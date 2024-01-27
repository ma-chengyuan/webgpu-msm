use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;

use crate::bytes::{read_points, write_points, N_BYTES_PER_POINT, N_U32S_PER_POINT};
use crate::gpu::{
    create_bind_group, create_buffer, create_buffer_init, create_pipeline, create_staging_buffer,
    dispatch, map_buffers, BufferBinding, GpuDeviceQueue, CURVE_WGSL, FIELD_MODULUS_WGSL,
    U256_WGSL,
};

pub struct GpuBucketer<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    n_buckets: usize,
    zeroes: Vec<u32>,
}

unsafe impl<'a> Sync for GpuBucketer<'a> {}

pub struct GpuBucketerPreprocessed {
    reshuffled: Vec<u32>,
    bucket_shuffled: Vec<usize>,
    idx_to_bkt: Vec<usize>,
    n_nonzero: usize,
}

impl<'a> GpuBucketer<'a> {
    pub fn new(dq: &'a GpuDeviceQueue, n_buckets: usize) -> Self {
        let (device, queue) = (&dq.device, &dq.queue);
        Self {
            device,
            queue,
            pipeline: create_pipeline(
                device,
                &format!(
                    "{}{}{}{}",
                    U256_WGSL,
                    FIELD_MODULUS_WGSL,
                    CURVE_WGSL,
                    include_str!("./wgsl/entry_bucket.wgsl")
                ),
                &[
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadWrite,
                ],
            ),
            n_buckets,
            zeroes: write_points(&vec![EdwardsProjective::zero(); n_buckets]),
        }
    }

    pub fn preprocess(
        &self,
        scalars: &[u32],
        points: &[EdwardsProjective],
    ) -> GpuBucketerPreprocessed {
        let n_buckets = self.n_buckets;
        assert_eq!(scalars.len(), points.len());

        let mut bucket = vec![0usize; n_buckets];
        for scalar in scalars {
            bucket[(*scalar) as usize] += 1;
        }
        // Ignore the 0th bucket because it's inconsequential for Pippenger's algorithm.
        bucket[0] = 0;
        let mut idx_to_bkt = (0..n_buckets).collect::<Vec<_>>();
        idx_to_bkt.sort_by_key(|&i| bucket[i]);
        for i in 1..n_buckets {
            // Compute the prefix sum.
            bucket[idx_to_bkt[i]] += bucket[idx_to_bkt[i - 1]];
            // bucket[i]: one-past-end index of the i-th bucket segment.
        }
        let bucket_shuffled = (0..n_buckets)
            .map(|i| bucket[idx_to_bkt[i]])
            .collect::<Vec<_>>();
        let n_nonzero = bucket[idx_to_bkt[n_buckets - 1]];
        let mut reshuffled = vec![EdwardsProjective::zero(); n_nonzero];
        for (scalar, point) in scalars.iter().zip(points.iter()) {
            let idx = (*scalar) as usize;
            if idx != 0 {
                bucket[idx] -= 1;
                reshuffled[bucket[idx]] = *point;
            }
        }
        GpuBucketerPreprocessed {
            reshuffled: write_points(&reshuffled),
            bucket_shuffled,
            idx_to_bkt,
            n_nonzero,
        }
    }

    pub async fn bucket(
        &self,
        preprocessed: GpuBucketerPreprocessed,
        interlude: impl FnOnce(),
    ) -> Vec<EdwardsProjective> {
        const MAX_BATCH_SIZE: usize = 1048576;
        const MAX_IN_FLIGHT: usize = 4;

        let n_buckets = self.n_buckets;
        let GpuBucketerPreprocessed {
            reshuffled,
            bucket_shuffled,
            idx_to_bkt,
            n_nonzero,
        } = preprocessed;

        let n_buffer_points = reshuffled.len().min(MAX_BATCH_SIZE);
        let points_buffer =
            create_buffer(self.device, (n_buffer_points * N_BYTES_PER_POINT) as u64);
        // 2 extra slots for the start bucket, n_threads.
        // the offsets array itself is n_buckets + 1 long where (+1) comes from the sentinel at the end.
        let n_buffer_points_rounded = (n_buffer_points + 63) / 64 * 64;
        let buckets_buffer = create_buffer(self.device, (n_buffer_points_rounded * 4) as u64);
        let offsets_buffer = create_buffer(self.device, ((n_buckets + 1) * 4) as u64);
        let sums_buffer = create_buffer_init(self.device, bytemuck::cast_slice(&self.zeroes));
        let staging_buffer =
            create_staging_buffer(self.device, (n_buckets * N_BYTES_PER_POINT) as u64);
        let bind_group = create_bind_group(
            self.device,
            &self.pipeline,
            &[
                &offsets_buffer,
                &points_buffer,
                &buckets_buffer,
                &sums_buffer,
            ],
        );

        let mut current_idx = 1;
        let n_batches = (n_nonzero + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
        let mut submitted: Vec<wgpu::SubmissionIndex> = vec![];
        for i in 0..n_batches {
            let start = i * MAX_BATCH_SIZE;
            let end = ((i + 1) * MAX_BATCH_SIZE).min(n_nonzero);
            let points = &reshuffled[start * N_U32S_PER_POINT..end * N_U32S_PER_POINT];

            // Find the first bucket that ends after the start of this batch.
            while current_idx < n_buckets && bucket_shuffled[current_idx] <= start {
                current_idx += 1
            }
            let start_idx = current_idx as u32;
            // Find the first bucket that starts after the end of this batch.
            while current_idx < n_buckets && bucket_shuffled[current_idx - 1] < end {
                current_idx += 1
            }
            let end_idx = current_idx as u32;
            // Back up one bucket. This is very important when a bucket segment
            // straddles the end of the batch. Say the segment is b, and we have
            // bucket[b - 1] < end < bucket[b]. end_bucket will be b + 1 and so
            // will be current_bucket. But when we start scanning in the next
            // batch, we want start_bucket to be b, not b + 1. So we back up to
            // give us a chance.
            current_idx -= 1;

            let n_threads = end_idx - start_idx;
            let n_workgroups = (n_threads + 63) / 64;
            let mut offsets = vec![0u32; n_threads as usize + 1];
            let mut buckets = vec![0u32; n_workgroups as usize * 64];

            for s in start_idx..end_idx {
                offsets[(s - start_idx) as usize] =
                    bucket_shuffled[s as usize - 1].saturating_sub(start) as u32;
                buckets[(s - start_idx) as usize] = idx_to_bkt[s as usize] as u32;
            }
            offsets[n_threads as usize] = (end - start) as u32;

            self.queue
                .write_buffer(&points_buffer, 0, bytemuck::cast_slice(points));
            self.queue
                .write_buffer(&offsets_buffer, 0, bytemuck::cast_slice(&offsets));
            self.queue
                .write_buffer(&buckets_buffer, 0, bytemuck::cast_slice(&buckets));
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            dispatch(
                &mut command_encoder,
                &self.pipeline,
                &bind_group,
                n_workgroups,
            );
            // Limits the number of in-flight batches. This prevents TDRs for
            // large instances when run directly (not through WebGPU). Not a
            // concern for WebGPU because device is automatically polled.
            if submitted.len() >= MAX_IN_FLIGHT {
                let wait_idx = &submitted[submitted.len() - MAX_IN_FLIGHT];
                self.device
                    .poll(wgpu::Maintain::wait_for(wait_idx.clone()))
                    .panic_on_timeout();
            }
            let submission_idx = self.queue.submit(Some(command_encoder.finish()));
            submitted.push(submission_idx);
        }

        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        command_encoder.copy_buffer_to_buffer(
            &sums_buffer,
            0,
            &staging_buffer,
            0,
            (n_buckets * N_BYTES_PER_POINT) as u64,
        );
        self.queue.submit(Some(command_encoder.finish()));
        std::mem::drop(reshuffled);
        std::mem::drop(bucket_shuffled);
        std::mem::drop(idx_to_bkt);

        interlude();

        let mut sums = vec![];
        map_buffers! {
            self.device,
            (staging_buffer, [..]) => |view: &[u32]| { sums = read_points(view); }
        }
        sums
    }
}
