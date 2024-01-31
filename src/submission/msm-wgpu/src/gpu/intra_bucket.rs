use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;
use bytemuck::{Pod, Zeroable};

use crate::bytes::{write_points, N_BYTES_PER_POINT};
use crate::gpu::{
    create_bind_group, create_buffer, create_buffer_init, create_pipeline, create_staging_buffer,
    dispatch, map_buffers, BufferBinding, GpuDeviceQueue, CURVE_WGSL, FIELD_MODULUS_WGSL,
    U256_WGSL,
};
use crate::PodPoint;

pub struct GpuIntraBucketReducer<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    n_buckets: usize,
    zeroes: Vec<u32>,
}

#[derive(Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct PaddIndices {
    in_idx_1: u32,
    in_idx_2: u32,
    out_idx: u32,
}

const PADD_INDEX_NO_INPUT_2: u32 = 0xffffffff;
const PADD_INDEX_OUTPUT_TO_BUCKET: u32 = 0x80000000;

impl<'a> GpuIntraBucketReducer<'a> {
    pub fn new(dq: &'a GpuDeviceQueue, n_buckets: usize) -> Self {
        let (device, queue) = (&dq.device, &dq.queue);
        Self {
            device,
            queue,
            pipeline: create_pipeline(
                device,
                &format!(
                    "{}{}{}{}",
                    include_str!("./wgsl/entry_padd_idx.wgsl"),
                    U256_WGSL,
                    FIELD_MODULUS_WGSL,
                    CURVE_WGSL,
                ),
                &[
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadWrite,
                    BufferBinding::ReadWrite,
                    BufferBinding::ReadOnly,
                ],
            ),
            n_buckets,
            zeroes: write_points(&vec![EdwardsProjective::zero(); n_buckets]),
        }
    }

    pub async fn reduce(&self, scalars: &[u32], points: &[PodPoint]) -> Vec<u32> {
        let mut idx_by_bucket = vec![vec![]; self.n_buckets];
        for (i, scalar) in scalars.iter().enumerate() {
            if scalar == &0 {
                continue;
            }
            idx_by_bucket[(*scalar) as usize].push(i as u32);
        }
        let mut compute_next_padd_indices = |indices: &mut Vec<PaddIndices>| {
            indices.clear();
            let mut next_output_idx = 0;
            let mut new_idx_by_bucket = vec![vec![]; self.n_buckets];
            for (bucket, idxs) in idx_by_bucket.iter().enumerate() {
                match idxs.len() {
                    0 => continue,
                    1 => {
                        indices.push(PaddIndices {
                            in_idx_1: idxs[0],
                            in_idx_2: PADD_INDEX_NO_INPUT_2,
                            out_idx: PADD_INDEX_OUTPUT_TO_BUCKET | bucket as u32,
                        });
                    }
                    2 => {
                        indices.push(PaddIndices {
                            in_idx_1: idxs[0],
                            in_idx_2: idxs[1],
                            out_idx: PADD_INDEX_OUTPUT_TO_BUCKET | bucket as u32,
                        });
                    }
                    x => {
                        for i in (0..x).step_by(2) {
                            indices.push(PaddIndices {
                                in_idx_1: idxs[i],
                                in_idx_2: idxs.get(i + 1).copied().unwrap_or(PADD_INDEX_NO_INPUT_2),
                                out_idx: next_output_idx,
                            });
                            new_idx_by_bucket[bucket].push(next_output_idx);
                            next_output_idx += 1;
                        }
                    }
                }
            }
            idx_by_bucket = new_idx_by_bucket;
        };

        let mut padd_indices = vec![];
        compute_next_padd_indices(&mut padd_indices);

        // let _limits = self.device.limits();
        let batch_size = points.len();
        // // batch_size = batch_size
        // //     .min(limits.max_storage_buffer_binding_size as usize / N_BYTES_PER_POINT)
        // //     // Every workgroup can montgomery-ify 2 components, so half a point
        // //     .min(limits.max_compute_workgroups_per_dimension as usize / 2);
        let n_bytes = (batch_size * N_BYTES_PER_POINT) as u64;
        let n_indices_bytes = (padd_indices.len() * std::mem::size_of::<PaddIndices>()) as u64;
        let indices_buffers = [
            create_buffer(self.device, n_indices_bytes),
            create_buffer(self.device, n_indices_bytes),
        ];
        let indices_length_buffers = [create_buffer(self.device, 4), create_buffer(self.device, 4)];
        let inout_buffers = [
            create_buffer(self.device, n_bytes),
            create_buffer(self.device, n_bytes),
        ];
        let bucket_buffer = create_buffer_init(self.device, bytemuck::cast_slice(&self.zeroes));
        let bind_groups = (0..2)
            .map(|i| {
                create_bind_group(
                    self.device,
                    &self.pipeline,
                    &[
                        &indices_buffers[i],
                        &inout_buffers[i],
                        &inout_buffers[1 - i],
                        &bucket_buffer,
                        &indices_length_buffers[i],
                    ],
                )
            })
            .collect::<Vec<_>>();

        let mut current_bind_group = 0usize;

        self.queue.write_buffer(
            &inout_buffers[current_bind_group],
            0,
            bytemuck::cast_slice(points),
        );

        let mut submitted: Vec<wgpu::SubmissionIndex> = vec![];
        while !padd_indices.is_empty() {
            self.queue.write_buffer(
                &indices_buffers[current_bind_group],
                0,
                bytemuck::cast_slice(&padd_indices),
            );
            self.queue.write_buffer(
                &indices_length_buffers[current_bind_group],
                0,
                bytemuck::cast_slice(&[padd_indices.len() as u32]),
            );
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let n_workgroups = padd_indices.len().div_ceil(64);
            dispatch(
                &mut command_encoder,
                &self.pipeline,
                &bind_groups[current_bind_group],
                n_workgroups as u32,
            );
            // Limits the number of in-flight batches. This prevents TDRs for
            // large instances when run directly (not through WebGPU). Not a
            // concern for WebGPU because device is automatically polled.
            const MAX_IN_FLIGHT: usize = 4;
            if submitted.len() >= MAX_IN_FLIGHT {
                let wait_idx = &submitted[submitted.len() - MAX_IN_FLIGHT];
                self.device
                    .poll(wgpu::Maintain::wait_for(wait_idx.clone()))
                    .panic_on_timeout();
            }
            let submission_idx = self.queue.submit(Some(command_encoder.finish()));
            submitted.push(submission_idx);
            current_bind_group = 1 - current_bind_group;
            compute_next_padd_indices(&mut padd_indices);
        }
        let mut command_encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let bucket_staging_buffer = create_staging_buffer(self.device, bucket_buffer.size());
        command_encoder.copy_buffer_to_buffer(
            &bucket_buffer,
            0,
            &bucket_staging_buffer,
            0,
            bucket_buffer.size(),
        );
        self.queue.submit(Some(command_encoder.finish()));
        let mut bucket = vec![];
        map_buffers! {
            self.device,
            (bucket_staging_buffer, [..]) => |view: &[u32]| { bucket.extend_from_slice(view); }
        }
        bucket
    }
}
