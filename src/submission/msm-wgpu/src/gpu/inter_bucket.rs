use ark_ed_on_bls12_377::EdwardsProjective;
use ark_ff::Zero;
use static_assertions::const_assert;

use crate::bytes::{read_points, write_points, N_BYTES_PER_POINT, N_U32S_PER_POINT};
use crate::gpu::{
    create_bind_group, create_buffer, create_pipeline, create_staging_buffer, dispatch,
    map_buffers, BufferBinding, GpuDeviceQueue, CURVE_WGSL, FIELD_MODULUS_WGSL, U256_WGSL,
};
pub struct GpuInterBucketReducer<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuInterBucketReducer<'a> {
    pub fn new(dq: &'a GpuDeviceQueue) -> Self {
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
                    include_str!("./wgsl/entry_inter_bucket.wgsl")
                ),
                &[
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadOnly,
                    BufferBinding::ReadWrite,
                    BufferBinding::ReadWrite,
                ],
            ),
        }
    }

    /// Given a slice of points, compute
    ///   sum_{i=0}^{n-1} i * points[i]
    pub async fn reduce(&self, points: &[EdwardsProjective]) -> EdwardsProjective {
        // Total VRAM
        const MAX_BATCH_SIZE: usize = 32768;
        const_assert!(MAX_BATCH_SIZE % 64 == 0);

        let mut n_points = points.len();
        let mut sum_n = write_points(points);
        let mut sum_i = write_points(&vec![EdwardsProjective::zero(); n_points]);

        let buffer_size = (n_points.min(MAX_BATCH_SIZE) * N_BYTES_PER_POINT) as u64;
        let i_sum_n_buffer = create_buffer(self.device, buffer_size);
        let i_sum_i_buffer = create_buffer(self.device, buffer_size);
        let o_sum_n_buffer = create_buffer(self.device, buffer_size / 64);
        let o_sum_i_buffer = create_buffer(self.device, buffer_size / 64);

        let bind_group = create_bind_group(
            self.device,
            &self.pipeline,
            &[
                &i_sum_n_buffer,
                &i_sum_i_buffer,
                &o_sum_n_buffer,
                &o_sum_i_buffer,
            ],
        );

        let o_sum_n_staging_buffer = create_staging_buffer(self.device, buffer_size / 64);
        let o_sum_i_staging_buffer = create_staging_buffer(self.device, buffer_size / 64);

        while n_points > 64 && n_points % 64 == 0 {
            let n_batches = (n_points + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
            let result_u32_size = n_points / 64 * N_U32S_PER_POINT;
            let mut result_sum_n = Vec::with_capacity(result_u32_size);
            let mut result_sum_i = Vec::with_capacity(result_u32_size);

            for i in 0..n_batches {
                let start_point = i * MAX_BATCH_SIZE;
                let start_u32 = start_point * N_U32S_PER_POINT;
                let end_point = ((i + 1) * MAX_BATCH_SIZE).min(n_points);
                let end_u32 = end_point * N_U32S_PER_POINT;
                self.queue.write_buffer(
                    &i_sum_n_buffer,
                    0,
                    bytemuck::cast_slice(&sum_n[start_u32..end_u32]),
                );
                self.queue.write_buffer(
                    &i_sum_i_buffer,
                    0,
                    bytemuck::cast_slice(&sum_i[start_u32..end_u32]),
                );
                let n_workgroups = (end_point - start_point) / 64;
                let mut command_encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                dispatch(
                    &mut command_encoder,
                    &self.pipeline,
                    &bind_group,
                    n_workgroups as u32,
                );
                let result_size = (n_workgroups * N_BYTES_PER_POINT) as u64;
                command_encoder.copy_buffer_to_buffer(
                    &o_sum_n_buffer,
                    0,
                    &o_sum_n_staging_buffer,
                    0,
                    result_size,
                );
                command_encoder.copy_buffer_to_buffer(
                    &o_sum_i_buffer,
                    0,
                    &o_sum_i_staging_buffer,
                    0,
                    result_size,
                );
                self.queue.submit(Some(command_encoder.finish()));
                map_buffers! {
                    self.device,
                    (o_sum_n_staging_buffer, [..result_size]) => |view: &[u32]| { result_sum_n.extend(view); },
                    (o_sum_i_staging_buffer, [..result_size]) => |view: &[u32]| { result_sum_i.extend(view); }
                }
            }
            sum_n = result_sum_n;
            sum_i = result_sum_i;
            n_points /= 64;
        }

        let mut sum = EdwardsProjective::zero();
        let mut carry = EdwardsProjective::zero();
        let sum_n = read_points(&sum_n);
        let sum_i = read_points(&sum_i);
        for (sn, si) in sum_n.into_iter().zip(sum_i.into_iter()).rev() {
            sum += carry + si;
            carry += sn;
        }
        sum
    }
}
