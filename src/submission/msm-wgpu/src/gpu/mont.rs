use ark_ed_on_bls12_377::EdwardsProjective;
use bytemuck::{Pod, Zeroable};

use crate::bytes::{read_points_le, write_points_le, N_BYTES_PER_POINT};

use super::*;

pub struct GpuMontgomeryConverter<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    to_mont_pipeline: wgpu::ComputePipeline,
    from_mont_pipeline: wgpu::ComputePipeline,
}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct U256(pub [u32; 8]);

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable, Default)]
pub struct MontgomeryPoint {
    pub x: U256,
    pub y: U256,
    pub t: U256,
    pub z: U256,
}

impl<'a> GpuMontgomeryConverter<'a> {
    pub fn new(dq: &'a GpuDeviceQueue) -> Self {
        let (device, queue) = (&dq.device, &dq.queue);
        Self {
            device,
            queue,
            to_mont_pipeline: create_pipeline(
                device,
                &format!(
                    "{}{}",
                    ARITH_WGSL,
                    include_str!("./wgsl/entry_to_mont.wgsl")
                ),
                &[BufferBinding::ReadOnly, BufferBinding::ReadWrite],
            ),
            from_mont_pipeline: create_pipeline(
                device,
                &format!(
                    "{}{}",
                    ARITH_WGSL,
                    include_str!("./wgsl/entry_from_mont.wgsl")
                ),
                &[BufferBinding::ReadOnly, BufferBinding::ReadWrite],
            ),
        }
    }

    pub async fn to_mont(&self, points: &[EdwardsProjective]) -> Vec<MontgomeryPoint> {
        let limits = self.device.limits();
        let mut batch_size = points.len();
        batch_size = batch_size
            .min(limits.max_storage_buffer_binding_size as usize / N_BYTES_PER_POINT)
            // Every workgroup can montgomery-ify 2 components, so half a point
            .min(limits.max_compute_workgroups_per_dimension as usize / 2);

        let n_bytes = batch_size * N_BYTES_PER_POINT;
        let n_batches = (points.len() + batch_size - 1) / batch_size;
        log::info!(
            "to_mont: batch_size: {}, n_bytes: {}, n_batches: {}",
            batch_size,
            n_bytes,
            n_batches
        );

        let input_buffer = create_buffer(self.device, n_bytes as u64);
        let output_buffer = create_buffer(self.device, n_bytes as u64);

        let staging_buffer_size = (points
            .len()
            .min(limits.max_buffer_size as usize / N_BYTES_PER_POINT)
            * N_BYTES_PER_POINT) as u64;
        let staging_buffer = create_staging_buffer(self.device, staging_buffer_size);

        let bind_group = create_bind_group(
            self.device,
            &self.to_mont_pipeline,
            &[&input_buffer, &output_buffer],
        );
        let mut results = Vec::with_capacity(points.len());
        let mut batches = points.chunks(batch_size);

        let mut opt_batch = batches.next();
        let mut opt_batch_bytes = opt_batch.map(write_points_le);

        let mut staging_buffer_offset = 0;

        while let Some(batch) = opt_batch {
            let batch_bytes = &opt_batch_bytes.take().unwrap();
            self.queue
                .write_buffer(&input_buffer, 0, bytemuck::cast_slice(batch_bytes));
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let n_workgroups = batch.len() * 2;
            let result_size = (batch.len() * N_BYTES_PER_POINT) as u64;
            dispatch(
                &mut command_encoder,
                &self.to_mont_pipeline,
                &bind_group,
                n_workgroups as u32,
            );
            assert!(staging_buffer_offset + result_size <= staging_buffer_size);
            command_encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                staging_buffer_offset,
                result_size,
            );
            staging_buffer_offset += result_size;
            self.queue.submit(Some(command_encoder.finish()));
            // While the GPU is working, we can prepare the next batch
            opt_batch = batches.next();
            opt_batch_bytes = opt_batch.map(write_points_le);
            // Only map and transfer data when the staging buffer is full
            let needs_mapping = match opt_batch {
                Some(next_batch) => {
                    staging_buffer_offset + (next_batch.len() * N_BYTES_PER_POINT) as u64
                        > staging_buffer_size
                }
                None => true,
            };
            if needs_mapping {
                map_buffers!(
                    self.device,
                    (staging_buffer, [..staging_buffer_offset]) => |view: &[MontgomeryPoint]| { results.extend_from_slice(view); }
                )
            }
        }
        results
    }

    pub async fn from_mont(&self, points: &[MontgomeryPoint]) -> Vec<EdwardsProjective> {
        let limits = self.device.limits();
        let mut batch_size = points.len();
        batch_size = batch_size
            .min(limits.max_storage_buffer_binding_size as usize / N_BYTES_PER_POINT)
            // Every workgroup can montgomery-ify 2 components, so half a point
            .min(limits.max_compute_workgroups_per_dimension as usize / 2);
        let n_bytes = batch_size * N_BYTES_PER_POINT;
        let n_batches = (points.len() + batch_size - 1) / batch_size;
        log::info!(
            "from_mont: batch_size: {}, n_bytes: {}, n_batches: {}",
            batch_size,
            n_bytes,
            n_batches
        );

        let input_buffer = create_buffer(self.device, n_bytes as u64);
        let output_buffer = create_buffer(self.device, n_bytes as u64);

        let staging_buffer_size = (points
            .len()
            .min(limits.max_buffer_size as usize / N_BYTES_PER_POINT)
            * N_BYTES_PER_POINT) as u64;
        let staging_buffer = create_staging_buffer(self.device, staging_buffer_size);

        let bind_group = create_bind_group(
            self.device,
            &self.from_mont_pipeline,
            &[&input_buffer, &output_buffer],
        );
        let mut results = Vec::with_capacity(points.len());
        let mut batches = points.chunks(batch_size);

        let mut opt_batch = batches.next();

        let mut staging_buffer_offset = 0;

        while let Some(batch) = opt_batch {
            self.queue
                .write_buffer(&input_buffer, 0, bytemuck::cast_slice(batch));
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let n_workgroups = batch.len() * 2;
            let result_size = (batch.len() * N_BYTES_PER_POINT) as u64;
            dispatch(
                &mut command_encoder,
                &self.from_mont_pipeline,
                &bind_group,
                n_workgroups as u32,
            );
            assert!(staging_buffer_offset + result_size <= staging_buffer_size);
            command_encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                staging_buffer_offset,
                result_size,
            );
            staging_buffer_offset += result_size;
            self.queue.submit(Some(command_encoder.finish()));
            // While the GPU is working, we can prepare the next batch
            opt_batch = batches.next();
            // Only map and transfer data when the staging buffer is full
            let needs_mapping = match opt_batch {
                Some(next_batch) => {
                    staging_buffer_offset + (next_batch.len() * N_BYTES_PER_POINT) as u64
                        > staging_buffer_size
                }
                None => true,
            };
            if needs_mapping {
                map_buffers!(
                    self.device,
                    (staging_buffer, [..staging_buffer_offset]) => |view: &[u32]| { results.extend(read_points_le(view)); }
                )
            }
        }
        results
    }
}

pub struct GpuPadd<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuPadd<'a> {
    pub fn new(dq: &'a GpuDeviceQueue) -> Self {
        let (device, queue) = (&dq.device, &dq.queue);
        Self {
            device,
            queue,
            pipeline: create_pipeline(
                device,
                &format!("{}{}", ARITH_WGSL, include_str!("./wgsl/entry_padd.wgsl")),
                &[BufferBinding::ReadOnly, BufferBinding::ReadWrite],
            ),
        }
    }

    pub async fn add(&self, points: &[MontgomeryPoint]) -> Vec<MontgomeryPoint> {
        let limits = self.device.limits();
        let mut batch_size = points.len();
        assert!(points.len() % 4 == 0);
        batch_size = batch_size
            .min((limits.max_storage_buffer_binding_size as usize / N_BYTES_PER_POINT) / 2 * 2)
            // Every workgroup and add 4 points
            .min(limits.max_compute_workgroups_per_dimension as usize * 4);
        assert!(batch_size % 4 == 0);

        let n_bytes = batch_size * N_BYTES_PER_POINT;
        let n_batches = (points.len() + batch_size - 1) / batch_size;

        log::info!(
            "padd: batch_size: {}, n_bytes: {}, n_batches: {}",
            batch_size,
            n_bytes,
            n_batches
        );

        let input_buffer = create_buffer(self.device, n_bytes as u64);
        let output_buffer = create_buffer(self.device, (n_bytes / 2) as u64);

        let staging_buffer_size = (points
            .len()
            .min(limits.max_buffer_size as usize / N_BYTES_PER_POINT)
            * N_BYTES_PER_POINT) as u64;
        let staging_buffer = create_staging_buffer(self.device, staging_buffer_size);

        let bind_group = create_bind_group(
            self.device,
            &self.pipeline,
            &[&input_buffer, &output_buffer],
        );
        let mut results = Vec::with_capacity(points.len() / 2);
        let mut batches = points.chunks(batch_size);

        let mut opt_batch = batches.next();

        let mut staging_buffer_offset = 0;

        while let Some(batch) = opt_batch {
            self.queue
                .write_buffer(&input_buffer, 0, bytemuck::cast_slice(batch));
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let n_workgroups = batch.len() / 4;
            let result_size = (batch.len() / 2 * N_BYTES_PER_POINT) as u64;
            dispatch(
                &mut command_encoder,
                &self.pipeline,
                &bind_group,
                n_workgroups as u32,
            );
            assert!(staging_buffer_offset + result_size <= staging_buffer_size);
            command_encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                staging_buffer_offset,
                result_size,
            );
            staging_buffer_offset += result_size;
            self.queue.submit(Some(command_encoder.finish()));
            // While the GPU is working, we can prepare the next batch
            opt_batch = batches.next();
            // Only map and transfer data when the staging buffer is full
            let needs_mapping = match opt_batch {
                Some(next_batch) => {
                    staging_buffer_offset + (next_batch.len() / 2 * N_BYTES_PER_POINT) as u64
                        > staging_buffer_size
                }
                None => true,
            };
            if needs_mapping {
                map_buffers!(
                    self.device,
                    (staging_buffer, [..staging_buffer_offset]) => |view: &[MontgomeryPoint]| { results.extend_from_slice(view); }
                )
            }
        }
        results
    }
}

pub struct GpuPaddOriginal<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl<'a> GpuPaddOriginal<'a> {
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
                    include_str!("./wgsl/entry_padd_old.wgsl")
                ),
                &[BufferBinding::ReadOnly, BufferBinding::ReadWrite],
            ),
        }
    }

    pub async fn add(&self, points: &[EdwardsProjective]) -> Vec<EdwardsProjective> {
        let limits = self.device.limits();
        let mut batch_size = points.len();
        assert!(points.len() % 2 == 0);
        batch_size = batch_size
            .min((limits.max_storage_buffer_binding_size as usize / N_BYTES_PER_POINT) / 2 * 2)
            // Every workgroup and add 64 points
            .min(limits.max_compute_workgroups_per_dimension as usize * 64);
        assert!(batch_size % 2 == 0);

        let n_bytes = batch_size * N_BYTES_PER_POINT;
        let n_batches = (points.len() + batch_size - 1) / batch_size;

        log::info!(
            "padd_original: batch_size: {}, n_bytes: {}, n_batches: {}",
            batch_size,
            n_bytes,
            n_batches
        );

        let input_buffer = create_buffer(self.device, n_bytes as u64);
        let output_buffer = create_buffer(self.device, (n_bytes / 2) as u64);

        let staging_buffer_size = (points
            .len()
            .min(limits.max_buffer_size as usize / N_BYTES_PER_POINT)
            * N_BYTES_PER_POINT) as u64;
        let staging_buffer = create_staging_buffer(self.device, staging_buffer_size);

        let bind_group = create_bind_group(
            self.device,
            &self.pipeline,
            &[&input_buffer, &output_buffer],
        );
        let mut results = Vec::with_capacity(points.len() / 2);
        let mut batches = points.chunks(batch_size);

        let mut opt_batch = batches.next();
        let mut opt_batch_bytes = opt_batch.map(crate::bytes::write_points);

        let mut staging_buffer_offset = 0;

        while let Some(batch) = opt_batch {
            let batch_bytes = &opt_batch_bytes.take().unwrap();
            self.queue
                .write_buffer(&input_buffer, 0, bytemuck::cast_slice(batch_bytes));
            let mut command_encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let n_workgroups = batch.len() / 128;
            let result_size = (batch.len() / 2 * N_BYTES_PER_POINT) as u64;
            dispatch(
                &mut command_encoder,
                &self.pipeline,
                &bind_group,
                n_workgroups as u32,
            );
            assert!(staging_buffer_offset + result_size <= staging_buffer_size);
            command_encoder.copy_buffer_to_buffer(
                &output_buffer,
                0,
                &staging_buffer,
                staging_buffer_offset,
                result_size,
            );
            staging_buffer_offset += result_size;
            self.queue.submit(Some(command_encoder.finish()));
            // While the GPU is working, we can prepare the next batch
            opt_batch = batches.next();
            opt_batch_bytes = opt_batch.map(crate::bytes::write_points);
            // Only map and transfer data when the staging buffer is full
            let needs_mapping = match opt_batch {
                Some(next_batch) => {
                    staging_buffer_offset + (next_batch.len() / 2 * N_BYTES_PER_POINT) as u64
                        > staging_buffer_size
                }
                None => true,
            };
            if needs_mapping {
                map_buffers!(
                    self.device,
                    (staging_buffer, [..staging_buffer_offset]) => |view: &[u32]| { results.extend_from_slice(&crate::bytes::read_points(view)); }
                )
            }
        }
        results
    }
}

pub fn test_ser(points: &[EdwardsProjective]) {
    let now = std::time::Instant::now();
    let res = crate::bytes::write_points(points);
    log::info!(
        "writing points in big endian ({} u32s) took {:?}",
        res.len(),
        now.elapsed()
    );

    let now = std::time::Instant::now();
    let res = crate::bytes::write_points_le(points);
    log::info!(
        "writing points in little endian ({} u32s) took {:?}",
        res.len(),
        now.elapsed()
    );
}
