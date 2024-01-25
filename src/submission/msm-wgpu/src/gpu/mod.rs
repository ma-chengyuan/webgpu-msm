#![allow(dead_code)]
pub mod bucket;
pub mod bucket_sum;

use wgpu::util::DeviceExt;

const U256_WGSL: &str = include_str!("./wgsl/u256.wgsl");
const FIELD_MODULUS_WGSL: &str = include_str!("./wgsl/field_modulus.wgsl");
const CURVE_WGSL: &str = include_str!("./wgsl/curve.wgsl");

pub struct GpuDeviceQueue {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuDeviceQueue {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .expect("failed to find a suitable adapter");
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .expect("failed to create device");
        Self { device, queue }
    }

    pub async fn probe_max_vram(&self) -> u64 {
        let max_buffer_size = self.device.limits().max_storage_buffer_binding_size;
        log::info!("max_buffer_size: {}", max_buffer_size);
        let mut buffers = vec![];
        let mut successful_allocation = true;
        while successful_allocation {
            let mut allocation_size = max_buffer_size;
            successful_allocation = false;
            while allocation_size >= 64 * 1024 * 1024 {
                self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: allocation_size as u64,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                log::info!("allocation_size: {}", allocation_size);
                match self.device.pop_error_scope().await {
                    None => {
                        successful_allocation = true;
                        buffers.push(new_buffer);
                        break;
                    }
                    Some(_) => {
                        allocation_size /= 2;
                        log::info!("out of memory: {}", allocation_size);
                    }
                }
            }
        }
        buffers
            .into_iter()
            .map(|buffer| {
                let size = buffer.size();
                buffer.destroy();
                size
            })
            .sum()
    }
}

enum BufferBinding {
    ReadOnly,
    ReadWrite,
}

fn create_pipeline(
    device: &wgpu::Device,
    shader_source: &str,
    buffer_bindings: &[BufferBinding],
) -> wgpu::ComputePipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let binding_group_layout_entries = buffer_bindings
        .iter()
        .enumerate()
        .map(|(i, binding)| {
            let read_only = match binding {
                BufferBinding::ReadOnly => true,
                BufferBinding::ReadWrite => false,
            };
            wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect::<Vec<_>>();
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &binding_group_layout_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    })
}

fn create_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_buffer_init(device: &wgpu::Device, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn create_staging_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}

fn dispatch(
    command_encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    n_workgroups: u32,
) {
    let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    compute_pass.set_pipeline(pipeline);
    compute_pass.set_bind_group(0, bind_group, &[]);
    compute_pass.dispatch_workgroups(n_workgroups, 1, 1);
}

macro_rules! map_buffers {
    ($device:expr, $(($buffer:expr , [$slice:expr]) => $func:expr),+) => {{
        let slices = [$($buffer.slice($slice)),+];
        let (sender, receiver) = flume::bounded(slices.len());
        for slice in slices.iter() {
            let sender = sender.clone();
            slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
        }
        std::mem::drop(sender);
        $device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        for _ in 0..slices.len() {
            receiver.recv_async().await.unwrap().unwrap();
        }
        let mut idx = 0;
        $(
            {
                let view = slices[idx].get_mapped_range();
                #[allow(clippy::redundant_closure_call)]
                $func(bytemuck::cast_slice(&view));
                idx += 1;
            }
            $buffer.unmap();
        )+
        let _ = idx;
    }};
}

pub(crate) use map_buffers;
