//! GPU-Accelerated Distance Calculations
//!
//! Parallelizes distance calculations for spatial queries and similarity scoring.

use std::sync::Arc;
use wgpu::{BindGroupLayout, ComputePipeline, Device, Queue};

use crate::gpu::device::GpuDevice;

/// GPU-accelerated distance calculator
pub struct GpuDistanceCalculator {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
}

/// Position for distance calculation
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub _padding: f32,
}

/// Distance result
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDistanceResult {
    pub distance: f32,
    pub index: u32,
    pub _padding: [f32; 2],
}

impl GpuDistanceCalculator {
    /// Create a new GPU distance calculator
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Distance Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/distance.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Distance Bind Group Layout"),
            entries: &[
                // Positions buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Center buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counts buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Distance Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Distance Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            workgroup_size,
        })
    }

    /// Calculate distances from a center point to all positions
    pub fn distances_from_center(
        &self,
        positions: &[GpuVec3],
        center: [f32; 3],
    ) -> Result<Vec<GpuDistanceResult>, String> {
        let count = positions.len() as u32;
        if count == 0 {
            return Ok(Vec::new());
        }

        // Create buffers
        let position_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Buffer"),
            size: (count as u64) * std::mem::size_of::<GpuVec3>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let center_data = [center[0], center[1], center[2], 0.0];
        let center_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Center Buffer"),
                contents: bytemuck::cast_slice(&center_data),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (count as u64) * std::mem::size_of::<GpuDistanceResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counts = [count, 0u32, 0u32, 0u32];
        let counts_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Counts Buffer"),
                contents: bytemuck::cast_slice(&counts),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Write data
        self.queue
            .write_buffer(&position_buffer, 0, bytemuck::cast_slice(positions));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: center_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: counts_buffer.as_entire_binding(),
                },
            ],
        });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (count as u64) * std::mem::size_of::<GpuDistanceResult>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Run compute pass
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Distance Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Distance Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = (count + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (count as u64) * std::mem::size_of::<GpuDistanceResult>() as u64,
        );

        self.queue.submit(std::iter::once(encoder.finish()));

        // Read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| format!("Failed to receive mapping result: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {}", e))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuDistanceResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }

    /// Find all positions within a radius
    pub fn within_radius(
        &self,
        positions: &[GpuVec3],
        center: [f32; 3],
        radius: f32,
    ) -> Result<Vec<usize>, String> {
        let results = self.distances_from_center(positions, center)?;

        Ok(results
            .into_iter()
            .enumerate()
            .filter(|(_, r)| r.distance <= radius)
            .map(|(i, _)| i)
            .collect())
    }
}

/// Calculate Euclidean distance between two 3D points (CPU fallback)
pub fn euclidean_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Find nearby positions using GPU if available
pub fn find_nearby_gpu(
    positions: &[[f32; 3]],
    center: [f32; 3],
    radius: f32,
) -> Option<Vec<usize>> {
    let gpu = crate::gpu::global_device()?;
    let calc = GpuDistanceCalculator::new(&gpu).ok()?;

    let gpu_positions: Vec<GpuVec3> = positions
        .iter()
        .map(|p| GpuVec3 {
            x: p[0],
            y: p[1],
            z: p[2],
            _padding: 0.0,
        })
        .collect();

    calc.within_radius(&gpu_positions, center, radius).ok()
}
