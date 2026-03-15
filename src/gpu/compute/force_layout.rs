//! GPU-Accelerated Force-Directed Layout
//!
//! Implements O(n²) force calculations on GPU for spatial positioning.
//! This is the most compute-intensive operation in the engine.

use std::sync::Arc;
use wgpu::{Device, Queue, ComputePipeline, BindGroupLayout};

use crate::config::SemanticMapConfig;
use crate::types::Unit;
use crate::gpu::device::GpuDevice;

/// GPU-accelerated force-directed layout
pub struct GpuForceLayout {
    device: Arc<Device>,
    queue: Arc<Queue>,
    repulsive_pipeline: ComputePipeline,
    attractive_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
}

/// Position data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuPosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub is_anchor: u32,       // 0 = false, 1 = true
    pub is_process_unit: u32, // 0 = false, 1 = true
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
}

/// Force accumulator for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuForce {
    pub fx: f32,
    pub fy: f32,
    pub fz: f32,
    pub _padding: f32,
}

/// Edge data for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuEdge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    pub _padding: f32,
}

/// Layout configuration for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLayoutConfig {
    pub k: f32,                    // ideal distance
    pub repulsive_coeff: f32,      // repulsive force coefficient
    pub attractive_coeff: f32,     // attractive force coefficient
    pub temperature: f32,          // current temperature
    pub max_displacement: f32,     // max displacement per iteration
    pub boundary: f32,             // layout boundary
    pub node_count: u32,
    pub edge_count: u32,
    pub _padding: [u32; 2],
}

impl GpuForceLayout {
    /// Create a new GPU force layout
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        // Create shader modules
        let repulsive_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Repulsive Force Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/force_repulsive.wgsl").into()),
        });

        let attractive_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Attractive Force Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/force_attractive.wgsl").into()),
        });

        // Create bind group layout (wgpu v22 API requires 'count' field)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Layout Bind Group Layout"),
            entries: &[
                // Positions buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Forces buffer (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Edges buffer (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Config buffer (uniform)
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Force Layout Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipelines
        let repulsive_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Repulsive Force Pipeline"),
            layout: Some(&pipeline_layout),
            module: &repulsive_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let attractive_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Attractive Force Pipeline"),
            layout: Some(&pipeline_layout),
            module: &attractive_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            repulsive_pipeline,
            attractive_pipeline,
            bind_group_layout,
            workgroup_size,
        })
    }

    /// Run one iteration of force-directed layout on GPU
    pub fn iterate_gpu(
        &self,
        positions: &mut [GpuPosition],
        edges: &[GpuEdge],
        config: &GpuLayoutConfig,
    ) -> Result<f32, String> {
        let node_count = positions.len() as u32;
        let edge_count = edges.len() as u32;

        // Create buffers
        let position_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Positions Buffer"),
            size: (node_count as u64) * std::mem::size_of::<GpuPosition>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let force_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Forces Buffer"),
            size: (node_count as u64) * std::mem::size_of::<GpuForce>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let edge_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edges Buffer"),
            size: (edge_count.max(1) as u64) * std::mem::size_of::<GpuEdge>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let config_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: bytemuck::cast_slice(&[*config]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Write initial data
        self.queue.write_buffer(&position_buffer, 0, bytemuck::cast_slice(positions));
        self.queue.write_buffer(&edge_buffer, 0, bytemuck::cast_slice(edges));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Layout Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: force_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: edge_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Force Layout Encoder"),
        });

        // Run repulsive forces pass (O(n²) - most expensive)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Repulsive Forces Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.repulsive_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Each workgroup handles a tile of the interaction matrix
            let workgroups = (node_count + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, workgroups, 1);
        }

        // Run attractive forces pass (O(edges))
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Attractive Forces Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.attractive_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let edge_workgroups = (edge_count + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(edge_workgroups.max(1), 1, 1);
        }

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back positions
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (node_count as u64) * std::mem::size_of::<GpuPosition>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &position_buffer,
            0,
            &staging_buffer,
            0,
            (node_count as u64) * std::mem::size_of::<GpuPosition>() as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv().map_err(|e| format!("Failed to receive mapping result: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {}", e))?;

        let data = buffer_slice.get_mapped_range();
        let new_positions: Vec<GpuPosition> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Calculate mean displacement
        let total_displacement: f32 = positions
            .iter()
            .zip(new_positions.iter())
            .map(|(old, new)| {
                if old.is_anchor == 1 {
                    0.0
                } else {
                    ((new.x - old.x).powi(2) + (new.y - old.y).powi(2) + (new.z - old.z).powi(2)).sqrt()
                }
            })
            .sum();

        // Update positions
        positions.copy_from_slice(&new_positions);

        Ok(total_displacement / node_count as f32)
    }
}

/// Layout outcome from GPU
pub struct GpuLayoutOutcome {
    pub position_updates: Vec<(uuid::Uuid, [f32; 3])>,
    pub mean_displacement: f32,
    pub rolled_back: bool,
}

/// Run force-directed layout on GPU if available
pub fn force_layout_gpu(
    units: &[Unit],
    config: &SemanticMapConfig,
) -> Option<GpuLayoutOutcome> {
    let gpu = crate::gpu::global_device()?;
    let scorer = GpuForceLayout::new(&gpu).ok()?;

    // Select units for layout
    let selected = select_layout_units(units, config.max_layout_units);
    if selected.len() < 2 {
        return Some(GpuLayoutOutcome {
            position_updates: Vec::new(),
            mean_displacement: 0.0,
            rolled_back: false,
        });
    }

    // Prepare GPU data
    let mut positions: Vec<GpuPosition> = selected
        .iter()
        .map(|u| GpuPosition {
            x: u.semantic_position[0],
            y: u.semantic_position[1],
            z: u.semantic_position[2],
            is_anchor: if u.anchor_status { 1 } else { 0 },
            is_process_unit: if u.is_process_unit { 1 } else { 0 },
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
        })
        .collect();

    let edges = prepare_edges(&selected);
    
    let k = ideal_distance(selected.len(), config);
    let mut gpu_config = GpuLayoutConfig {
        k,
        repulsive_coeff: config.repulsive_force_coefficient.max(0.0),
        attractive_coeff: config.attractive_force_coefficient.max(0.0),
        temperature: config.max_displacement_per_iteration,
        max_displacement: config.max_displacement_per_iteration,
        boundary: config.layout_boundary,
        node_count: selected.len() as u32,
        edge_count: edges.len() as u32,
        _padding: [0, 0],
    };

    let initial_positions = positions.clone();
    let mut best_positions = positions.clone();
    let mut best_energy = calculate_energy(&positions, &edges, &gpu_config);
    let initial_energy = best_energy;
    let mut previous_energy = best_energy;
    let mut increases = 0usize;

    // Run iterations
    for _ in 0..config.max_layout_iterations {
        if let Ok(mean_disp) = scorer.iterate_gpu(&mut positions, &edges, &gpu_config) {
            let energy = calculate_energy(&positions, &edges, &gpu_config);
            
            if energy < best_energy {
                best_energy = energy;
                best_positions = positions.clone();
            }
            
            if energy > previous_energy {
                increases += 1;
                if increases >= config.energy_rollback_threshold as usize {
                    let rolled_back = best_energy <= initial_energy;
                    let final_positions = if rolled_back { best_positions } else { initial_positions };
                    
                    return Some(build_outcome(&selected, &final_positions, rolled_back));
                }
            } else {
                increases = 0;
            }
            
            previous_energy = energy;
            gpu_config.temperature *= 0.90;
            
            if mean_disp <= config.convergence_tolerance {
                break;
            }
        }
    }

    Some(build_outcome(&selected, &best_positions, false))
}

// Helper functions (duplicated from spatial_index.rs for GPU)
fn select_layout_units(units: &[Unit], max: usize) -> Vec<&Unit> {
    let mut sorted: Vec<_> = units.iter().collect();
    sorted.sort_by(|a, b| {
        b.utility_score
            .partial_cmp(&a.utility_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.into_iter().take(max).collect()
}

fn ideal_distance(count: usize, config: &SemanticMapConfig) -> f32 {
    (config.layout_boundary / (count.max(1) as f32).sqrt()) * 0.8
}

fn prepare_edges(units: &[&Unit]) -> Vec<GpuEdge> {
    let mut edges = Vec::new();
    let id_to_idx: std::collections::HashMap<uuid::Uuid, u32> = units
        .iter()
        .enumerate()
        .map(|(i, u)| (u.id, i as u32))
        .collect();

    for (idx, unit) in units.iter().enumerate() {
        for link in &unit.links {
            if let Some(&target_idx) = id_to_idx.get(&link.target_id) {
                if target_idx > idx as u32 {
                    edges.push(GpuEdge {
                        source: idx as u32,
                        target: target_idx,
                        weight: link.weight.max(0.12),
                        _padding: 0.0,
                    });
                }
            }
        }
    }

    // Add default edges if none exist
    if edges.is_empty() && units.len() > 1 {
        for i in 1..units.len() {
            edges.push(GpuEdge {
                source: (i - 1) as u32,
                target: i as u32,
                weight: 0.16,
                _padding: 0.0,
            });
        }
    }

    edges
}

fn calculate_energy(positions: &[GpuPosition], edges: &[GpuEdge], config: &GpuLayoutConfig) -> f32 {
    let mut energy = 0.0;

    // Repulsive energy
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let dx = positions[i].x - positions[j].x;
            let dy = positions[i].y - positions[j].y;
            let dz = positions[i].z - positions[j].z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.05);
            energy += (config.k * config.k) / dist * config.repulsive_coeff;
        }
    }

    // Attractive energy
    for edge in edges {
        let p1 = &positions[edge.source as usize];
        let p2 = &positions[edge.target as usize];
        let dx = p1.x - p2.x;
        let dy = p1.y - p2.y;
        let dz = p1.z - p2.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        energy += (dist * dist) / config.k * edge.weight * config.attractive_coeff;
    }

    energy
}

fn build_outcome(units: &[&Unit], positions: &[GpuPosition], rolled_back: bool) -> GpuLayoutOutcome {
    let position_updates: Vec<_> = units
        .iter()
        .zip(positions.iter())
        .map(|(unit, pos)| (unit.id, [pos.x, pos.y, pos.z]))
        .collect();

    GpuLayoutOutcome {
        position_updates,
        mean_displacement: 0.0,
        rolled_back,
    }
}
