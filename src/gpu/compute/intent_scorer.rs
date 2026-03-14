//! GPU-Accelerated Intent Scoring
//!
//! Parallelizes intent classification scoring across 23 intent types using GPU compute.
//! Falls back to CPU when GPU is unavailable.

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BindGroupLayout, BindGroupDescriptor, ComputePipeline, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, ShaderModuleDescriptor, ShaderSource};

use crate::types::IntentKind;
use crate::gpu::device::GpuDevice;
use crate::gpu::is_gpu_available;
use once_cell::sync::Lazy;
use bytemuck::{Pod, Zeroable};

/// Global cached GPU intent scorer (lazy initialized)
static GPU_INTENT_SCORER: Lazy<Option<Arc<GpuIntentScorer>>> = Lazy::new(|| {
    crate::gpu::global_device().and_then(|gpu| {
        match GpuIntentScorer::new(&gpu) {
            Ok(scorer) => {
                log::info!("GPU intent scorer initialized");
                Some(Arc::new(scorer))
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU intent scorer: {}", e);
                None
            }
        }
    })
});

/// Get the cached GPU intent scorer if available
fn get_gpu_intent_scorer() -> Option<Arc<GpuIntentScorer>> {
    GPU_INTENT_SCORER.clone()
}

/// GPU data for intent scoring input
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuIntentInput {
    /// Token count
    token_count: u32,
    /// Has temporal cues
    has_temporal: u32,
    /// Has domain hints
    has_domain_hints: u32,
    /// Has preference cues
    has_preference: u32,
    /// References document context
    references_doc: u32,
    /// Wants brief output
    wants_brief: u32,
    /// Certainty bias (as f32 bits)
    certainty_bias: f32,
    /// Turn index
    turn_index: u32,
    /// Padding for alignment (8 u32s = 32 bytes to reach 64 byte aligned size)
    _padding: [u32; 8],
}

/// GPU data for intent score output
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuIntentScore {
    /// Intent kind index
    intent_index: u32,
    /// Score value
    score: f32,
    /// Padding
    _padding: [u32; 2],
}

/// GPU-accelerated intent scorer
pub struct GpuIntentScorer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
}

impl GpuIntentScorer {
    /// Create a new GPU intent scorer
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        // Shader for parallel intent scoring
        let shader_source = include_str!("../shaders/intent_scorer.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Intent Scorer Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout for input/output buffers
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Intent Scorer Bind Group Layout"),
            entries: &[
                // Input buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer (storage)
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
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Intent Scorer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Intent Scoring Pipeline"),
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

    /// Score intents on GPU
    pub fn score_gpu(
        &self,
        input: &GpuIntentInput,
    ) -> Result<Vec<GpuIntentScore>, String> {
        const INTENT_COUNT: usize = 23;

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intent Input Buffer"),
            size: std::mem::size_of::<GpuIntentInput>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intent Output Buffer"),
            size: (INTENT_COUNT * std::mem::size_of::<GpuIntentScore>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Intent Staging Buffer"),
            size: (INTENT_COUNT * std::mem::size_of::<GpuIntentScore>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write input data
        self.queue.write_buffer(&input_buffer, 0, bytemuck::bytes_of(input));

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Intent Scorer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Intent Scorer Encoder"),
        });

        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Intent Scorer Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (INTENT_COUNT as u32 + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (INTENT_COUNT * std::mem::size_of::<GpuIntentScore>()) as u64,
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map staging buffer and read results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| format!("Failed to map buffer: {}", e))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuIntentScore> = data
            .chunks_exact(std::mem::size_of::<GpuIntentScore>())
            .map(|chunk| *bytemuck::from_bytes(chunk))
            .collect();

        Ok(results)
    }
}

/// Score intents using GPU if available, otherwise CPU
pub fn score_intents_gpu(
    token_count: usize,
    has_temporal: bool,
    has_domain_hints: bool,
    has_preference: bool,
    references_doc: bool,
    wants_brief: bool,
    certainty_bias: f32,
    turn_index: usize,
) -> Option<Vec<(IntentKind, f32)>> {
    // For small operations, CPU is faster (no GPU overhead)
    // GPU is beneficial when batch processing many queries
    if !is_gpu_available() {
        return None;
    }

    if let Some(scorer) = get_gpu_intent_scorer() {
        let input = GpuIntentInput {
            token_count: token_count as u32,
            has_temporal: has_temporal as u32,
            has_domain_hints: has_domain_hints as u32,
            has_preference: has_preference as u32,
            references_doc: references_doc as u32,
            wants_brief: wants_brief as u32,
            certainty_bias,
            turn_index: turn_index as u32,
            _padding: [0; 8],
        };

        match scorer.score_gpu(&input) {
            Ok(results) => {
                // Map GPU results to intent kinds
                let intent_kinds = [
                    IntentKind::Greeting, IntentKind::Gratitude, IntentKind::Farewell,
                    IntentKind::Help, IntentKind::Clarify, IntentKind::Rewrite,
                    IntentKind::Verify, IntentKind::Continue, IntentKind::Forget,
                    IntentKind::Summarize, IntentKind::Explain, IntentKind::Compare,
                    IntentKind::Extract, IntentKind::Analyze, IntentKind::Plan,
                    IntentKind::Act, IntentKind::Recommend, IntentKind::Classify,
                    IntentKind::Translate, IntentKind::Debug, IntentKind::Critique,
                    IntentKind::Brainstorm, IntentKind::Question,
                ];

                return Some(results
                    .into_iter()
                    .zip(intent_kinds.iter())
                    .map(|(gpu_score, kind)| (*kind, gpu_score.score))
                    .collect());
            }
            Err(e) => {
                log::warn!("GPU intent scoring failed: {}", e);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_intent_scorer_input_size() {
        assert_eq!(std::mem::size_of::<GpuIntentInput>(), 48);
    }

    #[test]
    fn gpu_intent_score_size() {
        assert_eq!(std::mem::size_of::<GpuIntentScore>(), 16);
    }
}
