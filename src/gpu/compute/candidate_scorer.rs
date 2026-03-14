//! GPU-Accelerated Candidate Scoring
//!
//! Parallelizes candidate scoring across thousands of candidates using GPU compute shaders.
//! Falls back to CPU when GPU is unavailable.

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BindGroup, BindGroupLayout, ComputePipeline, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, BindGroupDescriptor, BindingType, ShaderStages, BufferBindingType, ShaderModuleDescriptor, ShaderSource};

use crate::config::ScoringWeights;
use crate::types::{Unit, ScoredCandidate, ScoreBreakdown, ContextMatrix, SequenceState, MergedState, MemoryType};
use crate::gpu::device::GpuDevice;
use crate::gpu::is_gpu_available;
use once_cell::sync::Lazy;

/// Global cached GPU candidate scorer (lazy initialized)
static GPU_SCORER: Lazy<Option<Arc<GpuCandidateScorer>>> = Lazy::new(|| {
    crate::gpu::global_device().and_then(|gpu| {
        match GpuCandidateScorer::new(&gpu) {
            Ok(scorer) => {
                log::info!("GPU candidate scorer initialized");
                Some(Arc::new(scorer))
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU scorer: {}", e);
                None
            }
        }
    })
});

/// Get the cached GPU scorer if available
fn get_gpu_scorer() -> Option<Arc<GpuCandidateScorer>> {
    GPU_SCORER.clone()
}

/// GPU-accelerated candidate scorer
pub struct GpuCandidateScorer {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
}

impl GpuCandidateScorer {
    /// Create a new GPU candidate scorer
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        // Create shader module
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Candidate Scoring Shader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/candidate_scoring.wgsl").into()),
        });

        // Create bind group layout (wgpu v22 API requires 'count' field)
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Candidate Scoring Bind Group Layout"),
            entries: &[
                // Candidates buffer (input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Weights buffer (input)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results buffer (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Counts buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Candidate Scoring Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Candidate Scoring Pipeline"),
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

    /// Score candidates on GPU
    pub fn score_gpu(
        &self,
        candidates: &[GpuCandidateData],
        weights: &GpuWeightData,
    ) -> Result<Vec<GpuScoreResult>, String> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let candidate_count = candidates.len() as u32;

        // Create candidate buffer
        let candidate_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Candidates Buffer"),
            size: (candidate_count as u64) * std::mem::size_of::<GpuCandidateData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create weights buffer
        let weights_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Weights Buffer"),
            size: std::mem::size_of::<GpuWeightData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create results buffer
        let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (candidate_count as u64) * std::mem::size_of::<GpuScoreResult>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (candidate_count as u64) * std::mem::size_of::<GpuScoreResult>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create counts buffer (use util::DeviceExt for create_buffer_init)
        let counts = [candidate_count, 0u32, 0u32, 0u32]; // pad to 16 bytes
        let counts_buffer = wgpu::util::DeviceExt::create_buffer_init(
            &*self.device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Counts Buffer"),
                contents: bytemuck::cast_slice(&counts),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        // Write data to buffers
        self.queue.write_buffer(&candidate_buffer, 0, bytemuck::cast_slice(candidates));
        self.queue.write_buffer(&weights_buffer, 0, bytemuck::cast_slice(&[*weights]));

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Candidate Scoring Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: candidate_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights_buffer.as_entire_binding(),
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

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Candidate Scoring Encoder"),
        });

        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Candidate Scoring Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (candidate_count + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (candidate_count as u64) * std::mem::size_of::<GpuScoreResult>() as u64,
        );

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv().map_err(|e| format!("Failed to receive mapping result: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {}", e))?;

        let data = buffer_slice.get_mapped_range();
        let results: Vec<GpuScoreResult> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(results)
    }
}

/// Candidate data formatted for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCandidateData {
    pub spatial_fit: f32,
    pub context_fit: f32,
    pub sequence_fit: f32,
    pub transition_fit: f32,
    pub utility_fit: f32,
    pub confidence_fit: f32,
    pub evidence_support: f32,
    pub _padding: f32,
    pub unit_id_low: u32,
    pub unit_id_high: u32,
}

/// Weight data formatted for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuWeightData {
    pub spatial: f32,
    pub context: f32,
    pub sequence: f32,
    pub transition: f32,
    pub utility: f32,
    pub confidence: f32,
    pub evidence: f32,
    pub freshness_boost: f32,
}

/// Score result from GPU
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuScoreResult {
    pub score: f32,
    pub unit_id_low: u32,
    pub unit_id_high: u32,
    pub _padding: f32,
}

impl GpuWeightData {
    pub fn from_weights(weights: &ScoringWeights, freshness_boost: f32) -> Self {
        Self {
            spatial: weights.spatial,
            context: weights.context,
            sequence: weights.sequence,
            transition: weights.transition,
            utility: weights.utility,
            confidence: weights.confidence,
            evidence: weights.evidence,
            freshness_boost,
        }
    }
}

/// Score candidates using GPU if available, otherwise CPU
pub fn score_candidates(
    candidates: &[Unit],
    context: &ContextMatrix,
    sequence: &SequenceState,
    merged: &MergedState,
    weights: &ScoringWeights,
) -> Vec<ScoredCandidate> {
    // For small batches, CPU is faster (no GPU overhead)
    if candidates.len() < 256 || !is_gpu_available() {
        return score_candidates_cpu(candidates, context, sequence, merged, weights);
    }

    // Try GPU scoring with cached scorer
    if let Some(scorer) = get_gpu_scorer() {
        // Prepare GPU data
        let gpu_candidates = prepare_gpu_candidates(candidates, context, sequence, merged);
        let gpu_weights = GpuWeightData::from_weights(weights, merged.freshness_boost);

        match scorer.score_gpu(&gpu_candidates, &gpu_weights) {
            Ok(results) => {
                return results
                    .into_iter()
                    .zip(candidates.iter())
                    .map(|(result, unit)| {
                        let uuid = uuid::Uuid::from_u128(
                            (result.unit_id_high as u128) << 32 | result.unit_id_low as u128
                        );
                        ScoredCandidate {
                            unit_id: unit.id,
                            content: unit.content.clone(),
                            score: result.score,
                            breakdown: ScoreBreakdown::default(), // GPU doesn't return breakdown
                            memory_type: unit.memory_type,
                        }
                    })
                    .collect();
            }
            Err(e) => {
                log::warn!("GPU scoring failed, falling back to CPU: {}", e);
            }
        }
    }

    // Fallback to CPU
    score_candidates_cpu(candidates, context, sequence, merged, weights)
}

/// CPU fallback for candidate scoring
fn score_candidates_cpu(
    candidates: &[Unit],
    context: &ContextMatrix,
    sequence: &SequenceState,
    merged: &MergedState,
    weights: &ScoringWeights,
) -> Vec<ScoredCandidate> {
    crate::layers::search::CandidateScorer::score(candidates, context, sequence, merged, weights, None, None)
}

/// Prepare candidate data for GPU
fn prepare_gpu_candidates(
    candidates: &[Unit],
    context: &ContextMatrix,
    sequence: &SequenceState,
    merged: &MergedState,
) -> Vec<GpuCandidateData> {
    use std::collections::HashSet;

    let merged_candidate_ids: HashSet<_> = merged.candidate_ids.iter().copied().collect();
    let recent_unit_ids: HashSet<_> = sequence.recent_unit_ids.iter().copied().collect();
    let task_entities: Vec<_> = sequence.task_entities.iter().map(|e| e.to_lowercase()).collect();

    candidates
        .iter()
        .map(|unit| {
            let lowered = unit.content.to_lowercase();
            
            let spatial_fit = if merged_candidate_ids.contains(&unit.id) {
                0.9
            } else {
                0.35
            };

            let level_mult = level_multiplier(unit.level);
            let context_fit = context_match(&lowered, context) * level_mult;
            
            let sequence_fit = if recent_unit_ids.contains(&unit.id) {
                0.95 * level_mult
            } else if task_entities.iter().any(|e| lowered.contains(e)) {
                0.65 * level_mult
            } else {
                0.25 * level_mult
            };

            let transition_fit = ((unit.links.len() as f32 / 5.0).clamp(0.0, 1.0)) * level_mult;
            let utility_fit = unit.utility_score.clamp(0.0, 1.0) * level_mult;
            let confidence_fit = ((unit.confidence + unit.trust_score) / 2.0).clamp(0.0, 1.0);
            let evidence_support = evidence_match(&lowered, merged) * level_mult;

            let uuid_bytes = unit.id.as_bytes();
            let unit_id_low = u32::from_le_bytes([uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3]]);
            let unit_id_high = u32::from_le_bytes([uuid_bytes[4], uuid_bytes[5], uuid_bytes[6], uuid_bytes[7]]);

            GpuCandidateData {
                spatial_fit,
                context_fit,
                sequence_fit,
                transition_fit,
                utility_fit,
                confidence_fit,
                evidence_support,
                _padding: 0.0,
                unit_id_low,
                unit_id_high,
            }
        })
        .collect()
}

// Helper functions (duplicated from search.rs for GPU preparation)
fn level_multiplier(level: crate::types::UnitLevel) -> f32 {
    use crate::types::UnitLevel;
    match level {
        UnitLevel::Char => 0.5,
        UnitLevel::Subword => 0.6,
        UnitLevel::Word => 0.7,
        UnitLevel::Phrase => 0.85,
        UnitLevel::Pattern => 1.0,
    }
}

fn context_match(lowered: &str, context: &ContextMatrix) -> f32 {
    if context.summary.contains(lowered) {
        0.9
    } else {
        0.3
    }
}

fn evidence_match(lowered: &str, merged: &MergedState) -> f32 {
    // Use evidence_support as proxy for evidence match
    if merged.evidence_support > 0.5 {
        0.8
    } else {
        0.2
    }
}
