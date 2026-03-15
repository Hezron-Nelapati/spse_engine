//! GPU-Accelerated Classification Calculator
//!
//! Parallelizes signature similarity scoring and vote aggregation using GPU compute.
//! Falls back to CPU when GPU is unavailable.

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BindGroupLayout, ComputePipeline, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, ShaderModuleDescriptor, ShaderSource};

use crate::types::{IntentKind, ToneKind, ResolverMode, ClassificationResult, CalculationMethod};
use crate::classification::{ClassificationSignature, ClassificationPattern};
use crate::config::ClassificationConfig;
use crate::gpu::device::GpuDevice;
use crate::gpu::is_gpu_available;
use once_cell::sync::Lazy;
use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

/// Global cached GPU classification calculator (lazy initialized)
static GPU_CLASSIFIER: Lazy<Option<Arc<GpuClassificationCalculator>>> = Lazy::new(|| {
    crate::gpu::global_device().and_then(|gpu| {
        match GpuClassificationCalculator::new(&gpu) {
            Ok(calc) => {
                log::info!("GPU classification calculator initialized");
                Some(Arc::new(calc))
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU classification calculator: {}", e);
                None
            }
        }
    })
});

/// Get the cached GPU classification calculator if available
pub fn get_gpu_classifier() -> Option<Arc<GpuClassificationCalculator>> {
    GPU_CLASSIFIER.clone()
}

/// GPU data for classification signature (16 floats = 64 bytes, aligned)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSignature {
    /// Structure scores (4 floats)
    structure_scores: [f32; 4],
    /// Punctuation scores (4 floats)
    punctuation_scores: [f32; 4],
    /// Semantic centroid (3 floats) + 1 padding
    semantic_centroid: [f32; 3],
    /// Padding for alignment
    _pad0: f32,
    /// Derived scores (3 floats) + 1 padding
    derived_scores: [f32; 3],
    /// Padding for 16-byte alignment
    _pad1: f32,
}

impl From<&ClassificationSignature> for GpuSignature {
    fn from(sig: &ClassificationSignature) -> Self {
        Self {
            // Map structure fields to array
            structure_scores: [
                sig.byte_length_norm,
                sig.sentence_entropy,
                sig.token_count_norm,
                0.0, // padding
            ],
            // Map punctuation vector + padding
            punctuation_scores: [
                sig.punct_vector[0],
                sig.punct_vector[1],
                sig.punct_vector[2],
                0.0, // padding
            ],
            semantic_centroid: sig.semantic_centroid,
            _pad0: 0.0,
            // Map derived scores
            derived_scores: [
                sig.urgency_score,
                sig.formality_score,
                sig.technical_score,
            ],
            _pad1: 0.0,
        }
    }
}

/// GPU data for classification pattern
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuPattern {
    /// Pattern signature
    signature: GpuSignature,
    /// Intent kind index (0-22)
    intent_index: u32,
    /// Tone kind index (0-7)
    tone_index: u32,
    /// Resolver mode index (0-2)
    resolver_index: u32,
    /// Pattern confidence
    confidence: f32,
    /// Success count
    success_count: u32,
    /// Failure count
    failure_count: u32,
    /// Padding
    _padding: [u32; 2],
}

/// GPU data for similarity result
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSimilarityResult {
    /// Pattern index
    pattern_index: u32,
    /// Similarity score
    similarity: f32,
    /// Final score (similarity * confidence)
    final_score: f32,
    /// Intent index
    intent_index: u32,
    /// Tone index
    tone_index: u32,
    /// Resolver index
    resolver_index: u32,
    /// Padding for 16-byte alignment (6 u32s = 24 bytes, need 8 more for 32)
    _padding: [u32; 2],
}

/// GPU data for aggregated votes (160 bytes, 16-byte aligned)
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuVoteAggregation {
    /// Intent scores (24 slots for alignment, only 23 used)
    intent_scores: [f32; 24],
    /// Tone scores (8 tones)
    tone_scores: [f32; 8],
    /// Resolver scores (4 slots for alignment, only 3 used)
    resolver_scores: [f32; 4],
    /// Best intent index
    best_intent: u32,
    /// Best tone index
    best_tone: u32,
    /// Best resolver index
    best_resolver: u32,
    /// Overall confidence
    confidence: f32,
    /// Candidate count
    candidate_count: u32,
    /// Padding for 16-byte alignment
    _padding: [u32; 3],
}

/// Minimum pattern count to justify GPU dispatch overhead
#[allow(dead_code)]
const GPU_THRESHOLD: usize = 64;

/// Maximum patterns to process in single GPU dispatch (prevents buffer overflow)
const MAX_PATTERNS_PER_DISPATCH: usize = 4096;

/// GPU-accelerated classification calculator
#[allow(dead_code)]
pub struct GpuClassificationCalculator {
    device: Arc<Device>,
    queue: Arc<Queue>,
    similarity_pipeline: ComputePipeline,
    aggregation_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
    /// Reusable buffers for common pattern counts (reduces allocations)
    cached_buffers: std::sync::Mutex<HashMap<usize, CachedBuffers>>,
}

/// Cached GPU buffers for a specific pattern count
struct CachedBuffers {
    query_buffer: Buffer,
    patterns_buffer: Buffer,
    results_buffer: Buffer,
    staging_buffer: Buffer,
}

impl GpuClassificationCalculator {
    /// Create a new GPU classification calculator
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device();
        let queue = gpu.queue();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size.max(64);
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("ClassificationBindGroupLayout"),
            entries: &[
                // Query signature (uniform)
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
                // Patterns buffer (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Results buffer (storage, read/write)
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
                // Config uniform
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
        
        // Create similarity pipeline
        let similarity_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ClassificationSimilarityShader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/classification_similarity.wgsl").into()),
        });
        
        let similarity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ClassificationSimilarityPipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ClassificationSimilarityLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &similarity_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create aggregation pipeline
        let aggregation_shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("ClassificationAggregationShader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/classification_aggregation.wgsl").into()),
        });
        
        let aggregation_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ClassificationAggregationPipeline"),
            layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("ClassificationAggregationLayout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &aggregation_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        Ok(Self {
            device: device.clone(),
            queue: queue.clone(),
            similarity_pipeline,
            aggregation_pipeline,
            bind_group_layout,
            workgroup_size,
            cached_buffers: std::sync::Mutex::new(HashMap::new()),
        })
    }
    
    /// Get or create cached buffers for pattern count
    fn get_buffers(&self, pattern_count: usize) -> (Buffer, Buffer, Buffer, Buffer) {
        // Round up to next power of 2 for better cache reuse
        let bucket = pattern_count.next_power_of_two().min(MAX_PATTERNS_PER_DISPATCH);
        
        let mut cache = self.cached_buffers.lock().unwrap();
        
        if !cache.contains_key(&bucket) {
            // Create new buffers for this bucket
            let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("QuerySignatureBuffer"),
                size: std::mem::size_of::<GpuSignature>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let patterns_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PatternsBuffer"),
                size: (std::mem::size_of::<GpuPattern>() * bucket) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let results_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ResultsBuffer"),
                size: (std::mem::size_of::<GpuSimilarityResult>() * bucket) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            
            let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("StagingBuffer"),
                size: (std::mem::size_of::<GpuSimilarityResult>() * bucket) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            cache.insert(bucket, CachedBuffers {
                query_buffer,
                patterns_buffer,
                results_buffer,
                staging_buffer,
            });
        }
        
        let cached = cache.get(&bucket).unwrap();
        (
            cached.query_buffer.clone(),
            cached.patterns_buffer.clone(),
            cached.results_buffer.clone(),
            cached.staging_buffer.clone(),
        )
    }
    
    /// Calculate classification using GPU acceleration
    /// Returns None if GPU fails, caller should fall back to CPU
    pub fn calculate_gpu(
        &self,
        query_sig: &ClassificationSignature,
        patterns: &[ClassificationPattern],
        config: &ClassificationConfig,
    ) -> Option<ClassificationResult> {
        let start = std::time::Instant::now();
        
        if patterns.is_empty() {
            return Some(ClassificationResult {
                intent: IntentKind::Unknown,
                tone: ToneKind::NeutralProfessional,
                resolver_mode: ResolverMode::Exploratory,
                confidence: 0.0,
                method: CalculationMethod::MemoryLookup,
                candidate_count: 0,
            });
        }
        
        // Limit patterns to prevent buffer overflow
        let effective_count = patterns.len().min(MAX_PATTERNS_PER_DISPATCH);
        let patterns = &patterns[..effective_count];
        let pattern_count = effective_count as u32;
        
        // Convert patterns to GPU format (pre-allocate)
        let mut gpu_patterns = Vec::with_capacity(effective_count);
        for p in patterns {
            gpu_patterns.push(GpuPattern {
                signature: GpuSignature::from(&p.signature),
                intent_index: p.intent_kind as u32,
                tone_index: p.tone_kind as u32,
                resolver_index: p.resolver_mode as u32,
                confidence: p.confidence(),
                success_count: p.success_count as u32,
                failure_count: p.failure_count as u32,
                _padding: [0; 2],
            });
        }
        
        // Get cached buffers
        let (query_buffer, patterns_buffer, results_buffer, staging_buffer) = 
            self.get_buffers(effective_count);
        
        // Write data
        let gpu_query = GpuSignature::from(query_sig);
        self.queue.write_buffer(&query_buffer, 0, bytemuck::bytes_of(&gpu_query));
        self.queue.write_buffer(&patterns_buffer, 0, bytemuck::cast_slice(&gpu_patterns));
        
        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ClassificationBindGroup"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: patterns_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: results_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Dispatch compute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ClassificationEncoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ClassificationSimilarityPass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.similarity_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (pattern_count + self.workgroup_size - 1) / self.workgroup_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        
        // Copy results to staging
        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (std::mem::size_of::<GpuSimilarityResult>() * effective_count) as u64,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Read back results
        let results = futures::executor::block_on(async {
            let slice = staging_buffer.slice(..(std::mem::size_of::<GpuSimilarityResult>() * effective_count) as u64);
            slice.map_async(wgpu::MapMode::Read, |_| ());
            self.device.poll(wgpu::Maintain::Wait);
            
            let data = slice.get_mapped_range().to_vec();
            // slice is dropped here (Copy type, drop is no-op but scope ends)
            staging_buffer.unmap();
            
            bytemuck::cast_slice::<u8, GpuSimilarityResult>(&data).to_vec()
        });
        
        let elapsed = start.elapsed();
        log::debug!(
            "GPU classification: {} patterns in {:?} ({:.2} patterns/ms)",
            effective_count,
            elapsed,
            effective_count as f64 / elapsed.as_secs_f64() / 1000.0
        );
        
        // Filter by threshold and aggregate
        let scored: Vec<_> = results.into_iter()
            .filter(|r| r.final_score >= config.min_similarity_threshold)
            .collect();
        
        if scored.is_empty() {
            return Some(ClassificationResult {
                intent: IntentKind::Unknown,
                tone: ToneKind::NeutralProfessional,
                resolver_mode: ResolverMode::Exploratory,
                confidence: 0.0,
                method: CalculationMethod::MemoryLookup,
                candidate_count: 0,
            });
        }
        
        // Aggregate votes on CPU (small dataset, not worth GPU dispatch)
        let mut intent_scores = [0.0f32; 23];
        let mut tone_scores = [0.0f32; 8];
        let mut resolver_scores = [0.0f32; 3];
        
        for result in &scored {
            if (result.intent_index as usize) < 23 {
                intent_scores[result.intent_index as usize] += result.final_score;
            }
            if (result.tone_index as usize) < 8 {
                tone_scores[result.tone_index as usize] += result.final_score;
            }
            if (result.resolver_index as usize) < 3 {
                resolver_scores[result.resolver_index as usize] += result.final_score;
            }
        }
        
        // Find best scores
        let (best_intent, intent_score) = intent_scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i as u32, s))
            .unwrap_or((0, 0.0));
        
        let (best_tone, tone_score) = tone_scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i as u32, s))
            .unwrap_or((0, 0.0));
        
        let (best_resolver, _) = resolver_scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &s)| (i as u32, s))
            .unwrap_or((1, 0.0));
        
        let confidence = (intent_score + tone_score) / 2.0;
        
        // Apply confidence-driven resolver override
        let final_resolver = if confidence < config.low_confidence_threshold {
            ResolverMode::Exploratory
        } else if confidence > config.high_confidence_threshold && best_resolver == 1 {
            ResolverMode::Deterministic
        } else {
            match best_resolver {
                0 => ResolverMode::Deterministic,
                1 => ResolverMode::Balanced,
                _ => ResolverMode::Exploratory,
            }
        };
        
        Some(ClassificationResult {
            intent: intent_from_index(best_intent),
            tone: tone_from_index(best_tone),
            resolver_mode: final_resolver,
            confidence,
            method: CalculationMethod::MemoryLookup,
            candidate_count: scored.len(),
        })
    }
}

/// Check if GPU classification is available
pub fn is_gpu_classification_available() -> bool {
    is_gpu_available() && GPU_CLASSIFIER.is_some()
}

/// Convert intent index to IntentKind
fn intent_from_index(index: u32) -> IntentKind {
    match index {
        0 => IntentKind::Greeting,
        1 => IntentKind::Gratitude,
        2 => IntentKind::Farewell,
        3 => IntentKind::Help,
        4 => IntentKind::Clarify,
        5 => IntentKind::Rewrite,
        6 => IntentKind::Verify,
        7 => IntentKind::Continue,
        8 => IntentKind::Forget,
        9 => IntentKind::Question,
        10 => IntentKind::Summarize,
        11 => IntentKind::Explain,
        12 => IntentKind::Compare,
        13 => IntentKind::Extract,
        14 => IntentKind::Analyze,
        15 => IntentKind::Plan,
        16 => IntentKind::Act,
        17 => IntentKind::Recommend,
        18 => IntentKind::Classify,
        19 => IntentKind::Translate,
        20 => IntentKind::Debug,
        21 => IntentKind::Critique,
        22 => IntentKind::Brainstorm,
        _ => IntentKind::Unknown,
    }
}

/// Convert tone index to ToneKind
fn tone_from_index(index: u32) -> ToneKind {
    match index {
        0 => ToneKind::NeutralProfessional,
        1 => ToneKind::Empathetic,
        2 => ToneKind::Direct,
        3 => ToneKind::Technical,
        4 => ToneKind::Casual,
        5 => ToneKind::Formal,
        _ => ToneKind::NeutralProfessional,
    }
}
