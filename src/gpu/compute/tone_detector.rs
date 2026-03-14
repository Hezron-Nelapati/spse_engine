//! GPU-Accelerated Tone Detection
//!
//! Parallelizes tone keyword matching across multiple tone types using GPU compute.
//! Falls back to CPU when GPU is unavailable.

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BindGroupLayout, BindGroupDescriptor, ComputePipeline, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, ShaderModuleDescriptor, ShaderSource};

use crate::types::ToneKind;
use crate::gpu::device::GpuDevice;
use crate::gpu::is_gpu_available;
use once_cell::sync::Lazy;
use bytemuck::{Pod, Zeroable};

/// Global cached GPU tone detector (lazy initialized)
static GPU_TONE_DETECTOR: Lazy<Option<Arc<GpuToneDetector>>> = Lazy::new(|| {
    crate::gpu::global_device().and_then(|gpu| {
        match GpuToneDetector::new(&gpu) {
            Ok(detector) => {
                log::info!("GPU tone detector initialized");
                Some(Arc::new(detector))
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU tone detector: {}", e);
                None
            }
        }
    })
});

/// Get the cached GPU tone detector if available
fn get_gpu_tone_detector() -> Option<Arc<GpuToneDetector>> {
    GPU_TONE_DETECTOR.clone()
}

/// Maximum keywords per tone type
const MAX_KEYWORDS_PER_TONE: usize = 32;

/// Number of tone types
const TONE_COUNT: usize = 6;

/// GPU data for tone detection input
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuToneInput {
    /// Input text length in characters
    text_length: u32,
    /// Exclamation mark count
    exclamation_count: u32,
    /// Padding
    _padding: [u32; 2],
    /// Input text as u32 characters (padded to 1024 chars)
    text_data: [u32; 1024],
}

/// GPU data for tone keyword set
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuToneKeywords {
    /// Keyword count for this tone
    keyword_count: u32,
    /// Padding
    _padding: [u32; 3],
    /// Keyword offsets and lengths packed as (offset << 16 | length)
    keyword_meta: [u32; MAX_KEYWORDS_PER_TONE],
    /// Keyword data as u32 characters (shared buffer)
    keyword_data: [u32; 256],
}

/// GPU data for tone score output
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuToneScore {
    /// Tone kind index
    tone_index: u32,
    /// Score value
    score: f32,
    /// Padding
    _padding: [u32; 2],
}

/// GPU-accelerated tone detector
pub struct GpuToneDetector {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
    /// Pre-compiled keyword sets for each tone
    keyword_sets: [GpuToneKeywords; TONE_COUNT],
}

impl GpuToneDetector {
    /// Create a new GPU tone detector
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        // Pre-compile keyword sets
        let keyword_sets = Self::build_keyword_sets();

        // Shader for parallel tone detection
        let shader_source = include_str!("../shaders/tone_detector.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Tone Detector Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Tone Detector Bind Group Layout"),
            entries: &[
                // Input text buffer
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
                // Keyword sets buffer
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
                // Output scores buffer
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
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Tone Detector Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Tone Detection Pipeline"),
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
            keyword_sets,
        })
    }

    /// Build keyword sets for each tone type
    fn build_keyword_sets() -> [GpuToneKeywords; TONE_COUNT] {
        let urgency_keywords = [
            "urgent", "emergency", "asap", "immediately", "now", "critical",
            "important", "quickly", "hurry", "fast", "right now",
        ];
        let sadness_keywords = [
            "sad", "upset", "worried", "anxious", "depressed", "lonely",
            "hurt", "pain", "suffering", "struggling", "difficult", "hard",
            "lost", "grief", "cry", "tears", "hopeless",
        ];
        let technical_keywords = [
            "code", "function", "api", "algorithm", "debug", "error",
            "variable", "method", "class", "system", "architecture",
            "implementation", "database", "server", "client", "protocol",
            "interface", "module", "component", "service", "endpoint",
        ];
        let casual_keywords = [
            "hey", "yo", "sup", "cool", "awesome", "yeah", "nah",
            "gonna", "wanna", "kinda", "sorta", "lol", "haha",
        ];
        let formal_keywords = [
            "please", "thank you", "regarding", "sincerely", "respectfully",
            "would you", "could you", "may i", "i would like", "kindly",
        ];
        let empathetic_keywords = [
            "sorry", "understand", "feel", "difficult", "help", "sad",
            "upset", "worried", "anxious", "hope", "care", "support",
        ];

        fn build_set(keywords: &[&str]) -> GpuToneKeywords {
            let mut keyword_meta = [0u32; MAX_KEYWORDS_PER_TONE];
            let mut keyword_data = [0u32; 256];
            let mut offset = 0usize;

            for (i, kw) in keywords.iter().enumerate().take(MAX_KEYWORDS_PER_TONE) {
                let kw_bytes = kw.as_bytes();
                let len = kw_bytes.len() as u32;
                keyword_meta[i] = ((offset as u32) << 16) | len;
                
                for (j, &byte) in kw_bytes.iter().enumerate() {
                    if offset + j < 256 {
                        keyword_data[offset + j] = byte as u32;
                    }
                }
                offset += kw_bytes.len();
            }

            GpuToneKeywords {
                keyword_count: keywords.len().min(MAX_KEYWORDS_PER_TONE) as u32,
                _padding: [0; 3],
                keyword_meta,
                keyword_data,
            }
        }

        [
            build_set(&urgency_keywords),
            build_set(&sadness_keywords),
            build_set(&technical_keywords),
            build_set(&casual_keywords),
            build_set(&formal_keywords),
            build_set(&empathetic_keywords),
        ]
    }

    /// Detect tone on GPU
    pub fn detect_gpu(
        &self,
        input_lower: &str,
        exclamation_count: u32,
    ) -> Result<Vec<GpuToneScore>, String> {
        // Create input
        let mut text_data = [0u32; 1024];
        let text_bytes = input_lower.as_bytes();
        for (i, &byte) in text_bytes.iter().enumerate().take(1024) {
            text_data[i] = byte as u32;
        }

        let input = GpuToneInput {
            text_length: text_bytes.len().min(1024) as u32,
            exclamation_count,
            _padding: [0; 2],
            text_data,
        };

        // Create buffers
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tone Input Buffer"),
            size: std::mem::size_of::<GpuToneInput>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let keywords_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tone Keywords Buffer"),
            size: (TONE_COUNT * std::mem::size_of::<GpuToneKeywords>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tone Output Buffer"),
            size: (TONE_COUNT * std::mem::size_of::<GpuToneScore>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tone Staging Buffer"),
            size: (TONE_COUNT * std::mem::size_of::<GpuToneScore>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue.write_buffer(&input_buffer, 0, bytemuck::bytes_of(&input));
        
        // Write keyword sets
        for (i, kw_set) in self.keyword_sets.iter().enumerate() {
            let offset = (i * std::mem::size_of::<GpuToneKeywords>()) as u64;
            self.queue.write_buffer(&keywords_buffer, offset, bytemuck::bytes_of(kw_set));
        }

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Tone Detector Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keywords_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Tone Detector Encoder"),
        });

        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tone Detector Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (TONE_COUNT as u32 + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (TONE_COUNT * std::mem::size_of::<GpuToneScore>()) as u64,
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
        let results: Vec<GpuToneScore> = data
            .chunks_exact(std::mem::size_of::<GpuToneScore>())
            .map(|chunk| *bytemuck::from_bytes(chunk))
            .collect();

        Ok(results)
    }
}

/// Detect tone using GPU if available, otherwise CPU
pub fn detect_tone_gpu(
    input_lower: &str,
    exclamation_count: u32,
) -> Option<Vec<(ToneKind, f32)>> {
    if !is_gpu_available() {
        return None;
    }

    if let Some(detector) = get_gpu_tone_detector() {
        match detector.detect_gpu(input_lower, exclamation_count) {
            Ok(results) => {
                let tone_kinds = [
                    ToneKind::Direct,      // Urgency
                    ToneKind::Empathetic,  // Sadness
                    ToneKind::Technical,   // Technical
                    ToneKind::Casual,      // Casual
                    ToneKind::Formal,      // Formal
                    ToneKind::Empathetic,  // Empathetic (duplicate, but different keyword set)
                ];

                return Some(results
                    .into_iter()
                    .zip(tone_kinds.iter())
                    .map(|(gpu_score, kind)| (*kind, gpu_score.score))
                    .collect());
            }
            Err(e) => {
                log::warn!("GPU tone detection failed: {}", e);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_tone_input_size() {
        assert_eq!(std::mem::size_of::<GpuToneInput>(), 4112);
    }

    #[test]
    fn gpu_tone_score_size() {
        assert_eq!(std::mem::size_of::<GpuToneScore>(), 16);
    }
}
