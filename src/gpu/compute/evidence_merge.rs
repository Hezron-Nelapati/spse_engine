//! GPU-Accelerated Evidence Merge
//!
//! Parallelizes document-context overlap detection using GPU compute.
//! Falls back to CPU when GPU is unavailable.

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BindGroupLayout, BindGroupDescriptor, ComputePipeline, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, ShaderModuleDescriptor, ShaderSource};

use crate::types::{ContextMatrix, RetrievedDocument};
use crate::gpu::device::GpuDevice;
use crate::gpu::is_gpu_available;
use once_cell::sync::Lazy;
use bytemuck::{Pod, Zeroable};

/// Global cached GPU evidence merger (lazy initialized)
static GPU_EVIDENCE_MERGER: Lazy<Option<Arc<GpuEvidenceMerger>>> = Lazy::new(|| {
    crate::gpu::global_device().and_then(|gpu| {
        match GpuEvidenceMerger::new(&gpu) {
            Ok(merger) => {
                log::info!("GPU evidence merger initialized");
                Some(Arc::new(merger))
            }
            Err(e) => {
                log::warn!("Failed to initialize GPU evidence merger: {}", e);
                None
            }
        }
    })
});

/// Get the cached GPU evidence merger if available
fn get_gpu_evidence_merger() -> Option<Arc<GpuEvidenceMerger>> {
    GPU_EVIDENCE_MERGER.clone()
}

/// Maximum context cells
const MAX_CONTEXT_CELLS: usize = 64;

/// Maximum documents
const MAX_DOCUMENTS: usize = 128;

/// Maximum cell content length (in u32 chars)
const MAX_CELL_LENGTH: usize = 128;

/// Maximum document content length (in u32 chars)
const MAX_DOC_LENGTH: usize = 2048;

/// GPU data for context cell
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuContextCell {
    /// Cell content length
    length: u32,
    /// Padding
    _padding: [u32; 3],
    /// Cell content as lowercase u32 characters
    content: [u32; MAX_CELL_LENGTH],
}

/// GPU data for document
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuDocument {
    /// Document content length
    length: u32,
    /// Trust score (as f32)
    trust_score: f32,
    /// Padding
    _padding: [u32; 2],
    /// Document content as lowercase u32 characters
    content: [u32; MAX_DOC_LENGTH],
}

/// GPU data for overlap result
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GpuOverlapResult {
    /// Document index
    doc_index: u32,
    /// Overlap count with context cells
    overlap_count: u32,
    /// Total overlap score
    overlap_score: f32,
    /// Padding
    _padding: u32,
}

/// GPU-accelerated evidence merger
pub struct GpuEvidenceMerger {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    workgroup_size: u32,
}

impl GpuEvidenceMerger {
    /// Create a new GPU evidence merger
    pub fn new(gpu: &GpuDevice) -> Result<Self, String> {
        let device = gpu.device().clone();
        let queue = gpu.queue().clone();
        let workgroup_size = gpu.capabilities().preferred_workgroup_size;

        // Shader for parallel overlap detection
        let shader_source = include_str!("../shaders/evidence_merge.wgsl");
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Evidence Merge Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Evidence Merge Bind Group Layout"),
            entries: &[
                // Context cells buffer
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
                // Documents buffer
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
                // Cell count (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Output buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("Evidence Merge Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Evidence Merge Pipeline"),
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

    /// Check overlap on GPU
    pub fn check_overlap_gpu(
        &self,
        context: &ContextMatrix,
        documents: &[RetrievedDocument],
    ) -> Result<Vec<GpuOverlapResult>, String> {
        let doc_count = documents.len().min(MAX_DOCUMENTS);
        
        // Prepare context cells
        let mut cells = Vec::with_capacity(MAX_CONTEXT_CELLS);
        for cell in context.cells.iter().take(MAX_CONTEXT_CELLS) {
            let mut content = [0u32; MAX_CELL_LENGTH];
            let cell_lower = cell.content.to_lowercase();
            let bytes = cell_lower.as_bytes();
            for (i, &byte) in bytes.iter().enumerate().take(MAX_CELL_LENGTH) {
                content[i] = byte as u32;
            }
            cells.push(GpuContextCell {
                length: bytes.len().min(MAX_CELL_LENGTH) as u32,
                _padding: [0; 3],
                content,
            });
        }
        // Pad remaining cells
        while cells.len() < MAX_CONTEXT_CELLS {
            cells.push(GpuContextCell {
                length: 0,
                _padding: [0; 3],
                content: [0; MAX_CELL_LENGTH],
            });
        }

        // Prepare documents
        let mut docs = Vec::with_capacity(MAX_DOCUMENTS);
        for doc in documents.iter().take(MAX_DOCUMENTS) {
            let mut content = [0u32; MAX_DOC_LENGTH];
            let doc_lower = doc.normalized_content.to_lowercase();
            let bytes = doc_lower.as_bytes();
            for (i, &byte) in bytes.iter().enumerate().take(MAX_DOC_LENGTH) {
                content[i] = byte as u32;
            }
            docs.push(GpuDocument {
                length: bytes.len().min(MAX_DOC_LENGTH) as u32,
                trust_score: doc.trust_score,
                _padding: [0; 2],
                content,
            });
        }
        // Pad remaining docs
        while docs.len() < MAX_DOCUMENTS {
            docs.push(GpuDocument {
                length: 0,
                trust_score: 0.0,
                _padding: [0; 2],
                content: [0; MAX_DOC_LENGTH],
            });
        }

        // Create buffers
        let cells_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Context Cells Buffer"),
            size: (MAX_CONTEXT_CELLS * std::mem::size_of::<GpuContextCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let docs_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Documents Buffer"),
            size: (MAX_DOCUMENTS * std::mem::size_of::<GpuDocument>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Buffer"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Overlap Output Buffer"),
            size: (MAX_DOCUMENTS * std::mem::size_of::<GpuOverlapResult>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Overlap Staging Buffer"),
            size: (MAX_DOCUMENTS * std::mem::size_of::<GpuOverlapResult>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        for (i, cell) in cells.iter().enumerate() {
            let offset = (i * std::mem::size_of::<GpuContextCell>()) as u64;
            self.queue.write_buffer(&cells_buffer, offset, bytemuck::bytes_of(cell));
        }

        for (i, doc) in docs.iter().enumerate() {
            let offset = (i * std::mem::size_of::<GpuDocument>()) as u64;
            self.queue.write_buffer(&docs_buffer, offset, bytemuck::bytes_of(doc));
        }

        let counts: [u32; 4] = [
            context.cells.len().min(MAX_CONTEXT_CELLS) as u32,
            doc_count as u32,
            0, 0
        ];
        self.queue.write_buffer(&count_buffer, 0, bytemuck::bytes_of(&counts));

        // Create bind group
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Evidence Merge Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cells_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: docs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Evidence Merge Encoder"),
        });

        // Run compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Evidence Merge Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroups = (doc_count as u32 + self.workgroup_size - 1) / self.workgroup_size;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            (MAX_DOCUMENTS * std::mem::size_of::<GpuOverlapResult>()) as u64,
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
        let results: Vec<GpuOverlapResult> = data
            .chunks_exact(std::mem::size_of::<GpuOverlapResult>())
            .take(doc_count)
            .map(|chunk| *bytemuck::from_bytes(chunk))
            .collect();

        Ok(results)
    }
}

/// Check evidence overlap using GPU if available, otherwise CPU
pub fn check_overlap_gpu(
    context: &ContextMatrix,
    documents: &[RetrievedDocument],
) -> Option<Vec<(usize, bool)>> {
    // GPU is beneficial for large document sets
    if documents.len() < 16 || !is_gpu_available() {
        return None;
    }

    if let Some(merger) = get_gpu_evidence_merger() {
        match merger.check_overlap_gpu(context, documents) {
            Ok(results) => {
                return Some(results
                    .into_iter()
                    .map(|r| (r.doc_index as usize, r.overlap_count > 0))
                    .collect());
            }
            Err(e) => {
                log::warn!("GPU evidence merge failed: {}", e);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_context_cell_size() {
        assert_eq!(std::mem::size_of::<GpuContextCell>(), 528);
    }

    #[test]
    fn gpu_document_size() {
        assert_eq!(std::mem::size_of::<GpuDocument>(), 8208);
    }

    #[test]
    fn gpu_overlap_result_size() {
        assert_eq!(std::mem::size_of::<GpuOverlapResult>(), 16);
    }
}
