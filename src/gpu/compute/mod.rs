//! GPU Compute Operations
//!
//! Implements GPU-accelerated versions of compute-intensive operations.

pub mod candidate_scorer;
pub mod force_layout;
pub mod distance;

pub use candidate_scorer::GpuCandidateScorer;
pub use force_layout::GpuForceLayout;
pub use distance::GpuDistanceCalculator;

use std::sync::Arc;
use wgpu::{Device, Queue, Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePass};

use crate::gpu::device::GpuDevice;

/// Helper to create a GPU buffer
fn create_buffer(device: &Device, label: &str, size: u64, usage: BufferUsages) -> Buffer {
    device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Helper to create a staging buffer for CPU readback
fn create_staging_buffer(device: &Device, size: u64) -> Buffer {
    create_buffer(
        device,
        "Staging Buffer",
        size,
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    )
}

/// Helper to create a storage buffer for GPU data
fn create_storage_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    create_buffer(
        device,
        label,
        size,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    )
}
