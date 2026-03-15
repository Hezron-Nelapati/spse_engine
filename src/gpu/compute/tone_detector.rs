//! GPU-Accelerated Tone Detection (DEPRECATED)
//!
//! Tone detection is now handled by the ClassificationCalculator.
//! This module is retained as a stub to preserve module structure.
//! GPU classification is available via gpu/compute/classification.rs.

use crate::gpu::device::GpuDevice;

/// GPU-accelerated tone detector (deprecated — classification system is primary)
pub struct GpuToneDetector;

impl GpuToneDetector {
    pub fn new(_gpu: &GpuDevice) -> Result<Self, String> {
        Err("GpuToneDetector is deprecated; use ClassificationCalculator".to_string())
    }
}
