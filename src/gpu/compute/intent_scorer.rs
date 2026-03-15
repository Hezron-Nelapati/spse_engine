//! GPU-Accelerated Intent Scoring (DEPRECATED)
//!
//! Intent scoring is now handled by the ClassificationCalculator.
//! This module is retained as a stub to preserve module structure.
//! GPU classification is available via gpu/compute/classification.rs.

use crate::gpu::device::GpuDevice;

/// GPU-accelerated intent scorer (deprecated — classification system is primary)
pub struct GpuIntentScorer;

impl GpuIntentScorer {
    pub fn new(_gpu: &GpuDevice) -> Result<Self, String> {
        Err("GpuIntentScorer is deprecated; use ClassificationCalculator".to_string())
    }
}
