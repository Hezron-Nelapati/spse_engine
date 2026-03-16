//! GPU Acceleration Module
//!
//! Provides GPU-accelerated operations for compute-intensive tasks:
//! - Candidate scoring (parallel dot products)
//! - Force-directed layout (O(n²) force calculations)
//! - Spatial distance calculations
//! - Batch training operations
//!
//! Uses wgpu for cross-platform GPU support (macOS/Linux/Windows/WebAssembly).
//! Falls back to CPU when GPU is unavailable or feature disabled.

#[cfg(feature = "gpu")]
pub mod compute;
#[cfg(feature = "gpu")]
mod device;

#[cfg(feature = "gpu")]
pub use compute::{GpuCandidateScorer, GpuDistanceCalculator, GpuForceLayout};
#[cfg(feature = "gpu")]
pub use device::{GpuCapabilities, GpuConfig as DeviceGpuConfig, GpuDevice};

#[cfg(feature = "gpu")]
use once_cell::sync::Lazy;
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Global GPU device instance (lazy initialized)
#[cfg(feature = "gpu")]
static GPU_DEVICE: Lazy<Option<Arc<GpuDevice>>> =
    Lazy::new(|| match futures::executor::block_on(GpuDevice::new()) {
        Ok(device) => {
            log::info!("GPU acceleration enabled: {:?}", device.capabilities());
            Some(Arc::new(device))
        }
        Err(e) => {
            log::warn!("GPU acceleration disabled: {}", e);
            None
        }
    });

/// Get the global GPU device if available
#[cfg(feature = "gpu")]
pub fn global_device() -> Option<Arc<GpuDevice>> {
    GPU_DEVICE.clone()
}

/// Check if GPU acceleration is available
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    GPU_DEVICE.is_some()
}

/// Get GPU capabilities description
#[cfg(feature = "gpu")]
pub fn gpu_info() -> Option<String> {
    GPU_DEVICE.as_ref().map(|d| d.info_string())
}

// CPU-only fallbacks when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool {
    false
}

#[cfg(not(feature = "gpu"))]
pub fn gpu_info() -> Option<String> {
    None
}
