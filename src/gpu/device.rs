//! GPU Device Detection and Management
//!
//! Handles GPU device discovery, capability detection, and resource management.

use std::sync::Arc;
use wgpu::{Backends, Device, Instance, PowerPreference, Queue, RequestAdapterOptions};

/// GPU device wrapper with capabilities info
pub struct GpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    capabilities: GpuCapabilities,
}

/// GPU capabilities and limits
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// Device name (e.g., "Apple M1", "NVIDIA RTX 3080")
    pub name: String,
    /// Backend type (Metal, Vulkan, DX12, etc.)
    pub backend: String,
    /// Total GPU memory in bytes (approximate)
    pub memory_bytes: u64,
    /// Maximum buffer size
    pub max_buffer_size: u64,
    /// Maximum workgroup size
    pub max_workgroup_size: u32,
    /// Supports storage buffers
    pub supports_storage_buffers: bool,
    /// Supports compute shaders
    pub supports_compute: bool,
    /// Preferred workgroup size for this device
    pub preferred_workgroup_size: u32,
}

/// GPU configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GpuConfig {
    /// Enable GPU acceleration (default: true if available)
    pub enabled: bool,
    /// Force CPU mode even if GPU available
    pub force_cpu: bool,
    /// Power preference (low, high)
    pub power_preference: GpuPowerPreference,
    /// Minimum GPU memory required (bytes)
    pub min_memory_bytes: u64,
    /// Batch size for GPU operations
    pub batch_size: usize,
    /// Timeout for GPU operations (ms)
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuPowerPreference {
    Low,
    High,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            force_cpu: false,
            power_preference: GpuPowerPreference::High,
            min_memory_bytes: 512 * 1024 * 1024, // 512MB minimum
            batch_size: 1024,
            timeout_ms: 5000,
        }
    }
}

impl GpuDevice {
    /// Create a new GPU device instance
    pub async fn new() -> Result<Self, String> {
        Self::new_with_config(GpuConfig::default()).await
    }

    /// Create GPU device with specific configuration
    pub async fn new_with_config(config: GpuConfig) -> Result<Self, String> {
        if config.force_cpu {
            return Err("GPU force-disabled by configuration".to_string());
        }

        // Create instance with all backends (wgpu v24 API)
        let descriptor = wgpu::InstanceDescriptor {
            backends: Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
        };
        let instance = Instance::new(&descriptor);

        // Request adapter with power preference
        let power_pref = match config.power_preference {
            GpuPowerPreference::Low => PowerPreference::LowPower,
            GpuPowerPreference::High => PowerPreference::HighPerformance,
        };

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: power_pref,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "No suitable GPU adapter found".to_string())?;

        // Get adapter info
        let info = adapter.get_info();

        // Request device with appropriate limits (wgpu v24 API)
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("SPSE Engine GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create GPU device: {}", e))?;

        let limits = device.limits();

        let capabilities = GpuCapabilities {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            memory_bytes: estimate_gpu_memory(&info),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: limits.max_compute_workgroup_size_x,
            supports_storage_buffers: limits.max_storage_buffer_binding_size > 0,
            supports_compute: true, // wgpu adapters always support compute
            preferred_workgroup_size: optimal_workgroup_size(&info),
        };

        // Check minimum requirements
        if capabilities.memory_bytes < config.min_memory_bytes {
            return Err(format!(
                "GPU memory too low: {} MB (need {} MB)",
                capabilities.memory_bytes / 1_048_576,
                config.min_memory_bytes / 1_048_576
            ));
        }

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        })
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &GpuCapabilities {
        &self.capabilities
    }

    /// Get human-readable info string
    pub fn info_string(&self) -> String {
        format!(
            "{} ({}, {} MB, workgroup: {})",
            self.capabilities.name,
            self.capabilities.backend,
            self.capabilities.memory_bytes / 1_048_576,
            self.capabilities.preferred_workgroup_size
        )
    }
}

/// Estimate GPU memory based on device info
fn estimate_gpu_memory(info: &wgpu::AdapterInfo) -> u64 {
    // Heuristic estimates based on device type
    match info.device_type {
        wgpu::DeviceType::IntegratedGpu => {
            // Integrated GPUs share system memory, estimate 25% of typical system
            4 * 1024 * 1024 * 1024 // 4GB estimate
        }
        wgpu::DeviceType::DiscreteGpu => {
            // Discrete GPUs have dedicated VRAM
            // Estimate based on name patterns
            if info.name.contains("RTX 40") || info.name.contains("RX 7") {
                16 * 1024 * 1024 * 1024 // 16GB
            } else if info.name.contains("RTX 30") || info.name.contains("RX 6") {
                12 * 1024 * 1024 * 1024 // 12GB
            } else if info.name.contains("RTX 20") || info.name.contains("GTX 16") {
                8 * 1024 * 1024 * 1024 // 8GB
            } else {
                4 * 1024 * 1024 * 1024 // 4GB default
            }
        }
        wgpu::DeviceType::VirtualGpu => {
            2 * 1024 * 1024 * 1024 // 2GB for virtual
        }
        wgpu::DeviceType::Cpu => {
            0 // No GPU memory
        }
        _ => {
            2 * 1024 * 1024 * 1024 // 2GB default
        }
    }
}

/// Determine optimal workgroup size for device
fn optimal_workgroup_size(info: &wgpu::AdapterInfo) -> u32 {
    // Most GPUs work well with 256 threads per workgroup
    // Apple Silicon prefers 32 or 64
    // NVIDIA/AMD typically prefer 256 or 512

    if info.name.contains("Apple")
        || info.name.contains("M1")
        || info.name.contains("M2")
        || info.name.contains("M3")
    {
        64
    } else if info.backend == wgpu::Backend::Metal {
        64
    } else {
        256
    }
}
