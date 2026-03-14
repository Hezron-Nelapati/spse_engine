//! Dynamic Memory Allocation for Phase 4.2
//!
//! Manages reasoning memory buffers that are allocated on-demand when reasoning
//! is triggered and released immediately after reasoning completes.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Default thought buffer size in KB
pub const DEFAULT_THOUGHT_BUFFER_SIZE_KB: usize = 64;

/// Configuration for dynamic memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DynamicMemoryConfig {
    /// Enable dynamic memory allocation
    pub enabled: bool,
    /// Base memory limit when idle (MB)
    pub base_memory_limit_mb: usize,
    /// Maximum memory limit during reasoning (MB)
    pub max_memory_limit_mb: usize,
    /// Thought buffer size per reasoning step (KB)
    pub thought_buffer_size_kb: usize,
}

impl Default for DynamicMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_memory_limit_mb: 350,
            max_memory_limit_mb: 550,
            thought_buffer_size_kb: DEFAULT_THOUGHT_BUFFER_SIZE_KB,
        }
    }
}

/// Thought buffer for storing reasoning thoughts
#[derive(Debug, Clone)]
pub struct ThoughtBuffer {
    /// Buffer contents (thought strings)
    pub thoughts: Vec<String>,
    /// Total capacity in bytes
    pub capacity_bytes: usize,
    /// Used bytes
    pub used_bytes: usize,
}

impl ThoughtBuffer {
    /// Create a new thought buffer
    pub fn new(capacity_kb: usize) -> Self {
        Self {
            thoughts: Vec::new(),
            capacity_bytes: capacity_kb * 1024,
            used_bytes: 0,
        }
    }

    /// Add a thought to the buffer
    pub fn add_thought(&mut self, thought: &str) -> Result<(), String> {
        let thought_bytes = thought.len();
        if self.used_bytes + thought_bytes > self.capacity_bytes {
            return Err("Thought buffer capacity exceeded".to_string());
        }

        self.thoughts.push(thought.to_string());
        self.used_bytes += thought_bytes;
        Ok(())
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.thoughts.clear();
        self.used_bytes = 0;
    }

    /// Get the number of thoughts
    pub fn len(&self) -> usize {
        self.thoughts.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.thoughts.is_empty()
    }

    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.capacity_bytes == 0 {
            return 0.0;
        }
        self.used_bytes as f32 / self.capacity_bytes as f32
    }
}

/// Dynamic memory allocator for reasoning buffers
pub struct DynamicMemoryAllocator {
    /// Configuration
    config: DynamicMemoryConfig,
    /// Current memory usage in KB
    current_usage_kb: AtomicUsize,
    /// Whether reasoning mode is active
    reasoning_active: AtomicBool,
    /// Number of allocated thought buffers
    buffer_count: AtomicUsize,
    /// Peak memory usage in KB
    peak_usage_kb: AtomicUsize,
    /// Total allocations
    total_allocations: AtomicUsize,
    /// Total deallocations
    total_deallocations: AtomicUsize,
}

impl DynamicMemoryAllocator {
    /// Create a new dynamic memory allocator
    pub fn new(config: DynamicMemoryConfig) -> Self {
        Self {
            config,
            current_usage_kb: AtomicUsize::new(0),
            reasoning_active: AtomicBool::new(false),
            buffer_count: AtomicUsize::new(0),
            peak_usage_kb: AtomicUsize::new(0),
            total_allocations: AtomicUsize::new(0),
            total_deallocations: AtomicUsize::new(0),
        }
    }

    /// Get the current memory limit based on reasoning state
    pub fn current_limit_kb(&self) -> usize {
        if self.reasoning_active.load(Ordering::Relaxed) {
            self.config.max_memory_limit_mb * 1024
        } else {
            self.config.base_memory_limit_mb * 1024
        }
    }

    /// Enter reasoning mode (increases memory limit)
    pub fn enter_reasoning_mode(&self) {
        self.reasoning_active.store(true, Ordering::Relaxed);
    }

    /// Exit reasoning mode (decreases memory limit)
    pub fn exit_reasoning_mode(&self) {
        self.reasoning_active.store(false, Ordering::Relaxed);
    }

    /// Check if reasoning mode is active
    pub fn is_reasoning_active(&self) -> bool {
        self.reasoning_active.load(Ordering::Relaxed)
    }

    /// Allocate a thought buffer
    /// Returns None if memory limit would be exceeded
    pub fn allocate_thought_buffer(&self) -> Option<ThoughtBuffer> {
        if !self.config.enabled {
            return Some(ThoughtBuffer::new(self.config.thought_buffer_size_kb));
        }

        let buffer_size_kb = self.config.thought_buffer_size_kb;
        let current = self.current_usage_kb.load(Ordering::Relaxed);
        let limit = self.current_limit_kb();

        if current + buffer_size_kb > limit {
            return None;
        }

        // Attempt to allocate
        match self.current_usage_kb.compare_exchange(
            current,
            current + buffer_size_kb,
            Ordering::SeqCst,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                self.buffer_count.fetch_add(1, Ordering::Relaxed);
                self.total_allocations.fetch_add(1, Ordering::Relaxed);

                // Update peak
                let new_usage = current + buffer_size_kb;
                let mut peak = self.peak_usage_kb.load(Ordering::Relaxed);
                while new_usage > peak {
                    match self.peak_usage_kb.compare_exchange_weak(
                        peak,
                        new_usage,
                        Ordering::SeqCst,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }

                Some(ThoughtBuffer::new(buffer_size_kb))
            }
            Err(_) => {
                // Another thread beat us, retry would be needed
                None
            }
        }
    }

    /// Deallocate a thought buffer
    pub fn deallocate_thought_buffer(&self, buffer: ThoughtBuffer) {
        if !self.config.enabled {
            return;
        }

        let buffer_size_kb = self.config.thought_buffer_size_kb;
        self.current_usage_kb.fetch_sub(buffer_size_kb, Ordering::Relaxed);
        self.buffer_count.fetch_sub(1, Ordering::Relaxed);
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);

        // Ensure buffer is dropped
        drop(buffer);
    }

    /// Force release all reasoning memory
    pub fn release_all_reasoning_memory(&self) -> usize {
        let buffers = self.buffer_count.load(Ordering::Relaxed);
        let buffer_size = self.config.thought_buffer_size_kb;

        // Reset usage
        let released = self.current_usage_kb.swap(0, Ordering::Relaxed);
        self.buffer_count.store(0, Ordering::Relaxed);

        released
    }

    /// Get current memory usage in KB
    pub fn current_usage_kb(&self) -> usize {
        self.current_usage_kb.load(Ordering::Relaxed)
    }

    /// Get current memory usage in MB
    pub fn current_usage_mb(&self) -> usize {
        self.current_usage_kb.load(Ordering::Relaxed) / 1024
    }

    /// Get peak memory usage in KB
    pub fn peak_usage_kb(&self) -> usize {
        self.peak_usage_kb.load(Ordering::Relaxed)
    }

    /// Get number of active buffers
    pub fn active_buffer_count(&self) -> usize {
        self.buffer_count.load(Ordering::Relaxed)
    }

    /// Get memory utilization percentage
    pub fn utilization(&self) -> f32 {
        let limit = self.current_limit_kb();
        if limit == 0 {
            return 0.0;
        }
        self.current_usage_kb.load(Ordering::Relaxed) as f32 / limit as f32
    }

    /// Check if memory is available for allocation
    pub fn can_allocate(&self) -> bool {
        let current = self.current_usage_kb.load(Ordering::Relaxed);
        let limit = self.current_limit_kb();
        let buffer_size = self.config.thought_buffer_size_kb;
        current + buffer_size <= limit
    }

    /// Get statistics
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage_kb: self.current_usage_kb.load(Ordering::Relaxed),
            peak_usage_kb: self.peak_usage_kb.load(Ordering::Relaxed),
            current_limit_kb: self.current_limit_kb(),
            base_limit_kb: self.config.base_memory_limit_mb * 1024,
            max_limit_kb: self.config.max_memory_limit_mb * 1024,
            active_buffers: self.buffer_count.load(Ordering::Relaxed),
            total_allocations: self.total_allocations.load(Ordering::Relaxed),
            total_deallocations: self.total_deallocations.load(Ordering::Relaxed),
            reasoning_active: self.reasoning_active.load(Ordering::Relaxed),
        }
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage in KB
    pub current_usage_kb: usize,
    /// Peak memory usage in KB
    pub peak_usage_kb: usize,
    /// Current limit in KB
    pub current_limit_kb: usize,
    /// Base limit in KB (idle)
    pub base_limit_kb: usize,
    /// Max limit in KB (reasoning)
    pub max_limit_kb: usize,
    /// Number of active buffers
    pub active_buffers: usize,
    /// Total allocations
    pub total_allocations: usize,
    /// Total deallocations
    pub total_deallocations: usize,
    /// Whether reasoning mode is active
    pub reasoning_active: bool,
}

impl MemoryStats {
    /// Get utilization percentage
    pub fn utilization(&self) -> f32 {
        if self.current_limit_kb == 0 {
            return 0.0;
        }
        self.current_usage_kb as f32 / self.current_limit_kb as f32
    }

    /// Check if memory is healthy (not near limit)
    pub fn is_healthy(&self) -> bool {
        self.utilization() < 0.8
    }
}

/// RAII guard for reasoning mode
pub struct ReasoningGuard<'a> {
    allocator: &'a DynamicMemoryAllocator,
}

impl<'a> ReasoningGuard<'a> {
    /// Create a new reasoning guard
    pub fn new(allocator: &'a DynamicMemoryAllocator) -> Self {
        allocator.enter_reasoning_mode();
        Self { allocator }
    }
}

impl Drop for ReasoningGuard<'_> {
    fn drop(&mut self) {
        self.allocator.exit_reasoning_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_buffer() {
        let mut buffer = ThoughtBuffer::new(1); // 1KB

        buffer.add_thought("First thought").unwrap();
        buffer.add_thought("Second thought").unwrap();

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.is_empty());
        assert!(buffer.utilization() > 0.0);
    }

    #[test]
    fn test_thought_buffer_capacity() {
        let mut buffer = ThoughtBuffer::new(1); // 1KB

        // Try to add a thought larger than capacity
        let large_thought = "x".repeat(2000);
        let result = buffer.add_thought(&large_thought);

        assert!(result.is_err());
    }

    #[test]
    fn test_dynamic_allocator_basic() {
        let config = DynamicMemoryConfig {
            enabled: true,
            base_memory_limit_mb: 10,  // 10MB
            max_memory_limit_mb: 20,   // 20MB
            thought_buffer_size_kb: 64,
        };

        let allocator = DynamicMemoryAllocator::new(config);

        // Should be in idle mode initially
        assert!(!allocator.is_reasoning_active());
        assert_eq!(allocator.current_limit_kb(), 10 * 1024);

        // Allocate a buffer
        let buffer = allocator.allocate_thought_buffer().unwrap();
        assert!(allocator.active_buffer_count() >= 1);

        // Deallocate
        allocator.deallocate_thought_buffer(buffer);
    }

    #[test]
    fn test_reasoning_mode() {
        let config = DynamicMemoryConfig {
            enabled: true,
            base_memory_limit_mb: 10,
            max_memory_limit_mb: 50,
            thought_buffer_size_kb: 64,
        };

        let allocator = DynamicMemoryAllocator::new(config);

        // Enter reasoning mode
        allocator.enter_reasoning_mode();
        assert!(allocator.is_reasoning_active());
        assert_eq!(allocator.current_limit_kb(), 50 * 1024);

        // Exit reasoning mode
        allocator.exit_reasoning_mode();
        assert!(!allocator.is_reasoning_active());
        assert_eq!(allocator.current_limit_kb(), 10 * 1024);
    }

    #[test]
    fn test_memory_limit_enforcement() {
        let config = DynamicMemoryConfig {
            enabled: true,
            base_memory_limit_mb: 1,   // 1MB - very small
            max_memory_limit_mb: 2,    // 2MB
            thought_buffer_size_kb: 512, // 512KB per buffer
        };

        let allocator = DynamicMemoryAllocator::new(config);

        // Should be able to allocate 2 buffers (1MB total) in idle mode
        let buf1 = allocator.allocate_thought_buffer();
        let buf2 = allocator.allocate_thought_buffer();

        // Third should fail (would exceed 1MB limit)
        let buf3 = allocator.allocate_thought_buffer();
        assert!(buf3.is_none());

        // But if we enter reasoning mode, we have 2MB limit
        allocator.enter_reasoning_mode();
        let buf4 = allocator.allocate_thought_buffer();
        assert!(buf4.is_some());

        // Cleanup
        if let Some(b) = buf1 { allocator.deallocate_thought_buffer(b); }
        if let Some(b) = buf2 { allocator.deallocate_thought_buffer(b); }
        if let Some(b) = buf4 { allocator.deallocate_thought_buffer(b); }
    }

    #[test]
    fn test_reasoning_guard() {
        let config = DynamicMemoryConfig::default();
        let allocator = DynamicMemoryAllocator::new(config);

        {
            let _guard = ReasoningGuard::new(&allocator);
            assert!(allocator.is_reasoning_active());
        }

        // Guard dropped, should exit reasoning mode
        assert!(!allocator.is_reasoning_active());
    }

    #[test]
    fn test_memory_stats() {
        let config = DynamicMemoryConfig {
            enabled: true,
            base_memory_limit_mb: 10,
            max_memory_limit_mb: 20,
            thought_buffer_size_kb: 64,
        };

        let allocator = DynamicMemoryAllocator::new(config);

        let _buffer = allocator.allocate_thought_buffer().unwrap();
        let stats = allocator.stats();

        assert!(stats.current_usage_kb > 0);
        assert!(stats.total_allocations >= 1);
        assert!(!stats.reasoning_active);
    }
}
