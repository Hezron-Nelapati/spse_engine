//! Latency Monitoring for Phase 4.2
//!
//! Tracks p50, p95, p99 latencies across layers with alert thresholds
//! for low-spec devices. Emits telemetry events when thresholds exceeded.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Default alert threshold for low-spec devices (200ms)
pub const DEFAULT_ALERT_THRESHOLD_MS: u64 = 200;

/// Latency metrics for a single layer
#[derive(Debug, Clone, Default)]
pub struct LayerLatencyMetrics {
    /// Total operations recorded
    pub total_ops: u64,
    /// Sum of all latencies (for mean calculation)
    pub total_latency_ms: u64,
    /// Maximum latency observed
    pub max_latency_ms: u64,
    /// Minimum latency observed
    pub min_latency_ms: u64,
    /// Recent latencies for percentile calculation
    recent_latencies: VecDeque<u64>,
    /// Window size for percentile calculation
    window_size: usize,
}

impl LayerLatencyMetrics {
    pub fn new(window_size: usize) -> Self {
        Self {
            total_ops: 0,
            total_latency_ms: 0,
            max_latency_ms: 0,
            min_latency_ms: u64::MAX,
            recent_latencies: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Record a latency measurement
    pub fn record(&mut self, latency_ms: u64) {
        self.total_ops += 1;
        self.total_latency_ms += latency_ms;
        self.max_latency_ms = self.max_latency_ms.max(latency_ms);
        self.min_latency_ms = self.min_latency_ms.min(latency_ms);

        // Maintain sliding window
        if self.recent_latencies.len() >= self.window_size {
            self.recent_latencies.pop_front();
        }
        self.recent_latencies.push_back(latency_ms);
    }

    /// Calculate mean latency
    pub fn mean_ms(&self) -> f64 {
        if self.total_ops == 0 {
            return 0.0;
        }
        self.total_latency_ms as f64 / self.total_ops as f64
    }

    /// Calculate percentile from recent latencies
    pub fn percentile(&self, p: f64) -> u64 {
        if self.recent_latencies.is_empty() {
            return 0;
        }

        let mut sorted: Vec<u64> = self.recent_latencies.iter().copied().collect();
        sorted.sort_unstable();

        let index = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[index]
    }

    /// Get p50 (median) latency
    pub fn p50(&self) -> u64 {
        self.percentile(50.0)
    }

    /// Get p95 latency
    pub fn p95(&self) -> u64 {
        self.percentile(95.0)
    }

    /// Get p99 latency
    pub fn p99(&self) -> u64 {
        self.percentile(99.0)
    }
}

/// Configuration for latency monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LatencyMonitorConfig {
    /// Alert threshold in milliseconds (default 200ms for low-spec)
    pub alert_threshold_ms: u64,
    /// Window size for percentile calculation
    pub window_size: usize,
    /// Enable latency monitoring
    pub enabled: bool,
    /// Sample rate for latency recording (0.0-1.0)
    pub sample_rate: f32,
}

impl Default for LatencyMonitorConfig {
    fn default() -> Self {
        Self {
            alert_threshold_ms: DEFAULT_ALERT_THRESHOLD_MS,
            window_size: 1000,
            enabled: true,
            sample_rate: 1.0,
        }
    }
}

/// Latency monitor for tracking layer latencies
pub struct LatencyMonitor {
    /// Per-layer metrics (indexed by layer number)
    layer_metrics: Mutex<Vec<LayerLatencyMetrics>>,
    /// Global p50 latency (atomic for fast reads)
    global_p50: AtomicU64,
    /// Global p95 latency (atomic for fast reads)
    global_p95: AtomicU64,
    /// Global p99 latency (atomic for fast reads)
    global_p99: AtomicU64,
    /// Alert threshold
    alert_threshold_ms: u64,
    /// Window size for percentile calculation
    window_size: usize,
    /// Whether monitoring is enabled
    enabled: bool,
    /// Sample rate
    sample_rate: f32,
    /// Total alerts triggered
    alerts_triggered: AtomicU64,
}

impl LatencyMonitor {
    /// Create a new latency monitor
    pub fn new(config: LatencyMonitorConfig) -> Self {
        let layer_count = 22; // Layers 0-21
        let metrics = (0..layer_count)
            .map(|_| LayerLatencyMetrics::new(config.window_size))
            .collect();

        Self {
            layer_metrics: Mutex::new(metrics),
            global_p50: AtomicU64::new(0),
            global_p95: AtomicU64::new(0),
            global_p99: AtomicU64::new(0),
            alert_threshold_ms: config.alert_threshold_ms,
            window_size: config.window_size,
            enabled: config.enabled,
            sample_rate: config.sample_rate,
            alerts_triggered: AtomicU64::new(0),
        }
    }

    /// Record a latency measurement for a layer
    /// Returns true if the latency exceeded the alert threshold
    pub fn record(&self, layer: u8, latency_ms: u64) -> bool {
        if !self.enabled {
            return false;
        }

        // Apply sample rate
        if self.sample_rate < 1.0 && rand::random::<f32>() > self.sample_rate {
            return false;
        }

        // Record to layer metrics
        if let Ok(mut metrics) = self.layer_metrics.lock() {
            if (layer as usize) < metrics.len() {
                metrics[layer as usize].record(latency_ms);
                self.update_global_percentiles(&metrics);
            }
        }

        // Check alert threshold
        if latency_ms > self.alert_threshold_ms {
            self.alerts_triggered.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Update global percentile values from all layers
    fn update_global_percentiles(&self, metrics: &[LayerLatencyMetrics]) {
        // Collect all recent latencies across layers
        let mut all_latencies: Vec<u64> = metrics
            .iter()
            .flat_map(|m| m.recent_latencies.iter().copied())
            .collect();

        if all_latencies.is_empty() {
            return;
        }

        all_latencies.sort_unstable();

        let p50_idx = ((50.0 / 100.0) * (all_latencies.len() - 1) as f64).round() as usize;
        let p95_idx = ((95.0 / 100.0) * (all_latencies.len() - 1) as f64).round() as usize;
        let p99_idx = ((99.0 / 100.0) * (all_latencies.len() - 1) as f64).round() as usize;

        self.global_p50.store(all_latencies[p50_idx], Ordering::Relaxed);
        self.global_p95.store(all_latencies[p95_idx], Ordering::Relaxed);
        self.global_p99.store(all_latencies[p99_idx], Ordering::Relaxed);
    }

    /// Get global p50 latency
    pub fn p50(&self) -> u64 {
        self.global_p50.load(Ordering::Relaxed)
    }

    /// Get global p95 latency
    pub fn p95(&self) -> u64 {
        self.global_p95.load(Ordering::Relaxed)
    }

    /// Get global p99 latency
    pub fn p99(&self) -> u64 {
        self.global_p99.load(Ordering::Relaxed)
    }

    /// Get metrics for a specific layer
    pub fn layer_metrics(&self, layer: u8) -> Option<LayerLatencyMetrics> {
        if let Ok(metrics) = self.layer_metrics.lock() {
            if (layer as usize) < metrics.len() {
                return Some(metrics[layer as usize].clone());
            }
        }
        None
    }

    /// Get total alerts triggered
    pub fn alerts_triggered(&self) -> u64 {
        self.alerts_triggered.load(Ordering::Relaxed)
    }

    /// Get alert threshold
    pub fn alert_threshold(&self) -> u64 {
        self.alert_threshold_ms
    }

    /// Check if a latency would trigger an alert
    pub fn would_alert(&self, latency_ms: u64) -> bool {
        latency_ms > self.alert_threshold_ms
    }

    /// Get a summary of all layer latencies
    pub fn summary(&self) -> LatencySummary {
        let mut layer_summaries = Vec::new();

        if let Ok(metrics) = self.layer_metrics.lock() {
            for (i, m) in metrics.iter().enumerate() {
                if m.total_ops > 0 {
                    layer_summaries.push(LayerLatencySummary {
                        layer: i as u8,
                        mean_ms: m.mean_ms(),
                        p50_ms: m.p50(),
                        p95_ms: m.p95(),
                        p99_ms: m.p99(),
                        max_ms: m.max_latency_ms,
                        min_ms: m.min_latency_ms,
                        total_ops: m.total_ops,
                    });
                }
            }
        }

        LatencySummary {
            global_p50_ms: self.p50(),
            global_p95_ms: self.p95(),
            global_p99_ms: self.p99(),
            alert_threshold_ms: self.alert_threshold_ms,
            alerts_triggered: self.alerts_triggered(),
            layers: layer_summaries,
        }
    }
}

/// Summary of latency metrics for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    /// Global p50 latency
    pub global_p50_ms: u64,
    /// Global p95 latency
    pub global_p95_ms: u64,
    /// Global p99 latency
    pub global_p99_ms: u64,
    /// Alert threshold
    pub alert_threshold_ms: u64,
    /// Total alerts triggered
    pub alerts_triggered: u64,
    /// Per-layer summaries
    pub layers: Vec<LayerLatencySummary>,
}

/// Summary for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLatencySummary {
    /// Layer number
    pub layer: u8,
    /// Mean latency
    pub mean_ms: f64,
    /// P50 latency
    pub p50_ms: u64,
    /// P95 latency
    pub p95_ms: u64,
    /// P99 latency
    pub p99_ms: u64,
    /// Maximum latency
    pub max_ms: u64,
    /// Minimum latency
    pub min_ms: u64,
    /// Total operations
    pub total_ops: u64,
}

/// RAII guard for timing an operation
pub struct LatencyTimer<'a> {
    monitor: &'a LatencyMonitor,
    layer: u8,
    start: Instant,
}

impl<'a> LatencyTimer<'a> {
    /// Create a new latency timer
    pub fn new(monitor: &'a LatencyMonitor, layer: u8) -> Self {
        Self {
            monitor,
            layer,
            start: Instant::now(),
        }
    }
}

impl Drop for LatencyTimer<'_> {
    fn drop(&mut self) {
        let latency_ms = self.start.elapsed().as_millis() as u64;
        self.monitor.record(self.layer, latency_ms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_latency_metrics() {
        let mut metrics = LayerLatencyMetrics::new(100);

        metrics.record(10);
        metrics.record(20);
        metrics.record(30);
        metrics.record(40);
        metrics.record(50);

        assert_eq!(metrics.total_ops, 5);
        assert_eq!(metrics.max_latency_ms, 50);
        assert_eq!(metrics.min_latency_ms, 10);
        assert_eq!(metrics.mean_ms(), 30.0);
    }

    #[test]
    fn test_latency_monitor() {
        let config = LatencyMonitorConfig {
            alert_threshold_ms: 100,
            window_size: 100,
            enabled: true,
            sample_rate: 1.0,
        };

        let monitor = LatencyMonitor::new(config);

        // Record some latencies
        let alert1 = monitor.record(5, 50);  // Below threshold
        let alert2 = monitor.record(5, 150); // Above threshold

        assert!(!alert1);
        assert!(alert2);
        assert_eq!(monitor.alerts_triggered(), 1);
    }

    #[test]
    fn test_latency_timer() {
        let config = LatencyMonitorConfig::default();
        let monitor = LatencyMonitor::new(config);

        {
            let _timer = LatencyTimer::new(&monitor, 5);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        let metrics = monitor.layer_metrics(5).unwrap();
        assert!(metrics.total_ops > 0);
        assert!(metrics.mean_ms() >= 10.0);
    }

    #[test]
    fn test_latency_summary() {
        let config = LatencyMonitorConfig::default();
        let monitor = LatencyMonitor::new(config);

        monitor.record(2, 10);
        monitor.record(2, 20);
        monitor.record(5, 30);
        monitor.record(5, 40);

        let summary = monitor.summary();
        assert!(summary.layers.len() >= 2);
    }
}
