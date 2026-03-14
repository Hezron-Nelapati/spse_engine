pub mod hot_store;
pub mod latency;
pub mod test_observer;
pub mod trace;
pub mod worker;

// Re-export key types for convenience
pub use hot_store::{HotStore, HotStoreConfig, ReasoningStepRecord};
pub use latency::{
    LatencyMonitor, LatencyMonitorConfig, LatencySummary, LatencyTimer, LayerLatencyMetrics,
    LayerLatencySummary, DEFAULT_ALERT_THRESHOLD_MS,
};
pub use trace::{ReasoningTrace, SessionId, TraceContext, TraceId};
pub use worker::{
    AppendOnlyLog, SqliteHotStore, TelemetryEvent, TelemetryWorker, TelemetryWorkerConfig,
};
