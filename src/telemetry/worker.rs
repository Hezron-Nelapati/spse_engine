//! Async Telemetry Worker for Layer 20
//!
//! Non-blocking worker that emits structured JSON events to hot SQLite store
//! and cold append-only log files. Captures reasoning steps for LLM-like core.

use crate::types::{IntentKind, MemoryChannel, MemoryType};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::TryRecvError;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use uuid::Uuid;

/// Telemetry event types for Layer 20 logging
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryEvent {
    /// Calculation event at layer boundary
    Calculation {
        layer: u8,
        operation: String,
        duration_ms: u64,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Database push event
    DbPush {
        unit_id: Uuid,
        memory_type: MemoryType,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Retrieval event
    Retrieval {
        source: String,
        results: usize,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Morph action event
    MorphAction {
        action: String,
        before: String,
        after: String,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Intent label event with channel tracking
    IntentLabel {
        label: IntentKind,
        channel: MemoryChannel,
        score: f32,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Reasoning step event (Phase 4.1)
    ReasoningStep {
        step: usize,
        thought: String,
        confidence: f32,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Latency spike event (Phase 4.2)
    LatencySpike {
        layer: u8,
        latency_ms: u64,
        threshold_ms: u64,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Memory allocation event (Phase 4.2)
    MemoryAllocation {
        allocated_kb: usize,
        total_kb: usize,
        limit_kb: usize,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Process anchor protected from pruning (Phase 5)
    ProcessAnchorProtected {
        unit_id: Uuid,
        structure_hash: u64,
        utility_score: f32,
        session_id: Uuid,
        trace_id: Uuid,
    },
}

/// Configuration for telemetry worker
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryWorkerConfig {
    /// Enable async telemetry worker
    pub enabled: bool,
    /// Hot store path (SQLite)
    pub hot_store_path: String,
    /// Cold log path (append-only file)
    pub cold_log_path: String,
    /// Batch size for flushing events
    pub batch_size: usize,
    /// Flush interval in milliseconds
    pub flush_interval_ms: u64,
    /// Maximum channel capacity before backpressure
    pub channel_capacity: usize,
    /// Sample rate for high-frequency events (0.0-1.0)
    pub sample_rate: f32,
}

impl Default for TelemetryWorkerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            hot_store_path: "telemetry/hot.db".to_string(),
            cold_log_path: "telemetry/cold.log".to_string(),
            batch_size: 100,
            flush_interval_ms: 100,
            channel_capacity: 10000,
            sample_rate: 1.0,
        }
    }
}

/// Hot store for real-time UI queries
#[allow(dead_code)]
pub struct SqliteHotStore {
    path: String,
    connection: Option<rusqlite::Connection>,
}

impl SqliteHotStore {
    pub fn new(path: &str) -> Result<Self, String> {
        let connection = Self::create_connection(path)?;
        Ok(Self {
            path: path.to_string(),
            connection: Some(connection),
        })
    }

    fn create_connection(path: &str) -> Result<rusqlite::Connection, String> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create telemetry directory: {e}"))?;
            }
        }

        let conn = rusqlite::Connection::open(path)
            .map_err(|e| format!("Failed to open hot store: {e}"))?;

        // Create schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                session_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_session_id ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_trace_id ON events(trace_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
            "#,
        )
        .map_err(|e| format!("Failed to create schema: {e}"))?;

        Ok(conn)
    }

    pub fn insert_event(&self, event: &TelemetryEvent) -> Result<(), String> {
        let conn = self
            .connection
            .as_ref()
            .ok_or_else(|| "Connection not initialized".to_string())?;

        let (event_type, session_id, trace_id) = match event {
            TelemetryEvent::Calculation {
                session_id,
                trace_id,
                ..
            } => ("calculation", *session_id, *trace_id),
            TelemetryEvent::DbPush {
                session_id,
                trace_id,
                ..
            } => ("db_push", *session_id, *trace_id),
            TelemetryEvent::Retrieval {
                session_id,
                trace_id,
                ..
            } => ("retrieval", *session_id, *trace_id),
            TelemetryEvent::MorphAction {
                session_id,
                trace_id,
                ..
            } => ("morph_action", *session_id, *trace_id),
            TelemetryEvent::IntentLabel {
                session_id,
                trace_id,
                ..
            } => ("intent_label", *session_id, *trace_id),
            TelemetryEvent::ReasoningStep {
                session_id,
                trace_id,
                ..
            } => ("reasoning_step", *session_id, *trace_id),
            TelemetryEvent::LatencySpike {
                session_id,
                trace_id,
                ..
            } => ("latency_spike", *session_id, *trace_id),
            TelemetryEvent::MemoryAllocation {
                session_id,
                trace_id,
                ..
            } => ("memory_allocation", *session_id, *trace_id),
            TelemetryEvent::ProcessAnchorProtected {
                session_id,
                trace_id,
                ..
            } => ("process_anchor_protected", *session_id, *trace_id),
        };

        let event_data =
            serde_json::to_string(event).map_err(|e| format!("Failed to serialize event: {e}"))?;

        conn.execute(
            "INSERT INTO events (event_type, event_data, session_id, trace_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![event_type, event_data, session_id.to_string(), trace_id.to_string()],
        )
        .map_err(|e| format!("Failed to insert event: {e}"))?;

        Ok(())
    }

    pub fn query_by_trace(&self, trace_id: Uuid) -> Result<Vec<TelemetryEvent>, String> {
        let conn = self
            .connection
            .as_ref()
            .ok_or_else(|| "Connection not initialized".to_string())?;

        let mut stmt = conn
            .prepare("SELECT event_data FROM events WHERE trace_id = ?1 ORDER BY id")
            .map_err(|e| format!("Failed to prepare query: {e}"))?;

        let events = stmt
            .query_map(rusqlite::params![trace_id.to_string()], |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str::<TelemetryEvent>(&data))
            })
            .map_err(|e| format!("Failed to query events: {e}"))?;

        Ok(events
            .filter_map(|r| r.ok())
            .filter_map(|r| r.ok())
            .collect())
    }
}

/// Cold log for long-term storage
#[allow(dead_code)]
pub struct AppendOnlyLog {
    path: String,
    file: Option<std::fs::File>,
}

impl AppendOnlyLog {
    pub fn new(path: &str) -> Result<Self, String> {
        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create cold log directory: {e}"))?;
            }
        }

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("Failed to open cold log: {e}"))?;

        Ok(Self {
            path: path.to_string(),
            file: Some(file),
        })
    }

    pub fn append(&mut self, event: &TelemetryEvent) -> Result<(), String> {
        use std::io::Write;

        let file = self
            .file
            .as_mut()
            .ok_or_else(|| "File not initialized".to_string())?;

        let json =
            serde_json::to_string(event).map_err(|e| format!("Failed to serialize event: {e}"))?;

        writeln!(file, "{json}").map_err(|e| format!("Failed to write to cold log: {e}"))?;

        Ok(())
    }
}

/// Async telemetry worker
pub struct TelemetryWorker {
    sender: std::sync::mpsc::SyncSender<TelemetryEvent>,
    handle: Option<JoinHandle<()>>,
    config: TelemetryWorkerConfig,
}

impl TelemetryWorker {
    /// Create a new telemetry worker with async processing
    pub fn new(config: TelemetryWorkerConfig) -> Result<Self, String> {
        if !config.enabled {
            // Return a dummy worker that drops events
            let (sender, _receiver) = std::sync::mpsc::sync_channel(1);
            return Ok(Self {
                sender,
                handle: None,
                config,
            });
        }

        let (sender, receiver): (std::sync::mpsc::SyncSender<TelemetryEvent>, _) =
            std::sync::mpsc::sync_channel(config.channel_capacity);

        let hot_store_path = config.hot_store_path.clone();
        let cold_log_path = config.cold_log_path.clone();
        let batch_size = config.batch_size;
        let flush_interval_ms = config.flush_interval_ms;

        let handle = thread::spawn(move || {
            let hot_store = match SqliteHotStore::new(&hot_store_path) {
                Ok(store) => store,
                Err(e) => {
                    eprintln!("Failed to initialize hot store: {e}");
                    return;
                }
            };

            let mut cold_log = match AppendOnlyLog::new(&cold_log_path) {
                Ok(log) => log,
                Err(e) => {
                    eprintln!("Failed to initialize cold log: {e}");
                    return;
                }
            };

            let mut batch: Vec<TelemetryEvent> = Vec::with_capacity(batch_size);
            let flush_duration = Duration::from_millis(flush_interval_ms);

            loop {
                // Try to receive with timeout
                match receiver.try_recv() {
                    Ok(event) => {
                        batch.push(event);
                        if batch.len() >= batch_size {
                            Self::flush_batch(&hot_store, &mut cold_log, &mut batch);
                        }
                    }
                    Err(TryRecvError::Empty) => {
                        // Flush any pending events
                        if !batch.is_empty() {
                            Self::flush_batch(&hot_store, &mut cold_log, &mut batch);
                        }
                        // Wait before trying again
                        thread::sleep(flush_duration);
                    }
                    Err(TryRecvError::Disconnected) => {
                        // Channel closed, flush remaining and exit
                        if !batch.is_empty() {
                            Self::flush_batch(&hot_store, &mut cold_log, &mut batch);
                        }
                        break;
                    }
                }
            }
        });

        Ok(Self {
            sender,
            handle: Some(handle),
            config,
        })
    }

    fn flush_batch(
        hot_store: &SqliteHotStore,
        cold_log: &mut AppendOnlyLog,
        batch: &mut Vec<TelemetryEvent>,
    ) {
        for event in batch.drain(..) {
            if let Err(e) = hot_store.insert_event(&event) {
                eprintln!("Failed to insert event to hot store: {e}");
            }
            if let Err(e) = cold_log.append(&event) {
                eprintln!("Failed to append event to cold log: {e}");
            }
        }
    }

    /// Emit a telemetry event (non-blocking)
    pub fn emit(&self, event: TelemetryEvent) -> Result<(), String> {
        if !self.config.enabled {
            return Ok(());
        }

        // Apply sample rate for high-frequency events
        if self.config.sample_rate < 1.0 {
            match &event {
                TelemetryEvent::Calculation { .. } => {
                    if rand::random::<f32>() > self.config.sample_rate {
                        return Ok(());
                    }
                }
                _ => {}
            }
        }

        self.sender
            .send(event)
            .map_err(|e| format!("Failed to send telemetry event: {e}"))?;

        Ok(())
    }

    /// Check if the worker channel is full (backpressure)
    pub fn is_backpressured(&self) -> bool {
        // If we can send without blocking, there's no backpressure
        self.sender
            .try_send(TelemetryEvent::Calculation {
                layer: 0,
                operation: "ping".to_string(),
                duration_ms: 0,
                session_id: Uuid::nil(),
                trace_id: Uuid::nil(),
            })
            .is_err()
    }

    /// Shutdown the worker gracefully
    pub fn shutdown(mut self) {
        // Drop sender to close channel
        drop(self.sender);

        // Wait for worker thread to finish
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_telemetry_event_serialization() {
        let event = TelemetryEvent::ReasoningStep {
            step: 1,
            thought: "Analyzing query structure".to_string(),
            confidence: 0.45,
            session_id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("reasoning_step"));
        assert!(json.contains("Analyzing query structure"));
    }

    #[test]
    fn test_hot_store_insert_and_query() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hot.db").to_str().unwrap().to_string();

        let store = SqliteHotStore::new(&path).unwrap();
        let session_id = Uuid::new_v4();
        let trace_id = Uuid::new_v4();

        let event = TelemetryEvent::Calculation {
            layer: 5,
            operation: "routing".to_string(),
            duration_ms: 42,
            session_id,
            trace_id,
        };

        store.insert_event(&event).unwrap();

        let events = store.query_by_trace(trace_id).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_worker_emit_event() {
        let dir = tempdir().unwrap();
        let config = TelemetryWorkerConfig {
            enabled: true,
            hot_store_path: dir.path().join("hot.db").to_str().unwrap().to_string(),
            cold_log_path: dir.path().join("cold.log").to_str().unwrap().to_string(),
            batch_size: 10,
            flush_interval_ms: 10,
            channel_capacity: 100,
            sample_rate: 1.0,
        };

        let worker = TelemetryWorker::new(config).unwrap();

        let event = TelemetryEvent::ReasoningStep {
            step: 0,
            thought: "Initial thought".to_string(),
            confidence: 0.35,
            session_id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
        };

        worker.emit(event).unwrap();

        // Give worker time to process
        thread::sleep(Duration::from_millis(100));

        worker.shutdown();
    }

    #[test]
    fn test_worker_disabled() {
        let config = TelemetryWorkerConfig {
            enabled: false,
            ..Default::default()
        };

        let worker = TelemetryWorker::new(config).unwrap();

        let event = TelemetryEvent::Calculation {
            layer: 1,
            operation: "test".to_string(),
            duration_ms: 0,
            session_id: Uuid::new_v4(),
            trace_id: Uuid::new_v4(),
        };

        // Should succeed even though worker is disabled
        worker.emit(event).unwrap();
    }
}
