//! Hot Store for Layer 20 Telemetry
//!
//! SQLite-based hot store for real-time UI queries of telemetry events.
//! Provides fast access to recent events for debugging and visualization.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

// Helper to convert serde_json errors to string
fn from_json<T: serde::de::DeserializeOwned>(data: &str) -> Result<T, rusqlite::Error> {
    serde_json::from_str(data).map_err(|e| rusqlite::Error::InvalidParameterName(e.to_string()))
}

use super::worker::TelemetryEvent;

/// Hot store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HotStoreConfig {
    /// Path to SQLite database
    pub path: String,
    /// Maximum events to retain (older events pruned)
    pub max_events: usize,
    /// Prune interval in seconds
    pub prune_interval_secs: u64,
}

impl Default for HotStoreConfig {
    fn default() -> Self {
        Self {
            path: "telemetry/hot.db".to_string(),
            max_events: 100_000,
            prune_interval_secs: 300,
        }
    }
}

/// Hot store for telemetry events
pub struct HotStore {
    connection: Mutex<Connection>,
    config: HotStoreConfig,
}

impl HotStore {
    /// Create a new hot store
    pub fn new(config: HotStoreConfig) -> Result<Self, String> {
        let connection = Self::create_connection(&config.path)?;
        Ok(Self {
            connection: Mutex::new(connection),
            config,
        })
    }

    fn create_connection(path: &str) -> Result<Connection, String> {
        // Ensure parent directory exists
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create telemetry directory: {e}"))?;
            }
        }

        let conn = Connection::open(path)
            .map_err(|e| format!("Failed to open hot store: {e}"))?;

        // Enable WAL mode for better concurrency
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
            "#,
        )
        .map_err(|e| format!("Failed to set pragmas: {e}"))?;

        // Create schema
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT NOT NULL,
                session_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                layer INTEGER,
                timestamp TEXT NOT NULL DEFAULT (datetime('now'))
            );
            
            CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_trace_id ON events(trace_id);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_layer ON events(layer);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            "#,
        )
        .map_err(|e| format!("Failed to create schema: {e}"))?;

        Ok(conn)
    }

    /// Insert an event into the hot store
    pub fn insert(&self, event: &TelemetryEvent) -> Result<i64, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let (event_type, layer, session_id, trace_id) = Self::extract_event_metadata(event);

        let event_data = serde_json::to_string(event)
            .map_err(|e| format!("Failed to serialize event: {e}"))?;

        conn.execute(
            "INSERT INTO events (event_type, event_data, session_id, trace_id, layer) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![event_type, event_data, session_id.to_string(), trace_id.to_string(), layer],
        )
        .map_err(|e| format!("Failed to insert event: {e}"))?;

        let id = conn.last_insert_rowid();
        Ok(id)
    }

    /// Extract metadata from a telemetry event
    fn extract_event_metadata(event: &TelemetryEvent) -> (&'static str, Option<u8>, Uuid, Uuid) {
        match event {
            TelemetryEvent::Calculation { layer, session_id, trace_id, .. } => {
                ("calculation", Some(*layer), *session_id, *trace_id)
            }
            TelemetryEvent::DbPush { session_id, trace_id, .. } => {
                ("db_push", Some(4), *session_id, *trace_id)
            }
            TelemetryEvent::Retrieval { session_id, trace_id, .. } => {
                ("retrieval", Some(11), *session_id, *trace_id)
            }
            TelemetryEvent::MorphAction { session_id, trace_id, .. } => {
                ("morph_action", None, *session_id, *trace_id)
            }
            TelemetryEvent::IntentLabel { session_id, trace_id, .. } => {
                ("intent_label", Some(7), *session_id, *trace_id)
            }
            TelemetryEvent::ReasoningStep { session_id, trace_id, .. } => {
                ("reasoning_step", None, *session_id, *trace_id)
            }
            TelemetryEvent::LatencySpike { layer, session_id, trace_id, .. } => {
                ("latency_spike", Some(*layer), *session_id, *trace_id)
            }
            TelemetryEvent::MemoryAllocation { session_id, trace_id, .. } => {
                ("memory_allocation", None, *session_id, *trace_id)
            }
        }
    }

    /// Query events by trace ID
    pub fn query_by_trace(&self, trace_id: Uuid) -> Result<Vec<TelemetryEvent>, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let mut stmt = conn.prepare(
            "SELECT event_data FROM events WHERE trace_id = ?1 ORDER BY id"
        ).map_err(|e| format!("Failed to prepare query: {e}"))?;

        let events = stmt.query_map(
            params![trace_id.to_string()],
            |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str::<TelemetryEvent>(&data))
            }
        ).map_err(|e| format!("Failed to query events: {e}"))?;

        Ok(events
            .filter_map(|r| r.ok())
            .filter_map(|r| r.ok())
            .collect())
    }

    /// Query events by session ID
    pub fn query_by_session(&self, session_id: Uuid) -> Result<Vec<TelemetryEvent>, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let mut stmt = conn.prepare(
            "SELECT event_data FROM events WHERE session_id = ?1 ORDER BY id"
        ).map_err(|e| format!("Failed to prepare query: {e}"))?;

        let events = stmt.query_map(
            params![session_id.to_string()],
            |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str::<TelemetryEvent>(&data))
            }
        ).map_err(|e| format!("Failed to query events: {e}"))?;

        Ok(events
            .filter_map(|r| r.ok())
            .filter_map(|r| r.ok())
            .collect())
    }

    /// Query events by layer
    pub fn query_by_layer(&self, layer: u8, limit: usize) -> Result<Vec<TelemetryEvent>, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let mut stmt = conn.prepare(
            "SELECT event_data FROM events WHERE layer = ?1 ORDER BY id DESC LIMIT ?2"
        ).map_err(|e| format!("Failed to prepare query: {e}"))?;

        let events = stmt.query_map(
            params![layer, limit as i64],
            |row| {
                let data: String = row.get(0)?;
                Ok(serde_json::from_str::<TelemetryEvent>(&data))
            }
        ).map_err(|e| format!("Failed to query events: {e}"))?;

        Ok(events
            .filter_map(|r| r.ok())
            .filter_map(|r| r.ok())
            .collect())
    }

    /// Query reasoning steps for a trace
    pub fn query_reasoning_steps(&self, trace_id: Uuid) -> Result<Vec<ReasoningStepRecord>, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let mut stmt = conn.prepare(
            r#"SELECT event_data FROM events 
               WHERE trace_id = ?1 AND event_type = 'reasoning_step' 
               ORDER BY id"#
        ).map_err(|e| format!("Failed to prepare query: {e}"))?;

        let steps = stmt.query_map(
            params![trace_id.to_string()],
            |row| {
                let data: String = row.get(0)?;
                let event: TelemetryEvent = from_json(&data)?;
                Ok(event)
            }
        ).map_err(|e| format!("Failed to query events: {e}"))?;

        Ok(steps
            .filter_map(|r| r.ok())
            .filter_map(|event| {
                if let TelemetryEvent::ReasoningStep { step, thought, confidence, .. } = event {
                    Some(ReasoningStepRecord {
                        step,
                        thought,
                        confidence,
                    })
                } else {
                    None
                }
            })
            .collect())
    }

    /// Get event count
    pub fn event_count(&self) -> Result<usize, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM events",
            [],
            |row| row.get(0)
        ).map_err(|e| format!("Failed to count events: {e}"))?;

        Ok(count as usize)
    }

    /// Prune old events to maintain max_events limit
    pub fn prune(&self) -> Result<usize, String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        // Count directly without re-locking
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM events",
            [],
            |row| row.get::<_, i64>(0)
        ).map_err(|e| format!("Failed to count events: {e}"))?;

        if count as usize <= self.config.max_events {
            return Ok(0);
        }

        let to_delete = count as usize - self.config.max_events;

        conn.execute(
            "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id LIMIT ?1)",
            params![to_delete as i64]
        ).map_err(|e| format!("Failed to prune events: {e}"))?;

        Ok(to_delete)
    }

    /// Clear all events
    pub fn clear(&self) -> Result<(), String> {
        let conn = self.connection.lock()
            .map_err(|e| format!("Failed to lock connection: {e}"))?;

        conn.execute("DELETE FROM events", [])
            .map_err(|e| format!("Failed to clear events: {e}"))?;

        Ok(())
    }
}

/// Reasoning step record for easy access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStepRecord {
    pub step: usize,
    pub thought: String,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_hot_store_insert_and_query() {
        let dir = tempdir().unwrap();
        let config = HotStoreConfig {
            path: dir.path().join("hot.db").to_str().unwrap().to_string(),
            max_events: 1000,
            prune_interval_secs: 60,
        };

        let store = HotStore::new(config).unwrap();
        let session_id = Uuid::new_v4();
        let trace_id = Uuid::new_v4();

        let event = TelemetryEvent::ReasoningStep {
            step: 1,
            thought: "Test thought".to_string(),
            confidence: 0.75,
            session_id,
            trace_id,
        };

        store.insert(&event).unwrap();

        let events = store.query_by_trace(trace_id).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_query_reasoning_steps() {
        let dir = tempdir().unwrap();
        let config = HotStoreConfig {
            path: dir.path().join("hot.db").to_str().unwrap().to_string(),
            ..Default::default()
        };

        let store = HotStore::new(config).unwrap();
        let session_id = Uuid::new_v4();
        let trace_id = Uuid::new_v4();

        // Insert multiple reasoning steps
        for i in 0..3 {
            let event = TelemetryEvent::ReasoningStep {
                step: i,
                thought: format!("Thought {}", i),
                confidence: 0.3 + (i as f32 * 0.1),
                session_id,
                trace_id,
            };
            store.insert(&event).unwrap();
        }

        let steps = store.query_reasoning_steps(trace_id).unwrap();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].step, 0);
        assert_eq!(steps[2].confidence, 0.5);
    }

    #[test]
    fn test_prune() {
        let dir = tempdir().unwrap();
        let config = HotStoreConfig {
            path: dir.path().join("hot.db").to_str().unwrap().to_string(),
            max_events: 5,
            prune_interval_secs: 60,
        };

        let store = HotStore::new(config).unwrap();
        let session_id = Uuid::new_v4();
        let trace_id = Uuid::new_v4();

        // Insert 10 events
        for i in 0..10 {
            let event = TelemetryEvent::Calculation {
                layer: i as u8,
                operation: format!("op_{}", i),
                duration_ms: i as u64,
                session_id,
                trace_id,
            };
            store.insert(&event).unwrap();
        }

        assert_eq!(store.event_count().unwrap(), 10);

        let pruned = store.prune().unwrap();
        assert_eq!(pruned, 5);
        assert_eq!(store.event_count().unwrap(), 5);
    }
}
