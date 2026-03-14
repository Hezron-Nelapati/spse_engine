//! Trace and Session ID management for Layer 20 telemetry
//!
//! Provides unique identifiers for correlating telemetry events across
//! query lifecycles and engine sessions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// Global session counter for generating unique session IDs
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Global trace counter for generating unique trace IDs
static TRACE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Session identifier for grouping all events in an engine instance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

impl SessionId {
    /// Generate a new unique session ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Trace identifier for grouping all events in a single query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TraceId(pub Uuid);

impl TraceId {
    /// Generate a new unique trace ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TraceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Reasoning trace for capturing reasoning steps and confidence trajectory
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReasoningTrace {
    /// Session ID for this reasoning trace
    pub session_id: SessionId,
    /// Trace ID for this reasoning trace
    pub trace_id: TraceId,
    /// Number of reasoning steps taken
    pub reasoning_steps_taken: usize,
    /// Confidence trajectory (one entry per step)
    pub confidence_trajectory: Vec<f32>,
    /// Whether reasoning was triggered
    pub reasoning_triggered: bool,
    /// Whether max steps were reached
    pub max_steps_reached: bool,
    /// Timestamp when reasoning started
    pub started_at: DateTime<Utc>,
    /// Timestamp when reasoning completed
    pub completed_at: Option<DateTime<Utc>>,
}

impl ReasoningTrace {
    /// Create a new reasoning trace
    pub fn new(session_id: SessionId, trace_id: TraceId) -> Self {
        Self {
            session_id,
            trace_id,
            reasoning_steps_taken: 0,
            confidence_trajectory: Vec::new(),
            reasoning_triggered: false,
            max_steps_reached: false,
            started_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Record a reasoning step
    pub fn record_step(&mut self, confidence: f32) {
        self.reasoning_steps_taken += 1;
        self.confidence_trajectory.push(confidence);
        self.reasoning_triggered = true;
    }

    /// Mark reasoning as completed
    pub fn complete(&mut self, max_steps_reached: bool) {
        self.max_steps_reached = max_steps_reached;
        self.completed_at = Some(Utc::now());
    }

    /// Get the final confidence from the trajectory
    pub fn final_confidence(&self) -> Option<f32> {
        self.confidence_trajectory.last().copied()
    }

    /// Get the confidence delta (improvement) from first to last step
    pub fn confidence_delta(&self) -> Option<f32> {
        if self.confidence_trajectory.len() >= 2 {
            let first = self.confidence_trajectory.first()?;
            let last = self.confidence_trajectory.last()?;
            Some(last - first)
        } else {
            None
        }
    }

    /// Get the total duration of reasoning in milliseconds
    pub fn duration_ms(&self) -> Option<i64> {
        self.completed_at
            .map(|completed| (completed - self.started_at).num_milliseconds())
    }
}

/// Trace context for storing session and trace IDs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Current session ID
    pub session_id: SessionId,
    /// Current trace ID (changes per query)
    pub trace_id: TraceId,
    /// Current reasoning trace (if any)
    pub reasoning_trace: Option<ReasoningTrace>,
}

impl TraceContext {
    /// Create a new trace context with fresh session and trace IDs
    pub fn new() -> Self {
        Self {
            session_id: SessionId::new(),
            trace_id: TraceId::new(),
            reasoning_trace: None,
        }
    }

    /// Start a new trace (call at the beginning of each query)
    pub fn start_new_trace(&mut self) {
        self.trace_id = TraceId::new();
        self.reasoning_trace = None;
    }

    /// Start a reasoning trace
    pub fn start_reasoning(&mut self) {
        self.reasoning_trace = Some(ReasoningTrace::new(self.session_id, self.trace_id));
    }

    /// Record a reasoning step in the current trace
    pub fn record_reasoning_step(&mut self, confidence: f32) {
        if let Some(trace) = &mut self.reasoning_trace {
            trace.record_step(confidence);
        }
    }

    /// Complete the current reasoning trace
    pub fn complete_reasoning(&mut self, max_steps_reached: bool) {
        if let Some(trace) = &mut self.reasoning_trace {
            trace.complete(max_steps_reached);
        }
    }

    /// Get the current reasoning trace
    pub fn reasoning_trace(&self) -> Option<&ReasoningTrace> {
        self.reasoning_trace.as_ref()
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_uniqueness() {
        let id1 = SessionId::new();
        let id2 = SessionId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_trace_id_uniqueness() {
        let id1 = TraceId::new();
        let id2 = TraceId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_reasoning_trace() {
        let session_id = SessionId::new();
        let trace_id = TraceId::new();

        let mut trace = ReasoningTrace::new(session_id, trace_id);
        assert!(!trace.reasoning_triggered);

        trace.record_step(0.35);
        trace.record_step(0.50);
        trace.record_step(0.62);
        trace.complete(false);

        assert!(trace.reasoning_triggered);
        assert_eq!(trace.reasoning_steps_taken, 3);
        assert_eq!(trace.confidence_trajectory.len(), 3);
        assert_eq!(trace.final_confidence(), Some(0.62));
        assert_eq!(trace.confidence_delta(), Some(0.27));
        assert!(!trace.max_steps_reached);
        assert!(trace.completed_at.is_some());
    }

    #[test]
    fn test_trace_context() {
        let mut ctx = TraceContext::new();
        let initial_trace_id = ctx.trace_id;

        ctx.start_new_trace();
        assert_ne!(ctx.trace_id, initial_trace_id);

        ctx.start_reasoning();
        assert!(ctx.reasoning_trace.is_some());

        ctx.record_reasoning_step(0.40);
        ctx.record_reasoning_step(0.55);
        ctx.complete_reasoning(false);

        let trace = ctx.reasoning_trace().unwrap();
        assert_eq!(trace.reasoning_steps_taken, 2);
    }
}
