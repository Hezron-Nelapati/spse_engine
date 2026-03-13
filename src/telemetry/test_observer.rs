use crate::config::EngineConfig;
use crate::types::ScoreBreakdown as CandidateScoreBreakdown;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs::{create_dir_all, OpenOptions};
use std::io::{Error, Write};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestObservation {
    pub timestamp: String,
    pub test_id: String,
    pub query: String,
    pub total_latency_ms: u64,
    pub layer_2_time_ms: u64,
    pub layer_5_routing_time_ms: u64,
    pub layer_9_decision_time_ms: u64,
    pub layer_11_retrieval_time_ms: u64,
    pub layer_13_merge_time_ms: u64,
    pub layer_14_scoring_time_ms: u64,
    pub layer_16_resolution_time_ms: u64,
    pub units_discovered: u32,
    pub units_activated: u32,
    pub new_units_created: u32,
    pub units_pruned: u32,
    pub memory_delta_kb: i64,
    pub episodic_count: u32,
    pub core_count: u32,
    pub anchors_protected: u32,
    pub retrieval_triggered: bool,
    pub retrieval_reason: Option<String>,
    pub sources_consulted: Vec<String>,
    pub evidence_merged: u32,
    pub candidates_scored: u32,
    pub top_candidate_score: f32,
    pub score_breakdown: Option<ScoreBreakdown>,
    pub final_answer_confidence: f32,
    pub config_values_used: ConfigSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub spatial_fit: f32,
    pub context_fit: f32,
    pub sequence_fit: f32,
    pub transition_fit: f32,
    pub utility_fit: f32,
    pub confidence_fit: f32,
    pub evidence_support: f32,
}

impl From<&CandidateScoreBreakdown> for ScoreBreakdown {
    fn from(value: &CandidateScoreBreakdown) -> Self {
        Self {
            spatial_fit: value.spatial_fit,
            context_fit: value.context_fit,
            sequence_fit: value.sequence_fit,
            transition_fit: value.transition_fit,
            utility_fit: value.utility_fit,
            confidence_fit: value.confidence_fit,
            evidence_support: value.evidence_support,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigSnapshot {
    pub entropy_threshold: f32,
    pub retrieve_threshold: f32,
    pub recency_threshold_hours: u32,
    pub w_entropy: f32,
    pub w_recency: f32,
    pub w_disagreement: f32,
    pub w_cost: f32,
    pub w_spatial: f32,
    pub w_context: f32,
    pub w_sequence: f32,
    pub w_transition: f32,
    pub w_utility: f32,
    pub w_confidence: f32,
    pub w_evidence: f32,
    pub selection_temperature: f32,
    pub min_confidence_floor: f32,
    pub max_query_expansion_terms: usize,
    pub max_retrieval_results: usize,
    pub retrieval_timeout_ms: u64,
}

impl From<&EngineConfig> for ConfigSnapshot {
    fn from(config: &EngineConfig) -> Self {
        Self {
            entropy_threshold: config.retrieval.entropy_threshold,
            retrieve_threshold: config.retrieval.decision_threshold,
            recency_threshold_hours: config.retrieval.recency_threshold_hours,
            w_entropy: config.retrieval.w_entropy,
            w_recency: config.retrieval.w_recency,
            w_disagreement: config.retrieval.w_disagreement,
            w_cost: config.retrieval.w_cost,
            w_spatial: config.scoring.spatial,
            w_context: config.scoring.context,
            w_sequence: config.scoring.sequence,
            w_transition: config.scoring.transition,
            w_utility: config.scoring.utility,
            w_confidence: config.scoring.confidence,
            w_evidence: config.scoring.evidence,
            selection_temperature: config.resolver.selection_temperature,
            min_confidence_floor: config.resolver.min_confidence_floor,
            max_query_expansion_terms: config.query.max_query_expansion_terms,
            max_retrieval_results: config.retrieval_io.max_retrieval_results,
            retrieval_timeout_ms: config.retrieval_io.retrieval_timeout_ms,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestObserver {
    log_path: String,
    sample_rate: f32,
}

impl TestObserver {
    pub fn new(log_path: &str, sample_rate: f32) -> Self {
        Self {
            log_path: log_path.to_string(),
            sample_rate,
        }
    }

    pub fn from_config(config: &EngineConfig) -> Option<Self> {
        config
            .telemetry
            .observation_log_path
            .as_deref()
            .map(|path| Self::new(path, config.telemetry.telemetry_sample_rate))
    }

    pub fn log_observation(&self, obs: &TestObservation) -> Result<(), Error> {
        if rand::random::<f32>() > self.sample_rate {
            return Ok(());
        }

        self.ensure_parent_dir()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        let json = serde_json::to_string(obs).map_err(Error::other)?;
        writeln!(file, "{json}")?;
        Ok(())
    }

    pub fn log_batch_observations(&self, obs_batch: &[TestObservation]) -> Result<(), Error> {
        self.ensure_parent_dir()?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        for obs in obs_batch {
            let json = serde_json::to_string(obs).map_err(Error::other)?;
            writeln!(file, "{json}")?;
        }
        Ok(())
    }

    fn ensure_parent_dir(&self) -> Result<(), Error> {
        if let Some(parent) = Path::new(&self.log_path).parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }
        Ok(())
    }

    pub fn build_observation(
        &self,
        test_id: String,
        query: &str,
        config: &EngineConfig,
        timings: ObservationTimings,
        units: ObservationUnits,
        memory: ObservationMemory,
        retrieval: ObservationRetrieval,
        scoring: ObservationScoring,
        final_answer_confidence: f32,
    ) -> TestObservation {
        TestObservation {
            timestamp: Utc::now().to_rfc3339(),
            test_id,
            query: query.to_string(),
            total_latency_ms: timings.total_latency_ms,
            layer_2_time_ms: timings.layer_2_time_ms,
            layer_5_routing_time_ms: timings.layer_5_routing_time_ms,
            layer_9_decision_time_ms: timings.layer_9_decision_time_ms,
            layer_11_retrieval_time_ms: timings.layer_11_retrieval_time_ms,
            layer_13_merge_time_ms: timings.layer_13_merge_time_ms,
            layer_14_scoring_time_ms: timings.layer_14_scoring_time_ms,
            layer_16_resolution_time_ms: timings.layer_16_resolution_time_ms,
            units_discovered: units.units_discovered,
            units_activated: units.units_activated,
            new_units_created: units.new_units_created,
            units_pruned: units.units_pruned,
            memory_delta_kb: memory.memory_delta_kb,
            episodic_count: memory.episodic_count,
            core_count: memory.core_count,
            anchors_protected: memory.anchors_protected,
            retrieval_triggered: retrieval.retrieval_triggered,
            retrieval_reason: retrieval.retrieval_reason,
            sources_consulted: retrieval.sources_consulted,
            evidence_merged: retrieval.evidence_merged,
            candidates_scored: scoring.candidates_scored,
            top_candidate_score: scoring.top_candidate_score,
            score_breakdown: scoring.score_breakdown,
            final_answer_confidence,
            config_values_used: ConfigSnapshot::from(config),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ObservationTimings {
    pub total_latency_ms: u64,
    pub layer_2_time_ms: u64,
    pub layer_5_routing_time_ms: u64,
    pub layer_9_decision_time_ms: u64,
    pub layer_11_retrieval_time_ms: u64,
    pub layer_13_merge_time_ms: u64,
    pub layer_14_scoring_time_ms: u64,
    pub layer_16_resolution_time_ms: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ObservationUnits {
    pub units_discovered: u32,
    pub units_activated: u32,
    pub new_units_created: u32,
    pub units_pruned: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ObservationMemory {
    pub memory_delta_kb: i64,
    pub episodic_count: u32,
    pub core_count: u32,
    pub anchors_protected: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ObservationRetrieval {
    pub retrieval_triggered: bool,
    pub retrieval_reason: Option<String>,
    pub sources_consulted: Vec<String>,
    pub evidence_merged: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ObservationScoring {
    pub candidates_scored: u32,
    pub top_candidate_score: f32,
    pub score_breakdown: Option<ScoreBreakdown>,
}
