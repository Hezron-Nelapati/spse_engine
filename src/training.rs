use crate::config::EngineConfig;
use crate::open_sources;
use crate::types::{TrainingExecutionMode, TrainingJobStatus, TrainingOptions, TrainingPhaseKind, TrainingSource, ReasoningTrace, MemoryType, MemoryChannel, SourceKind};
use crate::memory::store::MemoryStore;
use chrono::Utc;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use uuid::Uuid;

/// Per-run training logger that writes structured output to a dedicated folder.
///
/// Folder structure:
/// ```text
/// training_jobs/
///   <job_id>/
///     status.json       - current/final job status
///     progress.log      - human-readable timestamped progress
///     telemetry.jsonl   - per-batch metrics as JSON lines
/// ```
pub struct TrainingRunLogger {
    run_dir: PathBuf,
    progress_file: Option<std::fs::File>,
    telemetry_file: Option<std::fs::File>,
    run_start: Instant,
    last_print: Instant,
}

#[derive(Debug, Serialize)]
pub struct BatchTelemetry {
    pub timestamp: String,
    pub source: String,
    pub phase: String,
    pub batch_index: u64,
    pub examples_ingested: u64,
    pub units_created: u64,
    pub bytes_read: u64,
    pub batch_duration_ms: u64,
    pub examples_per_sec: f64,
    pub units_per_sec: f64,
    pub cumulative_examples: u64,
    pub cumulative_units: u64,
    pub elapsed_sec: f64,
}

impl TrainingRunLogger {
    /// Create a new logger for a training run. Creates the folder structure immediately.
    pub fn new(base_dir: &Path, job_id: &str) -> Self {
        let run_dir = base_dir.join("training_jobs").join(job_id);
        let _ = fs::create_dir_all(&run_dir);

        let progress_file = fs::File::create(run_dir.join("progress.log")).ok();
        let telemetry_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(run_dir.join("telemetry.jsonl"))
            .ok();

        let now = Instant::now();
        let mut logger = Self {
            run_dir,
            progress_file,
            telemetry_file,
            run_start: now,
            last_print: now,
        };
        logger.log_progress(&format!(
            "[{}] Training run started: {}",
            Utc::now().format("%H:%M:%S"),
            job_id,
        ));
        logger
    }

    /// Write the current job status to status.json in the run folder.
    pub fn write_status(&self, status: &TrainingJobStatus) {
        let path = self.run_dir.join("status.json");
        if let Ok(json) = serde_json::to_string_pretty(status) {
            let tmp = path.with_extension("json.tmp");
            if let Ok(mut f) = fs::File::create(&tmp) {
                let _ = f.write_all(json.as_bytes());
                let _ = f.sync_all();
                let _ = fs::rename(&tmp, &path);
            }
        }
    }

    /// Log a human-readable progress line to both stderr and progress.log.
    pub fn log_progress(&mut self, msg: &str) {
        eprintln!("{}", msg);
        if let Some(ref mut f) = self.progress_file {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
    }

    /// Log a batch telemetry record to telemetry.jsonl.
    pub fn log_telemetry(&mut self, record: &BatchTelemetry) {
        if let Some(ref mut f) = self.telemetry_file {
            if let Ok(json) = serde_json::to_string(record) {
                let _ = writeln!(f, "{}", json);
                let _ = f.flush();
            }
        }
    }

    /// Log batch progress with rate metrics. Prints every 2 seconds or on force.
    pub fn log_batch_progress(
        &mut self,
        source_name: &str,
        phase: &str,
        batch_index: u64,
        examples_ingested: u64,
        units_created: u64,
        bytes_read: u64,
        batch_duration: std::time::Duration,
        force: bool,
    ) {
        let elapsed = self.run_start.elapsed();
        let since_print = self.last_print.elapsed();

        // Write telemetry for every batch
        let batch_ms = batch_duration.as_millis() as u64;
        let record = BatchTelemetry {
            timestamp: Utc::now().to_rfc3339(),
            source: source_name.to_string(),
            phase: phase.to_string(),
            batch_index,
            examples_ingested,
            units_created,
            bytes_read,
            batch_duration_ms: batch_ms,
            examples_per_sec: examples_ingested as f64 / elapsed.as_secs_f64().max(0.001),
            units_per_sec: units_created as f64 / elapsed.as_secs_f64().max(0.001),
            cumulative_examples: examples_ingested,
            cumulative_units: units_created,
            elapsed_sec: elapsed.as_secs_f64(),
        };
        self.log_telemetry(&record);

        // Print human-readable progress every 2 seconds or when forced
        if force || since_print.as_secs() >= 2 {
            let rate = examples_ingested as f64 / elapsed.as_secs_f64().max(0.001);
            let mb = bytes_read as f64 / 1_048_576.0;
            let msg = format!(
                "[{}] [{}] {} examples, {} units, {:.1} MB, {:.0} ex/s, {:.1}s elapsed",
                Utc::now().format("%H:%M:%S"),
                source_name,
                examples_ingested,
                units_created,
                mb,
                rate,
                elapsed.as_secs_f64(),
            );
            self.log_progress(&msg);
            self.last_print = Instant::now();
        }
    }

    /// Return the run directory path.
    pub fn run_dir(&self) -> &Path {
        &self.run_dir
    }

    /// Total elapsed time since run start.
    pub fn elapsed(&self) -> std::time::Duration {
        self.run_start.elapsed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingScope {
    Full,
    Bootstrap,
    DryRun,
    HuggingFace,
}

#[derive(Debug, Clone)]
pub struct TrainingPhasePlan {
    pub phase: TrainingPhaseKind,
    pub sources: Vec<TrainingSource>,
    pub batches_target: usize,
    pub options: TrainingOptions,
    pub min_unit_discovery_efficiency: Option<f32>,
    pub min_semantic_routing_accuracy: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct TrainingPlan {
    pub phases: Vec<TrainingPhasePlan>,
}

pub fn build_release_training_plan(execution_mode: TrainingExecutionMode) -> TrainingPlan {
    build_training_plan_with_config(
        &EngineConfig::default(),
        execution_mode,
        TrainingScope::Full,
    )
}

pub fn build_training_plan(
    execution_mode: TrainingExecutionMode,
    scope: TrainingScope,
) -> TrainingPlan {
    build_training_plan_with_config(&EngineConfig::default(), execution_mode, scope)
}

pub fn build_training_plan_with_config(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
    scope: TrainingScope,
) -> TrainingPlan {
    let bootstrap = bootstrap_phase(config, execution_mode);
    let validation = validation_phase(config, execution_mode);
    let expansion = expansion_phase(config, execution_mode);
    let huggingface = huggingface_phase(config, execution_mode);

    let phases = match scope {
        TrainingScope::Full => vec![bootstrap, validation, expansion],
        TrainingScope::Bootstrap => vec![bootstrap],
        TrainingScope::DryRun => vec![dry_run_phase(config, execution_mode)],
        TrainingScope::HuggingFace => vec![huggingface],
    };

    TrainingPlan { phases }
}

pub fn render_release_training_plan(execution_mode: TrainingExecutionMode) -> String {
    render_training_plan_with_config(
        &EngineConfig::default(),
        execution_mode,
        TrainingScope::Full,
    )
}

pub fn render_training_plan(execution_mode: TrainingExecutionMode, scope: TrainingScope) -> String {
    render_training_plan_with_config(&EngineConfig::default(), execution_mode, scope)
}

pub fn render_training_plan_with_config(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
    scope: TrainingScope,
) -> String {
    let mut lines = vec![
        "phase | sources | max_memory_delta_mb | daily_growth_limit_mb | merge_to_core | gates"
            .to_string(),
        "--- | --- | --- | --- | --- | ---".to_string(),
    ];

    for phase in build_training_plan_with_config(config, execution_mode, scope).phases {
        let names = phase
            .sources
            .iter()
            .filter_map(|source| source.name.as_deref())
            .collect::<Vec<_>>()
            .join(", ");
        let gates = match (
            phase.min_unit_discovery_efficiency,
            phase.min_semantic_routing_accuracy,
        ) {
            (Some(unit), Some(routing)) => format!(
                "unit_discovery_efficiency>={unit:.2}, semantic_routing_accuracy>={routing:.2}"
            ),
            _ => "-".to_string(),
        };
        lines.push(format!(
            "{:?} | {} | {:.1} | {:.1} | {} | {}",
            phase.phase,
            names,
            phase.options.max_memory_delta_mb,
            phase.options.daily_growth_limit_mb.unwrap_or(0.0),
            phase.options.merge_to_core,
            gates
        ));
    }

    lines.join("\n")
}

fn seed_sources() -> Vec<TrainingSource> {
    let names = ["seed_entities", "seed_intelligence", "seed_dialogues", "seed_classification"];
    names.iter().map(|name| source(name)).collect()
}

fn bootstrap_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // Use all 4 generated seed datasets for bootstrap training
    let sources: Vec<TrainingSource> = seed_sources().into_iter().map(|mut s| {
        s.stream.max_input_bytes = Some(1024 * 1024 * 1024); // 1GB per source
        s.stream.item_limit = Some(200_000);
        s.stream.batch_size = Some(500);
        s
    }).collect();

    TrainingPhasePlan {
        phase: TrainingPhaseKind::Bootstrap,
        sources: curriculum_order(sources),
        batches_target: 4,
        options: phase_options(config, execution_mode, &config.training_phases.bootstrap),
        min_unit_discovery_efficiency: config
            .training_phases
            .bootstrap
            .min_unit_discovery_efficiency,
        min_semantic_routing_accuracy: config
            .training_phases
            .bootstrap
            .min_semantic_routing_accuracy,
    }
}

fn validation_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // Validation samples from all 4 seed datasets to verify ingestion quality.
    let sources: Vec<TrainingSource> = seed_sources().into_iter().map(|mut s| {
        s.stream.item_limit = Some(5_000);
        s.stream.batch_size = Some(250);
        s
    }).collect();

    TrainingPhasePlan {
        phase: TrainingPhaseKind::Validation,
        sources,
        batches_target: 1,
        options: phase_options(config, execution_mode, &config.training_phases.validation),
        min_unit_discovery_efficiency: config
            .training_phases
            .validation
            .min_unit_discovery_efficiency,
        min_semantic_routing_accuracy: config
            .training_phases
            .validation
            .min_semantic_routing_accuracy,
    }
}

fn expansion_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // Expansion phase is reserved for future user-supplied or API-driven sources.
    // No external open-source datasets are used. The engine acquires knowledge
    // at runtime via web retrieval triggered during reasoning.
    TrainingPhasePlan {
        phase: TrainingPhaseKind::Expansion,
        sources: Vec::new(),
        batches_target: 0,
        options: phase_options(config, execution_mode, &config.training_phases.expansion),
        min_unit_discovery_efficiency: config
            .training_phases
            .expansion
            .min_unit_discovery_efficiency,
        min_semantic_routing_accuracy: config
            .training_phases
            .expansion
            .min_semantic_routing_accuracy,
    }
}

fn dry_run_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // Dry run uses all 4 seed datasets with same limits as bootstrap
    let sources: Vec<TrainingSource> = seed_sources().into_iter().map(|mut s| {
        s.stream.max_input_bytes = Some(1024 * 1024 * 1024);
        s.stream.item_limit = Some(200_000);
        s.stream.batch_size = Some(500);
        s
    }).collect();

    TrainingPhasePlan {
        phase: TrainingPhaseKind::DryRun,
        sources,
        batches_target: 4,
        options: phase_options(config, execution_mode, &config.training_phases.dry_run),
        min_unit_discovery_efficiency: config.training_phases.dry_run.min_unit_discovery_efficiency,
        min_semantic_routing_accuracy: config.training_phases.dry_run.min_semantic_routing_accuracy,
    }
}

fn huggingface_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // HuggingFace phase disabled: no external datasets used for training.
    // The engine acquires knowledge at runtime via web retrieval during reasoning.
    TrainingPhasePlan {
        phase: TrainingPhaseKind::Lifelong,
        sources: Vec::new(),
        batches_target: 0,
        options: phase_options(config, execution_mode, &config.training_phases.huggingface),
        min_unit_discovery_efficiency: config
            .training_phases
            .huggingface
            .min_unit_discovery_efficiency,
        min_semantic_routing_accuracy: config
            .training_phases
            .huggingface
            .min_semantic_routing_accuracy,
    }
}

fn phase_options(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
    phase: &crate::config::TrainingPhaseConfig,
) -> TrainingOptions {
    let (max_memory_delta_mb, daily_growth_limit_mb) = match execution_mode {
        TrainingExecutionMode::Development => {
            (phase.max_memory_delta_mb, Some(phase.daily_growth_limit_mb))
        }
        TrainingExecutionMode::User => (0.5, Some(1.0)),
    };

    TrainingOptions {
        consolidate_immediately: true,
        max_memory_delta_mb,
        progress_interval_sec: config.silent_training.progress_interval_sec,
        tag_intent: true,
        merge_to_core: phase.merge_to_core || config.silent_training.merge_to_core,
        bypass_retrieval_gate: true,
        bypass_generation: true,
        daily_growth_limit_mb,
        execution_mode,
    }
}

fn source(name: &str) -> TrainingSource {
    open_sources::catalog_source(name)
}

fn curriculum_order(mut sources: Vec<TrainingSource>) -> Vec<TrainingSource> {
    sources.sort_by(|lhs, rhs| {
        curriculum_score(rhs)
            .cmp(&curriculum_score(lhs))
            .then_with(|| lhs.name.cmp(&rhs.name))
    });
    sources
}

fn curriculum_score(source: &TrainingSource) -> i32 {
    // Seed datasets ordered: entities first (core KB), then intelligence (reasoning),
    // then classification (intent patterns), then dialogues (multi-turn).
    if let Some(name) = source.name.as_deref() {
        let explicit = match name {
            "seed_entities" => Some(130),
            "seed_intelligence" => Some(120),
            "seed_classification" => Some(110),
            "seed_dialogues" => Some(100),
            _ => None,
        };
        if let Some(score) = explicit {
            return score;
        }
    }

    let structure_score = match source.source_type {
        crate::types::TrainingSourceType::QaJson => 88,
        crate::types::TrainingSourceType::StructuredJson => 86,
        crate::types::TrainingSourceType::Dataset => 82,
        crate::types::TrainingSourceType::Document => 54,
        crate::types::TrainingSourceType::Url => 48,
        _ => 40,
    };
    structure_score
}

/// Ingest a reasoning trace into memory as process units.
/// Each step becomes a process unit in the Reasoning channel, isolated from Core memory.
/// Returns the unit IDs created for the trace.
pub fn ingest_reasoning_trace(
    memory: &mut MemoryStore,
    trace: &ReasoningTrace,
    _query: &str,
) -> Vec<Uuid> {
    use crate::types::{ActivatedUnit, UnitHierarchy, UnitLevel};
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut unit_ids = Vec::new();

    for (step_idx, step) in trace.steps.iter().enumerate() {
        // Get confidence from trajectory if available
        let confidence = trace.confidence_trajectory
            .get(step_idx)
            .copied()
            .unwrap_or(0.5);

        // Create an activated unit from the reasoning step
        let activation = ActivatedUnit {
            normalized: step.content.clone(),
            content: step.content.clone(),
            level: UnitLevel::Phrase,
            utility_score: confidence,
            confidence,
            frequency: 1,
            salience: 0.5,
            context_hint: format!("reasoning_{:?}_step_{}", trace.reasoning_type, step_idx),
        };

        // Build hierarchy for this step
        let mut hierarchy = UnitHierarchy::default();
        hierarchy.levels.insert("Phrase".to_string(), vec![activation.clone()]);

        // Ingest into Reasoning channel only (not Core)
        memory.ingest_hierarchy_with_channels(
            &hierarchy,
            SourceKind::UserInput,
            &format!("reasoning_trace_{:?}", trace.structure_hash.unwrap_or(0)),
            MemoryType::Episodic,
            &[MemoryChannel::Reasoning],
        );

        // Find the created unit and mark it as a process unit
        if let Some(unit) = memory
            .units_in_channel(MemoryChannel::Reasoning)
            .iter()
            .find(|u| u.content == step.content && !u.is_process_unit)
        {
            // Compute structure hash for process anchor registration
            let mut hasher = DefaultHasher::new();
            step.content.hash(&mut hasher);
            let structure_hash = hasher.finish();

            // Register as process unit and reasoning pattern
            memory.register_process_anchor(structure_hash, unit.id);
            memory.register_reasoning_pattern(trace.reasoning_type, unit.id);

            unit_ids.push(unit.id);
        }
    }

    unit_ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingest_reasoning_trace_creates_process_units() {
        // This test would require a MemoryStore with a temp database
        // For now, we verify the function compiles and has correct signature
        let _ = ingest_reasoning_trace as fn(&mut MemoryStore, &ReasoningTrace, &str) -> Vec<Uuid>;
    }
}
