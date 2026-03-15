use crate::config::EngineConfig;
use crate::open_sources;
use crate::types::{TrainingExecutionMode, TrainingOptions, TrainingPhaseKind, TrainingSource, ReasoningTrace, Unit, MemoryType, MemoryChannel, SourceKind};
use crate::memory::store::MemoryStore;
use uuid::Uuid;

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

fn bootstrap_phase(
    config: &EngineConfig,
    execution_mode: TrainingExecutionMode,
) -> TrainingPhasePlan {
    // Use generated high-density seed datasets for bootstrap training
    let mut intent_source = source("dryrun_intent_core");
    intent_source.stream.max_input_bytes = Some(3 * 1024 * 1024 * 1024); // 3GB limit
    intent_source.stream.item_limit = Some(100_000);
    intent_source.stream.batch_size = Some(500);
    intent_source.stream.chunk_char_limit = Some(8_000);

    let mut entity_source = source("dryrun_entity_seed");
    entity_source.stream.max_input_bytes = Some(300 * 1024 * 1024); // 300MB limit
    entity_source.stream.item_limit = Some(50_000);
    entity_source.stream.batch_size = Some(500);
    entity_source.stream.chunk_char_limit = Some(6_000);

    TrainingPhasePlan {
        phase: TrainingPhaseKind::Bootstrap,
        sources: curriculum_order(vec![entity_source, intent_source]),
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
    TrainingPhasePlan {
        phase: TrainingPhaseKind::Validation,
        sources: curriculum_order(vec![source("squad_v2"), source("natural_questions")]),
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
    TrainingPhasePlan {
        phase: TrainingPhaseKind::Expansion,
        sources: curriculum_order(vec![
            source("public_openapi_specs"),
            source("proofwriter"),
            source("common_crawl"),
        ]),
        batches_target: 1,
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
    // Use generated high-density seed datasets
    let mut intent_source = source("dryrun_intent_core");
    intent_source.stream.max_input_bytes = Some(3 * 1024 * 1024 * 1024); // 3GB limit
    intent_source.stream.item_limit = Some(100_000);
    intent_source.stream.batch_size = Some(500);
    intent_source.stream.chunk_char_limit = Some(8_000);

    let mut entity_source = source("dryrun_entity_seed");
    entity_source.stream.max_input_bytes = Some(300 * 1024 * 1024); // 300MB limit
    entity_source.stream.item_limit = Some(50_000);
    entity_source.stream.batch_size = Some(500);
    entity_source.stream.chunk_char_limit = Some(6_000);

    TrainingPhasePlan {
        phase: TrainingPhaseKind::DryRun,
        sources: vec![entity_source, intent_source],
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
    TrainingPhasePlan {
        phase: TrainingPhaseKind::Lifelong,
        sources: curriculum_order(vec![
            hf_source("hf_openai_gsm8k", 1),
            hf_source("hf_tb_smoltalk2", 1),
            hf_source("hf_h4_ultrachat_200k", 1),
            hf_source("hf_fw_fineweb_edu", 1),
            hf_source("hf_tb_cosmopedia", 1),
            hf_source("hf_karpathy_climbmix", 1),
            hf_source("hf_karpathy_fineweb_edu", 1),
            hf_source("hf_karpathy_tinystories", 1),
        ]),
        batches_target: 1,
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

fn hf_source(name: &str, shard_limit: usize) -> TrainingSource {
    let mut source = open_sources::catalog_source(name);
    if shard_limit > 0 {
        source.stream.shard_limit = Some(shard_limit);
    }
    source
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
    if let Some(name) = source.name.as_deref() {
        let explicit = match name {
            "wikidata" => Some(112),
            "public_openapi_specs" => Some(110),
            "proofwriter" => Some(102),
            "gsm8k_train" => Some(98),
            "natural_questions" => Some(94),
            "wikipedia" => Some(90),
            "squad_v2" => Some(88),
            "dolly_15k" => Some(84),
            "common_crawl" => Some(22),
            "hf_tb_smoltalk2" => Some(108),
            "hf_openai_gsm8k" => Some(106),
            "hf_h4_ultrachat_200k" => Some(104),
            "hf_fw_fineweb_edu" => Some(96),
            "hf_tb_cosmopedia" => Some(95),
            "hf_karpathy_climbmix" => Some(93),
            "hf_karpathy_fineweb_edu" => Some(92),
            "hf_karpathy_tinystories" => Some(82),
            _ => None,
        };
        if let Some(score) = explicit {
            return score;
        }
    }

    let structure_score = match source.source_type {
        crate::types::TrainingSourceType::WikidataTruthy => 100,
        crate::types::TrainingSourceType::DbpediaDump => 96,
        crate::types::TrainingSourceType::OpenApiSpec => 94,
        crate::types::TrainingSourceType::QaJson => 88,
        crate::types::TrainingSourceType::HuggingFaceDataset => 87,
        crate::types::TrainingSourceType::StructuredJson => 86,
        crate::types::TrainingSourceType::Dataset => 82,
        crate::types::TrainingSourceType::WikipediaDump => 72,
        crate::types::TrainingSourceType::ProjectGutenberg => 64,
        crate::types::TrainingSourceType::CodeRepository => 60,
        crate::types::TrainingSourceType::Document => 54,
        crate::types::TrainingSourceType::Url => 48,
        crate::types::TrainingSourceType::OpenWebText => 38,
        crate::types::TrainingSourceType::CommonCrawlWet => 22,
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
    use crate::types::{ReasoningStep, ReasoningType};
    use crate::telemetry::trace::{SessionId, TraceId};

    #[test]
    fn test_ingest_reasoning_trace_creates_process_units() {
        // This test would require a MemoryStore with a temp database
        // For now, we verify the function compiles and has correct signature
        let _ = ingest_reasoning_trace as fn(&mut MemoryStore, &ReasoningTrace, &str) -> Vec<Uuid>;
    }
}
