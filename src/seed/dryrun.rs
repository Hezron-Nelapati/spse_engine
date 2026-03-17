//! DryRun dataset generation for Phase 1 pipeline validation.
//!
//! This module generates high-density seed datasets that integrate with the
//! unified training system. Generated datasets include:
//! - Training phase hints (DryRun/Bootstrap alignment)
//! - Curriculum metadata for ordering
//! - Quality gates for validation
//! - TrainingOptions hints for the pipeline

use crate::seed::dialogue_generator::{
    templates, validate_dialogue_dataset, DialogueGenerator, DialogueJsonDataset,
};
use crate::seed::entity_generator::{validate_entity_dataset, EntityGenerator, EntityJsonDataset};
use crate::seed::{CurriculumMetadata, QualityGates, QualityMetrics};
use crate::types::{
    IntentKind, MemoryChannel, MemoryType, TrainingExecutionMode, TrainingOptions,
    TrainingPhaseKind,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for DryRun dataset generation (aligned with unified training system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunDatasetConfig {
    /// Target size for intent dataset in MB
    pub intent_dataset_size_mb: f32,
    /// Target size for entity dataset in MB
    pub entity_dataset_size_mb: f32,
    /// Number of dialogues per intent kind
    pub dialogues_per_intent: usize,
    /// Number of entities to generate
    pub entity_count: usize,
    /// Output directory
    pub output_dir: String,
    /// Target training phase
    #[serde(default)]
    pub phase_hint: TrainingPhaseKind,
    /// Quality gates for validation
    #[serde(default)]
    pub quality_gates: QualityGates,
    /// Training options
    #[serde(default)]
    pub training_options: TrainingOptions,
    /// Curriculum metadata
    #[serde(default)]
    pub curriculum: CurriculumMetadata,
}

impl Default for DryRunDatasetConfig {
    fn default() -> Self {
        Self {
            intent_dataset_size_mb: 1800.0, // Target ~2GB total
            entity_dataset_size_mb: 200.0,  // Larger entity dataset
            dialogues_per_intent: 100_000,  // 2.4M dialogues total (24 intents * 100K)
            entity_count: 200_000,          // High-density entity corpus
            output_dir: "datasets/dryrun".to_string(),
            phase_hint: TrainingPhaseKind::DryRun,
            quality_gates: QualityGates {
                min_unit_discovery_efficiency: Some(0.60),
                min_semantic_routing_accuracy: Some(0.75),
                min_corroboration_count: 2,
            },
            training_options: TrainingOptions {
                consolidate_immediately: true,
                max_memory_delta_mb: 50.0,
                progress_interval_sec: 10,
                tag_intent: true,
                merge_to_core: false,
                bypass_retrieval_gate: true,
                bypass_generation: true,
                daily_growth_limit_mb: Some(100.0),
                execution_mode: TrainingExecutionMode::Development,
            },
            curriculum: CurriculumMetadata {
                curriculum_score: 100,
                phase_hint: TrainingPhaseKind::DryRun,
                target_memory: MemoryType::Episodic,
                memory_channels: vec![MemoryChannel::Main, MemoryChannel::Intent],
                suggested_batch_size: 500,
                max_chunk_chars: 8000,
            },
        }
    }
}

/// Result of DryRun dataset generation (aligned with unified training system)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunGenerationResult {
    pub intent_dataset_path: String,
    pub entity_dataset_path: String,
    pub intent_dialogue_count: usize,
    pub entity_count: usize,
    pub intents_covered: Vec<String>,
    pub quality_passed: bool,
    pub warnings: Vec<String>,
    /// Quality metrics for the generated datasets
    #[serde(default)]
    pub quality_metrics: QualityMetrics,
    /// Training phase used
    #[serde(default)]
    pub phase_hint: TrainingPhaseKind,
    /// Curriculum metadata for training ordering
    #[serde(default)]
    pub curriculum: CurriculumMetadata,
}

/// Generate DryRun datasets for pipeline validation
pub fn generate_dryrun_datasets(config: &DryRunDatasetConfig) -> DryRunGenerationResult {
    let mut warnings = Vec::new();

    // Generate intent/dialogue dataset
    let intent_dataset = generate_intent_dataset(config.dialogues_per_intent);
    let intent_dialogue_count = intent_dataset.dialogues.len();
    let intents_covered: Vec<String> = intent_dataset
        .dialogues
        .iter()
        .map(|d| d.intent.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Check all 24 intents are covered
    if intents_covered.len() < 24 {
        warnings.push(format!(
            "Only {} of 24 intent kinds covered",
            intents_covered.len()
        ));
    }

    // Generate entity dataset
    let entity_dataset = generate_entity_dataset(config.entity_count);
    let entity_count = entity_dataset.entities.len();

    // Create output directory
    let output_path = Path::new(&config.output_dir);
    if let Err(e) = std::fs::create_dir_all(output_path) {
        warnings.push(format!("Failed to create output directory: {}", e));
    }

    // Write datasets to files (JSONL format for streaming large datasets)
    let intent_path = output_path.join("dryrun_intent_core.jsonl");
    let entity_path = output_path.join("dryrun_entity_seed.jsonl");

    let intent_dataset_path = intent_path.display().to_string();
    let entity_dataset_path = entity_path.display().to_string();

    // Write intent dataset as JSONL (one dialogue per line)
    match std::fs::File::create(&intent_path) {
        Ok(mut file) => {
            use std::io::Write;
            for dialogue in &intent_dataset.dialogues {
                match serde_json::to_string(dialogue) {
                    Ok(line) => {
                        if let Err(e) = writeln!(file, "{}", line) {
                            warnings.push(format!("Failed to write dialogue: {}", e));
                        }
                    }
                    Err(e) => warnings.push(format!("Failed to serialize dialogue: {}", e)),
                }
            }
        }
        Err(e) => warnings.push(format!("Failed to create intent dataset file: {}", e)),
    }

    // Write entity dataset as JSONL (one entity per line)
    match std::fs::File::create(&entity_path) {
        Ok(mut file) => {
            use std::io::Write;
            for entity in &entity_dataset.entities {
                match serde_json::to_string(entity) {
                    Ok(line) => {
                        if let Err(e) = writeln!(file, "{}", line) {
                            warnings.push(format!("Failed to write entity: {}", e));
                        }
                    }
                    Err(e) => warnings.push(format!("Failed to serialize entity: {}", e)),
                }
            }
        }
        Err(e) => warnings.push(format!("Failed to create entity dataset file: {}", e)),
    }

    // Compute quality metrics from actual generated datasets
    let dialogue_metrics = validate_dialogue_dataset(&intent_dataset);
    let entity_metrics = validate_entity_dataset(&entity_dataset);

    // Merge metrics: use entity density from entity validator, intent balance from dialogue validator
    let quality_metrics = QualityMetrics {
        entity_density: entity_metrics.entity_density,
        unique_ratio: entity_metrics.unique_ratio,
        link_coverage: entity_metrics.link_coverage,
        noise_ratio: dialogue_metrics.noise_ratio,
        intent_balance: dialogue_metrics.intent_balance,
        estimated_unit_discovery_efficiency: dialogue_metrics.estimated_unit_discovery_efficiency,
        estimated_semantic_routing_accuracy: entity_metrics.estimated_semantic_routing_accuracy,
    };

    // Validate against quality gates
    let gate_errors = quality_metrics.validate_against_gates(&config.quality_gates);
    warnings.extend(gate_errors);

    let quality_passed = warnings.is_empty()
        && intents_covered.len() >= 24  // 24 IntentKind variants
        && entity_count >= 100;

    DryRunGenerationResult {
        intent_dataset_path,
        entity_dataset_path,
        intent_dialogue_count,
        entity_count,
        intents_covered,
        quality_passed,
        warnings,
        quality_metrics,
        phase_hint: config.phase_hint,
        curriculum: config.curriculum.clone(),
    }
}

/// Generate intent/dialogue dataset for DryRun
fn generate_intent_dataset(dialogues_per_intent: usize) -> DialogueJsonDataset {
    let mut gen = DialogueGenerator::new();

    // Generate dialogues for all intent kinds
    let all_intents = [
        IntentKind::Greeting,
        IntentKind::Gratitude,
        IntentKind::Farewell,
        IntentKind::Help,
        IntentKind::Clarify,
        IntentKind::Rewrite,
        IntentKind::Verify,
        IntentKind::Continue,
        IntentKind::Forget,
        IntentKind::Question,
        IntentKind::Summarize,
        IntentKind::Explain,
        IntentKind::Compare,
        IntentKind::Extract,
        IntentKind::Analyze,
        IntentKind::Plan,
        IntentKind::Act,
        IntentKind::Recommend,
        IntentKind::Classify,
        IntentKind::Translate,
        IntentKind::Debug,
        IntentKind::Critique,
        IntentKind::Brainstorm,
        IntentKind::Unknown,
    ];

    for intent in all_intents {
        templates::generate_intent_dialogues(&mut gen, intent, dialogues_per_intent);
    }

    gen.build("dryrun_intent_core")
}

/// Generate entity dataset for DryRun
fn generate_entity_dataset(entity_count: usize) -> EntityJsonDataset {
    let mut gen = EntityGenerator::new();

    // Use bulk generation for high-density coverage
    gen.generate_bulk_entities(entity_count);

    gen.build("dryrun_entity_seed")
}
