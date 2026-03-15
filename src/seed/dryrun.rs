//! DryRun dataset generation for Phase 1 pipeline validation.

use crate::seed::dialogue_generator::{DialogueGenerator, DialogueJsonDataset, templates};
use crate::seed::entity_generator::{EntityGenerator, EntityJsonDataset};
use crate::types::IntentKind;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Configuration for DryRun dataset generation
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
}

impl Default for DryRunDatasetConfig {
    fn default() -> Self {
        Self {
            intent_dataset_size_mb: 1800.0,  // Target ~2GB total
            entity_dataset_size_mb: 200.0,   // Larger entity dataset
            dialogues_per_intent: 100_000,   // 2.4M dialogues total (24 intents * 100K)
            entity_count: 200_000,           // High-density entity corpus
            output_dir: "datasets/dryrun".to_string(),
        }
    }
}

/// Result of DryRun dataset generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunGenerationResult {
    pub intent_dataset_path: String,
    pub entity_dataset_path: String,
    pub intent_dialogue_count: usize,
    pub entity_count: usize,
    pub intents_covered: Vec<String>,
    pub quality_passed: bool,
    pub warnings: Vec<String>,
}

/// Generate DryRun datasets for pipeline validation
pub fn generate_dryrun_datasets(config: &DryRunDatasetConfig) -> DryRunGenerationResult {
    let mut warnings = Vec::new();
    
    // Generate intent/dialogue dataset
    let intent_dataset = generate_intent_dataset(config.dialogues_per_intent);
    let intent_dialogue_count = intent_dataset.dialogues.len();
    let intents_covered: Vec<String> = intent_dataset.dialogues.iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_dryrun_datasets() {
        let config = DryRunDatasetConfig {
            dialogues_per_intent: 10, // Small for testing
            entity_count: 50,
            output_dir: std::env::temp_dir().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let result = generate_dryrun_datasets(&config);
        
        assert!(result.intent_dialogue_count >= 10);
        assert!(result.entity_count >= 50);
        // Should cover most intents (may not cover all with small sample)
        assert!(!result.intents_covered.is_empty());
    }

    #[test]
    fn test_generate_aliases() {
        let aliases = generate_aliases("Approval Process");
        
        assert!(aliases.contains(&"approval process".to_string()));
        assert!(aliases.iter().any(|a| a.contains("Workflow")));
    }
}
