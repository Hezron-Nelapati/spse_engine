//! Seed module for generating high-density training datasets.
//!
//! This module implements dataset generators following the specifications in
//! DATASET_GENERATION_GUIDE.md for pre-production training.

mod entity_generator;
mod dialogue_generator;
mod dryrun;

pub use entity_generator::{EntityGenerator, EntityJsonDataset};
pub use dialogue_generator::{DialogueGenerator, DialogueJsonDataset, Dialogue, DialogueTurn, DialogueMetadata as SeedDialogueMetadata, ExpectedUnitCount, MemoryTarget as SeedMemoryTarget};
pub use dryrun::{generate_dryrun_datasets, DryRunDatasetConfig};

use serde::{Deserialize, Serialize};

/// Common dataset metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub dataset_id: String,
    pub version: String,
    pub generated_at: String,
    #[serde(rename = "type")]
    pub dataset_type: String,
    pub density_score: f32,
    pub unit_count_estimate: u64,
}

impl DatasetMetadata {
    pub fn new(dataset_id: &str, dataset_type: &str, density_score: f32, unit_count_estimate: u64) -> Self {
        Self {
            dataset_id: dataset_id.to_string(),
            version: "1.0.0".to_string(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            dataset_type: dataset_type.to_string(),
            density_score,
            unit_count_estimate,
        }
    }
}

/// Quality metrics for generated datasets
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub entity_density: f32,
    pub unique_ratio: f32,
    pub link_coverage: f32,
    pub noise_ratio: f32,
    pub intent_balance: f32,
}

impl QualityMetrics {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        
        if self.entity_density < 40.0 {
            errors.push(format!("entity_density {} < 40 entities per KB", self.entity_density));
        }
        if self.unique_ratio < 0.95 {
            errors.push(format!("unique_ratio {} < 0.95", self.unique_ratio));
        }
        if self.link_coverage < 0.80 {
            errors.push(format!("link_coverage {} < 0.80", self.link_coverage));
        }
        if self.noise_ratio > 0.05 {
            errors.push(format!("noise_ratio {} > 0.05", self.noise_ratio));
        }
        if self.intent_balance < 0.85 {
            errors.push(format!("intent_balance {} < 0.85", self.intent_balance));
        }
        
        errors
    }
}
