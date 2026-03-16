//! Seed module for generating high-density training datasets.
//!
//! This module implements dataset generators following the specifications in
//! DATASET_GENERATION_GUIDE.md for pre-production training.
//!
//! ## Unified Training System Alignment
//!
//! Seed generators produce datasets that integrate with the unified training pipeline:
//! - `TrainingExample` - canonical QA format with reasoning traces
//! - `CurriculumMetadata` - ordering hints for curriculum-based training
//! - `QualityGates` - minimum thresholds for training validation
//! - `TrainingOptions` - processing options (from types.rs)

pub mod bulk_generator;
pub mod classification_dataset_generator;
pub mod classification_generator;
pub mod consistency_generator;
pub mod dialogue_generator;
pub mod dryrun;
pub mod entity_generator;
pub mod intelligence_generator;
pub mod predictive_generator;
pub mod reasoning_generator;

pub use classification_dataset_generator::ClassificationDatasetGenerator;
pub use classification_generator::generate_bulk_classification;
pub use consistency_generator::ConsistencyDatasetGenerator;
pub use dialogue_generator::generate_bulk_dialogues;
pub use dialogue_generator::{
    Dialogue, DialogueGenerator, DialogueJsonDataset, DialogueMetadata as SeedDialogueMetadata,
    DialogueTurn, ExpectedUnitCount, MemoryTarget as SeedMemoryTarget,
};
pub use dryrun::{generate_dryrun_datasets, DryRunDatasetConfig};
pub use entity_generator::generate_bulk_entities;
pub use entity_generator::{EntityGenerator, EntityJsonDataset};
pub use intelligence_generator::{
    generate_bulk_intelligence, generate_intelligence_seeds, intelligence_seed_count,
};
pub use predictive_generator::PredictiveQAGenerator;
pub use reasoning_generator::ReasoningDatasetGenerator;

use crate::types::{
    IntentKind, MemoryChannel, MemoryType, ReasoningStep, ReasoningStepType, ReasoningTrace,
    ReasoningType, TrainingOptions, TrainingPhaseKind,
};
use serde::{Deserialize, Serialize};

//=============================================================================
// UNIFIED TRAINING EXAMPLE FORMAT
//
// This is the canonical format expected by the unified training system.
// All seed generators should produce data convertible to this format.
//=============================================================================

/// A single training example in the format expected by unified training.
/// This matches the QaJson format with embedded reasoning traces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    /// The question or input prompt
    pub question: String,
    /// The answer or expected output
    pub answer: String,
    /// Optional context for the QA pair
    #[serde(default)]
    pub context: Option<String>,
    /// Reasoning trace showing step-by-step derivation
    #[serde(default)]
    pub reasoning: Option<ReasoningTrace>,
    /// Intent classification for this example
    #[serde(default)]
    pub intent: Option<String>,
    /// Entities mentioned in the example
    #[serde(default)]
    pub entities: Vec<String>,
    /// Memory channels to route this example
    #[serde(default)]
    pub channels: Vec<MemoryChannel>,
    /// Curriculum metadata for training ordering and configuration
    #[serde(default)]
    pub curriculum: CurriculumMetadata,
    /// Quality gates for this example
    #[serde(default)]
    pub quality_gates: QualityGates,
    /// Training options for processing this example
    #[serde(default)]
    pub training_options: TrainingOptions,
}

impl TrainingExample {
    /// Create a simple QA pair without reasoning
    pub fn qa(question: &str, answer: &str) -> Self {
        Self {
            question: question.to_string(),
            answer: answer.to_string(),
            context: None,
            reasoning: None,
            intent: None,
            entities: Vec::new(),
            channels: vec![MemoryChannel::Main],
            curriculum: CurriculumMetadata::default(),
            quality_gates: QualityGates::default(),
            training_options: TrainingOptions::default(),
        }
    }

    /// Create a QA pair with reasoning trace
    pub fn qa_with_reasoning(
        question: &str,
        answer: &str,
        reasoning_type: ReasoningType,
        steps: Vec<(&str, ReasoningStepType)>,
    ) -> Self {
        let reasoning_steps: Vec<ReasoningStep> = steps
            .iter()
            .enumerate()
            .map(|(i, (content, step_type))| ReasoningStep {
                content: content.to_string(),
                step_type: *step_type,
                anchor_step: i == 0 || *step_type == ReasoningStepType::Conclusion,
                dependencies: if i > 0 { vec![i - 1] } else { vec![] },
                structure_hash: None,
            })
            .collect();

        let confidence_trajectory: Vec<f32> = (0..reasoning_steps.len())
            .map(|i| 0.3 + (i as f32 * 0.15).min(0.6))
            .collect();

        let mut curriculum = CurriculumMetadata::default();
        curriculum.memory_channels = vec![MemoryChannel::Main, MemoryChannel::Reasoning];

        Self {
            question: question.to_string(),
            answer: answer.to_string(),
            context: None,
            reasoning: Some(ReasoningTrace {
                steps: reasoning_steps,
                reasoning_type,
                confidence_trajectory,
                entities: Vec::new(),
                structure_hash: None,
            }),
            intent: None,
            entities: Vec::new(),
            channels: vec![MemoryChannel::Main, MemoryChannel::Reasoning],
            curriculum,
            quality_gates: QualityGates::default(),
            training_options: TrainingOptions::default(),
        }
    }

    /// Set intent for this example
    pub fn with_intent(mut self, intent: IntentKind) -> Self {
        self.intent = Some(format!("{:?}", intent));
        if !self.channels.contains(&MemoryChannel::Intent) {
            self.channels.push(MemoryChannel::Intent);
        }
        if !self
            .curriculum
            .memory_channels
            .contains(&MemoryChannel::Intent)
        {
            self.curriculum.memory_channels.push(MemoryChannel::Intent);
        }
        self
    }

    /// Set entities for this example
    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }

    /// Set context for this example
    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    /// Set curriculum score
    pub fn with_curriculum_score(mut self, score: i32) -> Self {
        self.curriculum.curriculum_score = score;
        self
    }

    /// Set training phase
    pub fn with_phase(mut self, phase: TrainingPhaseKind) -> Self {
        self.curriculum.phase_hint = phase;
        self
    }

    /// Set target memory type
    pub fn with_target_memory(mut self, memory_type: MemoryType) -> Self {
        self.curriculum.target_memory = memory_type;
        self
    }

    /// Set quality gates
    pub fn with_quality_gates(mut self, gates: QualityGates) -> Self {
        self.quality_gates = gates;
        self
    }

    /// Set training options
    pub fn with_training_options(mut self, options: TrainingOptions) -> Self {
        self.training_options = options;
        self
    }
}

/// Convert a Dialogue to TrainingExamples (one per turn pair)
impl From<&Dialogue> for Vec<TrainingExample> {
    fn from(dialogue: &Dialogue) -> Self {
        let mut examples = Vec::new();
        let turns = &dialogue.turns;

        // Convert user-assistant turn pairs to QA examples
        for i in (0..turns.len()).step_by(2) {
            if i + 1 < turns.len() && turns[i].role == "user" && turns[i + 1].role == "assistant" {
                let mut channels = dialogue.metadata.memory_channels.clone();

                // Add Intent channel if not present
                if !channels.contains(&MemoryChannel::Intent) {
                    channels.push(MemoryChannel::Intent);
                }

                let example = TrainingExample {
                    question: turns[i].content.clone(),
                    answer: turns[i + 1].content.clone(),
                    context: turns[i].context.clone(),
                    reasoning: turns[i + 1].reasoning_trace.clone(),
                    intent: Some(dialogue.intent.clone()),
                    entities: dialogue.metadata.entities_referenced.clone(),
                    channels,
                    curriculum: dialogue.metadata.curriculum.clone(),
                    quality_gates: dialogue.metadata.quality_gates.clone(),
                    training_options: dialogue.metadata.training_options.clone(),
                };

                examples.push(example);
            }
        }

        examples
    }
}

//=============================================================================
// DATASET METADATA
//=============================================================================

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
    pub fn new(
        dataset_id: &str,
        dataset_type: &str,
        density_score: f32,
        unit_count_estimate: u64,
    ) -> Self {
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

/// Curriculum metadata for training ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumMetadata {
    /// Curriculum priority score (higher = earlier in training)
    pub curriculum_score: i32,
    /// Target training phase
    #[serde(default)]
    pub phase_hint: TrainingPhaseKind,
    /// Target memory type for ingestion
    #[serde(default)]
    pub target_memory: MemoryType,
    /// Memory channels to route content
    #[serde(default)]
    pub memory_channels: Vec<MemoryChannel>,
    /// Suggested batch size for streaming
    #[serde(default)]
    pub suggested_batch_size: usize,
    /// Maximum chunk size in characters
    #[serde(default)]
    pub max_chunk_chars: usize,
}

impl Default for CurriculumMetadata {
    fn default() -> Self {
        Self {
            curriculum_score: 100,
            phase_hint: TrainingPhaseKind::DryRun,
            target_memory: MemoryType::Episodic,
            memory_channels: vec![MemoryChannel::Main, MemoryChannel::Intent],
            suggested_batch_size: 500,
            max_chunk_chars: 8000,
        }
    }
}

/// Quality gates for training validation (aligned with TrainingPhasePlan)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGates {
    /// Minimum unit discovery efficiency threshold
    #[serde(default)]
    pub min_unit_discovery_efficiency: Option<f32>,
    /// Minimum semantic routing accuracy threshold
    #[serde(default)]
    pub min_semantic_routing_accuracy: Option<f32>,
    /// Minimum corroboration count for Core promotion
    #[serde(default)]
    pub min_corroboration_count: u32,
}

impl Default for QualityGates {
    fn default() -> Self {
        Self {
            min_unit_discovery_efficiency: Some(0.60),
            min_semantic_routing_accuracy: Some(0.75),
            min_corroboration_count: 2,
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
    /// Estimated unit discovery efficiency based on dataset structure
    #[serde(default)]
    pub estimated_unit_discovery_efficiency: f32,
    /// Estimated semantic routing accuracy based on entity coverage
    #[serde(default)]
    pub estimated_semantic_routing_accuracy: f32,
}

impl QualityMetrics {
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.entity_density < 40.0 {
            errors.push(format!(
                "entity_density {} < 40 entities per KB",
                self.entity_density
            ));
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

    /// Validate against training quality gates
    pub fn validate_against_gates(&self, gates: &QualityGates) -> Vec<String> {
        let mut errors = self.validate();

        if let Some(min_efficiency) = gates.min_unit_discovery_efficiency {
            if self.estimated_unit_discovery_efficiency < min_efficiency {
                errors.push(format!(
                    "estimated_unit_discovery_efficiency {} < {} threshold",
                    self.estimated_unit_discovery_efficiency, min_efficiency
                ));
            }
        }

        if let Some(min_accuracy) = gates.min_semantic_routing_accuracy {
            if self.estimated_semantic_routing_accuracy < min_accuracy {
                errors.push(format!(
                    "estimated_semantic_routing_accuracy {} < {} threshold",
                    self.estimated_semantic_routing_accuracy, min_accuracy
                ));
            }
        }

        errors
    }
}

//=============================================================================
// TESTS: Unified Training Flow
//=============================================================================

#[cfg(test)]
mod unified_flow_tests {
    use super::*;
    use crate::types::{
        ReasoningStepType, ReasoningType, ResolverMode, ToneKind, TrainingSource,
        TrainingSourceType,
    };

    /// Test: Generate a single TrainingExample and verify it can flow through unified training
    #[test]
    fn test_single_example_generation_and_flow() {
        // Step 1: Generate a simple QA example
        let example = TrainingExample::qa(
            "What is the capital of France?",
            "The capital of France is Paris.",
        )
        .with_intent(IntentKind::Question)
        .with_entities(vec!["France".to_string(), "Paris".to_string()])
        .with_curriculum_score(110);

        // Verify example structure
        assert_eq!(example.question, "What is the capital of France?");
        assert_eq!(example.answer, "The capital of France is Paris.");
        assert!(example.intent.is_some());
        assert_eq!(example.entities.len(), 2);
        assert!(example.channels.contains(&MemoryChannel::Intent));
        assert_eq!(example.curriculum.curriculum_score, 110);

        println!("✓ Step 1: Generated TrainingExample: {:?}", example);
    }

    /// Test: Generate QA with reasoning trace
    #[test]
    fn test_qa_with_reasoning_trace() {
        let example = TrainingExample::qa_with_reasoning(
            "What is 15 + 27?",
            "15 + 27 = 42",
            ReasoningType::Mathematical,
            vec![
                ("Given: 15 + 27", ReasoningStepType::Premise),
                (
                    "15 + 20 = 35, then 35 + 7 = 42",
                    ReasoningStepType::Calculation,
                ),
                ("Result: 42", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(120);

        // Verify reasoning trace
        assert!(example.reasoning.is_some());
        let trace = example.reasoning.unwrap();
        assert_eq!(trace.steps.len(), 3);
        assert_eq!(trace.reasoning_type, ReasoningType::Mathematical);
        assert!(!trace.confidence_trajectory.is_empty());

        // Verify channels include Reasoning
        assert!(example.channels.contains(&MemoryChannel::Reasoning));

        println!("✓ Step 2: Generated QA with reasoning trace");
        println!(
            "  Steps: {:?}",
            trace.steps.iter().map(|s| &s.content).collect::<Vec<_>>()
        );
    }

    /// Test: Convert Dialogue to TrainingExamples
    #[test]
    fn test_dialogue_to_training_examples() {
        use crate::seed::dialogue_generator::DialogueGenerator;

        // Generate a dialogue using DialogueGenerator
        let mut generator = DialogueGenerator::new();
        let dialogue = generator.create_dialogue(
            IntentKind::Question,
            Some(ToneKind::NeutralProfessional),
            Some(ResolverMode::Deterministic),
            vec![
                ("user".to_string(), "What is machine learning?".to_string()),
                (
                    "assistant".to_string(),
                    "Machine learning is a subset of AI that enables systems to learn from data."
                        .to_string(),
                ),
            ],
            "technology",
            "moderate",
            vec!["machine_learning".to_string()],
            vec![MemoryChannel::Main, MemoryChannel::Intent],
            vec!["phrase".to_string(), "sentence".to_string()],
        );

        // Convert to TrainingExamples
        let examples: Vec<TrainingExample> = Vec::from(&dialogue);

        assert!(
            !examples.is_empty(),
            "Dialogue should produce at least one TrainingExample"
        );

        let first = &examples[0];
        assert!(!first.question.is_empty());
        assert!(!first.answer.is_empty());
        assert!(first.intent.is_some());

        println!(
            "✓ Step 3: Converted Dialogue to {} TrainingExample(s)",
            examples.len()
        );
        println!("  Question: {}", first.question);
        println!("  Answer: {}", first.answer);
    }

    /// Test: TrainingExample to TrainingSource conversion
    #[test]
    fn test_example_to_training_source() {
        let example =
            TrainingExample::qa("Test question", "Test answer").with_intent(IntentKind::Question);

        // Convert TrainingExample to TrainingSource format
        let json_content = serde_json::to_string(&example).expect("Failed to serialize example");

        let source = TrainingSource {
            source_type: TrainingSourceType::QaJson,
            name: Some("test_example".to_string()),
            value: None,
            mime: Some("application/json".to_string()),
            content: Some(json_content.clone()),
            target_memory: Some(example.curriculum.target_memory),
            memory_channels: Some(example.channels.clone()),
            stream: Default::default(),
        };

        // Verify source structure
        assert_eq!(source.source_type, TrainingSourceType::QaJson);
        assert!(source.content.is_some());

        // Verify we can deserialize back
        let parsed: TrainingExample =
            serde_json::from_str(&json_content).expect("Failed to deserialize");
        assert_eq!(parsed.question, "Test question");
        assert_eq!(parsed.answer, "Test answer");

        println!("✓ Step 4: Converted TrainingExample to TrainingSource");
        println!(
            "  JSON (truncated): {}",
            &json_content.chars().take(100).collect::<String>()
        );
    }

    /// Test: Full pipeline - generate, serialize, parse, validate
    #[test]
    fn test_full_pipeline_continuous_flow() {
        // 1. Generate example with reasoning
        let example = TrainingExample::qa_with_reasoning(
            "Explain photosynthesis",
            "Photosynthesis is the process by which plants convert sunlight into energy.",
            ReasoningType::Explanatory,
            vec![
                (
                    "Photosynthesis occurs in plant cells",
                    ReasoningStepType::Premise,
                ),
                (
                    "Sunlight + CO2 + H2O → Glucose + O2",
                    ReasoningStepType::Inference,
                ),
                (
                    "Plants produce oxygen as byproduct",
                    ReasoningStepType::Conclusion,
                ),
            ],
        )
        .with_intent(IntentKind::Explain)
        .with_entities(vec!["photosynthesis".to_string(), "plants".to_string()])
        .with_phase(TrainingPhaseKind::Bootstrap)
        .with_curriculum_score(105);

        // 2. Validate quality gates
        let validation_errors = example
            .quality_gates
            .min_unit_discovery_efficiency
            .map(|min| {
                if min > 0.0 {
                    vec![]
                } else {
                    vec!["gate failed".to_string()]
                }
            })
            .unwrap_or_default();
        assert!(validation_errors.is_empty(), "Quality gates should pass");

        // 3. Serialize to JSON
        let json = serde_json::to_string_pretty(&example).expect("Serialization failed");

        // 4. Deserialize back
        let parsed: TrainingExample = serde_json::from_str(&json).expect("Deserialization failed");

        // 5. Verify round-trip integrity
        assert_eq!(example.question, parsed.question);
        assert_eq!(example.answer, parsed.answer);
        assert_eq!(example.entities, parsed.entities);
        assert_eq!(
            example.curriculum.curriculum_score,
            parsed.curriculum.curriculum_score
        );

        println!("✓ Full pipeline test passed!");
        println!("  Example: {} → {}", example.question, example.answer);
        println!(
            "  Reasoning steps: {}",
            parsed.reasoning.map(|r| r.steps.len()).unwrap_or(0)
        );
        println!("  Channels: {:?}", example.channels);
        println!("  Phase: {:?}", example.curriculum.phase_hint);
    }
}
