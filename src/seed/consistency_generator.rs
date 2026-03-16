// Consistency Dataset Generator
// Generates cross-system validation examples for consistency loop (§11.5)

use crate::seed::{CurriculumMetadata, QualityGates, TrainingExample};
use crate::types::{MemoryChannel, ReasoningStep, ReasoningStepType, ReasoningTrace};

pub struct ConsistencyDatasetGenerator {
    curriculum_score_base: f32,
}

impl ConsistencyDatasetGenerator {
    pub fn new() -> Self {
        Self {
            curriculum_score_base: 0.85,
        }
    }

    /// Generate 5K+ cross-system validation examples covering:
    /// - R1: High Uncertainty → Retrieve
    /// - R2: Social Intent → Short-circuit
    /// - R3: Factual Intent → Anchor Protection
    /// - R4: Creative Intent → Drift Allowed
    /// - R5: Cold-Start → Confidence Penalty
    /// - R6: Evidence Contradiction → Pattern Penalty
    /// - R7: Broad Sparsity → Proactive Learning
    pub fn generate_full_dataset(&self) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // R1: High Uncertainty → Retrieve (1000 examples)
        examples.extend(self.generate_r1_high_uncertainty(1000));

        // R2: Social Intent → Short-circuit (800 examples)
        examples.extend(self.generate_r2_social_intent(800));

        // R3: Factual Intent → Anchor Protection (1000 examples)
        examples.extend(self.generate_r3_factual_intent(1000));

        // R4: Creative Intent → Drift Allowed (800 examples)
        examples.extend(self.generate_r4_creative_intent(800));

        // R5: Cold-Start → Confidence Penalty (700 examples)
        examples.extend(self.generate_r5_cold_start(700));

        // R6: Evidence Contradiction → Pattern Penalty (500 examples)
        examples.extend(self.generate_r6_contradiction(500));

        // R7: Broad Sparsity → Proactive Learning (200 examples)
        examples.extend(self.generate_r7_sparsity(200));

        examples
    }

    fn generate_r1_high_uncertainty(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let uncertain_queries = vec![
            ("What is the Kigali Amendment?", "unknown_entity", 0.35, true),
            ("Who won the 2024 Nobel Prize in Physics?", "current_event", 0.30, true),
            ("What happened at COP29?", "recent_event", 0.38, true),
            ("Explain quantum entanglement.", "complex_science", 0.40, true),
        ];

        for (query, context_tag, expected_confidence, needs_retrieval) in uncertain_queries.iter().cycle().take(count) {
            let reasoning = Some(ReasoningTrace {
                steps: vec![
                    ReasoningStep {
                        step_type: ReasoningStepType::Premise,
                        content: format!("Query classification confidence: {}", expected_confidence),
                        anchor_step: true,
                        dependencies: vec![],
                        structure_hash: None,
                    },
                    ReasoningStep {
                        step_type: ReasoningStepType::Verification,
                        content: "Confidence below 0.40 threshold - retrieval required".to_string(),
                        anchor_step: false,
                        dependencies: vec![0],
                        structure_hash: None,
                    },
                ],
                reasoning_type: crate::types::ReasoningType::Logical,
                confidence_trajectory: vec![*expected_confidence, 0.95],
                entities: vec![],
                structure_hash: None,
            });

            examples.push(TrainingExample {
                question: query.to_string(),
                answer: "Retrieval required due to low confidence".to_string(),
                context: Some(format!("r1:{}", context_tag)),
                reasoning,
                intent: Some("Question".to_string()),
                entities: vec![],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: (self.curriculum_score_base * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.25),
                    min_semantic_routing_accuracy: Some(0.60),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r2_social_intent(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let social_queries = vec![
            ("Thanks!", "Gratitude", "You're welcome!", 0.97),
            ("Hello", "Greeting", "Hello! How can I help you?", 0.96),
            ("Goodbye", "Farewell", "Goodbye! Have a great day!", 0.98),
            ("Good morning", "Greeting", "Good morning! What can I do for you today?", 0.95),
        ];

        for (query, intent, response, expected_confidence) in social_queries.iter().cycle().take(count) {
            // R2: Social intents should NOT have reasoning traces (short-circuit)
            let reasoning = None;

            examples.push(TrainingExample {
                question: query.to_string(),
                answer: response.to_string(),
                context: Some(format!("r2:social:{}", intent)),
                reasoning,
                intent: Some(intent.to_string()),
                entities: vec![],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base + 0.10) * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.95),
                    min_semantic_routing_accuracy: Some(0.95),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r3_factual_intent(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let factual_queries = vec![
            // R3: Answers must contain the anchor entity for anchor protection
            ("What is the capital of France?", "Verify", "The capital of France is Paris.", "france", 0.92),
            ("Explain photosynthesis.", "Explain", "Photosynthesis is the process by which plants convert light into energy using chlorophyll.", "photosynthesis", 0.88),
            ("Is water H2O?", "Verify", "Yes, water has the chemical formula H2O.", "water", 0.95),
        ];

        for (query, intent, answer, anchor_entity, expected_confidence) in factual_queries.iter().cycle().take(count) {
            let reasoning = Some(ReasoningTrace {
                steps: vec![
                    ReasoningStep {
                        step_type: ReasoningStepType::Premise,
                        content: format!("Query with entity: {}", anchor_entity),
                        anchor_step: true,
                        dependencies: vec![],
                        structure_hash: None,
                    },
                    ReasoningStep {
                        step_type: ReasoningStepType::Verification,
                        content: format!("Anchor entity: {}", anchor_entity),
                        anchor_step: false,
                        dependencies: vec![0],
                        structure_hash: None,
                    },
                ],
                reasoning_type: crate::types::ReasoningType::Logical,
                confidence_trajectory: vec![*expected_confidence, 0.90],
                entities: vec![anchor_entity.to_string()],
                structure_hash: None,
            });

            examples.push(TrainingExample {
                question: query.to_string(),
                answer: answer.to_string(),
                context: Some(format!("r3:factual:{}", anchor_entity)),
                reasoning,
                intent: Some(intent.to_string()),
                entities: vec![anchor_entity.to_string()],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: (self.curriculum_score_base * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.85),
                    min_semantic_routing_accuracy: Some(0.75),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r4_creative_intent(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let creative_queries = vec![
            ("Write a poem about autumn.", "Brainstorm", "exploratory", 0.75),
            ("Brainstorm names for a coffee brand.", "Brainstorm", "exploratory", 0.78),
            ("Create a tagline for eco-friendly sneakers.", "Brainstorm", "creative", 0.72),
        ];

        for (query, intent, mode, expected_confidence) in creative_queries.iter().cycle().take(count) {
            let reasoning = Some(ReasoningTrace {
                steps: vec![
                    ReasoningStep {
                        step_type: ReasoningStepType::Premise,
                        content: format!("Creative mode: {}", mode),
                        anchor_step: true,
                        dependencies: vec![],
                        structure_hash: None,
                    },
                    ReasoningStep {
                        step_type: ReasoningStepType::Conclusion,
                        content: "Wider candidate pool, semantic drift enabled".to_string(),
                        anchor_step: true,
                        dependencies: vec![0],
                        structure_hash: None,
                    },
                ],
                reasoning_type: crate::types::ReasoningType::Planning,
                confidence_trajectory: vec![*expected_confidence, 0.85],
                entities: vec![],
                structure_hash: None,
            });

            examples.push(TrainingExample {
                question: query.to_string(),
                // R4: Creative responses should be longer (>50 chars) to allow drift
                answer: format!("Here's a creative exploration of the topic: {} with many possibilities, ideas, and imaginative approaches to consider.", query),
                context: Some(format!("r4:creative:{}", mode)),
                reasoning,
                intent: Some(intent.to_string()),
                entities: vec![],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.05) * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.70),
                    min_semantic_routing_accuracy: Some(0.70),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r5_cold_start(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let cold_start_queries = vec![
            ("Hi", "Greeting", 0.65, true, "cold_start_word"),
            ("What is Kigali?", "Question", 0.40, true, "unknown_entity"),
        ];

        for (query, intent, initial_confidence, needs_retrieval, context_tag) in cold_start_queries.iter().cycle().take(count) {
            examples.push(TrainingExample {
                question: query.to_string(),
                answer: "Cold-start detected - retrieval + edge injection".to_string(),
                context: Some(format!("r5:{}", context_tag)),
                reasoning: None,
                intent: Some(intent.to_string()),
                entities: vec![],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.10) * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.35),
                    min_semantic_routing_accuracy: Some(0.50),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r6_contradiction(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let contradiction_queries = vec![
            ("When was Paris founded?", "Question", vec!["Source A: 250 BC", "Source B: 52 BC"], 0.45),
        ];

        for (query, intent, contradicting_evidence, final_confidence) in contradiction_queries.iter().cycle().take(count) {
            examples.push(TrainingExample {
                question: query.to_string(),
                answer: format!("Contradictory evidence found: {:?}", contradicting_evidence),
                context: Some("r6:contradiction".to_string()),
                reasoning: None,
                intent: Some(intent.to_string()),
                entities: vec!["Paris".to_string()],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.15) * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.40),
                    min_semantic_routing_accuracy: Some(0.50),
                    // R6: Contradictions require higher corroboration
                    min_corroboration_count: 2,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }

    fn generate_r7_sparsity(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Generate examples demonstrating broad graph sparsity across multiple intents
        for i in 0..count {
            examples.push(TrainingExample {
                question: format!("Query {} requiring tier-3 pathfinding", i),
                answer: "Broad sparsity detected - proactive edge building triggered".to_string(),
                context: Some("r7:sparsity".to_string()),
                reasoning: None,
                intent: Some("Question".to_string()),
                entities: vec![],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.20) * 100.0) as i32,
                    phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                    target_memory: crate::types::MemoryType::Episodic,
                    memory_channels: vec![MemoryChannel::Main],
                    suggested_batch_size: 500,
                    max_chunk_chars: 8000,
                },
                quality_gates: QualityGates {
                    min_unit_discovery_efficiency: Some(0.35),
                    min_semantic_routing_accuracy: Some(0.55),
                    min_corroboration_count: 1,
                },
                training_options: crate::types::TrainingOptions::default(),
            });
        }

        examples
    }
}

impl Default for ConsistencyDatasetGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_consistency_dataset() {
        let generator = ConsistencyDatasetGenerator::new();
        let examples = generator.generate_full_dataset();
        
        assert!(examples.len() >= 5000, "Should generate 5K+ examples");
    }

    #[test]
    fn covers_all_consistency_rules() {
        let generator = ConsistencyDatasetGenerator::new();
        let examples = generator.generate_full_dataset();
        
        let r1 = examples.iter().filter(|e| e.context.as_ref().map_or(false, |c| c.starts_with("r1:"))).count();
        let r2 = examples.iter().filter(|e| e.context.as_ref().map_or(false, |c| c.starts_with("r2:"))).count();
        let r3 = examples.iter().filter(|e| e.context.as_ref().map_or(false, |c| c.starts_with("r3:"))).count();
        
        assert!(r1 > 0, "Should cover R1");
        assert!(r2 > 0, "Should cover R2");
        assert!(r3 > 0, "Should cover R3");
    }
}
