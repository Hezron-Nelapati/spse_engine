//! Classification trainer with Layer 18 feedback integration.
//!
//! Implements online learning from labeled seed data with spatial adjustment.

use crate::classification::{ClassificationCalculator, ClassificationPattern, ClassificationSignature};
use crate::config::ClassificationConfig;
use crate::memory::MemoryStore;
use crate::spatial_index::SpatialGrid;
use crate::types::{GroundTruth, IntentKind, ResolverMode, ToneKind};
use serde::{Deserialize, Serialize};

/// Training outcome for a single labeled turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingOutcome {
    /// Whether intent prediction matched expected
    pub intent_correct: bool,
    /// Whether tone prediction matched expected
    pub tone_correct: bool,
    /// Whether resolver mode matched expected
    pub resolver_correct: bool,
    /// The prediction made before storing
    pub prediction: crate::types::ClassificationResult,
    /// The expected ground truth
    pub expected: GroundTruth,
}

/// Report for a single training iteration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IterationReport {
    /// Total samples processed
    pub total: u64,
    /// Correct intent predictions
    pub intent_correct: u64,
    /// Correct tone predictions
    pub tone_correct: u64,
    /// Correct resolver predictions
    pub resolver_correct: u64,
}

impl IterationReport {
    /// Calculate intent accuracy.
    pub fn intent_accuracy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.intent_correct as f32 / self.total as f32
    }
    
    /// Calculate tone accuracy.
    pub fn tone_accuracy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.tone_correct as f32 / self.total as f32
    }
    
    /// Calculate resolver accuracy.
    pub fn resolver_accuracy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        self.resolver_correct as f32 / self.total as f32
    }
    
    /// Calculate average accuracy across all classifications.
    pub fn average_accuracy(&self) -> f32 {
        (self.intent_accuracy() + self.tone_accuracy() + self.resolver_accuracy()) / 3.0
    }
}

/// Final report after training completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalReport {
    /// Whether training converged to target accuracy
    pub converged: bool,
    /// Number of iterations run
    pub iterations: usize,
    /// Final accuracy achieved
    pub final_accuracy: f32,
    /// Detailed iteration report
    pub report: IterationReport,
}

/// Classification trainer with Layer 18 feedback.
pub struct ClassificationTrainer {
    /// Classification calculator
    calculator: ClassificationCalculator,
    /// Training configuration
    config: ClassificationConfig,
    /// Pattern merge threshold (for deduplication)
    pattern_merge_threshold: f32,
}

impl ClassificationTrainer {
    /// Create new trainer.
    pub fn new(calculator: ClassificationCalculator, config: ClassificationConfig) -> Self {
        Self {
            calculator,
            config,
            pattern_merge_threshold: 0.95,
        }
    }
    
    /// Ingest a labeled turn with online evaluation.
    /// Returns training outcome with prediction vs expected comparison.
    pub fn ingest_labeled_turn(
        &mut self,
        text: &str,
        label: &GroundTruth,
        memory: &mut MemoryStore,
        spatial: &mut SpatialGrid,
    ) -> TrainingOutcome {
        let sig = ClassificationSignature::compute(text, self.calculator.hasher());
        
        // 1. Predict before storing (online evaluation)
        let prediction = self.calculator.calculate(text, memory, spatial, &self.config);
        
        let intent_correct = prediction.intent == label.intent;
        let tone_correct = prediction.tone == label.tone;
        let resolver_correct = prediction.resolver_mode == label.resolver_mode;
        
        // 2. Find or create pattern
        if let Some(mut existing) = self.find_similar_pattern(&sig, memory, spatial) {
            // Layer 18 Feedback: Update utility based on outcome
            if intent_correct && tone_correct {
                existing.record_success();
                
                // Reinforce: Move toward query centroid (attract)
                spatial.attract(existing.unit_id, &sig.semantic_centroid, 0.05);
            } else {
                existing.record_failure();
                
                // Penalize: Move away from query centroid (repel)
                spatial.repel(existing.unit_id, &sig.semantic_centroid, 0.10);
            }
            
            memory.update_classification_pattern(existing.clone());
        } else {
            // New Unit Discovery (Layer 2)
            let new_pattern = ClassificationPattern::new(
                sig.clone(),
                label.intent,
                label.tone,
                label.resolver_mode,
                label.domain.clone(),
            );
            
            // Set initial success/failure based on prediction match
            let mut new_pattern = new_pattern;
            if intent_correct && tone_correct {
                new_pattern.record_success();
            } else {
                new_pattern.record_failure();
            }
            
            // Store in memory (L4)
            memory.store_classification_pattern(new_pattern.clone());
            
            // Insert into spatial map (L5)
            spatial.insert(new_pattern.unit_id, &sig.semantic_centroid);
        }
        
        TrainingOutcome {
            intent_correct,
            tone_correct,
            resolver_correct,
            prediction,
            expected: label.clone(),
        }
    }
    
    /// Train until accuracy threshold is met or max iterations reached.
    pub fn train_until_threshold(
        &mut self,
        seed_data: &[LabeledDialogue],
        memory: &mut MemoryStore,
        spatial: &mut SpatialGrid,
        target_accuracy: f32,
        max_iterations: usize,
    ) -> FinalReport {
        let mut iteration = 0;
        let mut best_accuracy = 0.0;
        let mut best_report = IterationReport::default();
        
        while iteration < max_iterations {
            let mut report = IterationReport::default();
            
            for dialogue in seed_data {
                for turn in &dialogue.turns {
                    let outcome = self.ingest_labeled_turn(
                        &turn.content,
                        &GroundTruth::from(dialogue),
                        memory,
                        spatial,
                    );
                    
                    report.total += 1;
                    if outcome.intent_correct {
                        report.intent_correct += 1;
                    }
                    if outcome.tone_correct {
                        report.tone_correct += 1;
                    }
                    if outcome.resolver_correct {
                        report.resolver_correct += 1;
                    }
                }
            }
            
            let avg_accuracy = report.average_accuracy();
            
            if avg_accuracy >= target_accuracy {
                return FinalReport {
                    converged: true,
                    iterations: iteration + 1,
                    final_accuracy: avg_accuracy,
                    report,
                };
            }
            
            // Adjust weights based on performance
            self.adjust_weights(&report);
            
            if avg_accuracy > best_accuracy {
                best_accuracy = avg_accuracy;
                best_report = report.clone();
            }
            
            iteration += 1;
        }
        
        FinalReport {
            converged: false,
            iterations: max_iterations,
            final_accuracy: best_accuracy,
            report: best_report,
        }
    }
    
    /// Find a similar pattern in memory.
    fn find_similar_pattern(
        &self,
        sig: &ClassificationSignature,
        memory: &MemoryStore,
        spatial: &SpatialGrid,
    ) -> Option<ClassificationPattern> {
        let candidate_ids = spatial.nearby(sig.semantic_centroid, self.config.spatial_query_radius);
        
        for id in candidate_ids {
            if let Some(pattern) = memory.get_classification_pattern(id) {
                let similarity = self.calculator.cosine_similarity(sig, &pattern.signature);
                if similarity >= self.pattern_merge_threshold {
                    return Some(pattern);
                }
            }
        }
        
        None
    }
    
    /// Adjust feature weights based on iteration performance.
    fn adjust_weights(&mut self, report: &IterationReport) {
        // If intent accuracy low, increase semantic weight
        if report.intent_accuracy() < 0.7 {
            self.calculator.w_semantic *= 1.05;
        }
        
        // If tone accuracy low, increase derived scores weight
        if report.tone_accuracy() < 0.7 {
            self.calculator.w_derived *= 1.05;
        }
        
        // Normalize weights to sum to ~1.0
        let total = self.calculator.w_structure
            + self.calculator.w_punctuation
            + self.calculator.w_semantic
            + self.calculator.w_derived;
        
        if total > 1.5 || total < 0.5 {
            self.calculator.w_structure /= total;
            self.calculator.w_punctuation /= total;
            self.calculator.w_semantic /= total;
            self.calculator.w_derived /= total;
        }
    }
    
    /// Get reference to calculator.
    pub fn calculator(&self) -> &ClassificationCalculator {
        &self.calculator
    }
    
    /// Get mutable reference to calculator.
    pub fn calculator_mut(&mut self) -> &mut ClassificationCalculator {
        &mut self.calculator
    }
}

/// Labeled dialogue for training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledDialogue {
    /// Dialogue ID
    pub id: String,
    /// Expected intent
    pub intent: IntentKind,
    /// Expected tone
    pub expected_tone: ToneKind,
    /// Expected resolver mode
    pub resolver_mode: ResolverMode,
    /// Dialogue turns
    pub turns: Vec<LabeledTurn>,
    /// Metadata
    pub metadata: DialogueMetadata,
}

/// Expected unit counts per level for validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpectedUnitCount {
    /// Expected phrase-level units
    #[serde(default)]
    pub phrase: Option<u32>,
    /// Expected sentence-level units
    #[serde(default)]
    pub sentence: Option<u32>,
    /// Expected word-level units
    #[serde(default)]
    pub word: Option<u32>,
}

/// Labeled turn in a dialogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledTurn {
    /// Role (user/assistant)
    pub role: String,
    /// Turn content
    pub content: String,
    /// Expected entities to be extracted (for core training validation)
    #[serde(default)]
    pub expected_entities: Vec<String>,
    /// Expected anchor phrases (for Layer 8 anchor validation)
    #[serde(default)]
    pub expected_anchors: Vec<String>,
    /// Expected unit counts per level (for Layer 2 validation)
    #[serde(default)]
    pub expected_unit_count: ExpectedUnitCount,
    /// Source quality hint for trust scoring (assistant turns)
    #[serde(default)]
    pub source_quality: Option<f32>,
}

/// Memory target for core training (Layer 21 compliance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTarget {
    /// Staging episodic - high utility, requires corroboration for Core promotion
    #[default]
    StagingEpisodic,
    /// Direct to Core (only via consolidate_immediately or explicit corroboration)
    Core,
    /// Standard episodic memory
    Episodic,
}

/// Dialogue metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueMetadata {
    /// Domain context
    pub domain: Option<String>,
    /// Complexity level
    pub complexity: String,
    /// Target memory type (Core requires Layer 21 corroboration gate)
    #[serde(default)]
    pub memory_target: MemoryTarget,
    /// Memory channels to route to (Layer 4)
    #[serde(default)]
    pub channels: Vec<String>,
    /// Minimum corroboration count for Core promotion (default 2 per Appendix B5)
    #[serde(default = "default_corroboration_threshold")]
    pub corroboration_threshold: u32,
}

fn default_corroboration_threshold() -> u32 {
    2
}

/// Convert from dialogue_generator::Dialogue to LabeledDialogue for unified training
impl From<&crate::seed::Dialogue> for LabeledDialogue {
    fn from(dialogue: &crate::seed::Dialogue) -> Self {
        use crate::types::{IntentKind, ToneKind, ResolverMode};
        
        // Parse intent string to enum (default to Unknown if invalid)
        let intent = match dialogue.intent.as_str() {
            "Greeting" => IntentKind::Greeting,
            "Gratitude" => IntentKind::Gratitude,
            "Farewell" => IntentKind::Farewell,
            "Help" => IntentKind::Help,
            "Clarify" => IntentKind::Clarify,
            "Rewrite" => IntentKind::Rewrite,
            "Verify" => IntentKind::Verify,
            "Continue" => IntentKind::Continue,
            "Forget" => IntentKind::Forget,
            "Question" => IntentKind::Question,
            "Summarize" => IntentKind::Summarize,
            "Explain" => IntentKind::Explain,
            "Compare" => IntentKind::Compare,
            "Extract" => IntentKind::Extract,
            "Analyze" => IntentKind::Analyze,
            "Plan" => IntentKind::Plan,
            "Act" => IntentKind::Act,
            "Recommend" => IntentKind::Recommend,
            "Classify" => IntentKind::Classify,
            "Translate" => IntentKind::Translate,
            "Debug" => IntentKind::Debug,
            "Critique" => IntentKind::Critique,
            "Brainstorm" => IntentKind::Brainstorm,
            _ => IntentKind::Unknown,
        };
        
        // Parse tone (default to NeutralProfessional)
        let expected_tone = dialogue.expected_tone
            .as_ref()
            .and_then(|t| match t.as_str() {
                "Casual" => Some(ToneKind::Casual),
                "NeutralProfessional" => Some(ToneKind::NeutralProfessional),
                "Technical" => Some(ToneKind::Technical),
                "Formal" => Some(ToneKind::Formal),
                "Direct" => Some(ToneKind::Direct),
                "Empathetic" => Some(ToneKind::Empathetic),
                _ => None,
            })
            .unwrap_or(ToneKind::NeutralProfessional);
        
        // Parse resolver mode (default to Balanced)
        let resolver_mode = dialogue.resolver_mode
            .as_ref()
            .and_then(|r| match r.as_str() {
                "Deterministic" => Some(ResolverMode::Deterministic),
                "Balanced" => Some(ResolverMode::Balanced),
                "Exploratory" => Some(ResolverMode::Exploratory),
                _ => None,
            })
            .unwrap_or(ResolverMode::Balanced);
        
        // Convert turns with extended fields
        let turns = dialogue.turns.iter().map(|t| LabeledTurn {
            role: t.role.clone(),
            content: t.content.clone(),
            expected_entities: t.expected_entities.clone(),
            expected_anchors: t.expected_anchors.clone(),
            expected_unit_count: ExpectedUnitCount {
                phrase: t.expected_unit_count.phrase,
                sentence: t.expected_unit_count.sentence,
                word: t.expected_unit_count.word,
            },
            source_quality: t.source_quality,
        }).collect();
        
        // Convert metadata with memory_target from dialogue
        let metadata = DialogueMetadata {
            domain: Some(dialogue.metadata.domain.clone()),
            complexity: dialogue.metadata.complexity.clone(),
            memory_target: match dialogue.metadata.memory_target {
                crate::seed::SeedMemoryTarget::StagingEpisodic => MemoryTarget::StagingEpisodic,
                crate::seed::SeedMemoryTarget::Core => MemoryTarget::Core,
                crate::seed::SeedMemoryTarget::Episodic => MemoryTarget::Episodic,
            },
            channels: dialogue.metadata.memory_channels.iter().map(|c| format!("{:?}", c)).collect(),
            corroboration_threshold: dialogue.metadata.corroboration_threshold,
        };
        
        Self {
            id: dialogue.id.clone(),
            intent,
            expected_tone,
            resolver_mode,
            turns,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClassificationConfig;
    
    #[test]
    fn test_iteration_report_accuracy() {
        let mut report = IterationReport::default();
        
        assert_eq!(report.intent_accuracy(), 0.0);
        
        report.total = 10;
        report.intent_correct = 8;
        report.tone_correct = 7;
        report.resolver_correct = 9;
        
        assert!((report.intent_accuracy() - 0.8).abs() < 0.01);
        assert!((report.tone_accuracy() - 0.7).abs() < 0.01);
        assert!((report.resolver_accuracy() - 0.9).abs() < 0.01);
        assert!((report.average_accuracy() - 0.8).abs() < 0.01);
    }
    
    #[test]
    fn test_trainer_creation() {
        let calculator = ClassificationCalculator::new();
        let config = ClassificationConfig::default();
        let trainer = ClassificationTrainer::new(calculator, config);
        
        assert!((trainer.calculator.w_semantic - 0.35).abs() < 0.01);
    }
    
    #[test]
    fn test_weight_adjustment() {
        let calculator = ClassificationCalculator::new();
        let config = ClassificationConfig::default();
        let mut trainer = ClassificationTrainer::new(calculator, config);
        
        let mut report = IterationReport::default();
        report.total = 10;
        report.intent_correct = 5; // 50% accuracy
        report.tone_correct = 5; // 50% accuracy
        report.resolver_correct = 10;
        
        trainer.adjust_weights(&report);
        
        // Weights should have been adjusted
        assert!(trainer.calculator.w_semantic > 0.35);
        assert!(trainer.calculator.w_derived > 0.20);
    }
}
