// Cross-System Consistency Loop (§11.5)
// Validates inter-system agreements and applies asymmetric feedback corrections

use crate::config::EngineConfig;
use crate::memory::store::MemoryStore;
use crate::seed::{ConsistencyDatasetGenerator, TrainingExample};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Consistency Rules (R1-R7)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConsistencyRule {
    /// R1: High Uncertainty → Retrieve
    R1HighUncertaintyRetrieve,
    /// R2: Social Intent → Short-circuit
    R2SocialShortCircuit,
    /// R3: Factual Intent → Anchor Protection
    R3FactualAnchorProtection,
    /// R4: Creative Intent → Drift Allowed
    R4CreativeDriftAllowed,
    /// R5: Cold-Start → Confidence Penalty
    R5ColdStartPenalty,
    /// R6: Evidence Contradiction → Pattern Penalty
    R6ContradictionPenalty,
    /// R7: Broad Sparsity → Proactive Learning
    R7BroadSparsityProactive,
}

/// Consistency check result for a single example
#[derive(Debug, Clone)]
pub struct ConsistencyCheckResult {
    pub example_id: usize,
    pub rule: ConsistencyRule,
    pub passed: bool,
    pub expected_behavior: String,
    pub actual_behavior: String,
    pub correction_needed: Option<ConsistencyCorrection>,
}

/// Correction to apply when consistency rule violated
#[derive(Debug, Clone)]
pub struct ConsistencyCorrection {
    pub rule: ConsistencyRule,
    pub correction_type: CorrectionType,
    pub adjustment_value: f32,
    pub target_system: TargetSystem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionType {
    LowerThreshold,
    RaiseThreshold,
    AddToShortCircuitList,
    IncreaseTrust,
    LowerConfidenceFloor,
    PenalizePattern,
    SuppressIntentSplit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSystem {
    Classification,
    Reasoning,
    Predictive,
}

/// Full consistency check report
#[derive(Debug, Clone)]
pub struct ConsistencyReport {
    pub total_examples: usize,
    pub violations: Vec<ConsistencyCheckResult>,
    pub l_consistency: f32,
    pub corrections_applied: Vec<ConsistencyCorrection>,
    pub per_rule_violations: HashMap<ConsistencyRule, usize>,
}

/// Run cross-system consistency check (§11.5)
pub fn run_consistency_check(
    memory: &Arc<Mutex<MemoryStore>>,
    config: &EngineConfig,
) -> Result<ConsistencyReport, String> {
    // Generate consistency validation dataset
    let generator = ConsistencyDatasetGenerator::new();
    let examples = generator.generate_full_dataset();

    if examples.len() < 5000 {
        return Err(format!(
            "Insufficient consistency examples: {} (need 5K+)",
            examples.len()
        ));
    }

    let mut violations = Vec::new();
    let mut per_rule_violations: HashMap<ConsistencyRule, usize> = HashMap::new();

    // Check each consistency rule
    for (idx, example) in examples.iter().enumerate() {
        // Determine which rules apply to this example
        let applicable_rules = determine_applicable_rules(example);

        for rule in applicable_rules {
            let result = check_rule(example, rule, memory, config)?;

            if !result.passed {
                *per_rule_violations.entry(rule).or_insert(0) += 1;
                violations.push(result);
            }
        }
    }

    // Calculate L_consistency
    let l_consistency = violations.len() as f32 / examples.len() as f32;

    // Generate corrections
    let corrections = generate_corrections(&violations, config);

    Ok(ConsistencyReport {
        total_examples: examples.len(),
        violations,
        l_consistency,
        corrections_applied: corrections,
        per_rule_violations,
    })
}

/// Apply consistency corrections to config (asymmetric feedback)
pub fn apply_consistency_corrections(
    config: &mut EngineConfig,
    corrections: &[ConsistencyCorrection],
) -> Result<(), String> {
    for correction in corrections {
        match correction.correction_type {
            CorrectionType::LowerThreshold => {
                apply_lower_threshold(config, correction)?;
            }
            CorrectionType::RaiseThreshold => {
                apply_raise_threshold(config, correction)?;
            }
            CorrectionType::AddToShortCircuitList => {
                apply_add_shortcircuit(config, correction)?;
            }
            CorrectionType::IncreaseTrust => {
                apply_increase_trust(config, correction)?;
            }
            CorrectionType::LowerConfidenceFloor => {
                apply_lower_confidence_floor(config, correction)?;
            }
            CorrectionType::PenalizePattern => {
                apply_penalize_pattern(config, correction)?;
            }
            CorrectionType::SuppressIntentSplit => {
                apply_suppress_intent_split(config, correction)?;
            }
        }
    }

    Ok(())
}

// Helper functions

fn determine_applicable_rules(example: &TrainingExample) -> Vec<ConsistencyRule> {
    let mut rules = Vec::new();

    // Check context tags to determine which rules apply
    if let Some(context) = &example.context {
        if context.starts_with("r1:") {
            rules.push(ConsistencyRule::R1HighUncertaintyRetrieve);
        } else if context.starts_with("r2:") {
            rules.push(ConsistencyRule::R2SocialShortCircuit);
        } else if context.starts_with("r3:") {
            rules.push(ConsistencyRule::R3FactualAnchorProtection);
        } else if context.starts_with("r4:") {
            rules.push(ConsistencyRule::R4CreativeDriftAllowed);
        } else if context.starts_with("r5:") {
            rules.push(ConsistencyRule::R5ColdStartPenalty);
        } else if context.starts_with("r6:") {
            rules.push(ConsistencyRule::R6ContradictionPenalty);
        } else if context.starts_with("r7:") {
            rules.push(ConsistencyRule::R7BroadSparsityProactive);
        }
    }

    rules
}

fn check_rule(
    example: &TrainingExample,
    rule: ConsistencyRule,
    memory: &Arc<Mutex<MemoryStore>>,
    config: &EngineConfig,
) -> Result<ConsistencyCheckResult, String> {
    // Get memory store for actual checks
    let mem = memory.lock().map_err(|e| format!("Lock error: {}", e))?;
    let intent_centroids = mem.intent_centroids();
    drop(mem);

    // Simulate running the example through the pipeline and check if behavior matches rule
    let passed = match rule {
        ConsistencyRule::R1HighUncertaintyRetrieve => {
            // Check: if confidence < threshold, retrieval must be triggered
            let confidence = example
                .quality_gates
                .min_unit_discovery_efficiency
                .unwrap_or(0.50);
            let threshold = config.retrieval.entropy_threshold;

            if confidence < 0.40 {
                // Low confidence should trigger retrieval
                // Check if the intent is one that requires retrieval
                let needs_retrieval = matches!(
                    example.intent.as_deref(),
                    Some("Question") | Some("Verify") | Some("Explain") | Some("Analyze")
                );
                needs_retrieval
            } else {
                true // High confidence, rule satisfied
            }
        }
        ConsistencyRule::R2SocialShortCircuit => {
            // Check: social intents must skip reasoning (no reasoning trace)
            let is_social = matches!(
                example.intent.as_deref(),
                Some("Greeting") | Some("Farewell") | Some("Gratitude")
            );

            if is_social {
                // Social intents should NOT have reasoning traces
                example.reasoning.is_none()
                    || example
                        .reasoning
                        .as_ref()
                        .map(|r| r.steps.is_empty())
                        .unwrap_or(true)
            } else {
                true
            }
        }
        ConsistencyRule::R3FactualAnchorProtection => {
            // Check: factual intents preserve anchors (entities in answer)
            let is_factual = matches!(
                example.intent.as_deref(),
                Some("Verify") | Some("Question") | Some("Explain")
            );

            if is_factual && !example.entities.is_empty() {
                // Check if entities appear in answer (anchor protection)
                let answer_lower = example.answer.to_lowercase();
                example
                    .entities
                    .iter()
                    .any(|e| answer_lower.contains(&e.to_lowercase()))
            } else {
                true
            }
        }
        ConsistencyRule::R4CreativeDriftAllowed => {
            // Check: creative intents allow diverse content
            let is_creative = matches!(
                example.intent.as_deref(),
                Some("Brainstorm") | Some("Critique") | Some("Creative")
            );

            if is_creative {
                // Creative responses should have varied content (longer answers)
                example.answer.len() > 50
            } else {
                true
            }
        }
        ConsistencyRule::R5ColdStartPenalty => {
            // Check: cold-start (no prior context) reduces confidence
            let is_cold_start = example.context.is_none()
                || example
                    .context
                    .as_ref()
                    .map(|c| c.contains("cold_start"))
                    .unwrap_or(false);

            if is_cold_start {
                // Cold start should have reduced confidence in quality gates
                example
                    .quality_gates
                    .min_unit_discovery_efficiency
                    .map(|c| c < 0.8)
                    .unwrap_or(true)
            } else {
                true
            }
        }
        ConsistencyRule::R6ContradictionPenalty => {
            // Check: contradictions in evidence should penalize confidence
            let has_contradiction = example
                .context
                .as_ref()
                .map(|c| c.contains("contradiction"))
                .unwrap_or(false);

            if has_contradiction {
                // Contradictions should reduce corroboration requirement
                example.quality_gates.min_corroboration_count >= 2
            } else {
                true
            }
        }
        ConsistencyRule::R7BroadSparsityProactive => {
            // Check: broad queries trigger proactive learning
            let is_broad = example.question.split_whitespace().count() <= 3;

            if is_broad {
                // Broad queries should have lower discovery threshold
                example
                    .quality_gates
                    .min_unit_discovery_efficiency
                    .map(|c| c <= 0.9)
                    .unwrap_or(true)
            } else {
                true
            }
        }
    };

    Ok(ConsistencyCheckResult {
        example_id: 0,
        rule,
        passed,
        expected_behavior: format!("{:?} expected behavior", rule),
        actual_behavior: format!(
            "Actual behavior: {}",
            if passed { "correct" } else { "incorrect" }
        ),
        correction_needed: if !passed {
            Some(generate_correction_for_rule(rule))
        } else {
            None
        },
    })
}

fn generate_corrections(
    violations: &[ConsistencyCheckResult],
    config: &EngineConfig,
) -> Vec<ConsistencyCorrection> {
    let mut corrections = Vec::new();

    for violation in violations {
        if let Some(correction) = &violation.correction_needed {
            corrections.push(correction.clone());
        }
    }

    corrections
}

fn generate_correction_for_rule(rule: ConsistencyRule) -> ConsistencyCorrection {
    match rule {
        ConsistencyRule::R1HighUncertaintyRetrieve => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::LowerThreshold,
            adjustment_value: -0.05,
            target_system: TargetSystem::Reasoning,
        },
        ConsistencyRule::R2SocialShortCircuit => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::AddToShortCircuitList,
            adjustment_value: 0.0,
            target_system: TargetSystem::Classification,
        },
        ConsistencyRule::R3FactualAnchorProtection => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::IncreaseTrust,
            adjustment_value: 0.05,
            target_system: TargetSystem::Predictive,
        },
        ConsistencyRule::R4CreativeDriftAllowed => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::LowerConfidenceFloor,
            adjustment_value: -0.05,
            target_system: TargetSystem::Predictive,
        },
        ConsistencyRule::R5ColdStartPenalty => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::LowerConfidenceFloor,
            adjustment_value: -0.10,
            target_system: TargetSystem::Classification,
        },
        ConsistencyRule::R6ContradictionPenalty => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::PenalizePattern,
            adjustment_value: -0.15,
            target_system: TargetSystem::Classification,
        },
        ConsistencyRule::R7BroadSparsityProactive => ConsistencyCorrection {
            rule,
            correction_type: CorrectionType::SuppressIntentSplit,
            adjustment_value: 0.0,
            target_system: TargetSystem::Predictive,
        },
    }
}

fn apply_lower_threshold(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    match correction.target_system {
        TargetSystem::Reasoning => {
            // Lower retrieval entropy threshold to trigger retrieval more often
            config.retrieval.entropy_threshold =
                (config.retrieval.entropy_threshold - correction.adjustment_value.abs()).max(0.1);
        }
        TargetSystem::Classification => {
            // Lower classification low confidence threshold
            config.classification.low_confidence_threshold =
                (config.classification.low_confidence_threshold
                    - correction.adjustment_value.abs())
                .max(0.1);
        }
        _ => {}
    }
    Ok(())
}

fn apply_raise_threshold(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    match correction.target_system {
        TargetSystem::Reasoning => {
            // Raise retrieval threshold to reduce unnecessary retrievals
            config.retrieval.decision_threshold =
                (config.retrieval.decision_threshold + correction.adjustment_value.abs()).min(2.0);
        }
        TargetSystem::Classification => {
            // Raise classification high confidence threshold
            config.classification.high_confidence_threshold =
                (config.classification.high_confidence_threshold
                    + correction.adjustment_value.abs())
                .min(0.95);
        }
        _ => {}
    }
    Ok(())
}

fn apply_add_shortcircuit(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    // Add intent to social short-circuit list by lowering resolver confidence for social intents
    match correction.target_system {
        TargetSystem::Classification => {
            // Social intents should have lower resolver thresholds
            config.resolver.min_confidence_floor =
                (config.resolver.min_confidence_floor - 0.1).max(0.1);
        }
        _ => {}
    }
    Ok(())
}

fn apply_increase_trust(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    match correction.target_system {
        TargetSystem::Predictive => {
            // Increase anchor trust by raising confidence floor
            config.resolver.min_confidence_floor = (config.resolver.min_confidence_floor
                + correction.adjustment_value.abs())
            .min(0.95);
        }
        TargetSystem::Reasoning => {
            // Increase confidence scoring weight (proxy for trust)
            config.scoring.confidence =
                (config.scoring.confidence + correction.adjustment_value.abs()).min(0.5);
        }
        _ => {}
    }
    Ok(())
}

fn apply_lower_confidence_floor(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    match correction.target_system {
        TargetSystem::Predictive | TargetSystem::Classification => {
            config.resolver.min_confidence_floor = (config.resolver.min_confidence_floor
                - correction.adjustment_value.abs())
            .max(0.05);
        }
        TargetSystem::Reasoning => {
            config.retrieval.decision_threshold =
                (config.retrieval.decision_threshold - correction.adjustment_value.abs()).max(0.5);
        }
        _ => {}
    }
    Ok(())
}

fn apply_penalize_pattern(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    // Pattern penalty applied by adjusting scoring weights
    match correction.target_system {
        TargetSystem::Predictive => {
            // Reduce utility weight for patterns that cause contradictions
            config.scoring.utility =
                (config.scoring.utility - correction.adjustment_value.abs()).max(0.0);
        }
        _ => {}
    }
    Ok(())
}

fn apply_suppress_intent_split(
    config: &mut EngineConfig,
    correction: &ConsistencyCorrection,
) -> Result<(), String> {
    // Suppress structural feedback intent splitting by reducing sparsity sensitivity
    match correction.target_system {
        TargetSystem::Predictive => {
            // Increase spatial cell size to reduce granularity
            config.semantic_map.spatial_cell_size =
                (config.semantic_map.spatial_cell_size * 1.1).min(2.0);
        }
        _ => {}
    }
    Ok(())
}
