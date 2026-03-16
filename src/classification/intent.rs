use crate::config::{FineResolverConfig, ReasoningLoopConfig, RetrievalThresholds};
use crate::types::{
    ConfidenceStats, ContextMatrix, IntentFallbackMode, IntentKind, IntentProfile, ScoredCandidate,
    SearchDecision, SequenceState,
};

pub struct IntentDetector;

impl IntentDetector {
    pub fn calculate_entropy(scores: &[ScoredCandidate]) -> f32 {
        let total: f32 = scores
            .iter()
            .map(|candidate| candidate.score.max(0.0))
            .sum();
        if total <= f32::EPSILON {
            return 0.0;
        }

        let entropy = scores.iter().fold(0.0, |entropy, candidate| {
            let probability = candidate.score.max(0.0) / total;
            if probability <= f32::EPSILON {
                entropy
            } else {
                entropy - (probability * probability.ln())
            }
        });
        let max_entropy = (scores.len().max(2) as f32).ln();
        if max_entropy <= f32::EPSILON {
            0.0
        } else {
            (entropy / max_entropy).clamp(0.0, 1.0)
        }
    }

    pub fn assess(
        context: &ContextMatrix,
        sequence: &SequenceState,
        stats: &ConfidenceStats,
        scored: &[ScoredCandidate],
        thresholds: &RetrievalThresholds,
        resolver_config: &FineResolverConfig,
        raw_input: &str,
        intent: &IntentProfile,
    ) -> SearchDecision {
        let entropy = Self::calculate_entropy(scored);
        let freshness_need = freshness_signal(raw_input, context, sequence, thresholds);
        let disagreement = stats.disagreement.clamp(0.0, 1.0);
        let cost_penalty = thresholds.cost_penalty;
        let score = (thresholds.w_entropy * entropy)
            + (thresholds.w_recency * freshness_need)
            + (thresholds.w_disagreement * disagreement)
            - (thresholds.w_cost * cost_penalty);

        let mut reasons = Vec::new();
        if entropy > thresholds.entropy_threshold {
            reasons.push("high_internal_entropy".to_string());
        }
        if freshness_need > thresholds.freshness_threshold {
            reasons.push("freshness_sensitive_query".to_string());
        }
        if disagreement > thresholds.disagreement_threshold {
            reasons.push("candidate_disagreement".to_string());
        }
        if matches!(intent.primary, IntentKind::Verify) {
            reasons.push("verification_request".to_string());
        }
        if matches!(intent.fallback_mode, IntentFallbackMode::RetrieveUnknown) {
            reasons.push("low_confidence_force_retrieval".to_string());
        }

        let social_or_local = matches!(
            intent.primary,
            IntentKind::Greeting
                | IntentKind::Gratitude
                | IntentKind::Farewell
                | IntentKind::Help
                | IntentKind::Forget
                | IntentKind::Continue
                | IntentKind::Rewrite
                | IntentKind::Translate
                | IntentKind::Brainstorm
        );
        let open_world_force = should_force_external_retrieval(
            raw_input,
            intent,
            scored,
            stats.mean_confidence,
            resolver_config.factual_intent_retrieval_threshold,
        );
        if open_world_force {
            reasons.push("open_world_low_confidence_retrieval".to_string());
        }
        if reasons.is_empty() {
            reasons.push("internal_evidence_sufficient".to_string());
        }

        SearchDecision {
            should_retrieve: !social_or_local
                && (matches!(intent.fallback_mode, IntentFallbackMode::RetrieveUnknown)
                    || score >= thresholds.decision_threshold
                    || open_world_force),
            score,
            entropy,
            freshness_need,
            disagreement,
            cost_penalty,
            reasons,
        }
    }

    pub fn should_trigger_reasoning(intent: &IntentProfile, config: &ReasoningLoopConfig) -> bool {
        if intent.confidence < config.trigger_confidence_floor {
            return true;
        }
        matches!(
            intent.primary,
            IntentKind::Plan | IntentKind::Analyze | IntentKind::Compare | IntentKind::Critique
        )
    }

    /// Assess if reasoning pattern retrieval is needed
    pub fn assess_structural_uncertainty(
        intent: &IntentProfile,
        confidence_stats: &ConfidenceStats,
        _context: &ContextMatrix,
    ) -> crate::types::ReasoningGateDecision {
        // Check if facts are known but logic is complex
        let has_factual_knowledge = confidence_stats.mean_confidence > 0.5;
        let has_logical_complexity = matches!(
            intent.primary,
            IntentKind::Analyze | IntentKind::Plan | IntentKind::Debug | IntentKind::Explain
        );
        let has_low_structural_confidence = confidence_stats.disagreement > 0.4;

        crate::types::ReasoningGateDecision {
            should_retrieve_reasoning: has_factual_knowledge
                && has_logical_complexity
                && has_low_structural_confidence,
            reasoning_type_hint: Self::infer_reasoning_type(intent),
        }
    }

    fn infer_reasoning_type(intent: &IntentProfile) -> crate::types::ReasoningType {
        match intent.primary {
            IntentKind::Analyze => crate::types::ReasoningType::Logical,
            IntentKind::Plan => crate::types::ReasoningType::Planning,
            IntentKind::Debug => crate::types::ReasoningType::Debugging,
            IntentKind::Explain => crate::types::ReasoningType::Explanatory,
            IntentKind::Verify => crate::types::ReasoningType::Verification,
            _ => crate::types::ReasoningType::General,
        }
    }
}

fn normalize(text: &str) -> String {
    text.split_whitespace()
        .map(|segment| segment.trim_matches(|ch: char| ch.is_control()))
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn token_set(normalized: &str) -> Vec<String> {
    normalized
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric())
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn freshness_signal(
    raw_input: &str,
    context: &ContextMatrix,
    sequence: &SequenceState,
    thresholds: &RetrievalThresholds,
) -> f32 {
    let lowered = raw_input.to_lowercase();
    let mut freshness: f32 = 0.0;
    for term in [
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "recently",
        "last week",
        "now",
        "currently",
        "price",
        "weather",
    ] {
        if lowered.contains(term) {
            freshness += thresholds.freshness_temporal_term_weight;
        }
    }
    if context.summary.contains("freshness_sensitive") {
        freshness += thresholds.freshness_context_weight;
    }
    if !sequence.task_entities.is_empty() {
        freshness += thresholds.freshness_task_entity_weight;
    }
    freshness.clamp(0.0, thresholds.freshness_max_value)
}

fn should_force_external_retrieval(
    raw_input: &str,
    intent: &IntentProfile,
    scored: &[ScoredCandidate],
    mean_confidence: f32,
    factual_intent_threshold: f32,
) -> bool {
    if matches!(intent.fallback_mode, IntentFallbackMode::RetrieveUnknown) {
        return true;
    }

    let top = match scored.first() {
        Some(candidate) => candidate,
        None => return is_open_world_intent(intent.primary),
    };

    if !is_open_world_intent(intent.primary) {
        return false;
    }

    // For factual questions (Question, Verify, Explain, etc.), force retrieval if:
    // 1. The top candidate score is below a reasonable threshold
    // 2. The candidate doesn't appear to directly answer the question
    // This prevents settling for vaguely related but unhelpful candidates
    let factual_intent = matches!(
        intent.primary,
        IntentKind::Question | IntentKind::Verify | IntentKind::Explain | IntentKind::Extract
    );

    if factual_intent && top.score < factual_intent_threshold {
        return true;
    }

    let normalized_prompt = normalize(raw_input);
    let prompt_tokens = token_set(&normalized_prompt);
    let normalized_candidate = normalize(&top.content);
    let candidate_tokens = token_set(&normalized_candidate);
    let overlap = token_overlap(&prompt_tokens, &candidate_tokens);
    let echo_like = overlap >= 0.72 || normalized_candidate.ends_with('?');
    let terse = candidate_tokens.len() <= 3;
    let low_confidence = mean_confidence < 0.46
        || top.score < 0.52
        || intent.certainty_bias < 0.0
        || intent.ambiguous;
    let likely_open_world = raw_input.contains('?')
        || raw_input.split_whitespace().any(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
        });

    likely_open_world && (echo_like || terse || low_confidence)
}

fn is_open_world_intent(intent: IntentKind) -> bool {
    matches!(
        intent,
        IntentKind::Question
            | IntentKind::Explain
            | IntentKind::Compare
            | IntentKind::Analyze
            | IntentKind::Extract
            | IntentKind::Verify
            | IntentKind::Clarify
            | IntentKind::Recommend
            | IntentKind::Critique
            | IntentKind::Debug
    )
}

fn token_overlap(lhs: &[String], rhs: &[String]) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }

    let mut intersection = 0usize;
    for token in lhs {
        if rhs.contains(token) {
            intersection += 1;
        }
    }

    intersection as f32 / lhs.len().max(rhs.len()) as f32
}

#[cfg(test)]
mod tests {
    use super::IntentDetector;
    use crate::types::{MemoryType, ScoreBreakdown, ScoredCandidate};
    use uuid::Uuid;

    #[test]
    fn assess_triggers_retrieval_on_high_entropy() {
        let scored = vec![
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "test candidate".to_string(),
                score: 0.3,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "another candidate".to_string(),
                score: 0.3,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
        ];
        let entropy = IntentDetector::calculate_entropy(&scored);
        assert!(entropy > 0.5);
    }
}
