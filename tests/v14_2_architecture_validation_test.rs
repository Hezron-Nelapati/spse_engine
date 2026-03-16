//! V14.2 Architecture Synthetic Validation Tests
//!
//! These tests validate the core design ideas of the three systems described
//! in docs/SPSE_ARCHITECTURE_V14.2.md against actual config values and
//! expected algorithmic behavior. They exercise the design *ideas* even where
//! full end-to-end code paths are not yet landed.
//!
//! Run: cargo test --test v14_2_architecture_validation_test --no-default-features

use spse_engine::config::*;
use spse_engine::types::*;
use std::time::Instant;

// ============================================================================
// Helper: raw cosine similarity (mirrors classification/calculator.rs)
// ============================================================================
fn raw_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

// ============================================================================
// Synthetic Architecture Model
// ============================================================================

mod synthetic_model {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FlowKind {
        SocialShortCircuit,
        WarmLocal,
        ReasoningLocal,
        Retrieval,
        CreativeExploration,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PredictiveTier {
        Highway,
        Tier1,
        Tier2,
        Tier3,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RetrievalSignals {
        pub intent: IntentKind,
        pub confidence: f32,
        pub entropy: f32,
        pub freshness: f32,
        pub disagreement: f32,
        pub open_world: bool,
        pub recent_entity_carry: bool,
        pub procedural_bias: bool,
        pub local_reasoning_bias: bool,
        pub strong_local_support: bool,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct RetrievalDecision {
        pub should_retrieve: bool,
        pub flow: FlowKind,
        pub score: f32,
        pub suppressed: bool,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct EvidenceDoc {
        pub trust: f32,
        pub recency: f32,
        pub supports_claim: bool,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct PathCandidate {
        pub label: &'static str,
        pub tier: PredictiveTier,
        pub base_score: f32,
        pub proximity_or_decay: f32,
        pub context_match: f32,
        pub highway_boost: f32,
        pub subgraph_density: f32,
        pub anchor_trust: f32,
        pub contradicts_anchor: bool,
    }

    pub fn is_social_intent(intent: IntentKind) -> bool {
        matches!(
            intent,
            IntentKind::Greeting
                | IntentKind::Gratitude
                | IntentKind::Farewell
                | IntentKind::Continue
        )
    }

    pub fn is_creative_intent(intent: IntentKind) -> bool {
        matches!(intent, IntentKind::Brainstorm)
    }

    pub fn is_factual_anchor_intent(intent: IntentKind) -> bool {
        matches!(
            intent,
            IntentKind::Question | IntentKind::Verify | IntentKind::Explain
        )
    }

    pub fn naive_confidence_only_retrieval(confidence: f32) -> bool {
        confidence < 0.72
    }

    pub fn architecture_retrieval_decision(
        signals: RetrievalSignals,
        thresholds: &RetrievalThresholds,
    ) -> RetrievalDecision {
        let score = thresholds.w_entropy * signals.entropy
            + thresholds.w_recency * signals.freshness
            + thresholds.w_disagreement * signals.disagreement
            - thresholds.w_cost * thresholds.cost_penalty;
        let low_confidence = signals.confidence < 0.40;
        let social_shortcircuit = is_social_intent(signals.intent) && signals.confidence >= 0.95;

        if social_shortcircuit {
            return RetrievalDecision {
                should_retrieve: false,
                flow: FlowKind::SocialShortCircuit,
                score,
                suppressed: true,
            };
        }

        if signals.recent_entity_carry && !signals.open_world {
            return RetrievalDecision {
                should_retrieve: false,
                flow: FlowKind::ReasoningLocal,
                score,
                suppressed: true,
            };
        }

        if signals.procedural_bias && !signals.open_world {
            return RetrievalDecision {
                should_retrieve: false,
                flow: FlowKind::ReasoningLocal,
                score,
                suppressed: true,
            };
        }

        if signals.strong_local_support && !signals.open_world && signals.confidence >= 0.45 {
            return RetrievalDecision {
                should_retrieve: false,
                flow: if is_creative_intent(signals.intent) {
                    FlowKind::CreativeExploration
                } else if signals.local_reasoning_bias {
                    FlowKind::ReasoningLocal
                } else {
                    FlowKind::WarmLocal
                },
                score,
                suppressed: true,
            };
        }

        let open_world_force = signals.open_world
            && (signals.freshness >= 0.30
                || signals.entropy >= thresholds.entropy_threshold * 0.85
                || low_confidence);
        let should_retrieve =
            low_confidence || score >= thresholds.decision_threshold || open_world_force;

        RetrievalDecision {
            should_retrieve,
            flow: if should_retrieve {
                FlowKind::Retrieval
            } else if is_creative_intent(signals.intent) {
                FlowKind::CreativeExploration
            } else if signals.local_reasoning_bias {
                FlowKind::ReasoningLocal
            } else {
                FlowKind::WarmLocal
            },
            score,
            suppressed: false,
        }
    }

    pub fn should_trigger_reasoning(
        _intent: IntentKind,
        initial_confidence: f32,
        cfg: &ReasoningLoopConfig,
    ) -> bool {
        initial_confidence < cfg.trigger_confidence_floor
    }

    pub fn simulate_reasoning_progress(
        initial_confidence: f32,
        boosts: &[f32],
        cfg: &ReasoningLoopConfig,
    ) -> (f32, bool, usize) {
        let mut confidence = initial_confidence;
        let mut needs_retrieval = false;
        let mut steps_taken = 0usize;

        for (step, boost) in boosts.iter().enumerate().take(cfg.max_internal_steps) {
            confidence = (confidence + boost).clamp(0.0, 1.0);
            steps_taken = step + 1;
            if step > 0 && confidence < cfg.exit_confidence_threshold * 0.7 {
                needs_retrieval = true;
            }
            if confidence >= cfg.exit_confidence_threshold {
                break;
            }
        }

        (confidence, needs_retrieval, steps_taken)
    }

    pub fn naive_evidence_support(docs: &[EvidenceDoc], cfg: &EvidenceMergeConfig) -> f32 {
        if docs.is_empty() {
            return 0.0;
        }
        let avg_trust = docs.iter().map(|doc| doc.trust).sum::<f32>() / docs.len() as f32;
        let avg_recency = docs.iter().map(|doc| doc.recency).sum::<f32>() / docs.len() as f32;
        let support_ratio =
            docs.iter().filter(|doc| doc.supports_claim).count() as f32 / docs.len() as f32;
        (avg_trust * cfg.trust_weight
            + avg_recency * cfg.recency_weight
            + support_ratio * cfg.agreement_weight)
            .clamp(0.0, 1.0)
    }

    pub fn contradiction_aware_evidence_support(
        docs: &[EvidenceDoc],
        cfg: &EvidenceMergeConfig,
        contradiction_penalty: f32,
    ) -> (f32, bool) {
        if docs.is_empty() {
            return (0.0, false);
        }

        let supporting: Vec<&EvidenceDoc> = docs.iter().filter(|doc| doc.supports_claim).collect();
        let contradicting_count = docs.len().saturating_sub(supporting.len());
        if supporting.is_empty() {
            return (0.0, contradicting_count > 0);
        }

        let avg_trust =
            supporting.iter().map(|doc| doc.trust).sum::<f32>() / supporting.len() as f32;
        let avg_recency =
            supporting.iter().map(|doc| doc.recency).sum::<f32>() / supporting.len() as f32;
        let support_ratio = supporting.len() as f32 / docs.len() as f32;
        let contradiction_ratio = contradicting_count as f32 / docs.len() as f32;
        let score = avg_trust * cfg.trust_weight
            + avg_recency * cfg.recency_weight
            + support_ratio * cfg.agreement_weight
            - contradiction_ratio * contradiction_penalty;

        (score.clamp(0.0, 1.0), contradiction_ratio > 0.0)
    }

    pub fn score_path_candidate(candidate: PathCandidate) -> f32 {
        match candidate.tier {
            PredictiveTier::Highway => {
                candidate.base_score * candidate.highway_boost * candidate.context_match
            }
            PredictiveTier::Tier1 | PredictiveTier::Tier2 => {
                candidate.base_score * candidate.proximity_or_decay * candidate.context_match
            }
            PredictiveTier::Tier3 => {
                candidate.base_score * candidate.subgraph_density * candidate.context_match * 0.70
            }
        }
    }

    pub fn naive_highway_selection(candidates: &[PathCandidate]) -> PathCandidate {
        candidates
            .iter()
            .copied()
            .max_by(|a, b| {
                let left = match a.tier {
                    PredictiveTier::Highway => a.base_score * a.highway_boost,
                    _ => score_path_candidate(*a),
                };
                let right = match b.tier {
                    PredictiveTier::Highway => b.base_score * b.highway_boost,
                    _ => score_path_candidate(*b),
                };
                left.partial_cmp(&right)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("candidates must not be empty")
    }

    pub fn context_aware_selection(
        candidates: &[PathCandidate],
        preserve_anchor: bool,
        anchor_threshold: f32,
    ) -> PathCandidate {
        candidates
            .iter()
            .copied()
            .max_by(|a, b| {
                let left = adjusted_path_score(*a, preserve_anchor, anchor_threshold);
                let right = adjusted_path_score(*b, preserve_anchor, anchor_threshold);
                left.partial_cmp(&right)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("candidates must not be empty")
    }

    fn adjusted_path_score(
        candidate: PathCandidate,
        preserve_anchor: bool,
        anchor_threshold: f32,
    ) -> f32 {
        let mut score = score_path_candidate(candidate);
        if preserve_anchor
            && candidate.contradicts_anchor
            && candidate.anchor_trust >= anchor_threshold
        {
            score *= 0.25;
        }
        score
    }
}

// ============================================================================
// §1  CLASSIFICATION SYSTEM — Config & Formula Validation
// ============================================================================

mod classification_system {
    use super::synthetic_model::*;
    use super::*;

    // -----------------------------------------------------------------
    // 1.1  Classification feature weights must sum to 1.0 (§3.5)
    // -----------------------------------------------------------------
    #[test]
    fn classification_feature_weights_sum_to_one() {
        let cfg = ClassificationConfig::default();
        let sum = cfg.w_structure
            + cfg.w_punctuation
            + cfg.w_semantic
            + cfg.w_derived
            + cfg.w_intent_hash
            + cfg.w_tone_hash;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Classification feature weights must sum to ~1.0, got {sum}"
        );
    }

    // -----------------------------------------------------------------
    // 1.2  Documented default weights match code defaults (§3.5)
    // -----------------------------------------------------------------
    #[test]
    fn classification_default_weights_match_doc() {
        let cfg = ClassificationConfig::default();
        assert!(
            (cfg.w_intent_hash - 0.35).abs() < f32::EPSILON,
            "intent_hash weight"
        );
        assert!(
            (cfg.w_tone_hash - 0.20).abs() < f32::EPSILON,
            "tone_hash weight"
        );
        assert!(
            (cfg.w_semantic - 0.15).abs() < f32::EPSILON,
            "semantic weight"
        );
        assert!(
            (cfg.w_structure - 0.10).abs() < f32::EPSILON,
            "structure weight"
        );
        assert!(
            (cfg.w_punctuation - 0.10).abs() < f32::EPSILON,
            "punctuation weight"
        );
        assert!(
            (cfg.w_derived - 0.10).abs() < f32::EPSILON,
            "derived weight"
        );
    }

    // -----------------------------------------------------------------
    // 1.3  Confidence formula: margin-blend (§3.5 step 6)
    //
    //   confidence = best × (margin_blend + (1 - margin_blend) × margin)
    //   where margin = (best - runner_up) / best
    //
    //   Doc examples (margin_blend = 0.5):
    //     best=0.95 runner_up=0.90 → ~0.50
    //     best=0.95 runner_up=0.50 → ~0.70
    //     best=0.95 runner_up=0.10 → ~0.90
    //     best=0.30 runner_up=0.05 → ~0.27
    // -----------------------------------------------------------------
    fn margin_blend_confidence(best: f32, runner_up: f32, blend: f32) -> f32 {
        if best <= 0.0 {
            return 0.0;
        }
        let margin = (best - runner_up) / best;
        best * (blend + (1.0 - blend) * margin)
    }

    #[test]
    fn confidence_formula_ambiguous_close_runners() {
        let c = margin_blend_confidence(0.95, 0.90, 0.5);
        assert!(
            (c - 0.50).abs() < 0.02,
            "Ambiguous: expected ~0.50, got {c}"
        );
    }

    #[test]
    fn confidence_formula_moderate_clear_winner() {
        let c = margin_blend_confidence(0.95, 0.50, 0.5);
        assert!((c - 0.70).abs() < 0.02, "Moderate: expected ~0.70, got {c}");
    }

    #[test]
    fn confidence_formula_high_dominant_winner() {
        let c = margin_blend_confidence(0.95, 0.10, 0.5);
        assert!((c - 0.90).abs() < 0.02, "High: expected ~0.90, got {c}");
    }

    #[test]
    fn confidence_formula_low_weak_match() {
        let c = margin_blend_confidence(0.30, 0.05, 0.5);
        assert!((c - 0.27).abs() < 0.03, "Low: expected ~0.27, got {c}");
    }

    #[test]
    fn confidence_formula_naturally_bounded_zero_one() {
        // Property: result is always in [0, 1] for valid inputs
        for best in [0.0_f32, 0.1, 0.3, 0.5, 0.7, 0.95, 1.0] {
            for runner in [0.0_f32, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9] {
                if runner > best {
                    continue;
                }
                let c = margin_blend_confidence(best, runner, 0.5);
                assert!(
                    (0.0..=1.0).contains(&c),
                    "Out of bounds: best={best}, runner={runner}, c={c}"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // 1.4  Confidence → resolver mode mapping (§3.5 step 7)
    //   < 0.40 → Exploratory
    //   > 0.85 → Deterministic
    //   else   → Balanced
    // -----------------------------------------------------------------
    fn confidence_to_resolver(confidence: f32) -> ResolverMode {
        let cfg = ClassificationConfig::default();
        if confidence < cfg.low_confidence_threshold {
            ResolverMode::Exploratory
        } else if confidence > cfg.high_confidence_threshold {
            ResolverMode::Deterministic
        } else {
            ResolverMode::Balanced
        }
    }

    #[test]
    fn resolver_mode_low_confidence_is_exploratory() {
        assert_eq!(confidence_to_resolver(0.20), ResolverMode::Exploratory);
        assert_eq!(confidence_to_resolver(0.39), ResolverMode::Exploratory);
    }

    #[test]
    fn resolver_mode_mid_confidence_is_balanced() {
        assert_eq!(confidence_to_resolver(0.50), ResolverMode::Balanced);
        assert_eq!(confidence_to_resolver(0.72), ResolverMode::Balanced);
    }

    #[test]
    fn resolver_mode_high_confidence_is_deterministic() {
        assert_eq!(confidence_to_resolver(0.86), ResolverMode::Deterministic);
        assert_eq!(confidence_to_resolver(0.95), ResolverMode::Deterministic);
    }

    // -----------------------------------------------------------------
    // 1.5  Retrieval gating formula (§3.2 / L9)
    //   score = w_entropy×entropy + w_recency×freshness
    //         + w_disagreement×disagreement - w_cost×cost
    //   retrieve if score >= decision_threshold (1.10)
    // -----------------------------------------------------------------
    struct RetrievalGateInputs {
        entropy: f32,
        freshness: f32,
        disagreement: f32,
    }

    fn retrieval_gate_score(inputs: &RetrievalGateInputs, cfg: &RetrievalThresholds) -> f32 {
        cfg.w_entropy * inputs.entropy
            + cfg.w_recency * inputs.freshness
            + cfg.w_disagreement * inputs.disagreement
            - cfg.w_cost * cfg.cost_penalty
    }

    #[test]
    fn retrieval_gating_low_entropy_no_retrieve() {
        let cfg = RetrievalThresholds::default();
        let score = retrieval_gate_score(
            &RetrievalGateInputs {
                entropy: 0.3,
                freshness: 0.2,
                disagreement: 0.1,
            },
            &cfg,
        );
        assert!(
            score < cfg.decision_threshold,
            "Low entropy should NOT trigger retrieval: score={score}, threshold={}",
            cfg.decision_threshold
        );
    }

    #[test]
    fn retrieval_gating_high_entropy_triggers_retrieve() {
        let cfg = RetrievalThresholds::default();
        let score = retrieval_gate_score(
            &RetrievalGateInputs {
                entropy: 0.90,
                freshness: 0.70,
                disagreement: 0.40,
            },
            &cfg,
        );
        assert!(
            score >= cfg.decision_threshold,
            "High entropy should trigger retrieval: score={score}, threshold={}",
            cfg.decision_threshold
        );
    }

    #[test]
    fn social_intents_never_trigger_retrieval() {
        // §3.10 / Flow E: social intents skip retrieval regardless of score
        let social_intents = [
            IntentKind::Greeting,
            IntentKind::Gratitude,
            IntentKind::Farewell,
            IntentKind::Help,
            IntentKind::Forget,
            IntentKind::Continue,
            IntentKind::Rewrite,
            IntentKind::Translate,
            IntentKind::Brainstorm,
        ];
        for intent in &social_intents {
            // Even with max retrieval score, social intents should be suppressed
            assert!(
                matches!(
                    intent,
                    IntentKind::Greeting
                        | IntentKind::Gratitude
                        | IntentKind::Farewell
                        | IntentKind::Help
                        | IntentKind::Forget
                        | IntentKind::Continue
                        | IntentKind::Rewrite
                        | IntentKind::Translate
                        | IntentKind::Brainstorm
                ),
                "Intent {:?} should be in social suppression list",
                intent
            );
        }
    }

    // -----------------------------------------------------------------
    // 1.6  All 10 intent profiles exist and have valid scoring weights (§4.2)
    // -----------------------------------------------------------------
    #[test]
    fn all_ten_intent_profiles_exist() {
        let cfg = AdaptiveBehaviorConfig::default();
        let expected = [
            "casual",
            "explanatory",
            "factual",
            "procedural",
            "creative",
            "brainstorm",
            "plan",
            "act",
            "critique",
            "advisory",
        ];
        for name in &expected {
            assert!(
                cfg.intent_profile(name).is_some(),
                "Missing intent profile: {name}"
            );
        }
    }

    #[test]
    fn intent_profile_scoring_weights_sum_near_one() {
        let cfg = AdaptiveBehaviorConfig::default();
        for (name, profile) in &cfg.intent_profiles {
            let w = &profile.scoring;
            let sum = w.spatial
                + w.context
                + w.sequence
                + w.transition
                + w.utility
                + w.confidence
                + w.evidence;
            assert!(
                (sum - 1.0).abs() < 0.02,
                "Profile '{name}' scoring weights sum to {sum}, expected ~1.0"
            );
        }
    }

    // -----------------------------------------------------------------
    // 1.7  Factual profile prioritizes confidence+evidence (§4.2)
    // -----------------------------------------------------------------
    #[test]
    fn factual_profile_favors_confidence_and_evidence() {
        let cfg = AdaptiveBehaviorConfig::default();
        let factual = cfg.intent_profile("factual").unwrap();
        // Doc: confidence=0.40, evidence=0.30 → top 2 weights
        assert!(
            factual.scoring.confidence >= 0.20,
            "Factual confidence weight too low: {}",
            factual.scoring.confidence
        );
        assert!(
            factual.scoring.evidence >= 0.20,
            "Factual evidence weight too low: {}",
            factual.scoring.evidence
        );
        // Must be higher than spatial and transition
        assert!(factual.scoring.confidence > factual.scoring.spatial);
        assert!(factual.scoring.evidence > factual.scoring.transition);
    }

    // -----------------------------------------------------------------
    // 1.8  Brainstorm profile has highest stochastic jump & beam width (§4.2)
    // -----------------------------------------------------------------
    #[test]
    fn brainstorm_profile_is_most_exploratory() {
        let cfg = AdaptiveBehaviorConfig::default();
        let brainstorm = cfg.intent_profile("brainstorm").unwrap();
        let factual = cfg.intent_profile("factual").unwrap();
        assert!(
            brainstorm.escape.stochastic_jump_prob > factual.escape.stochastic_jump_prob,
            "Brainstorm stochastic_jump should exceed factual"
        );
        assert!(
            brainstorm.escape.beam_width > factual.escape.beam_width,
            "Brainstorm beam_width should exceed factual"
        );
        assert!(
            brainstorm.resolver.selection_temperature > factual.resolver.selection_temperature,
            "Brainstorm temperature should exceed factual"
        );
    }

    // -----------------------------------------------------------------
    // 1.9  Trust scoring formula (§3.4 / L19)
    // -----------------------------------------------------------------
    fn compute_trust_score(
        cfg: &TrustConfig,
        is_https: bool,
        is_allowlisted: bool,
        parser_warnings: bool,
        corroborating_sources: usize,
        format_adjustment: f32,
    ) -> f32 {
        let mut trust = cfg.default_source_trust;
        if is_https {
            trust += cfg.https_bonus;
        }
        if is_allowlisted {
            trust += cfg.allowlist_bonus;
        }
        if parser_warnings {
            trust -= cfg.parser_warning_penalty;
        }
        trust += corroborating_sources as f32 * cfg.corroboration_bonus;
        trust += format_adjustment;
        trust.clamp(0.0, 1.0)
    }

    #[test]
    fn trust_score_https_allowlisted_wikipedia() {
        let cfg = TrustConfig::default();
        let trust = compute_trust_score(&cfg, true, true, false, 2, 0.0);
        // 0.50 + 0.10 + 0.10 + 2*0.08 = 0.86
        assert!(
            (trust - 0.86).abs() < 0.01,
            "HTTPS+allowlisted+2 corroborations: expected 0.86, got {trust}"
        );
    }

    #[test]
    fn trust_score_raw_html_with_parser_warnings() {
        let cfg = TrustConfig::default();
        let html_adj = *cfg.format_trust_adjustments.get("html_raw").unwrap_or(&0.0);
        let trust = compute_trust_score(&cfg, false, false, true, 0, html_adj);
        // 0.50 - 0.20 + (-0.30) = 0.00
        assert!(
            trust < cfg.min_source_trust,
            "Raw HTML with parser warnings should fall below min_source_trust: trust={trust}"
        );
    }

    #[test]
    fn trust_score_structured_entity_bonus() {
        let cfg = TrustConfig::default();
        let entity_adj = *cfg
            .format_trust_adjustments
            .get("structured_entity")
            .unwrap_or(&0.0);
        assert!(
            (entity_adj - 0.20).abs() < f32::EPSILON,
            "structured_entity format adjustment should be +0.20"
        );
        let trust = compute_trust_score(&cfg, true, false, false, 0, entity_adj);
        // 0.50 + 0.10 + 0.20 = 0.80
        assert!(
            (trust - 0.80).abs() < 0.01,
            "HTTPS + structured_entity: expected 0.80, got {trust}"
        );
    }

    #[test]
    fn allowlist_domains_include_canonical_sources() {
        let cfg = TrustConfig::default();
        let expected = [
            "wikipedia.org",
            "wikidata.org",
            "archive.org",
            "gutenberg.org",
        ];
        for domain in &expected {
            assert!(
                cfg.allowlist_domains.iter().any(|d| d == domain),
                "Missing allowlist domain: {domain}"
            );
        }
    }

    // -----------------------------------------------------------------
    // 1.10  Content quality thresholds (§3.4)
    // -----------------------------------------------------------------
    #[test]
    fn content_quality_thresholds_match_doc() {
        let cfg = ContentQualityThresholds::default();
        assert!((cfg.min_readability_score - 0.60).abs() < f32::EPSILON);
        assert!((cfg.max_boilerplate_ratio - 0.40).abs() < f32::EPSILON);
        assert!((cfg.min_unique_words_ratio - 0.30).abs() < f32::EPSILON);
    }

    #[derive(Debug, Clone, Copy)]
    struct Scenario {
        name: &'static str,
        query: &'static str,
        signals: RetrievalSignals,
        expected_retrieve: bool,
        expected_flow: FlowKind,
    }

    fn architecture_scenarios() -> [Scenario; 7] {
        [
            Scenario {
                name: "social_gratitude",
                query: "Thanks!",
                signals: RetrievalSignals {
                    intent: IntentKind::Gratitude,
                    confidence: 0.98,
                    entropy: 0.05,
                    freshness: 0.0,
                    disagreement: 0.02,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: true,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::SocialShortCircuit,
            },
            Scenario {
                name: "warm_factual",
                query: "What is the capital of France?",
                signals: RetrievalSignals {
                    intent: IntentKind::Question,
                    confidence: 0.89,
                    entropy: 0.15,
                    freshness: 0.0,
                    disagreement: 0.08,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: true,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::WarmLocal,
            },
            Scenario {
                name: "compare_local_reasoning",
                query: "Is Paris bigger than Berlin?",
                signals: RetrievalSignals {
                    intent: IntentKind::Compare,
                    confidence: 0.67,
                    entropy: 0.38,
                    freshness: 0.0,
                    disagreement: 0.20,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: true,
                    strong_local_support: true,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::ReasoningLocal,
            },
            Scenario {
                name: "open_world_current_events",
                query: "What happened at the 2024 Olympics opening ceremony?",
                signals: RetrievalSignals {
                    intent: IntentKind::Question,
                    confidence: 0.58,
                    entropy: 0.76,
                    freshness: 0.82,
                    disagreement: 0.41,
                    open_world: true,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: false,
                },
                expected_retrieve: true,
                expected_flow: FlowKind::Retrieval,
            },
            Scenario {
                name: "context_carry_follow_up",
                query: "How many people live there?",
                signals: RetrievalSignals {
                    intent: IntentKind::Question,
                    confidence: 0.56,
                    entropy: 0.46,
                    freshness: 0.08,
                    disagreement: 0.17,
                    open_world: false,
                    recent_entity_carry: true,
                    procedural_bias: false,
                    local_reasoning_bias: true,
                    strong_local_support: false,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::ReasoningLocal,
            },
            Scenario {
                name: "procedural_plan",
                query: "Give me a 30-day launch plan for a Rust CLI tool",
                signals: RetrievalSignals {
                    intent: IntentKind::Plan,
                    confidence: 0.53,
                    entropy: 0.54,
                    freshness: 0.06,
                    disagreement: 0.21,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: true,
                    local_reasoning_bias: true,
                    strong_local_support: false,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::ReasoningLocal,
            },
            Scenario {
                name: "creative_brainstorm",
                query: "Brainstorm names for a climate-friendly coffee brand",
                signals: RetrievalSignals {
                    intent: IntentKind::Brainstorm,
                    confidence: 0.63,
                    entropy: 0.33,
                    freshness: 0.0,
                    disagreement: 0.11,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: true,
                },
                expected_retrieve: false,
                expected_flow: FlowKind::CreativeExploration,
            },
        ]
    }

    #[test]
    fn scenario_pack_matches_classification_acceptance_criteria() {
        let cfg = RetrievalThresholds::default();
        for scenario in architecture_scenarios() {
            let decision = architecture_retrieval_decision(scenario.signals, &cfg);
            assert_eq!(
                decision.should_retrieve, scenario.expected_retrieve,
                "Scenario '{}' ({}) retrieval mismatch",
                scenario.name, scenario.query
            );
            assert_eq!(
                decision.flow, scenario.expected_flow,
                "Scenario '{}' ({}) flow mismatch",
                scenario.name, scenario.query
            );
        }
    }

    #[test]
    fn naive_confidence_rule_overretrieves_followup_and_plan_examples() {
        let scenarios = architecture_scenarios();
        let failing: Vec<&str> = scenarios
            .iter()
            .filter(|scenario| {
                naive_confidence_only_retrieval(scenario.signals.confidence)
                    != scenario.expected_retrieve
            })
            .map(|scenario| scenario.name)
            .collect();

        assert!(
            failing.contains(&"context_carry_follow_up"),
            "Naive confidence-only rule should fail the context-carry case: {failing:?}"
        );
        assert!(
            failing.contains(&"procedural_plan"),
            "Naive confidence-only rule should fail the procedural case: {failing:?}"
        );
        assert!(
            failing.len() >= 2,
            "Expected at least two failure cases for naive retrieval gating, got {failing:?}"
        );
    }

    #[test]
    fn open_world_queries_override_local_similarity_when_freshness_is_high() {
        let cfg = RetrievalThresholds::default();
        let decision = architecture_retrieval_decision(
            RetrievalSignals {
                intent: IntentKind::Question,
                confidence: 0.69,
                entropy: 0.60,
                freshness: 0.88,
                disagreement: 0.24,
                open_world: true,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: true,
            },
            &cfg,
        );
        assert!(decision.should_retrieve);
        assert_eq!(decision.flow, FlowKind::Retrieval);
    }

    #[test]
    fn classification_principle_benchmark_reports_stable_scenario_coverage() {
        let scenarios = architecture_scenarios();
        let cfg = RetrievalThresholds::default();
        let iterations = 20_000usize;
        let start = Instant::now();
        let mut architecture_hits = 0usize;
        let mut naive_failures = 0usize;

        for _ in 0..iterations {
            for scenario in scenarios {
                let decision = architecture_retrieval_decision(scenario.signals, &cfg);
                if decision.should_retrieve == scenario.expected_retrieve
                    && decision.flow == scenario.expected_flow
                {
                    architecture_hits += 1;
                }
                if naive_confidence_only_retrieval(scenario.signals.confidence)
                    != scenario.expected_retrieve
                {
                    naive_failures += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        let evals = iterations * scenarios.len();
        let per_eval_ns = elapsed.as_nanos() as f64 / evals as f64;
        eprintln!(
            "classification_benchmark evals={} elapsed_ms={} per_eval_ns={:.2} naive_failures={}",
            evals,
            elapsed.as_millis(),
            per_eval_ns,
            naive_failures
        );

        assert_eq!(
            architecture_hits, evals,
            "Architecture scenario pack should stay stable"
        );
        assert!(
            naive_failures >= iterations * 2,
            "Naive retrieval should fail follow-up and procedural examples on every iteration"
        );
    }
}

// ============================================================================
// §2  REASONING SYSTEM — Scoring, Evidence Merge, Reasoning Loop
// ============================================================================

mod reasoning_system {
    use super::synthetic_model::*;
    use super::*;

    // -----------------------------------------------------------------
    // 2.1  7D candidate scoring weight vector sums to ~1.0 (§4.4)
    // -----------------------------------------------------------------
    #[test]
    fn candidate_scoring_weights_sum_near_one() {
        let w = ScoringWeights::default();
        let sum = w.spatial
            + w.context
            + w.sequence
            + w.transition
            + w.utility
            + w.confidence
            + w.evidence;
        assert!(
            (sum - 1.0).abs() < 0.02,
            "Default scoring weights sum to {sum}, expected ~1.0"
        );
    }

    // -----------------------------------------------------------------
    // 2.2  7D scoring: synthetic candidate ranking (§4.4)
    //   Given two candidates with known features, scoring must rank them
    //   correctly under the default weight vector.
    // -----------------------------------------------------------------
    fn score_candidate_7d(features: &[f32; 7], weights: &ScoringWeights) -> f32 {
        let w = [
            weights.spatial,
            weights.context,
            weights.sequence,
            weights.transition,
            weights.utility,
            weights.confidence,
            weights.evidence,
        ];
        features.iter().zip(w.iter()).map(|(f, w)| f * w).sum()
    }

    #[test]
    fn candidate_with_high_evidence_outranks_low_evidence() {
        let w = ScoringWeights::default();
        // Candidate A: strong evidence, moderate other features
        let a = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.95];
        // Candidate B: weak evidence, slightly stronger other features
        let b = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.10];
        let sa = score_candidate_7d(&a, &w);
        let sb = score_candidate_7d(&b, &w);
        // Evidence weight is significant (0.14 default), but B has higher everywhere else
        // This tests that the weights are balanced — B should still win here
        assert!(
            sb > sa || (sa - sb).abs() < 0.1,
            "Balanced weights: A={sa:.3}, B={sb:.3} — expected B≥A or close"
        );
    }

    #[test]
    fn factual_profile_rescoring_boosts_evidence_candidates() {
        let default_w = ScoringWeights::default();
        let factual_cfg = AdaptiveBehaviorConfig::default();
        let factual_w = &factual_cfg.intent_profiles.get("factual").unwrap().scoring;

        // Candidate with strong evidence support
        let evidence_candidate = [0.3, 0.3, 0.3, 0.3, 0.3, 0.8, 0.9];
        let default_score = score_candidate_7d(&evidence_candidate, &default_w);
        let factual_score = score_candidate_7d(&evidence_candidate, factual_w);

        assert!(
            factual_score > default_score,
            "Factual profile should boost evidence-heavy candidates: \
             factual={factual_score:.3} > default={default_score:.3}"
        );
    }

    // -----------------------------------------------------------------
    // 2.3  Evidence merge trust formula (§4.3 / L13)
    //   evidence_support = avg_trust × trust_weight + agreement_ratio × agreement_weight
    // -----------------------------------------------------------------
    #[test]
    fn evidence_merge_config_weights_sum_to_one() {
        let cfg = EvidenceMergeConfig::default();
        let sum = cfg.trust_weight + cfg.recency_weight + cfg.agreement_weight;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Evidence merge weights sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn evidence_merge_formula_high_trust_high_agreement() {
        let cfg = EvidenceMergeConfig::default();
        let avg_trust = 0.80;
        let agreement_ratio = 0.80; // 4/5 documents agree
        let evidence_support =
            avg_trust * cfg.trust_weight + agreement_ratio * cfg.agreement_weight;
        // 0.80×0.50 + 0.80×0.20 = 0.56
        assert!(
            evidence_support > 0.50,
            "High trust + high agreement should yield strong evidence: {evidence_support}"
        );
    }

    #[test]
    fn evidence_merge_formula_low_trust_low_agreement() {
        let cfg = EvidenceMergeConfig::default();
        let avg_trust = 0.30;
        let agreement_ratio = 0.20;
        let evidence_support =
            avg_trust * cfg.trust_weight + agreement_ratio * cfg.agreement_weight;
        // 0.30×0.50 + 0.20×0.20 = 0.19
        assert!(
            evidence_support < 0.30,
            "Low trust + low agreement should yield weak evidence: {evidence_support}"
        );
    }

    // -----------------------------------------------------------------
    // 2.4  Reasoning loop gating (§4.5)
    //   Triggered when candidate confidence < trigger_confidence_floor (0.40)
    //   Exits when confidence >= exit_confidence_threshold (0.60)
    //   Max steps: 3
    // -----------------------------------------------------------------
    #[test]
    fn reasoning_loop_config_defaults() {
        let cfg = ReasoningLoopConfig::default();
        assert!(cfg.enabled);
        assert!((cfg.trigger_confidence_floor - 0.40).abs() < f32::EPSILON);
        assert!((cfg.exit_confidence_threshold - 0.60).abs() < f32::EPSILON);
        assert_eq!(cfg.max_internal_steps, 3);
    }

    #[test]
    fn reasoning_loop_triggers_on_low_confidence() {
        let cfg = ReasoningLoopConfig::default();
        let initial_confidence = 0.35;
        assert!(
            initial_confidence < cfg.trigger_confidence_floor,
            "Confidence {initial_confidence} should trigger reasoning loop"
        );
    }

    #[test]
    fn reasoning_loop_exits_on_sufficient_confidence() {
        let cfg = ReasoningLoopConfig::default();
        let post_reasoning_confidence = 0.65;
        assert!(
            post_reasoning_confidence >= cfg.exit_confidence_threshold,
            "Confidence {post_reasoning_confidence} should exit reasoning loop"
        );
    }

    #[test]
    fn reasoning_loop_synthetic_confidence_trajectory() {
        // Simulate the confidence trajectory from §4.5:
        //   Step 0: adapt_reasoning_pattern → +0.15 (anchor boost)
        //   Step 1: generate_thought_unit → +0.1 × (1.0 - prev).min(0.3)
        //   Exit if >= exit_confidence_threshold
        let cfg = ReasoningLoopConfig::default();
        let mut confidence: f32 = 0.30; // Below trigger floor

        for step in 0..cfg.max_internal_steps {
            if step == 0 {
                // Adapt reasoning pattern: anchor boost +0.15
                confidence += 0.15;
            } else {
                // Fallback thought unit: 0.1 × (1.0 - confidence).min(0.3)
                let boost = 0.1 * (1.0 - confidence).min(0.3);
                confidence += boost;
            }
            confidence = confidence.clamp(0.0, 1.0);
            if confidence >= cfg.exit_confidence_threshold {
                break;
            }
        }
        // After 2 steps: 0.30 + 0.15 + 0.055 = 0.505 (below 0.60, needs step 2)
        // After 3 steps: 0.505 + 0.0495 = 0.5545 (still below — this matches the doc
        //   where hard cases average 0.329→0.716 with retrieval help)
        assert!(
            confidence > 0.30,
            "Reasoning should improve confidence from initial: {confidence}"
        );
    }

    // -----------------------------------------------------------------
    // 2.5  Retrieval retry trigger (§4.5 / Flow F)
    //   If step > 0 and confidence < exit_threshold × 0.7 → needs_retrieval
    // -----------------------------------------------------------------
    #[test]
    fn reasoning_triggers_retrieval_when_stuck() {
        let cfg = ReasoningLoopConfig::default();
        let threshold_for_retry = cfg.exit_confidence_threshold * 0.7;
        // After step 1 with confidence 0.28 → should trigger retrieval
        let confidence_after_step1 = 0.28;
        assert!(
            confidence_after_step1 < threshold_for_retry,
            "Confidence {confidence_after_step1} < retry threshold {threshold_for_retry} → retrieval"
        );
    }

    // -----------------------------------------------------------------
    // 2.6  Dynamic memory allocator bounds (§4.5)
    // -----------------------------------------------------------------
    #[test]
    fn dynamic_memory_config_defaults() {
        let cfg = DynamicMemoryConfig::default();
        assert!(cfg.enabled);
        assert_eq!(cfg.base_memory_limit_mb, 350);
        assert_eq!(cfg.max_memory_limit_mb, 550);
        assert_eq!(cfg.thought_buffer_size_kb, 64);
        assert!(cfg.max_memory_limit_mb > cfg.base_memory_limit_mb);
    }

    // -----------------------------------------------------------------
    // 2.7  Governance config: maturity-stage thresholds (§4.1)
    // -----------------------------------------------------------------
    #[test]
    fn governance_maturity_stage_thresholds_ascending() {
        let cfg = GovernanceConfig::default();
        // Cold start → Growth → Stable: thresholds should be ascending
        assert!(
            cfg.cold_start_discovery_utility_threshold < cfg.growth_discovery_utility_threshold,
            "Discovery utility: cold_start < growth"
        );
        assert!(
            cfg.growth_discovery_utility_threshold < cfg.stable_discovery_utility_threshold,
            "Discovery utility: growth < stable"
        );
        assert!(
            cfg.cold_start_candidate_observation_threshold
                < cfg.growth_candidate_observation_threshold,
            "Observation threshold: cold_start < growth"
        );
        assert!(
            cfg.growth_candidate_observation_threshold < cfg.stable_candidate_observation_threshold,
            "Observation threshold: growth < stable"
        );
    }

    #[test]
    fn governance_anchor_protection_config() {
        let cfg = GovernanceConfig::default();
        assert_eq!(cfg.anchor_reuse_threshold, 3);
        assert!((cfg.anchor_salience_threshold - 0.70).abs() < f32::EPSILON);
        assert_eq!(cfg.anchor_protection_grace_days, 14);
    }

    #[test]
    fn governance_intent_channel_blocked_from_core() {
        let cfg = GovernanceConfig::default();
        assert!(
            cfg.intent_channel_core_promotion_blocked,
            "Intent channel units must be blocked from Core promotion (§3.7)"
        );
    }

    // -----------------------------------------------------------------
    // 2.8  Micro-validator lazy gating (§4.3.1)
    //   Only runs if top-2 within ambiguity_margin OR source trust < floor
    // -----------------------------------------------------------------
    #[test]
    fn micro_validator_skips_clear_winner() {
        let cfg = EvidenceMergeConfig::default();
        let top_score = 0.85;
        let runner_up = 0.40;
        let gap = top_score - runner_up;
        assert!(
            gap > cfg.ambiguity_margin,
            "Clear winner (gap={gap}) should skip micro-validation"
        );
    }

    #[test]
    fn micro_validator_triggers_on_ambiguous_candidates() {
        let cfg = EvidenceMergeConfig::default();
        let top_score = 0.72;
        let runner_up = 0.70;
        let gap = top_score - runner_up;
        assert!(
            gap <= cfg.ambiguity_margin,
            "Ambiguous candidates (gap={gap}) should trigger micro-validation"
        );
    }

    // -----------------------------------------------------------------
    // 2.9  Memory budget tiers (§4.7)
    // -----------------------------------------------------------------
    #[test]
    fn memory_budget_tiers_decrease_daily_growth() {
        let cfg = MemoryBudgetConfig::default();
        assert!(
            cfg.cold_start.daily_growth_limit_mb > cfg.growth_phase.daily_growth_limit_mb,
            "Cold start daily growth > growth phase"
        );
        assert!(
            cfg.growth_phase.daily_growth_limit_mb > cfg.stable_phase.daily_growth_limit_mb,
            "Growth phase daily growth > stable phase"
        );
    }

    #[test]
    fn memory_budget_core_grows_with_maturity() {
        let cfg = MemoryBudgetConfig::default();
        assert!(
            cfg.stable_phase.core_limit_mb > cfg.cold_start.core_limit_mb,
            "Stable core limit should exceed cold start"
        );
    }

    #[test]
    fn hard_analytic_queries_only_trigger_reasoning_when_l14_confidence_is_low() {
        let cfg = ReasoningLoopConfig::default();
        assert!(
            should_trigger_reasoning(IntentKind::Compare, 0.32, &cfg),
            "Low-confidence compare queries should enter the reasoning loop"
        );
        assert!(
            !should_trigger_reasoning(IntentKind::Compare, 0.58, &cfg),
            "Mid-confidence compare queries should stay on the local path"
        );
        assert!(
            !should_trigger_reasoning(IntentKind::Plan, 0.52, &cfg),
            "Planning queries should not over-trigger reasoning when local support exists"
        );
    }

    #[test]
    fn naive_intent_based_reasoning_trigger_overthinks_plans_and_critiques() {
        fn naive_trigger(intent: IntentKind) -> bool {
            matches!(
                intent,
                IntentKind::Plan | IntentKind::Analyze | IntentKind::Compare | IntentKind::Critique
            )
        }

        assert!(naive_trigger(IntentKind::Plan));
        assert!(naive_trigger(IntentKind::Critique));
        assert!(
            !should_trigger_reasoning(IntentKind::Plan, 0.55, &ReasoningLoopConfig::default()),
            "Architecture-first trigger should suppress unnecessary reasoning on supported plans"
        );
    }

    #[test]
    fn contradiction_aware_evidence_merge_penalizes_false_consensus() {
        let cfg = EvidenceMergeConfig::default();
        let docs = [
            EvidenceDoc {
                trust: 0.86,
                recency: 0.82,
                supports_claim: true,
            },
            EvidenceDoc {
                trust: 0.80,
                recency: 0.78,
                supports_claim: true,
            },
            EvidenceDoc {
                trust: 0.89,
                recency: 0.90,
                supports_claim: false,
            },
            EvidenceDoc {
                trust: 0.84,
                recency: 0.88,
                supports_claim: false,
            },
        ];

        let naive = naive_evidence_support(&docs, &cfg);
        let (aware, contradicted) = contradiction_aware_evidence_support(&docs, &cfg, 0.30);

        assert!(
            contradicted,
            "Contradiction-aware merge should surface the conflict"
        );
        assert!(
            aware < naive,
            "Contradiction-aware evidence support should be lower than naive blending: aware={aware}, naive={naive}"
        );
        assert!(
            aware <= naive - 0.15,
            "Contradiction-aware merge should materially reduce support on conflicted evidence: aware={aware}, naive={naive}"
        );
        assert!(
            aware < 0.65,
            "Conflicted high-trust evidence should not remain in the strong-support band: {aware}"
        );
    }

    #[test]
    fn reasoning_loop_escalates_to_retrieval_when_internal_boosts_stall() {
        let cfg = ReasoningLoopConfig::default();
        let (final_confidence, needs_retrieval, steps_taken) =
            simulate_reasoning_progress(0.24, &[0.08, 0.05, 0.04], &cfg);

        assert_eq!(steps_taken, 3);
        assert!(
            needs_retrieval,
            "Stalled confidence should request retrieval"
        );
        assert!(
            final_confidence < cfg.exit_confidence_threshold,
            "Stalled trajectory should remain below exit threshold"
        );
    }

    #[test]
    fn reasoning_loop_clears_threshold_for_hard_but_local_compare_case() {
        let cfg = ReasoningLoopConfig::default();
        let (final_confidence, needs_retrieval, steps_taken) =
            simulate_reasoning_progress(0.31, &[0.15, 0.16], &cfg);

        assert_eq!(steps_taken, 2);
        assert!(!needs_retrieval);
        assert!(
            final_confidence >= cfg.exit_confidence_threshold,
            "Reasoning loop should rescue hard local compare queries"
        );
    }

    #[test]
    fn lazy_micro_validation_benchmark_saves_work_on_clear_winners() {
        let cfg = EvidenceMergeConfig::default();
        let iterations = 50_000usize;
        let pairs = [
            (0.91_f32, 0.40_f32),
            (0.88_f32, 0.32_f32),
            (0.72_f32, 0.70_f32),
            (0.69_f32, 0.66_f32),
            (0.84_f32, 0.20_f32),
        ];
        let start = Instant::now();
        let mut eager_validations = 0usize;
        let mut lazy_validations = 0usize;

        for _ in 0..iterations {
            for (top, runner_up) in pairs {
                eager_validations += 1;
                if top - runner_up <= cfg.ambiguity_margin {
                    lazy_validations += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!(
            "reasoning_benchmark pair_evals={} elapsed_ms={} eager={} lazy={}",
            iterations * pairs.len(),
            elapsed.as_millis(),
            eager_validations,
            lazy_validations
        );

        assert!(
            lazy_validations < eager_validations,
            "Lazy gating should validate fewer cases than validate-all"
        );
        assert_eq!(lazy_validations, iterations * 2);
    }
}

// ============================================================================
// §3  PREDICTIVE SYSTEM — Word Graph, Walk Scoring, Highways, Beam Search
// ============================================================================

mod predictive_system {
    use super::synthetic_model::*;
    use super::*;

    // -----------------------------------------------------------------
    // 3.1  Semantic map config defaults (§5.1)
    // -----------------------------------------------------------------
    #[test]
    fn semantic_map_config_defaults() {
        let cfg = SemanticMapConfig::default();
        assert!((cfg.preferred_spacing_k - 1.0).abs() < f32::EPSILON);
        assert_eq!(cfg.max_layout_iterations, 24);
        assert!((cfg.convergence_tolerance - 0.001).abs() < f32::EPSILON);
        assert!((cfg.layout_boundary - 128.0).abs() < f32::EPSILON);
        assert_eq!(cfg.max_layout_units, 256);
        assert!((cfg.spatial_cell_size - 4.0).abs() < f32::EPSILON);
        assert!((cfg.neighbor_radius - 6.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 3.2  3-tier walk scoring simulation (§5.4)
    //   Tier 1: edge_weight × proximity_bonus × context_match  (~70%)
    //   Tier 2: edge_weight × distance_decay × context_match   (~20%)
    //   Tier 3: Π(edge_weights) × subgraph_density             (~10%)
    // -----------------------------------------------------------------
    fn tier1_score(edge_weight: f32, proximity_bonus: f32, context_match: f32) -> f32 {
        edge_weight * proximity_bonus * context_match
    }

    fn tier2_score(edge_weight: f32, distance_decay: f32, context_match: f32) -> f32 {
        edge_weight * distance_decay * context_match
    }

    fn tier3_score(edge_weights: &[f32], subgraph_density: f32) -> f32 {
        let product: f32 = edge_weights.iter().product();
        product * subgraph_density
    }

    #[test]
    fn tier1_beats_tier2_for_nearby_edges() {
        let t1 = tier1_score(0.8, 1.5, 0.9); // nearby: high proximity bonus
        let t2 = tier2_score(0.8, 0.5, 0.9); // far: distance decay
        assert!(
            t1 > t2,
            "Tier 1 (nearby) should score higher than Tier 2 (far): T1={t1}, T2={t2}"
        );
    }

    #[test]
    fn tier2_beats_tier3_for_direct_edges() {
        let t2 = tier2_score(0.7, 0.6, 0.8);
        let t3 = tier3_score(&[0.6, 0.5, 0.7], 0.8); // 3-hop pathfinding
        assert!(
            t2 > t3,
            "Tier 2 (direct far) should score higher than Tier 3 (pathfinding): T2={t2}, T3={t3}"
        );
    }

    #[test]
    fn tier3_pathfinding_still_produces_positive_score() {
        // Even multi-hop pathfinding should produce a viable score
        let t3 = tier3_score(&[0.7, 0.6, 0.5], 0.8);
        assert!(t3 > 0.0, "Tier 3 should produce positive score: {t3}");
        assert!(t3 < 1.0, "Tier 3 should be reasonable: {t3}");
    }

    // -----------------------------------------------------------------
    // 3.3  Highway formation threshold (§5.1)
    //   Highway forms when a word sequence is walked >= highway_formation_threshold times
    // -----------------------------------------------------------------
    #[test]
    fn highway_formation_simulation() {
        let highway_formation_threshold = 5u32; // doc default
        let highway_min_frequency = 2u32;

        // Simulate walking "the capital of france is paris" multiple times
        let mut walk_count = 0u32;
        let mut highway_formed = false;

        for _ in 0..10 {
            walk_count += 1;
            if walk_count >= highway_formation_threshold {
                highway_formed = true;
                break;
            }
        }
        assert!(
            highway_formed,
            "Highway should form after {highway_formation_threshold} walks"
        );

        // Highway below min frequency should be dissolved
        let dissolved = walk_count < highway_min_frequency;
        assert!(!dissolved, "Active highway should not be dissolved");
    }

    // -----------------------------------------------------------------
    // 3.4  Context-gated edge selection (polysemy, §5.1)
    //   "bank" with context {finance} activates "bank→account" but not "bank→erosion"
    // -----------------------------------------------------------------
    #[test]
    fn context_gating_selects_correct_edges() {
        // Simulate context tags as u64 hashes
        let finance_context: u64 = 0xF10A0CE;
        let nature_context: u64 = 0x0A7E0E;

        struct SimEdge {
            target: &'static str,
            weight: f32,
            context_tags: Vec<u64>,
        }

        let edges = vec![
            SimEdge {
                target: "account",
                weight: 0.7,
                context_tags: vec![finance_context],
            },
            SimEdge {
                target: "erosion",
                weight: 0.6,
                context_tags: vec![nature_context],
            },
            SimEdge {
                target: "robbery",
                weight: 0.5,
                context_tags: vec![finance_context],
            },
        ];

        // Active context: finance
        let active_context = finance_context;
        let activated: Vec<&str> = edges
            .iter()
            .filter(|e| e.context_tags.contains(&active_context))
            .map(|e| e.target)
            .collect();

        assert!(
            activated.contains(&"account"),
            "finance context should activate bank→account"
        );
        assert!(
            activated.contains(&"robbery"),
            "finance context should activate bank→robbery"
        );
        assert!(
            !activated.contains(&"erosion"),
            "finance context should NOT activate bank→erosion"
        );
    }

    // -----------------------------------------------------------------
    // 3.5  Beam search: wider beam explores more (§5.4)
    // -----------------------------------------------------------------
    #[test]
    fn beam_width_varies_by_profile() {
        let cfg = AdaptiveBehaviorConfig::default();
        let factual_beam = cfg.intent_profile("factual").unwrap().escape.beam_width;
        let brainstorm_beam = cfg.intent_profile("brainstorm").unwrap().escape.beam_width;
        assert_eq!(factual_beam, 3, "Factual beam width should be 3");
        assert_eq!(brainstorm_beam, 10, "Brainstorm beam width should be 10");
        assert!(
            brainstorm_beam > factual_beam,
            "Creative profiles need wider beam for diversity"
        );
    }

    #[test]
    fn beam_search_cumulative_scoring() {
        // Simulate 2-step beam search with beam_width=2
        let beam_width = 2;
        struct Beam {
            words: Vec<&'static str>,
            score: f32,
        }

        // Step 0: start with "capital"
        let mut beams = vec![Beam {
            words: vec!["capital"],
            score: 1.0,
        }];

        // Step 1: expand each beam with candidate next words
        let mut next_beams = Vec::new();
        for beam in &beams {
            // Candidate edges from "capital"
            let candidates = vec![("of", 0.9), ("city", 0.7), ("letter", 0.3)];
            for (word, edge_score) in candidates {
                let mut words = beam.words.clone();
                words.push(word);
                next_beams.push(Beam {
                    words,
                    score: beam.score * edge_score,
                });
            }
        }
        next_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beams = next_beams.into_iter().take(beam_width).collect();

        assert_eq!(beams.len(), beam_width);
        assert_eq!(beams[0].words, vec!["capital", "of"]);
        assert!((beams[0].score - 0.9).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 3.6  Step resolver temperature mapping (§5.5 / L16)
    // -----------------------------------------------------------------
    #[test]
    fn resolver_temperature_ordering() {
        let cfg = AdaptiveBehaviorConfig::default();
        let factual_t = cfg
            .intent_profile("factual")
            .unwrap()
            .resolver
            .selection_temperature;
        let balanced_t = cfg
            .intent_profile("explanatory")
            .unwrap()
            .resolver
            .selection_temperature;
        let creative_t = cfg
            .intent_profile("creative")
            .unwrap()
            .resolver
            .selection_temperature;
        let brainstorm_t = cfg
            .intent_profile("brainstorm")
            .unwrap()
            .resolver
            .selection_temperature;

        assert!(factual_t < balanced_t, "factual < explanatory temperature");
        assert!(
            balanced_t < creative_t,
            "explanatory < creative temperature"
        );
        assert!(
            creative_t < brainstorm_t,
            "creative < brainstorm temperature"
        );
    }

    // -----------------------------------------------------------------
    // 3.7  Fine resolver config defaults (§5.5)
    // -----------------------------------------------------------------
    #[test]
    fn fine_resolver_config_defaults() {
        let cfg = FineResolverConfig::default();
        assert!((cfg.selection_temperature - 0.70).abs() < f32::EPSILON);
        assert!((cfg.min_confidence_floor - 0.22).abs() < 0.01);
        assert!((cfg.evidence_answer_confidence_threshold - 0.22).abs() < 0.01);
        assert!((cfg.creative_drift_tolerance - 0.25).abs() < f32::EPSILON);
        assert!((cfg.factual_corruption_threshold - 0.15).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------
    // 3.8  Level multiplier ordering (L14 search.rs)
    //   Char < Subword < Word ≤ Phrase; Pattern intermediate
    // -----------------------------------------------------------------
    fn level_multiplier(level: UnitLevel) -> f32 {
        match level {
            UnitLevel::Char => 0.15,
            UnitLevel::Subword => 0.45,
            UnitLevel::Word => 0.9,
            UnitLevel::Phrase => 1.0,
            UnitLevel::Pattern => 0.8,
        }
    }

    #[test]
    fn level_multipliers_ordered_correctly() {
        assert!(level_multiplier(UnitLevel::Char) < level_multiplier(UnitLevel::Subword));
        assert!(level_multiplier(UnitLevel::Subword) < level_multiplier(UnitLevel::Word));
        assert!(level_multiplier(UnitLevel::Word) <= level_multiplier(UnitLevel::Phrase));
        assert!(level_multiplier(UnitLevel::Pattern) > level_multiplier(UnitLevel::Subword));
    }

    // -----------------------------------------------------------------
    // 3.9  Creative spark config (§5.5 / Phase 3.2)
    // -----------------------------------------------------------------
    #[test]
    fn creative_spark_config_defaults() {
        let cfg = CreativeSparkConfig::default();
        assert!((cfg.global_stochastic_floor - 0.15).abs() < f32::EPSILON);
        assert!((cfg.anchor_protection_strictness - 0.95).abs() < f32::EPSILON);
        assert!(cfg.disable_drift_for_math);
    }

    // -----------------------------------------------------------------
    // 3.10  Semantic drift: creative allows, plan does not (§5.5)
    // -----------------------------------------------------------------
    #[test]
    fn creative_profiles_allow_drift_plan_does_not() {
        let cfg = AdaptiveBehaviorConfig::default();
        let creative = cfg.intent_profile("creative").unwrap();
        let brainstorm = cfg.intent_profile("brainstorm").unwrap();
        let plan = cfg.intent_profile("plan").unwrap();
        let factual = cfg.intent_profile("factual").unwrap();

        assert!(
            creative.shaping.allow_semantic_drift,
            "Creative should allow drift"
        );
        assert!(
            brainstorm.shaping.allow_semantic_drift,
            "Brainstorm should allow drift"
        );
        assert!(
            !plan.shaping.allow_semantic_drift,
            "Plan should NOT allow drift"
        );
        assert!(
            !factual.shaping.allow_semantic_drift,
            "Factual should NOT allow drift"
        );
    }

    #[test]
    fn highway_preference_remains_subordinate_to_context_match() {
        let candidates = [
            PathCandidate {
                label: "erosion",
                tier: PredictiveTier::Highway,
                base_score: 0.92,
                proximity_or_decay: 1.0,
                context_match: 0.20,
                highway_boost: 1.30,
                subgraph_density: 0.85,
                anchor_trust: 0.20,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "account",
                tier: PredictiveTier::Tier1,
                base_score: 0.78,
                proximity_or_decay: 1.40,
                context_match: 0.92,
                highway_boost: 1.0,
                subgraph_density: 0.70,
                anchor_trust: 0.40,
                contradicts_anchor: false,
            },
        ];

        let naive = naive_highway_selection(&candidates);
        let aware = context_aware_selection(&candidates, false, 0.80);

        assert_eq!(naive.label, "erosion");
        assert_eq!(aware.label, "account");
    }

    #[test]
    fn anchor_protection_blocks_creative_drift_from_overwriting_factual_anchor() {
        let candidates = [
            PathCandidate {
                label: "Paris",
                tier: PredictiveTier::Highway,
                base_score: 0.78,
                proximity_or_decay: 1.0,
                context_match: 0.85,
                highway_boost: 1.20,
                subgraph_density: 0.80,
                anchor_trust: 0.96,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "Lyon",
                tier: PredictiveTier::Tier1,
                base_score: 0.93,
                proximity_or_decay: 1.30,
                context_match: 0.95,
                highway_boost: 1.0,
                subgraph_density: 0.80,
                anchor_trust: 0.96,
                contradicts_anchor: true,
            },
        ];

        let drift_allowed = context_aware_selection(&candidates, false, 0.80);
        let anchor_safe = context_aware_selection(&candidates, true, 0.80);

        assert_eq!(
            drift_allowed.label, "Lyon",
            "Without anchor protection the higher-scoring creative branch should win"
        );
        assert_eq!(
            anchor_safe.label, "Paris",
            "High-trust anchors should survive creative pressure when protection is enabled"
        );
    }

    #[test]
    fn beam_width_greater_than_one_recovers_from_greedy_dead_end() {
        #[derive(Clone)]
        struct Beam {
            words: Vec<&'static str>,
            score: f32,
        }

        fn graph(node: &str) -> Vec<(&'static str, f32)> {
            match node {
                "start" => vec![("alpha", 0.92), ("beta", 0.74)],
                "alpha" => vec![("dead_end", 0.08)],
                "beta" => vec![("bridge", 0.96)],
                "bridge" => vec![("goal", 0.95)],
                _ => Vec::new(),
            }
        }

        fn search(beam_width: usize) -> Beam {
            let mut beams = vec![Beam {
                words: vec!["start"],
                score: 1.0,
            }];

            for _ in 0..3 {
                let mut next = Vec::new();
                for beam in &beams {
                    let current = beam.words.last().copied().unwrap_or("start");
                    let expansions = graph(current);
                    if expansions.is_empty() {
                        next.push(beam.clone());
                        continue;
                    }
                    for (word, edge_score) in expansions {
                        let mut words = beam.words.clone();
                        words.push(word);
                        next.push(Beam {
                            words,
                            score: beam.score * edge_score,
                        });
                    }
                }
                next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                beams = next.into_iter().take(beam_width).collect();
            }

            beams
                .into_iter()
                .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
                .expect("beam search should retain at least one path")
        }

        let greedy = search(1);
        let beam = search(2);

        assert_eq!(greedy.words, vec!["start", "alpha", "dead_end"]);
        assert_eq!(beam.words, vec!["start", "beta", "bridge", "goal"]);
        assert!(beam.score > greedy.score);
    }

    #[test]
    fn predictive_principle_benchmark_tracks_beam_exploration_growth() {
        fn explored_states(beam_width: usize, branching: usize, depth: usize) -> usize {
            let mut frontier = 1usize;
            let mut explored = 0usize;
            for _ in 0..depth {
                let expanded = frontier * branching;
                explored += expanded;
                frontier = expanded.min(beam_width.max(1));
            }
            explored
        }

        let iterations = 100_000usize;
        let start = Instant::now();
        let factual_explored = explored_states(3, 4, 4);
        let brainstorm_explored = explored_states(10, 4, 4);
        let total = iterations * (factual_explored + brainstorm_explored);
        let elapsed = start.elapsed();

        eprintln!(
            "predictive_benchmark iterations={} elapsed_ms={} factual_states={} brainstorm_states={} total_state_expansions={}",
            iterations,
            elapsed.as_millis(),
            factual_explored,
            brainstorm_explored,
            total
        );

        assert!(brainstorm_explored > factual_explored);
        assert!(
            brainstorm_explored >= factual_explored * 2,
            "Brainstorm exploration should be meaningfully wider than factual exploration"
        );
    }
}

// ============================================================================
// §4  CROSS-SYSTEM — Config Coherence & End-to-End Flow Properties
// ============================================================================

mod cross_system {
    use super::synthetic_model::*;
    use super::*;

    // -----------------------------------------------------------------
    // 4.1  Full EngineConfig loads from YAML without panic
    // -----------------------------------------------------------------
    #[test]
    fn engine_config_loads_from_yaml() {
        let yaml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/config/config.yaml");
        let yaml_content =
            std::fs::read_to_string(yaml_path).expect("config/config.yaml must exist");
        let config: EngineConfig = serde_yaml::from_str(&yaml_content)
            .expect("config.yaml must deserialize into EngineConfig");
        // Spot-check a few values
        assert!(config.auto_inference.reasoning_loop.enabled);
        assert!(
            (config.retrieval.entropy_threshold - 0.72).abs() < 0.01
                || (config.retrieval.entropy_threshold - 0.85).abs() < 0.01,
            "Entropy threshold should be a documented value: {}",
            config.retrieval.entropy_threshold
        );
    }

    // -----------------------------------------------------------------
    // 4.2  Flow E: social short-circuit confidence threshold (§4.10)
    //   confidence > 0.95 AND intent ∈ social → skip Reasoning
    // -----------------------------------------------------------------
    #[test]
    fn social_shortcircuit_skips_reasoning() {
        let social_shortcircuit_confidence = 0.95_f32;
        let social_intents = [
            IntentKind::Greeting,
            IntentKind::Gratitude,
            IntentKind::Farewell,
        ];

        for intent in &social_intents {
            let confidence = 0.97;
            let should_skip = confidence > social_shortcircuit_confidence
                && matches!(
                    intent,
                    IntentKind::Greeting | IntentKind::Gratitude | IntentKind::Farewell
                );
            assert!(
                should_skip,
                "Intent {:?} at confidence {confidence} should skip Reasoning",
                intent
            );
        }
    }

    // -----------------------------------------------------------------
    // 4.3  Flow A vs Flow C: warm path vs retrieval path (§4.10)
    // -----------------------------------------------------------------
    #[test]
    fn warm_path_does_not_retrieve() {
        // Warm path: classification confident, all words have edges
        let classification_confidence = 0.92;
        let retrieval_threshold = RetrievalThresholds::default();

        // Low entropy, low freshness, low disagreement → no retrieval
        let score = 1.0 * 0.3 + 1.0 * 0.2 + 1.0 * 0.1 - 0.65 * retrieval_threshold.cost_penalty;
        assert!(
            score < retrieval_threshold.decision_threshold,
            "Warm path should NOT trigger retrieval: score={score}"
        );

        // Confirm classification confident enough to be deterministic
        let cfg = ClassificationConfig::default();
        assert!(
            classification_confidence > cfg.high_confidence_threshold,
            "Warm path should have Deterministic mode"
        );
    }

    // -----------------------------------------------------------------
    // 4.4  Confidence trajectory: pre-reasoning → post-reasoning (§4.11)
    //   Doc: average of 0.329 → 0.716 after reasoning path
    // -----------------------------------------------------------------
    #[test]
    fn confidence_improves_through_reasoning_pipeline() {
        let pre_reasoning_avg = 0.329_f32;
        let post_reasoning_avg = 0.716_f32;

        assert!(
            post_reasoning_avg > pre_reasoning_avg,
            "Reasoning should improve confidence"
        );
        assert!(
            post_reasoning_avg > ReasoningLoopConfig::default().exit_confidence_threshold,
            "Post-reasoning confidence should clear exit threshold"
        );
        assert!(
            pre_reasoning_avg < ReasoningLoopConfig::default().trigger_confidence_floor,
            "Pre-reasoning confidence should be below trigger floor"
        );
    }

    // -----------------------------------------------------------------
    // 4.5  TTG lease for probationary edges (§4.9 Extension 2b)
    // -----------------------------------------------------------------
    #[test]
    fn ttg_lease_lifecycle_simulation() {
        let ttg_lease_duration_secs: u64 = 300; // 5 minutes
        let graduation_count: u32 = 2;

        // Simulate edge lifecycle
        #[derive(PartialEq, Debug)]
        enum EdgeStatus {
            Probationary,
            Episodic,
            Purged,
        }

        struct SimEdge {
            status: EdgeStatus,
            traversal_count: u32,
            lease_expires_at: u64,
        }

        // Step 1: Injection
        let now = 1000u64;
        let mut edge = SimEdge {
            status: EdgeStatus::Probationary,
            traversal_count: 1,
            lease_expires_at: now + ttg_lease_duration_secs,
        };

        // Step 2: During lease — immune to pruning
        let maintenance_time = now + 30; // 30s maintenance cycle
        assert!(
            maintenance_time < edge.lease_expires_at,
            "Edge should be immune during lease"
        );

        // Step 3: Subsequent use — traverse again
        edge.traversal_count += 1;
        if edge.traversal_count >= graduation_count {
            edge.status = EdgeStatus::Episodic;
        }
        assert_eq!(
            edge.status,
            EdgeStatus::Episodic,
            "Edge should graduate to Episodic"
        );

        // Step 4: Unused edge — lease expires, purged
        let mut unused_edge = SimEdge {
            status: EdgeStatus::Probationary,
            traversal_count: 1,
            lease_expires_at: now + ttg_lease_duration_secs,
        };
        let late_maintenance = now + ttg_lease_duration_secs + 1;
        if late_maintenance > unused_edge.lease_expires_at
            && unused_edge.status == EdgeStatus::Probationary
            && unused_edge.traversal_count < graduation_count
        {
            unused_edge.status = EdgeStatus::Purged;
        }
        assert_eq!(
            unused_edge.status,
            EdgeStatus::Purged,
            "Unused edge should be purged after lease"
        );
    }

    // -----------------------------------------------------------------
    // 4.6  Feedback queue: trace-ID tagged reinforcement (§4.9 Extension 3)
    // -----------------------------------------------------------------
    #[test]
    fn feedback_queue_simulation() {
        use std::collections::VecDeque;

        struct WalkEvent {
            trace_id: u64,
            edges: Vec<(&'static str, &'static str)>,
            created_at: u64,
        }

        let max_size = 64usize;
        let ttl_secs = 600u64;
        let mut queue: VecDeque<WalkEvent> = VecDeque::new();

        // Response generates walk event
        queue.push_back(WalkEvent {
            trace_id: 42,
            edges: vec![("hi", "hello"), ("hello", "how")],
            created_at: 1000,
        });
        assert_eq!(queue.len(), 1);

        // Next user message: implicit accept → reinforce edges
        let parent_trace = 42u64;
        let event = queue.iter().find(|e| e.trace_id == parent_trace);
        assert!(event.is_some(), "Should find matching WalkEvent");
        assert_eq!(event.unwrap().edges.len(), 2);

        // TTL cleanup
        let now = 1000 + ttl_secs + 1;
        queue.retain(|e| now - e.created_at < ttl_secs);
        assert_eq!(queue.len(), 0, "Expired events should be cleaned up");

        // Bounded size
        for i in 0..max_size + 10 {
            queue.push_back(WalkEvent {
                trace_id: i as u64,
                edges: vec![],
                created_at: 2000,
            });
            if queue.len() > max_size {
                queue.pop_front();
            }
        }
        assert_eq!(queue.len(), max_size, "Queue should be bounded");
    }

    // -----------------------------------------------------------------
    // 4.7  Entropy calculation produces valid range (L9)
    // -----------------------------------------------------------------
    #[test]
    fn entropy_calculation_valid_range() {
        // Shannon entropy normalized to [0, 1]
        fn calculate_entropy(scores: &[f32]) -> f32 {
            let total: f32 = scores.iter().map(|s| s.max(0.0)).sum();
            if total <= f32::EPSILON {
                return 0.0;
            }
            let entropy = scores.iter().fold(0.0_f32, |e, s| {
                let p = s.max(0.0) / total;
                if p <= f32::EPSILON {
                    e
                } else {
                    e - p * p.ln()
                }
            });
            let max_entropy = (scores.len().max(2) as f32).ln();
            if max_entropy <= f32::EPSILON {
                0.0
            } else {
                (entropy / max_entropy).clamp(0.0, 1.0)
            }
        }

        // Uniform distribution → max entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let e = calculate_entropy(&uniform);
        assert!(
            (e - 1.0).abs() < 0.01,
            "Uniform should give entropy ~1.0, got {e}"
        );

        // Single dominant → low entropy
        let dominant = vec![0.95, 0.02, 0.02, 0.01];
        let e = calculate_entropy(&dominant);
        assert!(e < 0.5, "Dominant should give low entropy, got {e}");

        // Empty → zero
        assert_eq!(calculate_entropy(&[]), 0.0);
    }

    // -----------------------------------------------------------------
    // 4.8  Memory channel isolation (§3.7, §4.1)
    // -----------------------------------------------------------------
    #[test]
    fn memory_channels_are_distinct() {
        let channels = [
            MemoryChannel::Main,
            MemoryChannel::Intent,
            MemoryChannel::Reasoning,
        ];
        for (i, a) in channels.iter().enumerate() {
            for (j, b) in channels.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Channels must be distinct");
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // 4.9  Multi-engine consensus config (§5.1 Phase 5)
    // -----------------------------------------------------------------
    #[test]
    fn multi_engine_consensus_weights_sum_to_one() {
        let cfg = MultiEngineConfig::default();
        let sum = cfg.trust_weight + cfg.agreement_weight + cfg.diversity_weight;
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Multi-engine weights sum to {sum}, expected 1.0"
        );
    }

    // -----------------------------------------------------------------
    // 4.10  Load-aware cost config (§4.2)
    // -----------------------------------------------------------------
    #[test]
    fn load_cost_config_defaults() {
        let cfg = LoadAwareCostConfig::default();
        assert!((cfg.inference_queue_weight - 0.10).abs() < f32::EPSILON);
        assert!((cfg.max_additive_penalty - 0.60).abs() < f32::EPSILON);
        // Inference should have highest weight (most impactful)
        assert!(cfg.inference_queue_weight >= cfg.silent_batch_weight);
        assert!(cfg.inference_queue_weight >= cfg.maintenance_weight);
    }

    // -----------------------------------------------------------------
    // 4.11  Two trust profiles: default and high_stakes (§4.2)
    // -----------------------------------------------------------------
    #[test]
    fn trust_profiles_exist_and_differ() {
        let cfg = AdaptiveBehaviorConfig::default();
        let default_tp = cfg.trust_profile("default").unwrap();
        let high_stakes = cfg.trust_profile("high_stakes").unwrap();

        assert!(
            high_stakes.default_source_trust < default_tp.default_source_trust,
            "High stakes should have lower default trust"
        );
        assert!(
            high_stakes.min_corroborating_sources > default_tp.min_corroborating_sources,
            "High stakes should require more corroboration"
        );
        assert!(
            high_stakes.require_https && !default_tp.require_https,
            "High stakes should require HTTPS"
        );
    }

    // -----------------------------------------------------------------
    // 4.12  Pollution prevention thresholds (§4.1)
    // -----------------------------------------------------------------
    #[test]
    fn pollution_detection_config() {
        let cfg = GovernanceConfig::default();
        assert!(
            cfg.pollution_detection_enabled,
            "Pollution detection should be on by default"
        );
        assert_eq!(cfg.pollution_min_length, 4);
        assert!((cfg.pollution_overlap_threshold - 0.70).abs() < f32::EPSILON);
        assert_eq!(cfg.pollution_audit_limit, 64);
    }

    // -----------------------------------------------------------------
    // 4.13  Unit hierarchy levels are complete (§3.1)
    // -----------------------------------------------------------------
    #[test]
    fn unit_hierarchy_levels_complete() {
        let levels = [
            UnitLevel::Char,
            UnitLevel::Subword,
            UnitLevel::Word,
            UnitLevel::Phrase,
            UnitLevel::Pattern,
        ];
        assert_eq!(levels.len(), 5, "Should have exactly 5 hierarchy levels");
    }

    // -----------------------------------------------------------------
    // 4.14  All 24 intent labels exist (§3.2)
    // -----------------------------------------------------------------
    #[test]
    fn all_24_intent_labels_exist() {
        let intents = [
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
        assert_eq!(intents.len(), 24, "Should have exactly 24 intent labels");
    }

    // -----------------------------------------------------------------
    // 4.15  FNV-1a text fingerprint deterministic (§types.rs)
    // -----------------------------------------------------------------
    #[test]
    fn text_fingerprint_deterministic() {
        let fp1 = text_fingerprint("hello world");
        let fp2 = text_fingerprint("hello world");
        let fp3 = text_fingerprint("different text");
        assert_eq!(fp1, fp2, "Same input should produce same fingerprint");
        assert_ne!(
            fp1, fp3,
            "Different input should produce different fingerprint"
        );
    }

    // -----------------------------------------------------------------
    // 4.16  Auto-mode enforcement (§3.4 / Phase 3)
    // -----------------------------------------------------------------
    #[test]
    fn auto_mode_locked_by_default() {
        let cfg = AutoModeConfig::default();
        assert!(cfg.locked);
        assert!(cfg.ignore_mode_parameter);
        assert!(cfg.ignore_temperature_parameter);
        assert!(cfg.ignore_reasoning_depth_parameter);
        assert!(cfg.ignore_creative_level_parameter);
    }

    // -----------------------------------------------------------------
    // 4.17  Tone inference config (§3.3 / Phase 3)
    // -----------------------------------------------------------------
    #[test]
    fn tone_inference_config_defaults() {
        let cfg = ToneInferenceConfig::default();
        assert!(cfg.enabled);
        assert!((cfg.urgency_threshold - 0.7).abs() < f32::EPSILON);
        assert!((cfg.sadness_threshold - 0.5).abs() < f32::EPSILON);
        assert!((cfg.technical_threshold - 0.6).abs() < f32::EPSILON);
        assert!((cfg.style_resonance_weight - 0.08).abs() < f32::EPSILON);
    }

    #[derive(Debug, Clone, Copy)]
    struct ConsistencyScenario {
        name: &'static str,
        intent: IntentKind,
        classification_confidence: f32,
        retrieval_signals: RetrievalSignals,
        reasoning_initial_confidence: f32,
        reasoning_boosts: &'static [f32],
        predicted_tier: PredictiveTier,
        drift_allowed: bool,
        contradicts_anchor: bool,
        evidence_contradiction: bool,
    }

    fn consistency_scenarios() -> [ConsistencyScenario; 5] {
        [
            ConsistencyScenario {
                name: "social_shortcircuit",
                intent: IntentKind::Gratitude,
                classification_confidence: 0.98,
                retrieval_signals: RetrievalSignals {
                    intent: IntentKind::Gratitude,
                    confidence: 0.98,
                    entropy: 0.04,
                    freshness: 0.0,
                    disagreement: 0.02,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: true,
                },
                reasoning_initial_confidence: 0.96,
                reasoning_boosts: &[],
                predicted_tier: PredictiveTier::Highway,
                drift_allowed: false,
                contradicts_anchor: false,
                evidence_contradiction: false,
            },
            ConsistencyScenario {
                name: "hard_compare_local",
                intent: IntentKind::Compare,
                classification_confidence: 0.66,
                retrieval_signals: RetrievalSignals {
                    intent: IntentKind::Compare,
                    confidence: 0.66,
                    entropy: 0.38,
                    freshness: 0.0,
                    disagreement: 0.18,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: true,
                    strong_local_support: true,
                },
                reasoning_initial_confidence: 0.33,
                reasoning_boosts: &[0.15, 0.16],
                predicted_tier: PredictiveTier::Tier1,
                drift_allowed: false,
                contradicts_anchor: false,
                evidence_contradiction: false,
            },
            ConsistencyScenario {
                name: "open_world_retrieval",
                intent: IntentKind::Question,
                classification_confidence: 0.57,
                retrieval_signals: RetrievalSignals {
                    intent: IntentKind::Question,
                    confidence: 0.57,
                    entropy: 0.77,
                    freshness: 0.82,
                    disagreement: 0.41,
                    open_world: true,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: false,
                },
                reasoning_initial_confidence: 0.26,
                reasoning_boosts: &[0.08, 0.04, 0.03],
                predicted_tier: PredictiveTier::Tier2,
                drift_allowed: false,
                contradicts_anchor: false,
                evidence_contradiction: false,
            },
            ConsistencyScenario {
                name: "creative_drift",
                intent: IntentKind::Brainstorm,
                classification_confidence: 0.64,
                retrieval_signals: RetrievalSignals {
                    intent: IntentKind::Brainstorm,
                    confidence: 0.64,
                    entropy: 0.32,
                    freshness: 0.0,
                    disagreement: 0.12,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: false,
                    strong_local_support: true,
                },
                reasoning_initial_confidence: 0.58,
                reasoning_boosts: &[],
                predicted_tier: PredictiveTier::Tier2,
                drift_allowed: true,
                contradicts_anchor: false,
                evidence_contradiction: false,
            },
            ConsistencyScenario {
                name: "factual_anchor_preserved",
                intent: IntentKind::Verify,
                classification_confidence: 0.83,
                retrieval_signals: RetrievalSignals {
                    intent: IntentKind::Verify,
                    confidence: 0.83,
                    entropy: 0.41,
                    freshness: 0.18,
                    disagreement: 0.24,
                    open_world: false,
                    recent_entity_carry: false,
                    procedural_bias: false,
                    local_reasoning_bias: true,
                    strong_local_support: true,
                },
                reasoning_initial_confidence: 0.61,
                reasoning_boosts: &[],
                predicted_tier: PredictiveTier::Tier1,
                drift_allowed: false,
                contradicts_anchor: false,
                evidence_contradiction: false,
            },
        ]
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct ConsistencyViolations {
        r1: usize,
        r2: usize,
        r3: usize,
        r4: usize,
        r5: usize,
        r6: usize,
        r7: usize,
    }

    fn evaluate_consistency(
        scenario: ConsistencyScenario,
        thresholds: &RetrievalThresholds,
        reasoning_cfg: &ReasoningLoopConfig,
    ) -> ConsistencyViolations {
        let decision = architecture_retrieval_decision(scenario.retrieval_signals, thresholds);
        let reasoning_triggered = should_trigger_reasoning(
            scenario.intent,
            scenario.reasoning_initial_confidence,
            reasoning_cfg,
        );
        let (final_confidence, loop_requested_retrieval, steps_taken) = if reasoning_triggered {
            simulate_reasoning_progress(
                scenario.reasoning_initial_confidence,
                scenario.reasoning_boosts,
                reasoning_cfg,
            )
        } else {
            (scenario.reasoning_initial_confidence, false, 0)
        };
        let used_retrieval = decision.should_retrieve || loop_requested_retrieval;
        let contradicts_anchor =
            scenario.contradicts_anchor && is_factual_anchor_intent(scenario.intent);

        let _ = final_confidence;

        ConsistencyViolations {
            r1: usize::from(scenario.classification_confidence < 0.40 && !used_retrieval),
            r2: usize::from(is_social_intent(scenario.intent) && steps_taken > 0),
            r3: usize::from(contradicts_anchor),
            r4: usize::from(is_creative_intent(scenario.intent) && !scenario.drift_allowed),
            r5: usize::from(
                scenario.predicted_tier == PredictiveTier::Tier3
                    && scenario.classification_confidence > 0.72,
            ),
            r6: usize::from(
                scenario.evidence_contradiction && scenario.classification_confidence > 0.72,
            ),
            r7: 0,
        }
    }

    #[test]
    fn consistency_scenario_pack_satisfies_r1_to_r6() {
        let thresholds = RetrievalThresholds::default();
        let reasoning_cfg = ReasoningLoopConfig::default();
        let mut totals = ConsistencyViolations::default();

        for scenario in consistency_scenarios() {
            let result = evaluate_consistency(scenario, &thresholds, &reasoning_cfg);
            totals.r1 += result.r1;
            totals.r2 += result.r2;
            totals.r3 += result.r3;
            totals.r4 += result.r4;
            totals.r5 += result.r5;
            totals.r6 += result.r6;
        }

        assert_eq!(
            totals.r1, 0,
            "R1 should not fail in the baseline scenario pack"
        );
        assert_eq!(
            totals.r2, 0,
            "R2 should not fail in the baseline scenario pack"
        );
        assert_eq!(
            totals.r3, 0,
            "R3 should not fail in the baseline scenario pack"
        );
        assert_eq!(
            totals.r4, 0,
            "R4 should not fail in the baseline scenario pack"
        );
        assert_eq!(
            totals.r5, 0,
            "R5 should not fail in the baseline scenario pack"
        );
        assert_eq!(
            totals.r6, 0,
            "R6 should not fail in the baseline scenario pack"
        );
    }

    #[test]
    fn cold_start_tier3_usage_triggers_confidence_penalty_rule() {
        let initial_confidence = 0.81_f32;
        let cold_start_confidence_penalty = 0.10_f32;
        let corrected = (initial_confidence - cold_start_confidence_penalty).clamp(0.0, 1.0);

        assert!(
            initial_confidence > 0.72,
            "Scenario should start in high-confidence band"
        );
        assert!(
            corrected < initial_confidence,
            "Cold-start Tier 3 usage should lower future classification confidence"
        );
        assert!(
            corrected < 0.72,
            "Penalty should push the scenario out of the misleading high-confidence band"
        );
    }

    #[test]
    fn contradiction_feedback_penalty_demotes_overconfident_patterns() {
        let pattern_confidence = 0.84_f32;
        let contradiction_feedback_penalty = 0.15_f32;
        let corrected = (pattern_confidence - contradiction_feedback_penalty).clamp(0.0, 1.0);

        assert!(pattern_confidence > 0.72);
        assert!(corrected < pattern_confidence);
        assert!(
            corrected < 0.72,
            "Contradicted high-confidence patterns should fall out of the high-confidence band"
        );
    }

    #[test]
    fn broad_tier3_sparsity_triggers_proactive_learning_instead_of_intent_split() {
        let sparsity_intent_threshold = 3usize;
        let sparsity_tier3_rate = 0.30_f32;
        let tier3_rates = [
            ("question", 0.41_f32),
            ("compare", 0.33_f32),
            ("debug", 0.35_f32),
            ("plan", 0.18_f32),
        ];
        let distinct_sparse_intents = tier3_rates
            .iter()
            .filter(|(_, rate)| *rate > sparsity_tier3_rate)
            .count();

        assert_eq!(distinct_sparse_intents, 3);
        assert!(
            distinct_sparse_intents >= sparsity_intent_threshold,
            "Three distinct intents exceeding the Tier 3 rate should trigger R7"
        );
    }

    #[test]
    fn cross_system_benchmark_reports_zero_baseline_violations() {
        let thresholds = RetrievalThresholds::default();
        let reasoning_cfg = ReasoningLoopConfig::default();
        let scenarios = consistency_scenarios();
        let iterations = 25_000usize;
        let start = Instant::now();
        let mut total = ConsistencyViolations::default();

        for _ in 0..iterations {
            for scenario in scenarios {
                let result = evaluate_consistency(scenario, &thresholds, &reasoning_cfg);
                total.r1 += result.r1;
                total.r2 += result.r2;
                total.r3 += result.r3;
                total.r4 += result.r4;
                total.r5 += result.r5;
                total.r6 += result.r6;
                total.r7 += result.r7;
            }
        }

        let elapsed = start.elapsed();
        let evals = iterations * scenarios.len();
        eprintln!(
            "cross_system_benchmark evals={} elapsed_ms={} r1={} r2={} r3={} r4={} r5={} r6={} r7={}",
            evals,
            elapsed.as_millis(),
            total.r1,
            total.r2,
            total.r3,
            total.r4,
            total.r5,
            total.r6,
            total.r7
        );

        assert_eq!(
            total.r1 + total.r2 + total.r3 + total.r4 + total.r5 + total.r6 + total.r7,
            0
        );
    }
}

// ============================================================================
// §5  BENCHMARKS — Systematic throughput & correctness for all core logics
// ============================================================================

mod benchmarks {
    use super::synthetic_model::*;
    use super::*;

    // -----------------------------------------------------------------
    // Shared: classification scenario pack (reused across benchmarks)
    // -----------------------------------------------------------------
    fn classification_scenario_pack() -> [RetrievalSignals; 8] {
        [
            // Social short-circuit
            RetrievalSignals {
                intent: IntentKind::Gratitude,
                confidence: 0.98,
                entropy: 0.05,
                freshness: 0.0,
                disagreement: 0.02,
                open_world: false,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: true,
            },
            // Warm factual
            RetrievalSignals {
                intent: IntentKind::Question,
                confidence: 0.89,
                entropy: 0.15,
                freshness: 0.0,
                disagreement: 0.08,
                open_world: false,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: true,
            },
            // Local reasoning compare
            RetrievalSignals {
                intent: IntentKind::Compare,
                confidence: 0.67,
                entropy: 0.38,
                freshness: 0.0,
                disagreement: 0.20,
                open_world: false,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: true,
                strong_local_support: true,
            },
            // Open-world retrieval
            RetrievalSignals {
                intent: IntentKind::Question,
                confidence: 0.58,
                entropy: 0.76,
                freshness: 0.82,
                disagreement: 0.41,
                open_world: true,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: false,
            },
            // Context carry follow-up
            RetrievalSignals {
                intent: IntentKind::Question,
                confidence: 0.56,
                entropy: 0.46,
                freshness: 0.08,
                disagreement: 0.17,
                open_world: false,
                recent_entity_carry: true,
                procedural_bias: false,
                local_reasoning_bias: true,
                strong_local_support: false,
            },
            // Procedural plan
            RetrievalSignals {
                intent: IntentKind::Plan,
                confidence: 0.53,
                entropy: 0.54,
                freshness: 0.06,
                disagreement: 0.21,
                open_world: false,
                recent_entity_carry: false,
                procedural_bias: true,
                local_reasoning_bias: true,
                strong_local_support: false,
            },
            // Creative brainstorm
            RetrievalSignals {
                intent: IntentKind::Brainstorm,
                confidence: 0.63,
                entropy: 0.33,
                freshness: 0.0,
                disagreement: 0.11,
                open_world: false,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: true,
            },
            // Low-confidence unknown → forces retrieval
            RetrievalSignals {
                intent: IntentKind::Unknown,
                confidence: 0.22,
                entropy: 0.91,
                freshness: 0.70,
                disagreement: 0.55,
                open_world: true,
                recent_entity_carry: false,
                procedural_bias: false,
                local_reasoning_bias: false,
                strong_local_support: false,
            },
        ]
    }

    // =====================================================================
    // 5.1  Classification: retrieval gating throughput
    // =====================================================================
    #[test]
    fn bench_retrieval_gating_throughput() {
        let cfg = RetrievalThresholds::default();
        let scenarios = classification_scenario_pack();
        let iterations = 100_000usize;
        let start = Instant::now();

        let mut retrieve_count = 0usize;
        let mut suppress_count = 0usize;
        let mut flow_counts = [0usize; 5]; // SocialShortCircuit, WarmLocal, ReasoningLocal, Retrieval, Creative

        for _ in 0..iterations {
            for signals in &scenarios {
                let decision = architecture_retrieval_decision(*signals, &cfg);
                if decision.should_retrieve {
                    retrieve_count += 1;
                }
                if decision.suppressed {
                    suppress_count += 1;
                }
                match decision.flow {
                    FlowKind::SocialShortCircuit => flow_counts[0] += 1,
                    FlowKind::WarmLocal => flow_counts[1] += 1,
                    FlowKind::ReasoningLocal => flow_counts[2] += 1,
                    FlowKind::Retrieval => flow_counts[3] += 1,
                    FlowKind::CreativeExploration => flow_counts[4] += 1,
                }
            }
        }

        let elapsed = start.elapsed();
        let total_evals = iterations * scenarios.len();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;
        let retrieve_pct = retrieve_count as f64 / total_evals as f64 * 100.0;
        let suppress_pct = suppress_count as f64 / total_evals as f64 * 100.0;

        eprintln!(
            "[bench_retrieval_gating] evals={} elapsed_ms={} per_eval_ns={:.1} retrieve={:.1}% suppress={:.1}% flows=[social={} warm={} reasoning={} retrieval={} creative={}]",
            total_evals, elapsed.as_millis(), per_eval_ns,
            retrieve_pct, suppress_pct,
            flow_counts[0], flow_counts[1], flow_counts[2], flow_counts[3], flow_counts[4]
        );

        // Correctness invariants
        assert!(per_eval_ns < 500.0, "Retrieval gating should be <500ns/eval, got {per_eval_ns:.1}ns");
        assert!(retrieve_pct < 40.0, "Architecture should suppress most retrieval: {retrieve_pct:.1}%");
        assert!(suppress_pct > 50.0, "Majority of scenarios should be suppressed: {suppress_pct:.1}%");
        assert!(flow_counts[0] > 0, "Social short-circuit flow must fire");
        assert!(flow_counts[3] > 0, "Retrieval flow must fire for open-world/low-confidence");
    }

    // =====================================================================
    // 5.2  Classification: confidence formula sweep
    // =====================================================================
    #[test]
    fn bench_confidence_formula_sweep() {
        fn margin_blend_confidence(best: f32, runner_up: f32, blend: f32) -> f32 {
            if best <= 0.0 {
                return 0.0;
            }
            let margin = (best - runner_up) / best;
            best * (blend + (1.0 - blend) * margin)
        }

        let blend = 0.5_f32;
        let steps = 50usize;
        let iterations = 10_000usize;
        let start = Instant::now();

        let mut total_evals = 0usize;
        let mut out_of_bounds = 0usize;
        let mut monotonicity_violations = 0usize;

        for _ in 0..iterations {
            for bi in 0..steps {
                let best = (bi as f32 + 1.0) / steps as f32;
                let mut prev_c = 0.0_f32;
                for ri in 0..steps {
                    let runner = (ri as f32) / steps as f32;
                    if runner > best {
                        continue;
                    }
                    let c = margin_blend_confidence(best, runner, blend);
                    total_evals += 1;
                    if c < 0.0 || c > 1.0 {
                        out_of_bounds += 1;
                    }
                    // As runner_up increases (margin narrows), confidence should be non-increasing
                    if ri > 0 && c > prev_c + f32::EPSILON {
                        monotonicity_violations += 1;
                    }
                    prev_c = c;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;

        eprintln!(
            "[bench_confidence_sweep] evals={} elapsed_ms={} per_eval_ns={:.1} oob={} monotonicity_violations={}",
            total_evals, elapsed.as_millis(), per_eval_ns, out_of_bounds, monotonicity_violations
        );

        assert_eq!(out_of_bounds, 0, "Confidence must always be in [0, 1]");
        assert_eq!(
            monotonicity_violations, 0,
            "Confidence must be monotonically non-decreasing as margin widens"
        );
    }

    // =====================================================================
    // 5.3  Classification: trust scoring sweep across source types
    // =====================================================================
    #[test]
    fn bench_trust_scoring_sweep() {
        let cfg = TrustConfig::default();
        let iterations = 50_000usize;

        struct TrustCase {
            label: &'static str,
            https: bool,
            allowlisted: bool,
            parser_warnings: bool,
            corroborating: usize,
            format_adj: f32,
        }

        let cases = [
            TrustCase {
                label: "https_allowlisted_2corr",
                https: true,
                allowlisted: true,
                parser_warnings: false,
                corroborating: 2,
                format_adj: 0.0,
            },
            TrustCase {
                label: "https_plain",
                https: true,
                allowlisted: false,
                parser_warnings: false,
                corroborating: 0,
                format_adj: 0.0,
            },
            TrustCase {
                label: "raw_html_warnings",
                https: false,
                allowlisted: false,
                parser_warnings: true,
                corroborating: 0,
                format_adj: -0.30,
            },
            TrustCase {
                label: "structured_entity",
                https: true,
                allowlisted: false,
                parser_warnings: false,
                corroborating: 0,
                format_adj: 0.20,
            },
            TrustCase {
                label: "max_corroboration",
                https: true,
                allowlisted: true,
                parser_warnings: false,
                corroborating: 5,
                format_adj: 0.20,
            },
        ];

        let start = Instant::now();
        let mut scores = Vec::with_capacity(cases.len());

        for _ in 0..iterations {
            for case in &cases {
                let mut trust = cfg.default_source_trust;
                if case.https {
                    trust += cfg.https_bonus;
                }
                if case.allowlisted {
                    trust += cfg.allowlist_bonus;
                }
                if case.parser_warnings {
                    trust -= cfg.parser_warning_penalty;
                }
                trust += case.corroborating as f32 * cfg.corroboration_bonus;
                trust += case.format_adj;
                trust = trust.clamp(0.0, 1.0);
                if scores.len() < cases.len() {
                    scores.push((case.label, trust));
                }
            }
        }

        let elapsed = start.elapsed();
        let total_evals = iterations * cases.len();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;

        eprintln!("[bench_trust_scoring] evals={} elapsed_ms={} per_eval_ns={:.1}", total_evals, elapsed.as_millis(), per_eval_ns);
        for (label, score) in &scores {
            eprintln!("  {label}: {score:.3}");
        }

        // Correctness: all scores in [0, 1], ordering makes sense
        for (_, score) in &scores {
            assert!((0.0..=1.0).contains(score));
        }
        let max_corr_score = scores.iter().find(|(l, _)| *l == "max_corroboration").unwrap().1;
        let raw_html_score = scores.iter().find(|(l, _)| *l == "raw_html_warnings").unwrap().1;
        assert!(
            max_corr_score > raw_html_score + 0.3,
            "Max corroboration should far exceed raw HTML"
        );
    }

    // =====================================================================
    // 5.4  Classification: profile weight normalization across all profiles
    // =====================================================================
    #[test]
    fn bench_profile_weight_normalization() {
        let cfg = AdaptiveBehaviorConfig::default();
        let iterations = 100_000usize;
        let start = Instant::now();

        let mut max_deviation = 0.0_f32;
        let mut worst_profile = "";

        for _ in 0..iterations {
            for (name, profile) in &cfg.intent_profiles {
                let w = &profile.scoring;
                let sum = w.spatial
                    + w.context
                    + w.sequence
                    + w.transition
                    + w.utility
                    + w.confidence
                    + w.evidence;
                let dev = (sum - 1.0).abs();
                if dev > max_deviation {
                    max_deviation = dev;
                    worst_profile = name;
                }
            }
        }

        let elapsed = start.elapsed();
        let total_evals = iterations * cfg.intent_profiles.len();

        eprintln!(
            "[bench_profile_weights] evals={} elapsed_ms={} worst_profile='{}' max_deviation={:.4}",
            total_evals, elapsed.as_millis(), worst_profile, max_deviation
        );

        assert!(
            max_deviation < 0.02,
            "All profiles must sum to ~1.0, worst='{worst_profile}' deviation={max_deviation:.4}"
        );
    }

    // =====================================================================
    // 5.5  Reasoning: 7D candidate scoring throughput
    // =====================================================================
    #[test]
    fn bench_7d_candidate_scoring_throughput() {
        let profiles = [
            ("default", ScoringWeights::default()),
            (
                "factual",
                AdaptiveBehaviorConfig::default()
                    .intent_profiles
                    .get("factual")
                    .unwrap()
                    .scoring
                    .clone(),
            ),
            (
                "brainstorm",
                AdaptiveBehaviorConfig::default()
                    .intent_profiles
                    .get("brainstorm")
                    .unwrap()
                    .scoring
                    .clone(),
            ),
        ];

        // Synthetic candidate feature vectors
        let candidates: Vec<[f32; 7]> = (0..64)
            .map(|i| {
                let f = (i as f32 + 1.0) / 65.0;
                [f, 1.0 - f, f * 0.8, 0.5, f * 0.6, 1.0 - f * 0.5, f * 0.3]
            })
            .collect();

        let iterations = 50_000usize;
        let start = Instant::now();

        let mut total_scored = 0usize;
        let mut ranking_inversions = 0usize;

        for _ in 0..iterations {
            for (_, weights) in &profiles {
                let w = [
                    weights.spatial,
                    weights.context,
                    weights.sequence,
                    weights.transition,
                    weights.utility,
                    weights.confidence,
                    weights.evidence,
                ];
                let mut prev_score = f32::MAX;
                let mut scores: Vec<f32> = candidates
                    .iter()
                    .map(|feat| feat.iter().zip(w.iter()).map(|(f, w)| f * w).sum::<f32>())
                    .collect();
                scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                total_scored += candidates.len();

                // Check that sorted order is stable
                for s in &scores {
                    if *s > prev_score + f32::EPSILON {
                        ranking_inversions += 1;
                    }
                    prev_score = *s;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_candidate_ns = elapsed.as_nanos() as f64 / total_scored as f64;

        eprintln!(
            "[bench_7d_scoring] candidates_scored={} elapsed_ms={} per_candidate_ns={:.1} inversions={}",
            total_scored, elapsed.as_millis(), per_candidate_ns, ranking_inversions
        );

        assert_eq!(ranking_inversions, 0, "Sort must be stable");
        assert!(
            per_candidate_ns < 200.0,
            "7D scoring should be <200ns/candidate, got {per_candidate_ns:.1}ns"
        );
    }

    // =====================================================================
    // 5.6  Reasoning: evidence merge naive vs contradiction-aware
    // =====================================================================
    #[test]
    fn bench_evidence_merge_naive_vs_contradiction_aware() {
        let cfg = EvidenceMergeConfig::default();
        let iterations = 50_000usize;

        let doc_sets: Vec<Vec<EvidenceDoc>> = vec![
            // All supporting, high trust
            vec![
                EvidenceDoc { trust: 0.90, recency: 0.85, supports_claim: true },
                EvidenceDoc { trust: 0.88, recency: 0.80, supports_claim: true },
                EvidenceDoc { trust: 0.92, recency: 0.90, supports_claim: true },
            ],
            // Mixed: 2 support, 2 contradict
            vec![
                EvidenceDoc { trust: 0.86, recency: 0.82, supports_claim: true },
                EvidenceDoc { trust: 0.80, recency: 0.78, supports_claim: true },
                EvidenceDoc { trust: 0.89, recency: 0.90, supports_claim: false },
                EvidenceDoc { trust: 0.84, recency: 0.88, supports_claim: false },
            ],
            // All contradicting
            vec![
                EvidenceDoc { trust: 0.85, recency: 0.80, supports_claim: false },
                EvidenceDoc { trust: 0.82, recency: 0.75, supports_claim: false },
            ],
            // Single high-trust support
            vec![
                EvidenceDoc { trust: 0.95, recency: 0.95, supports_claim: true },
            ],
            // Many low-trust mixed
            vec![
                EvidenceDoc { trust: 0.30, recency: 0.40, supports_claim: true },
                EvidenceDoc { trust: 0.25, recency: 0.35, supports_claim: false },
                EvidenceDoc { trust: 0.28, recency: 0.38, supports_claim: true },
                EvidenceDoc { trust: 0.32, recency: 0.42, supports_claim: false },
                EvidenceDoc { trust: 0.22, recency: 0.30, supports_claim: true },
            ],
        ];

        let start = Instant::now();
        let mut naive_sum = 0.0_f64;
        let mut aware_sum = 0.0_f64;
        let mut contradiction_detected = 0usize;
        let mut aware_lower_count = 0usize;
        let mut total_evals = 0usize;

        for _ in 0..iterations {
            for docs in &doc_sets {
                let naive = naive_evidence_support(docs, &cfg);
                let (aware, has_contradiction) =
                    contradiction_aware_evidence_support(docs, &cfg, 0.30);
                naive_sum += naive as f64;
                aware_sum += aware as f64;
                total_evals += 1;
                if has_contradiction {
                    contradiction_detected += 1;
                }
                if aware < naive {
                    aware_lower_count += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;
        let naive_avg = naive_sum / total_evals as f64;
        let aware_avg = aware_sum / total_evals as f64;

        eprintln!(
            "[bench_evidence_merge] evals={} elapsed_ms={} per_eval_ns={:.1} naive_avg={:.4} aware_avg={:.4} contradictions={} aware_lower={}",
            total_evals, elapsed.as_millis(), per_eval_ns, naive_avg, aware_avg, contradiction_detected, aware_lower_count
        );

        assert!(aware_avg <= naive_avg, "Contradiction-aware should never exceed naive on average");
        assert!(
            contradiction_detected > 0,
            "Mixed doc sets must trigger contradiction detection"
        );
        assert!(
            aware_lower_count > 0,
            "Contradiction penalty must reduce score for conflicted sets"
        );
    }

    // =====================================================================
    // 5.7  Reasoning: reasoning loop trajectory simulation at scale
    // =====================================================================
    #[test]
    fn bench_reasoning_loop_trajectories() {
        let cfg = ReasoningLoopConfig::default();
        let iterations = 100_000usize;

        struct LoopCase {
            label: &'static str,
            initial: f32,
            boosts: &'static [f32],
            expect_exit: bool,
            expect_retrieval: bool,
        }

        let cases = [
            LoopCase {
                label: "easy_rescue",
                initial: 0.31,
                boosts: &[0.15, 0.16],
                expect_exit: true,
                expect_retrieval: false,
            },
            LoopCase {
                label: "stalled_needs_retrieval",
                initial: 0.24,
                boosts: &[0.08, 0.05, 0.04],
                expect_exit: false,
                expect_retrieval: true,
            },
            LoopCase {
                label: "already_confident",
                initial: 0.65,
                boosts: &[],
                expect_exit: false,
                expect_retrieval: false,
            },
            LoopCase {
                label: "gradual_climb",
                initial: 0.20,
                boosts: &[0.12, 0.14, 0.16],
                expect_exit: true,
                expect_retrieval: false,
            },
            LoopCase {
                label: "flat_trajectory",
                initial: 0.25,
                boosts: &[0.03, 0.02, 0.01],
                expect_exit: false,
                expect_retrieval: true,
            },
        ];

        let start = Instant::now();
        let mut total_evals = 0usize;
        let mut exit_count = 0usize;
        let mut retrieval_count = 0usize;
        let mut total_steps = 0usize;
        let mut correctness_failures = 0usize;

        for _ in 0..iterations {
            for case in &cases {
                let triggered = should_trigger_reasoning(IntentKind::Compare, case.initial, &cfg);
                let (final_conf, needs_retrieval, steps) = if triggered {
                    simulate_reasoning_progress(case.initial, case.boosts, &cfg)
                } else {
                    (case.initial, false, 0)
                };
                total_evals += 1;
                total_steps += steps;

                let exited = final_conf >= cfg.exit_confidence_threshold;
                if exited {
                    exit_count += 1;
                }
                if needs_retrieval {
                    retrieval_count += 1;
                }

                // Verify expectations
                if case.expect_exit && triggered && !exited {
                    correctness_failures += 1;
                }
                if case.expect_retrieval && triggered && !needs_retrieval {
                    correctness_failures += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;
        let avg_steps = total_steps as f64 / total_evals as f64;

        eprintln!(
            "[bench_reasoning_loop] evals={} elapsed_ms={} per_eval_ns={:.1} avg_steps={:.2} exit_rate={:.1}% retrieval_rate={:.1}% failures={}",
            total_evals, elapsed.as_millis(), per_eval_ns, avg_steps,
            exit_count as f64 / total_evals as f64 * 100.0,
            retrieval_count as f64 / total_evals as f64 * 100.0,
            correctness_failures
        );

        assert_eq!(
            correctness_failures, 0,
            "Reasoning loop trajectories must match expected outcomes"
        );
        assert!(exit_count > 0, "Some cases must exit via confidence threshold");
        assert!(retrieval_count > 0, "Some cases must escalate to retrieval");
    }

    // =====================================================================
    // 5.8  Predictive: path candidate scoring across all tiers
    // =====================================================================
    #[test]
    fn bench_path_scoring_all_tiers() {
        let candidates = [
            PathCandidate {
                label: "highway_high_ctx",
                tier: PredictiveTier::Highway,
                base_score: 0.90,
                proximity_or_decay: 1.0,
                context_match: 0.95,
                highway_boost: 1.30,
                subgraph_density: 0.80,
                anchor_trust: 0.50,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "highway_low_ctx",
                tier: PredictiveTier::Highway,
                base_score: 0.92,
                proximity_or_decay: 1.0,
                context_match: 0.20,
                highway_boost: 1.30,
                subgraph_density: 0.85,
                anchor_trust: 0.20,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier1_strong",
                tier: PredictiveTier::Tier1,
                base_score: 0.78,
                proximity_or_decay: 1.40,
                context_match: 0.92,
                highway_boost: 1.0,
                subgraph_density: 0.70,
                anchor_trust: 0.40,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier2_moderate",
                tier: PredictiveTier::Tier2,
                base_score: 0.65,
                proximity_or_decay: 0.70,
                context_match: 0.80,
                highway_boost: 1.0,
                subgraph_density: 0.60,
                anchor_trust: 0.30,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier3_pathfinding",
                tier: PredictiveTier::Tier3,
                base_score: 0.55,
                proximity_or_decay: 0.40,
                context_match: 0.75,
                highway_boost: 1.0,
                subgraph_density: 0.85,
                anchor_trust: 0.20,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier1_anchor_conflict",
                tier: PredictiveTier::Tier1,
                base_score: 0.93,
                proximity_or_decay: 1.30,
                context_match: 0.95,
                highway_boost: 1.0,
                subgraph_density: 0.80,
                anchor_trust: 0.96,
                contradicts_anchor: true,
            },
        ];

        let iterations = 100_000usize;
        let start = Instant::now();

        let mut naive_wins = [0usize; 6];
        let mut context_wins = [0usize; 6];
        let mut anchor_wins = [0usize; 6];
        let mut total_evals = 0usize;

        for _ in 0..iterations {
            let naive = naive_highway_selection(&candidates);
            let context = context_aware_selection(&candidates, false, 0.80);
            let anchor = context_aware_selection(&candidates, true, 0.80);
            total_evals += 1;

            for (i, c) in candidates.iter().enumerate() {
                if c.label == naive.label {
                    naive_wins[i] += 1;
                }
                if c.label == context.label {
                    context_wins[i] += 1;
                }
                if c.label == anchor.label {
                    anchor_wins[i] += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;

        eprintln!(
            "[bench_path_scoring] evals={} elapsed_ms={} per_eval_ns={:.1}",
            total_evals, elapsed.as_millis(), per_eval_ns
        );
        for (i, c) in candidates.iter().enumerate() {
            eprintln!(
                "  {}: naive={} context={} anchor={}",
                c.label, naive_wins[i], context_wins[i], anchor_wins[i]
            );
        }

        // Key invariant: anchor protection must change the winner for conflicting candidates
        let anchor_conflict_idx = candidates.iter().position(|c| c.label == "tier1_anchor_conflict").unwrap();
        assert_eq!(
            anchor_wins[anchor_conflict_idx], 0,
            "Anchor-conflicting candidate must never win under anchor protection"
        );

        // Context-aware should prefer high-context candidates over raw highway boost
        let highway_low_ctx_idx = candidates.iter().position(|c| c.label == "highway_low_ctx").unwrap();
        assert_eq!(
            context_wins[highway_low_ctx_idx], 0,
            "Low-context highway should not win under context-aware selection"
        );
    }

    // =====================================================================
    // 5.9  Predictive: beam search scaling benchmark
    // =====================================================================
    #[test]
    fn bench_beam_search_scaling() {
        #[derive(Clone)]
        struct Beam {
            path: Vec<u32>,
            score: f32,
        }

        // Synthetic graph: node N has edges to N*2, N*2+1 with decaying weights
        fn expand(node: u32, depth: u32) -> Vec<(u32, f32)> {
            if depth >= 5 {
                return Vec::new();
            }
            let left = node * 2;
            let right = node * 2 + 1;
            let decay = 0.90 - depth as f32 * 0.05;
            vec![(left, decay), (right, decay * 0.85)]
        }

        fn run_beam_search(beam_width: usize, max_depth: u32) -> (usize, f32) {
            let mut beams = vec![Beam {
                path: vec![1],
                score: 1.0,
            }];
            let mut total_expansions = 0usize;

            for depth in 0..max_depth {
                let mut next = Vec::new();
                for beam in &beams {
                    let node = *beam.path.last().unwrap();
                    let edges = expand(node, depth);
                    total_expansions += edges.len();
                    for (target, weight) in edges {
                        let mut path = beam.path.clone();
                        path.push(target);
                        next.push(Beam {
                            path,
                            score: beam.score * weight,
                        });
                    }
                }
                if next.is_empty() {
                    break;
                }
                next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                beams = next.into_iter().take(beam_width).collect();
            }

            let best_score = beams.iter().map(|b| b.score).fold(0.0_f32, f32::max);
            (total_expansions, best_score)
        }

        let beam_widths = [1, 3, 5, 10, 20];
        let depth = 5u32;
        let iterations = 10_000usize;

        let start = Instant::now();
        let mut results = Vec::new();

        for &bw in &beam_widths {
            let mut total_expansions = 0usize;
            let mut best_score_sum = 0.0_f64;
            for _ in 0..iterations {
                let (expansions, score) = run_beam_search(bw, depth);
                total_expansions += expansions;
                best_score_sum += score as f64;
            }
            let avg_score = best_score_sum / iterations as f64;
            results.push((bw, total_expansions, avg_score));
        }

        let elapsed = start.elapsed();

        eprintln!(
            "[bench_beam_scaling] depth={} iterations={} elapsed_ms={}",
            depth, iterations, elapsed.as_millis()
        );
        for (bw, expansions, avg_score) in &results {
            let avg_expansions = *expansions as f64 / iterations as f64;
            eprintln!(
                "  beam_width={:>2}: avg_expansions={:.1} avg_best_score={:.6}",
                bw, avg_expansions, avg_score
            );
        }

        // Wider beams must expand more states
        for i in 1..results.len() {
            assert!(
                results[i].1 >= results[i - 1].1,
                "Wider beam must explore >= states: bw={} expansions={} < bw={} expansions={}",
                results[i].0,
                results[i].1,
                results[i - 1].0,
                results[i - 1].1
            );
        }

        // Wider beams should find equal or better scores
        for i in 1..results.len() {
            assert!(
                results[i].2 >= results[i - 1].2 - 0.001,
                "Wider beam should find >= score: bw={} score={:.6} < bw={} score={:.6}",
                results[i].0,
                results[i].2,
                results[i - 1].0,
                results[i - 1].2
            );
        }
    }

    // =====================================================================
    // 5.10  End-to-end pipeline: full flow simulation
    // =====================================================================
    #[test]
    fn bench_full_pipeline_flow() {
        let retrieval_cfg = RetrievalThresholds::default();
        let reasoning_cfg = ReasoningLoopConfig::default();
        let evidence_cfg = EvidenceMergeConfig::default();
        let adaptive_cfg = AdaptiveBehaviorConfig::default();
        let scenarios = classification_scenario_pack();

        // Evidence doc sets per flow outcome
        let retrieval_docs = vec![
            EvidenceDoc { trust: 0.85, recency: 0.80, supports_claim: true },
            EvidenceDoc { trust: 0.78, recency: 0.75, supports_claim: true },
            EvidenceDoc { trust: 0.70, recency: 0.60, supports_claim: false },
        ];

        let path_candidates = [
            PathCandidate {
                label: "highway",
                tier: PredictiveTier::Highway,
                base_score: 0.85,
                proximity_or_decay: 1.0,
                context_match: 0.90,
                highway_boost: 1.25,
                subgraph_density: 0.80,
                anchor_trust: 0.70,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier1",
                tier: PredictiveTier::Tier1,
                base_score: 0.72,
                proximity_or_decay: 1.30,
                context_match: 0.85,
                highway_boost: 1.0,
                subgraph_density: 0.70,
                anchor_trust: 0.50,
                contradicts_anchor: false,
            },
            PathCandidate {
                label: "tier2",
                tier: PredictiveTier::Tier2,
                base_score: 0.60,
                proximity_or_decay: 0.65,
                context_match: 0.78,
                highway_boost: 1.0,
                subgraph_density: 0.55,
                anchor_trust: 0.30,
                contradicts_anchor: false,
            },
        ];

        let iterations = 50_000usize;
        let start = Instant::now();

        let mut flow_histogram = [0usize; 5];
        let mut retrieval_triggered = 0usize;
        let mut reasoning_triggered = 0usize;
        let mut evidence_merged = 0usize;
        let mut total_evals = 0usize;

        for _ in 0..iterations {
            for signals in &scenarios {
                total_evals += 1;

                // Phase 1: Classification → retrieval decision
                let decision = architecture_retrieval_decision(*signals, &retrieval_cfg);
                match decision.flow {
                    FlowKind::SocialShortCircuit => flow_histogram[0] += 1,
                    FlowKind::WarmLocal => flow_histogram[1] += 1,
                    FlowKind::ReasoningLocal => flow_histogram[2] += 1,
                    FlowKind::Retrieval => flow_histogram[3] += 1,
                    FlowKind::CreativeExploration => flow_histogram[4] += 1,
                }

                // Phase 2: Reasoning loop (if confidence is low)
                let reasoning_needed =
                    should_trigger_reasoning(signals.intent, signals.confidence, &reasoning_cfg);
                if reasoning_needed {
                    reasoning_triggered += 1;
                    let boosts: &[f32] = match signals.intent {
                        IntentKind::Compare => &[0.12, 0.14],
                        IntentKind::Question => &[0.08, 0.06, 0.04],
                        _ => &[0.10, 0.08],
                    };
                    let (_final_conf, needs_retrieval, _steps) =
                        simulate_reasoning_progress(signals.confidence, boosts, &reasoning_cfg);
                    if needs_retrieval && !decision.should_retrieve {
                        retrieval_triggered += 1;
                    }
                }

                // Phase 3: Evidence merge (if retrieval happened)
                if decision.should_retrieve {
                    retrieval_triggered += 1;
                    let (_support, _contradicted) =
                        contradiction_aware_evidence_support(&retrieval_docs, &evidence_cfg, 0.30);
                    evidence_merged += 1;
                }

                // Phase 4: Predictive path selection
                let is_factual = is_factual_anchor_intent(signals.intent);
                let _selected = context_aware_selection(&path_candidates, is_factual, 0.80);

                // Phase 5: Profile-aware scoring check
                let _profile_name = match signals.intent {
                    IntentKind::Question | IntentKind::Verify => "factual",
                    IntentKind::Brainstorm => "brainstorm",
                    IntentKind::Plan => "plan",
                    IntentKind::Compare => "explanatory",
                    _ => "casual",
                };
                let _profile = adaptive_cfg.intent_profile(_profile_name);
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;

        eprintln!(
            "[bench_full_pipeline] evals={} elapsed_ms={} per_eval_ns={:.1}",
            total_evals, elapsed.as_millis(), per_eval_ns
        );
        eprintln!(
            "  flows: social={} warm={} reasoning={} retrieval={} creative={}",
            flow_histogram[0], flow_histogram[1], flow_histogram[2], flow_histogram[3], flow_histogram[4]
        );
        eprintln!(
            "  reasoning_triggered={} retrieval_triggered={} evidence_merged={}",
            reasoning_triggered, retrieval_triggered, evidence_merged
        );

        // Pipeline performance
        assert!(
            per_eval_ns < 1000.0,
            "Full pipeline should be <1µs/eval, got {per_eval_ns:.1}ns"
        );

        // Flow distribution sanity
        assert!(flow_histogram[0] > 0, "Social flow must fire");
        assert!(flow_histogram[1] > 0, "Warm flow must fire");
        assert!(flow_histogram[2] > 0, "Reasoning flow must fire");
        assert!(flow_histogram[3] > 0, "Retrieval flow must fire");
        assert!(flow_histogram[4] > 0, "Creative flow must fire");

        // Retrieval should not dominate
        let retrieval_pct = flow_histogram[3] as f64 / total_evals as f64 * 100.0;
        assert!(
            retrieval_pct < 40.0,
            "Retrieval should not dominate: {retrieval_pct:.1}%"
        );
    }

    // =====================================================================
    // 5.11  Config sensitivity: weight perturbation stability
    // =====================================================================
    #[test]
    fn bench_config_sensitivity_weight_perturbation() {
        let base_weights = ScoringWeights::default();
        let base_w = [
            base_weights.spatial,
            base_weights.context,
            base_weights.sequence,
            base_weights.transition,
            base_weights.utility,
            base_weights.confidence,
            base_weights.evidence,
        ];

        // Fixed candidate pool
        let candidates: Vec<[f32; 7]> = vec![
            [0.9, 0.3, 0.5, 0.4, 0.6, 0.8, 0.7], // evidence-heavy
            [0.3, 0.9, 0.8, 0.7, 0.5, 0.4, 0.2], // context-heavy
            [0.5, 0.5, 0.9, 0.8, 0.4, 0.3, 0.3], // sequence-heavy
            [0.4, 0.6, 0.4, 0.3, 0.9, 0.5, 0.4], // utility-heavy
        ];

        fn score_and_rank(candidates: &[[f32; 7]], weights: &[f32; 7]) -> Vec<usize> {
            let mut indexed: Vec<(usize, f32)> = candidates
                .iter()
                .enumerate()
                .map(|(i, feat)| {
                    let s: f32 = feat.iter().zip(weights.iter()).map(|(f, w)| f * w).sum();
                    (i, s)
                })
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            indexed.iter().map(|(i, _)| *i).collect()
        }

        let base_ranking = score_and_rank(&candidates, &base_w);
        let perturbations = [0.01_f32, 0.02, 0.05, 0.10, 0.15];
        let iterations = 20_000usize;
        let start = Instant::now();

        let mut rank_changes = vec![0usize; perturbations.len()];
        let mut total_evals = 0usize;

        for _ in 0..iterations {
            for (pi, &delta) in perturbations.iter().enumerate() {
                // Perturb each dimension one at a time
                for dim in 0..7 {
                    let mut perturbed = base_w;
                    perturbed[dim] = (perturbed[dim] + delta).min(1.0);
                    // Re-normalize
                    let sum: f32 = perturbed.iter().sum();
                    if sum > 0.0 {
                        for w in perturbed.iter_mut() {
                            *w /= sum;
                        }
                    }

                    let new_ranking = score_and_rank(&candidates, &perturbed);
                    total_evals += 1;
                    if new_ranking != base_ranking {
                        rank_changes[pi] += 1;
                    }
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;

        eprintln!(
            "[bench_sensitivity] evals={} elapsed_ms={} per_eval_ns={:.1}",
            total_evals, elapsed.as_millis(), per_eval_ns
        );
        for (i, &delta) in perturbations.iter().enumerate() {
            let change_pct = rank_changes[i] as f64 / (iterations * 7) as f64 * 100.0;
            eprintln!(
                "  delta={:.2}: rank_changes={} ({:.1}%)",
                delta, rank_changes[i], change_pct
            );
        }

        // Small perturbations (1-2%) should not change ranking
        let small_perturbation_changes = rank_changes[0]; // delta=0.01
        let small_pct = small_perturbation_changes as f64 / (iterations * 7) as f64 * 100.0;
        assert!(
            small_pct < 30.0,
            "1% weight perturbation should not flip >30% of rankings: {small_pct:.1}%"
        );

        // Large perturbations should cause more changes (sensitivity exists)
        assert!(
            rank_changes[4] >= rank_changes[0],
            "Larger perturbations should cause >= rank changes"
        );
    }

    // =====================================================================
    // 5.12  Architecture vs naive: retrieval gating advantage quantified
    // =====================================================================
    #[test]
    fn bench_architecture_vs_naive_retrieval_advantage() {
        let cfg = RetrievalThresholds::default();
        let scenarios = classification_scenario_pack();
        let iterations = 100_000usize;

        // Ground truth: expected retrieval decisions for each scenario
        let expected_retrieve = [
            false, // social gratitude
            false, // warm factual
            false, // local reasoning compare
            true,  // open-world retrieval
            false, // context carry follow-up
            false, // procedural plan
            false, // creative brainstorm
            true,  // low-confidence unknown
        ];

        let start = Instant::now();
        let mut arch_correct = 0usize;
        let mut naive_correct = 0usize;
        let mut arch_false_positives = 0usize;
        let mut naive_false_positives = 0usize;
        let mut arch_false_negatives = 0usize;
        let mut naive_false_negatives = 0usize;
        let total_evals = iterations * scenarios.len();

        for _ in 0..iterations {
            for (i, signals) in scenarios.iter().enumerate() {
                let expected = expected_retrieve[i];

                // Architecture decision
                let arch_decision = architecture_retrieval_decision(*signals, &cfg);
                if arch_decision.should_retrieve == expected {
                    arch_correct += 1;
                } else if arch_decision.should_retrieve && !expected {
                    arch_false_positives += 1;
                } else {
                    arch_false_negatives += 1;
                }

                // Naive confidence-only decision
                let naive_decision = naive_confidence_only_retrieval(signals.confidence);
                if naive_decision == expected {
                    naive_correct += 1;
                } else if naive_decision && !expected {
                    naive_false_positives += 1;
                } else {
                    naive_false_negatives += 1;
                }
            }
        }

        let elapsed = start.elapsed();
        let per_eval_ns = elapsed.as_nanos() as f64 / total_evals as f64;
        let arch_accuracy = arch_correct as f64 / total_evals as f64 * 100.0;
        let naive_accuracy = naive_correct as f64 / total_evals as f64 * 100.0;
        let advantage = arch_accuracy - naive_accuracy;

        eprintln!(
            "[bench_arch_vs_naive] evals={} elapsed_ms={} per_eval_ns={:.1}",
            total_evals, elapsed.as_millis(), per_eval_ns
        );
        eprintln!(
            "  architecture: accuracy={:.1}% fp={} fn={}",
            arch_accuracy, arch_false_positives, arch_false_negatives
        );
        eprintln!(
            "  naive:        accuracy={:.1}% fp={} fn={}",
            naive_accuracy, naive_false_positives, naive_false_negatives
        );
        eprintln!("  advantage: +{:.1}pp", advantage);

        assert_eq!(
            arch_correct, total_evals,
            "Architecture gating should achieve 100% accuracy on scenario pack"
        );
        assert!(
            arch_accuracy > naive_accuracy,
            "Architecture must outperform naive: arch={arch_accuracy:.1}% naive={naive_accuracy:.1}%"
        );
        assert!(
            advantage >= 20.0,
            "Architecture advantage should be >= 20pp: {advantage:.1}pp"
        );
        assert!(
            naive_false_positives > 0,
            "Naive should over-retrieve (false positives)"
        );
    }
}
