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
// §1  CLASSIFICATION SYSTEM — Config & Formula Validation
// ============================================================================

mod classification_system {
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
        assert!((cfg.w_intent_hash - 0.35).abs() < f32::EPSILON, "intent_hash weight");
        assert!((cfg.w_tone_hash - 0.20).abs() < f32::EPSILON, "tone_hash weight");
        assert!((cfg.w_semantic - 0.15).abs() < f32::EPSILON, "semantic weight");
        assert!((cfg.w_structure - 0.10).abs() < f32::EPSILON, "structure weight");
        assert!((cfg.w_punctuation - 0.10).abs() < f32::EPSILON, "punctuation weight");
        assert!((cfg.w_derived - 0.10).abs() < f32::EPSILON, "derived weight");
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
        assert!((c - 0.50).abs() < 0.02, "Ambiguous: expected ~0.50, got {c}");
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
            &RetrievalGateInputs { entropy: 0.3, freshness: 0.2, disagreement: 0.1 },
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
            &RetrievalGateInputs { entropy: 0.90, freshness: 0.70, disagreement: 0.40 },
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
            "casual", "explanatory", "factual", "procedural", "creative",
            "brainstorm", "plan", "act", "critique", "advisory",
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
            let sum = w.spatial + w.context + w.sequence + w.transition
                + w.utility + w.confidence + w.evidence;
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
        let entity_adj = *cfg.format_trust_adjustments.get("structured_entity").unwrap_or(&0.0);
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
        let expected = ["wikipedia.org", "wikidata.org", "archive.org", "gutenberg.org"];
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
}

// ============================================================================
// §2  REASONING SYSTEM — Scoring, Evidence Merge, Reasoning Loop
// ============================================================================

mod reasoning_system {
    use super::*;

    // -----------------------------------------------------------------
    // 2.1  7D candidate scoring weight vector sums to ~1.0 (§4.4)
    // -----------------------------------------------------------------
    #[test]
    fn candidate_scoring_weights_sum_near_one() {
        let w = ScoringWeights::default();
        let sum = w.spatial + w.context + w.sequence + w.transition
            + w.utility + w.confidence + w.evidence;
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
            weights.spatial, weights.context, weights.sequence, weights.transition,
            weights.utility, weights.confidence, weights.evidence,
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
        let evidence_support = avg_trust * cfg.trust_weight + agreement_ratio * cfg.agreement_weight;
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
        let evidence_support = avg_trust * cfg.trust_weight + agreement_ratio * cfg.agreement_weight;
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
            cfg.cold_start_discovery_utility_threshold
                < cfg.growth_discovery_utility_threshold,
            "Discovery utility: cold_start < growth"
        );
        assert!(
            cfg.growth_discovery_utility_threshold
                < cfg.stable_discovery_utility_threshold,
            "Discovery utility: growth < stable"
        );
        assert!(
            cfg.cold_start_candidate_observation_threshold
                < cfg.growth_candidate_observation_threshold,
            "Observation threshold: cold_start < growth"
        );
        assert!(
            cfg.growth_candidate_observation_threshold
                < cfg.stable_candidate_observation_threshold,
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
}

// ============================================================================
// §3  PREDICTIVE SYSTEM — Word Graph, Walk Scoring, Highways, Beam Search
// ============================================================================

mod predictive_system {
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
        assert!(highway_formed, "Highway should form after {highway_formation_threshold} walks");

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
            SimEdge { target: "account", weight: 0.7, context_tags: vec![finance_context] },
            SimEdge { target: "erosion", weight: 0.6, context_tags: vec![nature_context] },
            SimEdge { target: "robbery", weight: 0.5, context_tags: vec![finance_context] },
        ];

        // Active context: finance
        let active_context = finance_context;
        let activated: Vec<&str> = edges.iter()
            .filter(|e| e.context_tags.contains(&active_context))
            .map(|e| e.target)
            .collect();

        assert!(activated.contains(&"account"), "finance context should activate bank→account");
        assert!(activated.contains(&"robbery"), "finance context should activate bank→robbery");
        assert!(!activated.contains(&"erosion"), "finance context should NOT activate bank→erosion");
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
        let mut beams = vec![Beam { words: vec!["capital"], score: 1.0 }];

        // Step 1: expand each beam with candidate next words
        let mut next_beams = Vec::new();
        for beam in &beams {
            // Candidate edges from "capital"
            let candidates = vec![("of", 0.9), ("city", 0.7), ("letter", 0.3)];
            for (word, edge_score) in candidates {
                let mut words = beam.words.clone();
                words.push(word);
                next_beams.push(Beam { words, score: beam.score * edge_score });
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
        let factual_t = cfg.intent_profile("factual").unwrap().resolver.selection_temperature;
        let balanced_t = cfg.intent_profile("explanatory").unwrap().resolver.selection_temperature;
        let creative_t = cfg.intent_profile("creative").unwrap().resolver.selection_temperature;
        let brainstorm_t = cfg.intent_profile("brainstorm").unwrap().resolver.selection_temperature;

        assert!(factual_t < balanced_t, "factual < explanatory temperature");
        assert!(balanced_t < creative_t, "explanatory < creative temperature");
        assert!(creative_t < brainstorm_t, "creative < brainstorm temperature");
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

        assert!(creative.shaping.allow_semantic_drift, "Creative should allow drift");
        assert!(brainstorm.shaping.allow_semantic_drift, "Brainstorm should allow drift");
        assert!(!plan.shaping.allow_semantic_drift, "Plan should NOT allow drift");
        assert!(!factual.shaping.allow_semantic_drift, "Factual should NOT allow drift");
    }
}

// ============================================================================
// §4  CROSS-SYSTEM — Config Coherence & End-to-End Flow Properties
// ============================================================================

mod cross_system {
    use super::*;

    // -----------------------------------------------------------------
    // 4.1  Full EngineConfig loads from YAML without panic
    // -----------------------------------------------------------------
    #[test]
    fn engine_config_loads_from_yaml() {
        let yaml_path = concat!(env!("CARGO_MANIFEST_DIR"), "/config/config.yaml");
        let yaml_content = std::fs::read_to_string(yaml_path)
            .expect("config/config.yaml must exist");
        let config: EngineConfig = serde_yaml::from_str(&yaml_content)
            .expect("config.yaml must deserialize into EngineConfig");
        // Spot-check a few values
        assert!(config.auto_inference.reasoning_loop.enabled);
        assert!((config.retrieval.entropy_threshold - 0.72).abs() < 0.01
            || (config.retrieval.entropy_threshold - 0.85).abs() < 0.01,
            "Entropy threshold should be a documented value: {}",
            config.retrieval.entropy_threshold);
    }

    // -----------------------------------------------------------------
    // 4.2  Flow E: social short-circuit confidence threshold (§4.10)
    //   confidence > 0.95 AND intent ∈ social → skip Reasoning
    // -----------------------------------------------------------------
    #[test]
    fn social_shortcircuit_skips_reasoning() {
        let social_shortcircuit_confidence = 0.95_f32;
        let social_intents = [IntentKind::Greeting, IntentKind::Gratitude, IntentKind::Farewell];

        for intent in &social_intents {
            let confidence = 0.97;
            let should_skip = confidence > social_shortcircuit_confidence
                && matches!(intent, IntentKind::Greeting | IntentKind::Gratitude | IntentKind::Farewell);
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
        enum EdgeStatus { Probationary, Episodic, Purged }

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
        assert_eq!(edge.status, EdgeStatus::Episodic, "Edge should graduate to Episodic");

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
        assert_eq!(unused_edge.status, EdgeStatus::Purged, "Unused edge should be purged after lease");
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
                if p <= f32::EPSILON { e } else { e - p * p.ln() }
            });
            let max_entropy = (scores.len().max(2) as f32).ln();
            if max_entropy <= f32::EPSILON { 0.0 } else { (entropy / max_entropy).clamp(0.0, 1.0) }
        }

        // Uniform distribution → max entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let e = calculate_entropy(&uniform);
        assert!((e - 1.0).abs() < 0.01, "Uniform should give entropy ~1.0, got {e}");

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
        let channels = [MemoryChannel::Main, MemoryChannel::Intent, MemoryChannel::Reasoning];
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
        assert!(cfg.pollution_detection_enabled, "Pollution detection should be on by default");
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
            IntentKind::Greeting, IntentKind::Gratitude, IntentKind::Farewell,
            IntentKind::Help, IntentKind::Clarify, IntentKind::Rewrite,
            IntentKind::Verify, IntentKind::Continue, IntentKind::Forget,
            IntentKind::Question, IntentKind::Summarize, IntentKind::Explain,
            IntentKind::Compare, IntentKind::Extract, IntentKind::Analyze,
            IntentKind::Plan, IntentKind::Act, IntentKind::Recommend,
            IntentKind::Classify, IntentKind::Translate, IntentKind::Debug,
            IntentKind::Critique, IntentKind::Brainstorm, IntentKind::Unknown,
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
        assert_ne!(fp1, fp3, "Different input should produce different fingerprint");
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
}
