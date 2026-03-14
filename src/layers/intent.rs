use crate::config::{IntentConfig, ReasoningLoopConfig, RetrievalThresholds, ToneInferenceConfig};
use crate::types::{
    ConfidenceStats, ContextMatrix, IntentBlendReport, IntentFallbackMode, IntentKind,
    IntentProfile, IntentScore, ScoredCandidate, SearchDecision, SequenceState, StyleAnchor,
    ToneKind, Unit,
};
use std::collections::HashMap;

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

    pub fn classify(
        raw_input: &str,
        context: &ContextMatrix,
        sequence: &SequenceState,
        has_active_document_context: bool,
        config: &IntentConfig,
    ) -> IntentProfile {
        let normalized = normalize(raw_input);
        let tokens = token_set(&normalized);
        let lexical_token_count = tokens.iter().filter(|token| token.len() > 1).count();
        let temporal_cues = temporal_cues(&normalized);
        let domain_hints = domain_hints(&normalized);
        let preference_cues = preference_cues(&normalized);
        let certainty_bias = certainty_bias(&normalized, &tokens);
        let wants_brief = wants_brief(&tokens, &normalized);

        let references_document_context = has_active_document_context
            && (references_context(&normalized)
                || domain_hints.iter().any(|hint| is_document_hint(hint))
                || !preference_cues.is_empty()
                || sequence.turn_index > 0);

        let mut scored = vec![
            score(IntentKind::Greeting, score_greeting(&normalized, &tokens)),
            score(IntentKind::Gratitude, score_gratitude(&normalized, &tokens)),
            score(IntentKind::Farewell, score_farewell(&normalized, &tokens)),
            score(IntentKind::Help, score_help(&normalized, &tokens)),
            score(IntentKind::Clarify, score_clarify(&normalized, &tokens)),
            score(IntentKind::Rewrite, score_rewrite(&normalized, &tokens)),
            score(
                IntentKind::Verify,
                score_verify(
                    &normalized,
                    &tokens,
                    &temporal_cues,
                    references_document_context,
                ),
            ),
            score(
                IntentKind::Continue,
                score_continue(&normalized, &tokens, &preference_cues, sequence),
            ),
            score(IntentKind::Forget, score_forget(&normalized, &tokens)),
            score(
                IntentKind::Summarize,
                score_summarize(&normalized, &tokens, wants_brief, context),
            ),
            score(
                IntentKind::Explain,
                score_explain(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Compare,
                score_compare(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Extract,
                score_extract(&normalized, &tokens, references_document_context),
            ),
            score(
                IntentKind::Analyze,
                score_analyze(&normalized, &tokens, context),
            ),
            score(IntentKind::Plan, score_plan(&normalized, &tokens, context)),
            score(
                IntentKind::Act,
                score_act(&normalized, &tokens, references_document_context),
            ),
            score(
                IntentKind::Recommend,
                score_recommend(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Classify,
                score_classify(&normalized, &tokens, references_document_context),
            ),
            score(IntentKind::Translate, score_translate(&normalized, &tokens)),
            score(
                IntentKind::Debug,
                score_debug(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Critique,
                score_critique(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Brainstorm,
                score_brainstorm(&normalized, &tokens, context),
            ),
            score(
                IntentKind::Question,
                score_question(
                    raw_input,
                    &normalized,
                    &tokens,
                    references_document_context,
                    context,
                ),
            ),
        ];

        for item in &mut scored {
            if references_document_context
                && matches!(
                    item.intent,
                    IntentKind::Question
                        | IntentKind::Summarize
                        | IntentKind::Explain
                        | IntentKind::Extract
                        | IntentKind::Compare
                        | IntentKind::Verify
                        | IntentKind::Continue
                        | IntentKind::Plan
                        | IntentKind::Classify
                        | IntentKind::Critique
                        | IntentKind::Debug
                )
            {
                item.score += config.context_reference_bonus;
            }

            if !temporal_cues.is_empty()
                && matches!(
                    item.intent,
                    IntentKind::Question
                        | IntentKind::Verify
                        | IntentKind::Extract
                        | IntentKind::Analyze
                        | IntentKind::Recommend
                        | IntentKind::Critique
                )
            {
                item.score += config.temporal_cue_weight;
            }

            if domain_hints.iter().any(|hint| is_document_hint(hint))
                && matches!(
                    item.intent,
                    IntentKind::Question
                        | IntentKind::Summarize
                        | IntentKind::Extract
                        | IntentKind::Verify
                        | IntentKind::Clarify
                        | IntentKind::Plan
                        | IntentKind::Classify
                )
            {
                item.score += config.domain_hint_weight;
            }

            if domain_hints.iter().any(|hint| is_code_hint(hint))
                && matches!(
                    item.intent,
                    IntentKind::Explain
                        | IntentKind::Analyze
                        | IntentKind::Compare
                        | IntentKind::Clarify
                        | IntentKind::Debug
                        | IntentKind::Act
                        | IntentKind::Plan
                        | IntentKind::Critique
                )
            {
                item.score += config.domain_hint_weight;
            }

            if !preference_cues.is_empty()
                && matches!(
                    item.intent,
                    IntentKind::Continue
                        | IntentKind::Rewrite
                        | IntentKind::Clarify
                        | IntentKind::Recommend
                )
            {
                item.score += config.preference_signal_weight;
            }
        }

        if wants_brief {
            for item in &mut scored {
                if matches!(
                    item.intent,
                    IntentKind::Summarize | IntentKind::Rewrite | IntentKind::Classify
                ) {
                    item.score += config.briefness_keyword_weight;
                }
            }
        }

        if certainty_bias < 0.0 {
            for item in &mut scored {
                if matches!(
                    item.intent,
                    IntentKind::Verify
                        | IntentKind::Question
                        | IntentKind::Critique
                        | IntentKind::Recommend
                ) {
                    item.score += certainty_bias.abs() * 0.25;
                }
            }
        }

        scored.sort_by(|lhs, rhs| rhs.score.total_cmp(&lhs.score));
        let top = scored.first().map(|entry| entry.score).unwrap_or(0.0);
        let second = scored.get(1).map(|entry| entry.score).unwrap_or(0.0);
        let top_gap = (top - second).max(0.0);
        let floor_score = (top / config.intent_floor_threshold.max(0.01)).clamp(0.0, 1.0);
        let gap_score = (top_gap / config.intent_ambiguity_margin.max(0.01)).clamp(0.0, 1.0);
        let confidence = (0.55 * floor_score + 0.45 * gap_score
            - certainty_bias.min(0.0).abs() * config.certainty_softener_weight)
            .clamp(0.0, 1.0);
        let mut ambiguous = top_gap < config.intent_ambiguity_margin;

        let mut primary = scored
            .first()
            .filter(|entry| entry.score >= config.intent_floor_threshold)
            .map(|entry| entry.intent)
            .unwrap_or(IntentKind::Unknown);
        let mut fallback_mode = IntentFallbackMode::None;
        let explicit_task_override = ambiguous
            && top >= (config.intent_floor_threshold + 0.18)
            && matches!(
                primary,
                IntentKind::Plan
                    | IntentKind::Act
                    | IntentKind::Recommend
                    | IntentKind::Classify
                    | IntentKind::Translate
                    | IntentKind::Debug
                    | IntentKind::Critique
                    | IntentKind::Brainstorm
            );

        if explicit_task_override {
            ambiguous = false;
        } else if primary == IntentKind::Unknown || ambiguous {
            if references_document_context {
                primary = IntentKind::Question;
                fallback_mode = IntentFallbackMode::DocumentScope;
            } else if lexical_token_count < config.min_lexical_tokens {
                // Check if input contains proper nouns (capitalized words)
                // Proper nouns like "Donald Trump" are valid entity mentions
                let has_proper_noun = raw_input.split_whitespace().any(|token| {
                    token.chars().next().map(|ch| ch.is_uppercase()).unwrap_or(false)
                        && token.len() > 1
                        && !matches!(token.to_lowercase().as_str(), "the" | "a" | "an" | "i" | "hi" | "hey" | "hello")
                });
                
                if has_proper_noun {
                    // Proper noun mentions should trigger retrieval, not clarification
                    primary = IntentKind::Question;
                    fallback_mode = IntentFallbackMode::RetrieveUnknown;
                } else {
                    primary = IntentKind::Help;
                    fallback_mode = IntentFallbackMode::ClarifyHelp;
                }
            } else {
                primary = IntentKind::Unknown;
                fallback_mode = IntentFallbackMode::RetrieveUnknown;
            }
        }

        let mut reasons = Vec::new();
        collect_reasons(
            raw_input,
            &normalized,
            &tokens,
            primary,
            references_document_context,
            ambiguous,
            &temporal_cues,
            &domain_hints,
            &preference_cues,
            certainty_bias,
            fallback_mode,
            &mut reasons,
        );
        if explicit_task_override {
            reasons.push("explicit_task_intent_override".to_string());
        }
        if wants_brief {
            reasons.push("brief_response_requested".to_string());
        }
        if context.summary.contains("freshness_sensitive") {
            reasons.push("context_marks_freshness".to_string());
        }

        IntentProfile {
            primary,
            confidence,
            top_score: top,
            second_score: second,
            ambiguous,
            wants_brief,
            references_document_context,
            certainty_bias,
            fallback_mode,
            scores: scored,
            reasons,
        }
    }

    pub fn assess(
        context: &ContextMatrix,
        sequence: &SequenceState,
        stats: &ConfidenceStats,
        scored: &[ScoredCandidate],
        thresholds: &RetrievalThresholds,
        raw_input: &str,
        intent: &IntentProfile,
    ) -> SearchDecision {
        let entropy = Self::calculate_entropy(scored);
        let freshness_need = freshness_signal(raw_input, context, sequence);
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
        let open_world_force =
            should_force_external_retrieval(raw_input, intent, scored, stats.mean_confidence);
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

    /// Phase 3.1: Confidence gating for dynamic reasoning.
    /// Returns true if reasoning loop should be triggered based on low confidence or complex intent.
    pub fn should_trigger_reasoning(
        intent: &IntentProfile,
        config: &ReasoningLoopConfig,
    ) -> bool {
        if !config.enabled {
            return false;
        }

        // Trigger if confidence is below floor
        if intent.confidence < config.trigger_confidence_floor {
            return true;
        }

        // Trigger for complex intents that benefit from reasoning
        let complex_intent = matches!(
            intent.primary,
            IntentKind::Analyze
                | IntentKind::Plan
                | IntentKind::Critique
                | IntentKind::Debug
                | IntentKind::Brainstorm
                | IntentKind::Compare
                | IntentKind::Explain
        );

        // Complex intents with moderate confidence also trigger reasoning
        complex_intent && intent.confidence < config.exit_confidence_threshold
    }
}

// ============================================================================
// Phase 3.3: Internal Tone Inference
// ============================================================================

/// Tone inferrer for internal tone detection from input semantics.
/// Tone is inferred from input, NOT a user setting.
pub struct ToneInferrer {
    /// Style anchors for each tone kind
    style_anchors: HashMap<ToneKind, StyleAnchor>,
    /// Decay rate for session persistence
    decay_rate: f32,
    /// Current active tone for session
    active_tone: Option<ToneKind>,
}

impl ToneInferrer {
    /// Create a new tone inferrer with default style anchors
    pub fn new(config: &ToneInferenceConfig) -> Self {
        let mut style_anchors = HashMap::new();

        // Default style anchors with semantic positions and keywords
        style_anchors.insert(
            ToneKind::NeutralProfessional,
            StyleAnchor {
                tone: ToneKind::NeutralProfessional,
                embedding: [0.5, 0.5, 0.5],
                keywords: vec![
                    "please".to_string(), "thank you".to_string(), "regarding".to_string(),
                    "sincerely".to_string(), "respectfully".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        style_anchors.insert(
            ToneKind::Empathetic,
            StyleAnchor {
                tone: ToneKind::Empathetic,
                embedding: [0.3, 0.7, 0.4],
                keywords: vec![
                    "sorry".to_string(), "understand".to_string(), "feel".to_string(),
                    "difficult".to_string(), "help".to_string(), "sad".to_string(),
                    "upset".to_string(), "worried".to_string(), "anxious".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        style_anchors.insert(
            ToneKind::Direct,
            StyleAnchor {
                tone: ToneKind::Direct,
                embedding: [0.8, 0.2, 0.3],
                keywords: vec![
                    "urgent".to_string(), "immediately".to_string(), "asap".to_string(),
                    "now".to_string(), "quickly".to_string(), "emergency".to_string(),
                    "critical".to_string(), "important".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        style_anchors.insert(
            ToneKind::Technical,
            StyleAnchor {
                tone: ToneKind::Technical,
                embedding: [0.6, 0.3, 0.7],
                keywords: vec![
                    "code".to_string(), "function".to_string(), "api".to_string(),
                    "implementation".to_string(), "algorithm".to_string(), "debug".to_string(),
                    "error".to_string(), "variable".to_string(), "method".to_string(),
                    "class".to_string(), "system".to_string(), "architecture".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        style_anchors.insert(
            ToneKind::Casual,
            StyleAnchor {
                tone: ToneKind::Casual,
                embedding: [0.4, 0.6, 0.3],
                keywords: vec![
                    "hey".to_string(), "hi".to_string(), "cool".to_string(),
                    "awesome".to_string(), "thanks".to_string(), "lol".to_string(),
                    "yeah".to_string(), "ok".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        style_anchors.insert(
            ToneKind::Formal,
            StyleAnchor {
                tone: ToneKind::Formal,
                embedding: [0.7, 0.4, 0.6],
                keywords: vec![
                    "dear".to_string(), "sincerely".to_string(), "respectfully".to_string(),
                    "kindly".to_string(), "request".to_string(), "permit".to_string(),
                    "acknowledge".to_string(), "gratitude".to_string(),
                ],
                decay_rate: config.style_anchor_decay,
            },
        );

        Self {
            style_anchors,
            decay_rate: config.style_anchor_decay,
            active_tone: None,
        }
    }

    /// Infer tone from input text and conversation history.
    pub fn infer_tone(&mut self, input: &str, _history: &[Unit], config: &ToneInferenceConfig) -> ToneKind {
        if !config.enabled {
            return ToneKind::NeutralProfessional;
        }

        let input_lower = input.to_lowercase();

        // Detect emotional markers
        let urgency = self.detect_urgency(&input_lower, config);
        let sadness = self.detect_sadness(&input_lower, config);
        let technical = self.detect_technical_domain(&input_lower, config);

        // Priority: Urgency > Sadness > Technical > Default
        if urgency > config.urgency_threshold {
            self.active_tone = Some(ToneKind::Direct);
            return ToneKind::Direct;
        }

        if sadness > config.sadness_threshold {
            self.active_tone = Some(ToneKind::Empathetic);
            return ToneKind::Empathetic;
        }

        if technical > config.technical_threshold {
            self.active_tone = Some(ToneKind::Technical);
            return ToneKind::Technical;
        }

        // Check for casual/formal markers
        let casual_score = self.score_against_anchor(&input_lower, ToneKind::Casual);
        let formal_score = self.score_against_anchor(&input_lower, ToneKind::Formal);

        if casual_score > 0.3 && casual_score > formal_score {
            self.active_tone = Some(ToneKind::Casual);
            return ToneKind::Casual;
        }

        if formal_score > 0.3 {
            self.active_tone = Some(ToneKind::Formal);
            return ToneKind::Formal;
        }

        // Persist active tone for session if decay is 0
        if self.decay_rate == 0.0 {
            if let Some(active) = self.active_tone {
                return active;
            }
        }

        // Fallback to neutral
        ToneKind::NeutralProfessional
    }

    /// Detect urgency markers in input
    fn detect_urgency(&self, input_lower: &str, config: &ToneInferenceConfig) -> f32 {
        let urgency_keywords = [
            "urgent", "emergency", "asap", "immediately", "now", "critical",
            "important", "quickly", "hurry", "fast", "right now",
        ];

        let mut score = 0.0;
        for keyword in &urgency_keywords {
            if input_lower.contains(keyword) {
                score += 0.15;
            }
        }

        // Check for exclamation marks
        let exclamation_count = input_lower.matches('!').count();
        score += exclamation_count as f32 * 0.1;

        score.min(1.0)
    }

    /// Detect sadness/empathy markers in input
    fn detect_sadness(&self, input_lower: &str, _config: &ToneInferenceConfig) -> f32 {
        let sadness_keywords = [
            "sad", "upset", "worried", "anxious", "depressed", "lonely",
            "hurt", "pain", "suffering", "struggling", "difficult", "hard",
            "lost", "grief", "cry", "tears", "hopeless",
        ];

        let mut score: f32 = 0.0;
        for keyword in &sadness_keywords {
            if input_lower.contains(keyword) {
                score += 0.12;
            }
        }

        score.min(1.0)
    }

    /// Detect technical domain markers in input
    fn detect_technical_domain(&self, input_lower: &str, _config: &ToneInferenceConfig) -> f32 {
        let technical_keywords = [
            "code", "function", "api", "algorithm", "debug", "error",
            "variable", "method", "class", "system", "architecture",
            "implementation", "database", "server", "client", "protocol",
            "interface", "module", "component", "service", "endpoint",
        ];

        let mut score: f32 = 0.0;
        for keyword in &technical_keywords {
            if input_lower.contains(keyword) {
                score += 0.08;
            }
        }

        score.min(1.0)
    }

    /// Score input against a style anchor
    fn score_against_anchor(&self, input_lower: &str, tone: ToneKind) -> f32 {
        if let Some(anchor) = self.style_anchors.get(&tone) {
            let mut score: f32 = 0.0;
            for keyword in &anchor.keywords {
                if input_lower.contains(keyword) {
                    score += 0.1;
                }
            }
            return score.min(1.0);
        }
        0.0
    }

    /// Get current active tone
    pub fn active_tone(&self) -> Option<ToneKind> {
        self.active_tone
    }

    /// Calculate style resonance between candidate and active style anchor
    pub fn style_resonance(&self, candidate: &Unit, anchor: &StyleAnchor) -> f32 {
        // Simple semantic similarity based on keyword overlap
        let candidate_lower = candidate.content.to_lowercase();
        let mut resonance: f32 = 0.0;

        for keyword in &anchor.keywords {
            if candidate_lower.contains(keyword) {
                resonance += 0.1;
            }
        }

        resonance.min(1.0)
    }
}

fn score(intent: IntentKind, value: f32) -> IntentScore {
    IntentScore {
        intent,
        score: value.clamp(0.0, 1.0),
    }
}

fn score_greeting(normalized: &str, tokens: &[String]) -> f32 {
    if matches!(
        normalized,
        "hi" | "hello" | "hey" | "hi there" | "hello there" | "good morning" | "good evening"
    ) {
        return 0.98;
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "hi" | "hello" | "hey"))
    {
        0.8
    } else {
        0.0
    }
}

fn score_gratitude(normalized: &str, tokens: &[String]) -> f32 {
    if normalized.contains("thank you") || tokens.iter().any(|token| token == "thanks") {
        0.94
    } else if tokens.iter().any(|token| token == "thx") {
        0.82
    } else {
        0.0
    }
}

fn score_farewell(normalized: &str, tokens: &[String]) -> f32 {
    if matches!(normalized, "bye" | "goodbye" | "see you") {
        0.96
    } else if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "bye" | "goodbye"))
    {
        0.82
    } else {
        0.0
    }
}

fn score_help(normalized: &str, tokens: &[String]) -> f32 {
    let mut score = 0.0;
    if normalized == "help" {
        score += 0.95;
    }
    if tokens.iter().any(|token| token == "help") {
        score += 0.45;
    }
    if normalized.contains("what can you do") || normalized.contains("how do i use") {
        score += 0.42;
    }
    score
}

fn score_clarify(normalized: &str, tokens: &[String]) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("clarify", 0.42),
            ("be specific", 0.28),
            ("more specific", 0.28),
            ("which one", 0.26),
            ("what do you mean", 0.34),
            ("narrow it down", 0.26),
            ("refine", 0.22),
        ],
    );
    if normalized.starts_with("clarify ") {
        score += 0.18;
    }
    score
}

fn score_rewrite(normalized: &str, tokens: &[String]) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("rewrite", 0.48),
            ("rephrase", 0.48),
            ("paraphrase", 0.46),
            ("word this", 0.34),
            ("say this differently", 0.42),
            ("simplify", 0.26),
        ],
    );
    if normalized.starts_with("rewrite ") || normalized.starts_with("rephrase ") {
        score += 0.16;
    }
    score
}

fn score_verify(
    normalized: &str,
    tokens: &[String],
    temporal_cues: &[String],
    references_document_context: bool,
) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("verify", 0.52),
            ("fact check", 0.48),
            ("is this correct", 0.44),
            ("check if", 0.38),
            ("confirm", 0.34),
            ("validate", 0.34),
            ("double check", 0.34),
        ],
    );
    if !temporal_cues.is_empty() {
        score += 0.12;
    }
    if references_document_context {
        score += 0.08;
    }
    score
}

fn score_continue(
    normalized: &str,
    tokens: &[String],
    preference_cues: &[String],
    sequence: &SequenceState,
) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("continue", 0.54),
            ("go on", 0.34),
            ("carry on", 0.34),
            ("pick up", 0.24),
            ("resume", 0.42),
        ],
    );
    if !preference_cues.is_empty() {
        score += 0.24;
    }
    if sequence.turn_index > 0 && matches!(normalized, "continue" | "go on") {
        score += 0.16;
    }
    score
}

fn score_forget(normalized: &str, tokens: &[String]) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("forget", 0.52),
            ("clear", 0.46),
            ("reset", 0.46),
            ("start over", 0.40),
            ("drop context", 0.36),
            ("new topic", 0.26),
        ],
    );
    if normalized.contains("clear context") || normalized.contains("clear the document") {
        score += 0.2;
    }
    score
}

fn score_summarize(
    normalized: &str,
    tokens: &[String],
    wants_brief: bool,
    context: &ContextMatrix,
) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("summary", 0.42),
            ("summarize", 0.48),
            ("summarise", 0.48),
            ("overview", 0.34),
            ("mainly about", 0.4),
            ("about this document", 0.34),
            ("document about", 0.28),
            ("gist", 0.32),
            ("main points", 0.34),
            ("high level", 0.24),
            ("in short", 0.26),
            ("tldr", 0.3),
        ],
    );
    if normalized.starts_with("what is this document mainly about")
        || normalized.starts_with("what's this document mainly about")
    {
        score += 0.5;
    }
    if wants_brief {
        score += 0.08;
    }
    if context.summary.contains("document") {
        score += 0.06;
    }
    score
}

fn score_explain(normalized: &str, tokens: &[String], context: &ContextMatrix) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("tell me about", 0.44),
            ("tell me more about", 0.46),
            ("explain", 0.46),
            ("why", 0.26),
            ("how", 0.24),
            ("meaning", 0.34),
            ("clarify", 0.3),
            ("describe", 0.22),
            ("walk me through", 0.34),
            ("how does", 0.38),
        ],
    );
    if normalized.starts_with("why ") || normalized.starts_with("how ") {
        score += 0.12;
    }
    if normalized.starts_with("tell me about ") || normalized.starts_with("tell me more about ") {
        score += 0.18;
    }
    if !context.summary.is_empty() {
        score += 0.03;
    }
    score
}

fn score_compare(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    cue_score(
        normalized,
        tokens,
        &[
            ("compare", 0.46),
            ("difference", 0.42),
            ("versus", 0.4),
            ("vs", 0.34),
            ("tradeoff", 0.34),
            ("better than", 0.22),
        ],
    )
}

fn score_extract(normalized: &str, tokens: &[String], references_document_context: bool) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("list all", 0.42),
            ("show me", 0.28),
            ("list", 0.28),
            ("show", 0.22),
            ("find", 0.26),
            ("quote", 0.3),
            ("mention", 0.22),
            ("where", 0.18),
            ("who", 0.16),
            ("when", 0.16),
            ("which", 0.18),
        ],
    );
    if normalized.starts_with("list ") || normalized.starts_with("show ") {
        score += 0.18;
    }
    if references_document_context {
        score += 0.08;
    }
    score
}

fn score_analyze(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    cue_score(
        normalized,
        tokens,
        &[
            ("analyze", 0.46),
            ("analyse", 0.46),
            ("assess", 0.3),
            ("evaluate", 0.32),
            ("critique", 0.28),
            ("implications", 0.24),
            ("risks", 0.22),
            ("strengths", 0.18),
            ("weaknesses", 0.18),
        ],
    )
}

fn score_plan(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("plan", 0.34),
            ("roadmap", 0.34),
            ("outline", 0.26),
            ("steps", 0.28),
            ("step by step", 0.40),
            ("step-by-step", 0.40),
            ("next steps", 0.38),
            ("approach", 0.24),
            ("implementation plan", 0.44),
            ("how should we proceed", 0.44),
            ("what should we do next", 0.40),
        ],
    );
    if normalized.starts_with("plan ") || normalized.starts_with("outline ") {
        score += 0.16;
    }
    score
}

fn score_act(normalized: &str, tokens: &[String], references_document_context: bool) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("implement", 0.42),
            ("run", 0.28),
            ("execute", 0.32),
            ("create", 0.22),
            ("add", 0.20),
            ("remove", 0.20),
            ("delete", 0.24),
            ("install", 0.30),
            ("start", 0.20),
            ("stop", 0.20),
            ("patch", 0.34),
            ("update", 0.20),
            ("modify", 0.22),
            ("write", 0.18),
        ],
    );
    if imperative_start(tokens) {
        score += 0.18;
    }
    if normalized.contains("do it") || normalized.contains("make the change") {
        score += 0.22;
    }
    if references_document_context {
        score += 0.06;
    }
    score
}

fn score_recommend(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("recommend", 0.50),
            ("suggest", 0.34),
            ("best", 0.18),
            ("advice", 0.24),
            ("what should i use", 0.44),
            ("should i use", 0.38),
            ("which should i choose", 0.42),
            ("what do you suggest", 0.42),
            ("what would you pick", 0.38),
        ],
    );
    if normalized.starts_with("recommend ") || normalized.starts_with("suggest ") {
        score += 0.14;
    }
    score
}

fn score_classify(normalized: &str, tokens: &[String], references_document_context: bool) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("classify", 0.52),
            ("categorize", 0.48),
            ("category", 0.26),
            ("label this", 0.34),
            ("tag this", 0.30),
            ("what type", 0.32),
            ("which type", 0.32),
            ("belongs to", 0.26),
        ],
    );
    if references_document_context {
        score += 0.06;
    }
    score
}

fn score_translate(normalized: &str, tokens: &[String]) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("translate", 0.58),
            ("translation", 0.34),
            ("into english", 0.36),
            ("to english", 0.24),
            ("into spanish", 0.36),
            ("into french", 0.36),
            ("into hindi", 0.36),
        ],
    );
    if normalized.starts_with("translate ") {
        score += 0.18;
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "spanish" | "french" | "hindi" | "english"))
    {
        score += 0.06;
    }
    score
}

fn score_debug(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("debug", 0.56),
            ("fix", 0.24),
            ("broken", 0.26),
            ("bug", 0.34),
            ("issue", 0.18),
            ("error", 0.26),
            ("failing", 0.24),
            ("fails", 0.22),
            ("failure", 0.22),
            ("stack trace", 0.34),
            ("exception", 0.30),
            ("panic", 0.30),
            ("crash", 0.30),
        ],
    );
    if normalized.starts_with("debug ") || normalized.starts_with("fix ") {
        score += 0.16;
    }
    score
}

fn score_critique(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    cue_score(
        normalized,
        tokens,
        &[
            ("critique", 0.52),
            ("review", 0.34),
            ("audit", 0.34),
            ("assess quality", 0.30),
            ("what's wrong", 0.34),
            ("weaknesses", 0.24),
            ("flaws", 0.28),
            ("holes", 0.20),
        ],
    )
}

fn score_brainstorm(normalized: &str, tokens: &[String], _context: &ContextMatrix) -> f32 {
    let mut score = cue_score(
        normalized,
        tokens,
        &[
            ("brainstorm", 0.56),
            ("ideas", 0.26),
            ("creative", 0.22),
            ("options", 0.22),
            ("come up with", 0.34),
            ("ways to", 0.20),
            ("explore options", 0.34),
        ],
    );
    if normalized.starts_with("brainstorm ") {
        score += 0.14;
    }
    score
}

fn score_question(
    raw_input: &str,
    normalized: &str,
    tokens: &[String],
    references_document_context: bool,
    context: &ContextMatrix,
) -> f32 {
    let mut score = 0.0;
    if contains_question_marker(raw_input) {
        score += 0.26;
    }
    if tokens
        .first()
        .map(|token| is_question_starter(token))
        .unwrap_or(false)
    {
        score += 0.28;
    }
    if references_document_context {
        score += 0.1;
    }
    if normalized.contains("what does it say") || normalized.contains("what is") || normalized.contains("who is") {
        score += 0.18;
    }
    if !context.summary.is_empty() {
        score += 0.04;
    }
    score
}

fn cue_score(normalized: &str, tokens: &[String], cues: &[(&str, f32)]) -> f32 {
    let mut score = 0.0;
    for (cue, weight) in cues {
        if cue.contains(' ') {
            if normalized.contains(cue) {
                score += *weight;
            }
        } else if tokens.iter().any(|token| token == cue) {
            score += *weight;
        }
    }
    score
}

#[allow(clippy::too_many_arguments)]
fn collect_reasons(
    raw_input: &str,
    _normalized: &str,
    tokens: &[String],
    primary: IntentKind,
    references_document_context: bool,
    ambiguous: bool,
    temporal_cues: &[String],
    domain_hints: &[String],
    preference_cues: &[String],
    certainty_bias: f32,
    fallback_mode: IntentFallbackMode,
    reasons: &mut Vec<String>,
) {
    if references_document_context {
        reasons.push("active_document_context".to_string());
    }
    if contains_question_marker(raw_input) {
        reasons.push("question_punctuation".to_string());
    }
    if tokens
        .first()
        .map(|token| is_question_starter(token))
        .unwrap_or(false)
    {
        reasons.push("question_starter".to_string());
    }
    if ambiguous {
        reasons.push("intent_ambiguity_detected".to_string());
    }
    if !temporal_cues.is_empty() {
        reasons.push(format!("temporal_cues={}", temporal_cues.join(",")));
    }
    if !domain_hints.is_empty() {
        reasons.push(format!("domain_hints={}", domain_hints.join(",")));
    }
    if !preference_cues.is_empty() {
        reasons.push(format!("preference_cues={}", preference_cues.join(",")));
    }
    if certainty_bias < 0.0 {
        reasons.push("tentative_language_present".to_string());
    } else if certainty_bias > 0.0 {
        reasons.push("definitive_language_present".to_string());
    }
    match fallback_mode {
        IntentFallbackMode::DocumentScope => {
            reasons.push("low_confidence_document_scope_fallback".to_string())
        }
        IntentFallbackMode::ClarifyHelp => {
            reasons.push("low_confidence_clarification_fallback".to_string())
        }
        IntentFallbackMode::RetrieveUnknown => {
            reasons.push("low_confidence_retrieval_fallback".to_string())
        }
        IntentFallbackMode::None => {}
    }
    match primary {
        IntentKind::Summarize => reasons.push("compression_or_overview_request".to_string()),
        IntentKind::Explain => reasons.push("causal_or_mechanistic_request".to_string()),
        IntentKind::Compare => reasons.push("comparison_markers_detected".to_string()),
        IntentKind::Extract => reasons.push("targeted_lookup_request".to_string()),
        IntentKind::Analyze => reasons.push("evaluation_or_risk_request".to_string()),
        IntentKind::Plan => reasons.push("multi_step_planning_request".to_string()),
        IntentKind::Act => reasons.push("imperative_action_request".to_string()),
        IntentKind::Recommend => reasons.push("recommendation_request".to_string()),
        IntentKind::Classify => reasons.push("classification_request".to_string()),
        IntentKind::Translate => reasons.push("translation_request".to_string()),
        IntentKind::Debug => reasons.push("debugging_request".to_string()),
        IntentKind::Critique => reasons.push("critique_request".to_string()),
        IntentKind::Brainstorm => reasons.push("idea_generation_request".to_string()),
        IntentKind::Question => reasons.push("general_question_request".to_string()),
        IntentKind::Clarify => reasons.push("disambiguation_request".to_string()),
        IntentKind::Rewrite => reasons.push("surface_rewrite_request".to_string()),
        IntentKind::Verify => reasons.push("verification_request".to_string()),
        IntentKind::Continue => reasons.push("continuation_request".to_string()),
        IntentKind::Forget => reasons.push("session_reset_request".to_string()),
        IntentKind::Greeting => reasons.push("social_opening".to_string()),
        IntentKind::Gratitude => reasons.push("gratitude_signal".to_string()),
        IntentKind::Farewell => reasons.push("conversation_close_signal".to_string()),
        IntentKind::Help => reasons.push("capability_request".to_string()),
        IntentKind::Unknown => {}
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

fn references_context(normalized: &str) -> bool {
    normalized.starts_with("and ")
        || normalized.starts_with("also ")
        || normalized.starts_with("what about ")
        || normalized.contains(" it ")
        || normalized.contains(" they ")
        || normalized.contains(" that ")
        || normalized.contains(" those ")
        || normalized.contains(" this ")
}

fn wants_brief(tokens: &[String], normalized: &str) -> bool {
    tokens.iter().any(|token| {
        matches!(
            token.as_str(),
            "briefly" | "short" | "quickly" | "concise" | "brief" | "tldr"
        )
    }) || normalized.contains("in short")
}

fn temporal_cues(normalized: &str) -> Vec<String> {
    let mut found = Vec::new();
    for cue in [
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "recently",
        "last week",
        "last month",
        "now",
        "currently",
        "price",
        "weather",
    ] {
        if normalized.contains(cue) {
            found.push(cue.to_string());
        }
    }
    found
}

fn domain_hints(normalized: &str) -> Vec<String> {
    let mut found = Vec::new();
    for cue in [
        "in code",
        "in the code",
        "in the doc",
        "in the document",
        "in docs",
        "in my notes",
        "in notes",
    ] {
        if normalized.contains(cue) {
            found.push(cue.to_string());
        }
    }
    found
}

fn preference_cues(normalized: &str) -> Vec<String> {
    let mut found = Vec::new();
    for cue in [
        "as before",
        "like last time",
        "same as before",
        "as earlier",
    ] {
        if normalized.contains(cue) {
            found.push(cue.to_string());
        }
    }
    found
}

fn certainty_bias(normalized: &str, tokens: &[String]) -> f32 {
    let mut bias: f32 = 0.0;
    if normalized.contains("probably")
        || normalized.contains("maybe")
        || normalized.contains("might")
        || normalized.contains("not sure")
    {
        bias -= 0.2;
    }
    if tokens
        .iter()
        .any(|token| matches!(token.as_str(), "definitely" | "certainly"))
    {
        bias += 0.12;
    }
    bias.clamp(-0.3, 0.2)
}

fn is_document_hint(hint: &str) -> bool {
    hint.contains("doc") || hint.contains("document") || hint.contains("notes")
}

fn is_code_hint(hint: &str) -> bool {
    hint.contains("code")
}

fn contains_question_marker(raw_input: &str) -> bool {
    raw_input.contains('?') || raw_input.contains('？') || raw_input.contains('¿')
}

fn is_question_starter(token: &str) -> bool {
    matches!(
        token,
        "what"
            | "why"
            | "how"
            | "when"
            | "where"
            | "who"
            | "which"
            | "is"
            | "are"
            | "does"
            | "do"
            | "can"
            | "could"
            | "would"
            | "should"
    )
}

fn freshness_signal(raw_input: &str, context: &ContextMatrix, sequence: &SequenceState) -> f32 {
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
            freshness += 0.22;
        }
    }
    if context.summary.contains("freshness_sensitive") {
        freshness += 0.18;
    }
    if !sequence.task_entities.is_empty() {
        freshness += 0.08;
    }
    freshness.clamp(0.0, 1.2)
}

fn should_force_external_retrieval(
    raw_input: &str,
    intent: &IntentProfile,
    scored: &[ScoredCandidate],
    mean_confidence: f32,
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
    // 1. The top candidate score is below a reasonable threshold (0.65)
    // 2. The candidate doesn't appear to directly answer the question
    // This prevents settling for vaguely related but unhelpful candidates
    let factual_intent = matches!(
        intent.primary,
        IntentKind::Question | IntentKind::Verify | IntentKind::Explain | IntentKind::Extract
    );
    
    if factual_intent && top.score < 0.65 {
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

fn imperative_start(tokens: &[String]) -> bool {
    tokens
        .first()
        .map(|token| {
            matches!(
                token.as_str(),
                "implement"
                    | "run"
                    | "execute"
                    | "create"
                    | "add"
                    | "remove"
                    | "delete"
                    | "install"
                    | "start"
                    | "stop"
                    | "patch"
                    | "update"
                    | "modify"
                    | "write"
            )
        })
        .unwrap_or(false)
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

impl IntentDetector {
    /// Blend heuristic intent classification with memory-backed intent lookup.
    /// This validates intent detection by cross-referencing Intent-channel memory units.
    pub fn hybrid_blend(
        heuristic_profile: &IntentProfile,
        memory_backed_intent: Option<IntentKind>,
        memory_backed_confidence: f32,
        heuristic_weight: f32,
        memory_weight: f32,
    ) -> IntentBlendReport {
        let heuristic_intent = heuristic_profile.primary;
        let heuristic_confidence = heuristic_profile.confidence;

        let intents_agree = memory_backed_intent
            .map(|mem_intent| mem_intent == heuristic_intent)
            .unwrap_or(true);

        let drift_detected = !intents_agree && memory_backed_confidence > 0.3;

        let drift_reason = if drift_detected {
            Some(format!(
                "Heuristic {:?} ({:.2}) conflicts with memory-backed {:?} ({:.2})",
                heuristic_intent, heuristic_confidence, memory_backed_intent, memory_backed_confidence
            ))
        } else {
            None
        };

        // Blend the confidence scores
        let blended_confidence = if let Some(_mem_intent) = memory_backed_intent {
            if intents_agree {
                // Intents agree: boost confidence
                (heuristic_confidence * heuristic_weight + memory_backed_confidence * memory_weight)
                    .min(1.0)
            } else {
                // Intents disagree: use weighted average but flag drift
                heuristic_confidence * heuristic_weight + memory_backed_confidence * memory_weight
            }
        } else {
            // No memory-backed intent: use heuristic confidence
            heuristic_confidence
        };

        // Determine blended intent
        let blended_intent = if let Some(mem_intent) = memory_backed_intent {
            if intents_agree || memory_backed_confidence > heuristic_confidence + 0.2 {
                // Use memory-backed if it agrees or is significantly more confident
                mem_intent
            } else {
                heuristic_intent
            }
        } else {
            heuristic_intent
        };

        IntentBlendReport {
            heuristic_intent,
            heuristic_confidence,
            memory_backed_intent,
            memory_backed_confidence,
            blended_intent,
            blended_confidence,
            heuristic_weight,
            memory_weight,
            intents_agree,
            drift_detected,
            drift_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::IntentDetector;
    use crate::config::IntentConfig;
    use crate::config::RetrievalThresholds;
    use crate::types::{
        ConfidenceStats, ContextMatrix, IntentFallbackMode, IntentKind, MemoryType, ScoreBreakdown,
        ScoredCandidate, SequenceState,
    };
    use uuid::Uuid;

    #[test]
    fn low_confidence_request_forces_unknown_retrieval_fallback() {
        let profile = IntentDetector::classify(
            "stuff about the thing",
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Unknown);
        assert_eq!(profile.fallback_mode, IntentFallbackMode::RetrieveUnknown);
    }

    #[test]
    fn clear_intent_is_detected() {
        let profile = IntentDetector::classify(
            "clear the document context",
            &ContextMatrix::default(),
            &SequenceState::default(),
            true,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Forget);
    }

    #[test]
    fn document_reference_sets_context_flag() {
        let mut state = SequenceState::default();
        state.turn_index = 1;
        let profile = IntentDetector::classify(
            "what does it say about anchors?",
            &ContextMatrix::default(),
            &state,
            true,
            &IntentConfig::default(),
        );

        assert!(profile.references_document_context);
        assert!(matches!(
            profile.primary,
            IntentKind::Question | IntentKind::Extract
        ));
    }

    #[test]
    fn translate_request_prefers_translate_intent() {
        let profile = IntentDetector::classify(
            "Translate hello to Spanish.",
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Translate);
    }

    #[test]
    fn debug_request_prefers_debug_intent() {
        let profile = IntentDetector::classify(
            "Debug why this Rust test is failing with a panic.",
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Debug);
    }

    #[test]
    fn weighted_retrieval_score_matches_manual_formula() {
        let thresholds = RetrievalThresholds {
            w_entropy: 0.8,
            w_recency: 1.2,
            w_disagreement: 0.6,
            w_cost: 0.5,
            decision_threshold: 9.0,
            ..RetrievalThresholds::default()
        };
        let scored = vec![
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "alpha".to_string(),
                score: 0.7,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "beta".to_string(),
                score: 0.2,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "gamma".to_string(),
                score: 0.1,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
        ];
        let stats = ConfidenceStats {
            mean_confidence: 0.333,
            candidate_count: scored.len(),
            disagreement: 0.24,
        };
        let entropy = IntentDetector::calculate_entropy(&scored);
        let freshness_need = 0.0;
        let manual = (thresholds.w_entropy * entropy)
            + (thresholds.w_recency * freshness_need)
            + (thresholds.w_disagreement * stats.disagreement)
            - (thresholds.w_cost * thresholds.cost_penalty);
        let decision = IntentDetector::assess(
            &ContextMatrix::default(),
            &SequenceState::default(),
            &stats,
            &scored,
            &thresholds,
            "tell me about alpha",
            &crate::types::IntentProfile {
                primary: IntentKind::Analyze,
                ..crate::types::IntentProfile::default()
            },
        );

        assert!((decision.score - manual).abs() < 1e-6);
    }

    #[test]
    fn tell_me_about_prefers_explain_intent() {
        let profile = IntentDetector::classify(
            "Tell me about cars.",
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Explain);
    }

    #[test]
    fn list_all_prefers_extract_intent() {
        let profile = IntentDetector::classify(
            "List all cars by Ferrari.",
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &IntentConfig::default(),
        );

        assert_eq!(profile.primary, IntentKind::Extract);
    }
}
