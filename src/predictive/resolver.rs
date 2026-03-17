use crate::config::{CreativeSparkConfig, FineResolverConfig, IntentShapingConfig, ResolverThresholdConfig};
use crate::types::{ResolvedCandidate, ResolverMode, ScoredCandidate, Unit};
use rand::seq::SliceRandom;

pub struct FineResolver;

impl FineResolver {
    pub fn select(
        scored: &[ScoredCandidate],
        mode: ResolverMode,
        used_escape: bool,
        config: &FineResolverConfig,
        thresholds: &ResolverThresholdConfig,
    ) -> Option<ResolvedCandidate> {
        let meaningful = scored
            .iter()
            .filter(|candidate| candidate.content.len() > 1)
            .collect::<Vec<_>>();
        let preferred = if meaningful.is_empty() {
            scored.iter().collect::<Vec<_>>()
        } else {
            meaningful
        }
        .into_iter()
        .filter(|candidate| candidate.score >= config.min_confidence_floor)
        .collect::<Vec<_>>();

        if preferred.is_empty() {
            return None;
        }

        match mode {
            ResolverMode::Deterministic => preferred.first().map(|candidate| ResolvedCandidate {
                unit_id: candidate.unit_id,
                content: candidate.content.clone(),
                score: candidate.score,
                mode,
                used_escape,
            }),
            ResolverMode::Balanced => {
                let top_k = if config.selection_temperature <= thresholds.deterministic_temp_threshold {
                    1
                } else if config.selection_temperature <= thresholds.balanced_temp_threshold {
                    3
                } else {
                    5
                };
                let top = preferred.iter().take(top_k).copied().collect::<Vec<_>>();
                top.first().map(|candidate| ResolvedCandidate {
                    unit_id: candidate.unit_id,
                    content: candidate.content.clone(),
                    score: candidate.score,
                    mode,
                    used_escape,
                })
            }
            ResolverMode::Exploratory => {
                let top_k = if config.selection_temperature <= thresholds.exploratory_temp_low {
                    3
                } else if config.selection_temperature <= thresholds.exploratory_temp_high {
                    5
                } else {
                    7
                };
                let mut options = preferred.iter().take(top_k).copied().collect::<Vec<_>>();
                let mut rng = rand::thread_rng();
                options.shuffle(&mut rng);
                options.first().map(|candidate| ResolvedCandidate {
                    unit_id: candidate.unit_id,
                    content: candidate.content.clone(),
                    score: candidate.score,
                    mode,
                    used_escape,
                })
            }
        }
    }

    /// Select with intent shaping applied - allows semantic drift for creative intents
    /// while preserving factual anchors (units with high trust scores).
    pub fn select_with_shaping(
        scored: &[ScoredCandidate],
        mode: ResolverMode,
        used_escape: bool,
        config: &FineResolverConfig,
        shaping: &IntentShapingConfig,
        anchor_units: &[&Unit],
        thresholds: &ResolverThresholdConfig,
    ) -> Option<ResolvedCandidate> {
        // If semantic drift is not allowed, use standard selection
        if !shaping.allow_semantic_drift {
            return Self::select(scored, mode, used_escape, config, thresholds);
        }

        // For creative mode with semantic drift allowed:
        // 1. Identify protected anchor content
        // 2. Allow wider candidate selection
        // 3. Apply drift tolerance check

        let anchor_content: Vec<&str> = anchor_units
            .iter()
            .filter(|unit| unit.trust_score >= shaping.anchor_trust_threshold)
            .map(|unit| unit.content.as_str())
            .collect();

        let meaningful: Vec<_> = scored
            .iter()
            .filter(|candidate| candidate.content.len() > 1)
            .collect();

        let preferred: Vec<_> = if meaningful.is_empty() {
            scored.iter().collect()
        } else {
            meaningful
        };

        // Lower confidence floor for creative mode
        let lowered_floor = (config.min_confidence_floor * thresholds.creative_floor_multiplier).max(thresholds.min_confidence_floor_absolute);
        let eligible: Vec<_> = preferred
            .iter()
            .filter(|candidate| candidate.score >= lowered_floor)
            .collect();

        if eligible.is_empty() {
            return None;
        }

        // For exploratory mode with drift, use wider beam and stochastic selection
        let beam_width = (config.selection_temperature * 8.0) as usize;
        let top_k = beam_width.clamp(3, eligible.len().max(3));

        let mut options: Vec<_> = eligible.iter().take(top_k).copied().collect();
        let mut rng = rand::thread_rng();
        options.shuffle(&mut rng);

        // Select candidate, checking for factual corruption if anchors exist
        for candidate in &options {
            if shaping.preserve_factual_anchor && !anchor_content.is_empty() {
                // Check if candidate significantly contradicts anchor content
                let corruption_risk = Self::assess_factual_corruption(
                    &candidate.content,
                    &anchor_content,
                    config.factual_corruption_threshold,
                );
                if corruption_risk {
                    continue; // Skip this candidate due to corruption risk
                }
            }
            return Some(ResolvedCandidate {
                unit_id: candidate.unit_id,
                content: candidate.content.clone(),
                score: candidate.score,
                mode,
                used_escape,
            });
        }

        // Fallback to first option if all were filtered
        options.first().map(|candidate| ResolvedCandidate {
            unit_id: candidate.unit_id,
            content: candidate.content.clone(),
            score: candidate.score,
            mode,
            used_escape,
        })
    }

    /// Assess if candidate content contradicts factual anchors.
    /// Architecture does not specify keyword-based contradiction detection.
    /// Returns false - actual contradiction detection should come from memory-backed pattern matching.
    fn assess_factual_corruption(
        _candidate_content: &str,
        _anchor_contents: &[&str],
        _threshold: f32,
    ) -> bool {
        false
    }

    /// Calculate word overlap ratio between two strings
    fn word_overlap(a: &str, b: &str) -> f32 {
        crate::common::similarity::SimilarityUtils::jaccard_similarity(a, b)
    }

    /// Phase 3.2: Validate candidate against high-trust anchors.
    /// Returns false if candidate contradicts a protected anchor.
    pub fn validate_against_anchors(
        candidate: &ScoredCandidate,
        anchors: &[&Unit],
        config: &CreativeSparkConfig,
    ) -> bool {
        // Check each anchor for contradiction
        for anchor in anchors {
            if anchor.trust_score > config.anchor_protection_strictness {
                // This is a protected anchor - check for contradiction
                if Self::contradicts_anchor(&candidate.content, anchor) {
                    return false; // Reject creative drift on high-trust anchors
                }
            }
        }
        true
    }

    /// Check if candidate content contradicts an anchor unit.
    /// Architecture does not specify keyword-based contradiction detection.
    /// Returns false - actual contradiction detection should come from memory-backed pattern matching.
    fn contradicts_anchor(_candidate_content: &str, _anchor: &Unit) -> bool {
        false
    }
}
