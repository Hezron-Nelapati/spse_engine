use crate::config::ScoringWeights;
use crate::types::{
    ContextMatrix, IntentKind, MergedState, ScoreBreakdown, ScoredCandidate, SequenceState, Unit, UnitLevel,
};
use nalgebra::SVector;
use std::collections::HashSet;
use uuid::Uuid;

pub struct CandidateScorer;

impl CandidateScorer {
    pub fn score(
        candidates: &[Unit],
        context: &ContextMatrix,
        sequence: &SequenceState,
        merged: &MergedState,
        weights: &ScoringWeights,
        intent: Option<IntentKind>,
        original_query: Option<&str>,
    ) -> Vec<ScoredCandidate> {
        let mut scored = Vec::new();
        let weight_vector = scoring_weight_vector(weights);
        let merged_candidate_ids = merged.candidate_ids.iter().copied().collect::<HashSet<_>>();
        let recent_unit_ids = sequence
            .recent_unit_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let task_entities = sequence
            .task_entities
            .iter()
            .map(|entity| entity.to_lowercase())
            .collect::<Vec<_>>();
        
        // Extract query terms from context summary first (trained/dynamic intent resolution)
        // Fallback to original query for untrained cases
        let query_terms: Vec<String> = if !context.summary.is_empty() {
            context.summary
                .split_whitespace()
                .take(10)
                .map(|s| s.to_lowercase())
                .collect()
        } else {
            original_query
                .map(|q| q.split_whitespace().take(10).map(|s| s.to_lowercase()).collect())
                .unwrap_or_default()
        };

        for unit in candidates {
            let lowered = unit.content.to_lowercase();
            let spatial_fit = if merged_candidate_ids.contains(&unit.id) {
                0.9
            } else {
                0.35
            };
            let level_multiplier = level_multiplier(unit.level);
            let context_fit = context_match(&lowered, context) * level_multiplier;
            let sequence_fit = if recent_unit_ids.contains(&unit.id) {
                0.95 * level_multiplier
            } else if task_entities.iter().any(|entity| lowered.contains(entity)) {
                0.65 * level_multiplier
            } else {
                0.25 * level_multiplier
            };
            let transition_fit =
                ((unit.links.len() as f32 / 5.0).clamp(0.0, 1.0)) * level_multiplier;
            let utility_fit = unit.utility_score.clamp(0.0, 1.0) * level_multiplier;
            let confidence_fit = ((unit.confidence + unit.trust_score) / 2.0).clamp(0.0, 1.0);
            let evidence_support = evidence_match(&lowered, merged) * level_multiplier;
            
            // Exact match bonus: prioritize candidates that exactly match query terms
            // This helps disambiguate "Donald Trump" from "Donald Trump Jr."
            let exact_match_bonus = exact_match_score(&lowered, &query_terms, intent) * level_multiplier;

            let breakdown = ScoreBreakdown {
                spatial_fit,
                context_fit,
                sequence_fit,
                transition_fit,
                utility_fit,
                confidence_fit,
                evidence_support,
            };

            let feature_vector = SVector::<f32, 7>::from_row_slice(&[
                spatial_fit,
                context_fit,
                sequence_fit,
                transition_fit,
                utility_fit,
                confidence_fit,
                evidence_support,
            ]);
            let score = weight_vector.dot(&feature_vector) + merged.freshness_boost + exact_match_bonus;

            scored.push(ScoredCandidate {
                unit_id: unit.id,
                content: unit.content.clone(),
                score,
                breakdown,
                memory_type: unit.memory_type,
            });
        }

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored
    }

    pub fn confidence_stats(scored: &[ScoredCandidate]) -> crate::types::ConfidenceStats {
        if scored.is_empty() {
            return crate::types::ConfidenceStats::default();
        }

        let mean =
            scored.iter().map(|candidate| candidate.score).sum::<f32>() / scored.len() as f32;
        let variance = scored
            .iter()
            .map(|candidate| {
                let diff = candidate.score - mean;
                diff * diff
            })
            .sum::<f32>()
            / scored.len() as f32;

        crate::types::ConfidenceStats {
            mean_confidence: mean,
            candidate_count: scored.len(),
            disagreement: variance.sqrt().clamp(0.0, 1.0),
        }
    }
}

fn scoring_weight_vector(weights: &ScoringWeights) -> SVector<f32, 7> {
    SVector::<f32, 7>::from_row_slice(&[
        weights.spatial,
        weights.context,
        weights.sequence,
        weights.transition,
        weights.utility,
        weights.confidence,
        weights.evidence,
    ])
}

fn context_match(lowered: &str, context: &ContextMatrix) -> f32 {
    if context
        .cells
        .iter()
        .any(|cell| cell.content.to_lowercase() == lowered)
    {
        return 1.0;
    }
    if context.summary.to_lowercase().contains(lowered) {
        return 0.75;
    }
    0.3
}

fn evidence_match(lowered: &str, merged: &MergedState) -> f32 {
    let mentioned = merged
        .evidence
        .documents
        .iter()
        .any(|doc| doc.normalized_content.to_lowercase().contains(lowered));

    let corroboration_bonus = if mentioned {
        0.45 + (merged.evidence.average_trust * 0.4)
    } else {
        merged.evidence_support * 0.2
    };
    corroboration_bonus.clamp(0.0, 1.0)
}

fn level_multiplier(level: UnitLevel) -> f32 {
    match level {
        UnitLevel::Char => 0.15,
        UnitLevel::Subword => 0.45,
        UnitLevel::Word => 0.9,
        UnitLevel::Phrase => 1.0,
        UnitLevel::Pattern => 0.8,
    }
}

pub fn top_unit_ids(scored: &[ScoredCandidate], limit: usize) -> Vec<Uuid> {
    scored
        .iter()
        .take(limit)
        .map(|candidate| candidate.unit_id)
        .collect()
}

/// Calculate exact match bonus for entity disambiguation.
/// Prioritizes candidates that exactly match or closely match query terms.
/// This helps distinguish "Donald Trump" from "Donald Trump Jr." when querying for "Donald Trump".
fn exact_match_score(candidate: &str, query_terms: &[String], intent: Option<IntentKind>) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }
    
    // Only apply for factual/question intents where exact matching matters
    let is_factual = matches!(
        intent,
        Some(IntentKind::Question) | Some(IntentKind::Verify) | Some(IntentKind::Extract)
    );
    
    if !is_factual {
        return 0.0;
    }
    
    crate::common::similarity::SimilarityUtils::exact_match_score(candidate, query_terms)
}
