use crate::config::{EvidenceMergeConfig, MicroValidatorConfig};
use crate::types::{
    ConflictRecord, ContextMatrix, EvidenceState, MergedState, RetrievedDocument, ScoredCandidate,
    Unit,
};
use std::collections::HashMap;

pub struct EvidenceMerger;

impl EvidenceMerger {
    pub fn merge(
        &self,
        internal_units: &[Unit],
        context: &ContextMatrix,
        evidence: EvidenceState,
        scored_candidates: &[ScoredCandidate],
        config: &EvidenceMergeConfig,
    ) -> MergedState {
        let mut candidate_ids = internal_units
            .iter()
            .map(|unit| unit.id)
            .collect::<Vec<_>>();
        let mut conflict_records = Vec::new();
        let mut evidence_support = evidence.average_trust * config.trust_weight;
        let mut agreement_hits = 0.0f32;

        for doc in &evidence.documents {
            let overlaps_context = context.cells.iter().any(|cell| {
                doc.normalized_content
                    .contains(&cell.content.to_lowercase())
            });

            if overlaps_context {
                agreement_hits += 1.0;
            } else if !context.summary.is_empty() {
                conflict_records.push(ConflictRecord {
                    claim: context.summary.clone(),
                    resolution: format!("retrieved evidence diverged from {}", doc.source_url),
                    external_trust: doc.trust_score,
                });
            }
        }

        candidate_ids.sort();
        candidate_ids.dedup();
        let agreement_support = if evidence.documents.is_empty() {
            0.0
        } else {
            (agreement_hits / evidence.documents.len() as f32) * config.agreement_weight
        };
        evidence_support = (evidence_support + agreement_support).clamp(0.0, 1.0);

        if should_run_micro_validator(
            scored_candidates,
            &evidence.documents,
            &config.micro_validator,
        ) {
            let validation = run_micro_validator(
                &evidence.documents,
                &context.summary,
                &config.micro_validator,
            );
            if validation.penalty > 0.0 {
                evidence_support = (evidence_support - validation.penalty).clamp(0.0, 1.0);
            }
            conflict_records.extend(validation.conflicts);
        }

        MergedState {
            candidate_ids,
            evidence_support,
            freshness_boost: (evidence.documents.len() as f32 * config.recency_weight * 0.2)
                .clamp(0.0, 0.25 + config.ambiguity_margin.clamp(0.0, 0.25)),
            conflict_records,
            evidence,
        }
    }
}

#[derive(Default)]
struct MicroValidationResult {
    penalty: f32,
    conflicts: Vec<ConflictRecord>,
}

fn should_run_micro_validator(
    scored_candidates: &[ScoredCandidate],
    documents: &[RetrievedDocument],
    config: &MicroValidatorConfig,
) -> bool {
    if !config.enabled || documents.is_empty() {
        return false;
    }
    if !config.lazy_validation_enabled {
        return true;
    }

    let top_score = scored_candidates
        .first()
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);
    let second_score = scored_candidates
        .get(1)
        .map(|candidate| candidate.score)
        .unwrap_or(0.0);
    let ambiguous = (top_score - second_score).abs() < config.ambiguity_margin;
    let low_trust = documents
        .iter()
        .any(|document| document.trust_score < config.validation_trust_floor);

    ambiguous || low_trust
}

fn run_micro_validator(
    documents: &[RetrievedDocument],
    context_summary: &str,
    config: &MicroValidatorConfig,
) -> MicroValidationResult {
    let mut result = MicroValidationResult::default();
    let mut contradiction_count = 0u32;

    let mut numeric_groups: HashMap<(String, String), Vec<(f64, f32)>> = HashMap::new();
    for document in documents {
        for (entity, value, unit) in &document.metadata_summary.numbers {
            numeric_groups
                .entry((entity.clone(), unit.clone()))
                .or_default()
                .push((*value, document.trust_score));
        }
    }

    for ((entity, unit), values) in numeric_groups {
        if values.len() < 2 {
            continue;
        }
        let min_value = values
            .iter()
            .map(|(value, _)| *value)
            .fold(f64::INFINITY, f64::min);
        let max_value = values
            .iter()
            .map(|(value, _)| *value)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_trust = values
            .iter()
            .map(|(_, trust)| *trust)
            .fold(0.0f32, f32::max);
        if max_trust >= config.contradiction_override_trust {
            continue;
        }
        let spread = if max_value.abs() <= f64::EPSILON {
            0.0
        } else {
            ((max_value - min_value).abs() / max_value.abs()) as f32
        };
        if spread > config.numeric_contradiction_threshold {
            contradiction_count += 1;
            result.conflicts.push(ConflictRecord {
                claim: format!("{entity}:{unit}"),
                resolution: format!(
                    "numeric contradiction detected for {entity} ({unit}) while validating {}",
                    summarize_context(context_summary)
                ),
                external_trust: max_trust,
            });
        }
    }

    let mut date_groups: HashMap<(String, String), Vec<(u16, f32)>> = HashMap::new();
    for document in documents {
        for (entity, relation, year) in &document.metadata_summary.dates {
            date_groups
                .entry((entity.clone(), relation.clone()))
                .or_default()
                .push((*year, document.trust_score));
        }
    }

    for ((entity, relation), values) in date_groups {
        let Some((first_year, _)) = values.first().copied() else {
            continue;
        };
        let has_conflict = values.iter().any(|(year, _)| *year != first_year);
        let max_trust = values
            .iter()
            .map(|(_, trust)| *trust)
            .fold(0.0f32, f32::max);
        if has_conflict && max_trust < config.contradiction_override_trust {
            contradiction_count += 1;
            result.conflicts.push(ConflictRecord {
                claim: format!("{entity}:{relation}"),
                resolution: format!(
                    "temporal contradiction detected for {entity} ({relation}) while validating {}",
                    summarize_context(context_summary)
                ),
                external_trust: max_trust,
            });
        }
    }

    let mut property_groups: HashMap<(String, String), Vec<(String, f32)>> = HashMap::new();
    for document in documents {
        for (entity, property, value) in &document.metadata_summary.properties {
            property_groups
                .entry((entity.clone(), property.clone()))
                .or_default()
                .push((value.clone(), document.trust_score));
        }
    }

    for ((entity, property), values) in property_groups {
        let Some((first_value, _)) = values.first() else {
            continue;
        };
        let has_conflict = values
            .iter()
            .any(|(value, _)| !value.eq_ignore_ascii_case(first_value));
        let max_trust = values
            .iter()
            .map(|(_, trust)| *trust)
            .fold(0.0f32, f32::max);
        if has_conflict && max_trust < config.contradiction_override_trust {
            contradiction_count += 1;
            result.conflicts.push(ConflictRecord {
                claim: format!("{entity}:{property}"),
                resolution: format!(
                    "entity-property contradiction detected for {entity} ({property}) while validating {}",
                    summarize_context(context_summary)
                ),
                external_trust: max_trust,
            });
        }
    }

    result.penalty = (config.contradiction_penalty * contradiction_count as f32).clamp(0.0, 0.75);
    result
}

fn summarize_context(summary: &str) -> &str {
    if summary.is_empty() {
        "retrieved evidence"
    } else {
        summary
    }
}
