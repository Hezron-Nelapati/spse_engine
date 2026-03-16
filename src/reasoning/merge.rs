use crate::config::EvidenceMergeConfig;
use crate::types::{ConflictRecord, ContextMatrix, EvidenceState, MergedState, Unit};

pub struct EvidenceMerger;

impl EvidenceMerger {
    pub fn merge(
        &self,
        internal_units: &[Unit],
        context: &ContextMatrix,
        evidence: EvidenceState,
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
