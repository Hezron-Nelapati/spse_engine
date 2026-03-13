use crate::types::{
    FeedbackEvent, GovernanceReport, ProcessResult, RoutingResult, SearchDecision, SequenceState,
};

pub struct FeedbackController;

impl FeedbackController {
    pub fn learn(
        &self,
        decision: &SearchDecision,
        result: &ProcessResult,
        routing: &RoutingResult,
        sequence: &SequenceState,
        governance: Option<&GovernanceReport>,
    ) -> Vec<FeedbackEvent> {
        let mut events = Vec::new();
        let target_unit_id = sequence.recent_unit_ids.last().copied();

        events.push(FeedbackEvent {
            layer: 18,
            event: if decision.should_retrieve {
                "retrieval_path_observed".to_string()
            } else {
                "internal_path_observed".to_string()
            },
            impact: if result.used_retrieval { 0.6 } else { 0.4 },
            target_unit_id,
        });

        events.push(FeedbackEvent {
            layer: 5,
            event: "routing_adjustments_committed".to_string(),
            impact: (routing.map_adjustments as f32 / 20.0).clamp(0.1, 0.8),
            target_unit_id,
        });

        if !sequence.anchor_ids.is_empty() {
            events.push(FeedbackEvent {
                layer: 8,
                event: "anchor_reuse_detected".to_string(),
                impact: 0.5,
                target_unit_id,
            });
        }

        if let Some(report) = governance {
            events.push(FeedbackEvent {
                layer: 21,
                event: format!(
                    "maintenance_pruned={} promoted={}",
                    report.pruned_units, report.promoted_units
                ),
                impact: 0.4,
                target_unit_id,
            });
        }

        events
    }
}
