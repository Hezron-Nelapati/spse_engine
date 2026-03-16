use crate::types::{ContextCell, ContextMatrix, RoutingResult, SequenceState, Unit, UnitHierarchy};

pub struct ContextManager;

impl ContextManager {
    pub fn update(
        _routing: &RoutingResult,
        hierarchy: &UnitHierarchy,
        prior_state: &SequenceState,
        active_units: &[Unit],
        neighbor_units: &[Unit],
    ) -> (ContextMatrix, SequenceState) {
        let mut cells = Vec::new();
        let ranked_active = rank_units(active_units);
        let ranked_neighbors = rank_units(neighbor_units);

        for unit in ranked_active.iter().take(16) {
            cells.push(ContextCell {
                unit_id: Some(unit.id),
                content: unit.content.clone(),
                relevance: (unit.utility_score / 2.0).clamp(0.1, 1.0),
                recency: 1.0,
                salience: unit.salience_score.clamp(0.0, 1.0),
            });
        }

        for unit in ranked_neighbors.iter().take(12) {
            cells.push(ContextCell {
                unit_id: Some(unit.id),
                content: unit.content.clone(),
                relevance: (unit.confidence * 0.8).clamp(0.1, 1.0),
                recency: 0.65,
                salience: (unit.salience_score * 0.8).clamp(0.0, 1.0),
            });
        }

        cells.sort_by(|a, b| {
            let a_score = (a.relevance + a.recency + a.salience) / 3.0;
            let b_score = (b.relevance + b.recency + b.salience) / 3.0;
            b_score
                .partial_cmp(&a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cells.truncate(20);

        let summary = if cells.is_empty() {
            "no active context".to_string()
        } else {
            cells
                .iter()
                .take(6)
                .map(|cell| cell.content.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };

        let anchor_ids = active_units
            .iter()
            .filter(|unit| {
                hierarchy
                    .anchors
                    .iter()
                    .any(|anchor| anchor == &unit.content)
            })
            .map(|unit| unit.id)
            .collect::<Vec<_>>();

        let recent_unit_ids = active_units
            .iter()
            .take(8)
            .map(|unit| unit.id)
            .chain(prior_state.recent_unit_ids.iter().copied())
            .take(8)
            .collect::<Vec<_>>();

        let mut task_entities = prior_state.task_entities.clone();
        for entity in &hierarchy.entities {
            if !task_entities.contains(entity) {
                task_entities.push(entity.clone());
            }
        }
        task_entities.truncate(10);

        let mut context = ContextMatrix {
            cells,
            summary,
            ..Default::default()
        };
        context.precompute();
        (
            context,
            SequenceState {
                recent_unit_ids,
                anchor_ids,
                task_entities,
                turn_index: prior_state.turn_index + 1,
            },
        )
    }

    pub fn infer_context_summary(routing: &RoutingResult, hierarchy: &UnitHierarchy) -> String {
        let mut summary_parts = Vec::new();
        summary_parts.extend(routing.active_regions.iter().take(2).cloned());
        summary_parts.extend(hierarchy.anchors.iter().take(3).cloned());
        if summary_parts.is_empty() {
            "general_context".to_string()
        } else {
            summary_parts.join(" | ")
        }
    }
}

fn rank_units(units: &[Unit]) -> Vec<Unit> {
    let mut ranked = units
        .iter()
        .filter(|unit| unit.level != crate::types::UnitLevel::Char)
        .cloned()
        .collect::<Vec<_>>();

    if ranked.is_empty() {
        ranked = units.to_vec();
    }

    ranked.sort_by(|a, b| {
        let a_score = a.utility_score + a.salience_score + a.confidence;
        let b_score = b.utility_score + b.salience_score + b.confidence;
        b_score
            .partial_cmp(&a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}
