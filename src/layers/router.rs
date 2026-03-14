use crate::config::{CreativeSparkConfig, EscapeProfile, SemanticMapConfig};
use crate::spatial_index::{centroid, SpatialGrid};
use crate::types::{CandidateRoute, RoutingResult, ScoredCandidate, Unit};
use rand::Rng;
use std::collections::HashSet;

pub struct SemanticRouter {
    grid: SpatialGrid,
    neighbor_radius: f32,
}

impl SemanticRouter {
    pub fn new(config: &SemanticMapConfig) -> Self {
        Self {
            grid: SpatialGrid::new(config.spatial_cell_size),
            neighbor_radius: config.neighbor_radius,
        }
    }

    pub fn route(&mut self, active_units: &[Unit], all_units: &[Unit]) -> RoutingResult {
        self.grid.rebuild(all_units);
        let positions = active_units
            .iter()
            .map(|unit| unit.semantic_position)
            .collect::<Vec<_>>();
        let center = centroid(&positions);
        let mut neighbor_ids = self.grid.nearby(center, self.neighbor_radius);
        neighbor_ids.retain(|id| active_units.iter().all(|unit| unit.id != *id));

        let mut active_regions = Vec::new();
        for position in positions.iter().take(6) {
            active_regions.push(region_key(*position));
        }
        active_regions.sort();
        active_regions.dedup();

        RoutingResult {
            active_regions,
            neighbor_ids,
            map_adjustments: 0,
            position_updates: Vec::new(),
        }
    }

    pub fn route_candidates(
        &self,
        routing: &RoutingResult,
        all_units: &[Unit],
        max_candidates: usize,
        escape: &EscapeProfile,
    ) -> crate::types::CandidateRoute {
        let mut candidate_ids = routing.neighbor_ids.clone();
        let mut rationale = vec!["local_neighbors".to_string()];
        let target_candidates = max_candidates.max(1).min(escape.beam_width.max(1));
        let stochastic_escape = escape.stochastic_jump_prob > 0.0
            && rand::thread_rng().gen::<f32>() < escape.stochastic_jump_prob;

        if candidate_ids.len() < target_candidates / 2 || stochastic_escape {
            let mut global = all_units.to_vec();
            global.sort_by(|a, b| {
                b.utility_score
                    .partial_cmp(&a.utility_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let used_escape = true;
            if stochastic_escape {
                rationale.push("stochastic_jump".to_string());
            }
            for unit in global.into_iter().take(max_candidates) {
                if !candidate_ids.contains(&unit.id) {
                    candidate_ids.push(unit.id);
                }
                if candidate_ids.len() >= target_candidates {
                    break;
                }
            }
            rationale.push("global_escape".to_string());
            return crate::types::CandidateRoute {
                candidate_ids,
                used_escape,
                rationale,
            };
        }

        let mut dedup = HashSet::new();
        candidate_ids.retain(|id| dedup.insert(*id));
        candidate_ids.truncate(target_candidates);

        crate::types::CandidateRoute {
            candidate_ids,
            used_escape: false,
            rationale,
        }
    }

    /// Phase 3.2: Select candidate with creative spark (15% stochastic floor).
    /// Forces non-greedy selection 15% of the time, subject to anchor validation in resolver.
    pub fn select_with_creative_floor(
        candidates: &[ScoredCandidate],
        config: &CreativeSparkConfig,
    ) -> Option<ScoredCandidate> {
        if candidates.is_empty() {
            return None;
        }

        let mut rng = rand::thread_rng();
        let sample = rng.gen::<f32>();

        if sample < config.global_stochastic_floor {
            // Non-greedy selection: weighted random from top-K
            let top_k: Vec<_> = candidates.iter().take(5).collect();
            if top_k.is_empty() {
                return candidates.first().cloned();
            }

            // Apply softmax-like weighting with temperature
            let total_score: f32 = top_k
                .iter()
                .map(|c| (c.score / config.selection_temperature).exp())
                .sum();
            let mut cumulative = 0.0;
            let threshold = rng.gen::<f32>() * total_score;

            for candidate in &top_k {
                cumulative += (candidate.score / config.selection_temperature).exp();
                if cumulative >= threshold {
                    return Some((*candidate).clone());
                }
            }
            top_k.last().map(|c| (*c).clone())
        } else {
            // Greedy selection: return highest scored candidate
            candidates.first().cloned()
        }
    }
}

fn region_key(position: [f32; 3]) -> String {
    format!(
        "r:{:.0}:{:.0}:{:.0}",
        position[0].round(),
        position[1].round(),
        position[2].round()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SemanticMapConfig;
    use crate::types::{CandidateRoute, Link, MemoryChannel, MemoryType, Unit, UnitLevel};
    use chrono::Utc;
    use uuid::Uuid;

    fn unit(content: &str, utility: f32) -> Unit {
        Unit {
            id: Uuid::new_v4(),
            content: content.to_string(),
            normalized: content.to_string(),
            level: UnitLevel::Word,
            frequency: 1,
            utility_score: utility,
            confidence: utility,
            salience_score: utility,
            anchor_status: false,
            memory_type: MemoryType::Episodic,
            memory_channels: vec![MemoryChannel::Main],
            semantic_position: [utility, 0.0, 0.0],
            corroboration_count: 0,
            links: Vec::<Link>::new(),
            contexts: Vec::new(),
            created_at: Utc::now(),
            last_seen_at: Utc::now(),
            trust_score: utility,
        }
    }

    #[test]
    fn route_candidates_honors_beam_width() {
        let router = SemanticRouter::new(&SemanticMapConfig::default());
        let units = vec![unit("alpha", 0.9), unit("beta", 0.8), unit("gamma", 0.7)];
        let routing = RoutingResult {
            active_regions: vec!["r:0:0:0".to_string()],
            neighbor_ids: units.iter().map(|item| item.id).collect(),
            map_adjustments: 0,
            position_updates: Vec::new(),
        };
        let route: CandidateRoute = router.route_candidates(
            &routing,
            &units,
            10,
            &EscapeProfile {
                stochastic_jump_prob: 0.0,
                beam_width: 2,
            },
        );
        assert_eq!(route.candidate_ids.len(), 2);
    }
}
