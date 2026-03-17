use crate::config::{CreativeSparkConfig, EscapeProfile, OutputScoringConfig, SemanticMapConfig};
use crate::spatial_index::{centroid, SpatialGrid};
use crate::types::{RoutingResult, ScoredCandidate, Unit};
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use uuid::Uuid;

pub struct SemanticRouter {
    config: SemanticMapConfig,
    grid: SpatialGrid,
    neighbor_radius: f32,
}

impl SemanticRouter {
    pub fn new(config: &SemanticMapConfig) -> Self {
        Self {
            config: config.clone(),
            grid: SpatialGrid::new(config.spatial_cell_size),
            neighbor_radius: config.neighbor_radius,
        }
    }

    /// Route using a pre-built cached spatial grid (fast path for large datasets)
    pub fn route_with_cached_grid(
        &mut self,
        active_units: &[Unit],
        cached_grid: &SpatialGrid,
    ) -> RoutingResult {
        let positions: Vec<[f32; 3]> = active_units
            .iter()
            .filter(|u| !u.is_process_unit)
            .map(|unit| unit.semantic_position)
            .collect();

        let center = if positions.is_empty() {
            [0.5, 0.5, 0.5]
        } else {
            centroid(&positions)
        };

        let mut neighbor_ids = cached_grid.nearby(center, self.neighbor_radius);

        // Use HashSet for O(1) lookup instead of O(n*m) nested iteration
        let active_ids: HashSet<Uuid> = active_units.iter().map(|u| u.id).collect();
        neighbor_ids.retain(|id| !active_ids.contains(id));

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

    pub fn route(&mut self, active_units: &[Unit], all_units: &[Unit]) -> RoutingResult {
        // Filter out process units from semantic routing
        let content_units: Vec<Unit> = all_units
            .iter()
            .filter(|u| !u.is_process_unit)
            .cloned()
            .collect();

        self.grid.rebuild(&content_units);
        let positions = active_units
            .iter()
            .filter(|u| !u.is_process_unit)
            .map(|unit| unit.semantic_position)
            .collect::<Vec<_>>();

        let center = if positions.is_empty() {
            [0.5, 0.5, 0.5]
        } else {
            centroid(&positions)
        };

        let mut neighbor_ids = self.grid.nearby(center, self.neighbor_radius);

        // Use HashSet for O(1) lookup instead of O(n*m) nested iteration
        let active_ids: HashSet<Uuid> = active_units.iter().map(|u| u.id).collect();
        neighbor_ids.retain(|id| !active_ids.contains(id));

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
        active_units: &[Unit],
        routing: &RoutingResult,
        all_units: &[Unit],
        max_candidates: usize,
        escape: &EscapeProfile,
        scoring: &OutputScoringConfig,
    ) -> crate::types::CandidateRoute {
        let target_candidates = max_candidates.max(1).min(escape.beam_width.max(1));
        let active_ids: HashSet<Uuid> = active_units.iter().map(|unit| unit.id).collect();
        let unit_index: HashMap<Uuid, &Unit> = all_units
            .iter()
            .filter(|unit| !unit.is_process_unit)
            .map(|unit| (unit.id, unit))
            .collect();

        let near_edges = self.scored_direct_edges(active_units, &unit_index, &active_ids, true, scoring);
        let far_edges = self.scored_direct_edges(active_units, &unit_index, &active_ids, false, scoring);
        let best_near = near_edges.first().map(|(_, score)| *score).unwrap_or(0.0);
        let best_far = far_edges.first().map(|(_, score)| *score).unwrap_or(0.0);

        let mut scored_candidates = Vec::new();
        let mut rationale = Vec::new();
        let mut used_pathfinding = false;

        if best_near >= self.config.tier1_confidence_threshold {
            scored_candidates.extend(near_edges.into_iter());
            rationale.push("tier1_near_edges".to_string());
        } else if best_near.max(best_far) >= self.config.tier2_confidence_threshold {
            scored_candidates.extend(near_edges.into_iter());
            scored_candidates.extend(far_edges.into_iter());
            rationale.push("tier2_far_edges".to_string());
        } else {
            scored_candidates.extend(near_edges.into_iter());
            scored_candidates.extend(far_edges.into_iter());
            let path_candidates =
                self.pathfind_candidates(active_units, all_units, &unit_index, &active_ids, scoring);
            if !path_candidates.is_empty() {
                used_pathfinding = true;
                rationale.push("tier3_pathfinding".to_string());
                scored_candidates.extend(path_candidates);
            }
        }

        if scored_candidates.is_empty() {
            rationale.push("spatial_nearest".to_string());
            scored_candidates.extend(
                routing
                    .neighbor_ids
                    .iter()
                    .filter_map(|id| unit_index.get(id).map(|unit| (unit.id, unit.utility_score))),
            );
        }

        scored_candidates.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

        let mut dedup = HashSet::new();
        let candidate_ids = scored_candidates
            .into_iter()
            .filter_map(|(id, _)| dedup.insert(id).then_some(id))
            .take(target_candidates)
            .collect();

        crate::types::CandidateRoute {
            candidate_ids,
            used_escape: used_pathfinding,
            rationale,
        }
    }

    fn scored_direct_edges(
        &self,
        active_units: &[Unit],
        unit_index: &HashMap<Uuid, &Unit>,
        active_ids: &HashSet<Uuid>,
        near_only: bool,
        scoring: &OutputScoringConfig,
    ) -> Vec<(Uuid, f32)> {
        let mut scored = Vec::new();

        for source in active_units {
            for link in &source.links {
                if link.weight < self.config.minimum_edge_weight
                    || active_ids.contains(&link.target_id)
                {
                    continue;
                }
                let Some(target) = unit_index.get(&link.target_id) else {
                    continue;
                };
                let distance = spatial_distance(source.semantic_position, target.semantic_position);
                let is_near = distance <= self.neighbor_radius;
                if near_only != is_near {
                    continue;
                }
                let score = if near_only {
                    let proximity_bonus =
                        1.0 - (distance / self.neighbor_radius.max(0.001)).min(1.0);
                    link.weight * proximity_bonus.max(scoring.proximity_bonus_floor)
                        * utility_bonus(target.utility_score, scoring)
                } else {
                    let distance_scale = (distance / self.neighbor_radius.max(0.001)).max(1.0);
                    let decayed = self.config.distance_decay.powf(distance_scale - 1.0);
                    link.weight * decayed * utility_bonus(target.utility_score, scoring)
                };
                scored.push((target.id, score));
            }
        }

        scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));
        scored
    }

    fn pathfind_candidates(
        &self,
        active_units: &[Unit],
        all_units: &[Unit],
        unit_index: &HashMap<Uuid, &Unit>,
        active_ids: &HashSet<Uuid>,
        scoring: &OutputScoringConfig,
    ) -> Vec<(Uuid, f32)> {
        let mut goals: Vec<&Unit> = all_units
            .iter()
            .filter(|unit| !unit.is_process_unit && !active_ids.contains(&unit.id))
            .collect();
        goals.sort_by(|lhs, rhs| {
            rhs.utility_score
                .partial_cmp(&lhs.utility_score)
                .unwrap_or(Ordering::Equal)
        });
        goals.truncate(8);

        let mut scored = Vec::new();
        for start in active_units {
            for goal in &goals {
                if let Some(path) = self.a_star_path(start.id, goal.id, unit_index, scoring) {
                    if path.len() < 2 {
                        continue;
                    }
                    let score = self.path_score(&path, unit_index);
                    for node_id in path.into_iter().skip(1) {
                        if let Some(unit) = unit_index.get(&node_id) {
                            if unit.links.len() >= self.config.hub_link_threshold
                                && node_id != goal.id
                            {
                                continue;
                            }
                            scored.push((node_id, score));
                        }
                    }
                }
            }
        }

        scored.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));
        scored
    }

    fn a_star_path(
        &self,
        start_id: Uuid,
        goal_id: Uuid,
        unit_index: &HashMap<Uuid, &Unit>,
        scoring: &OutputScoringConfig,
    ) -> Option<Vec<Uuid>> {
        #[derive(Clone)]
        struct State {
            node_id: Uuid,
            priority: f32,
            hops: usize,
            path: Vec<Uuid>,
        }

        impl PartialEq for State {
            fn eq(&self, other: &Self) -> bool {
                self.node_id == other.node_id
            }
        }

        impl Eq for State {}

        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .priority
                    .partial_cmp(&self.priority)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let goal = unit_index.get(&goal_id)?;
        let mut frontier = BinaryHeap::new();
        frontier.push(State {
            node_id: start_id,
            priority: 0.0,
            hops: 0,
            path: vec![start_id],
        });

        let mut explored = 0usize;
        let mut visited = HashSet::new();

        while let Some(state) = frontier.pop() {
            if state.node_id == goal_id {
                return Some(state.path);
            }
            if !visited.insert(state.node_id) {
                continue;
            }
            explored += 1;
            if explored >= self.config.pathfind_max_explored_nodes
                || state.hops >= self.config.pathfind_max_hops
            {
                continue;
            }

            let Some(current) = unit_index.get(&state.node_id) else {
                continue;
            };
            for link in &current.links {
                if link.weight < self.config.minimum_edge_weight {
                    continue;
                }
                let Some(next) = unit_index.get(&link.target_id) else {
                    continue;
                };
                if visited.contains(&next.id) {
                    continue;
                }
                let hub_bonus = if next.links.len() >= self.config.hub_link_threshold {
                    scoring.hub_bonus_non_hub
                } else {
                    scoring.hub_bonus_hub
                };
                let heuristic = spatial_distance(next.semantic_position, goal.semantic_position)
                    / self.neighbor_radius.max(0.001);
                let edge_cost = (1.0 - link.weight).max(0.0) * hub_bonus;
                let mut path = state.path.clone();
                path.push(next.id);
                frontier.push(State {
                    node_id: next.id,
                    priority: state.priority + edge_cost + heuristic,
                    hops: state.hops + 1,
                    path,
                });
            }
        }

        None
    }

    fn path_score(&self, path: &[Uuid], unit_index: &HashMap<Uuid, &Unit>) -> f32 {
        if path.len() < 2 {
            return 0.0;
        }

        let mut weight_product = 1.0f32;
        let mut density_sum = 0.0f32;
        let mut density_count = 0usize;

        for window in path.windows(2) {
            let Some(source) = unit_index.get(&window[0]) else {
                return 0.0;
            };
            let Some(target) = unit_index.get(&window[1]) else {
                return 0.0;
            };
            let Some(link) = source.links.iter().find(|link| link.target_id == target.id) else {
                return 0.0;
            };
            weight_product *= link.weight.max(0.001);
            density_sum +=
                (target.links.len() as f32 / self.config.hub_link_threshold as f32).clamp(0.1, 1.0);
            density_count += 1;
        }

        let subgraph_density = if density_count == 0 {
            0.0
        } else {
            density_sum / density_count as f32
        };
        weight_product * subgraph_density.max(0.1)
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

fn spatial_distance(lhs: [f32; 3], rhs: [f32; 3]) -> f32 {
    let dx = lhs[0] - rhs[0];
    let dy = lhs[1] - rhs[1];
    let dz = lhs[2] - rhs[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn utility_bonus(utility: f32, scoring: &OutputScoringConfig) -> f32 {
    scoring.utility_bonus_base + utility.clamp(0.0, 1.0) * scoring.utility_bonus_multiplier
}
