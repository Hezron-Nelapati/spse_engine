use crate::config::SemanticMapConfig;
use crate::types::{MemoryType, Unit};
use petgraph::graph::NodeIndex;
use petgraph::{Graph, Undirected};
use std::collections::HashMap;
use uuid::Uuid;

type CellId = (i32, i32, i32);

pub struct SpatialGrid {
    cells: HashMap<CellId, Vec<(Uuid, [f32; 3])>>,
    cell_size: f32,
}

pub struct LayoutOutcome {
    pub position_updates: Vec<(Uuid, [f32; 3])>,
    pub mean_displacement: f32,
    pub rolled_back: bool,
}

impl SpatialGrid {
    pub fn new(cell_size: f32) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size,
        }
    }

    pub fn rebuild(&mut self, units: &[Unit]) {
        self.cells.clear();
        for unit in units {
            self.cells
                .entry(cell_id(unit.semantic_position, self.cell_size))
                .or_default()
                .push((unit.id, unit.semantic_position));
        }
    }

    pub fn nearby(&self, center: [f32; 3], radius: f32) -> Vec<Uuid> {
        let mut ids = Vec::new();
        let range = (radius / self.cell_size).ceil() as i32;
        let origin = cell_id(center, self.cell_size);
        for x in (origin.0 - range)..=(origin.0 + range) {
            for y in (origin.1 - range)..=(origin.1 + range) {
                for z in (origin.2 - range)..=(origin.2 + range) {
                    if let Some(candidates) = self.cells.get(&(x, y, z)) {
                        for (id, position) in candidates {
                            if euclidean_distance(*position, center) <= radius {
                                ids.push(*id);
                            }
                        }
                    }
                }
            }
        }
        ids
    }
}

pub fn force_directed_layout(units: &[Unit], config: &SemanticMapConfig) -> LayoutOutcome {
    let selected = select_layout_units(units, config.max_layout_units);
    if selected.len() < 2 {
        return LayoutOutcome {
            position_updates: Vec::new(),
            mean_displacement: 0.0,
            rolled_back: false,
        };
    }

    let mut graph = Graph::<Uuid, f32, Undirected>::new_undirected();
    let mut node_lookup = HashMap::new();
    let mut initial_positions = Vec::with_capacity(selected.len());
    let mut anchor_locks = Vec::with_capacity(selected.len());

    for unit in &selected {
        let node = graph.add_node(unit.id);
        node_lookup.insert(unit.id, node);
        initial_positions.push(unit.semantic_position);
        anchor_locks.push(unit.anchor_status);
    }

    for (index, unit) in selected.iter().enumerate() {
        let node = NodeIndex::new(index);
        for link in &unit.links {
            if let Some(&target) = node_lookup.get(&link.target_id) {
                if node.index() >= target.index() {
                    continue;
                }
                graph.update_edge(node, target, link.weight.max(0.12));
            }
        }
    }

    if graph.edge_count() == 0 {
        for index in 1..selected.len() {
            let lhs = NodeIndex::new(index - 1);
            let rhs = NodeIndex::new(index);
            graph.update_edge(lhs, rhs, 0.16);
        }
    }

    let mut positions = initial_positions.clone();
    let mut best_positions = initial_positions.clone();
    let mut best_energy = layout_energy(&graph, &positions);
    let initial_energy = best_energy;
    let mut previous_energy = best_energy;
    let mut increases = 0usize;
    let mut temperature = config.max_displacement_per_iteration;

    for _ in 0..config.max_layout_iterations {
        let mut displacements = vec![[0.0; 3]; positions.len()];
        let k = ideal_distance(positions.len(), config);

        for lhs in 0..positions.len() {
            for rhs in lhs + 1..positions.len() {
                let delta = subtract(positions[lhs], positions[rhs]);
                let distance = magnitude(delta).max(0.05);
                let direction = scale(delta, 1.0 / distance);
                let repulsive_force =
                    ((k * k) / distance) * config.repulsive_force_coefficient.max(0.0);
                let adjustment = scale(direction, repulsive_force);
                displacements[lhs] = add(displacements[lhs], adjustment);
                displacements[rhs] = subtract(displacements[rhs], adjustment);
            }
        }

        for edge in graph.edge_indices() {
            let Some((lhs, rhs)) = graph.edge_endpoints(edge) else {
                continue;
            };
            let weight = *graph.edge_weight(edge).unwrap_or(&0.15);
            let delta = subtract(positions[lhs.index()], positions[rhs.index()]);
            let distance = magnitude(delta).max(0.05);
            let direction = scale(delta, 1.0 / distance);
            let attractive_force = ((distance * distance) / k.max(0.1))
                * weight
                * config.attractive_force_coefficient.max(0.0);
            let adjustment = scale(direction, attractive_force);
            displacements[lhs.index()] = subtract(displacements[lhs.index()], adjustment);
            displacements[rhs.index()] = add(displacements[rhs.index()], adjustment);
        }

        for index in 0..positions.len() {
            if anchor_locks[index] {
                continue;
            }
            let displacement = displacements[index];
            let distance = magnitude(displacement).max(0.001);
            let capped = scale(displacement, (temperature / distance).min(1.0));
            positions[index] =
                clamp_position(add(positions[index], capped), config.layout_boundary);
        }

        let energy = layout_energy(&graph, &positions);
        if energy < best_energy {
            best_energy = energy;
            best_positions = positions.clone();
        }
        if energy > previous_energy {
            increases += 1;
            if increases >= config.energy_rollback_threshold as usize {
                let rolled_back = best_energy <= initial_energy;
                let final_positions = if rolled_back {
                    best_positions
                } else {
                    initial_positions
                };
                return build_layout_outcome(&selected, &final_positions, rolled_back);
            }
        } else {
            increases = 0;
        }
        previous_energy = energy;
        temperature *= 0.90;
        let mean_displacement = mean_displacement(&positions, &best_positions);
        if mean_displacement <= config.convergence_tolerance {
            break;
        }
    }

    build_layout_outcome(&selected, &best_positions, false)
}

pub fn centroid(positions: &[[f32; 3]]) -> [f32; 3] {
    if positions.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut total = [0.0, 0.0, 0.0];
    for position in positions {
        total[0] += position[0];
        total[1] += position[1];
        total[2] += position[2];
    }
    let count = positions.len() as f32;
    [total[0] / count, total[1] / count, total[2] / count]
}

fn select_layout_units(units: &[Unit], max_layout_units: usize) -> Vec<Unit> {
    let mut selected = units.to_vec();
    selected.sort_by(|lhs, rhs| {
        rhs.anchor_status
            .cmp(&lhs.anchor_status)
            .then((rhs.memory_type == MemoryType::Core).cmp(&(lhs.memory_type == MemoryType::Core)))
            .then(rhs.utility_score.total_cmp(&lhs.utility_score))
            .then(rhs.frequency.cmp(&lhs.frequency))
    });
    selected.truncate(max_layout_units.max(1));
    selected
}

fn build_layout_outcome(
    selected: &[Unit],
    final_positions: &[[f32; 3]],
    rolled_back: bool,
) -> LayoutOutcome {
    let mut position_updates = Vec::new();
    let mut total_displacement = 0.0;

    for (unit, position) in selected.iter().zip(final_positions.iter()) {
        let displacement = euclidean_distance(unit.semantic_position, *position);
        if displacement > 0.001 {
            total_displacement += displacement;
            position_updates.push((unit.id, *position));
        }
    }

    let mean_displacement = if position_updates.is_empty() {
        0.0
    } else {
        total_displacement / position_updates.len() as f32
    };

    LayoutOutcome {
        position_updates,
        mean_displacement,
        rolled_back,
    }
}

fn layout_energy(graph: &Graph<Uuid, f32, Undirected>, positions: &[[f32; 3]]) -> f32 {
    let mut energy = 0.0;
    for lhs in 0..positions.len() {
        for rhs in lhs + 1..positions.len() {
            let distance = euclidean_distance(positions[lhs], positions[rhs]).max(0.05);
            energy += 1.0 / distance;
        }
    }

    for edge in graph.edge_indices() {
        let Some((lhs, rhs)) = graph.edge_endpoints(edge) else {
            continue;
        };
        let weight = *graph.edge_weight(edge).unwrap_or(&0.15);
        let distance = euclidean_distance(positions[lhs.index()], positions[rhs.index()]);
        energy += distance * distance * weight;
    }

    energy
}

fn ideal_distance(node_count: usize, config: &SemanticMapConfig) -> f32 {
    let volume = (config.layout_boundary * 2.0).powi(3);
    ((volume / node_count.max(1) as f32).cbrt() * config.preferred_spacing_k.max(0.1))
        .clamp(2.0, 18.0)
}

fn add(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn subtract(lhs: [f32; 3], rhs: [f32; 3]) -> [f32; 3] {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn scale(vector: [f32; 3], factor: f32) -> [f32; 3] {
    [vector[0] * factor, vector[1] * factor, vector[2] * factor]
}

fn magnitude(vector: [f32; 3]) -> f32 {
    euclidean_distance(vector, [0.0, 0.0, 0.0])
}

fn euclidean_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    ((dx * dx) + (dy * dy) + (dz * dz)).sqrt()
}

fn clamp_position(position: [f32; 3], boundary: f32) -> [f32; 3] {
    [
        position[0].clamp(-boundary, boundary),
        position[1].clamp(-boundary, boundary),
        position[2].clamp(-boundary, boundary),
    ]
}

fn cell_id(position: [f32; 3], cell_size: f32) -> CellId {
    (
        (position[0] / cell_size).floor() as i32,
        (position[1] / cell_size).floor() as i32,
        (position[2] / cell_size).floor() as i32,
    )
}

fn mean_displacement(current: &[[f32; 3]], baseline: &[[f32; 3]]) -> f32 {
    if current.is_empty() || baseline.is_empty() {
        return 0.0;
    }
    let count = current.len().min(baseline.len());
    let total = (0..count)
        .map(|index| euclidean_distance(current[index], baseline[index]))
        .sum::<f32>();
    total / count as f32
}

#[cfg(test)]
mod tests {
    use super::{force_directed_layout, SpatialGrid};
    use crate::config::SemanticMapConfig;
    use crate::types::{Link, MemoryType, Unit, UnitLevel};
    use uuid::Uuid;

    #[test]
    fn anchor_positions_remain_locked_during_layout() {
        let anchor_id = Uuid::new_v4();
        let mut anchor = Unit::new(
            "anchor".to_string(),
            "anchor".to_string(),
            UnitLevel::Word,
            1.2,
            0.8,
            [0.0, 0.0, 0.0],
        );
        anchor.id = anchor_id;
        anchor.anchor_status = true;
        anchor.memory_type = MemoryType::Core;

        let mut movable = Unit::new(
            "movable".to_string(),
            "movable".to_string(),
            UnitLevel::Word,
            0.8,
            0.6,
            [8.0, 8.0, 8.0],
        );
        movable
            .links
            .push(Link::new(anchor_id, crate::types::EdgeType::Semantic, 0.8));

        let layout = force_directed_layout(
            &[anchor.clone(), movable.clone()],
            &SemanticMapConfig::default(),
        );
        assert!(layout
            .position_updates
            .iter()
            .all(|(id, _)| *id != anchor.id));
    }

    #[test]
    fn spatial_hash_returns_local_neighbors() {
        let near_id = Uuid::new_v4();
        let far_id = Uuid::new_v4();
        let mut near = Unit::new(
            "near".to_string(),
            "near".to_string(),
            UnitLevel::Word,
            0.8,
            0.7,
            [1.0, 1.0, 1.0],
        );
        near.id = near_id;
        let mut far = Unit::new(
            "far".to_string(),
            "far".to_string(),
            UnitLevel::Word,
            0.8,
            0.7,
            [32.0, 32.0, 32.0],
        );
        far.id = far_id;

        let mut grid = SpatialGrid::new(SemanticMapConfig::default().spatial_cell_size);
        grid.rebuild(&[near, far]);

        let nearby = grid.nearby([0.0, 0.0, 0.0], 6.0);
        assert!(nearby.contains(&near_id));
        assert!(!nearby.contains(&far_id));
    }
}
