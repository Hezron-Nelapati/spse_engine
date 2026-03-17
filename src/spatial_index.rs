use crate::config::SemanticMapConfig;
use crate::types::{MemoryType, Unit};
use petgraph::graph::NodeIndex;
use petgraph::{Graph, Undirected};
use rayon::prelude::*;
use std::collections::HashMap;
use uuid::Uuid;

type CellId = (i32, i32, i32);

#[derive(Clone)]
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

    /// Insert a unit with given position into the spatial grid.
    pub fn insert(&mut self, id: Uuid, position: &[f32; 3]) {
        let cell = cell_id(*position, self.cell_size);
        self.cells.entry(cell).or_default().push((id, *position));
    }

    /// Attract a unit toward a target position (Layer 18 feedback for correct predictions).
    /// Moves the unit closer to the target by a fraction of the distance.
    pub fn attract(&mut self, id: Uuid, target: &[f32; 3], strength: f32) {
        for candidates in self.cells.values_mut() {
            for (unit_id, position) in candidates.iter_mut() {
                if *unit_id == id {
                    // Move toward target by strength fraction
                    let delta = [
                        target[0] - position[0],
                        target[1] - position[1],
                        target[2] - position[2],
                    ];
                    position[0] += delta[0] * strength;
                    position[1] += delta[1] * strength;
                    position[2] += delta[2] * strength;
                    return;
                }
            }
        }
    }

    /// Repel a unit away from a source position (Layer 18 feedback for incorrect predictions).
    /// Moves the unit away from the source by a fraction of the distance.
    pub fn repel(&mut self, id: Uuid, source: &[f32; 3], strength: f32) {
        for candidates in self.cells.values_mut() {
            for (unit_id, position) in candidates.iter_mut() {
                if *unit_id == id {
                    // Move away from source by strength fraction
                    let delta = [
                        position[0] - source[0],
                        position[1] - source[1],
                        position[2] - source[2],
                    ];
                    let distance = magnitude(delta).max(0.001);
                    let direction = [
                        delta[0] / distance,
                        delta[1] / distance,
                        delta[2] / distance,
                    ];
                    position[0] += direction[0] * strength;
                    position[1] += direction[1] * strength;
                    position[2] += direction[2] * strength;
                    return;
                }
            }
        }
    }
}

pub fn force_directed_layout(units: &[Unit], config: &SemanticMapConfig) -> LayoutOutcome {
    // Try GPU acceleration first if available and beneficial
    #[cfg(feature = "gpu")]
    {
        if units.len() >= 100 && crate::gpu::is_gpu_available() {
            if let Some(outcome) = force_directed_layout_gpu(units, config) {
                return LayoutOutcome {
                    position_updates: outcome.position_updates,
                    mean_displacement: outcome.mean_displacement,
                    rolled_back: outcome.rolled_back,
                };
            }
        }
    }

    // CPU fallback
    force_directed_layout_cpu(units, config)
}

/// GPU-accelerated force-directed layout (requires gpu feature)
#[cfg(feature = "gpu")]
fn force_directed_layout_gpu(
    units: &[Unit],
    config: &SemanticMapConfig,
) -> Option<crate::gpu::compute::force_layout::GpuLayoutOutcome> {
    use crate::gpu::compute::force_layout::force_layout_gpu;
    force_layout_gpu(units, config)
}

/// CPU implementation of force-directed layout
fn force_directed_layout_cpu(units: &[Unit], config: &SemanticMapConfig) -> LayoutOutcome {
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
    let mut process_flags = Vec::with_capacity(selected.len());

    for unit in &selected {
        let node = graph.add_node(unit.id);
        node_lookup.insert(unit.id, node);
        initial_positions.push(unit.semantic_position);
        anchor_locks.push(unit.anchor_status);
        process_flags.push(unit.is_process_unit);
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
        let k = ideal_distance(positions.len(), config);
        let repulsive_coeff = config.repulsive_force_coefficient.max(0.0);

        // Parallelize O(n²) repulsion: each row computes its net displacement independently
        let n = positions.len();
        let displacements: Vec<[f32; 3]> = if n > 64 {
            (0..n)
                .into_par_iter()
                .map(|lhs| {
                    let mut disp = [0.0f32; 3];
                    for rhs in 0..n {
                        if lhs == rhs {
                            continue;
                        }
                        let delta = subtract(positions[lhs], positions[rhs]);
                        let distance = magnitude(delta).max(0.05);
                        let direction = scale(delta, 1.0 / distance);
                        let multiplier = if process_flags[lhs] != process_flags[rhs] {
                            3.0
                        } else {
                            1.0
                        };
                        let repulsive_force = ((k * k) / distance) * repulsive_coeff * multiplier;
                        disp = add(disp, scale(direction, repulsive_force));
                    }
                    disp
                })
                .collect()
        } else {
            // Small n: sequential is faster (no thread overhead)
            let mut displacements = vec![[0.0; 3]; n];
            for lhs in 0..n {
                for rhs in lhs + 1..n {
                    let delta = subtract(positions[lhs], positions[rhs]);
                    let distance = magnitude(delta).max(0.05);
                    let direction = scale(delta, 1.0 / distance);
                    let multiplier = if process_flags[lhs] != process_flags[rhs] {
                        3.0
                    } else {
                        1.0
                    };
                    let repulsive_force = ((k * k) / distance) * repulsive_coeff * multiplier;
                    let adjustment = scale(direction, repulsive_force);
                    displacements[lhs] = add(displacements[lhs], adjustment);
                    displacements[rhs] = subtract(displacements[rhs], adjustment);
                }
            }
            displacements
        };
        let mut displacements = displacements;

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

            // Confine process units to Z = -1.0 subspace
            if process_flags[index] {
                positions[index][2] = -1.0;
            }
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
    // Collect indices with sort key to avoid cloning all units
    let mut indexed: Vec<(usize, (bool, bool, f32, u64))> = units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            (
                i,
                (
                    u.anchor_status,
                    u.memory_type == MemoryType::Core,
                    u.utility_score,
                    u.frequency,
                ),
            )
        })
        .collect();

    indexed.sort_by(|lhs, rhs| {
        rhs.1
             .0
            .cmp(&lhs.1 .0) // anchor_status desc
            .then(rhs.1 .1.cmp(&lhs.1 .1)) // is_core desc
            .then(rhs.1 .2.total_cmp(&lhs.1 .2)) // utility_score desc
            .then(rhs.1 .3.cmp(&lhs.1 .3)) // frequency desc
    });

    indexed
        .into_iter()
        .take(max_layout_units.max(1))
        .filter_map(|(i, _)| units.get(i).cloned())
        .collect()
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
