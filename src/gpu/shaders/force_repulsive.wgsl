// Repulsive Force Compute Shader
//
// Calculates O(n²) repulsive forces between all node pairs.
// Uses tiled approach for better cache utilization.

struct Position {
    x: f32,
    y: f32,
    z: f32,
    is_anchor: u32,
    is_process_unit: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

struct Force {
    fx: f32,
    fy: f32,
    fz: f32,
    _padding: f32,
}

struct Edge {
    source_idx: u32,
    target_idx: u32,
    weight: f32,
    _padding: f32,
}

struct Config {
    k: f32,
    repulsive_coeff: f32,
    attractive_coeff: f32,
    temperature: f32,
    max_displacement: f32,
    boundary: f32,
    node_count: u32,
    edge_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read_write> positions: array<Position>;
@group(0) @binding(1) var<storage, read_write> forces: array<Force>;
@group(0) @binding(2) var<storage, read> edges: array<Edge>;
@group(0) @binding(3) var<uniform> config: Config;

var<workgroup> shared_pos: array<Position, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let node_count = config.node_count;
    let my_idx = global_id.x;
    
    if (my_idx >= node_count) {
        return;
    }
    
    var my_pos = positions[my_idx];
    var my_force = vec3<f32>(0.0, 0.0, 0.0);
    
    // Process all other nodes in tiles
    let num_tiles = (node_count + 63u) / 64u;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        // Load tile into shared memory
        let tile_start = tile * 64u;
        let tile_idx = tile_start + local_id.x;
        
        if (tile_idx < node_count) {
            shared_pos[local_id.x] = positions[tile_idx];
        } else {
            shared_pos[local_id.x] = Position(0.0, 0.0, 0.0, 1u, 0u, 0u, 0u, 0u); // anchor, skip
        }
        
        workgroupBarrier();
        
        // Compute forces with this tile
        for (var j = 0u; j < 64u; j++) {
            let other_idx = tile_start + j;
            
            // Skip self and anchors
            if (other_idx == my_idx || other_idx >= node_count) {
                continue;
            }
            
            let other_pos = shared_pos[j];
            if (other_pos.is_anchor == 1u) {
                continue;
            }
            
            // Calculate repulsive force
            let dx = my_pos.x - other_pos.x;
            let dy = my_pos.y - other_pos.y;
            let dz = my_pos.z - other_pos.z;
            let dist_sq = dx * dx + dy * dy + dz * dz;
            let dist = max(sqrt(dist_sq), 0.05);
            
            // Fruchterman-Reingold repulsive force: (k² / d)
            let k = config.k;
            
            // Process units vs Content units get 3x repulsion to prevent drift
            var multiplier = 1.0;
            if (my_pos.is_process_unit != other_pos.is_process_unit) {
                multiplier = 3.0;
            }
            
            let force_mag = (k * k / dist) * config.repulsive_coeff * multiplier;
            
            // Direction (normalized)
            let dir = vec3<f32>(dx, dy, dz) / dist;
            
            my_force += dir * force_mag;
        }
        
        workgroupBarrier();
    }
    
    // Apply force to position (if not anchor)
    if (my_pos.is_anchor == 0u) {
        let force_mag = length(my_force);
        if (force_mag > 0.001) {
            let displacement = my_force / force_mag * min(force_mag, config.temperature);
            
            var new_pos = vec3<f32>(
                my_pos.x + displacement.x,
                my_pos.y + displacement.y,
                my_pos.z + displacement.z
            );
            
            // Clamp to boundary
            let boundary = config.boundary;
            new_pos = clamp(new_pos, vec3<f32>(-boundary), vec3<f32>(boundary));
            
            // Confine process units to Z = -1.0 subspace
            if (my_pos.is_process_unit == 1u) {
                new_pos.z = -1.0;
            }
            
            positions[my_idx] = Position(new_pos.x, new_pos.y, new_pos.z, 0u, my_pos.is_process_unit, 0u, 0u, 0u);
        }
    }
    
    // Store force for next pass
    forces[my_idx] = Force(my_force.x, my_force.y, my_force.z, 0.0);
}
