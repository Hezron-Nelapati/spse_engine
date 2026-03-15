// Attractive Force Compute Shader
//
// Calculates attractive forces along edges.
// O(edges) complexity - much faster than repulsive forces.

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge_idx = global_id.x;
    
    if (edge_idx >= config.edge_count) {
        return;
    }
    
    let edge = edges[edge_idx];
    let source_idx = edge.source_idx;
    let target_idx_val = edge.target_idx;
    
    if (source_idx >= config.node_count || target_idx_val >= config.node_count) {
        return;
    }
    
    var source_pos = positions[source_idx];
    var target_pos_val = positions[target_idx_val];
    
    // Calculate distance
    let dx = target_pos_val.x - source_pos.x;
    let dy = target_pos_val.y - source_pos.y;
    let dz = target_pos_val.z - source_pos.z;
    let dist = max(sqrt(dx * dx + dy * dy + dz * dz), 0.05);
    
    // Fruchterman-Reingold attractive force: (d² / k)
    let k = config.k;
    let force_mag = (dist * dist / k) * edge.weight * config.attractive_coeff;
    
    // Direction (normalized, towards each other)
    let dir = vec3<f32>(dx, dy, dz) / dist;
    let force = dir * force_mag;
    
    // Apply forces (if not anchors)
    let temperature = config.temperature;
    let boundary = config.boundary;
    
    // Source moves towards target
    if (source_pos.is_anchor == 0u) {
        let displacement = min(force_mag, temperature);
        var new_pos = vec3<f32>(
            source_pos.x + dir.x * displacement,
            source_pos.y + dir.y * displacement,
            source_pos.z + dir.z * displacement
        );
        new_pos = clamp(new_pos, vec3<f32>(-boundary), vec3<f32>(boundary));
        
        if (source_pos.is_process_unit == 1u) {
            new_pos.z = -1.0;
        }
        
        positions[source_idx] = Position(new_pos.x, new_pos.y, new_pos.z, 0u, source_pos.is_process_unit, 0u, 0u, 0u);
    }
    
    // Target moves towards source
    if (target_pos_val.is_anchor == 0u) {
        let displacement = min(force_mag, temperature);
        var new_pos = vec3<f32>(
            target_pos_val.x - dir.x * displacement,
            target_pos_val.y - dir.y * displacement,
            target_pos_val.z - dir.z * displacement
        );
        new_pos = clamp(new_pos, vec3<f32>(-boundary), vec3<f32>(boundary));
        
        if (target_pos_val.is_process_unit == 1u) {
            new_pos.z = -1.0;
        }
        
        positions[target_idx_val] = Position(new_pos.x, new_pos.y, new_pos.z, 0u, target_pos_val.is_process_unit, 0u, 0u, 0u);
    }
}
