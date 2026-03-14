// Distance Calculation Compute Shader
//
// Calculates Euclidean distances from a center point to all positions.

struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
    _padding: f32,
}

struct DistanceResult {
    distance: f32,
    index: u32,
    _padding: vec2<f32>,
}

struct Counts {
    position_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<Vec3>;
@group(0) @binding(1) var<uniform> center: Vec3;
@group(0) @binding(2) var<storage, read_write> results: array<DistanceResult>;
@group(0) @binding(3) var<uniform> counts: Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= counts.position_count) {
        return;
    }
    
    let pos = positions[idx];
    
    // Calculate Euclidean distance
    let dx = pos.x - center.x;
    let dy = pos.y - center.y;
    let dz = pos.z - center.z;
    let dist = sqrt(dx * dx + dy * dy + dz * dz);
    
    results[idx] = DistanceResult(dist, idx, vec2<f32>(0.0, 0.0));
}
