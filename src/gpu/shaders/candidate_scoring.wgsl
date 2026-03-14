// Candidate Scoring Compute Shader
// 
// Parallelizes dot product scoring across candidates.
// Each work item processes one candidate.

struct Candidate {
    spatial_fit: f32,
    context_fit: f32,
    sequence_fit: f32,
    transition_fit: f32,
    utility_fit: f32,
    confidence_fit: f32,
    evidence_support: f32,
    _padding: f32,
    unit_id_low: u32,
    unit_id_high: u32,
}

struct Weights {
    spatial: f32,
    context: f32,
    sequence: f32,
    transition: f32,
    utility: f32,
    confidence: f32,
    evidence: f32,
    freshness_boost: f32,
}

struct ScoreResult {
    score: f32,
    unit_id_low: u32,
    unit_id_high: u32,
    _padding: u32,
}

struct Counts {
    candidate_count: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> candidates: array<Candidate>;
@group(0) @binding(1) var<storage, read> weights: Weights;
@group(0) @binding(2) var<storage, read_write> results: array<ScoreResult>;
@group(0) @binding(3) var<uniform> counts: Counts;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= counts.candidate_count) {
        return;
    }
    
    let candidate = candidates[idx];
    
    // Compute weighted dot product
    let score = weights.spatial * candidate.spatial_fit
             + weights.context * candidate.context_fit
             + weights.sequence * candidate.sequence_fit
             + weights.transition * candidate.transition_fit
             + weights.utility * candidate.utility_fit
             + weights.confidence * candidate.confidence_fit
             + weights.evidence * candidate.evidence_support
             + weights.freshness_boost;
    
    results[idx] = ScoreResult(
        score,
        candidate.unit_id_low,
        candidate.unit_id_high,
        0u
    );
}
