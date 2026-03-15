// Classification Similarity Compute Shader
// Computes cosine similarity between query signature and all candidate patterns

struct GpuSignature {
    structure_scores: vec4<f32>,
    punctuation_scores: vec4<f32>,
    semantic_centroid: vec3<f32>,
    derived_scores: vec3<f32>,
}

struct GpuPattern {
    signature: GpuSignature,
    intent_index: u32,
    tone_index: u32,
    resolver_index: u32,
    confidence: f32,
    success_count: u32,
    failure_count: u32,
    _padding: vec2<u32>,
}

struct GpuSimilarityResult {
    pattern_index: u32,
    similarity: f32,
    final_score: f32,
    intent_index: u32,
    tone_index: u32,
    resolver_index: u32,
    _padding: u32,
}

struct Config {
    min_similarity_threshold: f32,
    low_confidence_threshold: f32,
    high_confidence_threshold: f32,
    pattern_merge_threshold: f32,
    pattern_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> query_sig: GpuSignature;
@group(0) @binding(1) var<storage, read> patterns: array<GpuPattern>;
@group(0) @binding(2) var<storage, read_write> results: array<GpuSimilarityResult>;
@group(0) @binding(3) var<uniform> config: Config;

// Convert signature to 14-element feature vector
fn to_feature_vector(sig: GpuSignature) -> array<f32, 14> {
    var v: array<f32, 14>;
    v[0] = sig.structure_scores.x;
    v[1] = sig.structure_scores.y;
    v[2] = sig.structure_scores.z;
    v[3] = sig.structure_scores.w;
    v[4] = sig.punctuation_scores.x;
    v[5] = sig.punctuation_scores.y;
    v[6] = sig.punctuation_scores.z;
    v[7] = sig.punctuation_scores.w;
    v[8] = sig.semantic_centroid.x;
    v[9] = sig.semantic_centroid.y;
    v[10] = sig.semantic_centroid.z;
    v[11] = sig.derived_scores.x;
    v[12] = sig.derived_scores.y;
    v[13] = sig.derived_scores.z;
    return v;
}

// Compute cosine similarity between two signatures
fn cosine_similarity(a: GpuSignature, b: GpuSignature) -> f32 {
    let va = to_feature_vector(a);
    let vb = to_feature_vector(b);
    
    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    
    for (var i: u32 = 0u; i < 14u; i = i + 1u) {
        dot_product = dot_product + va[i] * vb[i];
        norm_a = norm_a + va[i] * va[i];
        norm_b = norm_b + vb[i] * vb[i];
    }
    
    let denom = sqrt(norm_a) * sqrt(norm_b);
    if (denom < 0.0001) {
        return 0.0;
    }
    return dot_product / denom;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= config.pattern_count) {
        return;
    }
    
    let pattern = patterns[idx];
    let similarity = cosine_similarity(query_sig, pattern.signature);
    let final_score = similarity * pattern.confidence;
    
    results[idx].pattern_index = idx;
    results[idx].similarity = similarity;
    results[idx].final_score = final_score;
    results[idx].intent_index = pattern.intent_index;
    results[idx].tone_index = pattern.tone_index;
    results[idx].resolver_index = pattern.resolver_index;
}
