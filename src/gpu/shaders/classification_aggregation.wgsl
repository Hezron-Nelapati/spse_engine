// Classification Vote Aggregation Compute Shader
// Aggregates similarity scores into final intent/tone/resolver votes

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

struct GpuVoteAggregation {
    intent_scores: array<f32, 23>,  // 23 intent kinds
    tone_scores: array<f32, 8>,      // 8 tone kinds
    resolver_scores: array<f32, 3>,  // 3 resolver modes
    best_intent: u32,
    best_tone: u32,
    best_resolver: u32,
    confidence: f32,
    candidate_count: u32,
}

struct Config {
    min_similarity_threshold: f32,
    low_confidence_threshold: f32,
    high_confidence_threshold: f32,
    pattern_merge_threshold: f32,
    pattern_count: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> query_sig: GpuSignature;  // Not used in aggregation
@group(0) @binding(1) var<storage, read> patterns: array<GpuPattern>;  // Not used
@group(0) @binding(2) var<storage, read_write> results: array<GpuSimilarityResult>;
@group(0) @binding(3) var<uniform> config: Config;

var<workgroup> shared_intent_scores: array<f32, 23>;
var<workgroup> shared_tone_scores: array<f32, 8>;
var<workgroup> shared_resolver_scores: array<f32, 3>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let local_idx = local_id.x;
    
    // Initialize shared memory
    if (local_idx < 23u) {
        shared_intent_scores[local_idx] = 0.0;
    }
    if (local_idx < 8u) {
        shared_tone_scores[local_idx] = 0.0;
    }
    if (local_idx < 3u) {
        shared_resolver_scores[local_idx] = 0.0;
    }
    
    workgroupBarrier();
    
    // Each thread accumulates scores for patterns above threshold
    let idx = global_id.x;
    if (idx < config.pattern_count) {
        let result = results[idx];
        if (result.final_score >= config.min_similarity_threshold) {
            // Accumulate intent score
            let intent_idx = result.intent_index;
            if (intent_idx < 23u) {
                shared_intent_scores[intent_idx] = shared_intent_scores[intent_idx] + result.final_score;
            }
            
            // Accumulate tone score
            let tone_idx = result.tone_index;
            if (tone_idx < 8u) {
                shared_tone_scores[tone_idx] = shared_tone_scores[tone_idx] + result.final_score;
            }
            
            // Accumulate resolver score
            let resolver_idx = result.resolver_index;
            if (resolver_idx < 3u) {
                shared_resolver_scores[resolver_idx] = shared_resolver_scores[resolver_idx] + result.final_score;
            }
        }
    }
    
    workgroupBarrier();
    
    // Thread 0 finds best scores and writes result
    if (local_idx == 0u) {
        var best_intent: u32 = 0u;
        var best_intent_score: f32 = 0.0;
        for (var i: u32 = 0u; i < 23u; i = i + 1u) {
            if (shared_intent_scores[i] > best_intent_score) {
                best_intent_score = shared_intent_scores[i];
                best_intent = i;
            }
        }
        
        var best_tone: u32 = 0u;
        var best_tone_score: f32 = 0.0;
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            if (shared_tone_scores[i] > best_tone_score) {
                best_tone_score = shared_tone_scores[i];
                best_tone = i;
            }
        }
        
        var best_resolver: u32 = 0u;
        var best_resolver_score: f32 = 0.0;
        for (var i: u32 = 0u; i < 3u; i = i + 1u) {
            if (shared_resolver_scores[i] > best_resolver_score) {
                best_resolver_score = shared_resolver_scores[i];
                best_resolver = i;
            }
        }
        
        let confidence = (best_intent_score + best_tone_score) / 2.0;
        
        // Apply confidence-driven resolver override
        var final_resolver = best_resolver;
        if (confidence < config.low_confidence_threshold) {
            final_resolver = 2u;  // Exploratory
        } else if (confidence > config.high_confidence_threshold && best_resolver == 1u) {
            final_resolver = 0u;  // Deterministic
        }
        
        // Write result to first result slot
        results[0].pattern_index = 0u;
        results[0].similarity = confidence;
        results[0].final_score = confidence;
        results[0].intent_index = best_intent;
        results[0].tone_index = best_tone;
        results[0].resolver_index = final_resolver;
    }
}

