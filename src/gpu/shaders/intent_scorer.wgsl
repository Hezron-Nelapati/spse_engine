// Intent Scoring Compute Shader
// Parallelizes scoring across 23 intent types

struct IntentInput {
    token_count: u32,
    has_temporal: u32,
    has_domain_hints: u32,
    has_preference: u32,
    references_doc: u32,
    wants_brief: u32,
    certainty_bias: f32,
    turn_index: u32,
    _padding: vec3<u32>,
}

struct IntentScore {
    intent_index: u32,
    score: f32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> input: IntentInput;
@group(0) @binding(1) var<storage, read_write> output: array<IntentScore, 23>;

// Intent scoring weights (derived from CPU implementation)
const GREETING_WEIGHT: f32 = 0.95;
const GRATITUDE_WEIGHT: f32 = 0.90;
const FAREWELL_WEIGHT: f32 = 0.96;
const HELP_WEIGHT: f32 = 0.85;
const CLARIFY_WEIGHT: f32 = 0.50;
const REWRITE_WEIGHT: f32 = 0.55;
const VERIFY_WEIGHT: f32 = 0.60;
const CONTINUE_WEIGHT: f32 = 0.55;
const FORGET_WEIGHT: f32 = 0.70;
const SUMMARIZE_WEIGHT: f32 = 0.65;
const EXPLAIN_WEIGHT: f32 = 0.70;
const COMPARE_WEIGHT: f32 = 0.65;
const EXTRACT_WEIGHT: f32 = 0.60;
const ANALYZE_WEIGHT: f32 = 0.70;
const PLAN_WEIGHT: f32 = 0.65;
const ACT_WEIGHT: f32 = 0.60;
const RECOMMEND_WEIGHT: f32 = 0.60;
const CLASSIFY_WEIGHT: f32 = 0.55;
const TRANSLATE_WEIGHT: f32 = 0.75;
const DEBUG_WEIGHT: f32 = 0.70;
const CRITIQUE_WEIGHT: f32 = 0.65;
const BRAINSTORM_WEIGHT: f32 = 0.60;
const QUESTION_WEIGHT: f32 = 0.50;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= 23u) {
        return;
    }
    
    var score: f32 = 0.0;
    
    // Base scoring based on intent index
    // 0: Greeting
    if (idx == 0u) {
        score = GREETING_WEIGHT;
        if (input.token_count < 3u) {
            score += 0.15;
        }
    }
    // 1: Gratitude
    else if (idx == 1u) {
        score = GRATITUDE_WEIGHT;
    }
    // 2: Farewell
    else if (idx == 2u) {
        score = FAREWELL_WEIGHT;
    }
    // 3: Help
    else if (idx == 3u) {
        score = HELP_WEIGHT;
        if (input.token_count < 4u) {
            score += 0.10;
        }
    }
    // 4: Clarify
    else if (idx == 4u) {
        score = CLARIFY_WEIGHT;
        if (input.has_domain_hints == 1u) {
            score += 0.15;
        }
    }
    // 5: Rewrite
    else if (idx == 5u) {
        score = REWRITE_WEIGHT;
        if (input.has_preference == 1u) {
            score += 0.20;
        }
    }
    // 6: Verify
    else if (idx == 6u) {
        score = VERIFY_WEIGHT;
        if (input.has_temporal == 1u) {
            score += 0.15;
        }
        if (input.references_doc == 1u) {
            score += 0.20;
        }
    }
    // 7: Continue
    else if (idx == 7u) {
        score = CONTINUE_WEIGHT;
        if (input.turn_index > 0u) {
            score += 0.25;
        }
        if (input.has_preference == 1u) {
            score += 0.15;
        }
    }
    // 8: Forget
    else if (idx == 8u) {
        score = FORGET_WEIGHT;
    }
    // 9: Summarize
    else if (idx == 9u) {
        score = SUMMARIZE_WEIGHT;
        if (input.wants_brief == 1u) {
            score += 0.25;
        }
        if (input.references_doc == 1u) {
            score += 0.15;
        }
    }
    // 10: Explain
    else if (idx == 10u) {
        score = EXPLAIN_WEIGHT;
        if (input.has_domain_hints == 1u) {
            score += 0.15;
        }
    }
    // 11: Compare
    else if (idx == 11u) {
        score = COMPARE_WEIGHT;
        if (input.token_count > 5u) {
            score += 0.10;
        }
    }
    // 12: Extract
    else if (idx == 12u) {
        score = EXTRACT_WEIGHT;
        if (input.references_doc == 1u) {
            score += 0.20;
        }
    }
    // 13: Analyze
    else if (idx == 13u) {
        score = ANALYZE_WEIGHT;
        if (input.token_count > 6u) {
            score += 0.10;
        }
    }
    // 14: Plan
    else if (idx == 14u) {
        score = PLAN_WEIGHT;
        if (input.token_count > 4u) {
            score += 0.10;
        }
    }
    // 15: Act
    else if (idx == 15u) {
        score = ACT_WEIGHT;
        if (input.references_doc == 1u) {
            score += 0.15;
        }
    }
    // 16: Recommend
    else if (idx == 16u) {
        score = RECOMMEND_WEIGHT;
        if (input.has_preference == 1u) {
            score += 0.20;
        }
    }
    // 17: Classify
    else if (idx == 17u) {
        score = CLASSIFY_WEIGHT;
        if (input.wants_brief == 1u) {
            score += 0.15;
        }
    }
    // 18: Translate
    else if (idx == 18u) {
        score = TRANSLATE_WEIGHT;
    }
    // 19: Debug
    else if (idx == 19u) {
        score = DEBUG_WEIGHT;
        if (input.has_domain_hints == 1u) {
            score += 0.15;
        }
    }
    // 20: Critique
    else if (idx == 20u) {
        score = CRITIQUE_WEIGHT;
    }
    // 21: Brainstorm
    else if (idx == 21u) {
        score = BRAINSTORM_WEIGHT;
    }
    // 22: Question
    else if (idx == 22u) {
        score = QUESTION_WEIGHT;
        if (input.token_count > 2u) {
            score += 0.15;
        }
        if (input.certainty_bias < 0.0) {
            score += abs(input.certainty_bias) * 0.25;
        }
    }
    
    // Clamp score to [0, 1]
    score = clamp(score, 0.0, 1.0);
    
    // Write output
    output[idx].intent_index = idx;
    output[idx].score = score;
}
