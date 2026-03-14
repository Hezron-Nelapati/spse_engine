// Tone Detection Compute Shader
// Parallelizes keyword matching across 6 tone types

struct ToneInput {
    text_length: u32,
    exclamation_count: u32,
    _padding: vec2<u32>,
    text_data: array<u32, 1024>,
}

struct ToneKeywords {
    keyword_count: u32,
    _padding: vec3<u32>,
    keyword_meta: array<u32, 32>,  // (offset << 16 | length)
    keyword_data: array<u32, 256>,
}

struct ToneScore {
    tone_index: u32,
    score: f32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<uniform> input: ToneInput;
@group(0) @binding(1) var<storage, read> keywords: array<ToneKeywords, 6>;
@group(0) @binding(2) var<storage, read_write> output: array<ToneScore, 6>;

// Score increment per keyword match
const SCORE_INCREMENT_URGENCY: f32 = 0.15;
const SCORE_INCREMENT_SADNESS: f32 = 0.12;
const SCORE_INCREMENT_TECHNICAL: f32 = 0.08;
const SCORE_INCREMENT_CASUAL: f32 = 0.10;
const SCORE_INCREMENT_FORMAL: f32 = 0.10;
const SCORE_INCREMENT_EMPATHETIC: f32 = 0.10;

// Check if keyword exists in text at position
fn keyword_matches_at(text: ptr<function, array<u32, 1024>>, text_len: u32, kw_data: ptr<function, array<u32, 256>>, kw_offset: u32, kw_len: u32) -> bool {
    if (kw_len == 0u) {
        return false;
    }
    
    // Scan through text looking for keyword
    let max_pos = text_len - kw_len + 1u;
    
    for (pos in 0u .. max_pos) {
        var match_found = true;
        
        for (i in 0u .. kw_len) {
            let text_char = (*text)[pos + i];
            let kw_char = (*kw_data)[kw_offset + i];
            
            // Case-insensitive comparison (lowercase only)
            if (text_char != kw_char) {
                match_found = false;
                break;
            }
        }
        
        if (match_found) {
            return true;
        }
    }
    
    return false;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= 6u) {
        return;
    }
    
    var score: f32 = 0.0;
    let kw_set = keywords[idx];
    
    // Get score increment based on tone type
    let increment = select(
        select(
            select(
                select(
                    select(SCORE_INCREMENT_URGENCY, SCORE_INCREMENT_SADNESS, idx == 1u),
                    SCORE_INCREMENT_TECHNICAL, idx == 2u
                ),
                SCORE_INCREMENT_CASUAL, idx == 3u
            ),
            SCORE_INCREMENT_FORMAL, idx == 4u
        ),
        SCORE_INCREMENT_EMPATHETIC, idx == 5u
    );
    
    // Check each keyword
    for (i in 0u .. kw_set.keyword_count) {
        let meta = kw_set.keyword_meta[i];
        let kw_offset = meta >> 16u;
        let kw_len = meta & 0xFFFFu;
        
        if (keyword_matches_at(&input.text_data, input.text_length, &kw_set.keyword_data, kw_offset, kw_len)) {
            score += increment;
        }
    }
    
    // Special handling for urgency: add exclamation mark bonus
    if (idx == 0u) {
        score += f32(input.exclamation_count) * 0.1;
    }
    
    // Clamp score to [0, 1]
    score = clamp(score, 0.0, 1.0);
    
    // Write output
    output[idx].tone_index = idx;
    output[idx].score = score;
}
