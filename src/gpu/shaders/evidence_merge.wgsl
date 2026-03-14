// Evidence Merge Compute Shader
// Parallelizes document-context overlap detection

struct ContextCell {
    length: u32,
    _padding: vec3<u32>,
    content: array<u32, 128>,
}

struct Document {
    length: u32,
    trust_score: f32,
    _padding: vec2<u32>,
    content: array<u32, 2048>,
}

struct Counts {
    cell_count: u32,
    doc_count: u32,
    _padding: vec2<u32>,
}

struct OverlapResult {
    doc_index: u32,
    overlap_count: u32,
    overlap_score: f32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> cells: array<ContextCell, 64>;
@group(0) @binding(1) var<storage, read> documents: array<Document, 128>;
@group(0) @binding(2) var<uniform> counts: Counts;
@group(0) @binding(3) var<storage, read_write> output: array<OverlapResult, 128>;

// Check if cell content exists anywhere in document
fn cell_in_doc(cell: ptr<function, ContextCell>, doc: ptr<function, Document>) -> bool {
    let cell_len = (*cell).length;
    let doc_len = (*doc).length;
    
    if (cell_len == 0u || cell_len > doc_len) {
        return false;
    }
    
    // Sliding window search
    let max_pos = doc_len - cell_len + 1u;
    
    for (pos in 0u .. max_pos) {
        var match_found = true;
        
        for (i in 0u .. cell_len) {
            if ((*doc).content[pos + i] != (*cell).content[i]) {
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
    let doc_idx = global_id.x;
    
    if (doc_idx >= counts.doc_count) {
        return;
    }
    
    let doc = documents[doc_idx];
    var overlap_count: u32 = 0u;
    var overlap_score: f32 = 0.0;
    
    // Check each context cell against this document
    for (cell_idx in 0u .. counts.cell_count) {
        let cell = cells[cell_idx];
        
        if (cell_in_doc(&cell, &doc)) {
            overlap_count += 1u;
            // Weight by trust score
            overlap_score += doc.trust_score * 0.1;
        }
    }
    
    // Write output
    output[doc_idx].doc_index = doc_idx;
    output[doc_idx].overlap_count = overlap_count;
    output[doc_idx].overlap_score = overlap_score;
}
