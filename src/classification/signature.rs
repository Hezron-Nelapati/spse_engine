//! Lightweight feature signature for classification.
//!
//! ClassificationSignature is a CPU-efficient feature vector (<20 floats) that captures
//! structural, punctuation, semantic, and derived features for text classification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lightweight feature vector for CPU-efficient classification.
/// Total: 14 floats for cosine similarity computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClassificationSignature {
    // Structural features (Layer 2 alignment)
    /// Normalized byte length (0.0 - 1.0)
    pub byte_length_norm: f32,
    /// Sentence entropy measure (complexity)
    pub sentence_entropy: f32,
    /// Normalized token count (0.0 - 1.0)
    pub token_count_norm: f32,
    
    // Punctuation vector (normalized ratios)
    /// [question_mark_ratio, exclamation_ratio, period_ratio]
    pub punct_vector: [f32; 3],
    
    // Semantic anchor (Layer 5 position from SemanticHasher)
    /// 3D spatial position for retrieval
    pub semantic_centroid: [f32; 3],
    
    // Derived scores (lightweight calculation)
    /// Urgency signal (0.0 - 1.0)
    pub urgency_score: f32,
    /// Formality signal (0.0 - 1.0)
    pub formality_score: f32,
    /// Technical domain signal (0.0 - 1.0)
    pub technical_score: f32,
    
    // Context indicators
    /// Encoded domain signal
    pub domain_hint: f32,
    /// Time-related signal
    pub temporal_cue: f32,
}

impl ClassificationSignature {
    /// Convert to feature vector for cosine similarity.
    /// Returns 14-element vector.
    pub fn to_feature_vector(&self) -> Vec<f32> {
        vec![
            self.byte_length_norm,
            self.sentence_entropy,
            self.token_count_norm,
            self.punct_vector[0],
            self.punct_vector[1],
            self.punct_vector[2],
            self.semantic_centroid[0],
            self.semantic_centroid[1],
            self.semantic_centroid[2],
            self.urgency_score,
            self.formality_score,
            self.technical_score,
            self.domain_hint,
            self.temporal_cue,
        ]
    }
    
    /// Compute signature from raw text using provided hasher.
    /// Runs in microseconds - no heavy NLP.
    pub fn compute(text: &str, hasher: &SemanticHasher) -> Self {
        let normalized = normalize_text(text);
        let tokens = tokenize(&normalized);
        
        Self {
            byte_length_norm: normalize_byte_length(text.len()),
            sentence_entropy: compute_sentence_entropy(&normalized),
            token_count_norm: normalize_token_count(tokens.len()),
            punct_vector: compute_punct_vector(text),
            semantic_centroid: hasher.hash(&normalized),
            urgency_score: compute_urgency(&normalized),
            formality_score: compute_formality(&normalized),
            technical_score: compute_technical(&normalized),
            domain_hint: compute_domain_hint(&normalized),
            temporal_cue: compute_temporal_cue(&normalized),
        }
    }
    
    /// Compute signature hash for deduplication.
    pub fn signature_hash(&self) -> u64 {
        let mut hash: u64 = 0;
        for (i, feature) in self.to_feature_vector().iter().enumerate() {
            hash = hash.wrapping_add((feature.to_bits() as u64).wrapping_mul((i + 1) as u64));
        }
        hash
    }
    
    /// Generate 8-character hex hash for pattern markers.
    pub fn signature_hash_8char(&self) -> String {
        format!("{:08x}", self.signature_hash() & 0xFFFFFFFF)
    }
}

impl Default for ClassificationSignature {
    fn default() -> Self {
        Self {
            byte_length_norm: 0.0,
            sentence_entropy: 0.0,
            token_count_norm: 0.0,
            punct_vector: [0.0; 3],
            semantic_centroid: [0.5; 3],
            urgency_score: 0.0,
            formality_score: 0.5,
            technical_score: 0.0,
            domain_hint: 0.0,
            temporal_cue: 0.0,
        }
    }
}

/// Lightweight semantic hasher for text → 3D coordinate mapping.
/// Uses trigram frequency hashing - no heavy NLP models.
#[derive(Debug, Clone, Default)]
pub struct SemanticHasher {
    /// Pre-computed anchors for common domains
    domain_anchors: HashMap<String, [f32; 3]>,
}

impl SemanticHasher {
    pub fn new() -> Self {
        let mut domain_anchors = HashMap::new();
        
        // Domain-specific semantic anchors
        domain_anchors.insert("technology".to_string(), [0.7, 0.3, 0.5]);
        domain_anchors.insert("finance".to_string(), [0.3, 0.7, 0.4]);
        domain_anchors.insert("healthcare".to_string(), [0.4, 0.5, 0.8]);
        domain_anchors.insert("research".to_string(), [0.6, 0.6, 0.6]);
        domain_anchors.insert("operations".to_string(), [0.5, 0.4, 0.3]);
        domain_anchors.insert("governance".to_string(), [0.2, 0.8, 0.6]);
        domain_anchors.insert("marketing".to_string(), [0.8, 0.2, 0.4]);
        domain_anchors.insert("analytics".to_string(), [0.55, 0.55, 0.55]);
        
        Self { domain_anchors }
    }
    
    /// Hash text to 3D coordinate (microseconds).
    pub fn hash(&self, text: &str) -> [f32; 3] {
        let trigram_freq = self.compute_trigram_frequencies(text);
        
        // Map to 3D using prime-based hashing
        let x = self.hash_dimension(&trigram_freq, 31);
        let y = self.hash_dimension(&trigram_freq, 37);
        let z = self.hash_dimension(&trigram_freq, 41);
        
        // Apply domain anchor if detected
        if let Some(domain) = self.detect_domain(text) {
            if let Some(&anchor) = self.domain_anchors.get(&domain) {
                // Blend with anchor (80% hash, 20% anchor)
                return [
                    x * 0.8 + anchor[0] * 0.2,
                    y * 0.8 + anchor[1] * 0.2,
                    z * 0.8 + anchor[2] * 0.2,
                ];
            }
        }
        
        [x, y, z]
    }
    
    fn hash_dimension(&self, freq: &HashMap<String, f32>, prime: u32) -> f32 {
        let mut sum = 0.0f32;
        for (trigram, count) in freq {
            let hash = self.simple_hash(trigram, prime);
            sum += hash * count;
        }
        // Normalize to [0, 1]
        let normalized = (sum.fract() + 1.0) / 2.0;
        normalized.clamp(0.0, 1.0)
    }
    
    fn simple_hash(&self, s: &str, prime: u32) -> f32 {
        let mut hash = 0u32;
        for c in s.chars() {
            hash = hash.wrapping_mul(prime).wrapping_add(c as u32);
        }
        hash as f32 / u32::MAX as f32
    }
    
    fn compute_trigram_frequencies(&self, text: &str) -> HashMap<String, f32> {
        let chars: Vec<char> = text.chars().collect();
        let mut freq = HashMap::new();
        
        if chars.len() < 3 {
            return freq;
        }
        
        for i in 0..(chars.len() - 2) {
            let trigram: String = chars[i..i+3].iter().collect();
            *freq.entry(trigram).or_insert(0.0) += 1.0;
        }
        
        // Normalize frequencies
        let total = freq.values().sum::<f32>();
        if total > 0.0 {
            for count in freq.values_mut() {
                *count /= total;
            }
        }
        
        freq
    }
    
    fn detect_domain(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        
        // Simple keyword-based domain detection
        let domain_keywords = [
            ("technology", vec!["software", "api", "code", "system", "data", "algorithm", "infrastructure"]),
            ("finance", vec!["budget", "investment", "financial", "portfolio", "compliance", "audit"]),
            ("healthcare", vec!["patient", "clinical", "medical", "health", "treatment", "diagnosis"]),
            ("research", vec!["study", "analysis", "hypothesis", "methodology", "findings", "research"]),
            ("operations", vec!["process", "workflow", "efficiency", "operations", "logistics"]),
            ("governance", vec!["policy", "regulation", "compliance", "governance", "standards"]),
            ("marketing", vec!["campaign", "brand", "customer", "market", "engagement", "conversion"]),
            ("analytics", vec!["metrics", "kpi", "dashboard", "analytics", "reporting", "insights"]),
        ];
        
        for (domain, keywords) in domain_keywords {
            let matches = keywords.iter().filter(|k| lower.contains(*k)).count();
            if matches >= 2 {
                return Some(domain.to_string());
            }
        }
        
        None
    }
}

// === Helper Functions ===

/// Normalize text: lowercase, collapse whitespace, remove special chars.
fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Tokenize normalized text into words.
fn tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace().collect()
}

/// Normalize byte length to [0, 1] range.
fn normalize_byte_length(len: usize) -> f32 {
    // Assume max useful length ~1000 chars
    (len as f32 / 1000.0).min(1.0)
}

/// Normalize token count to [0, 1] range.
fn normalize_token_count(count: usize) -> f32 {
    // Assume max useful tokens ~200
    (count as f32 / 200.0).min(1.0)
}

/// Compute sentence entropy (complexity measure).
fn compute_sentence_entropy(text: &str) -> f32 {
    let tokens = tokenize(text);
    if tokens.is_empty() {
        return 0.0;
    }
    
    // Count token frequencies
    let mut freq: HashMap<&str, u32> = HashMap::new();
    for token in &tokens {
        *freq.entry(token).or_insert(0) += 1;
    }
    
    // Compute entropy
    let total = tokens.len() as f32;
    let mut entropy = 0.0f32;
    for &count in freq.values() {
        let p = count as f32 / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    
    // Normalize by max entropy
    let max_entropy = (tokens.len() as f32).ln();
    if max_entropy > 0.0 {
        (entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Compute punctuation vector [?, !, .] as ratios.
fn compute_punct_vector(text: &str) -> [f32; 3] {
    let total = text.chars().filter(|c| c.is_alphanumeric()).count() as f32;
    if total == 0.0 {
        return [0.0; 3];
    }
    
    let question = text.chars().filter(|&c| c == '?').count() as f32;
    let exclamation = text.chars().filter(|&c| c == '!').count() as f32;
    let period = text.chars().filter(|&c| c == '.').count() as f32;
    
    [
        (question / total).min(1.0),
        (exclamation / total).min(1.0),
        (period / total).min(1.0),
    ]
}

/// Compute urgency score from markers.
fn compute_urgency(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let urgency_markers = [
        "urgent", "asap", "immediately", "emergency", "critical",
        "deadline", "overdue", "priority", "hurry", "quickly",
        "now", "tonight", "today", "soon", "fast",
    ];
    
    let matches = urgency_markers.iter().filter(|m| lower.contains(*m)).count();
    (matches as f32 / 3.0).min(1.0)
}

/// Compute formality score from markers.
fn compute_formality(text: &str) -> f32 {
    let lower = text.to_lowercase();
    
    // Formal markers
    let formal_markers = [
        "please", "kindly", "would you", "may i", "could you",
        "sincerely", "respectfully", "dear", "regards", "formally",
    ];
    
    // Casual markers
    let casual_markers = [
        "hey", "hi there", "what's up", "cool", "awesome",
        "thanks", "cheers", "lol", "ok", "yeah",
    ];
    
    let formal_count = formal_markers.iter().filter(|m| lower.contains(*m)).count();
    let casual_count = casual_markers.iter().filter(|m| lower.contains(*m)).count();
    
    // Base formality at 0.5, adjust by markers
    let score = 0.5 + (formal_count as f32 * 0.1) - (casual_count as f32 * 0.1);
    score.clamp(0.0, 1.0)
}

/// Compute technical domain score.
fn compute_technical(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let technical_markers = [
        "api", "code", "function", "algorithm", "database", "server",
        "implementation", "architecture", "debug", "error", "exception",
        "variable", "method", "class", "interface", "protocol",
        "configuration", "deployment", "infrastructure", "pipeline",
    ];
    
    let matches = technical_markers.iter().filter(|m| lower.contains(*m)).count();
    (matches as f32 / 4.0).min(1.0)
}

/// Compute domain hint score.
fn compute_domain_hint(text: &str) -> f32 {
    // Returns a normalized score based on domain-specific vocabulary density
    let lower = text.to_lowercase();
    let domain_markers = [
        "domain", "field", "industry", "sector", "vertical",
        "business", "enterprise", "organization", "company",
    ];
    
    let matches = domain_markers.iter().filter(|m| lower.contains(*m)).count();
    (matches as f32 / 3.0).min(1.0)
}

/// Compute temporal cue score.
fn compute_temporal_cue(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let temporal_markers = [
        "now", "today", "tomorrow", "yesterday", "week", "month",
        "year", "quarter", "annual", "monthly", "weekly", "daily",
        "schedule", "deadline", "due", "when", "time", "date",
    ];
    
    let matches = temporal_markers.iter().filter(|m| lower.contains(*m)).count();
    (matches as f32 / 3.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signature_compute() {
        let hasher = SemanticHasher::new();
        let sig = ClassificationSignature::compute("Hello, what is the weather today?", &hasher);
        
        assert!(sig.punct_vector[0] > 0.0); // Has question mark
        assert!(sig.token_count_norm > 0.0);
        assert!(sig.temporal_cue > 0.0); // Has "today"
    }
    
    #[test]
    fn test_semantic_hasher() {
        let hasher = SemanticHasher::new();
        let pos1 = hasher.hash("software api infrastructure");
        let pos2 = hasher.hash("budget investment portfolio");
        
        // Different domains should have different positions
        assert_ne!(pos1, pos2);
    }
    
    #[test]
    fn test_feature_vector_length() {
        let sig = ClassificationSignature::default();
        let vec = sig.to_feature_vector();
        assert_eq!(vec.len(), 14);
    }
    
    #[test]
    fn test_urgency_detection() {
        assert!(compute_urgency("This is urgent, respond ASAP!") > 0.3);
        assert!(compute_urgency("Hello there") < 0.1);
    }
    
    #[test]
    fn test_formality_detection() {
        assert!(compute_formality("Please kindly respond") > 0.6);
        assert!(compute_formality("Hey, what's up?") < 0.5);
    }
    
    #[test]
    fn test_technical_detection() {
        assert!(compute_technical("The API function returns an error") > 0.3);
        assert!(compute_technical("Hello, how are you?") < 0.1);
    }
    
    #[test]
    fn test_domain_anchor() {
        let hasher = SemanticHasher::new();
        let pos = hasher.hash("software systems API infrastructure code");
        
        // Should be pulled toward technology anchor
        assert!(pos[0] > 0.5); // Technology anchor x = 0.7
    }
}
