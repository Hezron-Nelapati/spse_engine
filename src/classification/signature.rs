//! Lightweight feature signature for classification.
//!
//! ClassificationSignature is a CPU-efficient feature vector (<20 floats) that captures
//! structural, punctuation, semantic, and derived features for text classification.

use once_cell::sync::Lazy;
use postagger::PerceptronTagger;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Lightweight feature vector for CPU-efficient classification.
/// Total: 78 floats (14 structural + 32 intent hash + 32 tone hash).
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

    // POS-based intent hash (32 buckets)
    // Only verbs (VB*), question/wh-words (W*), and modals (MD) are hashed.
    // These POS tags carry the intent signal; nouns and function words are dropped.
    #[serde(default = "default_hash_32")]
    pub intent_hash: [f32; 32],

    // POS-based tone hash (32 buckets)
    // Only adjectives (JJ*) and adverbs (RB*) are hashed.
    // These POS tags carry the tone/style signal.
    #[serde(default = "default_hash_32")]
    pub tone_hash: [f32; 32],
}

impl ClassificationSignature {
    /// Convert to feature vector for similarity computation.
    /// Returns 78-element vector, all dimensions in [0, 1].
    /// Layout: structure(3) + punctuation(3) + centroid(3) + derived(5) + intent_hash(32) + tone_hash(32)
    pub fn to_feature_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(78);
        // Structure (rescaled to [0,1] for short queries)
        v.push((self.byte_length_norm * 12.0).min(1.0));
        v.push(self.sentence_entropy);
        v.push((self.token_count_norm * 14.0).min(1.0));
        // Punctuation (rescaled)
        v.push((self.punct_vector[0] * 30.0).min(1.0));
        v.push((self.punct_vector[1] * 30.0).min(1.0));
        v.push((self.punct_vector[2] * 30.0).min(1.0));
        // Centroid
        v.extend_from_slice(&self.semantic_centroid);
        // Derived
        v.push(self.urgency_score);
        v.push(self.formality_score);
        v.push(self.technical_score);
        v.push(self.domain_hint);
        v.push(self.temporal_cue);
        // POS-based intent hash (verbs + wh-words + modals)
        v.extend_from_slice(&self.intent_hash);
        // POS-based tone hash (adjectives + adverbs)
        v.extend_from_slice(&self.tone_hash);
        v
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
            // Preserve the 78-dim legacy layout while injecting a dedicated
            // creative-task signal into the derived feature budget.
            domain_hint: compute_domain_hint(&normalized).max(compute_creative_cue(&normalized)),
            temporal_cue: compute_temporal_cue(&normalized),
            intent_hash: compute_pos_intent_hash(&normalized),
            tone_hash: compute_pos_tone_hash(&normalized),
        }
    }

    /// Convert to classification feature vector (excluding centroid).
    /// Returns 11-element vector: structure (3) + punctuation (3) + derived (5).
    /// Centroid is used for spatial pre-filtering only, not similarity scoring.
    pub fn to_classification_features(&self) -> Vec<f32> {
        vec![
            self.byte_length_norm,
            self.sentence_entropy,
            self.token_count_norm,
            self.punct_vector[0],
            self.punct_vector[1],
            self.punct_vector[2],
            self.urgency_score,
            self.formality_score,
            self.technical_score,
            self.domain_hint,
            self.temporal_cue,
        ]
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

fn default_hash_32() -> [f32; 32] {
    [0.0; 32]
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
            intent_hash: [0.0f32; 32],
            tone_hash: [0.0f32; 32],
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
        let mut hash = 2166136261u32; // FNV offset basis

        // Sort trigrams for deterministic output (HashMap order is random)
        let mut trigrams: Vec<_> = freq.iter().collect();
        trigrams.sort_by(|a, b| a.0.cmp(b.0));

        // XOR-fold each trigram + its frequency into the hash state
        for (trigram, count) in &trigrams {
            for c in trigram.chars() {
                hash ^= c as u32;
                hash = hash.wrapping_mul(16777619); // FNV prime
            }
            // Mix in quantized frequency
            hash ^= (**count * 65536.0) as u32;
            hash = hash.wrapping_mul(prime);
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
            let trigram: String = chars[i..i + 3].iter().collect();
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
            (
                "technology",
                vec![
                    "software",
                    "api",
                    "code",
                    "system",
                    "data",
                    "algorithm",
                    "infrastructure",
                ],
            ),
            (
                "finance",
                vec![
                    "budget",
                    "investment",
                    "financial",
                    "portfolio",
                    "compliance",
                    "audit",
                ],
            ),
            (
                "healthcare",
                vec![
                    "patient",
                    "clinical",
                    "medical",
                    "health",
                    "treatment",
                    "diagnosis",
                ],
            ),
            (
                "research",
                vec![
                    "study",
                    "analysis",
                    "hypothesis",
                    "methodology",
                    "findings",
                    "research",
                ],
            ),
            (
                "operations",
                vec![
                    "process",
                    "workflow",
                    "efficiency",
                    "operations",
                    "logistics",
                ],
            ),
            (
                "governance",
                vec![
                    "policy",
                    "regulation",
                    "compliance",
                    "governance",
                    "standards",
                ],
            ),
            (
                "marketing",
                vec![
                    "campaign",
                    "brand",
                    "customer",
                    "market",
                    "engagement",
                    "conversion",
                ],
            ),
            (
                "analytics",
                vec![
                    "metrics",
                    "kpi",
                    "dashboard",
                    "analytics",
                    "reporting",
                    "insights",
                ],
            ),
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
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
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
        "urgent",
        "asap",
        "immediately",
        "emergency",
        "critical",
        "deadline",
        "overdue",
        "priority",
        "hurry",
        "quickly",
        "now",
        "tonight",
        "today",
        "soon",
        "fast",
    ];

    let matches = urgency_markers
        .iter()
        .filter(|m| lower.contains(*m))
        .count();
    (matches as f32 / 3.0).min(1.0)
}

/// Compute formality score from markers.
fn compute_formality(text: &str) -> f32 {
    let normalized = normalize_text(text);
    let tokens = tokenize(&normalized);

    // Formal markers
    let formal_markers = [
        "please",
        "kindly",
        "would you",
        "may i",
        "could you",
        "sincerely",
        "respectfully",
        "dear",
        "regards",
        "formally",
    ];

    // Casual markers
    let casual_markers = [
        "hey",
        "hi there",
        "what's up",
        "cool",
        "awesome",
        "thanks",
        "cheers",
        "lol",
        "ok",
        "yeah",
    ];

    let token_matches = |marker: &str| {
        if marker.contains(' ') {
            normalized.contains(marker)
        } else {
            tokens.iter().any(|token| *token == marker)
        }
    };

    let formal_count = formal_markers
        .iter()
        .filter(|marker| token_matches(marker))
        .count();
    let casual_count = casual_markers
        .iter()
        .filter(|marker| token_matches(marker))
        .count();

    // Base formality at 0.5, adjust by markers
    let score = 0.5 + (formal_count as f32 * 0.1) - (casual_count as f32 * 0.1);
    score.clamp(0.0, 1.0)
}

/// Compute technical domain score.
fn compute_technical(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let technical_markers = [
        "api",
        "code",
        "function",
        "algorithm",
        "database",
        "server",
        "implementation",
        "architecture",
        "debug",
        "error",
        "exception",
        "variable",
        "method",
        "class",
        "interface",
        "protocol",
        "configuration",
        "deployment",
        "infrastructure",
        "pipeline",
    ];

    let matches = technical_markers
        .iter()
        .filter(|m| lower.contains(*m))
        .count();
    (matches as f32 / 4.0).min(1.0)
}

/// Compute domain hint score.
fn compute_domain_hint(text: &str) -> f32 {
    // Returns a normalized score based on domain-specific vocabulary density
    let lower = text.to_lowercase();
    let domain_markers = [
        "domain",
        "field",
        "industry",
        "sector",
        "vertical",
        "business",
        "enterprise",
        "organization",
        "company",
    ];

    let matches = domain_markers.iter().filter(|m| lower.contains(*m)).count();
    (matches as f32 / 3.0).min(1.0)
}

/// Compute creative-task cue used to separate imperative-creative prompts
/// from imperative-factual prompts without changing the legacy 78-dim layout.
fn compute_creative_cue(text: &str) -> f32 {
    let normalized = normalize_text(text);
    let tokens = tokenize(&normalized);
    if tokens.is_empty() {
        return 0.0;
    }

    let creative_terms = [
        "write",
        "create",
        "compose",
        "draw",
        "design",
        "invent",
        "brainstorm",
        "imagine",
        "generate",
        "craft",
        "draft",
        "poem",
        "story",
        "idea",
    ];
    let factual_imperatives = [
        "explain",
        "describe",
        "define",
        "list",
        "summarize",
        "calculate",
        "find",
        "show",
        "prove",
    ];

    let first = tokens.first().copied().unwrap_or_default();
    if factual_imperatives.contains(&first) {
        return 0.0;
    }
    if creative_terms.contains(&first) {
        return 1.0;
    }
    if tokens
        .iter()
        .take(4)
        .any(|token| creative_terms.contains(token))
    {
        return 0.8;
    }
    0.0
}

/// Compute temporal cue score.
fn compute_temporal_cue(text: &str) -> f32 {
    let lower = text.to_lowercase();
    let temporal_markers = [
        "now",
        "today",
        "tomorrow",
        "yesterday",
        "week",
        "month",
        "year",
        "quarter",
        "annual",
        "monthly",
        "weekly",
        "daily",
        "schedule",
        "deadline",
        "due",
        "when",
        "time",
        "date",
    ];

    let matches = temporal_markers
        .iter()
        .filter(|m| lower.contains(*m))
        .count();
    (matches as f32 / 3.0).min(1.0)
}

// ============================================================================
// POS-based word filtering using postagger (NLTK perceptron tagger)
// ============================================================================

/// Global POS tagger — loaded once from pre-trained NLTK perceptron model.
/// Uses Penn Treebank tagset (NN, VB, JJ, RB, WP, MD, etc.).
static POS_TAGGER: Lazy<Option<PerceptronTagger>> = Lazy::new(|| {
    // Try common paths relative to working directory
    let paths = [(
        "config/pos_tagger/weights.json",
        "config/pos_tagger/classes.txt",
        "config/pos_tagger/tags.json",
    )];
    for (w, c, t) in &paths {
        if std::path::Path::new(w).exists() {
            return Some(PerceptronTagger::new(w, c, t));
        }
    }
    eprintln!("[classification] POS tagger model not found in config/pos_tagger/");
    None
});

/// Check if a Penn Treebank POS tag belongs to the intent group.
/// Intent group: verbs, question/wh-words, modals, interjections.
fn is_intent_pos(tag: &str) -> bool {
    tag.starts_with("VB")  // VB, VBD, VBG, VBN, VBP, VBZ
        || tag.starts_with('W') // WP, WP$, WDT, WRB (question words)
        || tag == "MD"          // modals: can, could, should, would, will, might, must
        || tag == "UH"          // interjections: hello, goodbye, thanks, yes, no
        || tag == "RP" // particles: up, out, off (phrasal verb parts)
}

/// Check if a Penn Treebank POS tag belongs to the tone group.
/// Tone group: adjectives, adverbs.
fn is_tone_pos(tag: &str) -> bool {
    tag.starts_with("JJ")  // JJ, JJR, JJS
        || tag.starts_with("RB") // RB, RBR, RBS
}

/// Check if a POS tag is a function word (carries no content signal).
/// These are always excluded from hashing.
fn is_function_pos(tag: &str) -> bool {
    matches!(
        tag,
        "DT" | "IN"
            | "CC"
            | "TO"
            | "PRP"
            | "PRP$"
            | "EX"
            | "PDT"
            | ","
            | "."
            | ":"
            | "("
            | ")"
            | "``"
            | "''"
            | "#"
            | "$"
            | "SYM"
            | "LS"
    )
}

/// Sanitize text for the POS tagger: keep only ASCII alphanumeric and spaces.
/// The postagger crate panics on multi-byte Unicode characters.
fn sanitize_for_pos(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == ' ' {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Hash a list of words into 32 FNV-1a buckets. Returns normalized distribution.
fn hash_words_to_buckets(words: &[&str]) -> [f32; 32] {
    let mut buckets = [0.0f32; 32];
    if words.is_empty() {
        return buckets;
    }
    for word in words {
        let lower = word.to_lowercase();
        let hash = fnv1a_word(lower.as_bytes());
        buckets[(hash as usize) % 32] += 1.0;
    }
    let total = words.len() as f32;
    for b in &mut buckets {
        *b /= total;
    }
    buckets
}

/// Compute intent hash: POS-tag the text, keep intent-carrying words
/// (verbs, wh-words, modals, interjections), hash into 32 buckets.
/// Special rules:
/// - First content word is always included (handles imperative mistagging)
/// - Short sentences (≤3 content words): all content words included
fn compute_pos_intent_hash(text: &str) -> [f32; 32] {
    if let Some(tagger) = POS_TAGGER.as_ref() {
        let safe = sanitize_for_pos(text);
        if safe.is_empty() {
            return [0.0; 32];
        }
        let tags =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tagger.tag(&safe))) {
                Ok(t) => t,
                Err(_) => return [0.0; 32],
            };
        // Separate content words from function words
        let content_tags: Vec<_> = tags.iter().filter(|t| !is_function_pos(&t.tag)).collect();
        let is_short = content_tags.len() <= 3;
        let mut intent_words: Vec<&str> = Vec::new();
        for (i, t) in content_tags.iter().enumerate() {
            // Always include: intent POS, first content word, or all words in short sentences
            if is_intent_pos(&t.tag) || i == 0 || is_short {
                intent_words.push(&*t.word);
            }
        }
        hash_words_to_buckets(&intent_words)
    } else {
        [0.0; 32]
    }
}

/// Compute tone hash: POS-tag the text, keep tone-carrying words
/// (adjectives, adverbs), hash into 32 buckets.
/// For short sentences (≤3 content words), all content words are included.
fn compute_pos_tone_hash(text: &str) -> [f32; 32] {
    if let Some(tagger) = POS_TAGGER.as_ref() {
        let safe = sanitize_for_pos(text);
        if safe.is_empty() {
            return [0.0; 32];
        }
        let tags =
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tagger.tag(&safe))) {
                Ok(t) => t,
                Err(_) => return [0.0; 32],
            };
        let content_tags: Vec<_> = tags.iter().filter(|t| !is_function_pos(&t.tag)).collect();
        let is_short = content_tags.len() <= 3;
        let tone_words: Vec<&str> = content_tags
            .iter()
            .filter(|t| is_tone_pos(&t.tag) || is_short)
            .map(|t| &*t.word)
            .collect();
        hash_words_to_buckets(&tone_words)
    } else {
        [0.0; 32]
    }
}

/// FNV-1a hash for a single word.
fn fnv1a_word(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
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
        assert_eq!(vec.len(), 78);
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
        assert!(compute_formality("Thanks!") < 0.6);
    }

    #[test]
    fn test_technical_detection() {
        assert!(compute_technical("The API function returns an error") > 0.3);
        assert!(compute_technical("Hello, how are you?") < 0.1);
    }

    #[test]
    fn test_domain_anchor() {
        let hasher = SemanticHasher::new();
        let tech_pos = hasher.hash("software systems API infrastructure code");
        let neutral_pos = hasher.hash("the quick brown fox jumps over the lazy dog");

        // Technology text should be detected as technology domain and blended with anchor
        // Verify positions are different (domain anchor applied vs not)
        let diff = (tech_pos[0] - neutral_pos[0]).abs()
            + (tech_pos[1] - neutral_pos[1]).abs()
            + (tech_pos[2] - neutral_pos[2]).abs();
        assert!(
            diff > 0.01,
            "Domain anchor should shift position: tech={:?} neutral={:?}",
            tech_pos,
            neutral_pos
        );
    }

    #[test]
    fn test_creative_cue_distinguishes_write_from_explain() {
        assert!(compute_creative_cue("write a poem about autumn") > 0.5);
        assert_eq!(compute_creative_cue("explain photosynthesis"), 0.0);
    }
}
