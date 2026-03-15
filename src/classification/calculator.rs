//! Classification calculator using Layer 6 Spatial Index and Layer 4 Memory Store.
//!
//! Implements O(log N) retrieval via spatial query and weighted vote aggregation.

use crate::classification::{ClassificationPattern, ClassificationSignature, SemanticHasher};
use crate::config::ClassificationConfig;
use crate::memory::MemoryStore;
use crate::spatial_index::SpatialGrid;
use crate::types::{CalculationMethod, ClassificationResult, IntentKind, ResolverMode, ToneKind};
use std::collections::HashMap;

/// Main classification calculator.
/// Uses Layer 6 Spatial Index for O(log N) lookup and Layer 14 scoring logic.
pub struct ClassificationCalculator {
    /// Feature weights (adjusted via Layer 18 feedback)
    pub w_structure: f32,
    pub w_punctuation: f32,
    pub w_semantic: f32,
    pub w_derived: f32,
    
    /// Semantic hasher for new text
    semantic_hasher: SemanticHasher,
}

impl ClassificationCalculator {
    /// Create new calculator with default weights.
    pub fn new() -> Self {
        Self {
            w_structure: 0.25,
            w_punctuation: 0.20,
            w_semantic: 0.35,
            w_derived: 0.20,
            semantic_hasher: SemanticHasher::new(),
        }
    }
    
    /// Create calculator with custom weights.
    pub fn with_weights(
        w_structure: f32,
        w_punctuation: f32,
        w_semantic: f32,
        w_derived: f32,
    ) -> Self {
        Self {
            w_structure,
            w_punctuation,
            w_semantic,
            w_derived,
            semantic_hasher: SemanticHasher::new(),
        }
    }
    
/// Main calculation entry point.
    /// Returns classification result with intent, tone, resolver mode, and confidence.
    /// Uses GPU acceleration when available, falls back to CPU otherwise.
    pub fn calculate(
        &self,
        text: &str,
        memory: &MemoryStore,
        spatial: &SpatialGrid,
        config: &ClassificationConfig,
    ) -> ClassificationResult {
        // Compute signature once (shared by GPU and CPU paths)
        let query_sig = ClassificationSignature::compute(text, &self.semantic_hasher);
        
        // Get candidate IDs from spatial index
        let candidate_ids = spatial.nearby(query_sig.semantic_centroid, config.spatial_query_radius);
        
        // Early exit: no candidates
        if candidate_ids.is_empty() {
            return ClassificationResult {
                intent: IntentKind::Unknown,
                tone: ToneKind::NeutralProfessional,
                resolver_mode: ResolverMode::Exploratory,
                confidence: 0.0,
                method: CalculationMethod::MemoryLookup,
                candidate_count: 0,
            };
        }
        
        // CPU fallback (original implementation)
        
        // Try GPU acceleration if available and pattern count justifies overhead
        #[cfg(feature = "gpu")]
        {
            const GPU_THRESHOLD: usize = 64;
            if crate::gpu::is_gpu_available() && candidate_ids.len() >= GPU_THRESHOLD {
                if let Some(gpu_calc) = crate::gpu::compute::get_gpu_classifier() {
                    // Fetch patterns from memory
                    let patterns: Vec<_> = candidate_ids.iter()
                        .filter_map(|id| memory.get_classification_pattern(*id))
                        .collect();
                    
                    if patterns.len() >= GPU_THRESHOLD {
                        if let Some(result) = gpu_calc.calculate_gpu(&query_sig, &patterns, config) {
                            return result;
                        }
                    }
                    // Fall through to CPU on GPU failure or insufficient patterns
                }
            }
        }
        
        // CPU fallback (original implementation)
        self.calculate_cpu_with_signature(&query_sig, &candidate_ids, memory, spatial, config)
    }
    
    /// CPU implementation with pre-computed signature (avoids recomputation).
    fn calculate_cpu_with_signature(
        &self,
        query_sig: &ClassificationSignature,
        candidate_ids: &[uuid::Uuid],
        memory: &MemoryStore,
        _spatial: &SpatialGrid,
        config: &ClassificationConfig,
    ) -> ClassificationResult {
        // 1. Retrieve and score (L14 logic)
        let mut scored_candidates = Vec::new();
        
        for id in candidate_ids {
            if let Some(pattern) = memory.get_classification_pattern(*id) {
                let similarity = self.cosine_similarity(query_sig, &pattern.signature);
                let final_score = similarity * pattern.confidence();
                
                if final_score >= config.min_similarity_threshold {
                    scored_candidates.push((pattern, final_score));
                }
            }
        }
        
        // 2. Aggregate votes or return unknown
        if scored_candidates.is_empty() {
            return ClassificationResult {
                intent: IntentKind::Unknown,
                tone: ToneKind::NeutralProfessional,
                resolver_mode: ResolverMode::Exploratory, // Low confidence = exploratory
                confidence: 0.0,
                method: CalculationMethod::MemoryLookup,
                candidate_count: 0,
            };
        }
        
        self.aggregate_votes(scored_candidates, config)
    }
    
    /// Calculate intent only (for partial classification).
    pub fn calculate_intent(
        &self,
        text: &str,
        memory: &MemoryStore,
        spatial: &SpatialGrid,
        config: &ClassificationConfig,
    ) -> (IntentKind, f32) {
        let query_sig = ClassificationSignature::compute(text, &self.semantic_hasher);
        let candidate_ids = spatial.nearby(query_sig.semantic_centroid, config.spatial_query_radius);
        
        let mut intent_scores: HashMap<IntentKind, f32> = HashMap::new();
        
        for id in candidate_ids {
            if let Some(pattern) = memory.get_classification_pattern(id) {
                let similarity = self.cosine_similarity(&query_sig, &pattern.signature);
                let score = similarity * pattern.confidence();
                
                if score >= config.min_similarity_threshold {
                    *intent_scores.entry(pattern.intent_kind).or_default() += score;
                }
            }
        }
        
        if intent_scores.is_empty() {
            return (IntentKind::Unknown, 0.0);
        }
        
        let (best_intent, best_score) = self.max_score(intent_scores);
        (best_intent, best_score)
    }
    
    /// Calculate tone only (for partial classification).
    pub fn calculate_tone(
        &self,
        text: &str,
        memory: &MemoryStore,
        spatial: &SpatialGrid,
        config: &ClassificationConfig,
    ) -> (ToneKind, f32) {
        let query_sig = ClassificationSignature::compute(text, &self.semantic_hasher);
        let candidate_ids = spatial.nearby(query_sig.semantic_centroid, config.spatial_query_radius);
        
        let mut tone_scores: HashMap<ToneKind, f32> = HashMap::new();
        
        for id in candidate_ids {
            if let Some(pattern) = memory.get_classification_pattern(id) {
                let similarity = self.cosine_similarity(&query_sig, &pattern.signature);
                let score = similarity * pattern.confidence();
                
                if score >= config.min_similarity_threshold {
                    *tone_scores.entry(pattern.tone_kind).or_default() += score;
                }
            }
        }
        
        if tone_scores.is_empty() {
            return (ToneKind::NeutralProfessional, 0.0);
        }
        
        let (best_tone, best_score) = self.max_score(tone_scores);
        (best_tone, best_score)
    }
    
    /// Aggregate votes with weighted scoring.
    fn aggregate_votes(
        &self,
        candidates: Vec<(ClassificationPattern, f32)>,
        config: &ClassificationConfig,
    ) -> ClassificationResult {
        let mut intent_scores: HashMap<IntentKind, f32> = HashMap::new();
        let mut tone_scores: HashMap<ToneKind, f32> = HashMap::new();
        let mut resolver_scores: HashMap<ResolverMode, f32> = HashMap::new();
        
        for (pattern, score) in &candidates {
            *intent_scores.entry(pattern.intent_kind).or_default() += score;
            *tone_scores.entry(pattern.tone_kind).or_default() += score;
            *resolver_scores.entry(pattern.resolver_mode).or_default() += score;
        }
        
        // Select max with confidence
        let (intent, intent_score) = self.max_score(intent_scores);
        let (tone, tone_score) = self.max_score(tone_scores);
        let (resolver, _resolver_score) = self.max_score(resolver_scores);
        
        // Overall confidence is average of intent and tone scores
        let confidence = (intent_score + tone_score) / 2.0;
        
        // Apply confidence-driven resolver override (Layer 9 alignment)
        let final_resolver = self.apply_confidence_resolver_override(
            resolver,
            confidence,
            config,
        );
        
        ClassificationResult {
            intent,
            tone,
            resolver_mode: final_resolver,
            confidence,
            method: CalculationMethod::MemoryLookup,
            candidate_count: candidates.len(),
        }
    }
    
    /// Confidence-driven resolver mode (Layer 9 alignment).
    /// Rule: Low confidence forces Exploratory (retrieve more evidence).
    fn apply_confidence_resolver_override(
        &self,
        predicted: ResolverMode,
        confidence: f32,
        config: &ClassificationConfig,
    ) -> ResolverMode {
        // Low confidence: force Exploratory
        if confidence < config.low_confidence_threshold {
            return ResolverMode::Exploratory;
        }
        
        // High confidence: can upgrade to Deterministic
        if confidence > config.high_confidence_threshold && predicted == ResolverMode::Balanced {
            return ResolverMode::Deterministic;
        }
        
        predicted
    }
    
    /// Cosine similarity between signatures.
    pub fn cosine_similarity(&self, a: &ClassificationSignature, b: &ClassificationSignature) -> f32 {
        let va = a.to_feature_vector();
        let vb = b.to_feature_vector();
        
        // Apply weights to feature vector
        let wa = self.apply_weights(&va);
        let wb = self.apply_weights(&vb);
        
        let dot: f32 = wa.iter().zip(wb.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = wa.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = wb.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Apply feature weights to vector.
    fn apply_weights(&self, v: &[f32]) -> Vec<f32> {
        // Feature vector layout:
        // [0-2] structure (byte_length, entropy, token_count)
        // [3-5] punctuation (?, !, .)
        // [6-8] semantic centroid (x, y, z)
        // [9-13] derived (urgency, formality, technical, domain, temporal)
        
        vec![
            v[0] * self.w_structure, // byte_length
            v[1] * self.w_structure, // entropy
            v[2] * self.w_structure, // token_count
            v[3] * self.w_punctuation, // ?
            v[4] * self.w_punctuation, // !
            v[5] * self.w_punctuation, // .
            v[6] * self.w_semantic, // centroid x
            v[7] * self.w_semantic, // centroid y
            v[8] * self.w_semantic, // centroid z
            v[9] * self.w_derived, // urgency
            v[10] * self.w_derived, // formality
            v[11] * self.w_derived, // technical
            v[12] * self.w_derived, // domain
            v[13] * self.w_derived, // temporal
        ]
    }
    
    /// Get max score from HashMap.
    fn max_score<K: Clone + Eq + std::hash::Hash>(&self, scores: HashMap<K, f32>) -> (K, f32) {
        scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(|| panic!("Empty scores map"))
    }
    
    /// Get semantic hasher for external use.
    pub fn hasher(&self) -> &SemanticHasher {
        &self.semantic_hasher
    }
    
    /// Adjust weights based on training feedback.
    pub fn adjust_weights(&mut self, w_structure: f32, w_punctuation: f32, w_semantic: f32, w_derived: f32) {
        self.w_structure = w_structure;
        self.w_punctuation = w_punctuation;
        self.w_semantic = w_semantic;
        self.w_derived = w_derived;
    }
}

impl Default for ClassificationCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClassificationConfig;
    use crate::memory::MemoryStore;
    use crate::spatial_index::SpatialGrid;
    
    #[test]
    fn test_calculator_creation() {
        let calc = ClassificationCalculator::new();
        
        assert!((calc.w_structure - 0.25).abs() < 0.01);
        assert!((calc.w_semantic - 0.35).abs() < 0.01);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let calc = ClassificationCalculator::new();
        let hasher = SemanticHasher::new();
        
        let sig1 = ClassificationSignature::compute("Hello there", &hasher);
        let sig2 = ClassificationSignature::compute("Hello there", &hasher);
        
        // Same text should have high similarity
        let similarity = calc.cosine_similarity(&sig1, &sig2);
        assert!(similarity > 0.99);
    }
    
    #[test]
    fn test_cosine_similarity_different() {
        let calc = ClassificationCalculator::new();
        let hasher = SemanticHasher::new();
        
        let sig1 = ClassificationSignature::compute("Hello there", &hasher);
        let sig2 = ClassificationSignature::compute("What is the weather today?", &hasher);
        
        // Different text should have lower similarity (but short texts may share structure)
        let similarity = calc.cosine_similarity(&sig1, &sig2);
        assert!(similarity < 1.0); // Not identical
        assert!(similarity < 0.99); // Reasonably different
    }
    
    #[test]
    fn test_confidence_resolver_override() {
        let calc = ClassificationCalculator::new();
        let config = ClassificationConfig::default();
        
        // Low confidence -> Exploratory
        let result = calc.apply_confidence_resolver_override(
            ResolverMode::Deterministic,
            0.3,
            &config,
        );
        assert_eq!(result, ResolverMode::Exploratory);
        
        // High confidence -> can upgrade
        let result = calc.apply_confidence_resolver_override(
            ResolverMode::Balanced,
            0.9,
            &config,
        );
        assert_eq!(result, ResolverMode::Deterministic);
        
        // Medium confidence -> keep predicted
        let result = calc.apply_confidence_resolver_override(
            ResolverMode::Balanced,
            0.6,
            &config,
        );
        assert_eq!(result, ResolverMode::Balanced);
    }
    
    #[test]
    fn test_empty_memory_returns_unknown() {
        use std::env;
        let calc = ClassificationCalculator::new();
        let db_path = env::temp_dir().join(format!("spse_calc_test_{}.db", uuid::Uuid::new_v4()));
        let memory = MemoryStore::new(db_path.to_str().expect("db path"));
        let spatial = SpatialGrid::new(0.5);
        let config = ClassificationConfig::default();
        
        let result = calc.calculate("Hello", &memory, &spatial, &config);
        
        assert_eq!(result.intent, IntentKind::Unknown);
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.method, CalculationMethod::MemoryLookup);
    }
}
