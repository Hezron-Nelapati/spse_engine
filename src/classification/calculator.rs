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
    pub w_intent_hash: f32,
    pub w_tone_hash: f32,

    /// Semantic hasher for new text
    semantic_hasher: SemanticHasher,
}

impl ClassificationCalculator {
    /// Create new calculator with default weights.
    pub fn new() -> Self {
        Self {
            w_structure: 0.10,
            w_punctuation: 0.10,
            w_semantic: 0.15,
            w_derived: 0.10,
            w_intent_hash: 0.35,
            w_tone_hash: 0.20,
            semantic_hasher: SemanticHasher::new(),
        }
    }

    /// Create calculator with custom weights.
    pub fn with_weights(
        w_structure: f32,
        w_punctuation: f32,
        w_semantic: f32,
        w_derived: f32,
        w_intent_hash: f32,
        w_tone_hash: f32,
    ) -> Self {
        Self {
            w_structure,
            w_punctuation,
            w_semantic,
            w_derived,
            w_intent_hash,
            w_tone_hash,
            semantic_hasher: SemanticHasher::new(),
        }
    }

    /// Main calculation entry point.
    /// Uses Nearest Centroid Classifier: compares query feature vector against
    /// per-intent and per-tone mean centroids (~23 comparisons, not 23K patterns).
    /// Each centroid averages 1000+ training examples, so topic-specific noise
    /// cancels out while intent-discriminating keywords reinforce.
    pub fn calculate(
        &self,
        text: &str,
        memory: &MemoryStore,
        _spatial: &SpatialGrid,
        config: &ClassificationConfig,
    ) -> ClassificationResult {
        let query_sig = ClassificationSignature::compute(text, &self.semantic_hasher);
        let query_fv = query_sig.to_feature_vector();

        // Get centroids (built during training)
        let intent_centroids = memory.intent_centroids();
        let tone_centroids = memory.tone_centroids();

        // Early exit: no centroids yet
        if intent_centroids.is_empty() {
            return ClassificationResult {
                intent: IntentKind::Unknown,
                tone: ToneKind::NeutralProfessional,
                resolver_mode: ResolverMode::Exploratory,
                confidence: 0.0,
                method: CalculationMethod::MemoryLookup,
                candidate_count: 0,
            };
        }

        // Feature vector layout: structural(0-13) + intent_hash(14-45) + tone_hash(46-77)
        let structural = &query_fv[..14];
        let intent_hash = &query_fv[14..46];
        let tone_hash = &query_fv[46..78];

        // Score query against each intent centroid: structural + intent_hash dims
        let mut intent_scores: Vec<(IntentKind, f32)> = intent_centroids
            .iter()
            .map(|(intent, centroid)| {
                let c_struct = &centroid[..14];
                let c_intent = &centroid[14..46];
                let struct_sim = raw_cosine_similarity(structural, c_struct);
                let intent_sim = raw_cosine_similarity(intent_hash, c_intent);
                let blended =
                    self.w_intent_hash * intent_sim + (1.0 - self.w_intent_hash) * struct_sim;
                (*intent, blended)
            })
            .collect();
        intent_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Score query against each tone centroid: structural + tone_hash dims
        let mut tone_scores: Vec<(ToneKind, f32)> = tone_centroids
            .iter()
            .map(|(tone, centroid)| {
                let c_struct = &centroid[..14];
                let c_tone = &centroid[46..78];
                let struct_sim = raw_cosine_similarity(structural, c_struct);
                let tone_sim = raw_cosine_similarity(tone_hash, c_tone);
                let blended = self.w_tone_hash * tone_sim + (1.0 - self.w_tone_hash) * struct_sim;
                (*tone, blended)
            })
            .collect();
        tone_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (best_intent, best_intent_sim) = intent_scores
            .first()
            .copied()
            .unwrap_or((IntentKind::Unknown, 0.0));
        let (best_tone, _best_tone_sim) = tone_scores
            .first()
            .copied()
            .unwrap_or((ToneKind::NeutralProfessional, 0.0));

        // Confidence: how much the winner stands out from the runner-up
        let runner_up_sim = intent_scores.get(1).map(|s| s.1).unwrap_or(0.0);
        let confidence = if best_intent_sim > 0.0 {
            ((best_intent_sim - runner_up_sim) / best_intent_sim + best_intent_sim).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Determine resolver mode from confidence (Layer 9 alignment)
        let resolver_mode =
            self.apply_confidence_resolver_override(ResolverMode::Balanced, confidence, config);

        ClassificationResult {
            intent: best_intent,
            tone: best_tone,
            resolver_mode,
            confidence,
            method: CalculationMethod::MemoryLookup,
            candidate_count: intent_centroids.len(),
        }
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

        // 2. Top-k selection: only keep the most similar candidates for voting
        if scored_candidates.len() > config.top_k_candidates {
            scored_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored_candidates.truncate(config.top_k_candidates);
        }

        // 3. Aggregate votes or return unknown
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
        let candidate_ids =
            spatial.nearby(query_sig.semantic_centroid, config.spatial_query_radius);

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
        let candidate_ids =
            spatial.nearby(query_sig.semantic_centroid, config.spatial_query_radius);

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

        // Compute total score for normalization
        let intent_total: f32 = intent_scores.values().sum();
        let tone_total: f32 = tone_scores.values().sum();

        // Select max
        let (intent, intent_score) = self.max_score(intent_scores);
        let (tone, tone_score) = self.max_score(tone_scores);
        let (resolver, _resolver_score) = self.max_score(resolver_scores);

        // Confidence = share of winning class in total votes (normalized to [0,1])
        let intent_confidence = if intent_total > 0.0 {
            intent_score / intent_total
        } else {
            0.0
        };
        let tone_confidence = if tone_total > 0.0 {
            tone_score / tone_total
        } else {
            0.0
        };
        let confidence = ((intent_confidence + tone_confidence) / 2.0).clamp(0.0, 1.0);

        // Apply confidence-driven resolver override (Layer 9 alignment)
        let final_resolver = self.apply_confidence_resolver_override(resolver, confidence, config);

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

    /// Classification similarity using full 14-dim feature vector (including centroid).
    /// Uses weighted Euclidean distance converted to [0, 1] similarity.
    /// Centroid is the most discriminating feature: different texts → different trigram hashes.
    /// With Euclidean distance, centroid contributes proportionally (unlike cosine where it dominated).
    pub fn classification_similarity(
        &self,
        a: &ClassificationSignature,
        b: &ClassificationSignature,
    ) -> f32 {
        let va = a.to_feature_vector();
        let vb = b.to_feature_vector();

        // Per-dimension weights for all 14 features
        let weights = self.classification_weights();

        // Weighted squared Euclidean distance
        let dist_sq: f32 = va
            .iter()
            .zip(vb.iter())
            .zip(weights.iter())
            .map(|((ai, bi), w)| w * (ai - bi).powi(2))
            .sum();

        // Convert distance to similarity: 1 / (1 + sqrt(dist_sq))
        let dist = dist_sq.sqrt();
        (1.0 / (1.0 + dist * 5.0)).clamp(0.0, 1.0)
    }

    /// Full cosine similarity between signatures (used externally).
    pub fn cosine_similarity(
        &self,
        a: &ClassificationSignature,
        b: &ClassificationSignature,
    ) -> f32 {
        let va = a.to_feature_vector();
        let vb = b.to_feature_vector();

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

    /// Per-dimension weight array for all 78 features (matches to_feature_vector layout).
    /// Layout: [0-2] structure, [3-5] punctuation, [6-8] centroid, [9-13] derived,
    ///         [14-45] intent_hash, [46-77] tone_hash
    fn classification_weights(&self) -> Vec<f32> {
        let mut w = Vec::with_capacity(78);
        // Structure (3)
        w.extend_from_slice(&[self.w_structure; 3]);
        // Punctuation (3)
        w.extend_from_slice(&[self.w_punctuation; 3]);
        // Centroid (3)
        w.extend_from_slice(&[self.w_semantic; 3]);
        // Derived (5)
        w.extend_from_slice(&[self.w_derived; 5]);
        // Intent hash (32)
        w.extend_from_slice(&[self.w_intent_hash; 32]);
        // Tone hash (32)
        w.extend_from_slice(&[self.w_tone_hash; 32]);
        w
    }

    /// Apply feature weights to full vector (30 dims).
    fn apply_weights(&self, v: &[f32]) -> Vec<f32> {
        let weights = self.classification_weights();
        v.iter()
            .zip(weights.iter())
            .map(|(vi, wi)| vi * wi)
            .collect()
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
    pub fn adjust_weights(
        &mut self,
        w_structure: f32,
        w_punctuation: f32,
        w_semantic: f32,
        w_derived: f32,
    ) {
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

/// Cosine similarity between two raw feature vectors (no per-dimension weights).
/// Used by the Nearest Centroid Classifier to compare query against class centroids.
fn raw_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a * norm_b)).clamp(0.0, 1.0)
    } else {
        0.0
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

        assert!((calc.w_structure - 0.10).abs() < 0.01);
        assert!((calc.w_intent_hash - 0.35).abs() < 0.01);
        assert!((calc.w_tone_hash - 0.20).abs() < 0.01);
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
        let result =
            calc.apply_confidence_resolver_override(ResolverMode::Deterministic, 0.3, &config);
        assert_eq!(result, ResolverMode::Exploratory);

        // High confidence -> can upgrade
        let result = calc.apply_confidence_resolver_override(ResolverMode::Balanced, 0.9, &config);
        assert_eq!(result, ResolverMode::Deterministic);

        // Medium confidence -> keep predicted
        let result = calc.apply_confidence_resolver_override(ResolverMode::Balanced, 0.6, &config);
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
