//! Classification pattern stored in Intent memory channel.
//!
//! ClassificationPattern wraps a signature with labels (intent, tone, resolver mode)
//! and learning metrics for Layer 18 feedback integration.

use crate::types::{IntentKind, MemoryChannel, MemoryType, ResolverMode, ToneKind, Unit, UnitLevel};
use crate::classification::ClassificationSignature;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Specialized Unit stored in Intent Memory Channel (Layer 4).
/// Contains classification labels and learning metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationPattern {
    /// Unique identifier
    pub unit_id: Uuid,
    /// Feature signature for similarity matching
    pub signature: ClassificationSignature,
    
    // Labels from seed data
    /// Classified intent kind
    pub intent_kind: IntentKind,
    /// Classified tone kind
    pub tone_kind: ToneKind,
    /// Resolver mode for this pattern
    pub resolver_mode: ResolverMode,
    
    // Learning metrics (Layer 18 feedback)
    /// Number of successful predictions
    pub success_count: u64,
    /// Number of failed predictions
    pub failure_count: u64,
    /// Last reinforcement timestamp
    pub last_reinforced: DateTime<Utc>,
    
    // Context
    /// Optional domain context
    pub domain: Option<String>,
    
    /// Memory channels (always includes Intent)
    pub memory_channels: Vec<MemoryChannel>,
}

impl ClassificationPattern {
    /// Create a new classification pattern from signature and labels.
    pub fn new(
        signature: ClassificationSignature,
        intent_kind: IntentKind,
        tone_kind: ToneKind,
        resolver_mode: ResolverMode,
        domain: Option<String>,
    ) -> Self {
        Self {
            unit_id: Uuid::new_v4(),
            signature,
            intent_kind,
            tone_kind,
            resolver_mode,
            success_count: 0,
            failure_count: 0,
            last_reinforced: Utc::now(),
            domain,
            memory_channels: vec![MemoryChannel::Main, MemoryChannel::Intent],
        }
    }
    
    /// Confidence derived from success/failure ratio.
    /// Returns 0.5 for patterns with no observations.
    pub fn confidence(&self) -> f32 {
        let total = self.success_count + self.failure_count;
        if total == 0 {
            return 0.5;
        }
        (self.success_count as f32 / total as f32).clamp(0.1, 1.0)
    }
    
    /// Convert to Unit for storage in MemoryStore.
    /// Uses distinct marker pattern to prevent Layer 2 merging with raw text units.
    ///
    /// Marker format: `pattern:<intent>:<tone>:<hash8>`
    /// Example: `pattern:greeting:formal:a3f2b8c1`
    pub fn to_unit(&self) -> Unit {
        // Create unique marker to prevent accidental merging
        let signature_hash = self.signature.signature_hash_8char();
        let marker = format!(
            "pattern:{:?}:{:?}:{}",
            self.intent_kind,
            self.tone_kind,
            signature_hash
        ).to_lowercase();
        
        Unit {
            id: self.unit_id,
            content: marker.clone(),
            normalized: marker,
            level: UnitLevel::Pattern,
            frequency: self.success_count + self.failure_count,
            utility_score: self.confidence(),
            semantic_position: self.signature.semantic_centroid,
            anchor_status: self.success_count > 10, // Anchor after 10+ successes
            memory_type: MemoryType::Core, // Classification patterns go to Core
            memory_channels: self.memory_channels.clone(),
            confidence: self.confidence(),
            trust_score: self.confidence(), // Trust = confidence for patterns
            contexts: self.domain.iter().cloned().collect(),
            ..Unit::default()
        }
    }
    
    /// Record a successful prediction (Layer 18 feedback).
    pub fn record_success(&mut self) {
        self.success_count += 1;
        self.last_reinforced = Utc::now();
    }
    
    /// Record a failed prediction (Layer 18 feedback).
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_reinforced = Utc::now();
    }
    
    /// Check if pattern is an anchor (high confidence, many successes).
    pub fn is_anchor(&self) -> bool {
        self.success_count > 10 && self.confidence() > 0.8
    }
    
    /// Check if pattern should be pruned (low utility).
    pub fn should_prune(&self, threshold: f32) -> bool {
        let total = self.success_count + self.failure_count;
        total > 5 && self.confidence() < threshold
    }
    
    /// Create from Unit (for loading from memory).
    pub fn from_unit(unit: &Unit) -> Option<Self> {
        // Parse marker format: pattern:<intent>:<tone>:<hash8>
        let parts: Vec<&str> = unit.content.split(':').collect();
        if parts.len() != 4 || parts[0] != "pattern" {
            return None;
        }
        
        let intent_kind = parse_intent_kind(parts[1]);
        let tone_kind = parse_tone_kind(parts[2]);
        
        // Reconstruct signature from unit (partial)
        let signature = ClassificationSignature {
            semantic_centroid: unit.semantic_position,
            byte_length_norm: 0.0,
            sentence_entropy: 0.0,
            token_count_norm: 0.0,
            punct_vector: [0.0; 3],
            urgency_score: 0.0,
            formality_score: 0.0,
            technical_score: 0.0,
            domain_hint: 0.0,
            temporal_cue: 0.0,
        };
        
        Some(Self {
            unit_id: unit.id,
            signature,
            intent_kind,
            tone_kind,
            resolver_mode: ResolverMode::Balanced, // Default, should be stored separately
            success_count: unit.frequency,
            failure_count: 0, // Not stored in unit
            last_reinforced: Utc::now(),
            domain: unit.contexts.first().cloned(),
            memory_channels: unit.memory_channels.clone(),
        })
    }
}

impl Default for ClassificationPattern {
    fn default() -> Self {
        Self {
            unit_id: Uuid::nil(),
            signature: ClassificationSignature::default(),
            intent_kind: IntentKind::Unknown,
            tone_kind: ToneKind::NeutralProfessional,
            resolver_mode: ResolverMode::Balanced,
            success_count: 0,
            failure_count: 0,
            last_reinforced: Utc::now(),
            domain: None,
            memory_channels: vec![MemoryChannel::Main, MemoryChannel::Intent],
        }
    }
}

/// Parse intent kind from string (case-insensitive).
fn parse_intent_kind(s: &str) -> IntentKind {
    match s.to_lowercase().as_str() {
        "greeting" => IntentKind::Greeting,
        "gratitude" => IntentKind::Gratitude,
        "farewell" => IntentKind::Farewell,
        "help" => IntentKind::Help,
        "clarify" => IntentKind::Clarify,
        "rewrite" => IntentKind::Rewrite,
        "verify" => IntentKind::Verify,
        "continue" => IntentKind::Continue,
        "forget" => IntentKind::Forget,
        "question" => IntentKind::Question,
        "summarize" => IntentKind::Summarize,
        "explain" => IntentKind::Explain,
        "compare" => IntentKind::Compare,
        "extract" => IntentKind::Extract,
        "analyze" => IntentKind::Analyze,
        "plan" => IntentKind::Plan,
        "act" => IntentKind::Act,
        "recommend" => IntentKind::Recommend,
        "classify" => IntentKind::Classify,
        "translate" => IntentKind::Translate,
        "debug" => IntentKind::Debug,
        "critique" => IntentKind::Critique,
        "brainstorm" => IntentKind::Brainstorm,
        _ => IntentKind::Unknown,
    }
}

/// Parse tone kind from string (case-insensitive).
fn parse_tone_kind(s: &str) -> ToneKind {
    match s.to_lowercase().as_str() {
        "neutralprofessional" | "neutral_professional" => ToneKind::NeutralProfessional,
        "empathetic" => ToneKind::Empathetic,
        "direct" => ToneKind::Direct,
        "technical" => ToneKind::Technical,
        "casual" => ToneKind::Casual,
        "formal" => ToneKind::Formal,
        _ => ToneKind::NeutralProfessional,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classification::SemanticHasher;
    
    #[test]
    fn test_pattern_creation() {
        let hasher = SemanticHasher::new();
        let sig = ClassificationSignature::compute("Hello there", &hasher);
        
        let pattern = ClassificationPattern::new(
            sig,
            IntentKind::Greeting,
            ToneKind::Casual,
            ResolverMode::Balanced,
            Some("general".to_string()),
        );
        
        assert_eq!(pattern.intent_kind, IntentKind::Greeting);
        assert_eq!(pattern.tone_kind, ToneKind::Casual);
        assert_eq!(pattern.confidence(), 0.5); // No observations yet
    }
    
    #[test]
    fn test_pattern_to_unit_marker() {
        let pattern = ClassificationPattern::default();
        let unit = pattern.to_unit();
        
        assert!(unit.content.starts_with("pattern:"));
        assert!(unit.content.contains("unknown")); // Default intent
        assert_eq!(unit.level, UnitLevel::Pattern);
    }
    
    #[test]
    fn test_pattern_confidence() {
        let mut pattern = ClassificationPattern::default();
        
        assert_eq!(pattern.confidence(), 0.5); // No observations
        
        pattern.record_success();
        pattern.record_success();
        pattern.record_success();
        
        assert_eq!(pattern.confidence(), 1.0); // 3/3 = 1.0
        
        pattern.record_failure();
        
        assert!((pattern.confidence() - 0.75).abs() < 0.01); // 3/4 = 0.75
    }
    
    #[test]
    fn test_pattern_anchor_status() {
        let mut pattern = ClassificationPattern::default();
        
        assert!(!pattern.is_anchor());
        
        for _ in 0..15 {
            pattern.record_success();
        }
        
        assert!(pattern.is_anchor());
    }
    
    #[test]
    fn test_pattern_pruning() {
        let mut pattern = ClassificationPattern::default();
        
        // Not enough observations
        assert!(!pattern.should_prune(0.3));
        
        // Add some failures
        for _ in 0..10 {
            pattern.record_failure();
        }
        
        // Should prune if confidence < 0.3
        assert!(pattern.should_prune(0.3));
    }
    
    #[test]
    fn test_parse_intent_kind() {
        assert_eq!(parse_intent_kind("greeting"), IntentKind::Greeting);
        assert_eq!(parse_intent_kind("GREETING"), IntentKind::Greeting);
        assert_eq!(parse_intent_kind("question"), IntentKind::Question);
        assert_eq!(parse_intent_kind("unknown"), IntentKind::Unknown);
        assert_eq!(parse_intent_kind("invalid"), IntentKind::Unknown);
    }
    
    #[test]
    fn test_parse_tone_kind() {
        assert_eq!(parse_tone_kind("casual"), ToneKind::Casual);
        assert_eq!(parse_tone_kind("FORMAL"), ToneKind::Formal);
        assert_eq!(parse_tone_kind("technical"), ToneKind::Technical);
        assert_eq!(parse_tone_kind("invalid"), ToneKind::NeutralProfessional);
    }
}
