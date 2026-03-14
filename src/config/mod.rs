use crate::types::{MemoryType, ResolverMode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ScoringWeights {
    #[serde(rename = "w_spatial", alias = "spatial")]
    pub spatial: f32,
    #[serde(rename = "w_context", alias = "context")]
    pub context: f32,
    #[serde(rename = "w_sequence", alias = "sequence")]
    pub sequence: f32,
    #[serde(rename = "w_transition", alias = "transition")]
    pub transition: f32,
    #[serde(rename = "w_utility", alias = "utility")]
    pub utility: f32,
    #[serde(rename = "w_confidence", alias = "confidence")]
    pub confidence: f32,
    #[serde(rename = "w_evidence", alias = "evidence")]
    pub evidence: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            spatial: 0.12,
            context: 0.18,
            sequence: 0.16,
            transition: 0.12,
            utility: 0.14,
            confidence: 0.14,
            evidence: 0.14,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalThresholds {
    pub w_entropy: f32,
    pub w_recency: f32,
    pub w_disagreement: f32,
    pub w_cost: f32,
    pub entropy_threshold: f32,
    pub freshness_threshold: f32,
    pub disagreement_threshold: f32,
    #[serde(rename = "retrieve_threshold", alias = "decision_threshold")]
    pub decision_threshold: f32,
    pub cost_penalty: f32,
    pub recency_threshold_hours: u32,
}

impl Default for RetrievalThresholds {
    fn default() -> Self {
        Self {
            w_entropy: 1.0,
            w_recency: 1.0,
            w_disagreement: 1.0,
            w_cost: 1.0,
            entropy_threshold: 0.85,
            freshness_threshold: 0.65,
            disagreement_threshold: 0.35,
            decision_threshold: 1.1,
            cost_penalty: 0.15,
            recency_threshold_hours: 72,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrustConfig {
    pub min_source_trust: f32,
    pub corroboration_bonus: f32,
    pub max_results: usize,
    pub max_query_terms: usize,
    #[serde(default = "default_source_trust")]
    pub default_source_trust: f32,
    #[serde(default = "default_https_bonus")]
    pub https_bonus: f32,
    #[serde(default = "default_allowlist_bonus")]
    pub allowlist_bonus: f32,
    #[serde(default = "default_parser_warning_penalty")]
    pub parser_warning_penalty: f32,
    #[serde(default = "default_min_corroborating_sources")]
    pub min_corroborating_sources: usize,
    #[serde(default = "default_require_https")]
    pub require_https: bool,
    #[serde(default = "default_allowlist_domains")]
    pub allowlist_domains: Vec<String>,
    #[serde(default)]
    pub format_trust_adjustments: BTreeMap<String, f32>,
    #[serde(default)]
    pub content_quality_thresholds: ContentQualityThresholds,
    #[serde(default)]
    pub promotion_rules: PromotionRules,
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            min_source_trust: 0.35,
            corroboration_bonus: 0.08,
            max_results: 5,
            max_query_terms: 10,
            default_source_trust: default_source_trust(),
            https_bonus: default_https_bonus(),
            allowlist_bonus: default_allowlist_bonus(),
            parser_warning_penalty: default_parser_warning_penalty(),
            min_corroborating_sources: default_min_corroborating_sources(),
            require_https: default_require_https(),
            allowlist_domains: default_allowlist_domains(),
            format_trust_adjustments: default_format_trust_adjustments(),
            content_quality_thresholds: ContentQualityThresholds::default(),
            promotion_rules: PromotionRules::default(),
        }
    }
}

fn default_source_trust() -> f32 {
    0.50
}

fn default_https_bonus() -> f32 {
    0.10
}

fn default_allowlist_bonus() -> f32 {
    0.10
}

fn default_parser_warning_penalty() -> f32 {
    0.20
}

fn default_min_corroborating_sources() -> usize {
    2
}

fn default_require_https() -> bool {
    false
}

fn default_allowlist_domains() -> Vec<String> {
    vec![
        "wikimedia.org".to_string(),
        "wikipedia.org".to_string(),
        "wikidata.org".to_string(),
        "archive.org".to_string(),
        "ncbi.nlm.nih.gov".to_string(),
        "pmc.ncbi.nlm.nih.gov".to_string(),
        "nominatim.openstreetmap.org".to_string(),
        "openstreetmap.org".to_string(),
        "dbpedia.org".to_string(),
        "gutenberg.org".to_string(),
    ]
}

fn default_format_trust_adjustments() -> BTreeMap<String, f32> {
    BTreeMap::from([
        ("html_raw".to_string(), -0.30),
        ("qa_json_schema_keys".to_string(), -0.40),
        ("structured_entity".to_string(), 0.20),
        ("code_syntax".to_string(), -0.10),
    ])
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContentQualityThresholds {
    pub min_readability_score: f32,
    pub max_boilerplate_ratio: f32,
    pub min_unique_words_ratio: f32,
}

impl Default for ContentQualityThresholds {
    fn default() -> Self {
        Self {
            min_readability_score: 0.60,
            max_boilerplate_ratio: 0.40,
            min_unique_words_ratio: 0.30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PromotionRules {
    #[serde(default = "default_block_patterns")]
    pub block_patterns: Vec<String>,
    #[serde(default = "default_allow_patterns")]
    pub allow_patterns: Vec<String>,
}

impl Default for PromotionRules {
    fn default() -> Self {
        Self {
            block_patterns: default_block_patterns(),
            allow_patterns: default_allow_patterns(),
        }
    }
}

fn default_block_patterns() -> Vec<String> {
    vec![
        "^\\s*(question|answer|context|id|metadata)\\s*:".to_string(),
        "^\\s*[\\[\\]{}\\\"]+\\s*$".to_string(),
        "^\\s*[{}:,]+\\s*$".to_string(),
    ]
}

fn default_allow_patterns() -> Vec<String> {
    vec!["[a-zA-Z\\s\\.\\,\\?\\!]{20,}".to_string()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntentConfig {
    pub intent_floor_threshold: f32,
    pub intent_ambiguity_margin: f32,
    pub context_reference_bonus: f32,
    pub briefness_keyword_weight: f32,
    pub social_intent_shortcircuit: bool,
    pub min_lexical_tokens: usize,
    pub temporal_cue_weight: f32,
    pub domain_hint_weight: f32,
    pub preference_signal_weight: f32,
    pub certainty_softener_weight: f32,
}

impl Default for IntentConfig {
    fn default() -> Self {
        Self {
            intent_floor_threshold: 0.40,
            intent_ambiguity_margin: 0.15,
            context_reference_bonus: 0.20,
            briefness_keyword_weight: 0.35,
            social_intent_shortcircuit: true,
            min_lexical_tokens: 3,
            temporal_cue_weight: 0.16,
            domain_hint_weight: 0.14,
            preference_signal_weight: 0.14,
            certainty_softener_weight: 0.10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveBehaviorConfig {
    pub intent_profiles: BTreeMap<String, IntentAdaptiveProfile>,
    pub trust_profiles: BTreeMap<String, AdaptiveTrustProfile>,
    pub load_cost: LoadAwareCostConfig,
}

impl AdaptiveBehaviorConfig {
    pub fn intent_profile(&self, name: &str) -> Option<&IntentAdaptiveProfile> {
        self.intent_profiles.get(name)
    }

    pub fn trust_profile(&self, name: &str) -> Option<&AdaptiveTrustProfile> {
        self.trust_profiles.get(name)
    }
}

impl Default for AdaptiveBehaviorConfig {
    fn default() -> Self {
        let mut intent_profiles = BTreeMap::new();
        intent_profiles.insert(
            "casual".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.10,
                    context: 0.25,
                    sequence: 0.15,
                    transition: 0.05,
                    utility: 0.25,
                    confidence: 0.10,
                    evidence: 0.10,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.20,
                    beam_width: 7,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.70,
                    min_confidence_floor: 0.20,
                    mode: Some(ResolverMode::Exploratory),
                },
                shaping: IntentShapingConfig::default(),
            },
        );
        intent_profiles.insert(
            "explanatory".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.08,
                    context: 0.30,
                    sequence: 0.25,
                    transition: 0.12,
                    utility: 0.10,
                    confidence: 0.10,
                    evidence: 0.05,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.10,
                    beam_width: 5,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.30,
                    min_confidence_floor: 0.24,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig::default(),
            },
        );
        intent_profiles.insert(
            "factual".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.10,
                    context: 0.10,
                    sequence: 0.05,
                    transition: 0.04,
                    utility: 0.13,
                    confidence: 0.28,
                    evidence: 0.30,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.05,
                    beam_width: 3,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.10,
                    min_confidence_floor: 0.30,
                    mode: Some(ResolverMode::Deterministic),
                },
                shaping: IntentShapingConfig::default(),
            },
        );
        intent_profiles.insert(
            "procedural".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.08,
                    context: 0.18,
                    sequence: 0.30,
                    transition: 0.17,
                    utility: 0.12,
                    confidence: 0.10,
                    evidence: 0.05,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.08,
                    beam_width: 4,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.22,
                    min_confidence_floor: 0.28,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig::default(),
            },
        );
        intent_profiles.insert(
            "creative".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.08,
                    context: 0.24,
                    sequence: 0.18,
                    transition: 0.08,
                    utility: 0.18,
                    confidence: 0.08,
                    evidence: 0.06,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.22,
                    beam_width: 6,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.75,
                    min_confidence_floor: 0.18,
                    mode: Some(ResolverMode::Exploratory),
                },
                shaping: IntentShapingConfig {
                    allow_semantic_drift: true,
                    drift_tolerance: 0.25,
                    preserve_factual_anchor: true,
                    anchor_trust_threshold: 0.80,
                },
            },
        );
        intent_profiles.insert(
            "brainstorm".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.06,
                    context: 0.30,
                    sequence: 0.12,
                    transition: 0.06,
                    utility: 0.22,
                    confidence: 0.06,
                    evidence: 0.04,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.35,
                    beam_width: 10,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.90,
                    min_confidence_floor: 0.12,
                    mode: Some(ResolverMode::Exploratory),
                },
                shaping: IntentShapingConfig {
                    allow_semantic_drift: true,
                    drift_tolerance: 0.35,
                    preserve_factual_anchor: true,
                    anchor_trust_threshold: 0.80,
                },
            },
        );
        intent_profiles.insert(
            "plan".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.08,
                    context: 0.22,
                    sequence: 0.28,
                    transition: 0.14,
                    utility: 0.12,
                    confidence: 0.10,
                    evidence: 0.06,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.12,
                    beam_width: 5,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.35,
                    min_confidence_floor: 0.22,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig {
                    allow_semantic_drift: false,
                    drift_tolerance: 0.0,
                    preserve_factual_anchor: true,
                    anchor_trust_threshold: 0.75,
                },
            },
        );
        intent_profiles.insert(
            "act".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.06,
                    context: 0.20,
                    sequence: 0.24,
                    transition: 0.16,
                    utility: 0.16,
                    confidence: 0.12,
                    evidence: 0.06,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.08,
                    beam_width: 4,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.25,
                    min_confidence_floor: 0.25,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig {
                    allow_semantic_drift: false,
                    drift_tolerance: 0.0,
                    preserve_factual_anchor: true,
                    anchor_trust_threshold: 0.85,
                },
            },
        );
        intent_profiles.insert(
            "critique".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.08,
                    context: 0.26,
                    sequence: 0.16,
                    transition: 0.10,
                    utility: 0.14,
                    confidence: 0.16,
                    evidence: 0.10,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.10,
                    beam_width: 5,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.30,
                    min_confidence_floor: 0.24,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig {
                    allow_semantic_drift: false,
                    drift_tolerance: 0.0,
                    preserve_factual_anchor: true,
                    anchor_trust_threshold: 0.80,
                },
            },
        );
        intent_profiles.insert(
            "advisory".to_string(),
            IntentAdaptiveProfile {
                scoring: ScoringWeights {
                    spatial: 0.10,
                    context: 0.24,
                    sequence: 0.14,
                    transition: 0.08,
                    utility: 0.10,
                    confidence: 0.18,
                    evidence: 0.16,
                },
                escape: EscapeProfile {
                    stochastic_jump_prob: 0.09,
                    beam_width: 4,
                },
                resolver: AdaptiveResolverProfile {
                    selection_temperature: 0.26,
                    min_confidence_floor: 0.26,
                    mode: Some(ResolverMode::Balanced),
                },
                shaping: IntentShapingConfig::default(),
            },
        );

        let mut trust_profiles = BTreeMap::new();
        trust_profiles.insert(
            "default".to_string(),
            AdaptiveTrustProfile {
                default_source_trust: 0.50,
                min_corroborating_sources: 2,
                require_https: false,
            },
        );
        trust_profiles.insert(
            "high_stakes".to_string(),
            AdaptiveTrustProfile {
                default_source_trust: 0.30,
                min_corroborating_sources: 4,
                require_https: true,
            },
        );

        Self {
            intent_profiles,
            trust_profiles,
            load_cost: LoadAwareCostConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntentShapingConfig {
    pub allow_semantic_drift: bool,
    pub drift_tolerance: f32,
    pub preserve_factual_anchor: bool,
    pub anchor_trust_threshold: f32,
}

impl Default for IntentShapingConfig {
    fn default() -> Self {
        Self {
            allow_semantic_drift: false,
            drift_tolerance: 0.0,
            preserve_factual_anchor: true,
            anchor_trust_threshold: 0.80,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IntentAdaptiveProfile {
    pub scoring: ScoringWeights,
    pub escape: EscapeProfile,
    pub resolver: AdaptiveResolverProfile,
    pub shaping: IntentShapingConfig,
}

impl Default for IntentAdaptiveProfile {
    fn default() -> Self {
        Self {
            scoring: ScoringWeights::default(),
            escape: EscapeProfile::default(),
            resolver: AdaptiveResolverProfile::default(),
            shaping: IntentShapingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EscapeProfile {
    pub stochastic_jump_prob: f32,
    pub beam_width: usize,
}

impl Default for EscapeProfile {
    fn default() -> Self {
        Self {
            stochastic_jump_prob: 0.0,
            beam_width: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveResolverProfile {
    pub selection_temperature: f32,
    pub min_confidence_floor: f32,
    pub mode: Option<ResolverMode>,
}

impl Default for AdaptiveResolverProfile {
    fn default() -> Self {
        Self {
            selection_temperature: 0.7,
            min_confidence_floor: 0.30,
            mode: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AdaptiveTrustProfile {
    pub default_source_trust: f32,
    pub min_corroborating_sources: usize,
    pub require_https: bool,
}

impl Default for AdaptiveTrustProfile {
    fn default() -> Self {
        Self {
            default_source_trust: 0.50,
            min_corroborating_sources: 2,
            require_https: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadAwareCostConfig {
    pub inference_queue_weight: f32,
    pub interactive_training_weight: f32,
    pub silent_batch_weight: f32,
    pub maintenance_weight: f32,
    pub queue_saturation_depth: f32,
    pub max_additive_penalty: f32,
}

impl Default for LoadAwareCostConfig {
    fn default() -> Self {
        Self {
            inference_queue_weight: 0.10,
            interactive_training_weight: 0.08,
            silent_batch_weight: 0.06,
            maintenance_weight: 0.04,
            queue_saturation_depth: 4.0,
            max_additive_penalty: 0.60,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GovernanceConfig {
    pub maintenance_interval_secs: u64,
    pub anchor_reuse_threshold: u64,
    pub core_promotion_threshold: u64,
    pub promotion_min_corroborations: u32,
    pub prune_utility_threshold: f32,
    pub cold_start_prune_utility_threshold: f32,
    pub stable_prune_utility_threshold: f32,
    pub anchor_salience_threshold: f32,
    pub episodic_decay_days: u64,
    pub max_candidate_pool: usize,
    pub train_memory_ceiling_kb: i64,
    pub daily_growth_limit_mb: f32,
    pub bootstrap_seed_frequency: u64,
    pub bootstrap_seed_utility_floor: f32,
    pub cold_start_unit_threshold: usize,
    pub stable_unit_threshold: usize,
    pub cold_start_discovery_frequency: u64,
    pub growth_discovery_frequency: u64,
    pub stable_discovery_frequency: u64,
    pub cold_start_discovery_utility_threshold: f32,
    pub growth_discovery_utility_threshold: f32,
    pub stable_discovery_utility_threshold: f32,
    pub cold_start_candidate_observation_threshold: u64,
    pub growth_candidate_observation_threshold: u64,
    pub stable_candidate_observation_threshold: u64,
    pub candidate_activation_utility_threshold: f32,
    pub candidate_batch_size: usize,
    pub bloom_expected_items: usize,
    pub lfu_decay_factor: f32,
    pub recency_decay_rate: f32,
    pub anchor_protection_grace_days: u32,
    pub cold_archive_threshold_mb: f32,
    pub pollution_detection_enabled: bool,
    pub pollution_min_length: usize,
    pub pollution_edge_trim_limit: usize,
    pub pollution_overlap_threshold: f32,
    pub pollution_quality_margin: f32,
    pub pollution_audit_limit: usize,
    /// Whether to block Intent-channel units from being promoted to Core memory.
    /// Prevents noisy intent signals from polluting long-term core memory.
    pub intent_channel_core_promotion_blocked: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryBudgetTier {
    pub episodic_limit_mb: f32,
    pub core_limit_mb: f32,
    pub candidate_pool_mb: f32,
    pub daily_growth_limit_mb: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryBudgetConfig {
    pub cold_start: MemoryBudgetTier,
    pub growth_phase: MemoryBudgetTier,
    pub stable_phase: MemoryBudgetTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DocumentIngestionConfig {
    pub max_file_size_mb: f32,
    pub max_pdf_pages: usize,
    pub max_docx_xml_chars: usize,
    pub max_chunk_chars: usize,
    pub max_selected_passages: usize,
    pub max_carry_terms: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SourcePolicyConfig {
    #[serde(default = "default_extraction_mode_passthrough")]
    pub extraction_mode: String,
    #[serde(default)]
    pub fields_to_extract: Vec<String>,
    #[serde(default)]
    pub fields_to_skip: Vec<String>,
    #[serde(default)]
    pub strip_elements: Vec<String>,
    #[serde(default)]
    pub keep_only: Vec<String>,
    #[serde(default)]
    pub extract: Vec<String>,
    #[serde(default)]
    pub skip: Vec<String>,
    #[serde(default)]
    pub languages: Vec<String>,
    #[serde(default)]
    pub flatten_structure: bool,
    #[serde(default)]
    pub skip_internal_ids: bool,
    #[serde(default)]
    pub normalize_whitespace: bool,
    #[serde(default)]
    pub trim_lines: bool,
    #[serde(default)]
    pub merge_to_core: bool,
    #[serde(default)]
    pub trust_bonus: f32,
    #[serde(default)]
    pub memory_type: Option<MemoryType>,
    #[serde(default)]
    pub decay_days: Option<u64>,
    #[serde(default)]
    pub usage_tag: Option<String>,
    #[serde(default)]
    pub min_corroborations: Option<usize>,
}

impl Default for SourcePolicyConfig {
    fn default() -> Self {
        Self {
            extraction_mode: default_extraction_mode_passthrough(),
            fields_to_extract: Vec::new(),
            fields_to_skip: Vec::new(),
            strip_elements: Vec::new(),
            keep_only: Vec::new(),
            extract: Vec::new(),
            skip: Vec::new(),
            languages: Vec::new(),
            flatten_structure: false,
            skip_internal_ids: false,
            normalize_whitespace: true,
            trim_lines: true,
            merge_to_core: false,
            trust_bonus: 0.0,
            memory_type: None,
            decay_days: None,
            usage_tag: None,
            min_corroborations: None,
        }
    }
}

fn default_extraction_mode_passthrough() -> String {
    "passthrough".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SourcePoliciesConfig {
    #[serde(default = "default_html_source_policy")]
    pub html: SourcePolicyConfig,
    #[serde(default = "default_qa_json_source_policy")]
    pub qa_json: SourcePolicyConfig,
    #[serde(default = "default_structured_json_source_policy")]
    pub structured_json: SourcePolicyConfig,
    #[serde(default = "default_plain_text_source_policy")]
    pub plain_text: SourcePolicyConfig,
    #[serde(default = "default_code_source_policy")]
    pub code: SourcePolicyConfig,
    #[serde(default = "default_wikipedia_xml_source_policy")]
    pub wikipedia_xml: SourcePolicyConfig,
    #[serde(default = "default_wikidata_source_policy")]
    pub wikidata_truthy: SourcePolicyConfig,
    #[serde(default = "default_openapi_source_policy")]
    pub openapi_spec: SourcePolicyConfig,
    #[serde(default = "default_common_crawl_source_policy")]
    pub common_crawl_wet: SourcePolicyConfig,
}

impl Default for SourcePoliciesConfig {
    fn default() -> Self {
        Self {
            html: default_html_source_policy(),
            qa_json: default_qa_json_source_policy(),
            structured_json: default_structured_json_source_policy(),
            plain_text: default_plain_text_source_policy(),
            code: default_code_source_policy(),
            wikipedia_xml: default_wikipedia_xml_source_policy(),
            wikidata_truthy: default_wikidata_source_policy(),
            openapi_spec: default_openapi_source_policy(),
            common_crawl_wet: default_common_crawl_source_policy(),
        }
    }
}

fn default_html_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "readability".to_string(),
        strip_elements: vec![
            "nav".to_string(),
            "footer".to_string(),
            "script".to_string(),
            "style".to_string(),
            "aside".to_string(),
            "ads".to_string(),
        ],
        keep_only: vec![
            "article".to_string(),
            "main".to_string(),
            "section".to_string(),
            "p".to_string(),
            "h1".to_string(),
            "h2".to_string(),
            "h3".to_string(),
            "li".to_string(),
        ],
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(7),
        ..SourcePolicyConfig::default()
    }
}

fn default_qa_json_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "field_select".to_string(),
        fields_to_extract: vec![
            "question".to_string(),
            "context".to_string(),
            "instruction".to_string(),
            "prompt".to_string(),
            "input".to_string(),
            "query".to_string(),
            "reasoning".to_string(),
            "rationale".to_string(),
            "explanation".to_string(),
            "analysis".to_string(),
            "solution".to_string(),
            "proof".to_string(),
            "messages".to_string(),
            "turns".to_string(),
            "dialogue".to_string(),
            "conversation".to_string(),
        ],
        fields_to_skip: vec![
            "answer".to_string(),
            "answers".to_string(),
            "wellFormedAnswers".to_string(),
            "output".to_string(),
            "completion".to_string(),
            "target".to_string(),
            "label".to_string(),
            "id".to_string(),
            "metadata".to_string(),
            "options".to_string(),
            "choices".to_string(),
            "candidates".to_string(),
        ],
        flatten_structure: true,
        trust_bonus: 0.10,
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(14),
        usage_tag: Some("intent_training".to_string()),
        ..SourcePolicyConfig::default()
    }
}

fn default_structured_json_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "entity_extract".to_string(),
        fields_to_extract: vec![
            "question".to_string(),
            "context".to_string(),
            "instruction".to_string(),
            "prompt".to_string(),
            "input".to_string(),
            "query".to_string(),
            "label".to_string(),
            "description".to_string(),
            "aliases".to_string(),
            "claims".to_string(),
            "summary".to_string(),
            "reasoning".to_string(),
            "rationale".to_string(),
            "explanation".to_string(),
            "analysis".to_string(),
            "solution".to_string(),
            "proof".to_string(),
            "messages".to_string(),
            "turns".to_string(),
            "dialogue".to_string(),
            "conversation".to_string(),
            "steps".to_string(),
            "instructions".to_string(),
            "procedure".to_string(),
            "actions".to_string(),
            "workflow".to_string(),
            "plan".to_string(),
            "output".to_string(),
            "response".to_string(),
            "completion".to_string(),
            "target".to_string(),
            "label".to_string(),
        ],
        fields_to_skip: vec![
            "id".to_string(),
            "metadata".to_string(),
            "timestamp".to_string(),
        ],
        skip_internal_ids: true,
        trust_bonus: 0.20,
        merge_to_core: true,
        memory_type: Some(MemoryType::Core),
        min_corroborations: Some(1),
        ..SourcePolicyConfig::default()
    }
}

fn default_plain_text_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "passthrough".to_string(),
        normalize_whitespace: true,
        trim_lines: true,
        trust_bonus: 0.10,
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(30),
        ..SourcePolicyConfig::default()
    }
}

fn default_code_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "code_strip".to_string(),
        extract: vec![
            "docstrings".to_string(),
            "comments".to_string(),
            "readme".to_string(),
            "docs".to_string(),
        ],
        skip: vec![
            "syntax".to_string(),
            "indentation".to_string(),
            "punctuation".to_string(),
            "keywords".to_string(),
        ],
        languages: vec![
            "python".to_string(),
            "rust".to_string(),
            "javascript".to_string(),
            "typescript".to_string(),
        ],
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(14),
        usage_tag: Some("explanation_patterns".to_string()),
        ..SourcePolicyConfig::default()
    }
}

fn default_wikipedia_xml_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "article_text".to_string(),
        fields_to_extract: vec!["title".to_string(), "text".to_string()],
        fields_to_skip: vec![
            "revision".to_string(),
            "contributor".to_string(),
            "timestamp".to_string(),
        ],
        trust_bonus: 0.10,
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(30),
        usage_tag: Some("core_knowledge".to_string()),
        ..SourcePolicyConfig::default()
    }
}

fn default_wikidata_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "entity_extract".to_string(),
        fields_to_extract: vec![
            "label".to_string(),
            "description".to_string(),
            "claims".to_string(),
        ],
        skip_internal_ids: true,
        trust_bonus: 0.20,
        merge_to_core: true,
        memory_type: Some(MemoryType::Core),
        min_corroborations: Some(1),
        ..SourcePolicyConfig::default()
    }
}

fn default_openapi_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "entity_extract".to_string(),
        fields_to_extract: vec![
            "operationId".to_string(),
            "summary".to_string(),
            "description".to_string(),
            "parameters".to_string(),
            "requestBody".to_string(),
            "responses".to_string(),
        ],
        fields_to_skip: vec!["id".to_string(), "timestamp".to_string()],
        trust_bonus: 0.20,
        memory_type: Some(MemoryType::Core),
        usage_tag: Some("procedure_knowledge".to_string()),
        ..SourcePolicyConfig::default()
    }
}

fn default_common_crawl_source_policy() -> SourcePolicyConfig {
    SourcePolicyConfig {
        extraction_mode: "readability".to_string(),
        normalize_whitespace: true,
        trim_lines: true,
        memory_type: Some(MemoryType::Episodic),
        decay_days: Some(7),
        ..SourcePolicyConfig::default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SilentTrainingFormatRule {
    #[serde(default)]
    pub ingest_fields: Vec<String>,
    #[serde(default)]
    pub skip_fields: Vec<String>,
    #[serde(default)]
    pub extract_sections: Vec<String>,
    #[serde(default)]
    pub skip_metadata: Vec<String>,
    #[serde(default)]
    pub extract: Vec<String>,
    #[serde(default)]
    pub skip: Vec<String>,
    #[serde(default)]
    pub use_as: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MemoryPlacementRule {
    #[serde(default)]
    pub memory_type: Option<MemoryType>,
    #[serde(default)]
    pub decay_days: Option<u64>,
    #[serde(default)]
    pub min_corroborations: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SilentTrainingConfig {
    pub max_memory_delta_mb: f32,
    pub progress_interval_sec: u64,
    pub batch_size_sources: usize,
    pub merge_to_core: bool,
    #[serde(default)]
    pub parallel: ParallelTrainingConfig,
    #[serde(default)]
    pub format_ingestion_rules: BTreeMap<String, SilentTrainingFormatRule>,
    #[serde(default)]
    pub memory_placement: BTreeMap<String, MemoryPlacementRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ParallelTrainingConfig {
    pub enabled: bool,
    pub worker_count: usize,
    pub queue_capacity: usize,
    pub queue_capacity_per_worker: usize,
    pub commit_batch_size: usize,
    pub commit_flush_interval_ms: u64,
    pub total_memory_limit_mb: f32,
    pub non_worker_memory_reserve_mb: f32,
    pub local_shard_soft_limit_mb: f32,
    pub local_shard_hard_limit_mb: f32,
}

impl Default for ParallelTrainingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            worker_count: 0,
            queue_capacity: 64,
            queue_capacity_per_worker: 4,
            commit_batch_size: 8,
            commit_flush_interval_ms: 250,
            total_memory_limit_mb: 2560.0,
            non_worker_memory_reserve_mb: 1280.0,
            local_shard_soft_limit_mb: 96.0,
            local_shard_hard_limit_mb: 128.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HuggingFaceStreamingConfig {
    pub initial_rows_per_pull: usize,
    pub min_rows_per_pull: usize,
    pub max_rows_per_pull: usize,
    pub fast_pull_threshold_ms: u64,
    pub slow_pull_threshold_ms: u64,
    pub increase_factor: f32,
    pub decrease_factor: f32,
    pub request_delay_ms: u64,
    pub max_retries: u32,
    pub retry_base_delay_ms: u64,
    pub retry_max_delay_ms: u64,
}

impl Default for HuggingFaceStreamingConfig {
    fn default() -> Self {
        Self {
            initial_rows_per_pull: 100,
            min_rows_per_pull: 100,
            max_rows_per_pull: 300,
            fast_pull_threshold_ms: 1_200,
            slow_pull_threshold_ms: 3_500,
            increase_factor: 1.25,
            decrease_factor: 0.5,
            request_delay_ms: 200,
            max_retries: 6,
            retry_base_delay_ms: 1_000,
            retry_max_delay_ms: 15_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct UnitBuilderConfig {
    pub min_frequency_threshold: u64,
    pub initial_utility_score: f32,
    pub edit_distance_cluster_threshold: u32,
    pub rolling_hash_window_sizes: Vec<usize>,
    pub max_activated_units: usize,
    pub punctuation_ratio_limit: f32,
    pub hash_base: u64,
    pub min_fragment_length: usize,
    pub utf8_recovery_min_bytes: usize,
    pub utility_compression_gain_cap: f32,
    pub utility_span_weight: f32,
    pub utility_frequency_weight: f32,
    pub utility_density_weight: f32,
    pub utility_base_scale: f32,
    pub utility_full_boundary_bonus: f32,
    pub utility_edge_boundary_bonus: f32,
    pub utility_no_boundary_penalty: f32,
    pub salience_base: f32,
    pub salience_digit_weight: f32,
    pub salience_upper_weight: f32,
    pub salience_reuse_divisor: f32,
    pub confidence_base: f32,
    pub confidence_reuse_divisor: f32,
    pub confidence_reuse_cap: f32,
    pub confidence_length_weight: f32,
    pub confidence_full_boundary_bonus: f32,
    pub confidence_edge_boundary_bonus: f32,
    pub confidence_no_boundary_penalty: f32,
    pub global_corroboration_frequency_threshold: u64,
    pub global_corroboration_utility_threshold: f32,
    pub global_corroboration_confidence_threshold: f32,
    pub phrase_entity_promotion_frequency: u64,
    pub phrase_entity_promotion_salience: f32,
    pub phrase_entity_promotion_confidence: f32,
}

impl Default for UnitBuilderConfig {
    fn default() -> Self {
        Self {
            min_frequency_threshold: 2,
            initial_utility_score: 0.50,
            edit_distance_cluster_threshold: 1,
            rolling_hash_window_sizes: vec![2, 3, 4, 5, 6, 7, 8],
            max_activated_units: 96,
            punctuation_ratio_limit: 0.55,
            hash_base: 257,
            min_fragment_length: 4,
            utf8_recovery_min_bytes: 2,
            utility_compression_gain_cap: 1.2,
            utility_span_weight: 0.45,
            utility_frequency_weight: 0.30,
            utility_density_weight: 0.22,
            utility_base_scale: 0.16,
            utility_full_boundary_bonus: 0.18,
            utility_edge_boundary_bonus: 0.05,
            utility_no_boundary_penalty: -0.12,
            salience_base: 0.18,
            salience_digit_weight: 0.35,
            salience_upper_weight: 0.20,
            salience_reuse_divisor: 6.0,
            confidence_base: 0.22,
            confidence_reuse_divisor: 2.0,
            confidence_reuse_cap: 0.45,
            confidence_length_weight: 0.30,
            confidence_full_boundary_bonus: 0.10,
            confidence_edge_boundary_bonus: 0.03,
            confidence_no_boundary_penalty: -0.08,
            global_corroboration_frequency_threshold: 6,
            global_corroboration_utility_threshold: 0.45,
            global_corroboration_confidence_threshold: 0.45,
            phrase_entity_promotion_frequency: 4,
            phrase_entity_promotion_salience: 0.70,
            phrase_entity_promotion_confidence: 0.55,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticMapConfig {
    pub preferred_spacing_k: f32,
    pub max_displacement_per_iteration: f32,
    pub convergence_tolerance: f32,
    pub max_layout_iterations: u32,
    pub energy_rollback_threshold: u32,
    pub attractive_force_coefficient: f32,
    pub repulsive_force_coefficient: f32,
    pub layout_boundary: f32,
    pub max_layout_units: usize,
    pub spatial_cell_size: f32,
    pub neighbor_radius: f32,
}

impl Default for SemanticMapConfig {
    fn default() -> Self {
        Self {
            preferred_spacing_k: 1.0,
            max_displacement_per_iteration: 1.75,
            convergence_tolerance: 0.001,
            max_layout_iterations: 24,
            energy_rollback_threshold: 3,
            attractive_force_coefficient: 1.0,
            repulsive_force_coefficient: 1.0,
            layout_boundary: 128.0,
            max_layout_units: 256,
            spatial_cell_size: 4.0,
            neighbor_radius: 6.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EvidenceMergeConfig {
    pub trust_weight: f32,
    pub recency_weight: f32,
    pub agreement_weight: f32,
    pub ambiguity_margin: f32,
}

impl Default for EvidenceMergeConfig {
    fn default() -> Self {
        Self {
            trust_weight: 0.50,
            recency_weight: 0.30,
            agreement_weight: 0.20,
            ambiguity_margin: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QueryBuilderConfig {
    pub max_query_expansion_terms: usize,
    pub pii_stripping_aggressiveness: String,
}

impl Default for QueryBuilderConfig {
    fn default() -> Self {
        Self {
            max_query_expansion_terms: 5,
            pii_stripping_aggressiveness: "standard".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalIoConfig {
    pub retrieval_timeout_ms: u64,
    pub max_retrieval_results: usize,
    pub cache_ttl_seconds: u64,
    pub enable_retrieval: bool,
    pub max_retries: usize,
}

impl Default for RetrievalIoConfig {
    fn default() -> Self {
        Self {
            retrieval_timeout_ms: 2_000,
            max_retrieval_results: 10,
            cache_ttl_seconds: 3_600,
            enable_retrieval: true,
            max_retries: 2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FineResolverConfig {
    pub selection_temperature: f32,
    pub min_confidence_floor: f32,
    pub evidence_answer_confidence_threshold: f32,
    pub creative_drift_tolerance: f32,
    pub factual_corruption_threshold: f32,
}

impl Default for FineResolverConfig {
    fn default() -> Self {
        Self {
            selection_temperature: 0.7,
            min_confidence_floor: 0.22,
            evidence_answer_confidence_threshold: 0.22,
            creative_drift_tolerance: 0.25,
            factual_corruption_threshold: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TelemetryConfig {
    pub log_score_breakdowns: bool,
    pub telemetry_sample_rate: f32,
    pub warning_threshold_utility_drop: f32,
    pub observation_log_path: Option<String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            log_score_breakdowns: true,
            telemetry_sample_rate: 1.0,
            warning_threshold_utility_drop: 0.30,
            observation_log_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingPhaseConfig {
    pub max_memory_delta_mb: f32,
    pub daily_growth_limit_mb: f32,
    pub merge_to_core: bool,
    pub min_unit_discovery_efficiency: Option<f32>,
    pub min_semantic_routing_accuracy: Option<f32>,
}

impl Default for TrainingPhaseConfig {
    fn default() -> Self {
        Self {
            max_memory_delta_mb: 0.5,
            daily_growth_limit_mb: 1.0,
            merge_to_core: false,
            min_unit_discovery_efficiency: None,
            min_semantic_routing_accuracy: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingPhaseOverridesConfig {
    pub bootstrap: TrainingPhaseConfig,
    pub validation: TrainingPhaseConfig,
    pub expansion: TrainingPhaseConfig,
    pub dry_run: TrainingPhaseConfig,
    pub huggingface: TrainingPhaseConfig,
}

impl Default for TrainingPhaseOverridesConfig {
    fn default() -> Self {
        Self {
            bootstrap: TrainingPhaseConfig {
                max_memory_delta_mb: 5.0,
                daily_growth_limit_mb: 50.0,
                ..TrainingPhaseConfig::default()
            },
            validation: TrainingPhaseConfig {
                max_memory_delta_mb: 2.0,
                daily_growth_limit_mb: 10.0,
                ..TrainingPhaseConfig::default()
            },
            expansion: TrainingPhaseConfig {
                min_unit_discovery_efficiency: Some(0.90),
                min_semantic_routing_accuracy: Some(0.85),
                ..TrainingPhaseConfig::default()
            },
            dry_run: TrainingPhaseConfig {
                max_memory_delta_mb: 5.0,
                daily_growth_limit_mb: 50.0,
                ..TrainingPhaseConfig::default()
            },
            huggingface: TrainingPhaseConfig {
                max_memory_delta_mb: 2.0,
                daily_growth_limit_mb: 10.0,
                ..TrainingPhaseConfig::default()
            },
        }
    }
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            maintenance_interval_secs: 30,
            anchor_reuse_threshold: 3,
            core_promotion_threshold: 6,
            promotion_min_corroborations: 3,
            prune_utility_threshold: 0.12,
            cold_start_prune_utility_threshold: 0.08,
            stable_prune_utility_threshold: 0.18,
            anchor_salience_threshold: 0.70,
            episodic_decay_days: 30,
            max_candidate_pool: 48,
            train_memory_ceiling_kb: 512,
            daily_growth_limit_mb: 1.0,
            bootstrap_seed_frequency: 100,
            bootstrap_seed_utility_floor: 0.30,
            cold_start_unit_threshold: 1_000,
            stable_unit_threshold: 10_000,
            cold_start_discovery_frequency: 1,
            growth_discovery_frequency: 2,
            stable_discovery_frequency: 3,
            cold_start_discovery_utility_threshold: 0.18,
            growth_discovery_utility_threshold: 0.28,
            stable_discovery_utility_threshold: 0.42,
            cold_start_candidate_observation_threshold: 2,
            growth_candidate_observation_threshold: 3,
            stable_candidate_observation_threshold: 4,
            candidate_activation_utility_threshold: 0.55,
            candidate_batch_size: 128,
            bloom_expected_items: 1_000_000,
            lfu_decay_factor: 0.95,
            recency_decay_rate: 0.01,
            anchor_protection_grace_days: 14,
            cold_archive_threshold_mb: 100.0,
            pollution_detection_enabled: true,
            pollution_min_length: 4,
            pollution_edge_trim_limit: 3,
            pollution_overlap_threshold: 0.70,
            pollution_quality_margin: 0.08,
            pollution_audit_limit: 64,
            intent_channel_core_promotion_blocked: true,
        }
    }
}

impl Default for MemoryBudgetTier {
    fn default() -> Self {
        Self {
            episodic_limit_mb: 50.0,
            core_limit_mb: 100.0,
            candidate_pool_mb: 10.0,
            daily_growth_limit_mb: 1.0,
        }
    }
}

impl Default for MemoryBudgetConfig {
    fn default() -> Self {
        Self {
            cold_start: MemoryBudgetTier {
                episodic_limit_mb: 50.0,
                core_limit_mb: 10.0,
                candidate_pool_mb: 20.0,
                daily_growth_limit_mb: 10.0,
            },
            growth_phase: MemoryBudgetTier {
                episodic_limit_mb: 100.0,
                core_limit_mb: 50.0,
                candidate_pool_mb: 30.0,
                daily_growth_limit_mb: 5.0,
            },
            stable_phase: MemoryBudgetTier::default(),
        }
    }
}

impl Default for DocumentIngestionConfig {
    fn default() -> Self {
        Self {
            max_file_size_mb: 10.0,
            max_pdf_pages: 32,
            max_docx_xml_chars: 5_000_000,
            max_chunk_chars: 420,
            max_selected_passages: 4,
            max_carry_terms: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub version: String,
    pub schema_version: u32,
    #[serde(rename = "layer_2_unit_builder", alias = "builder")]
    pub builder: UnitBuilderConfig,
    #[serde(rename = "layer_5_semantic_map", alias = "semantic_map")]
    pub semantic_map: SemanticMapConfig,
    #[serde(rename = "layer_9_retrieval_gating", alias = "retrieval")]
    pub retrieval: RetrievalThresholds,
    #[serde(rename = "layer_13_evidence_merge", alias = "evidence_merge")]
    pub evidence_merge: EvidenceMergeConfig,
    #[serde(rename = "layer_14_candidate_scoring", alias = "scoring")]
    pub scoring: ScoringWeights,
    #[serde(rename = "layer_19_trust_heuristics", alias = "trust")]
    pub trust: TrustConfig,
    pub intent: IntentConfig,
    #[serde(default)]
    pub adaptive_behavior: AdaptiveBehaviorConfig,
    #[serde(rename = "layer_21_memory_governance", alias = "governance")]
    pub governance: GovernanceConfig,
    pub memory_budgets: MemoryBudgetConfig,
    pub document: DocumentIngestionConfig,
    #[serde(rename = "layer_10_query_builder", alias = "query")]
    pub query: QueryBuilderConfig,
    #[serde(rename = "layer_11_retrieval", alias = "retrieval_io")]
    pub retrieval_io: RetrievalIoConfig,
    #[serde(rename = "layer_16_fine_resolver", alias = "resolver")]
    pub resolver: FineResolverConfig,
    #[serde(rename = "layer_20_telemetry", alias = "telemetry")]
    pub telemetry: TelemetryConfig,
    #[serde(default)]
    pub training_phases: TrainingPhaseOverridesConfig,
    #[serde(default)]
    pub source_policies: SourcePoliciesConfig,
    #[serde(default)]
    pub silent_training: SilentTrainingConfig,
    #[serde(default)]
    pub huggingface_streaming: HuggingFaceStreamingConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            schema_version: 1,
            builder: UnitBuilderConfig::default(),
            semantic_map: SemanticMapConfig::default(),
            retrieval: RetrievalThresholds::default(),
            evidence_merge: EvidenceMergeConfig::default(),
            scoring: ScoringWeights::default(),
            trust: TrustConfig::default(),
            intent: IntentConfig::default(),
            adaptive_behavior: AdaptiveBehaviorConfig::default(),
            governance: GovernanceConfig::default(),
            memory_budgets: MemoryBudgetConfig::default(),
            document: DocumentIngestionConfig::default(),
            query: QueryBuilderConfig::default(),
            retrieval_io: RetrievalIoConfig::default(),
            resolver: FineResolverConfig::default(),
            telemetry: TelemetryConfig::default(),
            training_phases: TrainingPhaseOverridesConfig::default(),
            source_policies: SourcePoliciesConfig::default(),
            silent_training: default_silent_training_config(),
            huggingface_streaming: HuggingFaceStreamingConfig::default(),
        }
    }
}

fn default_silent_training_config() -> SilentTrainingConfig {
    let format_ingestion_rules = BTreeMap::from([
        (
            "qa_json".to_string(),
            SilentTrainingFormatRule {
                ingest_fields: vec![
                    "question".to_string(),
                    "context".to_string(),
                    "instruction".to_string(),
                    "reasoning".to_string(),
                ],
                skip_fields: vec![
                    "answer".to_string(),
                    "id".to_string(),
                    "metadata".to_string(),
                    "label".to_string(),
                ],
                extract_sections: Vec::new(),
                skip_metadata: Vec::new(),
                extract: Vec::new(),
                skip: Vec::new(),
                use_as: Some("intent_training".to_string()),
            },
        ),
        (
            "wikipedia_xml".to_string(),
            SilentTrainingFormatRule {
                ingest_fields: Vec::new(),
                skip_fields: Vec::new(),
                extract_sections: vec!["title".to_string(), "text".to_string()],
                skip_metadata: vec!["revision".to_string(), "contributor".to_string()],
                extract: Vec::new(),
                skip: Vec::new(),
                use_as: Some("core_knowledge".to_string()),
            },
        ),
        (
            "code".to_string(),
            SilentTrainingFormatRule {
                ingest_fields: Vec::new(),
                skip_fields: Vec::new(),
                extract_sections: Vec::new(),
                skip_metadata: Vec::new(),
                extract: vec!["docstrings".to_string(), "comments".to_string()],
                skip: vec!["syntax_tokens".to_string()],
                use_as: Some("explanation_patterns".to_string()),
            },
        ),
    ]);
    let memory_placement = BTreeMap::from([
        (
            "html".to_string(),
            MemoryPlacementRule {
                memory_type: Some(MemoryType::Episodic),
                decay_days: Some(7),
                min_corroborations: None,
            },
        ),
        (
            "qa_json".to_string(),
            MemoryPlacementRule {
                memory_type: Some(MemoryType::Episodic),
                decay_days: Some(14),
                min_corroborations: None,
            },
        ),
        (
            "wikidata_truthy".to_string(),
            MemoryPlacementRule {
                memory_type: Some(MemoryType::Core),
                decay_days: None,
                min_corroborations: Some(1),
            },
        ),
    ]);
    SilentTrainingConfig {
        max_memory_delta_mb: 0.5,
        progress_interval_sec: 10,
        batch_size_sources: 50,
        merge_to_core: false,
        parallel: ParallelTrainingConfig::default(),
        format_ingestion_rules,
        memory_placement,
    }
}

impl EngineConfig {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path)
            .map_err(|err| format!("failed to read config {}: {err}", path.display()))?;
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or_default();
        let config: Self = match extension {
            "yaml" | "yml" => serde_yaml::from_str(&raw)
                .map_err(|err| format!("failed to parse yaml config {}: {err}", path.display()))?,
            "json" => serde_json::from_str(&raw)
                .map_err(|err| format!("failed to parse json config {}: {err}", path.display()))?,
            _ => serde_yaml::from_str(&raw)
                .or_else(|_| serde_json::from_str(&raw))
                .map_err(|err| format!("failed to parse config {}: {err}", path.display()))?,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn load_default_file() -> Self {
        if let Ok(path) = std::env::var("SPSE_CONFIG_PATH") {
            return Self::load_from_file(path).unwrap_or_default();
        }

        let default_path = Path::new("config/config.yaml");
        if default_path.exists() {
            Self::load_from_file(default_path).unwrap_or_default()
        } else {
            let legacy_path = Path::new("spse_config.json");
            if legacy_path.exists() {
                Self::load_from_file(legacy_path).unwrap_or_default()
            } else {
                Self::default()
            }
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_range(
            "layer_9_retrieval_gating.entropy_threshold",
            self.retrieval.entropy_threshold,
            0.50,
            0.95,
        )?;
        validate_range(
            "layer_9_retrieval_gating.retrieve_threshold",
            self.retrieval.decision_threshold,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_5_semantic_map.max_displacement_per_iteration",
            self.semantic_map.max_displacement_per_iteration,
            0.01,
            5.0,
        )?;
        validate_range(
            "layer_21_memory_governance.daily_growth_limit_mb",
            self.governance.daily_growth_limit_mb,
            0.1,
            50.0,
        )?;
        if self.governance.pollution_min_length == 0 {
            return Err("layer_21_memory_governance.pollution_min_length must be >= 1".to_string());
        }
        if self.governance.pollution_edge_trim_limit == 0 {
            return Err(
                "layer_21_memory_governance.pollution_edge_trim_limit must be >= 1".to_string(),
            );
        }
        if self.governance.pollution_audit_limit == 0 {
            return Err(
                "layer_21_memory_governance.pollution_audit_limit must be >= 1".to_string(),
            );
        }
        validate_range(
            "layer_21_memory_governance.pollution_overlap_threshold",
            self.governance.pollution_overlap_threshold,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_21_memory_governance.pollution_quality_margin",
            self.governance.pollution_quality_margin,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_16_fine_resolver.selection_temperature",
            self.resolver.selection_temperature,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_20_telemetry.telemetry_sample_rate",
            self.telemetry.telemetry_sample_rate,
            0.0,
            1.0,
        )?;
        validate_range(
            "huggingface_streaming.increase_factor",
            self.huggingface_streaming.increase_factor,
            1.0,
            4.0,
        )?;
        validate_range(
            "huggingface_streaming.decrease_factor",
            self.huggingface_streaming.decrease_factor,
            0.1,
            1.0,
        )?;
        if self.huggingface_streaming.min_rows_per_pull == 0 {
            return Err("huggingface_streaming.min_rows_per_pull must be >= 1".to_string());
        }
        if self.huggingface_streaming.max_rows_per_pull
            < self.huggingface_streaming.min_rows_per_pull
        {
            return Err(
                "huggingface_streaming.max_rows_per_pull must be >= min_rows_per_pull".to_string(),
            );
        }
        if self.huggingface_streaming.initial_rows_per_pull
            < self.huggingface_streaming.min_rows_per_pull
            || self.huggingface_streaming.initial_rows_per_pull
                > self.huggingface_streaming.max_rows_per_pull
        {
            return Err(
                "huggingface_streaming.initial_rows_per_pull must be within min/max bounds"
                    .to_string(),
            );
        }
        if self.huggingface_streaming.retry_max_delay_ms
            < self.huggingface_streaming.retry_base_delay_ms
        {
            return Err(
                "huggingface_streaming.retry_max_delay_ms must be >= retry_base_delay_ms"
                    .to_string(),
            );
        }
        for (profile_name, profile) in &self.adaptive_behavior.intent_profiles {
            validate_range(
                &format!(
                    "adaptive_behavior.intent_profiles.{profile_name}.escape.stochastic_jump_prob"
                ),
                profile.escape.stochastic_jump_prob,
                0.0,
                1.0,
            )?;
            if profile.escape.beam_width == 0 {
                return Err(format!(
                    "adaptive_behavior.intent_profiles.{profile_name}.escape.beam_width must be >= 1"
                ));
            }
            validate_range(
                &format!(
                    "adaptive_behavior.intent_profiles.{profile_name}.resolver.selection_temperature"
                ),
                profile.resolver.selection_temperature,
                0.0,
                2.0,
            )?;
            validate_range(
                &format!(
                    "adaptive_behavior.intent_profiles.{profile_name}.resolver.min_confidence_floor"
                ),
                profile.resolver.min_confidence_floor,
                0.0,
                1.0,
            )?;
        }
        validate_range(
            "adaptive_behavior.load_cost.max_additive_penalty",
            self.adaptive_behavior.load_cost.max_additive_penalty,
            0.0,
            1.0,
        )?;
        validate_range(
            "adaptive_behavior.load_cost.queue_saturation_depth",
            self.adaptive_behavior.load_cost.queue_saturation_depth,
            1.0,
            128.0,
        )?;
        if self.builder.rolling_hash_window_sizes.is_empty() {
            return Err(
                "layer_2_unit_builder.rolling_hash_window_sizes must not be empty".to_string(),
            );
        }
        if self.builder.min_fragment_length == 0 {
            return Err("layer_2_unit_builder.min_fragment_length must be >= 1".to_string());
        }
        if self.builder.utf8_recovery_min_bytes == 0 {
            return Err("layer_2_unit_builder.utf8_recovery_min_bytes must be >= 1".to_string());
        }
        validate_range(
            "layer_2_unit_builder.punctuation_ratio_limit",
            self.builder.punctuation_ratio_limit,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_compression_gain_cap",
            self.builder.utility_compression_gain_cap,
            0.0,
            5.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_span_weight",
            self.builder.utility_span_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_frequency_weight",
            self.builder.utility_frequency_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_density_weight",
            self.builder.utility_density_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_base_scale",
            self.builder.utility_base_scale,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_full_boundary_bonus",
            self.builder.utility_full_boundary_bonus,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_edge_boundary_bonus",
            self.builder.utility_edge_boundary_bonus,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.utility_no_boundary_penalty",
            self.builder.utility_no_boundary_penalty,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.salience_base",
            self.builder.salience_base,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.salience_digit_weight",
            self.builder.salience_digit_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.salience_upper_weight",
            self.builder.salience_upper_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.salience_reuse_divisor",
            self.builder.salience_reuse_divisor,
            0.1,
            64.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_base",
            self.builder.confidence_base,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_reuse_divisor",
            self.builder.confidence_reuse_divisor,
            0.1,
            32.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_reuse_cap",
            self.builder.confidence_reuse_cap,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_length_weight",
            self.builder.confidence_length_weight,
            0.0,
            2.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_full_boundary_bonus",
            self.builder.confidence_full_boundary_bonus,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_edge_boundary_bonus",
            self.builder.confidence_edge_boundary_bonus,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.confidence_no_boundary_penalty",
            self.builder.confidence_no_boundary_penalty,
            -1.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.global_corroboration_utility_threshold",
            self.builder.global_corroboration_utility_threshold,
            0.0,
            2.5,
        )?;
        validate_range(
            "layer_2_unit_builder.global_corroboration_confidence_threshold",
            self.builder.global_corroboration_confidence_threshold,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.phrase_entity_promotion_salience",
            self.builder.phrase_entity_promotion_salience,
            0.0,
            1.0,
        )?;
        validate_range(
            "layer_2_unit_builder.phrase_entity_promotion_confidence",
            self.builder.phrase_entity_promotion_confidence,
            0.0,
            1.0,
        )?;
        Ok(())
    }

    pub fn default_training_options(&self) -> crate::types::TrainingOptions {
        crate::types::TrainingOptions {
            max_memory_delta_mb: self.silent_training.max_memory_delta_mb,
            progress_interval_sec: self.silent_training.progress_interval_sec,
            merge_to_core: self.silent_training.merge_to_core,
            ..crate::types::TrainingOptions::default()
        }
    }

    pub fn with_bootstrap_overrides(&self) -> Self {
        let mut config = self.clone();
        config.governance.daily_growth_limit_mb = 50.0;
        config.governance.episodic_decay_days = 90;
        config.governance.anchor_salience_threshold = 0.60;
        config.memory_budgets.cold_start.daily_growth_limit_mb = 50.0;
        config
    }
}

fn validate_range(name: &str, value: f32, min: f32, max: f32) -> Result<(), String> {
    if (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "{name} out of range: {value} not in [{min}, {max}]"
        ))
    }
}
