use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

/// FxHash-style fingerprint for fast text equality checks.
/// Converts text to a u64 hash suitable for O(1) comparisons.
pub fn text_fingerprint(s: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for byte in s.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }
    hash
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    #[default]
    Episodic,
    Core,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryChannel {
    #[default]
    Main,
    Intent,
    Reasoning,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseMaturityStage {
    #[default]
    ColdStart,
    Growth,
    Stable,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    #[default]
    Candidate,
    Validated,
    Active,
    Rejected,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum UnitLevel {
    Char,
    Subword,
    Word,
    Phrase,
    Pattern,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Semantic,
    Parent,
    Child,
    Sequence,
    SourceEvidence,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SourceKind {
    UserInput,
    Retrieval,
    TrainingDocument,
    TrainingUrl,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ResolverMode {
    Deterministic,
    Balanced,
    Exploratory,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Processing,
    Completed,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingExecutionMode {
    #[default]
    User,
    Development,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
#[serde(rename_all = "snake_case")]
pub enum TrainingPhaseKind {
    #[default]
    DryRun,
    Bootstrap,
    Validation,
    Expansion,
    Lifelong,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum IntentKind {
    Greeting,
    Gratitude,
    Farewell,
    Help,
    Clarify,
    Rewrite,
    Verify,
    Continue,
    Forget,
    Question,
    Summarize,
    Explain,
    Compare,
    Extract,
    Analyze,
    Plan,
    Act,
    Recommend,
    Classify,
    Translate,
    Debug,
    Critique,
    Brainstorm,
    #[default]
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum IntentFallbackMode {
    #[default]
    None,
    DocumentScope,
    RetrieveUnknown,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingSourceType {
    Url,
    Document,
    Dataset,
    HuggingFaceDataset,
    StructuredJson,
    OpenApiSpec,
    CodeRepository,
    WikipediaDump,
    WikidataTruthy,
    OpenWebText,
    DbpediaDump,
    ProjectGutenberg,
    CommonCrawlWet,
    QaJson,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningType {
    #[default]
    General,
    Mathematical,      // Step-by-step calculation (GSM8K style)
    Logical,           // Deductive inference chains (ProofWriter style)
    Explanatory,       // Why/how causal chains
    Planning,          // Multi-step action sequences
    Verification,      // Fact-checking with source tracing
    Debugging,         // Error isolation and hypothesis testing
    MultiHop,          // Multi-hop deduction across entities
}

/// Type of individual reasoning step
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningStepType {
    #[default]
    Premise,           // Starting fact or assumption
    Inference,         // Deduced conclusion from premises
    Calculation,       // Numeric or symbolic operation
    Verification,      // Checking a claim against evidence
    Hypothesis,        // Tentative explanation
    Conclusion,        // Final answer or result
}

/// Single step in a reasoning trace
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReasoningStep {
    /// The thought content for this step
    pub content: String,
    /// Type of reasoning step
    pub step_type: ReasoningStepType,
    /// Whether this step is an anchor (never pruned)
    #[serde(default)]
    pub anchor_step: bool,
    /// Indices of previous steps this builds on (for dependency tracking)
    #[serde(default)]
    pub dependencies: Vec<usize>,
    /// Abstracted structure hash (for pattern matching)
    #[serde(default)]
    pub structure_hash: Option<u64>,
}

/// Complete reasoning trace for a training example
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ReasoningTrace {
    /// All steps in the reasoning chain
    pub steps: Vec<ReasoningStep>,
    /// Classification of reasoning type
    pub reasoning_type: ReasoningType,
    /// Confidence progression across steps (learned from data)
    #[serde(default)]
    pub confidence_trajectory: Vec<f32>,
    /// Entities involved in this reasoning
    #[serde(default)]
    pub entities: Vec<String>,
    /// Abstracted structure hash for the entire trace
    #[serde(default)]
    pub structure_hash: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Link {
    pub target_id: Uuid,
    pub edge_type: EdgeType,
    pub weight: f32,
}

impl Link {
    pub fn new(target_id: Uuid, edge_type: EdgeType, weight: f32) -> Self {
        Self {
            target_id,
            edge_type,
            weight,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Unit {
    pub id: Uuid,
    pub content: String,
    pub normalized: String,
    pub level: UnitLevel,
    pub frequency: u64,
    pub utility_score: f32,
    pub semantic_position: [f32; 3],
    pub anchor_status: bool,
    pub memory_type: MemoryType,
    #[serde(default = "default_memory_channels")]
    pub memory_channels: Vec<MemoryChannel>,
    pub created_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub salience_score: f32,
    pub confidence: f32,
    pub trust_score: f32,
    pub corroboration_count: u32,
    pub links: Vec<Link>,
    pub contexts: Vec<String>,
    /// Process units are reasoning steps isolated from Core semantic space
    /// Default: false (content unit)
    #[serde(default)]
    pub is_process_unit: bool,
    /// Pre-computed lowercase content for hot-path scoring (avoids repeated to_lowercase)
    #[serde(skip)]
    pub content_lower: String,
    /// Pre-computed FNV fingerprint of content_lower for O(1) equality checks
    #[serde(skip)]
    pub content_fingerprint: u64,
}

impl Unit {
    pub fn new(
        content: String,
        normalized: String,
        level: UnitLevel,
        utility_score: f32,
        confidence: f32,
        semantic_position: [f32; 3],
    ) -> Self {
        let now = Utc::now();
        let content_lower = content.to_lowercase();
        let content_fingerprint = text_fingerprint(&content_lower);
        Self {
            id: Uuid::new_v4(),
            content,
            normalized,
            level,
            frequency: 1,
            utility_score,
            semantic_position,
            anchor_status: false,
            memory_type: MemoryType::Episodic,
            memory_channels: default_memory_channels(),
            created_at: now,
            last_seen_at: now,
            salience_score: 0.0,
            confidence,
            trust_score: 0.5,
            corroboration_count: 0,
            links: Vec::new(),
            contexts: Vec::new(),
            is_process_unit: false,
            content_lower,
            content_fingerprint,
        }
    }

    /// Ensure pre-computed fields are populated (call after deserialization or struct literal construction)
    pub fn ensure_precomputed(&mut self) {
        if self.content_lower.is_empty() && !self.content.is_empty() {
            self.content_lower = self.content.to_lowercase();
            self.content_fingerprint = text_fingerprint(&self.content_lower);
        }
    }
}

impl Default for Unit {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::nil(),
            content: String::new(),
            normalized: String::new(),
            level: UnitLevel::Phrase,
            frequency: 0,
            utility_score: 0.0,
            semantic_position: [0.5; 3],
            anchor_status: false,
            memory_type: MemoryType::Episodic,
            memory_channels: default_memory_channels(),
            created_at: now,
            last_seen_at: now,
            salience_score: 0.0,
            confidence: 0.0,
            trust_score: 0.5,
            corroboration_count: 0,
            links: Vec::new(),
            contexts: Vec::new(),
            is_process_unit: false,
            content_lower: String::new(),
            content_fingerprint: 0,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ActivatedUnit {
    pub content: String,
    pub normalized: String,
    pub level: UnitLevel,
    pub utility_score: f32,
    pub frequency: u64,
    pub salience: f32,
    pub confidence: f32,
    pub context_hint: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct BuildOutput {
    pub activated_units: Vec<ActivatedUnit>,
    pub new_units: Vec<ActivatedUnit>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UnitHierarchy {
    pub levels: BTreeMap<String, Vec<ActivatedUnit>>,
    pub anchors: Vec<String>,
    pub entities: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RoutingResult {
    pub active_regions: Vec<String>,
    pub neighbor_ids: Vec<Uuid>,
    pub position_updates: Vec<(Uuid, [f32; 3])>,
    pub map_adjustments: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContextCell {
    pub unit_id: Option<Uuid>,
    pub content: String,
    pub relevance: f32,
    pub recency: f32,
    pub salience: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ContextMatrix {
    pub cells: Vec<ContextCell>,
    pub summary: String,
    /// Pre-computed lowercase of summary (avoids repeated to_lowercase in scoring)
    #[serde(skip)]
    pub summary_lower: String,
    /// Pre-computed lowercase of each cell's content (parallel to cells vec)
    #[serde(skip)]
    pub cell_content_lower: Vec<String>,
    /// Pre-computed fingerprints of each cell's lowered content for O(1) equality checks
    #[serde(skip)]
    pub cell_fingerprints: Vec<u64>,
}

impl ContextMatrix {
    /// Populate pre-computed lowercase and fingerprint fields. Call after construction.
    pub fn precompute(&mut self) {
        self.summary_lower = self.summary.to_lowercase();
        self.cell_content_lower = self.cells.iter().map(|c| c.content.to_lowercase()).collect();
        self.cell_fingerprints = self.cell_content_lower.iter().map(|s| text_fingerprint(s)).collect();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SequenceState {
    pub recent_unit_ids: Vec<Uuid>,
    pub anchor_ids: Vec<Uuid>,
    pub task_entities: Vec<String>,
    pub turn_index: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ConfidenceStats {
    pub mean_confidence: f32,
    pub candidate_count: usize,
    pub disagreement: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SearchDecision {
    pub should_retrieve: bool,
    pub score: f32,
    pub entropy: f32,
    pub freshness_need: f32,
    pub disagreement: f32,
    pub cost_penalty: f32,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ReasoningGateDecision {
    pub should_retrieve_reasoning: bool,
    pub reasoning_type_hint: ReasoningType,
}

#[derive(Debug, Clone)]
pub struct ReasoningPatternMatch {
    pub unit_id: Uuid,
    pub content: String,
    pub similarity: f32,
    pub reasoning_type: ReasoningType,
    pub is_anchor: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntentScore {
    pub intent: IntentKind,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct IntentProfile {
    pub primary: IntentKind,
    pub confidence: f32,
    pub top_score: f32,
    pub second_score: f32,
    pub ambiguous: bool,
    pub wants_brief: bool,
    pub references_document_context: bool,
    pub certainty_bias: f32,
    pub fallback_mode: IntentFallbackMode,
    pub scores: Vec<IntentScore>,
    pub reasons: Vec<String>,
}


/// Report on channel isolation validation for MemoryStore.
/// Ensures that memory channels (Main, Intent, Reasoning) maintain proper isolation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelIsolationReport {
    /// Whether channel isolation is valid (no violations)
    pub is_valid: bool,
    /// Count of units in Main channel
    pub main_count: usize,
    /// Count of units in Intent channel
    pub intent_count: usize,
    /// Count of units in Reasoning channel
    pub reasoning_count: usize,
    /// List of isolation violations detected
    pub violations: Vec<ChannelIsolationViolation>,
}

/// A specific channel isolation violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelIsolationViolation {
    /// Type of violation detected
    pub violation_type: IsolationViolationType,
    /// Unit IDs involved in the violation
    pub unit_ids: Vec<Uuid>,
    /// Human-readable description of the violation
    pub description: String,
}

/// Types of channel isolation violations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum IsolationViolationType {
    /// Intent channel contains units not present in Main channel
    IntentNotInMain,
    /// Reasoning channel contains units not present in Main channel
    ReasoningNotInMain,
    /// Excessive overlap between Intent and Reasoning channels
    ExcessiveIntentReasoningOverlap,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SanitizedQuery {
    pub raw_query: String,
    pub sanitized_query: String,
    pub semantic_expansions: Vec<String>,
    pub removed_tokens: Vec<String>,
    pub pii_redacted: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RetrievedDocument {
    pub source_url: String,
    pub title: String,
    pub raw_content: String,
    pub normalized_content: String,
    pub retrieved_at: DateTime<Utc>,
    pub trust_score: f32,
    pub cached: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct EvidenceState {
    pub documents: Vec<RetrievedDocument>,
    pub evidence_units: Vec<ActivatedUnit>,
    pub warnings: Vec<String>,
    pub average_trust: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ConflictRecord {
    pub claim: String,
    pub resolution: String,
    pub external_trust: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MergedState {
    pub candidate_ids: Vec<Uuid>,
    pub evidence_support: f32,
    pub freshness_boost: f32,
    pub conflict_records: Vec<ConflictRecord>,
    pub evidence: EvidenceState,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ScoreBreakdown {
    pub spatial_fit: f32,
    pub context_fit: f32,
    pub sequence_fit: f32,
    pub transition_fit: f32,
    pub utility_fit: f32,
    pub confidence_fit: f32,
    pub evidence_support: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScoredCandidate {
    pub unit_id: Uuid,
    pub content: String,
    pub score: f32,
    pub breakdown: ScoreBreakdown,
    pub memory_type: MemoryType,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ResolvedCandidate {
    pub unit_id: Uuid,
    pub content: String,
    pub score: f32,
    pub mode: ResolverMode,
    pub used_escape: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DecodedOutput {
    pub text: String,
    pub grounded: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeedbackEvent {
    pub layer: u8,
    pub event: String,
    pub impact: f32,
    #[serde(default)]
    pub target_unit_id: Option<Uuid>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UnitCandidate {
    pub id: Uuid,
    pub content: String,
    pub normalized: String,
    pub level: UnitLevel,
    pub observation_count: u64,
    pub utility_score: f32,
    pub status: CandidateStatus,
    pub first_seen_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub promoted_at: Option<DateTime<Utc>>,
    pub memory_type: MemoryType,
    #[serde(default = "default_memory_channels")]
    pub memory_channels: Vec<MemoryChannel>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LayerNote {
    pub layer: u8,
    pub note: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DebugStep {
    pub layer: u8,
    pub stage: String,
    pub summary: String,
    pub details: BTreeMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ExplainTrace {
    pub intent_profile: IntentProfile,
    pub layer_notes: Vec<LayerNote>,
    pub debug_steps: Vec<DebugStep>,
    pub active_regions: Vec<String>,
    pub retrieval_query: Option<SanitizedQuery>,
    pub evidence_sources: Vec<String>,
    pub score_breakdowns: Vec<ScoredCandidate>,
    pub selected_unit: Option<String>,
    pub safety_warnings: Vec<String>,
    pub feedback_events: Vec<FeedbackEvent>,
    pub memory_summary: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ProcessResult {
    pub predicted_text: String,
    pub confidence: f32,
    pub used_retrieval: bool,
    pub trace: ExplainTrace,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InputPacket {
    pub original_text: String,
    pub normalized_text: String,
    pub bytes: Vec<u8>,
    pub training_mode: bool,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrustAssessment {
    pub source_url: String,
    pub trust_score: f32,
    pub accepted: bool,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GovernanceReport {
    pub pruned_units: u64,
    #[serde(default)]
    pub pruned_candidates: u64,
    #[serde(default)]
    pub purged_polluted_units: u64,
    #[serde(default)]
    pub purged_polluted_candidates: u64,
    pub promoted_units: u64,
    pub anchors_protected: u64,
    #[serde(default)]
    pub layout_adjustments: u64,
    #[serde(default)]
    pub mean_displacement: f32,
    #[serde(default)]
    pub layout_rolled_back: bool,
    pub snapshot_path: String,
    #[serde(default)]
    pub pruning_reasons: Vec<String>,
    #[serde(default)]
    pub pruned_references: Vec<PrunedUnitReference>,
    #[serde(default)]
    pub pollution_findings: Vec<PollutionFinding>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PrunedUnitReference {
    pub id: Uuid,
    pub content: String,
    pub normalized: String,
    pub level: UnitLevel,
    pub memory_type: MemoryType,
    pub utility_score: f32,
    pub salience_score: f32,
    pub confidence: f32,
    pub trust_score: f32,
    pub frequency: u64,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PollutionFinding {
    pub polluted_id: Uuid,
    pub polluted_content: String,
    pub polluted_normalized: String,
    pub polluted_level: UnitLevel,
    pub canonical_id: Uuid,
    pub canonical_content: String,
    pub canonical_normalized: String,
    pub canonical_level: UnitLevel,
    pub overlap_ratio: f32,
    pub quality_delta: f32,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingStreamConfig {
    #[serde(default)]
    pub cache_dir: Option<String>,
    #[serde(default)]
    pub shard_limit: Option<usize>,
    #[serde(default)]
    pub item_limit: Option<usize>,
    #[serde(default)]
    pub max_input_bytes: Option<usize>,
    #[serde(default)]
    pub batch_size: Option<usize>,
    #[serde(default)]
    pub chunk_char_limit: Option<usize>,
}

impl Default for TrainingStreamConfig {
    fn default() -> Self {
        Self {
            cache_dir: None,
            shard_limit: None,
            item_limit: None,
            max_input_bytes: None,
            batch_size: None,
            chunk_char_limit: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingSource {
    #[serde(rename = "type")]
    pub source_type: TrainingSourceType,
    #[serde(default)]
    pub name: Option<String>,
    pub value: Option<String>,
    pub mime: Option<String>,
    pub content: Option<String>,
    #[serde(default)]
    pub target_memory: Option<MemoryType>,
    #[serde(default)]
    pub memory_channels: Option<Vec<MemoryChannel>>,
    #[serde(default)]
    pub stream: TrainingStreamConfig,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingOptions {
    pub consolidate_immediately: bool,
    pub max_memory_delta_mb: f32,
    pub progress_interval_sec: u64,
    #[serde(default = "default_training_intent_tagging")]
    pub tag_intent: bool,
    #[serde(default)]
    pub merge_to_core: bool,
    #[serde(default = "default_training_bypass_retrieval_gate")]
    pub bypass_retrieval_gate: bool,
    #[serde(default = "default_training_bypass_generation")]
    pub bypass_generation: bool,
    #[serde(default)]
    pub daily_growth_limit_mb: Option<f32>,
    #[serde(default)]
    pub execution_mode: TrainingExecutionMode,
}

impl Default for TrainingOptions {
    fn default() -> Self {
        Self {
            consolidate_immediately: false,
            max_memory_delta_mb: 0.5,
            progress_interval_sec: 10,
            tag_intent: true,
            merge_to_core: false,
            bypass_retrieval_gate: default_training_bypass_retrieval_gate(),
            bypass_generation: default_training_bypass_generation(),
            daily_growth_limit_mb: None,
            execution_mode: TrainingExecutionMode::User,
        }
    }
}

fn default_training_intent_tagging() -> bool {
    true
}

fn default_training_bypass_retrieval_gate() -> bool {
    true
}

fn default_training_bypass_generation() -> bool {
    true
}

fn default_memory_channels() -> Vec<MemoryChannel> {
    vec![MemoryChannel::Main]
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainBatchRequest {
    pub mode: String,
    pub sources: Vec<TrainingSource>,
    pub options: TrainingOptions,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TrainRequest {
    #[serde(default = "default_train_mode")]
    pub mode: String,
    #[serde(default)]
    pub execution_mode: TrainingExecutionMode,
}

fn default_train_mode() -> String {
    "silent".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TrainingMetrics {
    pub new_unit_rate: f32,
    pub unit_discovery_efficiency: f32,
    pub semantic_routing_accuracy: f32,
    pub prediction_error: f32,
    pub memory_delta_kb: i64,
    #[serde(default)]
    pub search_trigger_precision: Option<f32>,
    #[serde(default)]
    pub examples_ingested: u64,
    #[serde(default)]
    pub units_created: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingPhaseStatus {
    pub phase: TrainingPhaseKind,
    pub status: JobState,
    pub batches_completed: usize,
    pub batches_target: usize,
    pub sources_processed: usize,
    pub sources_total: usize,
    pub metrics: TrainingMetrics,
}

impl TrainingPhaseStatus {
    pub fn new(phase: TrainingPhaseKind, batches_target: usize, sources_total: usize) -> Self {
        Self {
            phase,
            status: JobState::Queued,
            batches_completed: 0,
            batches_target,
            sources_processed: 0,
            sources_total,
            metrics: TrainingMetrics::default(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TrainingProgress {
    pub percent_complete: f32,
    pub sources_processed: usize,
    pub sources_total: usize,
    #[serde(default)]
    pub active_source: Option<String>,
    #[serde(default)]
    pub chunks_processed: u64,
    #[serde(default)]
    pub bytes_processed: u64,
    #[serde(default)]
    pub worker_count: usize,
    #[serde(default)]
    pub active_workers: usize,
    #[serde(default)]
    pub queued_chunks: usize,
    #[serde(default)]
    pub prepared_chunks: u64,
    #[serde(default)]
    pub committed_batches: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct LearningMetrics {
    pub new_units_discovered: u64,
    pub units_pruned: u64,
    pub memory_delta_kb: i64,
    pub map_adjustments: u64,
    pub anchors_protected: u64,
    #[serde(default)]
    pub efficiency: TrainingEfficiencyMetrics,
    #[serde(default)]
    pub database_health: DatabaseHealthMetrics,
    #[serde(default)]
    pub memory_governance: MemoryGovernanceMetrics,
    #[serde(default)]
    pub pruning_events: Vec<PruningEventLog>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PruningEventLog {
    pub trigger: String,
    pub pruned_units: u64,
    #[serde(default)]
    pub pruned_candidates: u64,
    #[serde(default)]
    pub purged_polluted_units: u64,
    #[serde(default)]
    pub purged_polluted_candidates: u64,
    pub anchors_protected: u64,
    pub snapshot_path: String,
    #[serde(default)]
    pub reasons: Vec<String>,
    #[serde(default)]
    pub pruned_references: Vec<PrunedUnitReference>,
    #[serde(default)]
    pub pollution_findings: Vec<PollutionFinding>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct TrainingEfficiencyMetrics {
    pub units_discovered_per_mb: f32,
    pub units_discovered_per_kb: f32,
    pub pruned_units_percent: f32,
    pub anchor_density: f32,
    pub candidate_to_active_ratio: f32,
    pub pruned_candidates_percent: f32,
    pub avg_observations_to_promotion: f32,
    pub map_rebuild_frequency: f32,
    pub cache_hit_rate: f32,
    pub bloom_filter_false_positive_rate: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct DatabaseHealthMetrics {
    pub total_units: u64,
    pub active_units: u64,
    pub core_units: u64,
    pub episodic_units: u64,
    pub anchor_units: u64,
    pub intent_units: u64,
    pub reasoning_units: u64,
    pub candidate_units: u64,
    pub validated_candidates: u64,
    pub active_candidates: u64,
    pub rejected_candidates: u64,
    pub pruned_units: u64,
    pub wal_size_mb: f32,
    pub index_fragmentation: f32,
    pub maturity_stage: DatabaseMaturityStage,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MemoryGovernanceMetrics {
    pub episodic_memory_mb: f32,
    pub core_memory_mb: f32,
    pub candidate_pool_mb: f32,
    pub daily_growth_mb: f32,
    pub pruning_rate_per_hour: f32,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct QueueDepths {
    pub inference: usize,
    pub interactive_training: usize,
    pub silent_batch: usize,
    pub maintenance: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PerformanceMetrics {
    pub avg_ms_per_source: u64,
    #[serde(default)]
    pub queue_depths: QueueDepths,
    #[serde(default)]
    pub snapshot_age_ms: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingJobStatus {
    pub job_id: String,
    pub status: JobState,
    pub active_phase: Option<TrainingPhaseKind>,
    pub phase_statuses: Vec<TrainingPhaseStatus>,
    pub progress: TrainingProgress,
    pub learning_metrics: LearningMetrics,
    pub performance: PerformanceMetrics,
    pub intent_distribution: BTreeMap<String, u64>,
    pub warnings: Vec<String>,
}

impl TrainingJobStatus {
    pub fn queued(job_id: String, total_sources: usize) -> Self {
        Self {
            job_id,
            status: JobState::Queued,
            active_phase: None,
            phase_statuses: Vec::new(),
            progress: TrainingProgress {
                percent_complete: 0.0,
                sources_processed: 0,
                sources_total: total_sources,
                active_source: None,
                chunks_processed: 0,
                bytes_processed: 0,
                worker_count: 0,
                active_workers: 0,
                queued_chunks: 0,
                prepared_chunks: 0,
                committed_batches: 0,
            },
            learning_metrics: LearningMetrics::default(),
            performance: PerformanceMetrics::default(),
            intent_distribution: BTreeMap::new(),
            warnings: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct CandidateRoute {
    pub candidate_ids: Vec<Uuid>,
    pub used_escape: bool,
    pub rationale: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DryRunReport {
    pub status: TrainingJobStatus,
    pub snapshot_path: String,
    pub snapshot_readable: bool,
    pub map_stable: bool,
    pub inference_ok: bool,
    pub inference_latency_ms: u128,
    pub latency_per_token_ms: u128,
    pub query_result: String,
    pub memory_summary: String,
}

// ============================================================================
// Phase 3: LLM-Like Core Types
// ============================================================================

/// Output type for dynamic reasoning - distinguishes between final answers and internal thoughts
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OutputType {
    /// Final answer to be shown to user
    FinalAnswer(String),
    /// Internal reasoning step - hidden from user
    SilentThought(String),
}

/// Thought unit for dynamic reasoning loop
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ThoughtUnit {
    /// The thought content
    pub content: String,
    /// Step number in reasoning loop (0-indexed)
    pub step: usize,
    /// Whether this thought is internal-only (not shown to user)
    pub internal_only: bool,
    /// Confidence after this thought step
    pub confidence: f32,
    /// Timestamp when thought was generated
    pub created_at: DateTime<Utc>,
}

impl ThoughtUnit {
    pub fn new(content: String, step: usize, confidence: f32) -> Self {
        Self {
            content,
            step,
            internal_only: true,
            confidence,
            created_at: Utc::now(),
        }
    }
}

/// Tone kinds for internal tone inference
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToneKind {
    #[default]
    NeutralProfessional,
    Empathetic,
    Direct,
    Technical,
    Casual,
    Formal,
}

/// Method used for classification calculation
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalculationMethod {
    /// Memory lookup via spatial index
    MemoryLookup,
}

/// Result of classification calculation
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassificationResult {
    /// Classified intent kind
    pub intent: IntentKind,
    /// Classified tone kind
    pub tone: ToneKind,
    /// Resolver mode for this classification
    pub resolver_mode: ResolverMode,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Method used for calculation
    pub method: CalculationMethod,
    /// Number of candidate patterns considered
    pub candidate_count: usize,
}

impl Default for ClassificationResult {
    fn default() -> Self {
        Self {
            intent: IntentKind::Unknown,
            tone: ToneKind::NeutralProfessional,
            resolver_mode: ResolverMode::Balanced,
            confidence: 0.0,
            method: CalculationMethod::MemoryLookup,
            candidate_count: 0,
        }
    }
}

/// Ground truth for training classification
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GroundTruth {
    /// Expected intent
    pub intent: IntentKind,
    /// Expected tone
    pub tone: ToneKind,
    /// Expected resolver mode
    pub resolver_mode: ResolverMode,
    /// Optional domain context
    pub domain: Option<String>,
}

impl Default for GroundTruth {
    fn default() -> Self {
        Self {
            intent: IntentKind::Unknown,
            tone: ToneKind::NeutralProfessional,
            resolver_mode: ResolverMode::Balanced,
            domain: None,
        }
    }
}

impl From<&crate::classification::trainer::LabeledDialogue> for GroundTruth {
    fn from(dialogue: &crate::classification::trainer::LabeledDialogue) -> Self {
        Self {
            intent: dialogue.intent,
            tone: dialogue.expected_tone,
            resolver_mode: dialogue.resolver_mode,
            domain: dialogue.metadata.domain.clone(),
        }
    }
}

/// Style anchor for tone inference - represents a reference style unit
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StyleAnchor {
    /// Tone kind this anchor represents
    pub tone: ToneKind,
    /// Semantic position in embedding space
    pub embedding: [f32; 3],
    /// Keywords associated with this style
    pub keywords: Vec<String>,
    /// Decay rate for session persistence (0.0 = persist for session)
    pub decay_rate: f32,
}

/// Reasoning state for dynamic reasoning loop
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ReasoningState {
    /// Whether reasoning loop is currently active
    pub active: bool,
    /// Current step in reasoning loop
    pub current_step: usize,
    /// Thoughts generated so far
    pub thoughts: Vec<ThoughtUnit>,
    /// Confidence trajectory (one entry per step)
    pub confidence_trajectory: Vec<f32>,
    /// Whether max steps reached
    pub max_steps_reached: bool,
    /// Whether reasoning determined that web retrieval is needed
    pub needs_retrieval: bool,
}

/// Result of dynamic reasoning execution
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReasoningResult {
    /// Final output type
    pub output: OutputType,
    /// Number of reasoning steps taken
    pub steps_taken: usize,
    /// Final confidence score
    pub final_confidence: f32,
    /// Whether reasoning was triggered
    pub reasoning_triggered: bool,
    /// All thoughts generated (for telemetry)
    pub thoughts: Vec<ThoughtUnit>,
    /// Whether reasoning flagged a need for web retrieval
    pub needs_retrieval: bool,
}

/// Validation error from unified training expected_* fields
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingValidationError {
    /// Field that failed validation
    pub field: String,
    /// Expected value
    pub expected: String,
    /// Actual value
    pub actual: String,
    /// Error category for Layer 18 feedback
    pub category: ValidationErrorCategory,
}

/// Category of validation error for Layer 18 feedback routing
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationErrorCategory {
    /// Unit building mismatch (Layer 2)
    UnitFragmentation,
    /// Anchor detection mismatch (Layer 8)
    AnchorDetection,
    /// Entity extraction mismatch
    EntityExtraction,
    /// Trust quality mismatch
    TrustQuality,
}

/// Result of unified training dialogue ingestion
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UnifiedTrainingReport {
    /// Dialogue ID processed
    pub dialogue_id: String,
    /// Total turns processed
    pub turns_processed: u32,
    /// Units built from all turns
    pub units_built: u64,
    /// Entities extracted
    pub entities_extracted: u64,
    /// Anchors detected
    pub anchors_detected: u64,
    /// Classification training outcome (if classification training enabled)
    pub classification_outcome: Option<crate::classification::TrainingOutcome>,
    /// Validation errors from expected_* fields
    pub validation_errors: Vec<TrainingValidationError>,
    /// Whether memory_target: Core was treated as staging (Layer 21 compliance)
    pub core_target_staged: bool,
    /// Corroboration count for promoted units
    pub corroboration_count: u32,
    /// Memory channels populated
    pub channels_populated: Vec<String>,
}
