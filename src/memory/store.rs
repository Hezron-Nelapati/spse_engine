use crate::bloom_filter::{BloomStats, UnitBloomFilter};
use crate::config::{GovernanceConfig, SemanticMapConfig};
use crate::persistence::Db;
use crate::spatial_index::{force_directed_layout, SpatialGrid};
use crate::types::{
    ActivatedUnit, CandidateStatus, ChannelIsolationReport, ChannelIsolationViolation,
    DatabaseHealthMetrics, DatabaseMaturityStage, FeedbackEvent, GovernanceReport,
    IsolationViolationType, Link, MemoryChannel, MemoryType, PollutionFinding, PrunedUnitReference,
    SequenceState, SourceKind, Unit, UnitCandidate, UnitHierarchy, UnitLevel,
};
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use uuid::Uuid;

const MAX_PRUNED_REFERENCE_SAMPLES: usize = 12;

struct CandidatePruneReport {
    current_memory_kb: i64,
    pruned_references: Vec<PrunedUnitReference>,
    pruned_candidates: u64,
}

#[derive(Default)]
struct PollutionPurgeReport {
    findings: Vec<PollutionFinding>,
    pruned_references: Vec<PrunedUnitReference>,
    purged_units: u64,
    purged_candidates: u64,
}

#[derive(Clone)]
struct StoredRecord {
    id: Uuid,
    content: String,
    normalized: String,
    level: UnitLevel,
    utility_score: f32,
    salience_score: f32,
    confidence: f32,
    trust_score: f32,
    frequency: u64,
    memory_type: MemoryType,
    kind: StoredRecordKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StoredRecordKind {
    Unit,
    Candidate,
}

pub struct MemoryStore {
    cache: HashMap<Uuid, Arc<Unit>>,
    content_index: HashMap<String, Uuid>,
    channel_index: HashMap<MemoryChannel, HashSet<Uuid>>,
    candidate_cache: HashMap<String, UnitCandidate>,
    candidate_id_index: HashMap<Uuid, String>,
    sequence_state: SequenceState,
    anchor_reuse_threshold: u64,
    core_promotion_threshold: u64,
    promotion_min_corroborations: u32,
    prune_utility_threshold: f32,
    cold_start_prune_utility_threshold: f32,
    stable_prune_utility_threshold: f32,
    anchor_salience_threshold: f32,
    episodic_decay_days: u64,
    bootstrap_seed_frequency: u64,
    bootstrap_seed_utility_floor: f32,
    cold_start_unit_threshold: usize,
    stable_unit_threshold: usize,
    cold_start_candidate_observation_threshold: u64,
    growth_candidate_observation_threshold: u64,
    stable_candidate_observation_threshold: u64,
    candidate_activation_utility_threshold: f32,
    candidate_batch_size: usize,
    lfu_decay_factor: f32,
    recency_decay_rate: f32,
    anchor_protection_grace_days: u32,
    cold_archive_threshold_mb: f32,
    pollution_detection_enabled: bool,
    pollution_min_length: usize,
    pollution_edge_trim_limit: usize,
    pollution_similarity_threshold: f32,
    pollution_penalty_factor: f32,
    pollution_overlap_threshold: f32,
    pollution_quality_margin: f32,
    pollution_audit_limit: usize,
    intent_channel_core_promotion_blocked: bool,
    semantic_map: SemanticMapConfig,
    /// Process anchors: reasoning patterns that should never be pruned
    process_anchors: HashMap<u64, Uuid>, // structure_hash -> unit_id
    /// Reasoning pattern index for retrieval
    reasoning_index: HashMap<crate::types::ReasoningType, HashSet<Uuid>>,
    bloom_filter: UnitBloomFilter,
    pruned_units_total: u64,
    pruned_candidates_total: u64,
    candidate_promotions_total: u64,
    pattern_cache_hits: u64,
    pattern_cache_lookups: u64,
    db: Arc<Db>,
    // Deferred writes buffer for batch flushing
    pending_writes: Vec<Unit>,
    pending_writes_threshold: usize,
    write_deferred: bool,
    // Training mode: skip candidate staging, pattern combos, larger write batches
    training_mode: bool,
    // Classification pattern storage (Intent channel)
    classification_patterns: HashMap<Uuid, crate::classification::ClassificationPattern>,
    classification_by_signature: HashMap<String, Uuid>,
    // Nearest Centroid Classifier: per-intent and per-tone accumulated feature vectors
    intent_centroids: HashMap<crate::types::IntentKind, (Vec<f32>, u64)>, // (sum, count)
    tone_centroids: HashMap<crate::types::ToneKind, (Vec<f32>, u64)>,     // (sum, count)
}

#[derive(Default)]
pub struct ShardMergeReport {
    pub active_ids: Vec<Uuid>,
    pub new_units: u64,
    pub reused_units: u64,
    pub candidate_observations: u64,
    pub candidate_promotions: u64,
    pub cache_hits: u64,
    pub cache_lookups: u64,
}

#[derive(Default)]
pub struct IngestReport {
    pub active_ids: Vec<Uuid>,
    pub new_units: u64,
    pub reused_units: u64,
    pub candidate_observations: u64,
    pub candidate_promotions: u64,
    pub cache_hits: u64,
    pub cache_lookups: u64,
    pub bloom_false_positives: u64,
}

enum ActivationOutcome {
    Active { id: Uuid, is_new: bool },
    Candidate,
}

const UNIT_METADATA_OVERHEAD_BYTES: usize = 160;
const CANDIDATE_METADATA_OVERHEAD_BYTES: usize = 112;

#[derive(Clone)]
pub struct MemorySnapshot {
    units: Vec<Arc<Unit>>,
    unit_index: HashMap<Uuid, Arc<Unit>>,
    normalized_index: HashMap<String, Uuid>,
    channel_index: HashMap<MemoryChannel, Vec<Uuid>>,
    sequence_state: SequenceState,
    /// Cached spatial grid for O(1) neighbor lookups
    spatial_grid: SpatialGrid,
}

impl MemorySnapshot {
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Returns references to all units (zero-copy)
    pub fn all_units_ref(&self) -> &[Arc<Unit>] {
        &self.units
    }

    /// Returns cloned units (use sparingly with large datasets)
    pub fn all_units(&self) -> Vec<Unit> {
        self.units.iter().map(|u| (**u).clone()).collect()
    }

    /// Get units by IDs (returns Arc references, zero-copy)
    pub fn get_units_arc(&self, ids: &[Uuid]) -> Vec<Arc<Unit>> {
        ids.iter()
            .filter_map(|id| self.unit_index.get(id).cloned())
            .collect()
    }

    /// Get units by IDs (clones, use sparingly)
    pub fn get_units(&self, ids: &[Uuid]) -> Vec<Unit> {
        ids.iter()
            .filter_map(|id| self.unit_index.get(id).map(|u| (**u).clone()))
            .collect()
    }

    pub fn get_by_normalized(&self, normalized: &str) -> Option<&Unit> {
        self.normalized_index
            .get(normalized)
            .and_then(|id| self.unit_index.get(id))
            .map(|u| u.as_ref())
    }

    pub fn top_units(&self, limit: usize) -> Vec<Unit> {
        self.top_units_arc(limit)
            .into_iter()
            .map(|u| (*u).clone())
            .collect()
    }

    /// Top units returning Arc (zero-copy, preferred for large datasets)
    pub fn top_units_arc(&self, limit: usize) -> Vec<Arc<Unit>> {
        if self.units.len() <= limit {
            return self.units.clone();
        }
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;
        struct Scored(Arc<Unit>);
        impl PartialEq for Scored {
            fn eq(&self, other: &Self) -> bool {
                self.0.utility_score == other.0.utility_score
            }
        }
        impl Eq for Scored {}
        // Min-heap: reverse ordering so smallest is at top
        impl PartialOrd for Scored {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for Scored {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0
                    .utility_score
                    .partial_cmp(&other.0.utility_score)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap = BinaryHeap::with_capacity(limit + 1);
        for unit in &self.units {
            heap.push(Scored(Arc::clone(unit)));
            if heap.len() > limit {
                heap.pop(); // remove smallest
            }
        }
        let mut result: Vec<Arc<Unit>> = heap.into_iter().map(|s| s.0).collect();
        result.sort_by(|a, b| {
            b.utility_score
                .partial_cmp(&a.utility_score)
                .unwrap_or(Ordering::Equal)
        });
        result
    }

    pub fn units_in_channel(&self, channel: MemoryChannel) -> Vec<Unit> {
        self.channel_index
            .get(&channel)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.unit_index.get(id).map(|u| (**u).clone()))
            .collect()
    }

    /// Units in channel returning Arc (zero-copy)
    pub fn units_in_channel_arc(&self, channel: MemoryChannel) -> Vec<Arc<Unit>> {
        self.channel_index
            .get(&channel)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.unit_index.get(id).cloned())
            .collect()
    }

    pub fn top_channel_matches(
        &self,
        channel: MemoryChannel,
        query: &str,
        limit: usize,
    ) -> Vec<Unit> {
        let query_terms = normalized_terms(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scored = self
            .units_in_channel(channel)
            .into_iter()
            .filter_map(|unit| {
                let overlap = lexical_overlap_score(&query_terms, &normalized_terms(&unit.content));
                let substring =
                    if query.contains(&unit.normalized) || unit.normalized.contains(query) {
                        1.0
                    } else {
                        0.0
                    };
                let score = (0.48 * overlap)
                    + (0.18 * substring)
                    + (0.14 * unit.utility_score.clamp(0.0, 1.0))
                    + (0.10 * unit.confidence.clamp(0.0, 1.0))
                    + (0.10 * unit.trust_score.clamp(0.0, 1.0));
                (score > 0.08).then_some((score, unit))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
        scored.truncate(limit);
        scored.into_iter().map(|(_, unit)| unit).collect()
    }

    pub fn sequence_state(&self) -> SequenceState {
        self.sequence_state.clone()
    }

    /// Get cached spatial grid for neighbor lookups
    pub fn spatial_grid(&self) -> &SpatialGrid {
        &self.spatial_grid
    }

    pub fn memory_summary(&self) -> String {
        let core = self
            .units
            .iter()
            .filter(|unit| unit.memory_type == MemoryType::Core)
            .count();
        let intent = self
            .channel_index
            .get(&MemoryChannel::Intent)
            .map(Vec::len)
            .unwrap_or(0);
        let reasoning = self
            .channel_index
            .get(&MemoryChannel::Reasoning)
            .map(Vec::len)
            .unwrap_or(0);
        let anchors = self.units.iter().filter(|unit| unit.anchor_status).count();
        format!(
            "units={}, core={}, intent={}, reasoning={}, anchors={}, turn_index={}",
            self.units.len(),
            core,
            intent,
            reasoning,
            anchors,
            self.sequence_state.turn_index
        )
    }
}

impl MemoryStore {
    pub fn new(db_path: &str) -> Self {
        Self::new_with_config(
            db_path,
            &GovernanceConfig::default(),
            &SemanticMapConfig::default(),
        )
    }

    pub fn new_with_governance(db_path: &str, governance: &GovernanceConfig) -> Self {
        Self::new_with_config(db_path, governance, &SemanticMapConfig::default())
    }

    pub fn new_training_shard(
        db_path: &str,
        governance: &GovernanceConfig,
        semantic_map: &SemanticMapConfig,
    ) -> Self {
        Self::new_with_options(db_path, governance, semantic_map, false)
    }

    pub fn new_with_config(
        db_path: &str,
        governance: &GovernanceConfig,
        semantic_map: &SemanticMapConfig,
    ) -> Self {
        Self::new_with_options(db_path, governance, semantic_map, true)
    }

    fn new_with_options(
        db_path: &str,
        governance: &GovernanceConfig,
        semantic_map: &SemanticMapConfig,
        seed_bootstrap_units: bool,
    ) -> Self {
        let db = Arc::new(Db::new(db_path).expect("failed to initialize database"));
        let units = db.load_all_units().unwrap_or_default();
        let candidates = db.load_all_candidates().unwrap_or_default();
        let sequence_state = db
            .load_snapshot()
            .ok()
            .flatten()
            .map(|(_, state)| state)
            .unwrap_or_default();
        let mut cache = HashMap::new();
        let mut content_index = HashMap::new();
        let mut channel_index: HashMap<MemoryChannel, HashSet<Uuid>> = HashMap::new();
        let mut candidate_cache = HashMap::new();
        let mut candidate_id_index = HashMap::new();

        let mut classification_patterns = HashMap::new();
        let mut classification_by_signature = HashMap::new();

        for mut unit in units {
            normalize_channels(&mut unit.memory_channels);
            content_index.insert(unit.normalized.clone(), unit.id);
            for channel in &unit.memory_channels {
                channel_index.entry(*channel).or_default().insert(unit.id);
            }
            // Reconstruct classification patterns from pattern-type units
            if unit.content.starts_with("pattern:") {
                if let Some(pattern) =
                    crate::classification::ClassificationPattern::from_unit(&unit)
                {
                    let sig_hash = pattern.signature.signature_hash().to_string();
                    classification_by_signature.insert(sig_hash, pattern.unit_id);
                    classification_patterns.insert(pattern.unit_id, pattern);
                }
            }
            cache.insert(unit.id, Arc::new(unit));
        }

        for mut candidate in candidates {
            normalize_channels(&mut candidate.memory_channels);
            candidate_id_index.insert(candidate.id, candidate.normalized.clone());
            candidate_cache.insert(candidate.normalized.clone(), candidate);
        }

        let mut bloom_filter = UnitBloomFilter::new(governance.bloom_expected_items);
        bloom_filter.rebuild(content_index.keys().map(String::as_str));

        let mut store = Self {
            cache,
            content_index,
            channel_index,
            candidate_cache,
            candidate_id_index,
            sequence_state,
            anchor_reuse_threshold: governance.anchor_reuse_threshold,
            core_promotion_threshold: governance.core_promotion_threshold,
            promotion_min_corroborations: governance.promotion_min_corroborations,
            prune_utility_threshold: governance.prune_utility_threshold,
            cold_start_prune_utility_threshold: governance.cold_start_prune_utility_threshold,
            stable_prune_utility_threshold: governance.stable_prune_utility_threshold,
            anchor_salience_threshold: governance.anchor_salience_threshold,
            episodic_decay_days: governance.episodic_decay_days,
            bootstrap_seed_frequency: governance.bootstrap_seed_frequency,
            bootstrap_seed_utility_floor: governance.bootstrap_seed_utility_floor,
            cold_start_unit_threshold: governance.cold_start_unit_threshold,
            stable_unit_threshold: governance.stable_unit_threshold,
            cold_start_candidate_observation_threshold: governance
                .cold_start_candidate_observation_threshold,
            growth_candidate_observation_threshold: governance
                .growth_candidate_observation_threshold,
            stable_candidate_observation_threshold: governance
                .stable_candidate_observation_threshold,
            candidate_activation_utility_threshold: governance
                .candidate_activation_utility_threshold,
            candidate_batch_size: governance.candidate_batch_size,
            lfu_decay_factor: governance.lfu_decay_factor,
            recency_decay_rate: governance.recency_decay_rate,
            anchor_protection_grace_days: governance.anchor_protection_grace_days,
            cold_archive_threshold_mb: governance.cold_archive_threshold_mb,
            pollution_detection_enabled: governance.pollution_detection_enabled,
            pollution_min_length: governance.pollution_min_length,
            pollution_edge_trim_limit: governance.pollution_edge_trim_limit,
            pollution_similarity_threshold: governance.pollution_similarity_threshold,
            pollution_penalty_factor: governance.pollution_penalty_factor,
            pollution_overlap_threshold: governance.pollution_overlap_threshold,
            pollution_quality_margin: governance.pollution_quality_margin,
            pollution_audit_limit: governance.pollution_audit_limit,
            intent_channel_core_promotion_blocked: governance.intent_channel_core_promotion_blocked,
            semantic_map: semantic_map.clone(),
            process_anchors: HashMap::new(),
            reasoning_index: HashMap::new(),
            bloom_filter,
            pruned_units_total: 0,
            pruned_candidates_total: 0,
            candidate_promotions_total: 0,
            pattern_cache_hits: 0,
            pattern_cache_lookups: 0,
            db,
            pending_writes: Vec::new(),
            pending_writes_threshold: 50_000,
            write_deferred: true,
            training_mode: false,
            classification_patterns,
            classification_by_signature,
            intent_centroids: HashMap::new(),
            tone_centroids: HashMap::new(),
        };
        
        // Load trained centroids from SQLite if available
        store.load_trained_centroids_from_db();
        
        if seed_bootstrap_units {
            store.ensure_bootstrap_seeds();
        }
        store
    }
    
    /// Load trained centroids from database persistence layer
    fn load_trained_centroids_from_db(&mut self) {
        // Load intent centroids
        if let Ok(centroids) = self.db.load_intent_centroids() {
            let mut loaded_count = 0;
            for (intent_str, centroid, count) in centroids {
                let intent = match intent_str.as_str() {
                    "Question" => crate::types::IntentKind::Question,
                    "Explain" => crate::types::IntentKind::Explain,
                    "Compare" => crate::types::IntentKind::Compare,
                    "Analyze" => crate::types::IntentKind::Analyze,
                    "Plan" => crate::types::IntentKind::Plan,
                    "Debug" => crate::types::IntentKind::Debug,
                    "Verify" => crate::types::IntentKind::Verify,
                    "Summarize" => crate::types::IntentKind::Summarize,
                    "Classify" => crate::types::IntentKind::Classify,
                    "Recommend" => crate::types::IntentKind::Recommend,
                    "Extract" => crate::types::IntentKind::Extract,
                    "Critique" => crate::types::IntentKind::Critique,
                    "Brainstorm" => crate::types::IntentKind::Brainstorm,
                    "Help" => crate::types::IntentKind::Help,
                    "Greeting" => crate::types::IntentKind::Greeting,
                    "Farewell" => crate::types::IntentKind::Farewell,
                    "Gratitude" => crate::types::IntentKind::Gratitude,
                    _ => continue,
                };
                // Store as sum (centroid * count) so mean calculation works
                let sum: Vec<f32> = centroid.iter().map(|v| v * count as f32).collect();
                self.intent_centroids.insert(intent, (sum, count));
                loaded_count += 1;
            }
            if loaded_count > 0 {
                eprintln!("[SPSE] Loaded {} intent centroids from database", loaded_count);
            }
        }
        
        // Load tone centroids
        if let Ok(centroids) = self.db.load_tone_centroids() {
            for (tone_str, centroid, count) in centroids {
                let tone = match tone_str.as_str() {
                    "NeutralProfessional" => crate::types::ToneKind::NeutralProfessional,
                    "Empathetic" => crate::types::ToneKind::Empathetic,
                    "Direct" => crate::types::ToneKind::Direct,
                    "Technical" => crate::types::ToneKind::Technical,
                    "Casual" => crate::types::ToneKind::Casual,
                    "Formal" => crate::types::ToneKind::Formal,
                    _ => continue,
                };
                let sum: Vec<f32> = centroid.iter().map(|v| v * count as f32).collect();
                self.tone_centroids.insert(tone, (sum, count));
            }
        }
    }

    pub fn apply_governance(&mut self, governance: &GovernanceConfig) {
        self.anchor_reuse_threshold = governance.anchor_reuse_threshold;
        self.core_promotion_threshold = governance.core_promotion_threshold;
        self.promotion_min_corroborations = governance.promotion_min_corroborations;
        self.prune_utility_threshold = governance.prune_utility_threshold;
        self.cold_start_prune_utility_threshold = governance.cold_start_prune_utility_threshold;
        self.stable_prune_utility_threshold = governance.stable_prune_utility_threshold;
        self.anchor_salience_threshold = governance.anchor_salience_threshold;
        self.episodic_decay_days = governance.episodic_decay_days;
        self.bootstrap_seed_frequency = governance.bootstrap_seed_frequency;
        self.bootstrap_seed_utility_floor = governance.bootstrap_seed_utility_floor;
        self.cold_start_unit_threshold = governance.cold_start_unit_threshold;
        self.stable_unit_threshold = governance.stable_unit_threshold;
        self.cold_start_candidate_observation_threshold =
            governance.cold_start_candidate_observation_threshold;
        self.growth_candidate_observation_threshold =
            governance.growth_candidate_observation_threshold;
        self.stable_candidate_observation_threshold =
            governance.stable_candidate_observation_threshold;
        self.candidate_activation_utility_threshold =
            governance.candidate_activation_utility_threshold;
        self.candidate_batch_size = governance.candidate_batch_size;
        self.lfu_decay_factor = governance.lfu_decay_factor;
        self.recency_decay_rate = governance.recency_decay_rate;
        self.anchor_protection_grace_days = governance.anchor_protection_grace_days;
        self.cold_archive_threshold_mb = governance.cold_archive_threshold_mb;
        self.pollution_detection_enabled = governance.pollution_detection_enabled;
        self.pollution_min_length = governance.pollution_min_length;
        self.pollution_edge_trim_limit = governance.pollution_edge_trim_limit;
        self.pollution_overlap_threshold = governance.pollution_overlap_threshold;
        self.pollution_quality_margin = governance.pollution_quality_margin;
        self.pollution_audit_limit = governance.pollution_audit_limit;
        self.intent_channel_core_promotion_blocked =
            governance.intent_channel_core_promotion_blocked;
    }

    pub fn apply_semantic_map_config(&mut self, semantic_map: &SemanticMapConfig) {
        self.semantic_map = semantic_map.clone();
    }

    pub fn db(&self) -> Arc<Db> {
        self.db.clone()
    }

    pub fn snapshot(&self) -> MemorySnapshot {
        // Wrap units in Arc for zero-copy sharing
        let units: Vec<Arc<Unit>> = self.cache.values().cloned().collect();
        let mut unit_index = HashMap::new();
        let mut normalized_index = HashMap::new();
        let mut channel_index: HashMap<MemoryChannel, Vec<Uuid>> = HashMap::new();
        for unit in &units {
            unit_index.insert(unit.id, Arc::clone(unit));
            normalized_index.insert(unit.normalized.clone(), unit.id);
            for channel in &unit.memory_channels {
                channel_index.entry(*channel).or_default().push(unit.id);
            }
        }

        // Build cached spatial grid (filter out process units)
        let mut spatial_grid = SpatialGrid::new(self.semantic_map.spatial_cell_size);
        for unit in &units {
            if !unit.is_process_unit {
                spatial_grid.insert(unit.id, &unit.semantic_position);
            }
        }

        MemorySnapshot {
            units,
            unit_index,
            normalized_index,
            channel_index,
            sequence_state: self.sequence_state(),
            spatial_grid,
        }
    }

    pub fn unit_count(&self) -> usize {
        self.cache.len()
    }

    pub fn database_health(&self) -> DatabaseHealthMetrics {
        database_health_from_units(
            self.cache.values().map(|u| u.as_ref()),
            &self.channel_index,
            self.candidate_cache.values(),
            self.cold_start_unit_threshold,
            self.stable_unit_threshold,
            self.pruned_units_total,
            self.db.wal_size_mb(),
            self.db.index_fragmentation(),
        )
    }

    pub fn bloom_stats(&self) -> BloomStats {
        self.bloom_filter.stats()
    }

    pub fn pattern_cache_stats(&self) -> (u64, u64) {
        (self.pattern_cache_hits, self.pattern_cache_lookups)
    }

    pub fn pruned_candidates_total(&self) -> u64 {
        self.pruned_candidates_total
    }

    pub fn candidate_promotions_total(&self) -> u64 {
        self.candidate_promotions_total
    }

    // === Classification Pattern Methods ===

    /// Store a classification pattern in Intent memory channel.
    pub fn store_classification_pattern(
        &mut self,
        pattern: crate::classification::ClassificationPattern,
    ) {
        let sig_hash = pattern.signature.signature_hash().to_string();
        let unit = pattern.to_unit();
        let fv = pattern.signature.to_feature_vector();

        self.accumulate_centroid(pattern.intent_kind, pattern.tone_kind, &fv);

        // Store pattern
        self.classification_patterns
            .insert(pattern.unit_id, pattern.clone());
        self.classification_by_signature
            .insert(sig_hash, pattern.unit_id);

        // Store unit in cache and channel index
        self.cache.insert(unit.id, Arc::new(unit.clone()));
        self.content_index.insert(unit.normalized.clone(), unit.id);
        for channel in &unit.memory_channels {
            self.channel_index
                .entry(*channel)
                .or_default()
                .insert(unit.id);
        }

        // Persist to database (deferred in training mode)
        self.persist(&unit);
    }

    /// Get a classification pattern by ID.
    pub fn get_classification_pattern(
        &self,
        id: uuid::Uuid,
    ) -> Option<crate::classification::ClassificationPattern> {
        self.classification_patterns.get(&id).cloned()
    }

    /// Update an existing classification pattern.
    pub fn update_classification_pattern(
        &mut self,
        pattern: crate::classification::ClassificationPattern,
    ) {
        if let Some(existing) = self.classification_patterns.get_mut(&pattern.unit_id) {
            *existing = pattern.clone();

            // Update unit in cache
            let unit = pattern.to_unit();
            if let Some(cached_unit) = self.cache.get_mut(&pattern.unit_id) {
                *cached_unit = Arc::new(unit.clone());
            }

            // Persist to database (deferred in training mode)
            self.persist(&unit);
        }
    }

    /// Get all classification patterns.
    pub fn all_classification_patterns(&self) -> Vec<crate::classification::ClassificationPattern> {
        self.classification_patterns.values().cloned().collect()
    }

    /// Look up classification pattern ID by signature hash.
    pub fn classification_pattern_by_signature(&self, sig_hash: &str) -> Option<uuid::Uuid> {
        self.classification_by_signature.get(sig_hash).copied()
    }

    /// Get all classification pattern IDs (for direct scoring without spatial pre-filtering).
    pub fn classification_pattern_ids(&self) -> Vec<uuid::Uuid> {
        self.classification_patterns.keys().copied().collect()
    }

    /// Count classification patterns.
    pub fn classification_pattern_count(&self) -> usize {
        self.classification_patterns.len()
    }

    /// Accumulate a feature vector into per-intent and per-tone centroids.
    /// Called during training for each classification pattern.
    pub fn accumulate_centroid(
        &mut self,
        intent: crate::types::IntentKind,
        tone: crate::types::ToneKind,
        feature_vector: &[f32],
    ) {
        // Intent centroid
        let (sum, count) = self
            .intent_centroids
            .entry(intent)
            .or_insert_with(|| (vec![0.0; feature_vector.len()], 0));
        for (s, v) in sum.iter_mut().zip(feature_vector.iter()) {
            *s += v;
        }
        *count += 1;

        // Tone centroid
        let (sum, count) = self
            .tone_centroids
            .entry(tone)
            .or_insert_with(|| (vec![0.0; feature_vector.len()], 0));
        for (s, v) in sum.iter_mut().zip(feature_vector.iter()) {
            *s += v;
        }
        *count += 1;
    }

    /// Get all intent centroids as (IntentKind, mean_feature_vector) pairs.
    pub fn intent_centroids(&self) -> Vec<(crate::types::IntentKind, Vec<f32>)> {
        self.intent_centroids
            .iter()
            .map(|(intent, (sum, count))| {
                let mean: Vec<f32> = sum.iter().map(|s| s / *count as f32).collect();
                (*intent, mean)
            })
            .collect()
    }

    /// Get all tone centroids as (ToneKind, mean_feature_vector) pairs.
    pub fn tone_centroids(&self) -> Vec<(crate::types::ToneKind, Vec<f32>)> {
        self.tone_centroids
            .iter()
            .map(|(tone, (sum, count))| {
                let mean: Vec<f32> = sum.iter().map(|s| s / *count as f32).collect();
                (*tone, mean)
            })
            .collect()
    }

    /// Rebuild centroids from all loaded classification patterns.
    /// Called on startup after patterns are loaded from DB.
    /// Skips rebuild if centroids were already loaded from database.
    pub fn rebuild_centroids_from_patterns(&mut self) {
        // Skip rebuild if centroids were already loaded from database
        if !self.intent_centroids.is_empty() {
            return;
        }
        let patterns: Vec<_> = self.classification_patterns.values().cloned().collect();
        for pattern in &patterns {
            let fv = pattern.signature.to_feature_vector();
            self.accumulate_centroid(pattern.intent_kind, pattern.tone_kind, &fv);
        }
    }

    pub fn all_units(&self) -> Vec<Unit> {
        self.cache.values().map(|u| (**u).clone()).collect()
    }

    pub fn all_candidates(&self) -> Vec<UnitCandidate> {
        self.candidate_cache.values().cloned().collect()
    }

    pub fn get_unit(&self, id: &Uuid) -> Option<&Unit> {
        self.cache.get(id).map(|v| v.as_ref())
    }

    /// O(1) content_index lookup by normalized text. Used to find unit IDs
    /// without scanning entire channels.
    pub fn find_unit_id_by_content(&self, normalized: &str) -> Option<Uuid> {
        self.content_index.get(normalized).copied()
    }

    pub fn get_units(&self, ids: &[Uuid]) -> Vec<Unit> {
        ids.iter()
            .filter_map(|id| self.cache.get(id).map(|u| (**u).clone()))
            .collect()
    }

    /// Register a process anchor (never-pruned reasoning pattern)
    pub fn register_process_anchor(&mut self, structure_hash: u64, unit_id: Uuid) {
        self.process_anchors.insert(structure_hash, unit_id);
    }

    /// Mark a unit as a process unit (reasoning step).
    /// Process units are filtered by reasoning_patterns_for_query.
    pub fn mark_as_process_unit(&mut self, unit_id: Uuid) {
        if let Some(arc) = self.cache.get_mut(&unit_id) {
            let unit = Arc::make_mut(arc);
            unit.is_process_unit = true;
        }
    }

    /// Check if a structure hash is a process anchor
    pub fn is_process_anchor(&self, structure_hash: u64) -> bool {
        self.process_anchors.contains_key(&structure_hash)
    }

    /// Check if a unit should be pruned based on utility, staleness, and anchor status.
    /// Process anchors are never pruned.
    pub fn should_prune_unit(
        &self,
        unit: &Unit,
        prune_utility_threshold: f32,
        stale_hours: f32,
        anchor_grace_active: bool,
    ) -> bool {
        // Process anchors are never pruned
        if self.process_anchors.values().any(|&id| id == unit.id) {
            return false;
        }
        // Anchors are never pruned
        if unit.anchor_status {
            return false;
        }
        // Grace period for new anchors
        if anchor_grace_active {
            return false;
        }
        // Only prune Episodic memory
        if unit.memory_type != MemoryType::Episodic {
            return false;
        }
        // Must be low utility and stale
        unit.utility_score < prune_utility_threshold && stale_hours > 24.0
    }

    /// Retrieve reasoning patterns matching a query, optionally filtered by reasoning type.
    /// Returns process units (is_process_unit=true) sorted by similarity to the query.
    pub fn reasoning_patterns_for_query(
        &self,
        query: &str,
        reasoning_type_hint: Option<crate::types::ReasoningType>,
        limit: usize,
    ) -> Vec<crate::types::ReasoningPatternMatch> {
        use crate::types::ReasoningPatternMatch;

        let query_terms = normalized_terms(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        // Get candidate unit IDs from reasoning index if type hint provided
        let type_filtered_ids: Option<HashSet<Uuid>> =
            reasoning_type_hint.and_then(|rt| self.reasoning_index.get(&rt).cloned());

        // Score process units
        let mut scored: Vec<(Uuid, String, f32, crate::types::ReasoningType, bool)> = self
            .cache
            .values()
            .filter(|unit| unit.is_process_unit)
            .filter(|unit| {
                type_filtered_ids
                    .as_ref()
                    .map(|ids| ids.contains(&unit.id))
                    .unwrap_or(true)
            })
            .filter_map(|unit| {
                let overlap = lexical_overlap_score(&query_terms, &normalized_terms(&unit.content));
                let substring =
                    if query.contains(&unit.normalized) || unit.normalized.contains(query) {
                        0.3
                    } else {
                        0.0
                    };
                let score = (0.5 * overlap) + substring + (0.2 * unit.salience_score);
                if score > 0.1 {
                    // Infer reasoning type from content heuristics (simplified)
                    let reasoning_type = self
                        .reasoning_index
                        .iter()
                        .find(|(_, ids)| ids.contains(&unit.id))
                        .map(|(rt, _)| *rt)
                        .unwrap_or_default();
                    Some((
                        unit.id,
                        unit.content.clone(),
                        score,
                        reasoning_type,
                        unit.anchor_status,
                    ))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
            .into_iter()
            .map(
                |(unit_id, content, similarity, reasoning_type, is_anchor)| ReasoningPatternMatch {
                    unit_id,
                    content,
                    similarity,
                    reasoning_type,
                    is_anchor,
                },
            )
            .collect()
    }

    /// Register a reasoning pattern unit in the reasoning index
    pub fn register_reasoning_pattern(
        &mut self,
        reasoning_type: crate::types::ReasoningType,
        unit_id: Uuid,
    ) {
        self.reasoning_index
            .entry(reasoning_type)
            .or_insert_with(HashSet::new)
            .insert(unit_id);
    }

    pub fn top_units(&self, limit: usize) -> Vec<Unit> {
        let mut units = self.all_units();
        units.sort_by(|a, b| {
            b.utility_score
                .partial_cmp(&a.utility_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        units.truncate(limit);
        units
    }

    pub fn units_in_channel(&self, channel: MemoryChannel) -> Vec<Unit> {
        self.channel_index
            .get(&channel)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.cache.get(id).map(|u| (**u).clone()))
            .collect()
    }

    pub fn top_channel_matches(
        &self,
        channel: MemoryChannel,
        query: &str,
        limit: usize,
    ) -> Vec<Unit> {
        let query_terms = normalized_terms(query);
        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scored = self
            .units_in_channel(channel)
            .into_iter()
            .filter_map(|unit| {
                let overlap = lexical_overlap_score(&query_terms, &normalized_terms(&unit.content));
                let substring =
                    if query.contains(&unit.normalized) || unit.normalized.contains(query) {
                        1.0
                    } else {
                        0.0
                    };
                let score = (0.48 * overlap)
                    + (0.18 * substring)
                    + (0.14 * unit.utility_score.clamp(0.0, 1.0))
                    + (0.10 * unit.confidence.clamp(0.0, 1.0))
                    + (0.10 * unit.trust_score.clamp(0.0, 1.0));
                (score > 0.08).then_some((score, unit))
            })
            .collect::<Vec<_>>();
        scored.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
        scored.truncate(limit);
        scored.into_iter().map(|(_, unit)| unit).collect()
    }

    /// Validate channel isolation - ensure units are properly isolated across channels.
    /// Returns a report of any isolation violations detected.
    pub fn validate_channel_isolation(&self) -> ChannelIsolationReport {
        let mut report = ChannelIsolationReport::default();

        // Get all unit IDs across all channels
        let main_ids: HashSet<Uuid> = self
            .channel_index
            .get(&MemoryChannel::Main)
            .cloned()
            .unwrap_or_default();
        let intent_ids: HashSet<Uuid> = self
            .channel_index
            .get(&MemoryChannel::Intent)
            .cloned()
            .unwrap_or_default();
        let reasoning_ids: HashSet<Uuid> = self
            .channel_index
            .get(&MemoryChannel::Reasoning)
            .cloned()
            .unwrap_or_default();

        // Check for proper isolation: Main should contain all units
        // Intent and Reasoning should be subsets of Main
        report.main_count = main_ids.len();
        report.intent_count = intent_ids.len();
        report.reasoning_count = reasoning_ids.len();

        // Verify Intent units are also in Main
        let intent_not_in_main: Vec<Uuid> = intent_ids.difference(&main_ids).copied().collect();
        if !intent_not_in_main.is_empty() {
            report.violations.push(ChannelIsolationViolation {
                violation_type: IsolationViolationType::IntentNotInMain,
                unit_ids: intent_not_in_main,
                description: "Intent channel contains units not present in Main channel"
                    .to_string(),
            });
        }

        // Verify Reasoning units are also in Main
        let reasoning_not_in_main: Vec<Uuid> =
            reasoning_ids.difference(&main_ids).copied().collect();
        if !reasoning_not_in_main.is_empty() {
            report.violations.push(ChannelIsolationViolation {
                violation_type: IsolationViolationType::ReasoningNotInMain,
                unit_ids: reasoning_not_in_main,
                description: "Reasoning channel contains units not present in Main channel"
                    .to_string(),
            });
        }

        // Check for content leakage between Intent and Reasoning channels
        // These should be independent - a unit shouldn't typically be in both
        let intent_reasoning_overlap: Vec<Uuid> =
            intent_ids.intersection(&reasoning_ids).copied().collect();

        // Allow some overlap but flag excessive overlap
        let overlap_ratio = if intent_ids.is_empty() || reasoning_ids.is_empty() {
            0.0
        } else {
            intent_reasoning_overlap.len() as f32 / intent_ids.len().max(reasoning_ids.len()) as f32
        };

        if overlap_ratio > 0.3 {
            report.violations.push(ChannelIsolationViolation {
                violation_type: IsolationViolationType::ExcessiveIntentReasoningOverlap,
                unit_ids: intent_reasoning_overlap,
                description: format!(
                    "Intent and Reasoning channels have excessive overlap ({:.1}%)",
                    overlap_ratio * 100.0
                ),
            });
        }

        report.is_valid = report.violations.is_empty();
        report
    }

    /// Get units that should be isolated to a specific channel (not present in other channels)
    pub fn isolated_units_for_channel(&self, channel: MemoryChannel) -> Vec<Unit> {
        let channel_ids: HashSet<Uuid> = self
            .channel_index
            .get(&channel)
            .cloned()
            .unwrap_or_default();

        let other_channel_ids: HashSet<Uuid> = match channel {
            MemoryChannel::Main => HashSet::new(), // Main contains all
            MemoryChannel::Intent => self
                .channel_index
                .get(&MemoryChannel::Reasoning)
                .cloned()
                .unwrap_or_default(),
            MemoryChannel::Reasoning => self
                .channel_index
                .get(&MemoryChannel::Intent)
                .cloned()
                .unwrap_or_default(),
        };

        channel_ids
            .difference(&other_channel_ids)
            .filter_map(|id| self.cache.get(id).map(|u| (**u).clone()))
            .collect()
    }

    pub fn sequence_state(&self) -> SequenceState {
        self.sequence_state.clone()
    }

    pub fn update_sequence_state(
        &mut self,
        recent_unit_ids: Vec<Uuid>,
        anchor_ids: Vec<Uuid>,
        task_entities: Vec<String>,
    ) {
        self.sequence_state.turn_index += 1;
        self.sequence_state.recent_unit_ids = recent_unit_ids;
        self.sequence_state.anchor_ids = anchor_ids;
        self.sequence_state.task_entities = task_entities;
    }

    pub fn ingest_hierarchy(
        &mut self,
        hierarchy: &UnitHierarchy,
        source: SourceKind,
        context_summary: &str,
    ) -> Vec<Uuid> {
        let default_memory_type = if source_defaults_to_core(source) {
            MemoryType::Core
        } else {
            MemoryType::Episodic
        };
        self.ingest_hierarchy_with_channels(
            hierarchy,
            source,
            context_summary,
            default_memory_type,
            &[MemoryChannel::Main],
        )
    }

    pub fn ingest_hierarchy_with_tiering(
        &mut self,
        hierarchy: &UnitHierarchy,
        source: SourceKind,
        context_summary: &str,
        default_memory_type: MemoryType,
        utility_boost: f32,
        memory_channels: &[MemoryChannel],
    ) -> Vec<Uuid> {
        self.ingest_hierarchy_with_tiering_report(
            hierarchy,
            source,
            context_summary,
            default_memory_type,
            utility_boost,
            memory_channels,
        )
        .active_ids
    }

    pub fn ingest_hierarchy_with_tiering_report(
        &mut self,
        hierarchy: &UnitHierarchy,
        source: SourceKind,
        context_summary: &str,
        default_memory_type: MemoryType,
        utility_boost: f32,
        memory_channels: &[MemoryChannel],
    ) -> IngestReport {
        let mut report = IngestReport::default();
        let trust_delta = match source {
            SourceKind::UserInput => 0.08,
            SourceKind::Retrieval => 0.04,
            SourceKind::TrainingDocument => 0.06,
            SourceKind::TrainingUrl => 0.05,
        };

        for units in hierarchy.levels.values() {
            for activation in units {
                match self.ingest_activation_with_utility_boost(
                    activation,
                    source,
                    context_summary,
                    trust_delta,
                    default_memory_type,
                    utility_boost,
                    memory_channels,
                ) {
                    ActivationOutcome::Active { id, is_new } => {
                        if is_new {
                            report.new_units += 1;
                        } else {
                            report.reused_units += 1;
                        }
                        report.active_ids.push(id);
                    }
                    ActivationOutcome::Candidate => {
                        report.candidate_observations += 1;
                    }
                }
            }
        }

        let promoted = self.promote_candidates_batch(default_memory_type, context_summary);
        report.candidate_promotions = promoted.len() as u64;
        report.new_units += promoted.len() as u64;
        report.active_ids.extend(promoted);
        report
    }

    pub fn ingest_hierarchy_with_channels(
        &mut self,
        hierarchy: &UnitHierarchy,
        source: SourceKind,
        context_summary: &str,
        default_memory_type: MemoryType,
        memory_channels: &[MemoryChannel],
    ) -> Vec<Uuid> {
        self.ingest_hierarchy_with_channels_report(
            hierarchy,
            source,
            context_summary,
            default_memory_type,
            memory_channels,
        )
        .active_ids
    }

    pub fn ingest_hierarchy_with_channels_report(
        &mut self,
        hierarchy: &UnitHierarchy,
        source: SourceKind,
        context_summary: &str,
        default_memory_type: MemoryType,
        memory_channels: &[MemoryChannel],
    ) -> IngestReport {
        let mut report = IngestReport::default();
        let trust_delta = match source {
            SourceKind::UserInput => 0.08,
            SourceKind::Retrieval => 0.04,
            SourceKind::TrainingDocument => 0.06,
            SourceKind::TrainingUrl => 0.05,
        };

        for units in hierarchy.levels.values() {
            for activation in units {
                match self.ingest_activation(
                    activation,
                    source,
                    context_summary,
                    trust_delta,
                    default_memory_type,
                    memory_channels,
                ) {
                    ActivationOutcome::Active { id, is_new } => {
                        if is_new {
                            report.new_units += 1;
                        } else {
                            report.reused_units += 1;
                        }
                        report.active_ids.push(id);
                    }
                    ActivationOutcome::Candidate => {
                        report.candidate_observations += 1;
                    }
                }
            }
        }

        // Skip per-hierarchy candidate promotion and pattern recording in training mode
        if !self.training_mode {
            let promoted = self.promote_candidates_batch(default_memory_type, context_summary);
            report.candidate_promotions = promoted.len() as u64;
            report.new_units += promoted.len() as u64;
            report.active_ids.extend(promoted);
            report.active_ids.sort();
            report.active_ids.dedup();

            let pattern_cache = self.record_pattern_combinations(&report.active_ids);
            report.cache_hits = pattern_cache.0;
            report.cache_lookups = pattern_cache.1;
        }
        report.bloom_false_positives = self.bloom_filter.stats().false_positives;

        report
    }

    /// Pre-ingestion content validation gate (Layer 21 pollution prevention).
    /// Returns true if the content passes quality checks and should be ingested.
    fn passes_content_validation(&self, activation: &ActivatedUnit, source: SourceKind) -> bool {
        if !self.pollution_detection_enabled {
            return true;
        }

        let normalized = &activation.normalized;
        let content_len = normalized.chars().count();

        // Gate 1: Reject empty or whitespace-only content
        if normalized.trim().is_empty() {
            return false;
        }

        // Gate 2: Reject content that is purely numeric or single-char repetition
        if content_len > 1
            && normalized
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == ',')
        {
            return false;
        }

        // Gate 3: Reject content with excessive repetition (pollution signature)
        if content_len >= self.pollution_min_length {
            let words: Vec<&str> = normalized.split_whitespace().collect();
            if words.len() >= 3 {
                let unique_words: std::collections::HashSet<&str> = words.iter().copied().collect();
                let uniqueness_ratio = unique_words.len() as f32 / words.len() as f32;
                // If less than 30% of words are unique, it's likely garbage/repetition
                if uniqueness_ratio < 0.30 {
                    return false;
                }
            }
        }

        // Gate 4: For retrieval-sourced content, apply stricter validation
        if matches!(source, SourceKind::Retrieval | SourceKind::TrainingUrl) {
            // Reject very short retrieval content (likely noise/fragments)
            if content_len < 3 {
                return false;
            }
            // Reject content that looks like HTML/script artifacts
            if normalized.contains("<script")
                || normalized.contains("javascript:")
                || normalized.contains("cookie") && normalized.contains("accept")
            {
                return false;
            }
        }

        // Gate 5: Check against existing content for near-duplicate pollution.
        // If an existing unit has the same normalized form with much higher quality,
        // reject the new activation to prevent polluted variants from entering.
        if let Some(existing_id) = self.content_index.get(normalized.as_str()) {
            if let Some(existing) = self.cache.get(existing_id) {
                let existing_quality = existing.trust_score * existing.utility_score;
                let new_quality = activation.confidence * activation.utility_score;
                // If existing is dramatically better, still allow (it's a re-observation)
                // But if new content is extremely low quality compared to existing, reject
                if new_quality < existing_quality * 0.1 && existing.frequency > 2 {
                    return false;
                }
            }
        }

        true
    }

    fn ingest_activation(
        &mut self,
        activation: &ActivatedUnit,
        source: SourceKind,
        context_summary: &str,
        trust_delta: f32,
        default_memory_type: MemoryType,
        memory_channels: &[MemoryChannel],
    ) -> ActivationOutcome {
        // Pre-ingestion pollution gate: validate content before accepting
        if !self.passes_content_validation(activation, source) {
            return ActivationOutcome::Candidate; // Silently reject as observation
        }

        // Intent channel gate: block Intent-channel units from Core memory promotion
        let is_intent_channel = memory_channels.contains(&MemoryChannel::Intent);
        let promote_to_core = default_memory_type == MemoryType::Core
            && !(self.intent_channel_core_promotion_blocked && is_intent_channel);
        let bloom_maybe = self.bloom_filter.contains(activation.normalized.as_str());
        if let Some(id) = self
            .content_index
            .get(activation.normalized.as_str())
            .copied()
        {
            let anchor_reuse_threshold = self.anchor_reuse_threshold;
            let anchor_salience_threshold = self.anchor_salience_threshold;
            if let Some(existing_arc) = self.cache.get_mut(&id) {
                let existing = Arc::make_mut(existing_arc);
                existing.frequency += activation.frequency.max(1);
                existing.utility_score =
                    (existing.utility_score * 0.7) + (activation.utility_score * 0.3);
                existing.salience_score =
                    (existing.salience_score * 0.6) + (activation.salience * 0.4);
                existing.confidence = (existing.confidence * 0.7) + (activation.confidence * 0.3);
                existing.last_seen_at = Utc::now();
                existing.trust_score = (existing.trust_score + trust_delta).min(1.0);
                if matches!(source, SourceKind::Retrieval | SourceKind::TrainingUrl) {
                    existing.corroboration_count += 1;
                }
                if promote_to_core {
                    existing.memory_type = MemoryType::Core;
                }
                merge_channels(&mut existing.memory_channels, memory_channels);
                Self::record_context(existing, context_summary);
                Self::apply_anchor_heuristics(
                    existing,
                    anchor_reuse_threshold,
                    anchor_salience_threshold,
                );
                // In training mode: skip clone + reindex + persist.
                // All updates are already in cache (source of truth).
                // Channels don't change within a source, so reindex is no-op.
                if !self.training_mode {
                    let unit_clone = existing.clone();
                    let _ = existing;
                    self.reindex_channels(&unit_clone);
                    self.persist(&unit_clone);
                }
            }
            return ActivationOutcome::Active { id, is_new: false };
        } else if bloom_maybe {
            self.bloom_filter.record_false_positive();
        }

        if self.should_stage_candidate(source, default_memory_type) {
            self.observe_candidate(activation, default_memory_type, memory_channels);
            return ActivationOutcome::Candidate;
        }

        let key = activation.normalized.clone();
        let mut unit = Unit::new(
            activation.content.clone(),
            activation.normalized.clone(),
            activation.level,
            activation.utility_score.max(0.05),
            activation.confidence.max(0.2),
            hashed_position(&activation.normalized, activation.level),
        );
        unit.salience_score = activation.salience;
        unit.trust_score = 0.45 + trust_delta;
        // Use promote_to_core result to determine actual memory type
        // Intent-channel units should stay Episodic when blocked
        unit.memory_type = if promote_to_core {
            MemoryType::Core
        } else {
            MemoryType::Episodic
        };
        unit.memory_channels = normalized_channels(memory_channels);
        if matches!(source, SourceKind::Retrieval | SourceKind::TrainingUrl) {
            unit.corroboration_count = 1;
        }
        Self::record_context(&mut unit, context_summary);
        Self::apply_anchor_heuristics(
            &mut unit,
            self.anchor_reuse_threshold,
            self.anchor_salience_threshold,
        );
        let id = unit.id;
        self.bloom_filter.insert(&key);
        self.content_index.insert(key, id);
        self.reindex_channels(&unit);
        self.persist(&unit);
        self.cache.insert(id, Arc::new(unit));
        ActivationOutcome::Active { id, is_new: true }
    }

    /// Ingest activation with utility boost for StagingEpisodic units (Layer 21 compliance).
    /// Used by unified training to give memory_target: Core units higher initial utility
    /// while still requiring corroboration for actual Core promotion.
    fn ingest_activation_with_utility_boost(
        &mut self,
        activation: &ActivatedUnit,
        source: SourceKind,
        context_summary: &str,
        trust_delta: f32,
        default_memory_type: MemoryType,
        utility_boost: f32,
        memory_channels: &[MemoryChannel],
    ) -> ActivationOutcome {
        // Intent channel gate: block Intent-channel units from Core memory promotion
        let is_intent_channel = memory_channels.contains(&MemoryChannel::Intent);
        let promote_to_core = default_memory_type == MemoryType::Core
            && !(self.intent_channel_core_promotion_blocked && is_intent_channel);
        let bloom_maybe = self.bloom_filter.contains(activation.normalized.as_str());
        if let Some(id) = self
            .content_index
            .get(activation.normalized.as_str())
            .copied()
        {
            let anchor_reuse_threshold = self.anchor_reuse_threshold;
            let anchor_salience_threshold = self.anchor_salience_threshold;
            if let Some(existing_arc) = self.cache.get_mut(&id) {
                let existing = Arc::make_mut(existing_arc);
                existing.frequency += activation.frequency.max(1);
                // Apply utility boost for StagingEpisodic units
                existing.utility_score = ((existing.utility_score * 0.7)
                    + (activation.utility_score * 0.3))
                    .max(utility_boost);
                existing.salience_score =
                    (existing.salience_score * 0.6) + (activation.salience * 0.4);
                existing.confidence = (existing.confidence * 0.7) + (activation.confidence * 0.3);
                existing.last_seen_at = Utc::now();
                existing.trust_score = (existing.trust_score + trust_delta).min(1.0);
                if matches!(
                    source,
                    SourceKind::Retrieval | SourceKind::TrainingUrl | SourceKind::TrainingDocument
                ) {
                    existing.corroboration_count += 1;
                }
                if promote_to_core {
                    existing.memory_type = MemoryType::Core;
                }
                merge_channels(&mut existing.memory_channels, memory_channels);
                Self::record_context(existing, context_summary);
                Self::apply_anchor_heuristics(
                    existing,
                    anchor_reuse_threshold,
                    anchor_salience_threshold,
                );
                if !self.training_mode {
                    let unit_clone = existing.clone();
                    let _ = existing;
                    self.reindex_channels(&unit_clone);
                    self.persist(&unit_clone);
                }
            }
            return ActivationOutcome::Active { id, is_new: false };
        } else if bloom_maybe {
            self.bloom_filter.record_false_positive();
        }

        if self.should_stage_candidate(source, default_memory_type) {
            self.observe_candidate(activation, default_memory_type, memory_channels);
            return ActivationOutcome::Candidate;
        }

        let key = activation.normalized.clone();
        let mut unit = Unit::new(
            activation.content.clone(),
            activation.normalized.clone(),
            activation.level,
            // Apply utility boost for new StagingEpisodic units
            activation.utility_score.max(utility_boost),
            activation.confidence.max(0.2),
            hashed_position(&activation.normalized, activation.level),
        );
        unit.salience_score = activation.salience;
        unit.trust_score = 0.45 + trust_delta;
        // Use promote_to_core result to determine actual memory type
        // Intent-channel units should stay Episodic when blocked
        unit.memory_type = if promote_to_core {
            MemoryType::Core
        } else {
            MemoryType::Episodic
        };
        unit.memory_channels = normalized_channels(memory_channels);
        if matches!(
            source,
            SourceKind::Retrieval | SourceKind::TrainingUrl | SourceKind::TrainingDocument
        ) {
            unit.corroboration_count = 1;
        }
        Self::record_context(&mut unit, context_summary);
        Self::apply_anchor_heuristics(
            &mut unit,
            self.anchor_reuse_threshold,
            self.anchor_salience_threshold,
        );
        let id = unit.id;
        self.bloom_filter.insert(&key);
        self.content_index.insert(key, id);
        self.reindex_channels(&unit);
        self.persist(&unit);
        self.cache.insert(id, Arc::new(unit));
        ActivationOutcome::Active { id, is_new: true }
    }

    pub fn update_positions(&mut self, updates: &[(Uuid, [f32; 3])]) {
        for (id, position) in updates {
            let mut persisted = None;
            if let Some(unit_arc) = self.cache.get_mut(id) {
                let unit = Arc::make_mut(unit_arc);
                unit.semantic_position = *position;
                persisted = Some(unit.clone());
            }
            if let Some(unit) = persisted.as_ref() {
                self.persist(unit);
            }
        }
    }

    pub fn connect_units(&mut self, source_id: Uuid, target_ids: &[Uuid], edge_weight: f32) {
        const MAX_LINKS_PER_UNIT: usize = 64;
        let mut persisted = None;
        if let Some(source_arc) = self.cache.get_mut(&source_id) {
            let source = Arc::make_mut(source_arc);
            let existing: HashSet<Uuid> = source.links.iter().map(|l| l.target_id).collect();
            let mut added = 0;
            for target_id in target_ids {
                if *target_id == source_id || existing.contains(target_id) {
                    continue;
                }
                if source.links.len() >= MAX_LINKS_PER_UNIT {
                    break;
                }
                source.links.push(Link::new(
                    *target_id,
                    crate::types::EdgeType::Semantic,
                    edge_weight,
                ));
                added += 1;
            }
            if added > 0 {
                persisted = Some(source.clone());
            }
        }
        if let Some(unit) = persisted.as_ref() {
            self.persist(unit);
        }
    }

    pub fn related_units(&self, ids: &[Uuid]) -> Vec<Uuid> {
        let mut related = Vec::new();
        let mut seen = HashSet::new();
        for id in ids {
            if let Some(unit) = self.cache.get(id) {
                for link in &unit.links {
                    if seen.insert(link.target_id) {
                        related.push(link.target_id);
                    }
                }
            }
        }
        related
    }

    pub fn apply_feedback(&mut self, feedback: &[FeedbackEvent]) {
        let fallback_target = self.sequence_state.recent_unit_ids.last().copied();
        let mut aggregated: HashMap<Uuid, f32> = HashMap::new();

        for event in feedback {
            if let Some(unit_id) = event.target_unit_id.or(fallback_target) {
                *aggregated.entry(unit_id).or_insert(0.0) += event.impact;
            }
        }

        for (unit_id, total_impact) in aggregated {
            let mut persisted = None;
            if let Some(unit_arc) = self.cache.get_mut(&unit_id) {
                let unit = Arc::make_mut(unit_arc);
                unit.confidence = (unit.confidence + total_impact * 0.05).clamp(0.05, 1.0);
                unit.utility_score = (unit.utility_score + total_impact * 0.03).clamp(0.05, 2.0);
                persisted = Some(unit.clone());
            }
            if let Some(unit) = persisted.as_ref() {
                self.persist(unit);
            }
        }
    }

    pub fn memory_summary(&self) -> String {
        let core = self
            .cache
            .values()
            .filter(|unit| unit.memory_type == MemoryType::Core)
            .count();
        let intent = self
            .channel_index
            .get(&MemoryChannel::Intent)
            .map(HashSet::len)
            .unwrap_or(0);
        let reasoning = self
            .channel_index
            .get(&MemoryChannel::Reasoning)
            .map(HashSet::len)
            .unwrap_or(0);
        let anchors = self
            .cache
            .values()
            .filter(|unit| unit.anchor_status)
            .count();
        format!(
            "units={}, core={}, intent={}, reasoning={}, anchors={}, turn_index={}",
            self.cache.len(),
            core,
            intent,
            reasoning,
            anchors,
            self.sequence_state.turn_index
        )
    }

    pub fn estimate_memory_kb(&self) -> i64 {
        let bytes: usize = self
            .cache
            .values()
            .map(|unit| {
                unit.content.len()
                    + unit.normalized.len()
                    + unit.contexts.iter().map(String::len).sum::<usize>()
                    + (unit.memory_channels.len() * 8)
                    + (unit.links.len() * 32)
                    + UNIT_METADATA_OVERHEAD_BYTES
            })
            .sum();
        let candidate_bytes: usize = self
            .candidate_cache
            .values()
            .filter(|candidate| candidate.status != CandidateStatus::Active)
            .map(|candidate| {
                candidate.content.len()
                    + candidate.normalized.len()
                    + CANDIDATE_METADATA_OVERHEAD_BYTES
            })
            .sum();
        (((bytes + candidate_bytes) as i64) / 1024).max(1)
    }

    /// Lightweight maintenance: pruning, promotion, decay — no SQLite writes, no layout, no snapshot.
    /// Suitable for inline calls during inference.
    pub fn run_maintenance_lightweight(
        &mut self,
        governance: &GovernanceConfig,
    ) -> GovernanceReport {
        self.apply_governance(governance);
        let health = self.database_health();
        let prune_utility_threshold = self.prune_threshold_for(health.maturity_stage);

        let mut to_remove = Vec::new();
        let mut promoted_units = 0u64;
        let mut anchors_protected = 0u64;

        for unit_arc in self.cache.values_mut() {
            let unit = Arc::make_mut(unit_arc);
            if unit.anchor_status {
                anchors_protected += 1;
            }
            if unit.frequency >= self.core_promotion_threshold
                || unit.corroboration_count >= self.promotion_min_corroborations
                || unit.anchor_status
            {
                if unit.memory_type != MemoryType::Core {
                    promoted_units += 1;
                    unit.memory_type = MemoryType::Core;
                }
            }
            let stale_hours = (Utc::now() - unit.last_seen_at).num_hours().max(0) as f32;
            let decay_window_hours = (self.episodic_decay_days.max(1) as f32) * 24.0;
            let decay = (stale_hours / decay_window_hours).min(0.4);
            let recency_penalty = decay * self.recency_decay_rate.max(0.0);
            unit.utility_score =
                ((unit.utility_score * self.lfu_decay_factor) - recency_penalty).max(0.02);

            if !unit.anchor_status
                && unit.memory_type == MemoryType::Episodic
                && unit.utility_score < prune_utility_threshold
                && stale_hours > 24.0
            {
                to_remove.push(unit.id);
            }
        }

        let pruned_units = self.archive_units_by_ids(&to_remove).len() as u64;
        self.pruned_units_total += pruned_units;

        GovernanceReport {
            pruned_units,
            pruned_candidates: 0,
            purged_polluted_units: 0,
            purged_polluted_candidates: 0,
            promoted_units,
            anchors_protected,
            layout_adjustments: 0,
            mean_displacement: 0.0,
            layout_rolled_back: false,
            snapshot_path: String::new(),
            pruning_reasons: Vec::new(),
            pruned_references: Vec::new(),
            pollution_findings: Vec::new(),
        }
    }

    pub fn run_maintenance(&mut self, governance: &GovernanceConfig) -> GovernanceReport {
        self.apply_governance(governance);
        let pollution = self.purge_polluted_memory();
        let health = self.database_health();
        let prune_utility_threshold = self.prune_threshold_for(health.maturity_stage);
        let layout = force_directed_layout(&self.all_units(), &self.semantic_map);
        for (id, position) in &layout.position_updates {
            if let Some(unit_arc) = self.cache.get_mut(id) {
                Arc::make_mut(unit_arc).semantic_position = *position;
            }
        }

        let mut to_remove = Vec::new();
        let mut promoted_units = 0;
        let mut anchors_protected = 0;

        for unit_arc in self.cache.values_mut() {
            let unit = Arc::make_mut(unit_arc);
            if unit.anchor_status {
                anchors_protected += 1;
            }
            if unit.frequency >= self.core_promotion_threshold
                || unit.corroboration_count >= self.promotion_min_corroborations
                || unit.anchor_status
            {
                if unit.memory_type != MemoryType::Core {
                    promoted_units += 1;
                    unit.memory_type = MemoryType::Core;
                }
            }

            let stale_hours = (Utc::now() - unit.last_seen_at).num_hours().max(0) as f32;
            let decay_window_hours = (self.episodic_decay_days.max(1) as f32) * 24.0;
            let decay = (stale_hours / decay_window_hours).min(0.4);
            let recency_penalty = decay * self.recency_decay_rate.max(0.0);
            unit.utility_score =
                ((unit.utility_score * self.lfu_decay_factor) - recency_penalty).max(0.02);
            let anchor_grace_hours = (self.anchor_protection_grace_days.max(1) as i64) * 24;
            let anchor_grace_active =
                (Utc::now() - unit.created_at).num_hours().max(0) < anchor_grace_hours;

            // Never prune process anchors
            let is_process_anchor = if let Some(_hash) = unit
                .is_process_unit
                .then(|| unit.content.as_str())
                .and_then(|_| {
                    // For simplicity here, we assume any process_unit in the process_anchors by ID is protected
                    Some(unit.id)
                }) {
                self.process_anchors.values().any(|&id| id == unit.id)
            } else {
                false
            };

            if !unit.anchor_status
                && !is_process_anchor
                && unit.memory_type == MemoryType::Episodic
                && !anchor_grace_active
                && unit.utility_score < prune_utility_threshold
                && stale_hours > 24.0
            {
                to_remove.push(unit.id);
            }
        }

        let mut pruning_reasons = Vec::new();
        let mut pruned_references = pollution.pruned_references.clone();
        if pollution.purged_units > 0 || pollution.purged_candidates > 0 {
            pruning_reasons.push("polluted_memory".to_string());
        }
        if !to_remove.is_empty() {
            pruning_reasons.push("stale_low_utility".to_string());
            pruned_references.extend(sampled_pruned_references(
                self.cache
                    .values()
                    .filter(|unit| to_remove.contains(&unit.id))
                    .map(|u| (**u).clone())
                    .collect(),
                "stale_low_utility",
            ));
            cap_pruned_references(&mut pruned_references);
        }
        let mut pruned_units = self.archive_units_by_ids(&to_remove).len() as u64;
        self.pruned_units_total += pruned_units;
        let cold_archive_target_kb = (self.cold_archive_threshold_mb.max(0.0) * 1024.0) as i64;
        if cold_archive_target_kb > 0 && self.estimate_memory_kb() > cold_archive_target_kb {
            let overflow = self.prune_to_memory_budget(cold_archive_target_kb);
            pruned_units += overflow.pruned_units;
            anchors_protected += overflow.anchors_protected;
            pruning_reasons.extend(overflow.pruning_reasons);
            pruned_references.extend(overflow.pruned_references);
            cap_pruned_references(&mut pruned_references);
        }
        let candidate_prune = self.prune_candidate_pool_to_budget(i64::MAX);
        if candidate_prune.pruned_candidates > 0 {
            pruning_reasons.push("candidate_pool_budget".to_string());
            pruned_references.extend(candidate_prune.pruned_references);
            cap_pruned_references(&mut pruned_references);
        }

        self.flush_pending_writes();
        let _ = self.db.batch_upsert_units(
            &self
                .cache
                .values()
                .map(|u| (**u).clone())
                .collect::<Vec<_>>(),
        );

        let snapshot_path = self
            .db
            .save_snapshot(&self.all_units(), &self.sequence_state)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "memory_snapshot.msgpack".to_string());

        GovernanceReport {
            pruned_units,
            pruned_candidates: candidate_prune.pruned_candidates,
            purged_polluted_units: pollution.purged_units,
            purged_polluted_candidates: pollution.purged_candidates,
            promoted_units,
            anchors_protected,
            layout_adjustments: layout.position_updates.len() as u64,
            mean_displacement: layout.mean_displacement,
            layout_rolled_back: layout.rolled_back,
            snapshot_path,
            pruning_reasons,
            pruned_references,
            pollution_findings: pollution.findings,
        }
    }

    pub fn prune_to_memory_budget(&mut self, target_memory_kb: i64) -> GovernanceReport {
        let health = self.database_health();
        let prune_threshold = self.prune_threshold_for(health.maturity_stage);
        let anchors_protected = self
            .cache
            .values()
            .filter(|unit| unit.anchor_status)
            .count() as u64;
        let mut current_memory_kb = self.estimate_memory_kb();
        if current_memory_kb <= target_memory_kb.max(0) {
            return GovernanceReport {
                pruned_units: 0,
                pruned_candidates: 0,
                purged_polluted_units: 0,
                purged_polluted_candidates: 0,
                promoted_units: 0,
                anchors_protected,
                layout_adjustments: 0,
                mean_displacement: 0.0,
                layout_rolled_back: false,
                snapshot_path: self
                    .db
                    .save_snapshot(&self.all_units(), &self.sequence_state)
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|_| "memory_snapshot.msgpack".to_string()),
                pruning_reasons: Vec::new(),
                pruned_references: Vec::new(),
                pollution_findings: Vec::new(),
            };
        }

        let candidate_prune = self.prune_candidate_pool_to_budget(target_memory_kb);
        current_memory_kb = candidate_prune.current_memory_kb;

        let mut candidates = self
            .cache
            .values()
            .filter(|unit| {
                let is_process_anchor = self.process_anchors.values().any(|&id| id == unit.id);
                !unit.anchor_status
                    && !is_process_anchor
                    && unit.memory_type == MemoryType::Episodic
            })
            .map(|unit| {
                (
                    unit.id,
                    unit.utility_score,
                    unit.salience_score,
                    unit.last_seen_at,
                    unit.frequency,
                    estimate_unit_kb(unit),
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|lhs, rhs| {
            (lhs.1 >= prune_threshold)
                .cmp(&(rhs.1 >= prune_threshold))
                .then(
                    lhs.1
                        .total_cmp(&rhs.1)
                        .then(lhs.2.total_cmp(&rhs.2))
                        .then(lhs.3.cmp(&rhs.3))
                        .then(lhs.4.cmp(&rhs.4)),
                )
        });

        let mut to_remove = Vec::new();
        for (id, _, _, _, _, estimated_kb) in candidates {
            if current_memory_kb <= target_memory_kb {
                break;
            }
            to_remove.push(id);
            current_memory_kb = (current_memory_kb - estimated_kb.max(1)).max(0);
        }

        let pruned_references = sampled_pruned_references(
            self.cache
                .values()
                .filter(|unit| to_remove.contains(&unit.id))
                .map(|u| (**u).clone())
                .collect(),
            "memory_budget",
        );
        let pruned_units = self.archive_units_by_ids(&to_remove).len() as u64;
        self.pruned_units_total += pruned_units;
        let mut pruning_reasons = vec!["memory_budget".to_string()];
        let mut pruned_references = pruned_references;
        if candidate_prune.pruned_candidates > 0 {
            pruning_reasons.push("candidate_pool_budget".to_string());
            pruned_references.extend(candidate_prune.pruned_references);
            cap_pruned_references(&mut pruned_references);
        }
        let snapshot_path = self
            .db
            .save_snapshot(&self.all_units(), &self.sequence_state)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "memory_snapshot.msgpack".to_string());

        GovernanceReport {
            pruned_units,
            pruned_candidates: candidate_prune.pruned_candidates,
            purged_polluted_units: 0,
            purged_polluted_candidates: 0,
            promoted_units: 0,
            anchors_protected,
            layout_adjustments: 0,
            mean_displacement: 0.0,
            layout_rolled_back: false,
            snapshot_path,
            pruning_reasons,
            pruned_references,
            pollution_findings: Vec::new(),
        }
    }

    pub fn archive_unit_ids(&mut self, ids: &[Uuid]) -> usize {
        self.archive_units_by_ids(ids).len()
    }

    pub fn archive_units_by_ids(&mut self, ids: &[Uuid]) -> Vec<Unit> {
        if ids.is_empty() {
            return Vec::new();
        }

        let mut archived: Vec<Unit> = Vec::new();
        for id in ids {
            if let Some(unit) = self.cache.remove(id) {
                self.content_index.remove(&unit.normalized);
                self.remove_from_channels(&unit);
                archived.push((*unit).clone());
            }
        }

        if !self.content_index.is_empty() {
            self.bloom_filter
                .rebuild(self.content_index.keys().map(String::as_str));
        }

        let _ = self.db.archive_units(&archived);
        archived
    }

    pub fn restore_archived_unit(&mut self, id: Uuid) -> Option<Unit> {
        let restored = self
            .db
            .restore_archived_unit(&id.to_string())
            .ok()
            .flatten()?;
        self.content_index
            .insert(restored.normalized.clone(), restored.id);
        self.bloom_filter.insert(&restored.normalized);
        self.reindex_channels(&restored);
        self.cache.insert(restored.id, Arc::new(restored.clone()));
        Some(restored)
    }

    pub fn check_daily_growth(&self, delta_kb: f32, limit_mb: f32) -> bool {
        let today = Utc::now().date_naive().to_string();
        let current = self.db.growth_for_date(&today).unwrap_or(0.0);
        (current + delta_kb.max(0.0)) <= limit_mb.max(0.0) * 1024.0
    }

    pub fn record_daily_growth(&self, delta_kb: f32) -> f32 {
        let today = Utc::now().date_naive().to_string();
        self.db
            .record_growth(&today, delta_kb)
            .unwrap_or(delta_kb.max(0.0))
    }

    pub fn growth_for_today(&self) -> f32 {
        let today = Utc::now().date_naive().to_string();
        self.db.growth_for_date(&today).unwrap_or(0.0)
    }

    pub fn candidate_pool_memory_kb(&self) -> i64 {
        let bytes: usize = self
            .candidate_cache
            .values()
            .filter(|candidate| candidate.status != CandidateStatus::Active)
            .map(|candidate| {
                candidate.content.len()
                    + candidate.normalized.len()
                    + (candidate.memory_channels.len() * 8)
                    + CANDIDATE_METADATA_OVERHEAD_BYTES
            })
            .sum();
        (bytes as i64 / 1024).max(if self.candidate_cache.is_empty() {
            0
        } else {
            1
        })
    }

    pub fn memory_usage_breakdown_mb(&self) -> (f32, f32, f32) {
        let mut episodic_bytes = 0usize;
        let mut core_bytes = 0usize;
        for unit in self.cache.values() {
            let unit_bytes = unit.content.len()
                + unit.normalized.len()
                + unit.contexts.iter().map(String::len).sum::<usize>()
                + (unit.memory_channels.len() * 8)
                + (unit.links.len() * 32)
                + UNIT_METADATA_OVERHEAD_BYTES;
            match unit.memory_type {
                MemoryType::Episodic => episodic_bytes += unit_bytes,
                MemoryType::Core => core_bytes += unit_bytes,
            }
        }
        let candidate_bytes = self.candidate_pool_memory_kb().max(0) as f32 * 1024.0;
        (
            episodic_bytes as f32 / (1024.0 * 1024.0),
            core_bytes as f32 / (1024.0 * 1024.0),
            candidate_bytes / (1024.0 * 1024.0),
        )
    }

    pub fn audit_pollution(&self, limit: usize) -> Vec<PollutionFinding> {
        let records = self
            .cache
            .values()
            .map(|u| StoredRecord::from_unit(u))
            .collect::<Vec<_>>();
        let mut findings = pollution_findings_for_records(
            &records,
            self.pollution_min_length,
            self.pollution_edge_trim_limit,
            self.pollution_overlap_threshold,
            self.pollution_quality_margin,
            self.pollution_similarity_threshold,
        );
        findings.truncate(limit.max(1));
        findings
    }

    pub fn run_shard_pruning(&mut self, governance: &GovernanceConfig) -> GovernanceReport {
        self.apply_governance(governance);
        let pollution = self.purge_polluted_memory();
        let candidate_prune = self.prune_candidate_pool_to_budget(i64::MAX);
        let snapshot_path = self
            .db
            .save_snapshot(&self.all_units(), &self.sequence_state)
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "memory_snapshot.msgpack".to_string());

        let mut pruning_reasons = Vec::new();
        let mut pruned_references = pollution.pruned_references.clone();
        if pollution.purged_units > 0 || pollution.purged_candidates > 0 {
            pruning_reasons.push("polluted_memory".to_string());
        }
        if candidate_prune.pruned_candidates > 0 {
            pruning_reasons.push("candidate_pool_budget".to_string());
            pruned_references.extend(candidate_prune.pruned_references);
            cap_pruned_references(&mut pruned_references);
        }

        GovernanceReport {
            pruned_units: 0,
            pruned_candidates: candidate_prune.pruned_candidates,
            purged_polluted_units: pollution.purged_units,
            purged_polluted_candidates: pollution.purged_candidates,
            promoted_units: 0,
            anchors_protected: self
                .cache
                .values()
                .filter(|unit| unit.anchor_status)
                .count() as u64,
            layout_adjustments: 0,
            mean_displacement: 0.0,
            layout_rolled_back: false,
            snapshot_path,
            pruning_reasons,
            pruned_references,
            pollution_findings: pollution.findings,
        }
    }

    pub fn merge_training_shard(
        &mut self,
        units: &[Unit],
        candidates: &[UnitCandidate],
    ) -> ShardMergeReport {
        let mut report = ShardMergeReport::default();
        let mut id_map = HashMap::new();

        for shard_unit in units {
            let target_id = if let Some(existing_id) =
                self.content_index.get(&shard_unit.normalized).copied()
            {
                report.reused_units += 1;
                self.merge_unit_record(existing_id, shard_unit);
                existing_id
            } else {
                report.new_units += 1;
                self.insert_merged_unit(shard_unit)
            };
            id_map.insert(shard_unit.id, target_id);
            report.active_ids.push(target_id);
        }

        for shard_unit in units {
            let Some(source_id) = id_map.get(&shard_unit.id).copied() else {
                continue;
            };
            let mut persisted = None;
            if let Some(source_arc) = self.cache.get_mut(&source_id) {
                let source = Arc::make_mut(source_arc);
                for link in &shard_unit.links {
                    let Some(target_id) = id_map.get(&link.target_id).copied() else {
                        continue;
                    };
                    if target_id == source_id {
                        continue;
                    }
                    if source.links.iter().all(|existing| {
                        existing.target_id != target_id || existing.edge_type != link.edge_type
                    }) {
                        source
                            .links
                            .push(Link::new(target_id, link.edge_type, link.weight));
                    }
                }
                persisted = Some(source.clone());
            }
            if let Some(unit) = persisted.as_ref() {
                self.persist(unit);
            }
        }

        for candidate in candidates {
            if self
                .content_index
                .contains_key(candidate.normalized.as_str())
            {
                report.candidate_promotions += 1;
                self.merge_candidate_into_active_unit(candidate);
                continue;
            }
            report.candidate_observations += 1;
            self.merge_candidate_record(candidate);
        }

        report.active_ids.sort();
        report.active_ids.dedup();
        let (cache_hits, cache_lookups) = self.record_pattern_combinations(&report.active_ids);
        report.cache_hits = cache_hits;
        report.cache_lookups = cache_lookups;
        report
    }

    pub fn merge_training_shard_db(
        &mut self,
        shard_db_path: &std::path::Path,
        batch_size: usize,
    ) -> Result<ShardMergeReport, String> {
        let shard_db =
            Db::new(shard_db_path).map_err(|err| format!("open_shard_db_failed:{err}"))?;
        let batch_size = batch_size.max(1);
        let mut report = ShardMergeReport::default();
        let mut id_map = HashMap::new();
        let mut offset = 0usize;

        loop {
            let units = shard_db
                .load_units_batch(offset, batch_size)
                .map_err(|err| format!("load_shard_units_failed:{err}"))?;
            if units.is_empty() {
                break;
            }
            for shard_unit in &units {
                let target_id = if let Some(existing_id) =
                    self.content_index.get(&shard_unit.normalized).copied()
                {
                    report.reused_units += 1;
                    self.merge_unit_record(existing_id, shard_unit);
                    existing_id
                } else {
                    report.new_units += 1;
                    self.insert_merged_unit(shard_unit)
                };
                id_map.insert(shard_unit.id, target_id);
                report.active_ids.push(target_id);
            }
            offset += units.len();
        }

        let mut offset = 0usize;
        loop {
            let units = shard_db
                .load_units_batch(offset, batch_size)
                .map_err(|err| format!("load_shard_units_failed:{err}"))?;
            if units.is_empty() {
                break;
            }
            for shard_unit in &units {
                let Some(source_id) = id_map.get(&shard_unit.id).copied() else {
                    continue;
                };
                let mut persisted = None;
                if let Some(source_arc) = self.cache.get_mut(&source_id) {
                    let source = Arc::make_mut(source_arc);
                    for link in &shard_unit.links {
                        let Some(target_id) = id_map.get(&link.target_id).copied() else {
                            continue;
                        };
                        if target_id == source_id {
                            continue;
                        }
                        if source.links.iter().all(|existing| {
                            existing.target_id != target_id || existing.edge_type != link.edge_type
                        }) {
                            source
                                .links
                                .push(Link::new(target_id, link.edge_type, link.weight));
                        }
                    }
                    persisted = Some(source.clone());
                }
                if let Some(unit) = persisted.as_ref() {
                    self.persist(unit);
                }
            }
            offset += units.len();
        }

        let mut offset = 0usize;
        loop {
            let candidates = shard_db
                .load_candidates_batch(offset, batch_size)
                .map_err(|err| format!("load_shard_candidates_failed:{err}"))?;
            if candidates.is_empty() {
                break;
            }
            for candidate in &candidates {
                if self
                    .content_index
                    .contains_key(candidate.normalized.as_str())
                {
                    report.candidate_promotions += 1;
                    self.merge_candidate_into_active_unit(candidate);
                    continue;
                }
                report.candidate_observations += 1;
                self.merge_candidate_record(candidate);
            }
            offset += candidates.len();
        }

        report.active_ids.sort();
        report.active_ids.dedup();
        let (cache_hits, cache_lookups) = self.record_pattern_combinations(&report.active_ids);
        report.cache_hits = cache_hits;
        report.cache_lookups = cache_lookups;
        Ok(report)
    }

    fn purge_polluted_memory(&mut self) -> PollutionPurgeReport {
        if !self.pollution_detection_enabled {
            return PollutionPurgeReport::default();
        }

        let records = self
            .cache
            .values()
            .map(|u| StoredRecord::from_unit(u))
            .chain(
                self.candidate_cache
                    .values()
                    .filter(|candidate| candidate.status != CandidateStatus::Rejected)
                    .map(StoredRecord::from_candidate),
            )
            .collect::<Vec<_>>();
        let findings = pollution_findings_for_records(
            &records,
            self.pollution_min_length,
            self.pollution_edge_trim_limit,
            self.pollution_overlap_threshold,
            self.pollution_quality_margin,
            self.pollution_similarity_threshold,
        );
        if findings.is_empty() {
            return PollutionPurgeReport::default();
        }

        let mut best_by_polluted = HashMap::new();
        for finding in findings {
            best_by_polluted
                .entry(finding.polluted_id)
                .and_modify(|existing: &mut PollutionFinding| {
                    if finding.quality_delta > existing.quality_delta {
                        *existing = finding.clone();
                    }
                })
                .or_insert(finding);
        }

        let mut report = PollutionPurgeReport::default();
        let mut archived_units = Vec::new();

        for finding in best_by_polluted.into_values() {
            if self.cache.contains_key(&finding.polluted_id) {
                let Some(polluted) = self.cache.get(&finding.polluted_id).cloned() else {
                    continue;
                };
                self.absorb_polluted_unit(&finding, &polluted);
                archived_units.push(finding.polluted_id);
                report.purged_units += 1;
                report.findings.push(finding.clone());
                report.pruned_references.push(PrunedUnitReference {
                    id: polluted.id,
                    content: polluted.content.clone(),
                    normalized: polluted.normalized.clone(),
                    level: polluted.level,
                    memory_type: polluted.memory_type,
                    utility_score: polluted.utility_score,
                    salience_score: polluted.salience_score,
                    confidence: polluted.confidence,
                    trust_score: polluted.trust_score,
                    frequency: polluted.frequency,
                    reason: finding.reason.clone(),
                });
                continue;
            }

            let Some(normalized) = self.candidate_id_index.get(&finding.polluted_id).cloned()
            else {
                continue;
            };
            let Some(polluted) = self.candidate_cache.get(&normalized).cloned() else {
                continue;
            };
            self.absorb_polluted_candidate(&finding, &polluted);
            self.candidate_cache.remove(&normalized);
            self.candidate_id_index.remove(&finding.polluted_id);
            let _ = self.db.delete_candidate(&finding.polluted_id);
            self.pruned_candidates_total += 1;
            report.purged_candidates += 1;
            report.findings.push(finding.clone());
            report.pruned_references.push(PrunedUnitReference {
                id: polluted.id,
                content: polluted.content,
                normalized: polluted.normalized,
                level: polluted.level,
                memory_type: polluted.memory_type,
                utility_score: polluted.utility_score,
                salience_score: 0.0,
                confidence: 0.0,
                trust_score: 0.0,
                frequency: polluted.observation_count,
                reason: finding.reason.clone(),
            });
        }

        if !archived_units.is_empty() {
            self.archive_units_by_ids(&archived_units);
            self.pruned_units_total += archived_units.len() as u64;
        }
        cap_pruned_references(&mut report.pruned_references);
        report.findings.truncate(self.pollution_audit_limit.max(1));
        report
    }

    fn merge_unit_record(&mut self, target_id: Uuid, incoming: &Unit) {
        let mut persisted = None;
        if let Some(existing_arc) = self.cache.get_mut(&target_id) {
            let existing = Arc::make_mut(existing_arc);
            existing.frequency += incoming.frequency.max(1);
            existing.utility_score =
                (existing.utility_score * 0.70) + (incoming.utility_score * 0.30);
            existing.salience_score =
                (existing.salience_score * 0.65) + (incoming.salience_score * 0.35);
            existing.confidence = existing.confidence.max(incoming.confidence);
            existing.trust_score = existing.trust_score.max(incoming.trust_score);
            existing.corroboration_count += incoming.corroboration_count;
            existing.last_seen_at = existing.last_seen_at.max(incoming.last_seen_at);
            existing.memory_type = if incoming.memory_type == MemoryType::Core
                || existing.memory_type == MemoryType::Core
            {
                MemoryType::Core
            } else {
                MemoryType::Episodic
            };
            merge_channels(&mut existing.memory_channels, &incoming.memory_channels);
            for context in &incoming.contexts {
                Self::record_context(existing, context);
            }
            existing.anchor_status |= incoming.anchor_status;
            if !incoming.content.is_empty()
                && (existing.content.len() < incoming.content.len()
                    || trim_outer_non_alphanumeric(&existing.content).len()
                        < trim_outer_non_alphanumeric(&incoming.content).len())
            {
                existing.content = incoming.content.clone();
            }
            Self::apply_anchor_heuristics(
                existing,
                self.anchor_reuse_threshold,
                self.anchor_salience_threshold,
            );
            persisted = Some(existing.clone());
        }
        if let Some(unit) = persisted.as_ref() {
            self.reindex_channels(unit);
            self.persist(unit);
        }
    }

    fn insert_merged_unit(&mut self, incoming: &Unit) -> Uuid {
        let mut unit = incoming.clone();
        if self.cache.contains_key(&unit.id) {
            unit.id = Uuid::new_v4();
        }
        Self::apply_anchor_heuristics(
            &mut unit,
            self.anchor_reuse_threshold,
            self.anchor_salience_threshold,
        );
        let id = unit.id;
        self.content_index.insert(unit.normalized.clone(), id);
        self.bloom_filter.insert(&unit.normalized);
        self.reindex_channels(&unit);
        self.persist(&unit);
        self.cache.insert(id, Arc::new(unit));
        id
    }

    fn merge_candidate_into_active_unit(&mut self, incoming: &UnitCandidate) {
        let Some(target_id) = self.content_index.get(&incoming.normalized).copied() else {
            return;
        };
        let mut persisted = None;
        if let Some(existing_arc) = self.cache.get_mut(&target_id) {
            let existing = Arc::make_mut(existing_arc);
            existing.frequency += incoming.observation_count.max(1);
            existing.utility_score =
                (existing.utility_score * 0.80) + (incoming.utility_score * 0.20);
            existing.salience_score = existing
                .salience_score
                .max(incoming.utility_score.clamp(0.1, 1.0));
            existing.memory_type = if incoming.memory_type == MemoryType::Core
                || existing.memory_type == MemoryType::Core
            {
                MemoryType::Core
            } else {
                MemoryType::Episodic
            };
            merge_channels(&mut existing.memory_channels, &incoming.memory_channels);
            Self::apply_anchor_heuristics(
                existing,
                self.anchor_reuse_threshold,
                self.anchor_salience_threshold,
            );
            persisted = Some(existing.clone());
        }
        if let Some(unit) = persisted.as_ref() {
            self.reindex_channels(unit);
            self.persist(unit);
        }
    }

    fn merge_candidate_record(&mut self, incoming: &UnitCandidate) {
        let normalized = incoming.normalized.clone();
        let stage = self.database_health().maturity_stage;
        let threshold = self.candidate_observation_threshold_for(stage);
        let candidate = self
            .candidate_cache
            .entry(normalized.clone())
            .or_insert_with(|| {
                let mut seeded = incoming.clone();
                if self.candidate_id_index.contains_key(&seeded.id) {
                    seeded.id = Uuid::new_v4();
                }
                seeded
            });
        if candidate.id != incoming.id || candidate.observation_count != incoming.observation_count
        {
            candidate.observation_count += incoming.observation_count.max(1);
            candidate.utility_score =
                (candidate.utility_score * 0.75) + (incoming.utility_score * 0.25);
            candidate.last_seen_at = candidate.last_seen_at.max(incoming.last_seen_at);
            candidate.first_seen_at = candidate.first_seen_at.min(incoming.first_seen_at);
            candidate.promoted_at = candidate.promoted_at.or(incoming.promoted_at);
            candidate.memory_type = if incoming.memory_type == MemoryType::Core
                || candidate.memory_type == MemoryType::Core
            {
                MemoryType::Core
            } else {
                MemoryType::Episodic
            };
            candidate.level = incoming.level;
            merge_channels(&mut candidate.memory_channels, &incoming.memory_channels);
            if candidate.status != CandidateStatus::Active
                && candidate.observation_count >= threshold
            {
                candidate.status = CandidateStatus::Validated;
            }
        } else if candidate.status != CandidateStatus::Active
            && candidate.observation_count >= threshold
        {
            candidate.status = CandidateStatus::Validated;
        }
        self.candidate_id_index
            .insert(candidate.id, candidate.normalized.clone());
        let _ = self.db.upsert_candidate(candidate);
    }

    fn absorb_polluted_unit(&mut self, finding: &PollutionFinding, polluted: &Unit) {
        let Some(canonical_arc) = self.cache.get_mut(&finding.canonical_id) else {
            return;
        };
        let canonical = Arc::make_mut(canonical_arc);
        canonical.frequency += polluted.frequency.max(1);
        let penalty = self.pollution_penalty_factor;
        canonical.utility_score =
            (canonical.utility_score * (1.0 - penalty)) + (polluted.utility_score * penalty);
        canonical.salience_score = (canonical.salience_score * (1.0 - penalty * 1.2).max(0.0))
            + (polluted.salience_score * (penalty * 1.2).min(1.0));
        canonical.confidence = canonical.confidence.max(polluted.confidence);
        canonical.trust_score = canonical.trust_score.max(polluted.trust_score);
        canonical.corroboration_count += polluted.corroboration_count;
        merge_channels(&mut canonical.memory_channels, &polluted.memory_channels);
        for context in &polluted.contexts {
            Self::record_context(canonical, context);
        }
        for link in &polluted.links {
            if canonical
                .links
                .iter()
                .all(|existing| existing.target_id != link.target_id)
            {
                canonical.links.push(link.clone());
            }
        }
        Self::apply_anchor_heuristics(
            canonical,
            self.anchor_reuse_threshold,
            self.anchor_salience_threshold,
        );
        let persisted = canonical.clone();
        self.reindex_channels(&persisted);
        self.persist(&persisted);
    }

    fn absorb_polluted_candidate(&mut self, finding: &PollutionFinding, polluted: &UnitCandidate) {
        if let Some(canonical_arc) = self.cache.get_mut(&finding.canonical_id) {
            let canonical = Arc::make_mut(canonical_arc);
            canonical.frequency += polluted.observation_count.max(1);
            let penalty = self.pollution_penalty_factor;
            canonical.utility_score =
                (canonical.utility_score * (1.0 - penalty)) + (polluted.utility_score * penalty);
            canonical.salience_score = canonical
                .salience_score
                .max(polluted.utility_score.clamp(0.1, 1.0));
            merge_channels(&mut canonical.memory_channels, &polluted.memory_channels);
            Self::apply_anchor_heuristics(
                canonical,
                self.anchor_reuse_threshold,
                self.anchor_salience_threshold,
            );
            let persisted = canonical.clone();
            self.reindex_channels(&persisted);
            self.persist(&persisted);
            return;
        }

        let Some(canonical_normalized) =
            self.candidate_id_index.get(&finding.canonical_id).cloned()
        else {
            return;
        };
        let Some(canonical) = self.candidate_cache.get_mut(&canonical_normalized) else {
            return;
        };
        canonical.observation_count += polluted.observation_count.max(1);
        canonical.utility_score =
            (canonical.utility_score * 0.75) + (polluted.utility_score * 0.25);
        canonical.last_seen_at = canonical.last_seen_at.max(polluted.last_seen_at);
        merge_channels(&mut canonical.memory_channels, &polluted.memory_channels);
        let _ = self.db.upsert_candidate(canonical);
    }

    fn should_stage_candidate(&self, source: SourceKind, default_memory_type: MemoryType) -> bool {
        // Training mode bypasses candidate staging for direct insert throughput
        if self.training_mode {
            return false;
        }
        matches!(
            source,
            SourceKind::TrainingDocument | SourceKind::TrainingUrl
        ) && default_memory_type != MemoryType::Core
    }

    fn observe_candidate(
        &mut self,
        activation: &ActivatedUnit,
        default_memory_type: MemoryType,
        memory_channels: &[MemoryChannel],
    ) {
        let now = Utc::now();
        let normalized = activation.normalized.clone();
        let threshold =
            self.candidate_observation_threshold_for(self.database_health().maturity_stage);
        let candidate = self
            .candidate_cache
            .entry(normalized.clone())
            .or_insert_with(|| UnitCandidate {
                id: Uuid::new_v4(),
                content: activation.content.clone(),
                normalized: normalized.clone(),
                level: activation.level,
                observation_count: 0,
                utility_score: activation.utility_score.max(0.05),
                status: CandidateStatus::Candidate,
                first_seen_at: now,
                last_seen_at: now,
                promoted_at: None,
                memory_type: default_memory_type,
                memory_channels: normalized_channels(memory_channels),
            });
        candidate.observation_count += activation.frequency.max(1);
        candidate.utility_score =
            (candidate.utility_score * 0.7) + (activation.utility_score.max(0.05) * 0.3);
        candidate.last_seen_at = now;
        candidate.level = activation.level;
        candidate.memory_type = default_memory_type;
        merge_channels(&mut candidate.memory_channels, memory_channels);
        if candidate.status != CandidateStatus::Active {
            candidate.status = if candidate.observation_count >= threshold {
                CandidateStatus::Validated
            } else {
                CandidateStatus::Candidate
            };
        }
        self.candidate_id_index
            .insert(candidate.id, candidate.normalized.clone());
        let _ = self.db.upsert_candidate(candidate);
    }

    fn promote_candidates_batch(
        &mut self,
        default_memory_type: MemoryType,
        context_summary: &str,
    ) -> Vec<Uuid> {
        let stage = self.database_health().maturity_stage;
        let observation_threshold = self.candidate_observation_threshold_for(stage);
        let activation_threshold = self.candidate_activation_threshold_for(stage);
        let mut eligible = self
            .candidate_cache
            .values()
            .filter(|candidate| {
                candidate.status != CandidateStatus::Active
                    && candidate.status != CandidateStatus::Rejected
                    && candidate.observation_count >= observation_threshold
                    && candidate.utility_score >= activation_threshold
            })
            .cloned()
            .collect::<Vec<_>>();
        if eligible.is_empty() && matches!(stage, DatabaseMaturityStage::ColdStart) {
            eligible = self
                .candidate_cache
                .values()
                .filter(|candidate| {
                    candidate.status != CandidateStatus::Active
                        && candidate.status != CandidateStatus::Rejected
                        && candidate.utility_score >= (activation_threshold * 0.5).max(0.20)
                })
                .cloned()
                .collect::<Vec<_>>();
        }
        eligible.sort_by(|lhs, rhs| {
            rhs.utility_score
                .total_cmp(&lhs.utility_score)
                .then(rhs.observation_count.cmp(&lhs.observation_count))
        });
        if matches!(stage, DatabaseMaturityStage::ColdStart) {
            eligible.truncate(self.candidate_batch_size.min(8).max(1));
        } else {
            eligible.truncate(self.candidate_batch_size);
        }

        let mut promoted = Vec::new();
        for candidate in eligible {
            if let Some(existing_id) = self.content_index.get(&candidate.normalized).copied() {
                if let Some(stored) = self.candidate_cache.get_mut(&candidate.normalized) {
                    stored.status = CandidateStatus::Active;
                    stored.promoted_at = Some(Utc::now());
                    let _ = self.db.upsert_candidate(stored);
                }
                promoted.push(existing_id);
                continue;
            }

            let mut unit = Unit::new(
                candidate.content.clone(),
                candidate.normalized.clone(),
                candidate.level,
                candidate.utility_score.max(0.05),
                (0.25 + candidate.utility_score * 0.45).clamp(0.2, 0.95),
                hashed_position(&candidate.normalized, candidate.level),
            );
            unit.frequency = candidate.observation_count.max(1);
            unit.salience_score = candidate.utility_score.clamp(0.1, 1.0);
            unit.trust_score = 0.55;
            unit.memory_type = default_memory_type;
            unit.memory_channels = normalized_channels(&candidate.memory_channels);
            Self::record_context(&mut unit, context_summary);
            Self::apply_anchor_heuristics(
                &mut unit,
                self.anchor_reuse_threshold,
                self.anchor_salience_threshold,
            );

            let id = unit.id;
            self.content_index.insert(candidate.normalized.clone(), id);
            self.bloom_filter.insert(&candidate.normalized);
            self.reindex_channels(&unit);
            self.persist(&unit);
            self.cache.insert(id, Arc::new(unit));

            if let Some(stored) = self.candidate_cache.get_mut(&candidate.normalized) {
                stored.status = CandidateStatus::Active;
                stored.promoted_at = Some(Utc::now());
                stored.memory_type = default_memory_type;
                let _ = self.db.upsert_candidate(stored);
            }

            self.candidate_promotions_total += 1;
            promoted.push(id);
        }

        promoted
    }

    fn record_pattern_combinations(&mut self, active_ids: &[Uuid]) -> (u64, u64) {
        if active_ids.len() < 2 {
            return (0, 0);
        }

        let mut hits = 0u64;
        let mut lookups = 0u64;
        for pair in active_ids.windows(2) {
            let Some(parent) = self.cache.get(&pair[0]) else {
                continue;
            };
            let Some(child) = self.cache.get(&pair[1]) else {
                continue;
            };
            let compression_gain =
                ((parent.utility_score + child.utility_score) / 2.0).clamp(0.05, 2.0);
            if let Ok(existed) =
                self.db
                    .upsert_pattern_combination(parent.id, child.id, compression_gain)
            {
                if existed {
                    hits += 1;
                }
                lookups += 1;
            }
        }
        self.pattern_cache_hits += hits;
        self.pattern_cache_lookups += lookups;
        (hits, lookups)
    }

    fn candidate_observation_threshold_for(&self, stage: DatabaseMaturityStage) -> u64 {
        match stage {
            DatabaseMaturityStage::ColdStart => self.cold_start_candidate_observation_threshold,
            DatabaseMaturityStage::Growth => self.growth_candidate_observation_threshold,
            DatabaseMaturityStage::Stable => self.stable_candidate_observation_threshold,
        }
    }

    fn candidate_activation_threshold_for(&self, stage: DatabaseMaturityStage) -> f32 {
        match stage {
            DatabaseMaturityStage::ColdStart => {
                (self.candidate_activation_utility_threshold * 0.75).max(0.30)
            }
            DatabaseMaturityStage::Growth => {
                (self.candidate_activation_utility_threshold * 0.9).max(0.40)
            }
            DatabaseMaturityStage::Stable => self.candidate_activation_utility_threshold,
        }
    }

    fn prune_candidate_pool_to_budget(&mut self, target_memory_kb: i64) -> CandidatePruneReport {
        let mut current_memory_kb = self.estimate_memory_kb();
        if current_memory_kb <= target_memory_kb.max(0) {
            return CandidatePruneReport {
                current_memory_kb,
                pruned_references: Vec::new(),
                pruned_candidates: 0,
            };
        }

        let mut candidates = self
            .candidate_cache
            .values()
            .filter(|candidate| candidate.status != CandidateStatus::Active)
            .map(|candidate| {
                let estimated_kb = ((candidate.content.len()
                    + candidate.normalized.len()
                    + CANDIDATE_METADATA_OVERHEAD_BYTES)
                    as i64
                    / 1024)
                    .max(1);
                (
                    candidate.id,
                    candidate.status,
                    candidate.utility_score,
                    candidate.last_seen_at,
                    estimated_kb,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|lhs, rhs| {
            candidate_status_rank(lhs.1)
                .cmp(&candidate_status_rank(rhs.1))
                .then(lhs.2.total_cmp(&rhs.2))
                .then(lhs.3.cmp(&rhs.3))
        });

        let mut removed = Vec::new();
        for (id, _, _, _, estimated_kb) in candidates {
            if current_memory_kb <= target_memory_kb {
                break;
            }
            removed.push(id);
            current_memory_kb = (current_memory_kb - estimated_kb).max(0);
        }

        let mut pruned_references = Vec::new();
        let mut pruned_candidates = 0u64;
        for id in removed {
            if let Some(normalized) = self.candidate_id_index.remove(&id) {
                if let Some(candidate) = self.candidate_cache.remove(&normalized) {
                    if pruned_references.len() < MAX_PRUNED_REFERENCE_SAMPLES {
                        pruned_references.push(PrunedUnitReference {
                            id: candidate.id,
                            content: candidate.content,
                            normalized: candidate.normalized,
                            level: candidate.level,
                            memory_type: candidate.memory_type,
                            utility_score: candidate.utility_score,
                            salience_score: 0.0,
                            confidence: 0.0,
                            trust_score: 0.0,
                            frequency: candidate.observation_count,
                            reason: "candidate_pool_budget".to_string(),
                        });
                    }
                }
            }
            let _ = self.db.delete_candidate(&id);
            self.pruned_candidates_total += 1;
            pruned_candidates += 1;
        }

        CandidatePruneReport {
            current_memory_kb,
            pruned_references,
            pruned_candidates,
        }
    }

    fn record_context(unit: &mut Unit, context_summary: &str) {
        if context_summary.is_empty() {
            return;
        }
        if !unit
            .contexts
            .iter()
            .any(|context| context == context_summary)
        {
            unit.contexts.push(context_summary.to_string());
        }
    }

    fn apply_anchor_heuristics(
        unit: &mut Unit,
        anchor_reuse_threshold: u64,
        anchor_salience_threshold: f32,
    ) {
        let is_identifier = unit.content.chars().any(|ch| ch.is_ascii_digit())
            || unit
                .content
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false);
        let reusable_span = unit.content.chars().count() >= 5 && unit.confidence >= 0.55;
        let repeated_reuse =
            unit.frequency >= anchor_reuse_threshold && unit.utility_score >= 0.9 && reusable_span;
        let sustained_salience = unit.salience_score >= anchor_salience_threshold
            && reusable_span
            && unit.utility_score >= 0.75;
        if is_identifier || repeated_reuse || sustained_salience {
            unit.anchor_status = true;
            if unit.frequency >= 2 {
                unit.memory_type = MemoryType::Core;
            }
        }
    }

    fn persist(&mut self, unit: &Unit) {
        // In training mode, skip all persistence — cache is source of truth.
        // Bulk flush happens when training_mode is disabled.
        if self.training_mode {
            return;
        }
        if self.write_deferred {
            self.pending_writes.push(unit.clone());
            if self.pending_writes.len() >= self.pending_writes_threshold {
                self.flush_pending_writes();
            }
        } else {
            let _ = self.db.upsert_unit(unit);
        }
    }

    /// Flush all pending writes to database in a single batch transaction
    pub fn flush_pending_writes(&mut self) {
        if self.pending_writes.is_empty() {
            return;
        }
        let units: Vec<Unit> = self.pending_writes.drain(..).collect();
        let _ = self.db.batch_upsert_units(&units);
    }

    /// Enable or disable deferred writes
    pub fn set_write_deferred(&mut self, deferred: bool) {
        self.write_deferred = deferred;
        if !deferred {
            self.flush_pending_writes();
        }
    }

    /// Enable or disable training mode.
    /// When enabled: skips candidate staging (direct insert), skips pattern combination
    /// recording, skips per-unit persistence (cache is source of truth).
    /// When disabled: bulk-flushes entire cache to SQLite in one transaction.
    pub fn set_training_mode(&mut self, enabled: bool) {
        self.training_mode = enabled;
        if enabled {
            self.pending_writes_threshold = 5000;
            self.write_deferred = true;
        } else {
            // Flush any pending writes from before training mode
            self.flush_pending_writes();
            // Bulk-write all cached units to SQLite
            self.flush_cache_to_db();
            self.pending_writes_threshold = 50_000;
        }
    }

    /// Bulk-write all units from the in-memory cache to SQLite.
    /// Used when exiting training mode to persist everything in one transaction.
    fn flush_cache_to_db(&self) {
        if self.cache.is_empty() {
            return;
        }
        let units: Vec<Unit> = self.cache.values().map(|u| (**u).clone()).collect();
        let _ = self.db.batch_upsert_units(&units);
    }

    fn remove_from_channels(&mut self, unit: &Unit) {
        for channel in &unit.memory_channels {
            if let Some(ids) = self.channel_index.get_mut(channel) {
                ids.remove(&unit.id);
            }
        }
    }

    fn reindex_channels(&mut self, unit: &Unit) {
        self.remove_from_channels(unit);
        for channel in &unit.memory_channels {
            self.channel_index
                .entry(*channel)
                .or_default()
                .insert(unit.id);
        }
    }

    fn prune_threshold_for(&self, stage: DatabaseMaturityStage) -> f32 {
        match stage {
            DatabaseMaturityStage::ColdStart => self.cold_start_prune_utility_threshold,
            DatabaseMaturityStage::Growth => self.prune_utility_threshold,
            DatabaseMaturityStage::Stable => self.stable_prune_utility_threshold,
        }
    }

    fn ensure_bootstrap_seeds(&mut self) {
        if !self.cache.is_empty() {
            return;
        }

        for (activation, channels) in bootstrap_seed_activations(
            self.bootstrap_seed_frequency,
            self.bootstrap_seed_utility_floor,
        ) {
            let _ = self.ingest_activation(
                &activation,
                SourceKind::TrainingDocument,
                "bootstrap_seed",
                0.12,
                MemoryType::Core,
                &channels,
            );
        }

        let _ = self
            .db
            .save_snapshot(&self.all_units(), &self.sequence_state);
    }
}

fn source_defaults_to_core(source: SourceKind) -> bool {
    !matches!(source, SourceKind::Retrieval)
}

fn hashed_position(content: &str, level: UnitLevel) -> [f32; 3] {
    let mut acc = [0i32; 3];
    for (idx, byte) in content.bytes().enumerate() {
        acc[idx % 3] += byte as i32 * ((idx as i32 % 5) + 1);
    }
    let level_bias = match level {
        UnitLevel::Char => -0.35,
        UnitLevel::Subword => -0.1,
        UnitLevel::Word => 0.15,
        UnitLevel::Phrase => 0.35,
        UnitLevel::Pattern => 0.55,
    };
    [
        ((acc[0] % 97) as f32 / 24.0) + level_bias,
        ((acc[1] % 113) as f32 / 28.0) - level_bias,
        ((acc[2] % 89) as f32 / 22.0) + (level_bias * 0.5),
    ]
}

fn normalized_channels(channels: &[MemoryChannel]) -> Vec<MemoryChannel> {
    let mut normalized = channels.to_vec();
    normalize_channels(&mut normalized);
    normalized
}

fn normalize_channels(channels: &mut Vec<MemoryChannel>) {
    if channels.is_empty() {
        channels.push(MemoryChannel::Main);
    }
    if !channels.contains(&MemoryChannel::Main) {
        channels.push(MemoryChannel::Main);
    }
    channels.sort_by_key(|channel| match channel {
        MemoryChannel::Main => 0,
        MemoryChannel::Intent => 1,
        MemoryChannel::Reasoning => 2,
    });
    channels.dedup();
}

fn bootstrap_seed_activations(
    seed_frequency: u64,
    utility_floor: f32,
) -> Vec<(ActivatedUnit, Vec<MemoryChannel>)> {
    let mut seeds = Vec::new();
    let frequency = seed_frequency.max(1);
    let utility_floor = utility_floor.clamp(0.05, 0.95);

    let char_seeds = "abcdefghijklmnopqrstuvwxyz0123456789-/:?."
        .chars()
        .map(|ch| ch.to_string())
        .collect::<Vec<_>>();
    let common_words = vec![
        "the", "and", "of", "to", "in", "for", "is", "on", "with", "that", "from", "by", "as",
        "at", "it", "an", "be", "or", "are", "was", "this", "which", "when", "where", "what",
        "how", "why",
    ];
    let reasoning_words = vec![
        "if",
        "then",
        "because",
        "therefore",
        "steps",
        "compare",
        "explain",
        "reasoning",
        "answer",
    ];
    let numeric_patterns = vec!["00", "01", "10", "12", "20", "2024", "2025", "2026"];

    for seed in char_seeds {
        seeds.push((
            ActivatedUnit {
                content: seed.clone(),
                normalized: seed,
                level: UnitLevel::Char,
                utility_score: utility_floor.max(0.32),
                frequency,
                salience: 0.22,
                confidence: 0.72,
                context_hint: "bootstrap_seed:char".to_string(),
            },
            vec![MemoryChannel::Main],
        ));
    }

    for seed in common_words {
        seeds.push((
            ActivatedUnit {
                content: seed.to_string(),
                normalized: seed.to_string(),
                level: UnitLevel::Word,
                utility_score: utility_floor.max(0.48),
                frequency,
                salience: 0.30,
                confidence: 0.78,
                context_hint: "bootstrap_seed:common_word".to_string(),
            },
            vec![MemoryChannel::Main, MemoryChannel::Intent],
        ));
    }

    for seed in reasoning_words {
        seeds.push((
            ActivatedUnit {
                content: seed.to_string(),
                normalized: seed.to_string(),
                level: UnitLevel::Word,
                utility_score: utility_floor.max(0.78),
                frequency,
                salience: 0.82,
                confidence: 0.84,
                context_hint: "bootstrap_seed:reasoning".to_string(),
            },
            vec![MemoryChannel::Main, MemoryChannel::Reasoning],
        ));
    }

    for seed in numeric_patterns {
        seeds.push((
            ActivatedUnit {
                content: seed.to_string(),
                normalized: seed.to_string(),
                level: UnitLevel::Subword,
                utility_score: utility_floor.max(0.44),
                frequency,
                salience: 0.52,
                confidence: 0.76,
                context_hint: "bootstrap_seed:numeric".to_string(),
            },
            vec![MemoryChannel::Main],
        ));
    }

    seeds
}

fn database_health_from_units<'a>(
    units: impl Iterator<Item = &'a Unit>,
    channel_index: &HashMap<MemoryChannel, HashSet<Uuid>>,
    candidates: impl Iterator<Item = &'a UnitCandidate>,
    cold_start_unit_threshold: usize,
    stable_unit_threshold: usize,
    pruned_units: u64,
    wal_size_mb: f32,
    index_fragmentation: f32,
) -> DatabaseHealthMetrics {
    let mut total_units = 0u64;
    let mut core_units = 0u64;
    let mut episodic_units = 0u64;
    let mut anchor_units = 0u64;

    for unit in units {
        total_units += 1;
        if unit.memory_type == MemoryType::Core {
            core_units += 1;
        } else {
            episodic_units += 1;
        }
        if unit.anchor_status {
            anchor_units += 1;
        }
    }

    let mut candidate_units = 0u64;
    let mut validated_candidates = 0u64;
    let mut active_candidates = 0u64;
    let mut rejected_candidates = 0u64;
    for candidate in candidates {
        candidate_units += 1;
        match candidate.status {
            CandidateStatus::Candidate => {}
            CandidateStatus::Validated => validated_candidates += 1,
            CandidateStatus::Active => active_candidates += 1,
            CandidateStatus::Rejected => rejected_candidates += 1,
        }
    }

    let maturity_stage = classify_maturity_stage(
        total_units as usize,
        cold_start_unit_threshold,
        stable_unit_threshold,
    );

    DatabaseHealthMetrics {
        total_units,
        active_units: total_units,
        core_units,
        episodic_units,
        anchor_units,
        intent_units: channel_index
            .get(&MemoryChannel::Intent)
            .map(HashSet::len)
            .unwrap_or(0) as u64,
        reasoning_units: channel_index
            .get(&MemoryChannel::Reasoning)
            .map(HashSet::len)
            .unwrap_or(0) as u64,
        candidate_units,
        validated_candidates,
        active_candidates,
        rejected_candidates,
        pruned_units,
        wal_size_mb,
        index_fragmentation,
        maturity_stage,
    }
}

fn classify_maturity_stage(
    total_units: usize,
    cold_start_unit_threshold: usize,
    stable_unit_threshold: usize,
) -> DatabaseMaturityStage {
    if total_units < cold_start_unit_threshold.max(1) {
        DatabaseMaturityStage::ColdStart
    } else if total_units < stable_unit_threshold.max(cold_start_unit_threshold + 1) {
        DatabaseMaturityStage::Growth
    } else {
        DatabaseMaturityStage::Stable
    }
}

fn normalized_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for token in text.split_whitespace() {
        let cleaned = token
            .trim_matches(|ch: char| !ch.is_alphanumeric())
            .to_ascii_lowercase();
        if cleaned.len() > 1 && !terms.contains(&cleaned) {
            terms.push(cleaned);
        }
    }
    terms
}

fn sampled_pruned_references(units: Vec<Unit>, reason: &str) -> Vec<PrunedUnitReference> {
    units
        .into_iter()
        .take(MAX_PRUNED_REFERENCE_SAMPLES)
        .map(|unit| PrunedUnitReference {
            id: unit.id,
            content: unit.content,
            normalized: unit.normalized,
            level: unit.level,
            memory_type: unit.memory_type,
            utility_score: unit.utility_score,
            salience_score: unit.salience_score,
            confidence: unit.confidence,
            trust_score: unit.trust_score,
            frequency: unit.frequency,
            reason: reason.to_string(),
        })
        .collect()
}

fn cap_pruned_references(references: &mut Vec<PrunedUnitReference>) {
    if references.len() > MAX_PRUNED_REFERENCE_SAMPLES {
        references.truncate(MAX_PRUNED_REFERENCE_SAMPLES);
    }
}

fn lexical_overlap_score(lhs: &[String], rhs: &[String]) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }
    let overlap = lhs.iter().filter(|term| rhs.contains(*term)).count() as f32;
    let normalizer = lhs.len().max(rhs.len()) as f32;
    (overlap / normalizer).clamp(0.0, 1.0)
}

fn merge_channels(target: &mut Vec<MemoryChannel>, incoming: &[MemoryChannel]) {
    target.extend_from_slice(incoming);
    normalize_channels(target);
}

impl StoredRecord {
    fn from_unit(unit: &Unit) -> Self {
        Self {
            id: unit.id,
            content: unit.content.clone(),
            normalized: unit.normalized.clone(),
            level: unit.level,
            utility_score: unit.utility_score,
            salience_score: unit.salience_score,
            confidence: unit.confidence,
            trust_score: unit.trust_score,
            frequency: unit.frequency,
            memory_type: unit.memory_type,
            kind: StoredRecordKind::Unit,
        }
    }

    fn from_candidate(candidate: &UnitCandidate) -> Self {
        Self {
            id: candidate.id,
            content: candidate.content.clone(),
            normalized: candidate.normalized.clone(),
            level: candidate.level,
            utility_score: candidate.utility_score,
            salience_score: candidate.utility_score.clamp(0.1, 1.0),
            confidence: candidate.utility_score.clamp(0.1, 0.9),
            trust_score: 0.0,
            frequency: candidate.observation_count,
            memory_type: candidate.memory_type,
            kind: StoredRecordKind::Candidate,
        }
    }
}

fn pollution_findings_for_records(
    records: &[StoredRecord],
    min_length: usize,
    edge_trim_limit: usize,
    overlap_threshold: f32,
    quality_margin: f32,
    similarity_threshold: f32,
) -> Vec<PollutionFinding> {
    let mut index: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, record) in records.iter().enumerate() {
        index
            .entry(record.normalized.clone())
            .or_default()
            .push(idx);
    }

    let mut findings = Vec::new();
    for (idx, polluted) in records.iter().enumerate() {
        let polluted_len = polluted.normalized.chars().count();
        if polluted_len < min_length {
            continue;
        }
        let polluted_quality = pollution_quality_score(polluted);
        let mut best: Option<PollutionFinding> = None;

        for (variant, reason) in pollution_variants(&polluted.normalized, edge_trim_limit) {
            let overlap_ratio =
                (variant.chars().count() as f32 / polluted_len.max(1) as f32).clamp(0.0, 1.0);
            if overlap_ratio < overlap_threshold {
                continue;
            }
            let Some(candidate_indexes) = index.get(&variant) else {
                continue;
            };
            for candidate_idx in candidate_indexes {
                if *candidate_idx == idx {
                    continue;
                }
                let canonical = &records[*candidate_idx];
                if polluted.kind == StoredRecordKind::Unit
                    && canonical.kind != StoredRecordKind::Unit
                {
                    continue;
                }
                if canonical.normalized.chars().count() < variant.chars().count() {
                    continue;
                }
                if is_edge_trim_reason(reason)
                    && is_clean_alphanumeric_word(polluted)
                    && is_clean_alphanumeric_word(canonical)
                {
                    continue;
                }

                // Similarity gate: require minimum content similarity for multi-word content.
                // Single-word variants are already validated by overlap_threshold above,
                // and word-level Jaccard is meaningless for single tokens.
                let is_multi_word =
                    polluted.normalized.contains(' ') || canonical.normalized.contains(' ');
                if is_multi_word {
                    let similarity = crate::common::similarity::SimilarityUtils::jaccard_similarity(
                        &polluted.normalized,
                        &canonical.normalized,
                    );
                    if similarity < similarity_threshold {
                        continue;
                    }
                }

                let canonical_quality = pollution_quality_score(canonical);
                let quality_delta = canonical_quality - polluted_quality;
                if quality_delta < quality_margin {
                    continue;
                }

                let finding = PollutionFinding {
                    polluted_id: polluted.id,
                    polluted_content: polluted.content.clone(),
                    polluted_normalized: polluted.normalized.clone(),
                    polluted_level: polluted.level,
                    canonical_id: canonical.id,
                    canonical_content: canonical.content.clone(),
                    canonical_normalized: canonical.normalized.clone(),
                    canonical_level: canonical.level,
                    overlap_ratio,
                    quality_delta,
                    reason: format!("{reason}:{}", canonical.normalized),
                };
                if best
                    .as_ref()
                    .map(|existing| finding.quality_delta > existing.quality_delta)
                    .unwrap_or(true)
                {
                    best = Some(finding);
                }
            }
        }

        if let Some(finding) = best {
            findings.push(finding);
        }
    }

    findings
}

fn pollution_variants(normalized: &str, edge_trim_limit: usize) -> Vec<(String, &'static str)> {
    let mut variants = Vec::new();
    let trimmed = trim_outer_non_alphanumeric(normalized);
    if trimmed != normalized {
        variants.push((trimmed.clone(), "outer_punctuation_variant"));
    }

    let chars = normalized.chars().collect::<Vec<_>>();
    for trim in 1..=edge_trim_limit.min(chars.len().saturating_sub(1)) {
        let leading = chars[trim..].iter().collect::<String>();
        variants.push((leading, "leading_fragment_variant"));
        let trailing = chars[..chars.len() - trim].iter().collect::<String>();
        variants.push((trailing, "trailing_fragment_variant"));
    }

    variants.sort();
    variants.dedup();
    variants
}

fn trim_outer_non_alphanumeric(text: &str) -> String {
    text.trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_string()
}

fn is_edge_trim_reason(reason: &str) -> bool {
    matches!(
        reason,
        "leading_fragment_variant" | "trailing_fragment_variant"
    )
}

fn is_clean_alphanumeric_word(record: &StoredRecord) -> bool {
    record.level == UnitLevel::Word
        && !record.normalized.is_empty()
        && record.normalized.chars().all(|ch| ch.is_alphanumeric())
}

fn punctuation_ratio(text: &str) -> f32 {
    let visible = text.chars().filter(|ch| !ch.is_whitespace()).count().max(1) as f32;
    let punctuation = text
        .chars()
        .filter(|ch| !ch.is_alphanumeric() && !ch.is_whitespace())
        .count() as f32;
    (punctuation / visible).clamp(0.0, 1.0)
}

fn pollution_quality_score(record: &StoredRecord) -> f32 {
    let char_len = record.normalized.chars().count().max(1) as f32;
    let start_clean = record
        .normalized
        .chars()
        .next()
        .map(|ch| ch.is_alphanumeric())
        .unwrap_or(false) as u8 as f32;
    let end_clean = record
        .normalized
        .chars()
        .last()
        .map(|ch| ch.is_alphanumeric())
        .unwrap_or(false) as u8 as f32;
    let level_bonus = match record.level {
        UnitLevel::Char => -0.10,
        UnitLevel::Subword => 0.05,
        UnitLevel::Word => 0.28,
        UnitLevel::Phrase => 0.22,
        UnitLevel::Pattern => -0.04,
    };
    let punctuation_penalty = punctuation_ratio(&record.normalized) * 0.45;
    let trimmed_core_bonus =
        (trim_outer_non_alphanumeric(&record.normalized) == record.normalized) as u8 as f32 * 0.08;
    let core_bonus = (record.memory_type == MemoryType::Core) as u8 as f32 * 0.04;
    let frequency_bonus = ((record.frequency as f32).ln_1p() / 8.0).clamp(0.0, 0.10);
    0.18 + (start_clean * 0.20)
        + (end_clean * 0.20)
        + level_bonus
        + (record.confidence.clamp(0.0, 1.0) * 0.18)
        + (record.utility_score.clamp(0.0, 1.5) * 0.12)
        + (record.salience_score.clamp(0.0, 1.0) * 0.08)
        + (record.trust_score.clamp(0.0, 1.0) * 0.04)
        + core_bonus
        + trimmed_core_bonus
        + frequency_bonus
        + (char_len / 12.0).clamp(0.0, 0.18)
        - punctuation_penalty
}

fn estimate_unit_kb(unit: &Unit) -> i64 {
    let bytes = unit.content.len()
        + unit.normalized.len()
        + unit.contexts.iter().map(String::len).sum::<usize>()
        + (unit.memory_channels.len() * 8)
        + (unit.links.len() * 32)
        + UNIT_METADATA_OVERHEAD_BYTES;
    (bytes as i64 / 1024).max(1)
}

fn candidate_status_rank(status: CandidateStatus) -> u8 {
    match status {
        CandidateStatus::Rejected => 0,
        CandidateStatus::Candidate => 1,
        CandidateStatus::Validated => 2,
        CandidateStatus::Active => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::{pollution_findings_for_records, MemoryStore, StoredRecord, StoredRecordKind};
    use crate::classification::builder::UnitBuilder;
    use crate::classification::hierarchy::HierarchicalUnitOrganizer;
    use crate::classification::input;
    use crate::config::{GovernanceConfig, UnitBuilderConfig};
    use crate::types::{
        ActivatedUnit, DatabaseMaturityStage, FeedbackEvent, MemoryChannel, MemoryType, SourceKind,
        Unit, UnitHierarchy, UnitLevel,
    };
    use chrono::Utc;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use uuid::Uuid;

    #[test]
    fn empty_store_bootstraps_seed_units() {
        let db_path = std::env::temp_dir().join(format!("spse_seed_{}.db", Uuid::new_v4()));
        let store = MemoryStore::new(db_path.to_str().expect("db path"));
        let health = store.database_health();

        assert!(health.total_units > 0);
        assert!(store
            .all_units()
            .iter()
            .any(|unit| unit.normalized == "the"));
        assert!(store
            .all_units()
            .iter()
            .any(|unit| unit.normalized == "because"));
        assert_eq!(health.maturity_stage, DatabaseMaturityStage::ColdStart);
    }

    #[test]
    fn targeted_feedback_updates_the_selected_unit() {
        let db_path = std::env::temp_dir().join(format!("spse_feedback_{}.db", Uuid::new_v4()));
        let mut store = MemoryStore::new(db_path.to_str().expect("db path"));
        let packet = input::ingest_raw("clarity anchors clarity", true);
        let build_output = UnitBuilder::ingest(&packet);
        let hierarchy =
            HierarchicalUnitOrganizer::organize(&build_output, &UnitBuilderConfig::default());
        let inserted =
            store.ingest_hierarchy(&hierarchy, SourceKind::TrainingDocument, "feedback_test");
        let target = inserted
            .iter()
            .find_map(|id| {
                store
                    .get_unit(id)
                    .filter(|unit| unit.normalized == "clarity")
                    .map(|unit| unit.id)
            })
            .expect("target unit");
        let before = store.get_unit(&target).expect("unit").utility_score;

        store.apply_feedback(&[FeedbackEvent {
            layer: 18,
            event: "test_feedback".to_string(),
            impact: 1.0,
            target_unit_id: Some(target),
        }]);

        let after = store.get_unit(&target).expect("unit").utility_score;
        assert!(after > before);
    }

    #[test]
    fn training_units_stage_in_candidates_before_activation() {
        let db_path = std::env::temp_dir().join(format!("spse_candidate_{}.db", Uuid::new_v4()));
        let mut governance = GovernanceConfig::default();
        governance.cold_start_candidate_observation_threshold = 3;
        governance.candidate_activation_utility_threshold = 10.0;
        let mut store =
            MemoryStore::new_with_governance(db_path.to_str().expect("db path"), &governance);
        let hierarchy = UnitHierarchy {
            levels: BTreeMap::from([(
                "word".to_string(),
                vec![ActivatedUnit {
                    content: "ReasoningAnchor".to_string(),
                    normalized: "reasoninganchor".to_string(),
                    level: UnitLevel::Word,
                    utility_score: 0.82,
                    frequency: 1,
                    salience: 0.64,
                    confidence: 0.78,
                    context_hint: "candidate_test".to_string(),
                }],
            )]),
            anchors: Vec::new(),
            entities: Vec::new(),
        };

        let first = store.ingest_hierarchy_with_channels_report(
            &hierarchy,
            SourceKind::TrainingDocument,
            "candidate_test",
            MemoryType::Episodic,
            &[MemoryChannel::Main],
        );
        assert!(first.active_ids.is_empty());
        assert_eq!(first.candidate_observations, 1);

        let second = store.ingest_hierarchy_with_channels_report(
            &hierarchy,
            SourceKind::TrainingDocument,
            "candidate_test",
            MemoryType::Episodic,
            &[MemoryChannel::Main],
        );
        assert!(second.active_ids.is_empty());

        let mut promotion_governance = governance.clone();
        promotion_governance.candidate_activation_utility_threshold = 0.55;
        store.apply_governance(&promotion_governance);

        let third = store.ingest_hierarchy_with_channels_report(
            &hierarchy,
            SourceKind::TrainingDocument,
            "candidate_test",
            MemoryType::Episodic,
            &[MemoryChannel::Main],
        );
        assert_eq!(third.candidate_promotions, 1);
        assert_eq!(third.active_ids.len(), 1);

        let health = store.database_health();
        assert_eq!(health.active_candidates, 1);
        assert!(health.candidate_units >= 1);
        assert!(store
            .all_units()
            .iter()
            .any(|unit| unit.normalized == "reasoninganchor"));
    }

    #[test]
    fn pollution_audit_does_not_trim_clean_full_words_into_other_words() {
        let records = vec![
            StoredRecord {
                id: Uuid::new_v4(),
                content: "then".to_string(),
                normalized: "then".to_string(),
                level: UnitLevel::Word,
                utility_score: 0.78,
                salience_score: 0.30,
                confidence: 0.70,
                trust_score: 0.50,
                frequency: 14,
                memory_type: MemoryType::Core,
                kind: StoredRecordKind::Unit,
            },
            StoredRecord {
                id: Uuid::new_v4(),
                content: "the".to_string(),
                normalized: "the".to_string(),
                level: UnitLevel::Word,
                utility_score: 0.82,
                salience_score: 0.35,
                confidence: 0.74,
                trust_score: 0.52,
                frequency: 28,
                memory_type: MemoryType::Core,
                kind: StoredRecordKind::Unit,
            },
        ];

        let findings = pollution_findings_for_records(&records, 3, 3, 0.65, 0.05, 0.3);
        assert!(findings.is_empty());
    }

    #[test]
    fn pollution_audit_still_catches_punctuation_wrapped_variants() {
        let records = vec![
            StoredRecord {
                id: Uuid::new_v4(),
                content: "\"Kraemer,\"".to_string(),
                normalized: "\"kraemer,\"".to_string(),
                level: UnitLevel::Word,
                utility_score: 0.61,
                salience_score: 0.22,
                confidence: 0.48,
                trust_score: 0.10,
                frequency: 4,
                memory_type: MemoryType::Episodic,
                kind: StoredRecordKind::Unit,
            },
            StoredRecord {
                id: Uuid::new_v4(),
                content: "Kraemer".to_string(),
                normalized: "kraemer".to_string(),
                level: UnitLevel::Word,
                utility_score: 0.84,
                salience_score: 0.34,
                confidence: 0.72,
                trust_score: 0.52,
                frequency: 9,
                memory_type: MemoryType::Core,
                kind: StoredRecordKind::Unit,
            },
        ];

        let findings = pollution_findings_for_records(&records, 3, 3, 0.65, 0.05, 0.3);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].canonical_normalized, "kraemer");
    }

    #[test]
    fn should_prune_unit_protects_process_anchors() {
        let db_path =
            std::env::temp_dir().join(format!("spse_process_anchor_{}.db", Uuid::new_v4()));
        let governance = GovernanceConfig::default();
        let mut store =
            MemoryStore::new_with_governance(db_path.to_str().expect("db path"), &governance);

        // Create a process unit
        let unit_id = Uuid::new_v4();
        let structure_hash = 12345u64;
        store.register_process_anchor(structure_hash, unit_id);

        // Create a unit that would normally be pruned
        let unit = Unit {
            id: unit_id,
            content: "test reasoning pattern".to_string(),
            normalized: "test reasoning pattern".to_string(),
            level: UnitLevel::Phrase,
            utility_score: 0.1, // Low utility
            salience_score: 0.3,
            confidence: 0.5,
            trust_score: 0.5,
            frequency: 1,
            memory_type: MemoryType::Episodic,
            anchor_status: false,
            created_at: Utc::now() - chrono::Duration::hours(48), // Old
            last_seen_at: Utc::now() - chrono::Duration::hours(48), // Stale
            is_process_unit: true,
            ..Default::default()
        };
        store.cache.insert(unit_id, Arc::new(unit.clone()));

        // Should NOT prune because it's a process anchor
        let should_prune = store.should_prune_unit(&unit, 0.2, 48.0, false);
        assert!(!should_prune, "Process anchor should not be pruned");
    }

    #[test]
    fn should_prune_unit_allows_normal_pruning() {
        let db_path = std::env::temp_dir().join(format!("spse_normal_prune_{}.db", Uuid::new_v4()));
        let governance = GovernanceConfig::default();
        let store =
            MemoryStore::new_with_governance(db_path.to_str().expect("db path"), &governance);

        // Create a normal unit that should be pruned
        let unit = Unit {
            id: Uuid::new_v4(),
            content: "test content".to_string(),
            normalized: "test content".to_string(),
            level: UnitLevel::Phrase,
            utility_score: 0.1, // Low utility
            salience_score: 0.3,
            confidence: 0.5,
            trust_score: 0.5,
            frequency: 1,
            memory_type: MemoryType::Episodic,
            anchor_status: false,
            created_at: Utc::now() - chrono::Duration::hours(48),
            last_seen_at: Utc::now() - chrono::Duration::hours(48),
            is_process_unit: false,
            ..Default::default()
        };

        // Should prune because it's low utility, stale, and not protected
        let should_prune = store.should_prune_unit(&unit, 0.2, 48.0, false);
        assert!(should_prune, "Normal low-utility unit should be pruned");
    }

    #[test]
    fn reasoning_patterns_for_query_returns_matching_patterns() {
        let db_path =
            std::env::temp_dir().join(format!("spse_reasoning_pattern_{}.db", Uuid::new_v4()));
        let governance = GovernanceConfig::default();
        let mut store =
            MemoryStore::new_with_governance(db_path.to_str().expect("db path"), &governance);

        // Create and register a reasoning pattern
        let pattern_id = Uuid::new_v4();
        let pattern = Unit {
            id: pattern_id,
            content: "Step 1: Analyze the problem. Step 2: Break it down.".to_string(),
            normalized: "step 1 analyze the problem step 2 break it down".to_string(),
            level: UnitLevel::Phrase,
            utility_score: 0.8,
            salience_score: 0.7,
            confidence: 0.9,
            trust_score: 0.8,
            frequency: 5,
            memory_type: MemoryType::Episodic,
            anchor_status: true,
            is_process_unit: true,
            ..Default::default()
        };
        store.cache.insert(pattern_id, Arc::new(pattern));
        store.register_reasoning_pattern(crate::types::ReasoningType::General, pattern_id);

        // Query for patterns
        let matches = store.reasoning_patterns_for_query("analyze the problem", None, 5);
        assert!(
            !matches.is_empty(),
            "Should find matching reasoning pattern"
        );
        assert_eq!(matches[0].unit_id, pattern_id);
        assert!(matches[0].similarity > 0.0);
    }

    #[test]
    fn register_reasoning_pattern_indexes_by_type() {
        let db_path =
            std::env::temp_dir().join(format!("spse_reasoning_index_{}.db", Uuid::new_v4()));
        let governance = GovernanceConfig::default();
        let mut store =
            MemoryStore::new_with_governance(db_path.to_str().expect("db path"), &governance);

        let pattern_id = Uuid::new_v4();
        store.register_reasoning_pattern(crate::types::ReasoningType::Mathematical, pattern_id);

        // Query with type hint should filter correctly
        let unit = Unit {
            id: pattern_id,
            content: "Calculate x + y = z".to_string(),
            normalized: "calculate x y z".to_string(),
            level: UnitLevel::Phrase,
            utility_score: 0.7,
            salience_score: 0.5,
            confidence: 0.8,
            trust_score: 0.6,
            frequency: 2,
            memory_type: MemoryType::Episodic,
            is_process_unit: true,
            ..Default::default()
        };
        store.cache.insert(pattern_id, Arc::new(unit));

        let matches = store.reasoning_patterns_for_query(
            "calculate",
            Some(crate::types::ReasoningType::Mathematical),
            5,
        );
        assert!(!matches.is_empty(), "Should find pattern with type hint");
    }
}
