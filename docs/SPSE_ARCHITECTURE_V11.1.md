# SPSE Engine Architecture Documentation v11.1

**Document Version:** 11.1 (Revised)  
**Last Updated:** Derived from live codebase  
**Status:** Reflects current implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [21-Layer Architecture](#3-21-layer-architecture)
4. [Core Data Structures](#4-core-data-structures)
5. [Configuration System](#5-configuration-system)
6. [Engine Pipeline](#6-engine-pipeline)
7. [Classification System](#7-classification-system)
8. [Memory Architecture](#8-memory-architecture)
9. [Dynamic Reasoning](#9-dynamic-reasoning)
10. [GPU Acceleration](#10-gpu-acceleration)
11. [Telemetry & Observability](#11-telemetry--observability)
12. [Training Pipeline & Seed Generation](#12-training-pipeline--seed-generation)
13. [API Specifications](#13-api-specifications)
14. [Priority Scheduler](#14-priority-scheduler)
15. [Security & Trust](#15-security--trust)
16. [Directory Structure](#16-directory-structure)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

The SPSE (Semantic Processing & Storage Engine) is a **privacy-first, config-driven, retrieval-augmented intelligence engine** written in Rust. It processes natural language through a **21-layer pipeline** that ingests text, discovers reusable semantic units via rolling hash, routes them through a 3D spatial map, scores candidates across 7 dimensions, and resolves answers using adaptive runtime profiles.

### Key Design Principles

- **No hardcoded thresholds** — every numeric value lives in `config/config.yaml` via `EngineConfig`
- **No external open sources** — training/seeding uses only internal datasets; the engine acquires knowledge at runtime via web retrieval triggered during reasoning
- **Auto-Mode only** — engine operates exclusively in auto-intelligence mode; user toggles for temperature, reasoning depth, and creative level are ignored
- **Auto-retrieval** — web retrieval is always enabled, triggered automatically by the reasoning/intent system when confidence is low or the query is open-world; no manual config gate
- **Intelligence, not knowledge** — seed datasets teach *how to think* (reasoning chains, retrieval triggers, confidence gating), not factual content
- **Calculation-based classification** — intent, tone, and resolver mode inferred via memory-backed spatial pattern matching (replaces heuristic detection)
- **Dynamic reasoning** — confidence-gated internal reasoning loop triggered automatically, not by user request; signals retrieval when internal confidence stays low
- **Privacy-first** — PII stripping (L10), no external telemetry, local SQLite persistence
- **GPU-optional** — `wgpu`-based acceleration behind `#[cfg(feature = "gpu")]` with transparent CPU fallback
- **Memory channels** — three isolated channels (Main, Intent, Reasoning) prevent cross-contamination
- **Pollution prevention at source** — pre-ingestion content validation gate rejects empty, repetitive, artifact-laden, and low-quality content before it enters memory; post-hoc detection remains as a safety net

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Rust (2021 edition) |
| Persistence | SQLite via `rusqlite` |
| HTTP Server | `axum` + `tokio` |
| Serialization | `serde` (JSON, YAML), `prost` (Protobuf) |
| GPU | `wgpu` (feature-gated) |
| Config | YAML (`config/config.yaml`) |
| Concurrency | `Arc<Mutex<T>>`, `arc_swap::ArcSwap`, `crossbeam_channel` |

---

## 2. System Overview

### Module Map (`src/lib.rs`)

```
pub mod api              // REST + OpenAI-compatible API (axum)
pub mod bloom_filter     // Probabilistic membership testing (UnitBloomFilter)
pub mod classification   // Calculation-based intent/tone/resolver inference
pub mod common           // Shared utilities (scoring, matching, selection, similarity, dedup)
pub mod config           // EngineConfig and all sub-configs
pub mod document         // Document ingestion (PDF, DOCX, plain text)
pub mod drill_lib        // Drill framework for system testing
pub mod engine           // Core Engine struct and 21-layer pipeline orchestration
pub mod gpu              // GPU acceleration (feature-gated wgpu)
pub mod layers           // 14 layer implementation files
pub mod memory           // MemoryStore (L4/L21) + DynamicMemoryAllocator
pub mod open_sources     // Open data source catalog (Wikidata, Wikipedia, etc.)
pub mod persistence      // SQLite persistence layer
pub mod proto            // Protobuf definitions (api.proto)
pub mod region_index     // Regional spatial index
pub mod scheduler        // Priority-based work scheduling (4 priorities)
pub mod seed             // Dataset generators (dialogue, entity, dryrun)
pub mod spatial_index    // SpatialGrid for O(log N) spatial queries
pub mod stress_drill_lib // Stress testing drills
pub mod crash_drill_lib  // Crash resilience drills
pub mod telemetry        // L20: worker, hot store, latency, trace
pub mod training         // Training pipeline orchestration
pub mod types            // All core type definitions (~1278 lines)
```

### Engine Struct (`src/engine.rs`)

The `Engine` is the central orchestrator:

```rust
pub struct Engine {
    config: EngineConfig,
    memory: Arc<Mutex<MemoryStore>>,              // L4/L21 — SQLite-backed unit store
    memory_snapshot: Arc<ArcSwap<MemorySnapshot>>, // Lock-free read path
    scheduler: Arc<PriorityScheduler>,             // 4-priority work queue
    retriever: RetrievalPipeline,                  // L11 — external search
    merger: EvidenceMerger,                        // L13 — conflict resolution
    decoder: OutputDecoder,                        // L17 — answer finalization
    feedback: FeedbackController,                  // L18 — learning events
    feedback_tx: Sender<Vec<FeedbackEvent>>,       // Async feedback channel (1024 cap)
    safety: TrustSafetyValidator,                  // L12/L19 — trust assessment
    jobs: Arc<Mutex<HashMap<String, TrainingJobStatus>>>,
    session_documents: Arc<Mutex<SessionDocuments>>,
    observer: Option<TestObserver>,                // Test observation hooks
    telemetry_worker: Option<TelemetryWorker>,     // L20 — async event emission
    latency_monitor: Arc<LatencyMonitor>,          // p50/p95/p99 tracking
    dynamic_memory: Arc<DynamicMemoryAllocator>,   // Reasoning buffer allocation
    trace_context: Arc<Mutex<TraceContext>>,        // Session/trace ID management
    classification_calculator: ClassificationCalculator, // Intent/tone/resolver inference
    spatial_grid: Arc<Mutex<SpatialGrid>>,         // Classification pattern retrieval
}
```

### Background Workers

Two background threads are spawned at engine initialization:

1. **Maintenance worker** — runs every `governance.maintenance_interval_secs` (30s), performs memory governance (pruning, promotion, layout)
2. **Feedback worker** — drains the `feedback_rx` channel and applies feedback events to memory store

---

## 3. 21-Layer Architecture

| Layer | Name | Source File | Config Section | Responsibility |
|-------|------|------------|----------------|----------------|
| L1 | Input Ingestion | `layers/input.rs` | — | Normalize raw text, create `InputPacket` |
| L2 | Unit Builder | `layers/builder.rs` | `layer_2_unit_builder` | Rolling hash discovery, unit activation, utility/salience/confidence scoring |
| L3 | Hierarchy Organizer | `layers/hierarchy.rs` | (uses builder config) | Level grouping (Char→Subword→Word→Phrase→Pattern), anchor/entity extraction |
| L4 | Memory Ingestion | `memory/store.rs` | `layer_21_memory_governance` | Unit persistence, candidate observation, channel-aware ingestion |
| L5 | Semantic Router | `layers/router.rs` | `layer_5_semantic_map` | 3D spatial routing, neighbor selection, force-directed layout, escape profiles |
| L6 | Context Manager | `layers/context.rs` | — | Context matrix construction, sequence state, task entities |
| L7 | Intent Detector | `layers/intent.rs` | `intent` | Intent classification (delegates to `ClassificationCalculator`), entropy calculation |
| L8 | Adaptive Runtime | `engine.rs` | `adaptive_behavior` | Profile selection (10 intent profiles), weight adjustment |
| L9 | Retrieval Decision | `layers/intent.rs` | `layer_9_retrieval_gating` | Entropy/freshness/disagreement/cost scoring → `should_retrieve` decision |
| L10 | Query Builder | `layers/query.rs` | `layer_10_query_builder` | Safe query construction, PII stripping, semantic expansion |
| L11 | Retrieval Pipeline | `layers/retrieval.rs` | `layer_11_retrieval` | External source fetching (SearxNG), response caching |
| L12 | Safety Validator | `layers/safety.rs` | `layer_19_trust_heuristics` | Trust assessment, document filtering, HTTPS enforcement |
| L13 | Evidence Merger | `layers/merge.rs` | `layer_13_evidence_merge` | Conflict detection, agreement scoring, trust-weighted merge |
| L14 | Candidate Scorer | `layers/search.rs` | `layer_14_candidate_scoring` | 7-dimensional feature scoring (GPU-accelerated when available) |
| L15 | Resolver Mode | `engine.rs` | `adaptive_behavior` | Deterministic / Balanced / Exploratory mode decision |
| L16 | Fine Resolver | `layers/resolver.rs` | `layer_16_fine_resolver` | Top-k selection with temperature, intent shaping, anchor protection |
| L17 | Output Decoder | `layers/output.rs` | — | Answer finalization, sentence extraction, evidence grounding |
| L18 | Feedback Controller | `layers/feedback.rs` | — | Learning events, impact scoring, weight adjustment signals |
| L19 | Trust/Safety | `layers/safety.rs` | `layer_19_trust_heuristics` | Source validation, allowlist management, format trust adjustments |
| L20 | Telemetry | `telemetry/` | `layer_20_telemetry` | Async event emission, hot SQLite store, cold log, latency monitoring |
| L21 | Governance | `memory/store.rs` | `layer_21_memory_governance` | Pruning, promotion, pollution detection, maintenance cycles |

### Layer Timing & Telemetry

The engine records wall-clock latency for key layers and feeds them into `LatencyMonitor`:

```
Layer 2  (Unit Building)     → latency_monitor.record(2, ms)
Layer 5  (Semantic Routing)  → latency_monitor.record(5, ms)
Layer 9  (Retrieval Decision)→ latency_monitor.record(9, ms)
Layer 11 (Retrieval I/O)     → latency_monitor.record(11, ms)
Layer 13 (Evidence Merge)    → latency_monitor.record(13, ms)
Layer 14 (Candidate Scoring) → latency_monitor.record(14, ms)
Layer 16 (Fine Resolution)   → latency_monitor.record(16, ms)
```

At the end of `process_prompt`, a `TelemetryEvent::Calculation` is emitted with `layer: 0` capturing total query processing duration.

---

## 4. Core Data Structures

All types are defined in `src/types.rs`.

### Key Enumerations

| Enum | Variants | Purpose |
|------|----------|---------|
| `MemoryType` | Episodic, Core | Episodic decays; Core is permanent |
| `MemoryChannel` | Main, Intent, Reasoning | Isolated memory lanes |
| `DatabaseMaturityStage` | ColdStart, Growth, Stable | Governs pruning/promotion thresholds |
| `CandidateStatus` | Candidate, Validated, Active, Rejected | Unit lifecycle states |
| `UnitLevel` | Char, Subword, Word, Phrase, Pattern | Granularity hierarchy |
| `EdgeType` | Semantic, Parent, Child, Sequence, SourceEvidence | Link types between units |
| `SourceKind` | UserInput, Retrieval, TrainingDocument, TrainingUrl | Provenance tracking |
| `ResolverMode` | Deterministic, Balanced, Exploratory | Resolution strategy |
| `IntentKind` | Greeting, Gratitude, Farewell, Help, Clarify, Rewrite, Verify, Continue, Forget, Question, Summarize, Explain, Compare, Extract, Analyze, Plan, Act, Recommend, Classify, Translate, Debug, Critique, Brainstorm, Unknown (24 total) | Intent labels |
| `IntentFallbackMode` | None, DocumentScope, ClarifyHelp, RetrieveUnknown | Fallback behavior |
| `ToneKind` | NeutralProfessional, Empathetic, Direct, Technical, Casual, Formal | Tone inference labels |
| `ReasoningType` | General, Mathematical, Logical, Explanatory, Planning, Verification, Debugging, MultiHop | Reasoning classification |
| `ReasoningStepType` | Premise, Inference, Calculation, Verification, Hypothesis, Conclusion | Step-level classification |
| `OutputType` | FinalAnswer(String), SilentThought(String) | Visible vs internal output |
| `CalculationMethod` | MemoryLookup | Classification method (spatial query) |
| `TrainingPhaseKind` | DryRun, Bootstrap, Validation, Expansion, Lifelong | Ordered training phases |
| `TrainingSourceType` | Url, Document, Dataset, HuggingFaceDataset, StructuredJson, OpenApiSpec, CodeRepository, WikipediaDump, WikidataTruthy, OpenWebText, DbpediaDump, ProjectGutenberg, CommonCrawlWet, QaJson (14 types) | Source formats |
| `JobState` | Queued, Processing, Completed, Failed | Job lifecycle |
| `TrainingExecutionMode` | User, Development | Training context |

### Unit (Primary Memory Element)

```rust
pub struct Unit {
    pub id: Uuid,
    pub content: String,
    pub normalized: String,
    pub level: UnitLevel,                    // Char..Pattern
    pub frequency: u64,
    pub utility_score: f32,
    pub semantic_position: [f32; 3],         // 3D spatial coordinates
    pub anchor_status: bool,
    pub memory_type: MemoryType,             // Episodic or Core
    pub memory_channels: Vec<MemoryChannel>, // Default: [Main]
    pub created_at: DateTime<Utc>,
    pub last_seen_at: DateTime<Utc>,
    pub salience_score: f32,
    pub confidence: f32,
    pub trust_score: f32,                    // Default: 0.5
    pub corroboration_count: u32,
    pub links: Vec<Link>,
    pub contexts: Vec<String>,
    pub is_process_unit: bool,               // true for reasoning artifacts
}
```

### ScoreBreakdown (7-Dimensional Scoring)

```rust
pub struct ScoreBreakdown {
    pub spatial_fit: f32,      // Distance in 3D space
    pub context_fit: f32,      // Context matrix relevance
    pub sequence_fit: f32,     // Recent unit sequence alignment
    pub transition_fit: f32,   // Transition probability
    pub utility_fit: f32,      // Intrinsic utility
    pub confidence_fit: f32,   // Corroboration confidence
    pub evidence_support: f32, // External evidence backing
}
```

Default weights: spatial=0.12, context=0.18, sequence=0.16, transition=0.12, utility=0.14, confidence=0.14, evidence=0.14.

### ReasoningTrace

```rust
pub struct ReasoningTrace {
    pub steps: Vec<ReasoningStep>,
    pub reasoning_type: ReasoningType,
    pub confidence_trajectory: Vec<f32>,
    pub entities: Vec<String>,
    pub structure_hash: Option<u64>,     // For pattern matching
}
pub struct ReasoningStep {
    pub content: String,
    pub step_type: ReasoningStepType,
    pub anchor_step: bool,               // Never pruned
    pub dependencies: Vec<usize>,        // DAG references
    pub structure_hash: Option<u64>,
}
```

### ProcessResult & ExplainTrace

```rust
pub struct ProcessResult {
    pub predicted_text: String,
    pub confidence: f32,
    pub used_retrieval: bool,
    pub trace: ExplainTrace,
}
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
```

### Governance Types

```rust
pub struct GovernanceReport {
    pub pruned_units: u64,
    pub pruned_candidates: u64,
    pub purged_polluted_units: u64,
    pub purged_polluted_candidates: u64,
    pub promoted_units: u64,
    pub anchors_protected: u64,
    pub layout_adjustments: u64,
    pub mean_displacement: f32,
    pub layout_rolled_back: bool,
    pub snapshot_path: String,
    pub pruning_reasons: Vec<String>,
    pub pruned_references: Vec<PrunedUnitReference>,
    pub pollution_findings: Vec<PollutionFinding>,
}
pub struct PollutionFinding {
    pub polluted_id: Uuid,
    pub polluted_content: String,
    pub canonical_id: Uuid,
    pub canonical_content: String,
    pub overlap_ratio: f32,
    pub quality_delta: f32,
    pub reason: String,
}
```

### Channel Isolation Types

```rust
pub struct ChannelIsolationReport {
    pub is_valid: bool,
    pub main_count: usize,
    pub intent_count: usize,
    pub reasoning_count: usize,
    pub violations: Vec<ChannelIsolationViolation>,
}
pub enum IsolationViolationType {
    IntentNotInMain,
    ReasoningNotInMain,
    ExcessiveIntentReasoningOverlap,
}
```

### Classification Types

```rust
pub struct ClassificationResult {
    pub intent: IntentKind,
    pub tone: ToneKind,
    pub resolver_mode: ResolverMode,
    pub confidence: f32,
    pub method: CalculationMethod,     // Always MemoryLookup
    pub candidate_count: usize,
}
pub struct GroundTruth {
    pub intent: IntentKind,
    pub tone: ToneKind,
    pub resolver_mode: ResolverMode,
    pub domain: Option<String>,
}
```

### Training Types

```rust
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
pub struct UnifiedTrainingReport {
    pub dialogue_id: String,
    pub turns_processed: u32,
    pub units_built: u64,
    pub entities_extracted: u64,
    pub anchors_detected: u64,
    pub classification_outcome: Option<TrainingOutcome>,
    pub validation_errors: Vec<TrainingValidationError>,
    pub core_target_staged: bool,
    pub corroboration_count: u32,
    pub channels_populated: Vec<String>,
}
```

---

## 5. Configuration System

All configuration is defined in `src/config/mod.rs` and loaded from `config/config.yaml`. The top-level struct is `EngineConfig`.

### EngineConfig Fields

| Field | Config Key | Type |
|-------|-----------|------|
| `builder` | `layer_2_unit_builder` | `UnitBuilderConfig` |
| `semantic_map` | `layer_5_semantic_map` | `SemanticMapConfig` |
| `retrieval` | `layer_9_retrieval_gating` | `RetrievalThresholds` |
| `evidence_merge` | `layer_13_evidence_merge` | `EvidenceMergeConfig` |
| `scoring` | `layer_14_candidate_scoring` | `ScoringWeights` |
| `trust` | `layer_19_trust_heuristics` | `TrustConfig` |
| `intent` | `intent` | `IntentConfig` |
| `adaptive_behavior` | `adaptive_behavior` | `AdaptiveBehaviorConfig` |
| `governance` | `layer_21_memory_governance` | `GovernanceConfig` |
| `memory_budgets` | `memory_budgets` | `MemoryBudgetConfig` |
| `document` | `document` | `DocumentIngestionConfig` |
| `query` | `layer_10_query_builder` | `QueryBuilderConfig` |
| `retrieval_io` | `layer_11_retrieval` | `RetrievalIoConfig` |
| `resolver` | `layer_16_fine_resolver` | `FineResolverConfig` |
| `telemetry` | `layer_20_telemetry` | `TelemetryConfig` |
| `training_phases` | `training_phases` | `TrainingPhaseOverridesConfig` |
| `source_policies` | `source_policies` | `SourcePoliciesConfig` |
| `silent_training` | `silent_training` | `SilentTrainingConfig` |
| `huggingface_streaming` | `huggingface_streaming` | `HuggingFaceStreamingConfig` |
| `auto_inference` | `auto_inference` | `AutoInferenceConfig` |
| `gpu` | `gpu` | `GpuConfig` |
| `multi_engine` | `multi_engine` | `MultiEngineConfig` |
| `config_sweep` | `config_sweep` | `ConfigSweepConfig` |
| `classification` | `classification` | `ClassificationConfig` |

### Key Thresholds Reference

| Threshold | Config Path | Default | Purpose |
|-----------|-------------|---------|---------|
| Evidence answer confidence | `resolver.evidence_answer_confidence_threshold` | 0.22 | Minimum for evidence answers |
| Min confidence floor | `resolver.min_confidence_floor` | 0.22 | Minimum candidate score |
| Intent floor | `intent.intent_floor_threshold` | 0.40 | Minimum intent score |
| Entropy threshold | `retrieval.entropy_threshold` | 0.85 | High entropy triggers retrieval |
| Freshness threshold | `retrieval.freshness_threshold` | 0.65 | Stale context triggers retrieval |
| Retrieval decision | `retrieval.decision_threshold` | 1.1 | Combined score for retrieval |
| Min source trust | `trust.min_source_trust` | 0.35 | Minimum trust for sources |
| Prune utility | `governance.prune_utility_threshold` | 0.12 | Utility floor to avoid pruning |
| Anchor salience | `governance.anchor_salience_threshold` | 0.70 | Salience for anchor status |
| Pollution similarity | `governance.pollution_similarity_threshold` | 0.65 | Jaccard similarity gate |
| Pollution overlap | `governance.pollution_overlap_threshold` | 0.70 | Overlap ratio for pollution |
| Reasoning trigger floor | `auto_inference.reasoning_loop.trigger_confidence_floor` | 0.40 | Below this triggers reasoning |
| Reasoning exit | `auto_inference.reasoning_loop.exit_confidence_threshold` | 0.60 | Above this exits reasoning |
| Stochastic floor | `auto_inference.creative_spark.global_stochastic_floor` | 0.15 | 15% non-greedy sampling |
| Classification low confidence | `classification.low_confidence_threshold` | 0.40 | Below → ambiguous |
| Classification high confidence | `classification.high_confidence_threshold` | 0.85 | Above → Deterministic upgrade |

### Adaptive Behavior — Intent Profiles

Ten named profiles with tuned scoring weights, escape profiles, and resolver settings:

| Profile | Resolver Mode | Temperature | Stochastic Jump | Beam Width |
|---------|--------------|-------------|-----------------|------------|
| `casual` | Exploratory | 0.70 | 0.20 | 7 |
| `explanatory` | Balanced | 0.30 | 0.10 | 5 |
| `factual` | Deterministic | 0.10 | 0.05 | 3 |
| `procedural` | Balanced | 0.22 | 0.08 | 4 |
| `creative` | Exploratory | 0.75 | 0.22 | 6 |
| `brainstorm` | Exploratory | 0.90 | 0.35 | 10 |
| `plan` | Balanced | 0.35 | 0.12 | 5 |
| `act` | Balanced | 0.25 | 0.08 | 4 |
| `critique` | Balanced | 0.30 | 0.10 | 5 |
| `advisory` | Balanced | 0.26 | 0.09 | 4 |

Two trust profiles: `default` (trust=0.50, corroboration=2, no HTTPS) and `high_stakes` (trust=0.30, corroboration=4, HTTPS required).

### Source Policies

Per-format ingestion policies:

| Policy | Extraction Mode | Memory Type | Trust Bonus | Decay Days |
|--------|----------------|-------------|-------------|------------|
| `seed_training` | field_select | Core | +0.30 | 30 |
| `html` | readability | Episodic | 0.00 | 7 |
| `qa_json` | field_select | Episodic | +0.10 | 14 |
| `structured_json` | entity_extract | Core | +0.20 | — |
| `plain_text` | passthrough | Episodic | +0.10 | 30 |
| `code` | code_strip | Episodic | 0.00 | 14 |
| `wikipedia_xml` | article_text | Episodic | +0.10 | 30 |
| `wikidata_truthy` | entity_extract | Core | +0.20 | — |
| `openapi_spec` | entity_extract | Core | +0.20 | — |
| `common_crawl_wet` | readability | Episodic | 0.00 | 7 |

### Memory Budget Tiers

| Stage | Episodic | Core | Candidate Pool | Daily Growth |
|-------|----------|------|----------------|--------------|
| ColdStart | 50 MB | 10 MB | 20 MB | 10 MB |
| Growth | 100 MB | 50 MB | 30 MB | 5 MB |
| Stable | 50 MB | 100 MB | 10 MB | 1 MB |

---

## 6. Engine Pipeline

### Entry Points

1. **`process(text)`** — Main inference entry point. Handles:
   - Inline document requests (file path detection → document loading via `document.rs`)
   - Mathematical expression evaluation
   - Intent-based short-circuits (Forget clears session, Continue resumes, social intents)
   - Session document follow-up queries (`answer_question` with carry terms)
   - Personal memory lookup (for Question, Verify, Extract, Explain intents)
   - Falls through to `process_prompt()` for full pipeline

2. **`process_with_retrieval(text, retrieval_enabled)`** — Explicit retrieval override

3. **`train()` / `train_batch()` / `train_with_scope()`** — Training entry points (see Section 12)

### `process_prompt()` — Full Pipeline

```
Input: text, optional local_documents, preset_sources, allow_retrieval flag

[L1]  Normalize input → InputPacket (ingest_raw)
[L2]  Build units via rolling hash → BuildOutput                     ⏱ record(2)
[L3]  Organize hierarchy → UnitHierarchy (Char..Pattern levels)
[L4]  Ingest hierarchy into MemoryStore → active_ids
      Publish memory snapshot (lock-free via ArcSwap)

[L5]  Route active units through 3D semantic space → RoutingResult   ⏱ record(5)
      Update unit positions, connect neighbors (trust=0.35)

[L6]  Prepare context matrix and sequence state from snapshot
[L7]  Resolve intent profile via ClassificationCalculator
[L8]  Resolve adaptive runtime settings from intent profile
      Route candidate units with escape profile

      Retrieve reasoning support from Reasoning memory channel
      Merge reasoning unit IDs into candidate pool

[L14] Initial candidate scoring (GPU-accelerated)
[L9]  Assess retrieval need via IntentDetector::assess               ⏱ record(9)

      --- Conditional Retrieval Path ---
[L10] Build safe query (PII stripped, expanded)
[L11] Fetch from external sources via RetrievalPipeline              ⏱ record(11)
[L12] Validate trust of retrieved documents
[L13] Merge evidence with candidates                                 ⏱ record(13)
      --- End Conditional Retrieval ---

      OR: If local_documents provided, merge directly (no L10/L11)

[L14] Final candidate scoring (7-dimensional, GPU-accelerated)       ⏱ record(14)
      Filter out Char-level units when higher-level units exist

[L15] Select resolver mode:
      - Retrieval used OR ambiguous OR negative certainty_bias → Balanced
      - Otherwise → Deterministic

[L16] Fine resolve with intent shaping and anchor protection         ⏱ record(16)
      FineResolver::select_with_shaping uses anchor_trust_threshold
      Fallback: first candidate with Deterministic mode

[Reasoning] If auto_inference.reasoning_loop.enabled:
      execute_reasoning_loop (confidence-gated, max 3 steps)

[L17] Decode output:
      - Retrieval failed → real-time query message OR "couldn't find" message
      - Evidence available + Extract intent → list_evidence_answer / grounded_evidence_answer
      - Evidence available + other intent → grounded_evidence_answer / answer_question
      - reshape_output_for_intent applies intent-specific formatting

[L21] Update sequence state (recent unit IDs, anchors, task entities)
      Run maintenance if memory > train_memory_ceiling_kb OR every 5 turns

[L18] Generate feedback events via FeedbackController::learn
      Enqueue feedback for background worker

      Build ExplainTrace, record test observation
[L20] Emit TelemetryEvent::Calculation (layer=0, total duration)

      Return ProcessResult
```

### Intent Resolution Flow

`resolve_intent_profile()` delegates entirely to `ClassificationCalculator`:

```rust
fn resolve_intent_profile(&self, raw_input, ..) -> IntentProfile {
    let result = self.classification_calculator.calculate(
        raw_input, &memory, &spatial, &self.config.classification,
    );
    IntentProfile {
        primary: result.intent,
        confidence: result.confidence,
        ambiguous: result.confidence < config.classification.low_confidence_threshold,
        reasons: vec![format!("classification_method={:?}", result.method)],
        ..
    }
}
```

### Reasoning Support from Memory

Before candidate scoring, the engine queries the Reasoning channel for prior reasoning patterns:
- `snapshot.top_channel_matches(MemoryChannel::Reasoning, &normalized, 12)` — top 12 matches
- Takes first 8, computes average of (utility + confidence + trust) / 3
- Contributes 20% of reasoning confidence to `merged.evidence_support`
- Reasoning unit IDs are added to the candidate pool

---

## 7. Classification System

Located in `src/classification/` with four submodules: `signature`, `pattern`, `calculator`, `trainer`.

### Architecture Alignment

| Layer | Role |
|-------|------|
| L2 | `ClassificationSignature` as specialized unit feature |
| L4 | `ClassificationPattern` stored in Intent memory channel |
| L6 | Spatial query for O(log N) pattern retrieval |
| L14 | Similarity scoring and candidate aggregation |
| L18 | Feedback-driven learning and spatial adjustment |

### ClassificationSignature (`classification/signature.rs`)

A 14-float CPU-efficient feature vector computed from raw text in microseconds:

| Category | Features | Count |
|----------|----------|-------|
| Structural | `byte_length_norm`, `sentence_entropy`, `token_count_norm` | 3 |
| Punctuation | `question_mark_ratio`, `exclamation_ratio`, `period_ratio` | 3 |
| Semantic | `semantic_centroid[0..3]` (via `SemanticHasher`) | 3 |
| Derived | `urgency_score`, `formality_score`, `technical_score`, `domain_hint`, `temporal_cue` | 5 |

`SemanticHasher` produces a deterministic 3D position from text using character-level hashing — no embedding model needed.

### ClassificationPattern (`classification/pattern.rs`)

Stored in Intent memory channel as a specialized Unit:

```rust
pub struct ClassificationPattern {
    pub unit_id: Uuid,
    pub signature: ClassificationSignature,
    pub intent_kind: IntentKind,
    pub tone_kind: ToneKind,
    pub resolver_mode: ResolverMode,
    pub success_count: u64,            // L18: successful predictions
    pub failure_count: u64,            // L18: failed predictions
    pub last_reinforced: DateTime<Utc>,
    pub domain: Option<String>,
    pub memory_channels: Vec<MemoryChannel>, // Always includes Intent
}
```

### ClassificationCalculator (`classification/calculator.rs`)

Feature weights (configurable via `ClassificationConfig`):
- Structure: 0.25
- Punctuation: 0.20
- Semantic: 0.35
- Derived: 0.20

Calculation flow:
1. Compute `ClassificationSignature` from input text
2. Query `SpatialGrid` for nearby patterns within `spatial_query_radius` (0.5)
3. If no spatial matches, fall back to `MemoryStore` keyword matching
4. Compute weighted cosine similarity between input and each pattern signature
5. Aggregate votes weighted by similarity × (success / (success + failure))
6. Return `ClassificationResult` with intent, tone, resolver_mode, confidence

### ClassificationTrainer (`classification/trainer.rs`)

Iterative training from labeled seed dialogues (`LabeledDialogue` / `LabeledTurn`):
- Converts dialogues to `ClassificationPattern` instances
- Inserts patterns into spatial grid and memory store
- Runs validation iterations adjusting spatial positions
- Tracks per-iteration accuracy via `IterationReport`
- Final report via `FinalReport` with overall accuracy and pattern count

---

## 8. Memory Architecture

### Memory Store (`src/memory/store.rs`)

SQLite-backed storage managing:
- **Units** — persistent semantic elements with full metadata
- **Candidates** — observation-stage units awaiting promotion
- **Channels** — isolated memory lanes (Main, Intent, Reasoning)
- **Sequence state** — recent unit IDs, anchors, task entities, turn index

Key operations:
- `ingest_hierarchy()` — channel-aware unit ingestion from L3 output
- `ingest_hierarchy_with_channels()` — explicit channel targeting (used by reasoning)
- `update_positions()` — apply L5 spatial routing results
- `connect_units()` — create links between neighboring units
- `run_maintenance()` — L21 governance cycle (prune, promote, detect pollution)
- `audit_pollution()` — scan for duplicate/degraded units
- `snapshot()` — create immutable `MemorySnapshot` for lock-free reads

### Memory Types

| Type | Behavior |
|------|----------|
| `Episodic` | Decays after `episodic_decay_days` (30); pruned by governance |
| `Core` | Permanent; requires corroboration for promotion from Episodic |

### Memory Channels

| Channel | Purpose | Isolation Rules |
|---------|---------|-----------------|
| `Main` | Primary content storage | All user content defaults here |
| `Intent` | Classification patterns | Blocked from Core promotion (`intent_channel_core_promotion_blocked: true`) |
| `Reasoning` | Internal reasoning thoughts | Process units with `is_process_unit: true` |

### Memory Snapshot (Lock-Free Read Path)

```
Write path: memory.lock() → modify → publish_memory_snapshot() → ArcSwap::store()
Read path:  memory_snapshot.load_full() → Arc<MemorySnapshot> (no lock)
```

Provides: `all_units()`, `get_units()`, `top_units()`, `sequence_state()`, `memory_summary()`, `top_channel_matches()`

### Candidate Lifecycle

```
Observation → Candidate → Validated → Active → (pruned or promoted to Core)
                                    → Rejected
```

Maturity-stage-dependent thresholds:

| Stage | Unit Threshold | Observation Threshold | Discovery Frequency | Discovery Utility |
|-------|---------------|----------------------|--------------------|--------------------|
| ColdStart | < 1,000 | 2 | 1 | 0.18 |
| Growth | 1,000–10,000 | 3 | 2 | 0.28 |
| Stable | > 10,000 | 4 | 3 | 0.42 |

### Pollution Prevention (Pre-Ingestion Gate)

Content is validated **before** entering memory via `passes_content_validation()` in `ingest_activation()`:

| Gate | Check | Action |
|------|-------|--------|
| Gate 1 | Empty or whitespace-only content | Reject |
| Gate 2 | Purely numeric/punctuation content | Reject |
| Gate 3 | Excessive word repetition (uniqueness < 30%) | Reject |
| Gate 4 | Retrieval-sourced: short fragments, HTML/script artifacts | Reject |
| Gate 5 | Near-duplicate with dramatically lower quality than existing | Reject |

This prevents pollution at the source rather than cleaning up post-hoc.

### Pollution Detection (Post-Hoc)

Runs during `run_maintenance()` as a safety net:

1. Normalize content and compute overlap ratio between unit pairs
2. Apply Jaccard similarity gate (`pollution_similarity_threshold`: 0.65)
3. Compare quality (utility × confidence × trust) — lower quality = pollutant
4. Apply `pollution_penalty_factor` (0.25) or purge
5. Min content length for detection: `pollution_min_length` (4 chars)

### Anchor System

Units earn anchor status when:
- `frequency >= anchor_reuse_threshold` (3)
- `salience_score >= anchor_salience_threshold` (0.70)

Anchors are protected from pruning for `anchor_protection_grace_days` (14 days).

### Bloom Filter (`src/bloom_filter.rs`)

`UnitBloomFilter` for O(1) probabilistic membership testing:
- 3 hash functions, 10× expected items bit count (minimum 1024 bits)
- Tracks `BloomStats`: queries, maybe_hits, false_positives
- Used for fast duplicate detection during unit ingestion

### Dynamic Memory Allocator (`src/memory/dynamic.rs`)

On-demand allocation for reasoning buffers:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `base_memory_limit_mb` | 350 | Idle memory limit |
| `max_memory_limit_mb` | 550 | Limit during reasoning |
| `thought_buffer_size_kb` | 64 | Per reasoning step |

- `DynamicMemoryAllocator` uses atomic counters for lock-free tracking
- `ReasoningGuard` — RAII guard that allocates on creation, releases on drop
- `ThoughtBuffer` — capacity-limited buffer; rejects additions when full

---

## 9. Dynamic Reasoning

### Trigger Mechanism

Reasoning is **not a user toggle** — triggered automatically when:
1. `reasoning_loop.enabled` is true (default: true)
2. Initial confidence from L16 is below `trigger_confidence_floor` (0.40)
3. `IntentDetector::should_trigger_reasoning(intent, config)` returns true

### Execution Loop

```
for step in 0..max_internal_steps (default 3):
    1. Try adapt_reasoning_pattern() from Reasoning channel
       → Top 5 patterns by similarity, best match adapted
       → Anchor boost: +0.15; non-anchor: +0.10; plus similarity × 0.2
    2. Fallback: generate_thought_unit()
       → Confidence improvement: 0.1 × (1.0 - previous).min(0.3)
    3. If step > 0 and confidence < exit_threshold × 0.7:
       → Set needs_retrieval = true (signals main pipeline to trigger web retrieval)
    4. Ingest thought into Reasoning channel as Episodic unit
       → MemoryType::Episodic, channels: [Reasoning]
       → NOT Core memory (prevents pollution)
    5. Track confidence trajectory
    6. Exit early if confidence >= exit_confidence_threshold (0.60)
```

### Reasoning-Triggered Retrieval

When the reasoning loop determines it cannot reach sufficient confidence from internal knowledge alone, it sets `needs_retrieval = true` on `ReasoningState`. The main pipeline checks this flag after the reasoning loop exits and automatically triggers web retrieval via the Layer 11 pipeline. This implements the "I don't know → let me check → here's the answer" pattern without requiring any manual retrieval configuration.

**Key**: Web retrieval is always enabled (`enable_retrieval` defaults to `true`). The `enable_retrieval` config gate has been removed from the main retrieval path — retrieval is triggered automatically based on `IntentDetector::assess()` and reasoning confidence signals.

### Output

```rust
ReasoningResult {
    output: OutputType::FinalAnswer(String),
    steps_taken: usize,
    final_confidence: f32,
    reasoning_triggered: bool,
    thoughts: Vec<ThoughtUnit>,
    needs_retrieval: bool,       // true if reasoning flagged retrieval need
}
```

`ThoughtUnit` contains: `content`, `step` index, `confidence`, `created_at`.

---

## 10. GPU Acceleration

Located in `src/gpu/`, entirely behind `#[cfg(feature = "gpu")]`.

### Technology

- **Backend**: `wgpu` (cross-platform: macOS Metal, Linux Vulkan, Windows DX12, WebAssembly)
- **Initialization**: Lazy global device via `once_cell::sync::Lazy`
- **Fallback**: Transparent CPU path when feature disabled or GPU unavailable

### GPU Operations

| Component | File | Shader(s) |
|-----------|------|-----------|
| `GpuCandidateScorer` | `compute/candidate_scorer.rs` | `candidate_scoring.wgsl` |
| `GpuForceLayout` | `compute/force_layout.rs` | `force_attractive.wgsl`, `force_repulsive.wgsl` |
| `GpuDistanceCalculator` | `compute/distance.rs` | `distance.wgsl` |
| GPU Evidence Merge | `compute/evidence_merge.rs` | `evidence_merge.wgsl` |
| GPU Classification | `compute/classification.rs` | `classification_similarity.wgsl`, `classification_aggregation.wgsl` |
| GPU Intent Scorer | `compute/intent_scorer.rs` | `intent_scorer.wgsl` |
| GPU Tone Detector | `compute/tone_detector.rs` | `tone_detector.wgsl` |

Minimum thresholds: `min_candidates_for_gpu`: 256, `min_units_for_gpu_layout`: 100.

### Configuration (`GpuConfig`)

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | true | Enable GPU acceleration |
| `force_cpu` | false | Override to CPU-only |
| `power_preference` | "high" | GPU power preference |
| `min_memory_mb` | 512 | Minimum GPU memory required |
| `batch_size` | 1024 | Batch size for GPU ops |
| `timeout_ms` | 5000 | GPU operation timeout |
| `use_for_scoring` | true | GPU candidate scoring |
| `use_for_layout` | true | GPU force layout |
| `use_for_distance` | true | GPU distance calc |

### CPU Fallback

```rust
#[cfg(not(feature = "gpu"))]
pub fn is_gpu_available() -> bool { false }
```

`score_candidates_gpu_accelerated()` in `layers/search.rs` automatically uses CPU when GPU is unavailable.

---

## 11. Telemetry & Observability

Located in `src/telemetry/` with five submodules.

### Module Map

| Module | Exports | Purpose |
|--------|---------|---------|
| `worker.rs` | `TelemetryWorker`, `TelemetryEvent`, `SqliteHotStore`, `AppendOnlyLog` | Async background event emission |
| `hot_store.rs` | `HotStore`, `HotStoreConfig`, `ReasoningStepRecord` | Dedicated SQLite store for real-time UI queries |
| `latency.rs` | `LatencyMonitor`, `LatencyTimer`, `LayerLatencyMetrics` | Per-layer p50/p95/p99 sliding window |
| `trace.rs` | `TraceContext`, `SessionId`, `TraceId` | Session/trace ID management |
| `test_observer.rs` | `TestObserver` | Structured observation during tests |

### TelemetryEvent Variants

```rust
pub enum TelemetryEvent {
    Calculation { layer: u8, operation: String, duration_ms: u64, session_id, trace_id },
    DbPush { unit_id: Uuid, memory_type: String, session_id, trace_id },
    Retrieval { source: String, results: usize, session_id, trace_id },
    MorphAction { action: String, before: String, after: String, session_id, trace_id },
    IntentLabel { label: String, channel: String, score: f32, session_id, trace_id },
    ReasoningStep { step: usize, thought: String, confidence: f32, session_id, trace_id },
    LatencySpike { layer: u8, latency_ms: u64, threshold_ms: u64, session_id, trace_id },
    MemoryAllocation { allocated_kb: usize, total_kb: usize, limit_kb: usize, session_id, trace_id },
    ProcessAnchorProtected { unit_id: Uuid, structure_hash: u64, utility_score: f32, session_id, trace_id },
}
```

### TelemetryWorker

- **Architecture**: Background thread with `SyncSender` channel (capacity: `channel_capacity`, default 10000)
- **Hot store**: `SqliteHotStore` — WAL-mode SQLite for real-time queries, indexed by session_id/trace_id/timestamp
- **Cold store**: `AppendOnlyLog` — append-only file for long-term archival
- **Batching**: Events buffered in batches of `batch_size` (100) or flushed every `flush_interval_ms` (100ms)
- **Sampling**: `sample_rate` (1.0) for high-frequency event filtering

### HotStore

- SQLite with WAL mode, `PRAGMA synchronous = NORMAL`, `cache_size = -64000`
- `max_events`: 100,000 (older events pruned every `prune_interval_secs`: 300s)
- Stores `ReasoningStepRecord` for reasoning step queries

### LatencyMonitor

- Per-layer sliding window (size: `window_size`, default 1000)
- Calculates p50, p95, p99 percentiles
- Alert threshold: `alert_threshold_ms` (200ms) — emits `LatencySpike` event when exceeded
- `LatencyTimer` — RAII guard that records duration on drop

### TraceContext

- `SessionId` — persists across a user session
- `TraceId` — per-query identifier, rotated via `start_new_trace()`
- Both are `Uuid`-based

---

## 12. Training Pipeline & Seed Generation

### Training Pipeline (`src/training.rs`)

#### Training Phases

| Phase | Purpose | Memory Delta | Growth Limit |
|-------|---------|-------------|--------------|
| `DryRun` | Validate pipeline end-to-end | 5.0 MB | 50 MB/day |
| `Bootstrap` | Initial knowledge seeding | 5.0 MB | 50 MB/day |
| `Validation` | Quality gate verification | 2.0 MB | 10 MB/day |
| `Expansion` | Broad knowledge ingestion | 0.5 MB | 1 MB/day |
| `Lifelong` | Continuous learning | 0.5 MB | 1 MB/day |

Expansion phase requires: `min_unit_discovery_efficiency ≥ 0.90`, `min_semantic_routing_accuracy ≥ 0.85`.

#### Training Plan

```rust
pub struct TrainingPlan {
    pub phases: Vec<TrainingPhasePlan>,
}
pub struct TrainingPhasePlan {
    pub phase: TrainingPhaseKind,
    pub batches_target: usize,
    pub sources: Vec<TrainingSource>,
    pub options: TrainingOptions,
    pub min_unit_discovery_efficiency: Option<f32>,
    pub min_semantic_routing_accuracy: Option<f32>,
}
```

Plans are built via `build_training_plan_with_config()` which reads `EngineConfig` to determine scope (Full, DryRun) and execution mode (User, Development).

#### Training Methods

| Method | Purpose |
|--------|---------|
| `train()` | Full training, User execution mode |
| `train_with_scope(mode, scope)` | Explicit scope (Full/DryRun) |
| `train_batch(request)` | Single-batch training from API/test harness |
| `start_train()` | Returns job_id for async training (spawn in API handler) |
| `run_phase0_dry_run()` | Validates pipeline + measures inference latency |

#### DryRun Report

`run_phase0_dry_run()` performs:
1. Run training with `TrainingScope::DryRun`
2. Warm inference path with probe query ("What does HIIT stand for?")
3. Measure inference latency per token (byte-span equivalent, not whitespace)
4. Check layout stability (no rollback warnings)

```rust
pub struct DryRunReport {
    pub status: TrainingJobStatus,
    pub snapshot_path: String,
    pub snapshot_readable: bool,
    pub map_stable: bool,
    pub inference_ok: bool,               // non-empty + < 200ms/token
    pub inference_latency_ms: u128,
    pub latency_per_token_ms: u128,
    pub query_result: String,
    pub memory_summary: String,
}
```

#### Parallel Training

```yaml
silent_training:
  parallel:
    enabled: true
    worker_count: 0                      # Auto-detect CPU cores
    queue_capacity: 64
    commit_batch_size: 8
    commit_flush_interval_ms: 250
    total_memory_limit_mb: 2560.0
    non_worker_memory_reserve_mb: 1280.0
    local_shard_soft_limit_mb: 96.0
    local_shard_hard_limit_mb: 128.0
```

#### HuggingFace Streaming

Adaptive batch sizing for streaming from HuggingFace datasets:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `initial_rows_per_pull` | 100 | Starting batch size |
| `min_rows_per_pull` | 100 | Floor |
| `max_rows_per_pull` | 300 | Ceiling |
| `fast_pull_threshold_ms` | 1,200 | Increase batch if faster |
| `slow_pull_threshold_ms` | 3,500 | Decrease batch if slower |
| `max_retries` | 6 | Retry count with exponential backoff |
| `retry_max_delay_ms` | 15,000 | Maximum retry delay |

### Seed Dataset Generation (`src/seed/`)

#### Unified Training Example Format

```rust
pub struct TrainingExample {
    pub question: String,
    pub answer: String,
    pub context: Option<String>,
    pub reasoning: Option<ReasoningTrace>,
    pub intent: Option<String>,
    pub entities: Vec<String>,
    pub channels: Vec<MemoryChannel>,
    pub curriculum: CurriculumMetadata,
    pub quality_gates: QualityGates,
}
```

#### Generators

| Generator | File | Purpose |
|-----------|------|---------|
| `EntityGenerator` | `seed/entity_generator.rs` | Produces entity-based QA datasets |
| `DialogueGenerator` | `seed/dialogue_generator.rs` | Produces multi-turn dialogue datasets |
| `generate_dryrun_datasets()` | `seed/dryrun.rs` | Generates DryRun validation datasets |
| `generate_intelligence_seeds()` | `seed/intelligence_generator.rs` | Intelligence-focused examples: reasoning chains, retrieval triggers, confidence gating |

#### Intelligence-Focused Seed Generation

The `intelligence_generator` module produces training examples that teach the engine *how to think*, not *what to know*. Five categories:

| Category | Count | Purpose |
|----------|-------|---------|
| Reasoning chains | 6+ | Step-by-step mathematical, logical, debugging, planning derivations |
| Retrieval triggers | 6+ | "I don't know → let me check → here's the answer" pattern |
| Confidence gating | 6+ | Recognizing high vs. low internal confidence; social short-circuits |
| Multi-hop reasoning | 3+ | Chaining multiple logical/arithmetic steps |
| Self-correction | 2+ | Catching and correcting initial reasoning errors |

Each example includes:
- `ReasoningTrace` with typed steps (`Premise`, `Inference`, `Calculation`, `Verification`, `Hypothesis`, `Conclusion`)
- `IntentKind` classification
- `context` hint (e.g., `retrieval_trigger:population_lookup`)
- `curriculum_score` for training ordering (higher = earlier)

### Internal Dataset Catalog (`src/open_sources.rs`)

**No external open-source datasets are used for training or seeding.** The engine acquires factual knowledge at runtime via web retrieval triggered during reasoning.

Internal-only datasets:

| Source | Category | License | Default Memory | Integration |
|--------|----------|---------|----------------|-------------|
| `dryrun_intent_core` | intent_dialogue | Internal | Episodic | structured_json |
| `dryrun_entity_seed` | core_kb | Internal | Core | structured_json |

Each source defines: `id`, `label`, `category`, `license`, `default_type`, `default_memory`, `default_batch_size`, `default_chunk_char_limit`.

### Config Sweep Benchmark Tool (`src/bin/config_sweep.rs`)

Sweeps key configuration values across ranges, runs probe queries against each configuration, and reports optimal settings. Dimensions swept:

- `evidence_answer_confidence_threshold`, `min_confidence_floor`, `intent_floor_threshold`
- `entropy_threshold`, `freshness_threshold`, `selection_temperature`
- `prune_utility_threshold_stable`, `reasoning_trigger_confidence_floor`, `reasoning_exit_confidence`

Output: `benchmarks/config_sweep_report.md`

### Pollution Control Sweep Tool (`src/bin/pollution_sweep.rs`)

Sweeps pollution-related configuration values, ingests a mix of valid and intentionally polluted content, then audits for contamination. Dimensions swept:

- `pollution_detection_enabled`, `pollution_min_length`, `pollution_similarity_threshold`
- `pollution_overlap_threshold`, `pollution_quality_margin`, `pollution_penalty_factor`, `pollution_edge_trim_limit`

Output: `benchmarks/pollution_sweep_report.md`

---

## 13. API Specifications

Located in `src/api.rs` and `src/api/openai_compat.rs`.

### REST API Routes

| Method | Path | Handler | Purpose |
|--------|------|---------|---------|
| POST | `/api/v1/train/batch` | `train_batch` | Submit batch training job |
| GET | `/api/v1/train/status/:job_id` | `training_status` | Poll training job status |
| GET | `/api/v1/status` | `auto_mode_status` | Engine status + auto-mode indicator |
| POST | `/v1/chat/completions` | `openai_compat::chat_completions` | OpenAI-compatible chat endpoint |
| GET | `/v1/models` | `openai_compat::list_models` | OpenAI-compatible model listing |

### Content Negotiation

- **JSON** (default): `application/json`
- **Protobuf**: `application/x-protobuf` — uses `prost::Message` for `proto/api.proto` definitions

### ApiState

```rust
pub struct ApiState {
    pub engine: Arc<Engine>,
    pub auto_mode_config: AutoModeConfig,
}
```

### Training Request Flow

1. Client POSTs to `/api/v1/train/batch` with `TrainRequest` (JSON or Protobuf)
2. Only `mode: "silent"` is accepted; other modes return 400
3. Request is converted to `TrainBatchRequest` with sources and options
4. Engine spawns async training, returns `AcceptedJob { job_id }`
5. Client polls `/api/v1/train/status/:job_id` for `TrainingJobStatus`

### OpenAI Compatibility (`api/openai_compat.rs`)

- `/v1/chat/completions` — accepts OpenAI-format messages, routes through `engine.process()`
- `/v1/models` — lists available model identifiers
- Auto-mode enforcement: `ignore_mode_parameter`, `ignore_temperature_parameter`, etc. per `AutoModeConfig`

---

## 14. Priority Scheduler

Located in `src/scheduler.rs`.

### Work Priorities

```rust
pub enum WorkPriority {
    Inference,            // Highest — user-facing queries
    InteractiveTraining,  // User-initiated training
    SilentBatch,          // Background batch training
    Maintenance,          // Lowest — governance, cleanup
}
```

### PriorityScheduler

- **Mechanism**: `Mutex<SchedulerState>` + `Condvar` for blocking acquisition
- **Preemption**: Higher-priority work blocks lower-priority acquisition
- **Starvation prevention**: Queued tasks wait only while higher-priority tasks are active or queued
- **Tracking**: Per-priority queued + active counts via `QueueDepths`

```rust
pub struct QueueDepths {
    pub inference: usize,
    pub interactive_training: usize,
    pub silent_batch: usize,
    pub maintenance: usize,
}
```

### WorkPermit

RAII guard — decrements active count and wakes waiting threads on drop.

---

## 15. Security & Trust

### Layer 12/19: Trust Assessment (`layers/safety.rs`)

Trust scoring formula per source:
```
trust = default_source_trust (0.50)
      + https_bonus (0.10)          if HTTPS
      + allowlist_bonus (0.10)      if domain in allowlist
      - parser_warning_penalty (0.20) if parser warnings
      + corroboration_bonus (0.08)  per corroborating source
      + format_trust_adjustments    per detected format
```

Format trust adjustments:
- `html_raw`: -0.30
- `qa_json_schema_keys`: -0.40
- `structured_entity`: +0.20
- `code_syntax`: -0.10

### Content Quality Thresholds

| Threshold | Default | Purpose |
|-----------|---------|---------|
| `min_readability_score` | 0.60 | Minimum readability |
| `max_boilerplate_ratio` | 0.40 | Maximum boilerplate content |
| `min_unique_words_ratio` | 0.30 | Minimum vocabulary diversity |

### Promotion Rules

- **Block patterns**: JSON schema keys (`question:`, `answer:`), bare punctuation
- **Allow patterns**: Natural language text ≥ 20 chars with alphabetic content

### PII Stripping (Layer 10)

`SafeQueryBuilder::build()` in `layers/query.rs` strips PII before external queries.
Aggressiveness level: configurable via `query.pii_stripping_aggressiveness` (default: "standard").

### Allowlist Domains

Default trusted domains: wikimedia.org, wikipedia.org, wikidata.org, archive.org, ncbi.nlm.nih.gov, pmc.ncbi.nlm.nih.gov, nominatim.openstreetmap.org, openstreetmap.org, dbpedia.org, gutenberg.org.

---

## 16. Directory Structure

```
spse_engine/
├── src/
│   ├── main.rs                    # CLI entry point
│   ├── lib.rs                     # Library exports (24 public modules)
│   ├── engine.rs                  # Core Engine struct + 21-layer pipeline
│   ├── types.rs                   # All core type definitions (~1278 lines)
│   ├── config/
│   │   └── mod.rs                 # EngineConfig + all sub-configs (~2758 lines)
│   ├── layers/
│   │   ├── mod.rs                 # Layer module exports
│   │   ├── input.rs               # L1: Input normalization
│   │   ├── builder.rs             # L2: Rolling hash unit discovery
│   │   ├── hierarchy.rs           # L3: Level grouping
│   │   ├── router.rs              # L5: 3D spatial routing
│   │   ├── context.rs             # L6: Context matrix
│   │   ├── intent.rs              # L7/L9: Intent detection + retrieval gating
│   │   ├── query.rs               # L10: Safe query building
│   │   ├── retrieval.rs           # L11: External retrieval
│   │   ├── safety.rs              # L12/L19: Trust validation
│   │   ├── merge.rs               # L13: Evidence merging
│   │   ├── search.rs              # L14: 7D candidate scoring
│   │   ├── resolver.rs            # L16: Fine resolution + shaping
│   │   ├── output.rs              # L17: Output decoding
│   │   └── feedback.rs            # L18: Learning events
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── store.rs               # L4/L21: MemoryStore + governance
│   │   └── dynamic.rs             # DynamicMemoryAllocator + ThoughtBuffer
│   ├── telemetry/
│   │   ├── mod.rs
│   │   ├── worker.rs              # TelemetryWorker + event types
│   │   ├── hot_store.rs           # HotStore (SQLite WAL)
│   │   ├── latency.rs             # LatencyMonitor (p50/p95/p99)
│   │   ├── trace.rs               # TraceContext (SessionId/TraceId)
│   │   └── test_observer.rs       # TestObserver
│   ├── classification/
│   │   ├── mod.rs
│   │   ├── signature.rs           # 14-float feature vector
│   │   ├── pattern.rs             # ClassificationPattern (Intent channel)
│   │   ├── calculator.rs          # Weighted vote aggregation
│   │   └── trainer.rs             # Iterative training from seed data
│   ├── gpu/
│   │   ├── mod.rs                 # Feature-gated GPU module
│   │   ├── device.rs              # GpuDevice initialization
│   │   ├── compute/
│   │   │   ├── mod.rs
│   │   │   ├── candidate_scorer.rs
│   │   │   ├── classification.rs
│   │   │   ├── distance.rs
│   │   │   ├── evidence_merge.rs
│   │   │   ├── force_layout.rs
│   │   │   ├── intent_scorer.rs
│   │   │   └── tone_detector.rs
│   │   └── shaders/               # WGSL compute shaders
│   │       ├── candidate_scoring.wgsl
│   │       ├── classification_aggregation.wgsl
│   │       ├── classification_similarity.wgsl
│   │       ├── distance.wgsl
│   │       ├── evidence_merge.wgsl
│   │       ├── force_attractive.wgsl
│   │       ├── force_repulsive.wgsl
│   │       ├── intent_scorer.wgsl
│   │       └── tone_detector.wgsl
│   ├── common/
│   │   ├── mod.rs
│   │   ├── scoring.rs             # ScoreUtils
│   │   ├── matching.rs            # KeywordMatcher
│   │   ├── selection.rs           # TopKSelector
│   │   ├── similarity.rs          # SimilarityUtils
│   │   └── dedup.rs               # DedupUtils
│   ├── api/
│   │   └── openai_compat.rs       # OpenAI-compatible endpoints
│   ├── seed/
│   │   ├── mod.rs                 # TrainingExample + CurriculumMetadata
│   │   ├── entity_generator.rs    # Entity-based QA datasets
│   │   ├── dialogue_generator.rs  # Multi-turn dialogue datasets
│   │   ├── intelligence_generator.rs # Intelligence-focused seeds (reasoning, retrieval triggers)
│   │   └── dryrun.rs              # DryRun validation datasets
│   ├── bin/
│   │   ├── benchmark_eval.rs      # Benchmark evaluation
│   │   ├── config_sweep.rs        # Config sweep benchmark tool
│   │   ├── crash_drill.rs         # Crash resilience drills
│   │   ├── drill_harness.rs       # Drill execution harness
│   │   ├── pollution_dev.rs       # Pollution detection development tool
│   │   ├── pollution_dev_lib.rs   # Pollution dev support library
│   │   ├── pollution_sweep.rs     # Pollution config sweep tool
│   │   ├── stress_drill.rs        # Stress testing drills
│   │   ├── test_harness.rs        # Config sweep harness
│   │   └── zero_shot_harness.rs   # Zero-shot scenario testing
│   ├── api.rs                     # REST API router (axum)
│   ├── bloom_filter.rs            # UnitBloomFilter
│   ├── document.rs                # Document processing (PDF, DOCX, text)
│   ├── open_sources.rs            # Internal dataset catalog (no external open sources)
│   ├── persistence.rs             # SQLite persistence layer
│   ├── scheduler.rs               # PriorityScheduler (4 priorities)
│   ├── spatial_index.rs           # SpatialGrid for O(log N) queries
│   ├── region_index.rs            # Regional spatial index
│   ├── training.rs                # Training pipeline orchestration
│   ├── drill_lib.rs               # Drill framework
│   ├── stress_drill_lib.rs        # Stress testing drills
│   └── crash_drill_lib.rs         # Crash resilience drills
├── config/
│   ├── config.yaml                # Main configuration file
│   ├── profiles.json              # Profile manifest
│   ├── layer_20_schema.json       # L20 telemetry JSON schema
│   └── profiles/                  # Profile overrides (12 YAML files)
│       ├── balanced.yaml
│       ├── deterministic.yaml
│       ├── confidence_heavy.yaml
│       └── ...
├── proto/
│   └── api.proto                  # Protobuf API definitions
├── tests/
│   ├── integration.rs             # Integration tests
│   ├── config_harness_assets_test.rs
│   ├── config_sweep_test.rs
│   ├── gpu_fallback_test.rs
│   └── layer_boundary_tests.rs
├── test_data/
│   ├── large_corpus/              # Test corpus files (14 text files)
│   ├── controlled_story_dataset.json
│   └── zero_shot_scenarios.json
├── scripts/
│   ├── analyze_results.py
│   ├── analyze_observations.py
│   ├── audit_pollution.py
│   ├── drill_corpus_generator.py
│   ├── generate_large_corpus.py
│   ├── test_data_generator.py
│   └── zero_shot_harness.py
├── docs/
│   ├── SPSE_ARCHITECTURE_V11.1.md # This document
│   ├── CONFIG_TUNING_GUIDE.md
│   ├── DATASET_GENERATION_GUIDE.md
│   ├── PRE_PRODUCTION_EXECUTION_PLAN.md
│   ├── PRE_PRODUCTION_TRAINING_PLAN.md
│   └── finalized_architecture_documentation_revised_v11.docx
├── web-ui/                        # Next.js web UI
│   ├── app/
│   ├── components/
│   ├── package.json
│   └── next.config.js
├── Cargo.toml
├── Cargo.lock
├── build.rs
├── AGENTS.md                      # Architecture compliance guide
└── README.md

```

---

## 17. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| **Unit** | Atomic semantic element in memory — the fundamental building block |
| **Anchor** | High-salience unit protected from pruning; factual reference point |
| **Candidate** | Observation-stage unit that may be promoted to active |
| **Channel** | Isolated memory lane (Main, Intent, Reasoning) |
| **Episodic** | Memory type that decays over time |
| **Core** | Permanent memory type requiring corroboration for promotion |
| **Escape** | Stochastic jump in semantic routing to explore non-obvious candidates |
| **Governance** | L21 maintenance cycle: pruning, promotion, pollution detection |
| **Pollution** | Duplicate or degraded units that lower memory quality |
| **Shaping** | Intent-specific output formatting and anchor protection |
| **StyleAnchor** | Exemplar text used to bias vocabulary selection |
| **TTFT** | Time To First Token |
| **Mode C** | Unified Auto+Reasoning architecture with dynamic switching |
| **SPS** | Semantic Processing System — the tokenizer-free approach |
| **ThoughtUnit** | Internal reasoning artifact stored in Reasoning channel |
| **DryRun** | Phase 0 validation pipeline that tests end-to-end readiness |

### Appendix B: References

- `AGENTS.md` — Architecture compliance guide and coding standards
- `config/config.yaml` — Main configuration file
- `README.md` — Project overview
- `docs/CONFIG_TUNING_GUIDE.md` — Configuration tuning guide
- `docs/DATASET_GENERATION_GUIDE.md` — Dataset generation specifications
- `docs/PRE_PRODUCTION_EXECUTION_PLAN.md` — Implementation roadmap
- `docs/PRE_PRODUCTION_TRAINING_PLAN.md` — Training plan