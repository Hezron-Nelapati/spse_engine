# SPSE Engine Pre-Production Execution Plan

**Document Version:** 1.0  
**Created:** March 14, 2026  
**Priority Order:** Creative > Drill Related > Core Logic/Tests > Web UI

---

## Phase 1: Creative Mode & Hybrid Intent Profiles

**Estimated Duration:** 2-3 weeks  
**Dependencies:** None (foundational for other phases)

### 1.1 Intent-Specific Config Profiles for Layers 14-17

**What to Implement:**
- Extend existing `adaptive_behavior.intent_profiles` in config to support creative mode with stochastic jumps and lower confidence floors
- Create new profile variants for `brainstorm`, `act`, `plan`, `critique` intent kinds
- Wire profile selection logic into resolver pipeline

**How to Implement:**

1. **Update Config Schema** (`src/config/mod.rs`):
   - Add `IntentShapingConfig` struct with `plan`, `act`, `brainstorm`, `critique` shaping parameters
   - Add `creative_drift_tolerance` and `factual_corruption_threshold` fields to `FineResolverConfig`

2. **Extend Config YAML** (`config/config.yaml`):
   ```yaml
   adaptive_behavior:
     intent_profiles:
       # Existing profiles...
       brainstorm:
         scoring:
           w_spatial: 0.06
           w_context: 0.30
           w_sequence: 0.12
           w_transition: 0.06
           w_utility: 0.22
           w_confidence: 0.06
           w_evidence: 0.04
         escape:
           stochastic_jump_prob: 0.35
           beam_width: 10
         resolver:
           selection_temperature: 0.90
           min_confidence_floor: 0.12
           mode: "exploratory"
         shaping:
           allow_semantic_drift: true
           drift_tolerance: 0.25
           preserve_factual_anchor: true
   ```

3. **Modify Resolver** (`src/layers/resolver.rs`):
   - Add `apply_intent_shaping()` method that adjusts candidate selection based on intent profile
   - Implement temperature-based stochastic selection for creative mode
   - Add factual anchor preservation check (units with `trust_score > 0.8` are protected)

4. **Update Engine** (`src/engine.rs`):
   - Wire intent profile lookup into `resolve_candidates()` flow
   - Pass active profile to `FineResolver::resolve()`

**Files to Modify:**
- `src/config/mod.rs` (lines ~1100-1200 for adaptive behavior)
- `config/config.yaml` (lines 131-245)
- `src/layers/resolver.rs` (full file)
- `src/engine.rs` (resolver integration section)

**Verification:**
```bash
cargo test --lib resolver::tests
cargo run --bin test_harness -- --profile creative --intent brainstorm
```

---

### 1.2 Hybrid Intent Score Validation

**What to Implement:**
- Validate runtime intent blending from heuristic classification + Intent-channel memory lookup
- Ensure creative outputs show semantic drift without factual corruption
- Verify Layer 17 split realization (Decoder + Engine-level shaping)

**How to Implement:**

1. **Create Intent Blend Validator** (`src/layers/intent.rs`):
   - Add `validate_hybrid_blend()` function that:
     - Computes heuristic score from `IntentDetector::classify()`
     - Retrieves memory-backed score from `intent_scores_from_memory()`
     - Calculates blended score: `0.6 * heuristic + 0.4 * memory_backed`
     - Returns `IntentBlendReport` with breakdown

2. **Add Telemetry Fields** (`src/telemetry/test_observer.rs`):
   ```rust
   pub struct IntentBlendReport {
       pub heuristic_score: f32,
       pub memory_backed_score: f32,
       pub blended_score: f32,
       pub memory_channel: MemoryChannel,
       pub intent_label: String,
   }
   ```
   - Add to `TestObservation` struct

3. **Implement Drift Detection** (`src/layers/output.rs`):
   - Add `detect_semantic_drift()` comparing output to source candidates
   - Add `detect_factual_corruption()` checking against anchor units
   - Return warnings if drift exceeds tolerance or corruption detected

4. **Wire Shaping Logic** (`src/engine.rs`):
   - In `build_explain_trace()`, add intent blend breakdown to debug steps
   - Apply `plan`, `act`, `brainstorm`, `critique` shaping based on `IntentKind`

**Files to Modify:**
- `src/layers/intent.rs` (add validation)
- `src/telemetry/test_observer.rs` (add fields)
- `src/layers/output.rs` (drift detection)
- `src/engine.rs` (wire shaping)

**Verification:**
```bash
cargo test --lib intent::tests::hybrid_blend
cargo test --lib output::tests::drift_detection
```

---

### 1.3 MemoryChannel::Intent Gate Validation

**What to Implement:**
- Verify `tag_intent=true` flag correctly gates promotion to durable intent memory
- Ensure noisy intent signals do not leak into Core Memory stability layer

**How to Implement:**

1. **Add Channel Isolation Tests** (`tests/integration.rs`):
   ```rust
   #[test]
   fn intent_channel_isolation_prevents_core_pollution() {
       // Train with intent_dialogue source
       // Verify units routed to Intent channel
       // Assert Core memory unaffected by noisy intent signals
   }
   ```

2. **Update Memory Store** (`src/memory/store.rs`):
   - Add `channel_isolation_check()` in `ingest_hierarchy_with_channels_report()`
   - Block promotion from `MemoryChannel::Intent` to `MemoryType::Core` without explicit `merge_to_core=true`

3. **Add Config Field** (`src/config/mod.rs`):
   ```rust
   pub struct GovernanceConfig {
       // Existing fields...
       pub intent_channel_core_promotion_blocked: bool,  // default: true
   }
   ```

**Files to Modify:**
- `tests/integration.rs` (new tests)
- `src/memory/store.rs` (isolation check)
- `src/config/mod.rs` (new field)
- `config/config.yaml` (add field under `layer_21_memory_governance`)

---

## Phase 2: Drill Suite & Pollution Integration

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 1 complete

### 2.1 Layer-Specific Drill Suite (100MB Corpus)

**What to Implement:**
- Create targeted corpus subsets to trigger specific failure modes
- Move existing pollution check logic from `pollution_dev.rs` into unified drill framework
- Test ALL layers with isolated failure scenarios, happy paths, and edge cases
- Include intent-driven input validation (L7, L9) and intent-driven output validation (L17)

**How to Implement:**

1. **Create Drill Framework** (`src/bin/drill_harness.rs`):
   ```rust
   enum DrillMode {
       // === Input Layer Drills ===
       Garbage,           // Layer 2: Low-quality fragment ingestion
       UnitActivation,    // Layer 2: Rolling hash activation edge cases
       
       // === Spatial Layer Drills ===
       Collisions,        // Layer 5: Spatial hash collisions
       RoutingEscape,     // Layer 5: Neighbor selection and escape logic
       
       // === Context Layer Drills ===
       AnchorLoss,        // Layer 6: Anchor protection failures
       ContextMatrix,     // Layer 6: Context matrix state consistency
       
       // === Intent-Driven Input Drills (NEW) ===
       IntentClassify,    // Layer 7: Intent classification accuracy
       IntentBlend,       // Layer 7: Hybrid heuristic + memory blend validation
       RetrievalGate,     // Layer 9: Entropy/freshness/cost scoring for retrieval
       IntentMemoryGate,  // Layer 9: MemoryChannel::Intent routing correctness
       
       // === Safety Layer Drills ===
       Poison,            // Layer 19: Malicious source injection
       TrustHeuristics,    // Layer 19: Trust score validation edge cases
       
       // === Memory Layer Drills ===
       Maintenance,       // Layer 21: Pruning edge cases
       Promotion,         // Layer 21: Candidate promotion logic
       ChannelIsolation,  // Layer 21: MemoryChannel isolation enforcement
       
       // === Intent-Driven Output Drills (NEW) ===
       OutputDecode,      // Layer 17: Answer finalization with intent shaping
       CreativeDrift,     // Layer 17: Semantic drift vs factual corruption detection
       IntentShaping,     // Layer 17: Intent-specific output profile application
   }
   
   enum DrillCategory {
       HappyPath,    // Expected normal behavior
       EdgeCase,     // Boundary conditions, unusual inputs
       FailureMode,  // Known failure scenarios, error handling
       Stress,       // High load, resource exhaustion
   }
   ```

2. **Generate Test Corpora** (`scripts/drill_corpus_generator.py`):
   
   **Input Layer Corpora:**
   - `garbage_corpus.json`: High punctuation ratio, low semantic content
   - `unit_activation_corpus.json`: Edge cases for rolling hash activation thresholds
   
   **Spatial Layer Corpora:**
   - `collision_corpus.json`: Deliberately similar semantic positions
   - `routing_escape_corpus.json`: Units requiring escape from dense neighborhoods
   
   **Context Layer Corpora:**
   - `anchor_corpus.json`: Units at prune threshold with anchor status
   - `context_matrix_corpus.json`: Long sequences testing context window limits
   
   **Intent-Driven Input Corpora (NEW):**
   - `intent_classify_corpus.json`: Labeled queries with known intent types (factual, brainstorm, plan, act, critique)
   - `intent_blend_corpus.json`: Queries with conflicting heuristic vs memory-backed scores
   - `retrieval_gate_corpus.json`: High/low entropy queries testing L9 gating thresholds
   - `intent_memory_corpus.json`: Queries targeting MemoryChannel::Intent vs Core routing
   
   **Safety Layer Corpora:**
   - `poison_corpus.json`: Sources with trust score below threshold
   - `trust_heuristics_corpus.json`: Edge cases for source trust validation
   
   **Memory Layer Corpora:**
   - `maintenance_corpus.json`: Units at various utility thresholds for pruning tests
   - `promotion_corpus.json`: Candidates at promotion boundaries
   - `channel_isolation_corpus.json`: Intent channel units that should NOT promote to Core
   
   **Intent-Driven Output Corpora (NEW):**
   - `output_decode_corpus.json`: Queries with expected answer formats per intent type
   - `creative_drift_corpus.json`: Inputs designed to trigger semantic drift in creative mode
   - `intent_shaping_corpus.json`: Queries testing per-intent output profiles (plan vs act vs brainstorm)

3. **Migrate Pollution Logic** (from `src/bin/pollution_dev_lib.rs`):
   - Move `purge_polluted_memory()` tests into drill framework
   - Integrate `pollution_findings` reporting into unified drill output
   - Add pollution ceiling assertion (<1%)

4. **SpatialGrid Layer 5-6 Tests**:
   - Test neighborhood search accuracy under high-density clustering
   - Assert no halo regions needed (fixed 3D hash implementation)
   - Verify `spatial_cell_size=4.0` and `neighbor_radius=6.0` config values
   - Test escape mechanism when neighborhood saturated

5. **Intent-Driven Input Drills (Layer 7, L9) - NEW:**
   
   **Layer 7 - Intent Classification Drills:**
   ```rust
   // Happy path: Correct classification
   fn intent_classify_happy_path();
   // Edge case: Ambiguous intent (low confidence across all types)
   fn intent_classify_ambiguous_low_entropy();
   // Edge case: High confidence wrong intent (entropy trap)
   fn intent_classify_entropy_trap();
   // Failure: Empty input handling
   fn intent_classify_empty_input();
   // Stress: Rapid intent switching in sequence
   fn intent_classify_rapid_switching();
   ```
   
   **Layer 7 - Hybrid Blend Drills:**
   ```rust
   // Happy path: Heuristic and memory agree
   fn intent_blend_agreement();
   // Edge case: Heuristic/memory conflict (test blend weights 0.6/0.4)
   fn intent_blend_conflict_resolution();
   // Edge case: Memory-backed score missing (fallback to heuristic)
   fn intent_blend_memory_missing();
   // Failure: Both scores below floor
   fn intent_blend_both_below_floor();
   ```
   
   **Layer 9 - Retrieval Gate Drills:**
   ```rust
   // Happy path: High entropy triggers retrieval
   fn retrieval_gate_high_entropy();
   // Happy path: Fresh context skips retrieval
   fn retrieval_gate_fresh_context();
   // Edge case: Entropy at threshold boundary
   fn retrieval_gate_entropy_threshold_boundary();
   // Edge case: Cost budget exhausted
   fn retrieval_gate_cost_budget_exhausted();
   // Failure: Retrieval source unavailable
   fn retrieval_gate_source_unavailable();
   ```

6. **Intent-Driven Output Drills (Layer 17) - NEW:**
   
   **Layer 17 - Output Decode Drills:**
   ```rust
   // Happy path: Factual intent produces concise answer
   fn output_decode_factual_concise();
   // Happy path: Brainstorm intent produces exploratory answer
   fn output_decode_brainstorm_exploratory();
   // Edge case: No candidates above confidence floor
   fn output_decode_no_confident_candidates();
   // Edge case: Multiple equally-scored candidates
   fn output_decode_tie_breaking();
   // Failure: Empty candidate pool
   fn output_decode_empty_pool();
   ```
   
   **Layer 17 - Creative Drift Drills:**
   ```rust
   // Happy path: Creative mode allows semantic drift
   fn creative_drift_allowed();
   // Edge case: Drift exceeds tolerance but preserves anchors
   fn creative_drift_at_tolerance();
   // Failure: Factual corruption detected (anchors modified)
   fn creative_drift_factual_corruption_blocked();
   // Stress: High drift tolerance with many anchors
   fn creative_drift_many_anchors();
   ```
   
   **Layer 17 - Intent Shaping Drills:**
   ```rust
   // Happy path: Plan intent applies plan profile
   fn intent_shaping_plan_profile();
   // Happy path: Act intent applies act profile
   fn intent_shaping_act_profile();
   // Edge case: Unknown intent falls back to default
   fn intent_shaping_unknown_fallback();
   // Edge case: Profile config missing fields
   fn intent_shaping_incomplete_profile();
   ```

7. **Module-Specific Drills (All Layers) - NEW:**
   
   **Layer 2 - Unit Builder:**
   ```rust
   fn unit_builder_rolling_hash_activation();     // Happy path
   fn unit_builder_frequency_threshold_boundary(); // Edge case
   fn unit_builder_duplicate_detection();         // Edge case
   fn unit_builder_max_units_limit();             // Stress
   ```
   
   **Layer 3 - Hierarchy Organizer:**
   ```rust
   fn hierarchy_level_grouping();           // Happy path
   fn hierarchy_entity_extraction();        // Happy path
   fn hierarchy_nested_structure();         // Edge case
   fn hierarchy_circular_reference();        // Failure mode
   ```
   
   **Layer 10 - Query Builder:**
   ```rust
   fn query_builder_safe_construction();    // Happy path
   fn query_builder_pii_stripping();        // Happy path
   fn query_builder_injection_attempt();    // Failure mode
   fn query_builder_max_length();           // Edge case
   ```
   
   **Layer 11 - Retrieval Pipeline:**
   ```rust
   fn retrieval_pipeline_fetch();           // Happy path
   fn retrieval_pipeline_caching();         // Happy path
   fn retrieval_pipeline_timeout();         // Failure mode
   fn retrieval_pipeline_rate_limit();      // Failure mode
   ```
   
   **Layer 12 - Safety Validator:**
   ```rust
   fn safety_validator_trust_pass();        // Happy path
   fn safety_validator_trust_fail();        // Happy path
   fn safety_validator_allowlist();         // Edge case
   fn safety_validator_malformed_doc();     // Failure mode
   ```
   
   **Layer 13 - Evidence Merger:**
   ```rust
   fn evidence_merge_agreement();           // Happy path
   fn evidence_merge_conflict();            // Edge case
   fn evidence_merge_source_priority();     // Edge case
   fn evidence_merge_empty_evidence();      // Failure mode
   ```
   
   **Layer 14 - Candidate Scorer:**
   ```rust
   fn candidate_scorer_7d_features();       // Happy path
   fn candidate_scorer_weight_application();// Happy path
   fn candidate_scorer_nan_handling();      // Failure mode
   fn candidate_scorer_tie_breaking();      // Edge case
   ```
   
   **Layer 16 - Fine Resolver:**
   ```rust
   fn fine_resolver_top_k_selection();      // Happy path
   fn fine_resolver_temperature_sampling(); // Happy path
   fn fine_resolver_empty_pool();           // Failure mode
   fn fine_resolver_all_below_floor();      // Edge case
   ```
   
   **Layer 18 - Feedback Controller:**
   ```rust
   fn feedback_controller_learning_event(); // Happy path
   fn feedback_controller_impact_scoring();// Happy path
   fn feedback_controller_invalid_signal(); // Failure mode
   fn feedback_controller_batch_learning(); // Stress
   ```

**Files to Create:**
- `src/bin/drill_harness.rs`
- `scripts/drill_corpus_generator.py`
- `test_data/drill_corpora/` directory with all corpora files

**Files to Modify:**
- `src/bin/pollution_dev.rs` → refactor into drill framework
- `tests/integration.rs` → add drill tests for all modes and categories
- `src/layers/intent.rs` → add drill test functions
- `src/layers/output.rs` → add drill test functions
- `src/layers/resolver.rs` → add drill test functions

**Verification:**
```bash
# Input layer drills
cargo run --bin drill_harness -- --mode garbage --category happy_path
cargo run --bin drill_harness -- --mode garbage --category edge_case
cargo run --bin drill_harness -- --mode unit_activation --category stress

# Spatial layer drills
cargo run --bin drill_harness -- --mode collisions --assert-halo-false
cargo run --bin drill_harness -- --mode routing_escape --category edge_case

# Intent-driven input drills (NEW)
cargo run --bin drill_harness -- --mode intent_classify --category happy_path
cargo run --bin drill_harness -- --mode intent_blend --category edge_case
cargo run --bin drill_harness -- --mode retrieval_gate --category failure_mode

# Intent-driven output drills (NEW)
cargo run --bin drill_harness -- --mode output_decode --category happy_path
cargo run --bin drill_harness -- --mode creative_drift --category edge_case
cargo run --bin drill_harness -- --mode intent_shaping --category failure_mode

# Memory layer drills
cargo run --bin drill_harness -- --mode maintenance --corpus 100mb
cargo run --bin drill_harness -- --mode channel_isolation --category edge_case

# Run all drill tests
cargo test --test integration -- drill_
cargo test --lib -- drill_
```

---

### 2.2 Unified Stress Drill (Heterogeneous Ingestion)

**What to Implement:**
- End-to-end 100MB ingestion with interleaved queries and forced maintenance cycles
- Validate Layer 12 normalization for HTML, QA JSON, Wikidata, OpenAPI, Common Crawl WET
- Monitor latency spikes, pollution ceilings, snapshot consistency

**How to Implement:**

1. **Create Stress Drill Runner** (`src/bin/stress_drill.rs`):
   ```rust
   struct StressDrillConfig {
       corpus_size_mb: usize,
       source_types: Vec<SourceTypeId>,
       query_interval_ms: u64,
       maintenance_interval_sec: u64,
       max_latency_spike_ms: u64,
       pollution_ceiling_percent: f32,
   }
   ```

2. **Implement Heterogeneous Source Generator** (`scripts/heterogeneous_corpus.py`):
   - Generate mixed corpus: 30% HTML, 20% QA JSON, 15% Wikidata truthy, 15% OpenAPI, 20% Common Crawl WET
   - Apply `source_policies` config for each type

3. **Add Latency Monitor** (`src/telemetry/latency_tracker.rs`):
   - Track per-layer latency during stress test
   - Flag spikes exceeding 2x baseline
   - Report to Layer 20 telemetry

4. **Wire Layer 12 Adapter** (`src/document.rs`):
   - Implement format-aware normalization for Hugging Face row formats
   - Add Wikipedia XML handler preserving relational structure
   - Apply `source_policies` extraction rules per format

**Files to Create:**
- `src/bin/stress_drill.rs`
- `src/telemetry/latency_tracker.rs`
- `scripts/heterogeneous_corpus.py`

**Files to Modify:**
- `src/document.rs` (format adapters)
- `src/telemetry/mod.rs` (exports)

---

### 2.3 Snapshot Atomicity & Recovery Drill

**What to Implement:**
- Simulate crashes during Layer 5 map updates or Layer 21 compaction
- Verify ArcSwap rollback mechanisms for Force-Directed updates and SpatialGrid re-indexing
- Ensure inference threads never blocked by background maintenance

**How to Implement:**

1. **Create Crash Simulator** (`src/bin/crash_drill.rs`):
   ```rust
   enum CrashPoint {
       ForceDirectedMidIteration,
       SpatialGridRebuild,
       LfuPruning,
       CandidatePromotion,
   }
   ```

2. **Add ArcSwap Rollback Tests** (`tests/integration.rs`):
   - Test snapshot consistency after simulated crash
   - Verify `load_memory_snapshot()` returns last valid state
   - Assert no partial writes in SQLite

3. **Implement Deferred Correction Queue** (`src/memory/store.rs`):
   - Queue spatial corrections during maintenance
   - Apply only after successful snapshot
   - Rollback queue on failure

**Files to Create:**
- `src/bin/crash_drill.rs`

**Files to Modify:**
- `src/memory/store.rs` (deferred queue)
- `src/engine.rs` (ArcSwap usage verification)
- `tests/integration.rs` (crash tests)

---

## Phase 3: Core Logic & Tests

**Estimated Duration:** 4-5 weeks  
**Dependencies:** Phase 2 complete

### 3.1 Global Logging Engine & Trace Visualization (Layer 20)

**What to Implement:**
- Non-blocking async worker emitting structured JSON events
- Hot logs in SQLite for real-time UI, cold logs in append-only files
- Capture `intent_label` markers from hybrid Intent-channel memory
- Distinguish `MemoryChannel::Intent` vs standard reasoning traces

**How to Implement:**

1. **Create Async Telemetry Worker** (`src/telemetry/worker.rs`):
   ```rust
   pub struct TelemetryWorker {
       sender: Sender<TelemetryEvent>,
       hot_store: SqliteHotStore,
       cold_log: AppendOnlyFile,
   }
   
   pub enum TelemetryEvent {
       Calculation { layer: u8, operation: String, duration_ms: u64 },
       DbPush { unit_id: Uuid, memory_type: MemoryType },
       Retrieval { source: String, results: usize },
       MorphAction { action: String, before: String, after: String },
       IntentLabel { label: IntentKind, channel: MemoryChannel, score: f32 },
   }
   ```

2. **Add Session/Trace IDs** (`src/telemetry/trace.rs`):
   - Generate `session_id` on engine init
   - Generate `trace_id` per query
   - Include in all telemetry events

3. **Create SQLite Hot Store** (`src/telemetry/hot_store.rs`):
   - Table schema for recent events (last 10,000)
   - Index on `session_id`, `trace_id`, `timestamp`
   - Auto-prune old events

4. **Wire into Engine** (`src/engine.rs`):
   - Initialize `TelemetryWorker` in `Engine::new()`
   - Emit events at each layer boundary
   - Add `intent_label` emission in intent classification

**Files to Create:**
- `src/telemetry/worker.rs`
- `src/telemetry/hot_store.rs`
- `src/telemetry/trace.rs`

**Files to Modify:**
- `src/telemetry/mod.rs` (exports)
- `src/engine.rs` (wire worker)
- `config/config.yaml` (Layer 20 config expansion)

---

### 3.2 Multi-Engine Consensus & Structured Parsing

**What to Implement:**
- Extend Layers 11-13 to aggregate multiple search engines
- Implement consensus scoring rather than simple merging
- Layer 12 adapter for Hugging Face row formats and Wikipedia XML

**How to Implement:**

1. **Add Multi-Engine Aggregator** (`src/layers/retrieval.rs`):
   ```rust
   pub struct MultiEngineAggregator {
       engines: Vec<SearchEngine>,
       consensus_threshold: f32,
   }
   
   impl MultiEngineAggregator {
       pub fn aggregate(&self, query: &str) -> ConsensusResult {
           // Query all engines
           // Apply consensus scoring: agreement across engines
           // Return merged with confidence boost for corroboration
       }
   }
   ```

2. **Update Evidence Merge** (`src/layers/merge.rs`):
   - Apply baseline policy: `0.5·Trust + 0.3·Recency + 0.2·Agreement`
   - Handle internal pseudo-sources vs external web evidence
   - Add conflict resolution for disagreement

3. **Add Format Adapters** (`src/document.rs`):
   - `HuggingFaceRowAdapter`: Parse dataset rows, preserve field structure
   - `WikipediaXmlAdapter`: Extract articles, preserve section hierarchy
   - Apply `source_policies.format_ingestion_rules`

**Files to Modify:**
- `src/layers/retrieval.rs` (multi-engine)
- `src/layers/merge.rs` (consensus)
- `src/document.rs` (adapters)
- `config/config.yaml` (source_policies expansion)

---

### 3.3 Concurrency Stress Test (Three-Class Model)

**What to Implement:**
- Run parallel Silent Training workers alongside Inference threads
- Enforce priority: Inference > Interactive Training > Silent Batch > Maintenance
- Verify inference latency <200ms/token under load

**How to Implement:**

1. **Create Priority Scheduler** (`src/scheduler.rs` - already exists, extend):
   ```rust
   pub enum ExecutionClass {
       Inference,         // Priority 1
       InteractiveTraining, // Priority 2
       SilentBatch,       // Priority 3
       Maintenance,       // Priority 4
   }
   ```

2. **Add Load Test** (`src/bin/concurrency_drill.rs`):
   - Spawn N Silent Training workers ingesting Hugging Face streams
   - Run concurrent inference queries
   - Measure latency percentiles (p50, p95, p99)
   - Assert p95 < 200ms

3. **Add Queue Depth Monitoring** (`src/telemetry/`):
   - Track `QueueDepths` struct
   - Report to Layer 20 telemetry
   - Alert on saturation

**Files to Create:**
- `src/bin/concurrency_drill.rs`

**Files to Modify:**
- `src/scheduler.rs` (extend)
- `src/engine.rs` (priority enforcement)

---

### 3.4 Config Sweeping & Benchmarking

**What to Implement:**
- Automate parameter sweeping on 100MB corpus
- Include Silent Training Mode parameter sweeps
- Output Pareto frontier graphs (Latency vs Pollution)
- Optimize SpatialGrid index rebuild cadence

**How to Implement:**

1. **Extend Config Sweep Harness** (`src/bin/test_harness.rs`):
   ```rust
   struct SweepConfig {
       entropy_range: Vec<f32>,
       trust_range: Vec<f32>,
       anchor_threshold_range: Vec<f32>,
       spatial_cell_size_range: Vec<f32>,
       max_memory_delta_range: Vec<f32>,
       batch_size_range: Vec<usize>,
   }
   ```

2. **Add Pareto Analysis** (`scripts/analyze_sweep.py`):
   - Read sweep results from JSON
   - Generate Pareto frontier plot
   - Identify optimal config for 200GB training run

3. **Add SpatialGrid Optimization**:
   - Sweep `spatial_cell_size` (2.0, 4.0, 6.0, 8.0)
   - Sweep `max_layout_iterations` (12, 24, 36)
   - Measure rebuild time vs query accuracy

**Files to Modify:**
- `src/bin/test_harness.rs` (extend)
- `scripts/analyze_results.py` (Pareto analysis)

---

## Phase 4: Web UI (SPSE Chat Interface)

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 3 complete (Layer 20 telemetry)

### 4.1 React/Next.js Interface

**What to Implement:**
- Web UI connecting to SPSE API
- Mode toggles: Creative / Precise / Research
- Real-time Layer 20 trace visualization

**How to Implement:**

1. **Create Next.js Project** (`web-ui/`):
   ```
   web-ui/
   ├── app/
   │   ├── page.tsx           # Main chat interface
   │   ├── layout.tsx
   │   └── api/
   │       └── chat/route.ts  # API proxy to SPSE
   ├── components/
   │   ├── ChatInterface.tsx
   │   ├── ModeToggle.tsx
   │   ├── TraceVisualization.tsx
   │   └── IntentBreakdown.tsx
   ├── lib/
   │   └── spse-client.ts
   └── package.json
   ```

2. **Implement Mode Toggle**:
   - Creative: `mode=exploratory`, high `stochastic_jump_prob`
   - Precise: `mode=deterministic`, low `stochastic_jump_prob`
   - Research: `mode=balanced`, retrieval-heavy

3. **Add Trace Visualization**:
   - WebSocket connection to SPSE telemetry stream
   - Display layer-by-layer processing
   - Show timing per layer
   - Highlight retrieval decisions

**Files to Create:**
- `web-ui/` directory with Next.js project

---

### 4.2 Hybrid Intent Score Breakdown Display

**What to Implement:**
- Display intent score breakdown (heuristic vs memory-backed)
- Show active `MemoryChannel` (Intent/Reasoning/Main)
- Demonstrate dual-memory architecture to user

**How to Implement:**

1. **Create IntentBreakdown Component** (`web-ui/components/IntentBreakdown.tsx`):
   ```tsx
   interface IntentBreakdownProps {
       heuristicScore: number;
       memoryBackedScore: number;
       blendedScore: number;
       activeChannel: 'main' | 'intent' | 'reasoning';
       intentLabel: string;
   }
   ```

2. **Add API Endpoint** (`src/api.rs`):
   - Extend `/query` response to include `IntentBlendReport`
   - Add `/telemetry/stream` WebSocket endpoint

**Files to Modify:**
- `src/api.rs` (extend endpoints)

---

## Execution Timeline

| Phase | Duration | Start | Dependencies |
|-------|----------|-------|--------------|
| Phase 1: Creative Mode | 2-3 weeks | Week 1 | None |
| Phase 2: Drill Suite | 3-4 weeks | Week 4 | Phase 1 |
| Phase 3: Core Logic | 4-5 weeks | Week 8 | Phase 2 |
| Phase 4: Web UI | 3-4 weeks | Week 13 | Phase 3 |

**Total Estimated Duration:** 12-16 weeks

---

## Success Criteria

### Phase 1
- [ ] Creative mode produces semantically drifted outputs without factual corruption
- [ ] Hybrid intent blend correctly weights heuristic vs memory-backed scores
- [ ] Intent channel isolation prevents Core memory pollution

### Phase 2
- [ ] All drill modes pass with pollution <1%
- [ ] SpatialGrid neighborhood search accurate without halo regions
- [ ] Crash recovery preserves snapshot consistency
- [ ] Heterogeneous ingestion handles all source types
- [ ] **Intent-driven input drills (L7, L9) pass all categories:**
  - [ ] Intent classification: happy path, edge cases (ambiguous, entropy trap), failure (empty input)
  - [ ] Hybrid blend: agreement, conflict resolution, memory missing fallback
  - [ ] Retrieval gate: high/low entropy, threshold boundary, cost exhausted, source unavailable
- [ ] **Intent-driven output drills (L17) pass all categories:**
  - [ ] Output decode: factual/brainstorm modes, no candidates, tie breaking, empty pool
  - [ ] Creative drift: allowed drift, tolerance boundary, factual corruption blocked
  - [ ] Intent shaping: plan/act profiles, unknown fallback, incomplete profile handling
- [ ] **Module-specific drills pass for all layers (L2-L18):**
  - [ ] Layer 2: Unit builder activation, frequency threshold, duplicates
  - [ ] Layer 3: Hierarchy grouping, entity extraction, circular reference
  - [ ] Layer 10: Query builder safe construction, PII stripping, injection
  - [ ] Layer 11: Retrieval fetch, caching, timeout, rate limit
  - [ ] Layer 12: Safety validator trust pass/fail, allowlist, malformed
  - [ ] Layer 13: Evidence merge agreement, conflict, source priority
  - [ ] Layer 14: Candidate scorer 7D features, weight application, NaN handling
  - [ ] Layer 16: Fine resolver top-k, temperature sampling, empty pool
  - [ ] Layer 18: Feedback controller learning, impact scoring, batch

### Phase 3
- [ ] Layer 20 telemetry captures all calculations with Session/Trace IDs
- [ ] Multi-engine consensus improves retrieval quality
- [ ] Inference latency <200ms under concurrent Silent Training load
- [ ] Pareto frontier identifies optimal config

### Phase 4
- [ ] Web UI displays real-time trace visualization
- [ ] Mode toggles correctly affect engine behavior
- [ ] Intent breakdown visible to user

---

## Risk Mitigation

1. **Creative Mode Factual Corruption**
   - Mitigation: Anchor protection with `trust_score > 0.8` threshold
   - Fallback: Disable creative mode for factual intent types

2. **SpatialGrid Collision Under Load**
   - Mitigation: Adaptive cell sizing based on density
   - Fallback: Fall back to linear search for dense regions

3. **Telemetry Performance Overhead**
   - Mitigation: Async worker with batching
   - Fallback: Configurable sample rate (default 1.0, reduce to 0.1 if needed)

4. **Concurrency Deadlock**
   - Mitigation: Priority queue with timeout
   - Fallback: Emergency maintenance pause under high inference load
