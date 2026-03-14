# SPSE Engine Pre-Production Execution Plan

**Document Version:** 1.3  
**Created:** March 14, 2026  
**Last Updated:** March 15, 2026  
**Priority Order:** Foundation > LLM-Like Core > Retrieval > Optimization > Interface

---

## Phase 1: Creative Mode & Hybrid Intent Profiles ✅ COMPLETE

**Estimated Duration:** 2-3 weeks  
**Actual Duration:** Completed
**Dependencies:** None (foundational for other phases)

### 1.1 Intent-Specific Config Profiles for Layers 14-17 ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `IntentShapingConfig` struct added to `src/config/mod.rs` (lines 565-571)
- `AdaptiveBehaviorConfig.intent_profiles` includes `creative`, `brainstorm`, `plan`, `act`, `critique` profiles
- `FineResolver::select_with_shaping()` implements intent shaping in `src/layers/resolver.rs` (lines 81-165)
- `config/config.yaml` contains full profile definitions (lines 131-334)

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

### 1.2 Hybrid Intent Score Validation ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `IntentDetector::hybrid_blend()` method added to `src/layers/intent.rs` (lines 1292-1358)
- `IntentBlendReport` struct defined in `src/types.rs` (lines 332-358)
- `OutputDecoder::detect_drift()` and `detect_corruption()` methods in `src/layers/output.rs` (lines 39-135)
- `DriftReport` and `CorruptionReport` structs for validation output

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

### 1.3 MemoryChannel::Intent Gate Validation ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `intent_channel_core_promotion_blocked` config field added to `GovernanceConfig` in `src/config/mod.rs` (line 723)
- `validate_channel_isolation()` method implemented in `src/memory/store.rs` (lines 544-619)
- `ChannelIsolationReport`, `ChannelIsolationViolation`, `IsolationViolationType` types in `src/types.rs` (lines 362-396)
- Intent channel gate logic blocks Core promotion for Intent-channel units in `ingest_activation()` (lines 784-787)
- Integration test `intent_channel_isolation_prevents_core_pollution()` added to `tests/integration.rs` (lines 1766-1817)
- Config field added to `config/config.yaml` (line 374)

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

## Phase 2: Drill Suite & Pollution Integration ✅ COMPLETE

**Estimated Duration:** 3-4 weeks  
**Actual Duration:** Completed
**Dependencies:** Phase 1 complete

### 2.1 Layer-Specific Drill Suite (100MB Corpus) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `drill_harness.rs` binary created with 20 drill modes across all layers (`src/bin/drill_harness.rs`)
- `drill_lib.rs` library module with `DrillMode`, `DrillCategory`, `DrillResult`, `DrillReport` structs (`src/drill_lib.rs`)
- `drill_corpus_generator.py` script generates JSON corpora for all drill modes (`scripts/drill_corpus_generator.py`)
- Drill modes implemented: Garbage, UnitActivation, Collisions, RoutingEscape, AnchorLoss, ContextMatrix, IntentClassify, IntentBlend, RetrievalGate, IntentMemoryGate, Poison, TrustHeuristics, Maintenance, Promotion, ChannelIsolation, OutputDecode, CreativeDrift, IntentShaping, PollutionCeiling, PollutionPurge

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

8. **Phase 1 Feature Drills (Creative Mode, Hybrid Blend, Intent Channel) - NEW:**
   
   **Creative Mode Drills (L14-L17):**
   ```rust
   // Happy path: Brainstorm intent triggers exploratory resolver mode
   fn creative_mode_brainstorm_exploratory();
   // Happy path: Plan intent applies structured output profile
   fn creative_mode_plan_structured();
   // Edge case: Creative mode disabled for factual queries
   fn creative_mode_factual_disabled();
   // Edge case: Stochastic jump probability at boundary (0.0, 1.0)
   fn creative_mode_stochastic_boundary();
   // Failure: Profile missing for intent type
   fn creative_mode_missing_profile();
   // Stress: Rapid mode switching between creative/precise
   fn creative_mode_rapid_switching();
   ```
   
   **Hybrid Blend Drills (L7):**
   ```rust
   // Happy path: 0.6/0.4 blend produces correct score
   fn hybrid_blend_weight_calculation();
   // Edge case: Heuristic score NaN fallback
   fn hybrid_blend_heuristic_nan();
   // Edge case: Memory channel returns empty
   fn hybrid_blend_memory_empty();
   // Failure: Intent channel corrupted data
   fn hybrid_blend_corrupted_memory();
   ```
   
   **Intent Channel Isolation Drills (L4, L21):**
   ```rust
   // Happy path: Intent units stay in Intent channel
   fn intent_channel_isolation_happy();
   // Edge case: Intent unit at promotion boundary
   fn intent_channel_promotion_boundary();
   // Failure: Intent-to-Core promotion attempt blocked
   fn intent_channel_core_promotion_blocked();
   // Stress: High-volume intent signals
   fn intent_channel_high_volume();
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

### 2.2 Unified Stress Drill (Heterogeneous Ingestion) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `stress_drill.rs` binary created with configurable corpus size, latency thresholds, pollution ceiling (`src/bin/stress_drill.rs`)
- `stress_drill_lib.rs` library module with `StressDrillConfig`, `StressDrillResult`, `LatencyReport` structs (`src/stress_drill_lib.rs`)
- Heterogeneous corpus generator supports HTML, QA JSON, Wikidata, OpenAPI, Common Crawl WET formats
- Distribution: 30% HTML, 20% QA JSON, 15% Wikidata, 15% OpenAPI, 20% Common Crawl WET
- Latency tracking with avg/max/p99 metrics and spike detection
- Pollution ceiling validation (<1% default)

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

### 2.3 Snapshot Atomicity & Recovery Drill ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `crash_drill.rs` binary created with crash simulation at multiple points (`src/bin/crash_drill.rs`)
- `crash_drill_lib.rs` library module with `CrashPoint`, `CrashDrillConfig`, `CrashDrillResult` structs (`src/crash_drill_lib.rs`)
- Crash points: ForceDirectedMidIteration, SpatialGridRebuild, LfuPruning, CandidatePromotion
- Recovery verification via memory counts comparison before/after crash
- Consistency error detection for partial writes

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

### 2.4 Cross-Layer Integration Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `layer_boundary_tests.rs` created with 16 layer boundary contract tests (`tests/layer_boundary_tests.rs`)
- Layer boundary tests: L1→L2, L2→L3, L3→L4, L5→L6, L6→L7, L7→L9, L14→L16, L16→L17
- State consistency tests: unit ID preservation, context overflow, unit dropped, high throughput
- Backpressure tests: normal flow, retrieval bottleneck, queue overflow, sustained high rate

**What to Implement:**
- Validate data flow between adjacent layers
- Test layer boundary contracts (input/output schemas)
- Verify state consistency across layer transitions

**How to Implement:**

1. **Layer Boundary Contract Tests:**
   ```rust
   // L1→L2: InputPacket to Unit activation
   fn l1_to_l2_input_packet_schema();
   // L2→L3: Unit stream to Hierarchy grouping
   fn l2_to_l3_unit_stream();
   // L3→L4: Hierarchy to Memory ingestion
   fn l3_to_l4_hierarchy_memory();
   // L5→L6: Spatial routing to Context matrix
   fn l5_to_l6_spatial_context();
   // L6→L7: Context state to Intent detection
   fn l6_to_l7_context_intent();
   // L7→L9: Intent to Retrieval decision
   fn l7_to_l9_intent_retrieval();
   // L14→L16: Candidate scores to Resolver selection
   fn l14_to_l16_candidate_resolver();
   // L16→L17: Resolver output to Output decoder
   fn l16_to_l17_resolver_output();
   ```

2. **State Consistency Drills:**
   ```rust
   // Happy path: Unit ID preserved across all layers
   fn state_consistency_unit_id();
   // Edge case: Context window overflow mid-pipeline
   fn state_consistency_context_overflow();
   // Failure: Unit dropped between layers
   fn state_consistency_unit_dropped();
   // Stress: High-throughput pipeline pressure
   fn state_consistency_high_throughput();
   ```

3. **Pipeline Backpressure Drills:**
   ```rust
   // Happy path: Normal flow rate
   fn backpressure_normal_flow();
   // Edge case: L11 retrieval bottleneck
   fn backpressure_retrieval_bottleneck();
   // Failure: Queue overflow at L14
   fn backpressure_queue_overflow();
   // Stress: Sustained high input rate
   fn backpressure_sustained_high_rate();
   ```

**Files to Create:**
- `tests/layer_boundary_tests.rs`

**Files to Modify:**
- `tests/integration.rs` (cross-layer tests)

---

### 2.5 End-to-End Scenario Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- 20 e2e scenario tests added to `tests/integration.rs`
- Single query lifecycle tests: factual, brainstorm, retrieval triggered, layer failure, concurrent
- Multi-turn conversation tests: context preserved, topic shift, intent change, context loss, long conversation
- Training mode tests: silent ingestion, interactive feedback, interrupted, data corruption, large corpus
- Error recovery tests: graceful degradation, partial output, pipeline failure, cascading failures

**What to Implement:**
- Full query lifecycle from input to output
- Real-world usage patterns and user scenarios
- Multi-turn conversation flows

**How to Implement:**

1. **Single Query Lifecycle Drills:**
   ```rust
   // Happy path: Factual query complete lifecycle
   fn e2e_factual_query_lifecycle();
   // Happy path: Brainstorm query with creative mode
   fn e2e_brainstorm_creative_lifecycle();
   // Edge case: Query with retrieval triggered
   fn e2e_retrieval_triggered_lifecycle();
   // Failure: Query fails at specific layer
   fn e2e_query_layer_failure();
   // Stress: Concurrent query processing
   fn e2e_concurrent_queries();
   ```

2. **Multi-Turn Conversation Drills:**
   ```rust
   // Happy path: Follow-up query with context
   fn e2e_multi_turn_context_preserved();
   // Edge case: Topic shift mid-conversation
   fn e2e_multi_turn_topic_shift();
   // Edge case: Intent change between turns
   fn e2e_multi_turn_intent_change();
   // Failure: Context loss between turns
   fn e2e_multi_turn_context_loss();
   // Stress: Long conversation (50+ turns)
   fn e2e_multi_turn_long_conversation();
   ```

3. **Training Mode Scenario Drills:**
   ```rust
   // Happy path: Silent training ingestion
   fn e2e_silent_training_ingestion();
   // Happy path: Interactive training with feedback
   fn e2e_interactive_training_feedback();
   // Edge case: Training interrupted by inference
   fn e2e_training_interrupted();
   // Failure: Training data corruption
   fn e2e_training_data_corruption();
   // Stress: Large corpus training (100MB+)
   fn e2e_large_corpus_training();
   ```

4. **Error Recovery Scenario Drills:**
   ```rust
   // Happy path: Graceful degradation on retrieval failure
   fn e2e_graceful_degradation();
   // Edge case: Partial output on resource exhaustion
   fn e2e_partial_output();
   // Failure: Complete pipeline failure recovery
   fn e2e_pipeline_failure_recovery();
   // Stress: Cascading failures
   fn e2e_cascading_failures();
   ```

**Files to Modify:**
- `tests/integration.rs` (scenario tests)

---

## Phase 3: Core Infrastructure

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 2 complete

**Objective:** Build foundational infrastructure that enables all subsequent LLM-like features. Telemetry provides debugging visibility, Reasoning Loop enables complex queries, and Concurrency validation ensures system stability.

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
n       sender: Sender<TelemetryEvent>,
n       hot_store: SqliteHotStore,
n       cold_log: AppendOnlyFile,
n   }
n   
n   pub enum TelemetryEvent {
n       Calculation { layer: u8, operation: String, duration_ms: u64 },
n       DbPush { unit_id: Uuid, memory_type: MemoryType },
n       Retrieval { source: String, results: usize },
n       MorphAction { action: String, before: String, after: String },
n       IntentLabel { label: IntentKind, channel: MemoryChannel, score: f32 },
n   }
n   ```

2. **Add Session/Trace IDs** (`src/telemetry/trace.rs`):
n   - Generate `session_id` on engine init
n   - Generate `trace_id` per query
n   - Include in all telemetry events

3. **Create SQLite Hot Store** (`src/telemetry/hot_store.rs`):
n   - Table schema for recent events (last 10,000)
n   - Index on `session_id`, `trace_id`, `timestamp`
n   - Auto-prune old events

4. **Wire into Engine** (`src/engine.rs`):
n   - Initialize `TelemetryWorker` in `Engine::new()`
n   - Emit events at each layer boundary
n   - Add `intent_label` emission in intent classification

**Files to Create:**
- `src/telemetry/worker.rs`
- `src/telemetry/hot_store.rs`
- `src/telemetry/trace.rs`

**Files to Modify:**
- `src/telemetry/mod.rs` (exports)
- `src/engine.rs` (wire worker)
- `config/config.yaml` (Layer 20 config expansion)

---

### 3.2 Recursive Self-Ingestion Loop (Chain-of-Thought)

**What to Implement:**
- Generate silent "Thought Units" that loop back as input
- Solve complex logic/coding problems step-by-step
- Max 5 internal steps before final answer
- Requires Telemetry (3.1) for reasoning step logging

**How to Implement:**

1. **Add Silent Thought Output** (`src/engine.rs`):
   ```rust
   pub enum OutputType {
       FinalAnswer(String),
       SilentThought(String),  // Internal reasoning step
   }
   ```

2. **Add Thought Unit Type** (`src/types.rs`):
   ```rust
   pub struct ThoughtUnit {
       pub content: String,
       pub step: usize,
       pub internal_only: bool,  // true = hidden from user
   }
   ```

3. **Log Reasoning Steps** (`src/telemetry/worker.rs`):
   ```rust
   pub enum TelemetryEvent {
       // ... existing variants
       ReasoningStep { step: usize, thought: String, confidence: f32 },
   }
   ```

4. **Update Config** (`config/config.yaml`):
   ```yaml
   resolver:
     reasoning_loop:
       enabled: true
       max_internal_steps: 5
       trigger_confidence_floor: 0.4
   ```

**Files to Modify:**
- `src/engine.rs` (reasoning loop)
- `src/types.rs` (ThoughtUnit)
- `src/telemetry/worker.rs` (reasoning step logging)
- `config/config.yaml` (reasoning_loop config)

---

### 3.3 Concurrency Stress Test (Three-Class Model)

**What to Implement:**
- Run parallel Silent Training workers alongside Inference threads
- Enforce priority: Inference > Interactive Training > Silent Batch > Maintenance
- Verify inference latency <200ms/token under load

**How to Implement:**

1. **Create Priority Scheduler** (`src/scheduler.rs` - already exists, extend):
   ```rust
   pub enum TaskPriority {
       Inference,        // Highest: user-facing queries
       InteractiveTraining, // User-guided learning
       SilentBatch,       // Background corpus ingestion
       Maintenance,       // Lowest: pruning, compaction
   }
   ```

2. **Add Latency Monitor** (`src/telemetry/latency.rs`):
   - Track p50, p95, p99 latency per priority class
   - Alert when inference latency exceeds 200ms

**Files to Create:**
- `src/telemetry/latency.rs`
- `src/bin/concurrency_stress.rs`

**Files to Modify:**
- `src/scheduler.rs` (priority enforcement)
- `src/engine.rs` (thread-safe access patterns)

---

### 3.4 Core Infrastructure Drills

**Drill Coverage:**

1. **Telemetry Worker Drills:**
   ```rust
   fn telemetry_event_emission();
   fn telemetry_id_propagation();
   fn telemetry_high_event_rate();
   fn telemetry_channel_full();
   ```

2. **Reasoning Loop Drills:**
   ```rust
   fn reasoning_loop_complex_logic();
   fn reasoning_loop_low_confidence();
   fn reasoning_loop_max_steps();
   fn reasoning_loop_infinite_prevention();
   ```

3. **Concurrency Drills:**
   ```rust
   fn concurrency_inference_preemption();
   fn concurrency_priority_inversion();
   fn concurrency_all_saturated();
   fn concurrency_deadlock_recovery();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `tests/integration.rs` (integration tests)

---

## Phase 4: LLM-Like Core

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 3 complete (Telemetry required for debugging)

**Objective:** Transform SPSE into an **Autonomous Personal Intelligence** with LLM-like fluidity. Auto-Mode inference, controlled creativity, and style resonance work together.

**Constraint:** MVP is **Auto-Mode Only** (no user toggles). Creativity is internally gated (10-20% drift).

### 4.1 Auto-Mode Inference Engine

**What to Implement:**
- System infers intent, tone, and creativity level dynamically
- No UI toggles or API parameters for `mode`, `temperature`, or `profile`
- Full `RuntimeProfile` inference from query + history

**How to Implement:**

1. **Create Auto-Profile Inferrer** (`src/layers/auto_profile.rs`):
   ```rust
   pub struct RuntimeProfile {
       pub intent: IntentKind,
       pub tone: ToneKind,
       pub creativity_level: f32,
       pub style_anchor_id: Option<Uuid>,
   }
   
   pub enum ToneKind {
       Empathetic,
       Direct,
       Exploratory,
       NeutralProfessional,
   }
   ```

2. **Update Intent Detector** (`src/layers/intent.rs`):
   - Modify `classify()` to return `RuntimeProfile`
   - Add `infer_auto_profile()` method

3. **Remove External Control** (`src/api.rs`):
   - Remove `mode`, `temperature`, `top_p` from request schema
   - All requests map to Auto-Mode

**Files to Create:**
- `src/layers/auto_profile.rs`

**Files to Modify:**
- `src/layers/intent.rs` (return RuntimeProfile)
- `src/api.rs` (remove mode/temp params)
- `src/config/mod.rs` (AutoInferenceConfig)
- `config/config.yaml` (auto_inference section)

---

### 4.2 Creative Spark & Stochastic Walks (MERGED)

**What to Implement:**
- Force 15% stochastic drift in candidate selection for human-like variability
- Anchor validation gate prevents factual corruption
- Stochastic semantic walks for metaphor generation (brainstorm mode)
- **MERGED:** Both modify `router.rs` traversal logic

**How to Implement:**

1. **Add Traversal Mode** (`src/layers/router.rs`):
   ```rust
   pub enum TraversalMode {
       GreedyNearest,       // Default
       StochasticWalk,      // Random walk for brainstorm
       DriftFloor,          // 15% non-greedy sampling
   }
   
   impl Router {
       pub fn sample_with_drift_floor(
           candidates: &[ScoredCandidate],
           floor_ratio: f32,
       ) -> Option<ScoredCandidate>;
       
       pub fn traverse_stochastic_walk(
           &self,
           start_position: Vec3,
           depth: usize,
           temperature: f32,
       ) -> Vec3;
   }
   ```

2. **Add Anchor Validation Gate** (`src/layers/resolver.rs`):
   ```rust
   impl FineResolver {
       /// Validate sampled candidate against Core anchors
       pub fn validate_against_anchors(
           candidate: &Unit,
           anchors: &[Unit],
       ) -> AnchorValidationResult {
           // Check if candidate contradicts any Core anchor
           // Immutable anchor types: mathematical_constant, user_identity, verified_code_syntax
           // If contradiction -> reject and re-sample greedily
       }
   }
   ```

3. **Wire into Engine** (`src/engine.rs`):
   - Replace greedy selection with `sample_with_drift_floor()`
   - Apply anchor validation after sampling
   - Re-sample greedily if anchor validation fails

4. **Update Config** (`config/config.yaml`):
   ```yaml
   resolver:
     global_stochastic_floor: 0.15  # Enforced 15% non-greedy sampling
     max_factual_drift_tolerance: 0.05  # Max deviation on anchored facts
     immutable_anchor_types:
       - "mathematical_constant"
       - "user_identity"
       - "verified_code_syntax"
   ```

**Files to Modify:**
- `src/layers/router.rs` (drift sampling)
- `src/layers/resolver.rs` (anchor validation)
- `src/engine.rs` (wire drift + validation)
- `config/config.yaml` (resolver.stochastic_floor)

---

### 4.3 Style Anchor Resonance (Emergent Tone)

**What to Implement:**
- Dynamic retrieval of "Style Anchors" (exemplar texts) to bias vocabulary selection
- Replace rule-based tone weights with semantic resonance
- Session-persistent style state

**How to Implement:**

1. **Add Style Anchor Support** (`src/memory/store.rs`):
   ```rust
   pub struct StyleAnchor {
       pub id: Uuid,
       pub name: String,           // "Shakespeare", "Legal", "Noir", "Academic"
       pub exemplar_text: String,
       pub style_vector: Vec<f32>,  // Semantic embedding of style
   }
   
   impl MemoryStore {
       pub fn ingest_style_anchor(&mut self, anchor: StyleAnchor);
       pub fn retrieve_style_anchor(&self, tone: ToneKind) -> Option<StyleAnchor>;
   }
   ```

2. **Add Style Resonance Scoring** (`src/layers/search.rs`):
   ```rust
   impl CandidateScorer {
       /// Add style resonance component to 7D scoring
       pub fn score_with_style_resonance(
           &self,
           candidate: &Unit,
           style_anchor: &StyleAnchor,
       ) -> f32 {
           let resonance = 1.0 - cosine_distance(&candidate.embedding, &style_anchor.style_vector);
           // Boost candidates close to active style anchor
           resonance
       }
   }
   ```

3. **Persist Style State** (`src/layers/context.rs`):
   ```rust
   pub struct SessionStyleState {
       pub active_style_anchor_id: Option<Uuid>,
       pub decay_rate: f32,  // 0.0 = permanent for session
   }
   ```

4. **Ingest Default Style Anchors** (`data/style_anchors.json`):
   - 50 style exemplars covering common tones
   - Empathetic, Direct, Exploratory, Academic, Noir, Legal, etc.

**Files to Create:**
- `data/style_anchors.json`

**Files to Modify:**
- `src/memory/store.rs` (style anchor storage)
- `src/layers/search.rs` (style resonance scoring)
- `src/layers/context.rs` (session style state)
- `config/config.yaml` (style_anchors config)

---

### 4.4 LLM-Like Core Drills

**Drill Coverage:**

1. **Auto-Mode Inference Drills:**
   ```rust
   // Happy path: Sad query triggers empathetic tone
   fn auto_mode_empathetic_tone();
   // Happy path: Urgent query triggers direct tone
   fn auto_mode_direct_tone();
   // Happy path: Brainstorm triggers exploratory tone
   fn auto_mode_exploratory_tone();
   // Edge case: Mixed signals (sad + urgent)
   fn auto_mode_mixed_signals();
   // Failure: No tone match (fallback to neutral)
   fn auto_mode_fallback_neutral();
   ```

2. **Creative Drift Safety Drills:**
   ```rust
   // Happy path: 15% drift on factual query produces synonym variation
   fn creative_drift_factual_synonym();
   // Happy path: Math query "2+2" stays "4" despite drift
   fn creative_drift_math_anchor();
   // Edge case: Drift suggests wrong answer, anchor blocks
   fn creative_drift_anchor_blocks();
   // Failure: All anchors corrupted (should never happen)
   fn creative_drift_anchor_corruption();
   // Stress: High drift on many anchors
   fn creative_drift_many_anchors();
   ```

3. **Style Consistency Drills:**
   ```rust
   // Happy path: Noir style maintained across 500 words
   fn style_consistency_noir();
   // Happy path: Academic style for technical query
   fn style_consistency_academic();
   // Edge case: Style conflict (query tone != style anchor)
   fn style_consistency_conflict();
   // Failure: Style anchor missing
   fn style_consistency_missing_anchor();
   ```

4. **Stochastic Walk Drills:**
   ```rust
   // Happy path: Walk produces metaphorically related result
   fn stochastic_walk_metaphor();
   // Happy path: Walk depth 3 reaches distant concept
   fn stochastic_walk_depth();
   // Edge case: Walk returns to start (loop)
   fn stochastic_walk_loop();
   // Failure: Empty neighborhood during walk
   fn stochastic_walk_empty_neighborhood();
   ```

5. **Reasoning Loop Drills:**
   ```rust
   // Happy path: Complex logic solved in 3 steps
   fn reasoning_loop_complex_logic();
   // Happy path: Low confidence triggers loop
   fn reasoning_loop_low_confidence();
   // Edge case: Max steps reached without solution
   fn reasoning_loop_max_steps();
   // Failure: Infinite loop prevention
   fn reasoning_loop_infinite_prevention();
   // Stress: Very complex problem (many steps)
   fn reasoning_loop_complex_problem();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `src/drill_lib.rs` (drill implementations)
- `tests/integration.rs` (integration tests)

---

## Phase 5: Retrieval & Optimization

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 4 complete

### 5.1 Multi-Engine Consensus & Structured Parsing

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

3. **Add Format Adapters** (`src/document.rs`):
   - `HuggingFaceRowAdapter`: Parse dataset rows
   - `WikipediaXmlAdapter`: Extract articles with section hierarchy

**Files to Modify:**
- `src/layers/retrieval.rs` (multi-engine)
- `src/layers/merge.rs` (consensus)
- `src/document.rs` (adapters)

---

### 5.2 Config Sweeping & Benchmarking

**What to Implement:**
- Automate parameter sweeping on 100MB corpus
- Output Pareto frontier graphs (Latency vs Pollution)
- Optimize SpatialGrid index rebuild cadence

**How to Implement:**

1. **Extend Config Sweep Harness** (`src/bin/test_harness.rs`):
   - Sweep `global_stochastic_floor` (0.10, 0.15, 0.20, 0.25)
   - Sweep `reasoning_loop_max_steps` (3, 5, 7)
   - Sweep `spatial_cell_size` (2.0, 4.0, 6.0, 8.0)

**Files to Modify:**
- `src/bin/test_harness.rs` (extend)
- `scripts/analyze_results.py` (Pareto analysis)

---

### 5.3 Retrieval & Optimization Drills

**Drill Coverage:**

1. **Multi-Engine Aggregation Drills:**
   ```rust
   fn multi_engine_agreement();
   fn multi_engine_disagreement();
   fn multi_engine_all_unavailable();
   ```

2. **Format Adapter Drills:**
   ```rust
   fn adapter_huggingface_row();
   fn adapter_wikipedia_xml();
   fn adapter_malformed_json();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `tests/integration.rs` (integration tests)

---

## Phase 6: User Interface

**Estimated Duration:** 3-4 weeks  
**Dependencies:** Phase 5 complete

**Objective:** Deliver user-facing interface with Auto-Mode only (no toggles). OpenAI-compatible API enables LLM replacement for existing clients.

### 6.1 Web UI (Auto-Mode Interface)

**What to Implement:**
- Web UI connecting to SPSE API
- **DELETE Mode Toggles** - Replace with "Auto-Mode Active" indicator
- Real-time Layer 20 trace visualization
- Intent breakdown display

**How to Implement:**

1. **Create Next.js Project** (`web-ui/`):
   - `AutoModeIndicator.tsx` - Shows active inferred profile
   - `TraceVisualization.tsx` - WebSocket telemetry stream
   - `IntentBreakdown.tsx` - Hybrid blend score display

**Files to Create:**
- `web-ui/` directory with Next.js project

---

### 6.2 OpenAI-Compatible API Layer (Auto-Mode Locked)

**What to Implement:**
- Full OpenAI Chat Completions API compatibility for LLM replacement
- Model selection maps to SPSE profiles
- System prompt handling via L6 Context Manager
- Streaming SSE output for token-by-token responses
- **Auto-Mode locked:** No temperature/mode params accepted

**How to Implement:**

1. **Create OpenAI API adapter** (`src/api/openai_compat.rs`):
   ```rust
   pub struct ChatCompletionRequest {
       pub model: String,           // Maps to SPSE profile
       pub messages: Vec<Message>,
       // temperature, top_p IGNORED - Auto-Mode only
   }
   ```

2. **Add Streaming Output** (`src/api/streaming.rs`):
   - SSE format for token-by-token emission
   - Stop sequence handling

**Files to Create:**
- `src/api/openai_compat.rs`
- `src/api/streaming.rs`

**Files to Modify:**
- `src/api.rs` (route OpenAI endpoints)

---

### 6.3 User Interface Drills

**Drill Coverage:**

1. **Auto-Mode Indicator Drills:**
   ```rust
   fn ui_auto_mode_indicator();
   fn ui_mode_toggle_disabled();
   fn ui_inferred_tone_display();
   ```

2. **OpenAI API Drills:**
   ```rust
   fn openai_chat_completion();
   fn openai_streaming();
   fn openai_temperature_ignored();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `tests/integration.rs` (integration tests)
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

**Drill Coverage for Multi-Engine Consensus - NEW:**

1. **Multi-Engine Aggregation Drills:**
   ```rust
   // Happy path: Two engines agree on result
   fn multi_engine_agreement();
   // Happy path: Three engines with consensus
   fn multi_engine_triple_consensus();
   // Edge case: Engines disagree (conflict resolution)
   fn multi_engine_disagreement();
   // Edge case: One engine timeout
   fn multi_engine_partial_timeout();
   // Failure: All engines unavailable
   fn multi_engine_all_unavailable();
   // Stress: Rapid parallel queries to all engines
   fn multi_engine_parallel_stress();
   ```

2. **Evidence Merge Drills:**
   ```rust
   // Happy path: Baseline policy (0.5·Trust + 0.3·Recency + 0.2·Agreement)
   fn evidence_merge_baseline_policy();
   // Edge case: Internal pseudo-source vs external web evidence
   fn evidence_merge_internal_vs_external();
   // Edge case: Conflict with equal scores
   fn evidence_merge_equal_conflict();
   // Failure: Empty evidence from all sources
   fn evidence_merge_all_empty();
   ```

3. **Format Adapter Drills:**
   ```rust
   // Happy path: HuggingFace row parsing
   fn adapter_huggingface_row();
   // Happy path: Wikipedia XML section hierarchy
   fn adapter_wikipedia_xml();
   // Edge case: Malformed JSON row
   fn adapter_malformed_json();
   // Edge case: Missing required fields
   fn adapter_missing_fields();
   // Failure: Invalid XML structure
   fn adapter_invalid_xml();
   ```

---

## Execution Timeline

| Phase | Duration | Start | Dependencies |
|-------|----------|-------|--------------|
| Phase 1: Creative Mode | 2-3 weeks | Week 1 | None |
| Phase 2: Drill Suite | 3-4 weeks | Week 4 | Phase 1 |
| Phase 3: Core Infrastructure | 3-4 weeks | Week 8 | Phase 2 |
| Phase 4: LLM-Like Core | 3-4 weeks | Week 12 | Phase 3 |
| Phase 5: Retrieval & Optimization | 3-4 weeks | Week 16 | Phase 4 |
| Phase 6: User Interface | 3-4 weeks | Week 20 | Phase 5 |

**Total Estimated Duration:** 20-24 weeks

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
- [ ] **Phase 1 Feature Drills pass:**
  - [ ] Creative mode drills: brainstorm/plan profiles, factual disabled, stochastic boundary, missing profile, rapid switching
  - [ ] Hybrid blend drills: weight calculation, heuristic NaN, memory empty, corrupted memory
  - [ ] Intent channel isolation drills: isolation happy, promotion boundary, core promotion blocked, high volume
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
- [ ] Telemetry captures all calculations with Session/Trace IDs
- [ ] Reasoning loop solves complex queries via silent thought steps
- [ ] Inference latency <200ms under concurrent load
- [ ] **Core Infrastructure Drills pass:**
  - [ ] Telemetry: event emission, ID propagation, high event rate, channel full
  - [ ] Reasoning loop: complex logic, low confidence, max steps, infinite prevention
  - [ ] Concurrency: inference preemption, priority inversion, all saturated, deadlock recovery

### Phase 4
- [ ] **Auto-Mode inference works:** intent, tone, creativity inferred from query + history
- [ ] **15% creative drift enforced:** stochastic floor applied to all candidate selection
- [ ] **Anchor validation gate blocks factual corruption:** Core anchors protected from drift
- [ ] **Style anchor resonance active:** tone consistency maintained across session
- [ ] **Stochastic walks enabled:** metaphor generation for brainstorm/creative intents
- [ ] **LLM-Like Core Drills pass:**
  - [ ] Auto-mode inference: empathetic tone, direct tone, exploratory tone, mixed signals, fallback neutral
  - [ ] Creative drift safety: factual synonym, math anchor, anchor blocks, anchor corruption, many anchors
  - [ ] Style consistency: noir, academic, conflict, missing anchor
  - [ ] Stochastic walk: metaphor, depth, loop, empty neighborhood

### Phase 5
- [ ] Multi-engine consensus improves retrieval quality
- [ ] Config sweep identifies optimal parameters for latency vs pollution tradeoff
- [ ] **Retrieval & Optimization Drills pass:**
  - [ ] Multi-engine aggregation: agreement, disagreement, all unavailable
  - [ ] Format adapters: HuggingFace row, Wikipedia XML, malformed JSON

### Phase 6
- [ ] User Interface provides accurate and intuitive user experience
- [ ] OpenAI-compatible API enables LLM replacement for existing clients
- [ ] Auto-Mode indicator displays correctly without mode toggles
- [ ] **User Interface Drills pass:**
  - [ ] Auto-mode indicator: displays correctly, toggle disabled, inferred tone display
  - [ ] OpenAI API: chat completion, streaming, temperature ignored

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

5. **Auto-Mode Tone Misclassification**
   - Risk: Query tone inferred incorrectly, leading to inappropriate response style
   - Mitigation: Multi-signal tone detection (keywords + intent + history)
   - Fallback: Default to NeutralProfessional tone on ambiguous signals

6. **Creative Drift Factual Corruption**
   - Risk: 15% stochastic drift selects candidate that contradicts Core anchor
   - Mitigation: Anchor validation gate re-samples greedily on contradiction
   - Fallback: Disable drift entirely for mathematical/identity queries

7. **Reasoning Loop Infinite Recursion**
   - Risk: Silent thought steps loop without reaching final answer
   - Mitigation: Max 5 steps enforced, confidence threshold check
   - Fallback: Force final answer with partial reasoning

8. **Style Anchor Drift Mid-Session**
   - Risk: Active style anchor changes unexpectedly during conversation
   - Mitigation: Session-persistent style state with decay_rate=0
   - Fallback: Lock style anchor after first 3 turns

9. **Stochastic Walk Semantic Disconnect**
   - Risk: Random walk reaches semantically unrelated concept
   - Mitigation: Temperature-weighted step selection, max depth 4
   - Fallback: Fall back to greedy nearest neighbor selection

10. **OpenAI API Incompatibility**
    - Risk: Subtle differences in API behavior break existing LLM integrations
    - Mitigation: Comprehensive API compatibility test suite against OpenAI spec
    - Fallback: Compatibility shims that emulate missing behaviors
