# SPSE Engine Pre-Production Execution Plan

**Document Version:** 1.4  
**Created:** March 14, 2026  
**Last Updated:** March 15, 2026  
**Priority Order:** LLM-Like Core > Infrastructure > Retrieval > Optimization > Interface  
**Target Hardware:** Edge Devices (2GB RAM / Dual-Core CPU)

**Architecture Mode:** Mode C (Unified Auto+Reasoning) - Dynamic reasoning triggered by confidence thresholds. System defaults to lightweight Auto-Only path, only allocating reasoning resources when necessary.

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
- `FineResolver::select_with_shaping()` implements intent shaping in `src/predictive/resolver.rs` (lines 81-165)
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

3. **Modify Resolver** (`src/predictive/resolver.rs`):
   - Add `apply_intent_shaping()` method that adjusts candidate selection based on intent profile
   - Implement temperature-based stochastic selection for creative mode
   - Add factual anchor preservation check (units with `trust_score > 0.8` are protected)

4. **Update Engine** (`src/engine.rs`):
   - Wire intent profile lookup into `resolve_candidates()` flow
   - Pass active profile to `FineResolver::resolve()`

**Files to Modify:**
- `src/config/mod.rs` (lines ~1100-1200 for adaptive behavior)
- `config/config.yaml` (lines 131-245)
- `src/predictive/resolver.rs` (full file)
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
- `IntentDetector::hybrid_blend()` method added to `src/classification/intent.rs` (lines 1292-1358)
- `IntentBlendReport` struct defined in `src/types.rs` (lines 332-358)
- `OutputDecoder::detect_drift()` and `detect_corruption()` methods in `src/predictive/output.rs` (lines 39-135)
- `DriftReport` and `CorruptionReport` structs for validation output

**What to Implement:**
- Validate runtime intent blending from heuristic classification + Intent-channel memory lookup
- Ensure creative outputs show semantic drift without factual corruption
- Verify Layer 17 split realization (Decoder + Engine-level shaping)

**How to Implement:**

1. **Create Intent Blend Validator** (`src/classification/intent.rs`):
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

3. **Implement Drift Detection** (`src/predictive/output.rs`):
   - Add `detect_semantic_drift()` comparing output to source candidates
   - Add `detect_factual_corruption()` checking against anchor units
   - Return warnings if drift exceeds tolerance or corruption detected

4. **Wire Shaping Logic** (`src/engine.rs`):
   - In `build_explain_trace()`, add intent blend breakdown to debug steps
   - Apply `plan`, `act`, `brainstorm`, `critique` shaping based on `IntentKind`

**Files to Modify:**
- `src/classification/intent.rs` (add validation)
- `src/telemetry/test_observer.rs` (add fields)
- `src/predictive/output.rs` (drift detection)
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

### 2.1 Layer-Specific Drill Suite (7MB Corpus) ✅

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
- `src/classification/intent.rs` → add drill test functions
- `src/predictive/output.rs` → add drill test functions
- `src/predictive/resolver.rs` → add drill test functions

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
cargo run --bin drill_harness -- --mode maintenance --corpus 7mb
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
- Heterogeneous corpus generator supports custom_training, runtime_html, plain_text, local_document formats
- Distribution: 40% custom_training, 30% runtime_html, 20% plain_text, 10% local_document
- Latency tracking with avg/max/p99 metrics and spike detection
- Pollution ceiling validation (<1% default)

**What to Implement:**
- End-to-end 7MB ingestion with interleaved queries and forced maintenance cycles
- Validate Layer 12 normalization for custom_training, runtime_html, plain_text, local_document
- Monitor latency spikes, pollution ceilings, snapshot consistency

**How to Implement:**

1. **Create Stress Drill Runner** (`src/bin/stress_drill.rs`):
   ```rust
   struct StressDrillConfig {
       corpus_size_mb: usize, // Default: 7MB for faster test execution
       source_types: Vec<SourceTypeId>,
       query_interval_ms: u64,
       maintenance_interval_sec: u64,
       max_latency_spike_ms: u64,
       pollution_ceiling_percent: f32,
   }
   ```

2. **Implement Heterogeneous Source Generator** (`scripts/heterogeneous_corpus.py`):
   - Generate mixed corpus using custom dataset generators (classification, reasoning, predictive, consistency)
   - Apply `ingestion_policies` config for each format (custom_training, runtime_html, plain_text, local_document)

3. **Add Latency Monitor** (`src/telemetry/latency_tracker.rs`):
   - Track per-layer latency during stress test
   - Flag spikes exceeding 2x baseline
   - Report to Layer 20 telemetry

4. **Wire Layer 12 Adapter** (`src/document.rs`):
   - Implement format-aware normalization for uploaded document formats (PDF, DOCX, plain text)
   - Apply `ingestion_policies` extraction rules per format

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
   // Stress: Large corpus training (7MB+)
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

## Phase 3: LLM-Like Core (HIGHEST PRIORITY) ✅ COMPLETE

**Estimated Duration:** 4-5 weeks  
**Actual Duration:** Completed
**Dependencies:** Phase 2 complete  
**Architecture Mode:** Mode C (Unified Auto+Reasoning)

**Objective:** Implement autonomous LLM-like behavior with dynamic reasoning. System defaults to lightweight Auto-Only path (~350MB RAM, <100ms latency), automatically triggering reasoning loop only when confidence is low. This achieves optimal resource usage on low-spec hardware while maintaining high capability for complex tasks.

### 3.1 Dynamic Reasoning Type (The "Thinking" Engine) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `ReasoningLoopConfig` added to `src/config/mod.rs` with `enabled`, `max_internal_steps`, `trigger_confidence_floor`, `exit_confidence_threshold`, `thought_channel`, `step_budget_tokens`
- `should_trigger_reasoning()` method added to `IntentDetector` in `src/classification/intent.rs`
- Confidence gating triggers reasoning when `confidence < 0.40` OR complex intent with moderate confidence
- `ThoughtUnit` type added to `src/types.rs` with `content`, `step`, `internal_only`, `confidence`, `created_at`
- `OutputType` enum added with `FinalAnswer(String)` and `SilentThought(String)`
- `execute_reasoning_loop()`, `generate_thought_unit()`, `ingest_silent_thought()`, `assess_confidence()` methods added to `src/engine.rs`
- Silent thoughts ingested into `MemoryChannel::Reasoning` to prevent Core memory pollution
- Config values in `config/config.yaml` under `auto_inference.reasoning_loop`

**How to Implement:**

1. **Add Confidence Gating in Layer 9** (`src/classification/intent.rs`):
   ```rust
   pub fn should_trigger_reasoning(&self, confidence: f32, intent: IntentKind) -> bool {
       confidence < self.config.reasoning_trigger_floor 
           || intent == IntentKind::ComplexLogic
   }
   ```

2. **Add Silent Thought Output** (`src/engine.rs`):
   ```rust
   pub enum OutputType {
       FinalAnswer(String),
       SilentThought(String),  // Internal reasoning step - not shown to user
   }
   
   impl Engine {
       pub fn resolve_with_dynamic_reasoning(&mut self, query: &str) -> OutputType {
           let initial_confidence = self.assess_confidence(query);
           
           if !self.should_trigger_reasoning(initial_confidence, self.current_intent) {
               return self.resolve_direct(query);  // Fast path: ~350MB, <100ms
           }
           
           // Reasoning path: temporarily allocate ~550MB
           self.execute_reasoning_loop(query, 0)
       }
       
       fn execute_reasoning_loop(&mut self, query: &str, step: usize) -> OutputType {
           if step >= self.config.max_internal_steps {
               return self.resolve_direct(query);
           }
           
           let thought = self.generate_thought_unit(query, step);
           self.ingest_silent_thought(&thought);
           
           let new_confidence = self.assess_confidence(query);
           if new_confidence >= self.config.exit_confidence_threshold {
               return self.resolve_direct(query);
           }
           
           self.execute_reasoning_loop(query, step + 1)
       }
   }
   ```

3. **Add Thought Unit Type** (`src/types.rs`):
   ```rust
   pub struct ThoughtUnit {
       pub content: String,
       pub step: usize,
       pub internal_only: bool,  // true = hidden from user
   }
   ```

**Files to Modify:**
- `src/classification/intent.rs` (confidence gating)
- `src/engine.rs` (dynamic reasoning loop)
- `src/types.rs` (ThoughtUnit)
- `config/config.yaml` (reasoning_loop config)

**Performance Guarantees:**
- Simple queries: ~350MB RAM, <100ms TTFT (no reasoning loop entered)
- Complex queries: ~550MB RAM, 500ms-2s TTFT (temporary spike during reasoning)

---

### 3.2 Controlled Creative Spark (15% Stochastic Floor) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `CreativeSparkConfig` added to `src/config/mod.rs` with `global_stochastic_floor`, `selection_temperature`, `anchor_protection_strictness`
- `select_with_creative_floor()` method added to `SemanticRouter` in `src/predictive/router.rs`
- 15% stochastic floor enforced using weighted random selection from top-K candidates
- Softmax-like weighting with configurable temperature for selection randomness
- `validate_against_anchors()` method added to `FineResolver` in `src/predictive/resolver.rs`
- Anchor validation gate blocks creative drift on high-trust anchors (math, identity, factual)
- Contradiction detection for negation patterns and semantic overlap
- Config values in `config/config.yaml` under `auto_inference.creative_spark`

**What to Implement:**
- Fixed **15% non-greedy sampling rate** enforced globally
- Subject to factual anchor validation - never drift on high-trust anchors
- Prevents robotic output while maintaining factual accuracy

**How to Implement:**

1. **Enforce Stochastic Floor in Router** (`src/predictive/router.rs`):
   ```rust
   impl Router {
       pub fn select_with_creative_floor(
           &self,
           candidates: &[Candidate],
           config: &CreativeSparkConfig,
       ) -> Candidate {
           // Force 15% probability mass onto non-greedy selection
           let mut rng = rand::thread_rng();
           let sample = rng.gen::<f32>();
           
           if sample < config.global_stochastic_floor {
               // Sample from top-K neighbors (not just argmax)
               self.weighted_random_select(candidates, config.temperature)
           } else {
               // Greedy selection
               candidates.iter().max_by_key(|c| c.score).unwrap().clone()
           }
       }
   }
   ```

2. **Add Anchor Validation Gate** (`src/predictive/resolver.rs`):
   ```rust
   fn validate_against_anchors(&self, candidate: &Candidate, anchors: &[Anchor]) -> bool {
       for anchor in anchors {
           if anchor.trust_score > self.config.anchor_protection_strictness {
               if candidate.contradicts(anchor) {
                   return false;  // Reject creative drift on high-trust anchors
               }
           }
       }
       true
   }
   ```

**Files to Modify:**
- `src/predictive/router.rs` (stochastic floor)
- `src/predictive/resolver.rs` (anchor validation)
- `config/config.yaml` (creative_spark config)

---

### 3.3 Internal Tone & Intent Inference ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `ToneInferenceConfig` added to `src/config/mod.rs` with `enabled`, `style_anchor_decay`, `urgency_threshold`, `sadness_threshold`, `technical_threshold`
- `ToneInferrer` struct added to `src/classification/intent.rs` with style anchors for each tone kind
- `ToneKind` enum added to `src/types.rs` with `NeutralProfessional`, `Empathetic`, `Direct`, `Technical`, `Casual`, `Formal`
- `StyleAnchor` struct added with `tone`, `embedding`, `keywords`, `decay_rate`
- Tone inference from input semantics: urgency detection, sadness detection, technical domain detection
- Priority: Urgency > Sadness > Technical > Casual/Formal > Neutral
- `style_resonance()` method calculates keyword overlap between candidate and style anchor
- Session persistence via `active_tone` field with configurable decay
- Config values in `config/config.yaml` under `auto_inference.tone_inference`

**What to Implement:**
- Tone is **inferred** from input semantics and conversation history
- **NOT a user setting** - system dynamically retrieves Style Anchors
- Layer 9 analyzes emotional markers (urgency, sadness) and domain context
- Layer 14 scoring biased toward units close to active Style Anchor

**How to Implement:**

1. **Add Tone Inferrer** (`src/classification/intent.rs`):
   ```rust
   pub struct ToneInferrer {
       style_anchors: HashMap<ToneKind, StyleAnchor>,
       decay_rate: f32,  // 0.0 = persist for session
   }
   
   impl ToneInferrer {
       pub fn infer_tone(&self, input: &str, history: &[Unit]) -> ToneKind {
           // Analyze emotional markers
           let urgency = self.detect_urgency(input);
           let sadness = self.detect_sadness(input);
           let technical = self.detect_technical_domain(input);
           
           if urgency > 0.7 { return ToneKind::Direct; }
           if sadness > 0.5 { return ToneKind::Empathetic; }
           if technical > 0.6 { return ToneKind::Technical; }
           ToneKind::NeutralProfessional
       }
   }
   ```

2. **Add Style Anchor Resonance** (`src/reasoning/search.rs`):
   ```rust
   fn score_style_resonance(&self, candidate: &Unit, anchor: &StyleAnchor) -> f32 {
       self.semantic_similarity(candidate.embedding, anchor.embedding)
   }
   ```

**Files to Modify:**
- `src/classification/intent.rs` (tone inference)
- `src/reasoning/search.rs` (style resonance scoring)
- `src/reasoning/context.rs` (session style state)
- `config/config.yaml` (style_anchors config)

---

### 3.4 Auto-Mode Enforcement (User Controls Removal) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `AutoModeConfig` added to `src/config/mod.rs` with `locked`, `indicator_label`, `ignore_mode_parameter`, `ignore_temperature_parameter`
- `config()` method added to `Engine` to expose config for API
- `/api/v1/status` endpoint added to `src/api.rs` returning auto-mode status
- `AutoModeStatus` struct returns `mode: "auto"`, `locked: bool`, `indicator: String`
- All external parameters ignored: `mode`, `temperature`, `reasoning_depth`, `creative_level`
- Engine operates in `Auto-Mode` exclusively
- Config values in `config/config.yaml` under `auto_inference.auto_mode`

**What to Implement:**
- Remove all external parameters: `mode`, `temperature`, `reasoning_depth`, `creative_level`
- Engine operates in `Auto-Mode` exclusively
- UI shows static "Auto-Intelligence Active" indicator

**How to Implement:**

1. **Strip Mode Fields from API** (`src/api.rs`):
   ```rust
   pub struct QueryRequest {
       pub query: String,
       pub context: Option<Vec<String>>,
       // REMOVED: mode, temperature, reasoning_depth, creative_level
   }
   ```

2. **Update Web UI** (`web-ui/`):
   - Remove `ModeToggle` component
   - Remove `ReasoningSwitch` component
   - Add `AutoModeIndicator.tsx` - static display

**Files to Modify:**
- `src/api.rs` (strip mode fields)
- `web-ui/components/` (remove toggles, add indicator)

---

### 3.5 LLM-Like Core Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- 7 new drill modes added to `DrillMode` enum in `src/drill_lib.rs`:
  - `DynamicReasoning` - Tests confidence gating and reasoning loop triggering
  - `SilentThought` - Tests thought unit creation and internal_only flag
  - `CreativeSpark` - Tests 15% stochastic floor with weighted random selection
  - `AnchorValidation` - Tests anchor contradiction detection and blocking
  - `ToneInference` - Tests urgency, sadness, technical tone detection
  - `StyleResonance` - Tests style anchor keyword overlap scoring
  - `AutoModeEnforcement` - Tests locked mode and parameter ignoring
- Corpus generators for each drill mode added
- Drill execution functions implemented with happy path, edge case, and failure mode tests
- All drills use config structs from `src/config/mod.rs` for threshold validation
- Metrics tracked: `reasoning_triggered`, `non_greedy_ratio`, `urgency_detected`, `resonance_score`, etc.

**Drill Coverage:**

1. **Dynamic Reasoning Drills:**
   ```rust
   // Happy path: Simple query skips reasoning loop
   fn reasoning_skips_simple_query();
   // Happy path: Complex query triggers reasoning
   fn reasoning_triggers_complex_query();
   // Happy path: Low confidence triggers reasoning
   fn reasoning_triggers_low_confidence();
   // Edge case: Max steps reached without solution
   fn reasoning_max_steps_reached();
   // Failure: Infinite loop prevention
   fn reasoning_infinite_prevention();
   // Stress: Memory allocation/deallocation during reasoning
   fn reasoning_memory_management();
   ```

2. **Creative Spark Drills:**
   ```rust
   // Happy path: 15% drift produces synonym variation
   fn creative_drift_synonym();
   // Happy path: Math query "2+2" stays "4" despite drift
   fn creative_drift_math_anchor();
   // Edge case: Drift suggests wrong answer, anchor blocks
   fn creative_drift_anchor_blocks();
   // Failure: All anchors corrupted (should never happen)
   fn creative_drift_anchor_corruption();
   ```

3. **Tone Inference Drills:**
   ```rust
   // Happy path: Sad query triggers empathetic tone
   fn tone_empathetic_detection();
   // Happy path: Urgent query triggers direct tone
   fn tone_direct_detection();
   // Happy path: Technical query triggers technical tone
   fn tone_technical_detection();
   // Edge case: Mixed signals (sad + urgent)
   fn tone_mixed_signals();
   // Failure: No tone match (fallback to neutral)
   fn tone_fallback_neutral();
   ```

4. **Auto-Mode Enforcement Drills:**
   ```rust
   // Happy path: Mode parameter ignored
   fn auto_mode_parameter_ignored();
   // Happy path: Temperature parameter ignored
   fn auto_mode_temperature_ignored();
   // Happy path: Auto-Mode indicator displays
   fn auto_mode_indicator_displays();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `src/drill_lib.rs` (drill implementations)
- `tests/integration.rs` (integration tests)

---

## Phase 4: Core Infrastructure ✅ COMPLETE

**Estimated Duration:** 2-3 weeks  
**Actual Duration:** Completed
**Dependencies:** Phase 3 complete (LLM-Like Core requires telemetry for debugging)

**Objective:** Build foundational infrastructure supporting LLM-like features. Telemetry provides debugging visibility for reasoning loops, Latency monitoring ensures low-spec performance, Dynamic memory allocation enables Mode C efficiency.

### 4.1 Global Logging Engine & Trace Visualization (Layer 20) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `TelemetryWorker` async worker with batching and backpressure in `src/telemetry/worker.rs`
- `HotStore` SQLite hot store for real-time queries in `src/telemetry/hot_store.rs`
- `AppendOnlyLog` cold storage for long-term archival
- `TraceContext` with `SessionId` and `TraceId` in `src/telemetry/trace.rs`
- `TelemetryEvent::ReasoningStep` captures step, thought, confidence
- All events include session_id and trace_id for correlation

**What to Implement:**
- Non-blocking async worker emitting structured JSON events
- Hot logs in SQLite for real-time UI, cold logs in append-only files
- **Reasoning trace logging** - capture `reasoning_steps_taken` and `confidence_trajectory`
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
       ReasoningStep { step: usize, thought: String, confidence: f32 },  // NEW
   }
   ```

2. **Add Session/Trace IDs** (`src/telemetry/trace.rs`):
   - Generate `session_id` on engine init
   - Generate `trace_id` per query
   - Include in all telemetry events

**Files to Create:**
- `src/telemetry/worker.rs`
- `src/telemetry/hot_store.rs`
- `src/telemetry/trace.rs`

**Files to Modify:**
- `src/telemetry/mod.rs` (exports)
- `src/engine.rs` (wire worker)
- `config/config.yaml` (Layer 20 config expansion)

---

### 4.2 Latency Monitoring & Dynamic Memory Allocation ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `LatencyMonitor` with p50/p95/p99 tracking in `src/telemetry/latency.rs`
- `LatencySummary` for reporting latency metrics
- `LatencyTimer` RAII guard for automatic timing
- `DynamicMemoryAllocator` for reasoning buffer management in `src/memory/dynamic.rs`
- `ThoughtBuffer` RAII guard for automatic deallocation
- `MemoryStats` for tracking usage, limits, and buffer counts
- Config-driven thresholds: 200ms alert, 350MB base, 550MB max

**What to Implement:**
- Track p50, p95, p99 latency per priority class
- **Dynamic memory allocation** for Mode C efficiency
- Alert when inference latency exceeds 200ms on low-spec devices

**How to Implement:**

1. **Add Latency Monitor** (`src/telemetry/latency.rs`):
   ```rust
   pub struct LatencyMonitor {
       p50: AtomicU64,
       p95: AtomicU64,
       p99: AtomicU64,
       alert_threshold_ms: u64,  // 200ms for low-spec
   }
   ```

2. **Add Dynamic Memory Allocator** (`src/memory/dynamic.rs`):
   ```rust
   pub struct DynamicMemoryAllocator {
       base_limit_mb: usize,    // 350MB idle
       max_limit_mb: usize,     // 550MB during reasoning
       current_usage: AtomicUsize,
   }
   
   impl DynamicMemoryAllocator {
       pub fn allocate_thought_buffer(&mut self) -> Option<ThoughtBuffer> {
           // Allocate only when reasoning triggered
           if self.current_usage.load() + THOUGHT_BUFFER_SIZE <= self.max_limit_mb {
               self.current_usage.fetch_add(THOUGHT_BUFFER_SIZE);
               Some(ThoughtBuffer::new())
           } else {
               None
           }
       }
       
       pub fn deallocate_thought_buffer(&mut self) {
           // Release immediately after reasoning completes
           self.current_usage.fetch_sub(THOUGHT_BUFFER_SIZE);
       }
   }
   ```

**Files to Create:**
- `src/telemetry/latency.rs`
- `src/memory/dynamic.rs`

**Files to Modify:**
- `src/engine.rs` (dynamic allocation hooks)
- `config/config.yaml` (memory_budgets config)

---

### 4.3 Core Infrastructure Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- Added 9 Phase 4 drill modes to `DrillMode` enum in `src/drill_lib.rs`
- `TelemetryEmission`, `TelemetryReasoningStep`, `TelemetryBackpressure` drills
- `LatencyNormalLoad`, `LatencyReasoningSpike`, `LatencyThresholdExceeded` drills
- `DynamicMemoryAllocate`, `DynamicMemoryRelease`, `DynamicMemoryLimit` drills
- All drills test happy path, edge cases, failure modes, and stress scenarios
- Wired into `run_drill()` and `generate_drill_corpus()` functions

**Drill Coverage:**

1. **Telemetry Worker Drills:**
   ```rust
   // Happy path: Event emitted at layer boundary
   fn telemetry_event_emission();
   // Happy path: Reasoning step logged
   fn telemetry_reasoning_step();
   // Edge case: High event rate (backpressure)
   fn telemetry_high_event_rate();
   // Failure: Worker channel full
   fn telemetry_channel_full();
   ```

2. **Latency Monitor Drills:**
   ```rust
   // Happy path: p95 < 200ms under normal load
   fn latency_normal_load();
   // Edge case: Latency spike during reasoning
   fn latency_reasoning_spike();
   // Failure: Latency exceeds threshold
   fn latency_threshold_exceeded();
   ```

3. **Dynamic Memory Drills:**
   ```rust
   // Happy path: Memory allocated on reasoning trigger
   fn dynamic_memory_allocate_on_reasoning();
   // Happy path: Memory released after reasoning
   fn dynamic_memory_release_after_reasoning();
   // Edge case: Memory limit reached
   fn dynamic_memory_limit_reached();
   // Stress: Repeated reasoning cycles
   fn dynamic_memory_repeated_cycles();
   ```

**Files to Modify:**
- `src/bin/drill_harness.rs` (add drill modes)
- `tests/integration.rs` (integration tests)

---

## Phase 5: Retrieval & Optimization ✅ COMPLETE

**Estimated Duration:** 2-3 weeks  
**Actual Duration:** Completed
**Dependencies:** Phase 4 complete

**Objective:** Enhance retrieval quality and optimize parameters for low-spec hardware. Multi-engine consensus improves answer accuracy, Config sweeping identifies optimal settings for Mode C efficiency.

### 5.1 Multi-Engine Consensus & Structured Parsing ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `MultiEngineAggregator` struct added to `src/reasoning/retrieval.rs`
- `EngineResult` and `ConsensusDocument` structs for multi-engine results
- `StructuredParser` for runtime-retrieved HTML content
- `MultiEngineConfig` added to `src/config/mod.rs` (lines 732-767)
- Config validation for consensus thresholds (lines 2541-2568)

**Key Features:**
- Parallel querying via SearxNG (runtime web retrieval)
- Consensus scoring with trust/agreement/diversity weights
- Structured parsing for runtime-retrieved HTML content
- Configurable timeout and max engines per query

---

### 5.2 Config Sweeping & Benchmarking ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `ConfigSweepConfig` struct added to `src/config/mod.rs` (lines 769-810)
- Config fields for sweep parameter ranges (reasoning trigger, steps, stochastic floor, memory)
- Validation for sweep parameters (lines 2569-2580)
- YAML configuration in `config/config.yaml` (lines 686-709)

**Key Parameters:**
- `reasoning_trigger_floor_values`: [0.30, 0.40, 0.50]
- `max_internal_steps_values`: [2, 3, 5]
- `global_stochastic_floor_values`: [0.10, 0.15, 0.20]
- `memory_limit_mb_values`: [350, 450, 550]
- `latency_target_ms`: 200
- `pollution_ceiling_percent`: 1.0

---

### 5.3 Retrieval & Optimization Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- Phase 5 drill modes added to `DrillMode` enum in `src/drill_lib.rs` (lines 121-133)
- Corpus generators for each drill mode (lines 1755-1799)
- Drill implementations for multi-engine and config sweep testing (lines 1801-1997)
- Drill harness updated in `src/bin/drill_harness.rs` (lines 121-127, 163-169, 274-280)

**Drill Coverage:**

1. **Multi-Engine Drills:**
   - `MultiEngineConsensus`: Happy path consensus agreement
   - `MultiEngineDisagreement`: Edge case engine disagreement
   - `MultiEngineUnavailable`: Failure mode all engines unavailable
   - `StructuredParsing`: Parsing validation for runtime-retrieved content formats

2. **Config Sweep Drills:**
   - `ConfigSweepPareto`: Pareto frontier identification
   - `ConfigSweepNoOptimal`: No optimal config found handling

---

## Phase 6: User Interface ✅ COMPLETE

**Estimated Duration:** 2-3 weeks  
**Actual Duration:** Completed
**Dependencies:** Phase 5 complete

**Objective:** Deliver user-facing interface with Auto-Mode only (no toggles). OpenAI-compatible API enables LLM replacement for existing clients.

### 6.1 Web UI (Auto-Mode Interface) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- Next.js 14 project created in `web-ui/` directory with TypeScript and TailwindCSS
- `AutoModeIndicator.tsx` component displays static "Auto-Intelligence Active" badge with animated pulse
- `IntentBreakdown.tsx` component shows detected intent and confidence percentage
- `page.tsx` main chat interface with message history, input form, and inferred tone display
- `api/chat/route.ts` API proxy to SPSE OpenAI-compatible endpoint
- No mode toggles - Auto-Mode indicator only

**Key Features:**
- Responsive chat interface with message bubbles
- Real-time intent and confidence display
- Inferred tone indicator (Empathetic, Direct, Technical, NeutralProfessional)
- Streaming support ready for SSE responses
- Clean, modern UI with TailwindCSS styling

---

### 6.2 OpenAI-Compatible API Layer (Auto-Mode Locked) ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- `src/api/openai_compat.rs` module with full OpenAI Chat Completions API compatibility
- `ChatCompletionRequest` and `ChatCompletionResponse` structs matching OpenAI spec
- `POST /v1/chat/completions` endpoint for chat completions
- `GET /v1/models` endpoint listing available models (all map to spse-auto)
- Streaming SSE output via `ChatCompletionChunk` and `StreamChoice` structs
- All parameters ignored in Auto-Mode: `model`, `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`
- SPSE-specific response fields: `intent`, `confidence`, `tone`
- Router integration in `src/api.rs` with `openai_router()` function

**Key Features:**
- Full OpenAI API compatibility for drop-in LLM replacement
- Streaming SSE support with `data: {...}` format and `[DONE]` marker
- Model names ignored - all resolve to spse-auto in Auto-Mode
- Temperature and other parameters ignored - engine uses internal control
- Intent and tone metadata in responses

---

### 6.3 User Interface Drills ✅

**Status:** IMPLEMENTED

**Implementation Details:**
- 7 Phase 6 drill modes added to `DrillMode` enum in `src/drill_lib.rs`:
  - `UiAutoModeIndicator` - Tests indicator display and locked status
  - `UiInferredTone` - Tests tone inference display (Empathetic, Direct, Technical)
  - `UiModeParameterIgnored` - Tests mode parameter ignoring in Auto-Mode
  - `OpenAiChatCompletion` - Tests OpenAI chat completion request/response
  - `OpenAiStreaming` - Tests SSE streaming format and [DONE] marker
  - `OpenAiTemperatureIgnored` - Tests temperature parameter ignoring
  - `OpenAiModelIgnored` - Tests model parameter ignoring (maps to spse-auto)
- Corpus generators for each drill mode
- Drill implementations for happy path, edge case, and failure mode categories
- Metrics tracked: `indicator_displayed`, `locked`, `tone_inferred`, `streaming_enabled`, etc.

**Drill Coverage:**

1. **Auto-Mode Indicator Drills:**
   ```rust
   // Happy path: Auto-Mode indicator displays
   fn ui_auto_mode_indicator();
   // Happy path: Inferred tone displayed
   fn ui_inferred_tone_display();
   // Edge case: Mode parameter ignored
   fn ui_mode_parameter_ignored();
   ```

2. **OpenAI API Drills:**
   ```rust
   // Happy path: Chat completion request
   fn openai_chat_completion();
   // Happy path: Streaming SSE output
   fn openai_streaming();
   // Edge case: Temperature param ignored (Auto-Mode)
   fn openai_temperature_ignored();
   // Edge case: Model param ignored (Auto-Mode)
   fn openai_model_ignored();
   ```

---

## Execution Timeline

| Phase | Duration | Start | Dependencies | Status |
|-------|----------|-------|--------------|--------|
| Phase 1: Creative Mode | 2-3 weeks | Week 1 | None | ✅ COMPLETE |
| Phase 2: Drill Suite | 3-4 weeks | Week 4 | Phase 1 | ✅ COMPLETE |
| Phase 3: LLM-Like Core | 4-5 weeks | Week 8 | Phase 2 | ✅ COMPLETE |
| Phase 4: Core Infrastructure | 3-4 weeks | Week 13 | Phase 3 | ✅ COMPLETE |
| Phase 5: Retrieval & Optimization | 2-3 weeks | Week 17 | Phase 4 | ✅ COMPLETE |
| Phase 6: User Interface | 2-3 weeks | Week 20 | Phase 5 | ✅ COMPLETE |

**Total Estimated Duration:** 19-24 weeks

**Low-Spec Performance Targets:**
- Simple queries: ~350MB RAM, <100ms TTFT
- Complex queries: ~550MB RAM, 500ms-2s TTFT
- Baseline efficiency: 90% of queries run in lightweight mode

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

### Phase 3
- [ ] **Dynamic reasoning triggers correctly:** confidence < 0.40 or complex intent
- [ ] **15% stochastic floor enforced** with anchor validation
- [ ] **Tone inference works:** empathetic, direct, technical, neutral
- [ ] **Auto-Mode enforced:** no user toggles, parameters ignored
- [ ] **Low-spec performance:** ~350MB idle, ~550MB during reasoning
- [ ] **LLM-Like Core Drills pass:**
  - [ ] Dynamic reasoning: skip simple, trigger complex, max steps, infinite prevention
  - [ ] Creative spark: synonym drift, math anchor, anchor blocks
  - [ ] Tone inference: empathetic, direct, technical, mixed signals, fallback
  - [ ] Auto-mode: parameter ignored, indicator displays

### Phase 4
- [x] Telemetry captures all calculations with Session/Trace IDs
- [x] Reasoning trace logging captures `reasoning_steps_taken` and `confidence_trajectory`
- [x] Dynamic memory allocation works: allocate on reasoning, release after
- [x] **Core Infrastructure Drills pass:**
  - [x] Telemetry: event emission, reasoning step, high rate, channel full
  - [x] Latency: normal load, reasoning spike, threshold exceeded
  - [x] Dynamic memory: allocate, release, limit reached, repeated cycles

### Phase 5
- [x] Multi-engine consensus improves retrieval quality
- [x] Config sweep identifies optimal parameters for low-spec hardware
- [x] **Retrieval & Optimization Drills pass:**
  - [x] Multi-engine: consensus agreement, disagreement, all unavailable
  - [x] Config sweep: Pareto frontier, no optimal

### Phase 6
- [ ] User Interface provides accurate and intuitive user experience
- [ ] OpenAI-compatible API enables LLM replacement for existing clients
- [ ] Auto-Mode indicator displays correctly without mode toggles
- [ ] **User Interface Drills pass:**
  - [ ] Auto-mode indicator: displays correctly, inferred tone, parameter ignored
  - [ ] OpenAI API: chat completion, streaming, temperature ignored

---

## Configuration Schema (Mode C - Unified Auto+Reasoning)

```yaml
engine:
  mode: "auto_unified"  # LOCKED. No user override.
  
auto_inference:
  # Dynamic Reasoning Configuration
  reasoning_loop:
    enabled: true
    trigger_confidence_floor: 0.40  # Below this, system starts "thinking"
    max_internal_steps: 3           # Cap loops to save CPU on low-spec
    exit_confidence_threshold: 0.60
    
  # Dynamic Tone Inference
  tone_inference:
    enabled: true
    style_anchor_decay: 0.0         # Persist tone for session
    
  # Controlled Creativity
  creative_spark:
    global_stochastic_floor: 0.15   # Enforce 15% drift
    anchor_protection_strictness: 0.95  # Never drift on high-trust anchors

memory:
  # Optimization for Low Spec
  dynamic_allocation:
    enable_thought_buffer_on_demand: true  # Don't allocate reasoning RAM until needed
    base_memory_limit_mb: 350
    max_memory_limit_mb: 550               # Allow spike during reasoning
```

---

## Risk Mitigation

1. **Dynamic Reasoning Infinite Loop**
   - Risk: Silent thought steps loop without reaching final answer
   - Mitigation: Max 3 steps enforced, confidence threshold check
   - Fallback: Force final answer with partial reasoning

2. **Creative Drift Factual Corruption**
   - Risk: 15% stochastic drift selects candidate that contradicts Core anchor
   - Mitigation: Anchor validation gate re-samples greedily on contradiction
   - Fallback: Disable drift entirely for mathematical/identity queries

3. **Auto-Mode Tone Misclassification**
   - Risk: Query tone inferred incorrectly, leading to inappropriate response style
   - Mitigation: Multi-signal tone detection (keywords + intent + history)
   - Fallback: Default to NeutralProfessional tone on ambiguous signals

4. **Memory Exhaustion on Low-Spec**
   - Risk: Reasoning loop allocates memory beyond device capacity
   - Mitigation: Dynamic memory allocation with hard limits (350MB base, 550MB max)
   - Fallback: Skip reasoning, return direct answer with lower confidence

5. **Telemetry Performance Overhead**
   - Risk: High-frequency events overwhelm low-spec devices
   - Mitigation: Async worker with batching, configurable sample rate
   - Fallback: Reduce sample rate to 0.1 if needed

6. **OpenAI API Incompatibility**
   - Risk: Subtle differences in API behavior break existing LLM integrations
   - Mitigation: Comprehensive API compatibility test suite against OpenAI spec
   - Fallback: Compatibility shims that emulate missing behaviors
