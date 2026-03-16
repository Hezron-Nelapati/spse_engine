# SPS Training Architecture — Comprehensive Coding Plan

**Created:** March 2026  
**Status:** Implementation Plan  
**Reference:** `docs/SPSE_ARCHITECTURE_V11.1.md` §11

---

## Table of Contents

1. [Overview & Execution Order](#1-overview--execution-order)
2. [Phase 1: Foundation — Shared Infrastructure](#2-phase-1-foundation--shared-infrastructure)
3. [Phase 2: Classification System Training](#3-phase-2-classification-system-training)
4. [Phase 3: Reasoning System Training](#4-phase-3-reasoning-system-training)
5. [Phase 4: Predictive System Training](#5-phase-4-predictive-system-training)
6. [Phase 5: Cross-System Consistency](#6-phase-5-cross-system-consistency)
7. [Phase 6: Dataset Generators](#7-phase-6-dataset-generators)
8. [Phase 7: Efficiency Optimizations](#8-phase-7-efficiency-optimizations)
9. [Phase 8: Training Sweep Harness](#9-phase-8-training-sweep-harness)
10. [Phase 9: Integration Testing](#10-phase-9-integration-testing)
11. [Phase 10: On-the-fly Learning](#11-phase-10-on-the-fly-learning)
12. [Phase 11: Hardening Improvements](#12-phase-11-hardening-improvements)
13. [Config Changes Summary](#13-config-changes-summary)
14. [File Manifest](#14-file-manifest)

---

## 1. Overview & Execution Order

### Dependency Graph

```
Phase 1 (Foundation)
  ├── Phase 2 (Classification Training)
  ├── Phase 3 (Reasoning Training)
  │     └── depends on Phase 2 (needs intent profiles)
  └── Phase 4 (Predictive Training)
        └── depends on Phase 3 (needs scored candidates)

Phase 5 (Consistency) → depends on Phases 2, 3, 4
Phase 6 (Dataset Generators) → can start in parallel with Phase 1
Phase 7 (Efficiency) → can start after Phase 2
Phase 8 (Sweep Harness) → depends on Phases 2, 3, 4
Phase 9 (Integration Tests) → depends on all phases
Phase 10 (On-the-fly Learning) → depends on Phase 4 (Word Graph) + Phase 3 (Evidence Merge)
Phase 11 (Hardening + Robustness + Edge Cases) → depends on Phases 2, 4, 5, 10
  ├── 11A (Semantic Probes) → depends on Phase 2 (Classification)
  ├── 11B (Micro-Validator + Lazy Gating) → depends on Phase 3 (Evidence Merge)
  ├── 11C (Adaptive Dims + Anchor Locking) → depends on Phase 4 (Word Graph)
  ├── 11D (TTG Lease Edges) → depends on Phase 10 (On-the-fly Learning)
  ├── 11E (Structural Feedback + Hysteresis) → depends on Phase 5 (Consistency)
  ├── 11F (POS Blending) → depends on Phase 4 (Word Graph, SemanticHasher)
  ├── 11G (Hub Management) → depends on Phase 4 (Word Graph) + 11D (EdgeStatus)
  ├── 11H (Context Bloom Filter) → depends on Phase 4 (Word Graph, edge context tags)
  ├── 11I (Silent Training Quarantine) → depends on Phase 2 (Classification Training)
  ├── 11J (Dynamic Hub Election) → depends on 11G (Hub Management)
  └── 11K (Feedback Queue) → depends on Phase 10 (On-the-fly Learning, feedback)
```

### Estimated Effort

| Phase | Files New | Files Modified | Estimated LOC | Priority |
|-------|-----------|---------------|---------------|----------|
| 1. Foundation | 1 | 4 | ~400 | P0 |
| 2. Classification Training | 1 | 4 | ~800 | P0 |
| 3. Reasoning Training | 2 | 3 | ~1200 | P0 |
| 4. Predictive Training | 1 | 3 | ~1000 | P0 |
| 5. Consistency | 1 | 2 | ~500 | P1 |
| 6. Dataset Generators | 4 | 2 | ~2000 | P0 |
| 7. Efficiency | 0 | 8 | ~600 | P2 |
| 8. Sweep Harness | 1 | 1 | ~400 | P1 |
| 9. Integration Tests | 1 | 1 | ~600 | P1 |
| 10. On-the-fly Learning | 0 | 3 | ~120 | P1 |
| 11. Hardening + Edge Cases | 6 | 12 | ~1900 | P0–P2 |
| **Total** | **16** | **~22** | **~9520** | |

---

## 2. Phase 1: Foundation — Shared Infrastructure

### Task 1.1: Training Loss Types (`src/types.rs`)

Add loss function result types used by all three system trainers.

```rust
// Add to src/types.rs

/// Loss function result for Classification System training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClassificationLoss {
    pub l_intent: f32,          // 1 - accuracy(predicted_intent, true_intent)
    pub l_tone: f32,            // 1 - accuracy(predicted_tone, true_tone)
    pub l_gate: f32,            // binary_cross_entropy(retrieval_triggered, retrieval_needed)
    pub l_class: f32,           // l_intent + 0.5 * l_tone + 0.3 * l_gate
    pub intent_accuracy: f32,
    pub tone_accuracy: f32,
    pub gate_accuracy: f32,
    pub sample_count: usize,
}

/// Loss function result for Reasoning System training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningLoss {
    pub l_ranking: f32,         // 1 - mean_reciprocal_rank
    pub l_merge_f1: f32,        // 1 - F1(predicted_conflicts, actual_conflicts)
    pub l_chain: f32,           // 1 - (chains_correct / chains_total)
    pub l_reason: f32,          // l_ranking + 0.3 * l_merge_f1 + 0.5 * l_chain
    pub mean_reciprocal_rank: f32,
    pub merge_f1: f32,
    pub chain_accuracy: f32,
    pub sample_count: usize,
}

/// Loss function result for Predictive System training
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictiveLoss {
    pub l_next_word: f32,       // -mean(log(P(correct_word | walked_edges)))
    pub l_spatial_energy: f32,  // layout_energy / num_edges
    pub l_edge_quality: f32,    // mean(edge_weight_variance_per_node)
    pub l_pred: f32,            // l_next_word + 0.1 * l_spatial_energy + 0.2 * l_edge_quality
    pub next_word_accuracy: f32,
    pub sequences_trained: usize,
    pub steps_trained: usize,
}

/// Cross-system consistency loss
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsistencyLoss {
    pub r1_violations: usize,   // High uncertainty but no retrieval
    pub r2_violations: usize,   // Social intent but reasoning ran
    pub r3_violations: usize,   // Factual intent but anchor contradicted
    pub r4_violations: usize,   // Creative intent but drift blocked
    pub l_consistency: f32,     // sum of violations / sample_count
    pub sample_count: usize,
}

/// Unified training report across all systems
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemTrainingReport {
    pub classification: Option<ClassificationLoss>,
    pub reasoning: Option<ReasoningLoss>,
    pub predictive: Option<PredictiveLoss>,
    pub consistency: Option<ConsistencyLoss>,
    pub total_duration_ms: u128,
    pub hardware_used: String,  // "cpu_only" | "cpu+gpu"
}
```

**Files modified:** `src/types.rs`  
**Tests:** Unit test asserting default values and serialization round-trip.

### Task 1.2: Training Example Extensions (`src/seed/mod.rs`)

Extend `TrainingExample` with new optional fields.

```rust
// Modify existing TrainingExample in src/seed/mod.rs
pub struct TrainingExample {
    // ... existing fields ...
    pub tone: Option<String>,             // NEW: tone label for classification training
    pub needs_retrieval: Option<bool>,    // NEW: retrieval gate ground truth
    pub correct_unit_id: Option<String>,  // NEW: for MRR evaluation in reasoning training
    pub sub_questions: Option<Vec<String>>, // NEW: for decomposition template learning
    pub unit_sequence: Option<Vec<UnitSequenceEntry>>, // NEW: for predictive training
    pub expected_classification: Option<ExpectedClassification>, // NEW: consistency
    pub expected_reasoning: Option<ExpectedReasoning>,           // NEW: consistency
    pub expected_prediction: Option<ExpectedPrediction>,         // NEW: consistency
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitSequenceEntry {
    pub content: String,
    pub level: String,       // "Word", "Phrase", etc.
    pub position: [f32; 3],  // initial spatial position
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedClassification {
    pub intent: String,
    pub tone: String,
    pub needs_retrieval: bool,
    pub min_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedReasoning {
    pub should_retrieve: bool,
    pub max_reasoning_steps: usize,
    pub correct_answer_contains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedPrediction {
    pub drift_allowed: bool,
    pub must_preserve_anchors: bool,
    pub expected_resolver_mode: String,
}
```

**Files modified:** `src/seed/mod.rs`  
**Tests:** Deserialization test from JSON fixture.

### Task 1.3: Config Additions (`src/config/mod.rs`, `config/config.yaml`)

Add training-specific config sections.

```rust
// Add to src/config/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationTrainingConfig {
    pub centroid_batch_size: usize,           // default: 500
    pub weight_sweep_iterations: usize,       // default: 10
    pub weight_sweep_dimensions: Vec<String>, // ["w_intent_hash", "w_tone_hash", ...]
    pub calibration_bins: usize,              // default: 10
    pub validation_split: f32,               // default: 0.1
    pub min_examples_per_intent: usize,      // default: 100
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningTrainingConfig {
    pub scoring_sweep_iterations: usize,      // default: 20
    pub decomposition_min_examples: usize,    // default: 50
    pub mrr_target: f32,                     // default: 0.70
    pub threshold_sweep_iterations: usize,   // default: 10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveTrainingConfig {
    pub attract_strength: f32,               // default: 0.02
    pub repel_strength: f32,                 // default: 0.01
    pub correct_attract_bonus: f32,          // default: 0.01 (extra for correct)
    pub mini_batch_size: usize,              // default: 32
    pub max_walk_steps: usize,               // default: 50
    pub convergence_tolerance: f32,          // default: 0.001
    pub temperature_anneal_rate: f32,        // default: 0.95
    pub momentum: f32,                       // default: 0.9
    pub walk_sweep_iterations: usize,        // default: 10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyCheckConfig {
    pub r1_threshold_adjustment: f32,        // default: 0.05
    pub r3_anchor_adjustment: f32,           // default: 0.05
    pub r4_confidence_adjustment: f32,       // default: 0.05
    pub max_correction_rounds: usize,        // default: 3
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemTrainingConfig {
    pub classification: ClassificationTrainingConfig,
    pub reasoning: ReasoningTrainingConfig,
    pub predictive: PredictiveTrainingConfig,
    pub consistency: ConsistencyCheckConfig,
}
```

Add corresponding YAML section to `config/config.yaml`:

```yaml
system_training:
  classification:
    centroid_batch_size: 500
    weight_sweep_iterations: 10
    weight_sweep_dimensions:
      - w_intent_hash
      - w_tone_hash
      - w_structure
      - w_punctuation
      - w_semantic
      - w_derived
      - low_confidence_threshold
      - high_confidence_threshold
      - retrieval_gate_threshold
    calibration_bins: 10
    validation_split: 0.1
    min_examples_per_intent: 100
  reasoning:
    scoring_sweep_iterations: 20
    decomposition_min_examples: 50
    mrr_target: 0.70
    threshold_sweep_iterations: 10
  predictive:
    attract_strength: 0.02
    repel_strength: 0.01
    correct_attract_bonus: 0.01
    mini_batch_size: 32
    max_walk_steps: 50
    convergence_tolerance: 0.001
    temperature_anneal_rate: 0.95
    momentum: 0.9
    walk_sweep_iterations: 10
  consistency:
    r1_threshold_adjustment: 0.05
    r3_anchor_adjustment: 0.05
    r4_confidence_adjustment: 0.05
    max_correction_rounds: 3
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`  
**Tests:** Config deserialization test, default value assertions.

### Task 1.4: Centroid Storage in MemoryStore (`src/memory/store.rs`)

The `MemoryStore` already has `intent_centroids()` and `tone_centroids()` methods. Add mutation methods for training:

```rust
// Add to MemoryStore impl in src/memory/store.rs

/// Set or update intent centroids (called during classification training)
pub fn set_intent_centroids(&mut self, centroids: HashMap<IntentKind, Vec<f32>>) {
    self.intent_centroids = centroids;
}

/// Set or update tone centroids (called during classification training)
pub fn set_tone_centroids(&mut self, centroids: HashMap<ToneKind, Vec<f32>>) {
    self.tone_centroids = centroids;
}

/// Incrementally update a single intent centroid via running mean
pub fn update_intent_centroid(&mut self, intent: IntentKind, new_vector: &[f32], count: usize) {
    let centroid = self.intent_centroids.entry(intent).or_insert_with(|| vec![0.0; 78]);
    let n = count as f32;
    for (i, val) in centroid.iter_mut().enumerate() {
        *val = *val * ((n - 1.0) / n) + new_vector[i] / n;
    }
}

/// Incrementally update a single tone centroid via running mean
pub fn update_tone_centroid(&mut self, tone: ToneKind, new_vector: &[f32], count: usize) {
    let centroid = self.tone_centroids.entry(tone).or_insert_with(|| vec![0.0; 78]);
    let n = count as f32;
    for (i, val) in centroid.iter_mut().enumerate() {
        *val = *val * ((n - 1.0) / n) + new_vector[i] / n;
    }
}
```

**Files modified:** `src/memory/store.rs`  
**Tests:** Test that incremental centroid update converges to batch mean.

---

## 3. Phase 2: Classification System Training

### Task 2.1: Classification Trainer Overhaul (`src/classification/trainer.rs`)

Replace existing `ClassificationTrainer` with the three-phase pipeline from §11.2.

**Current state:** The trainer converts labeled dialogues to patterns and inserts them into the spatial grid. It does not build centroids or optimize weights.

**New implementation:**

```rust
// src/classification/trainer.rs — full rewrite

pub struct ClassificationTrainer {
    semantic_hasher: SemanticHasher,
}

impl ClassificationTrainer {
    pub fn new() -> Self {
        Self { semantic_hasher: SemanticHasher::new() }
    }

    /// Phase 1: Build per-intent and per-tone centroids from labeled examples.
    /// Uses incremental running mean for memory efficiency.
    pub fn build_centroids(
        &self,
        examples: &[TrainingExample],
        memory: &mut MemoryStore,
    ) -> CentroidReport {
        // For each example:
        //   1. Parse intent label → IntentKind
        //   2. Parse tone label → ToneKind
        //   3. Compute ClassificationSignature::compute(text, &self.semantic_hasher)
        //   4. Convert to 78-float feature vector
        //   5. memory.update_intent_centroid(intent, &fv, running_count)
        //   6. memory.update_tone_centroid(tone, &fv, running_count)
        // Return CentroidReport { intents_built, tones_built, examples_processed }
        todo!()
    }

    /// Phase 2: Bayesian/grid sweep over feature weights and thresholds.
    /// Evaluates L_class on validation set for each configuration.
    pub fn optimize_weights(
        &self,
        validation_examples: &[TrainingExample],
        memory: &MemoryStore,
        spatial: &SpatialGrid,
        base_config: &ClassificationConfig,
        sweep_config: &ClassificationTrainingConfig,
    ) -> WeightOptimizationReport {
        // Generate weight configurations:
        //   For grid sweep: linspace each dimension in sweep_config.weight_sweep_dimensions
        //   For Bayesian: use TPE-like sampling (simplified: Latin hypercube + best-so-far)
        // For each config:
        //   1. Create temporary ClassificationCalculator with those weights
        //   2. Run calculate() on all validation examples
        //   3. Compute ClassificationLoss
        //   4. Record (config, l_class)
        // Return best config + all results
        todo!()
    }

    /// Phase 3: Confidence calibration via Platt scaling.
    /// Tracks actual accuracy per confidence band.
    pub fn calibrate_confidence(
        &self,
        calibration_examples: &[TrainingExample],
        memory: &MemoryStore,
        spatial: &SpatialGrid,
        config: &ClassificationConfig,
        bins: usize,
    ) -> CalibrationReport {
        // For each example:
        //   1. Run classification → get raw_confidence
        //   2. Check if prediction was correct
        //   3. Bucket into confidence band (0.0-0.1, 0.1-0.2, ...)
        // For each band:
        //   actual_accuracy = correct_in_band / total_in_band
        //   platt_a, platt_b = fit logistic(a * raw + b) to actual
        // Return CalibrationReport { bands, platt_params }
        todo!()
    }

    /// Full training pipeline: centroid → weights → calibration
    pub fn train_full(
        &self,
        train_examples: &[TrainingExample],
        val_examples: &[TrainingExample],
        memory: &mut MemoryStore,
        spatial: &SpatialGrid,
        config: &ClassificationConfig,
        training_config: &ClassificationTrainingConfig,
    ) -> ClassificationLoss {
        // 1. Split train into centroid_set + weight_set (or use all for centroid)
        // 2. build_centroids(centroid_set, memory)
        // 3. optimize_weights(val_examples, memory, spatial, config, training_config)
        // 4. calibrate_confidence(val_examples, memory, spatial, best_config, bins)
        // 5. Return final ClassificationLoss evaluated on val_examples with best config
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct CentroidReport {
    pub intents_built: usize,
    pub tones_built: usize,
    pub examples_processed: usize,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
pub struct WeightOptimizationReport {
    pub best_config: ClassificationConfig,
    pub best_loss: f32,
    pub iterations: usize,
    pub all_results: Vec<(ClassificationConfig, f32)>,
    pub duration_ms: u128,
}

#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub bands: Vec<CalibrationBand>,
    pub platt_a: f32,
    pub platt_b: f32,
}

#[derive(Debug, Clone)]
pub struct CalibrationBand {
    pub lower: f32,
    pub upper: f32,
    pub raw_accuracy: f32,
    pub calibrated_accuracy: f32,
    pub sample_count: usize,
}
```

**Files modified:** `src/classification/trainer.rs`  
**Tests:**
- `build_centroids_produces_correct_mean` — 10 examples per intent, verify centroid ≈ mean
- `optimize_weights_improves_loss` — verify best config has lower L_class than default
- `calibrate_confidence_produces_monotonic_bands` — higher raw → higher calibrated

### Task 2.2: L_class Evaluation Function (`src/classification/calculator.rs`)

Add a method to `ClassificationCalculator` that evaluates L_class over a batch:

```rust
// Add to ClassificationCalculator impl

/// Evaluate classification loss over a batch of labeled examples.
/// Returns ClassificationLoss with per-component breakdown.
pub fn evaluate_loss(
    &self,
    examples: &[TrainingExample],
    memory: &MemoryStore,
    spatial: &SpatialGrid,
    config: &ClassificationConfig,
) -> ClassificationLoss {
    let mut correct_intent = 0usize;
    let mut correct_tone = 0usize;
    let mut gate_loss_sum = 0.0f32;
    let total = examples.len();

    for example in examples {
        let result = self.calculate(&example.question, memory, spatial, config);
        
        // Intent accuracy
        if let Some(ref expected_intent) = example.intent {
            let expected = parse_intent_kind(expected_intent);
            if result.intent == expected { correct_intent += 1; }
        }
        
        // Tone accuracy
        if let Some(ref expected_tone) = example.tone {
            let expected = parse_tone_kind(expected_tone);
            if result.tone == expected { correct_tone += 1; }
        }
        
        // Gate accuracy (binary cross-entropy)
        if let Some(needs_retrieval) = example.needs_retrieval {
            let predicted = result.confidence < config.low_confidence_threshold;
            let p = if predicted { 0.99 } else { 0.01 };
            let y = if needs_retrieval { 1.0 } else { 0.0 };
            gate_loss_sum += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }
    }

    let intent_acc = correct_intent as f32 / total.max(1) as f32;
    let tone_acc = correct_tone as f32 / total.max(1) as f32;
    let gate_acc_avg = gate_loss_sum / total.max(1) as f32;

    let l_intent = 1.0 - intent_acc;
    let l_tone = 1.0 - tone_acc;
    let l_gate = gate_acc_avg;
    let l_class = l_intent + 0.5 * l_tone + 0.3 * l_gate;

    ClassificationLoss {
        l_intent, l_tone, l_gate, l_class,
        intent_accuracy: intent_acc,
        tone_accuracy: tone_acc,
        gate_accuracy: 1.0 - gate_acc_avg,
        sample_count: total,
    }
}
```

**Files modified:** `src/classification/calculator.rs`  
**Tests:** `evaluate_loss_perfect_predictions_gives_zero_loss`

### Task 2.3: Wire into Engine (`src/engine.rs`)

Add `train_classification()` method to `Engine`:

```rust
// Add to Engine impl in src/engine.rs

/// Train the Classification System independently.
/// Builds centroids, optimizes weights, calibrates confidence.
pub fn train_classification(
    &self,
    train_examples: &[TrainingExample],
    val_examples: &[TrainingExample],
) -> ClassificationLoss {
    let trainer = ClassificationTrainer::new();
    let mut memory = self.memory.lock().expect("memory lock");
    let spatial = self.spatial_grid.lock().expect("spatial lock");
    
    trainer.train_full(
        train_examples,
        val_examples,
        &mut memory,
        &spatial,
        &self.config.classification,
        &self.config.system_training.classification,
    )
}
```

**Files modified:** `src/engine.rs`

---

## 4. Phase 3: Reasoning System Training

### Task 3.1: Query Decomposer (`src/reasoning/decomposer.rs`) — NEW FILE

Implements structured query decomposition for multi-hop reasoning.

```rust
// src/reasoning/decomposer.rs — NEW FILE

use crate::types::IntentKind;
use crate::config::ReasoningTrainingConfig;

/// Decomposes a complex query into sub-questions based on intent type.
pub struct QueryDecomposer {
    /// Templates per intent: Vec of pattern strings with {X}, {Y}, {N} placeholders
    templates: HashMap<IntentKind, Vec<DecompositionTemplate>>,
}

#[derive(Debug, Clone)]
pub struct DecompositionTemplate {
    pub pattern: String,           // e.g., "What is {X}?"
    pub slot_extractors: Vec<SlotExtractor>,
}

#[derive(Debug, Clone)]
pub enum SlotExtractor {
    FirstNoun,                     // Extract first noun phrase
    SecondNoun,                    // Extract second noun phrase
    NumberedItem(usize),           // Extract Nth item from list
    AfterKeyword(String),          // Extract text after keyword
}

#[derive(Debug, Clone)]
pub struct DecomposedQuery {
    pub original: String,
    pub sub_questions: Vec<SubQuestion>,
    pub dependency_graph: Vec<(usize, usize)>, // (from_idx, to_idx) edges
}

#[derive(Debug, Clone)]
pub struct SubQuestion {
    pub text: String,
    pub depends_on: Vec<usize>,    // indices of sub-questions this depends on
    pub confidence: f32,
}

impl QueryDecomposer {
    /// Create with default templates per intent type
    pub fn new() -> Self { todo!() }

    /// Load templates learned from training data
    pub fn with_templates(templates: HashMap<IntentKind, Vec<DecompositionTemplate>>) -> Self { todo!() }

    /// Decompose query into sub-questions
    pub fn decompose(&self, query: &str, intent: IntentKind) -> DecomposedQuery { todo!() }

    /// Learn templates from labeled multi-hop examples
    pub fn learn_templates(
        &mut self,
        examples: &[TrainingExample],
        config: &ReasoningTrainingConfig,
    ) -> TemplateReport { todo!() }
}
```

**Built-in default templates (hardcoded, overridden by learning):**

| Intent | Template Pattern |
|--------|-----------------|
| Compare | `["What is {X}?", "What is {Y}?", "How do {X} and {Y} differ?"]` |
| Analyze | `["What are the components of {X}?", "How does each component work?", "What is the overall behavior?"]` |
| Plan | `["What is the goal of {X}?", "What steps are needed?", "What resources for step {N}?"]` |
| Debug | `["What is the error?", "What are possible causes?", "How to verify each cause?"]` |
| Explain | `["What is {X}?", "Why does {X} work this way?", "What are the implications?"]` |
| Critique | `["What is the claim?", "What evidence supports it?", "What evidence contradicts it?"]` |

**Files created:** `src/reasoning/decomposer.rs`  
**Files modified:** `src/reasoning/mod.rs` (add `pub mod decomposer;`)  
**Tests:**
- `decompose_compare_query_produces_three_subquestions`
- `decompose_simple_question_returns_single_subquestion`
- `learn_templates_from_examples_updates_templates`

### Task 3.2: Implement Real `generate_thought_unit()` (`src/engine.rs`)

Replace the placeholder stub with actual structured reasoning.

**Current:** Returns a formatted string with incrementally improved confidence.  
**New:** Uses `QueryDecomposer` + memory search per sub-question + confidence chaining.

```rust
// Replace generate_thought_unit in src/engine.rs

fn generate_thought_unit(
    &self,
    query: &str,
    step: usize,
    state: &ReasoningState,
) -> ThoughtUnit {
    let intent = state.current_intent.unwrap_or(IntentKind::Question);
    
    // Step 1: Decompose (only on first step)
    let decomposed = if step == 0 {
        self.query_decomposer.decompose(query, intent)
    } else {
        // Subsequent steps: focus on lowest-confidence sub-question
        let weakest_idx = state.sub_question_confidences
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        DecomposedQuery::single(state.sub_questions[weakest_idx].clone())
    };

    // Step 2: Search memory for each sub-question
    let snapshot = self.memory_snapshot.load_full();
    let mut sub_answers = Vec::new();
    let mut chain_confidence = 1.0f32;

    for sub_q in &decomposed.sub_questions {
        let matches = snapshot.top_channel_matches(
            MemoryChannel::Reasoning,
            &sub_q.text.to_lowercase(),
            12,
        );
        let best_match = matches.first();
        let sub_confidence = best_match
            .map(|u| (u.utility_score + u.confidence + u.trust_score) / 3.0)
            .unwrap_or(0.1);
        
        sub_answers.push(SubAnswer {
            question: sub_q.text.clone(),
            answer: best_match.map(|u| u.content.clone()).unwrap_or_default(),
            confidence: sub_confidence,
        });
        chain_confidence *= sub_confidence;
    }

    // Step 3: Synthesize
    let content = sub_answers.iter()
        .filter(|sa| !sa.answer.is_empty())
        .map(|sa| sa.answer.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    ThoughtUnit {
        content,
        confidence: chain_confidence.clamp(0.0, 1.0),
        reasoning_type: classify_reasoning_type(intent),
        step,
        source: ThoughtSource::Internal,
        timestamp: Utc::now(),
    }
}
```

**Files modified:** `src/engine.rs`  
**Tests:**
- `generate_thought_unit_decomposes_compare_query`
- `generate_thought_unit_chains_confidence_correctly`
- `generate_thought_unit_flags_low_confidence_for_retrieval`

### Task 3.3: L_reason Evaluation Function (`src/reasoning/search.rs`)

Add MRR evaluation to `CandidateScorer`:

```rust
// Add to src/reasoning/search.rs

impl CandidateScorer {
    /// Compute Mean Reciprocal Rank of correct answer in scored candidate list.
    pub fn evaluate_mrr(
        scored: &[ScoredCandidate],
        correct_unit_id: Uuid,
    ) -> f32 {
        for (rank, candidate) in scored.iter().enumerate() {
            if candidate.unit_id == correct_unit_id {
                return 1.0 / (rank as f32 + 1.0);
            }
        }
        0.0 // correct answer not in candidate list
    }
}
```

**Files modified:** `src/reasoning/search.rs`

### Task 3.4: Reasoning Training Pipeline (`src/training/pipeline.rs`)

Add `train_reasoning()` to the training module:

```rust
// Add to src/training.rs or src/engine.rs

/// Train the Reasoning System independently.
/// Phase 1: Optimize 7D scoring weights via MRR maximization.
/// Phase 2: Learn decomposition templates from multi-hop examples.
/// Phase 3: Optimize reasoning loop thresholds.
pub fn train_reasoning(
    &self,
    qa_examples: &[TrainingExample],      // single-hop QA
    multihop_examples: &[TrainingExample], // multi-hop with sub_questions
    gate_examples: &[TrainingExample],     // retrieval gate ground truth
) -> ReasoningLoss {
    // Phase 1: Scoring weight sweep
    //   For each weight config (Bayesian sampling of 7 dimensions):
    //     For each qa_example:
    //       1. Ingest question, build context
    //       2. Score candidates with current weights
    //       3. Compute MRR against correct_unit_id
    //     Record mean MRR
    //   Select weights maximizing mean MRR
    
    // Phase 2: Template learning
    //   self.query_decomposer.learn_templates(multihop_examples, config)
    
    // Phase 3: Threshold sweep
    //   For each (trigger_floor, exit_threshold, retrieval_flag) config:
    //     For each gate_example:
    //       Run reasoning loop → check if retrieval decision matches ground truth
    //     Record gate accuracy
    //   Select thresholds maximizing gate accuracy
    
    // Return combined ReasoningLoss
    todo!()
}
```

**Files modified:** `src/engine.rs` (or `src/training.rs`)  
**Tests:**
- `train_reasoning_improves_mrr_over_default`
- `train_reasoning_learns_compare_template`
- `train_reasoning_optimizes_retrieval_gate`

---

## 5. Phase 4: Predictive System Training

### Task 4.1: Word Graph Trainer (`src/predictive/walk_trainer.rs`) — NEW FILE

Implements Word Graph edge formation training with attract/repel layout.

```rust
// src/predictive/walk_trainer.rs — NEW FILE

use crate::config::PredictiveTrainingConfig;
use crate::spatial_index::SpatialGrid;
use crate::reasoning::search::CandidateScorer;
use crate::predictive::resolver::FineResolver;
use crate::types::*;

/// Trains unit positions via autoregressive spatial walk.
pub struct SpatialWalkTrainer {
    config: PredictiveTrainingConfig,
    /// Accumulated position deltas (batched before applying)
    position_deltas: HashMap<Uuid, [f32; 3]>,
    /// Momentum state per unit
    momentum: HashMap<Uuid, [f32; 3]>,
    /// Training metrics
    total_steps: usize,
    correct_steps: usize,
}

impl SpatialWalkTrainer {
    pub fn new(config: PredictiveTrainingConfig) -> Self { todo!() }

    /// Train on a single sequence. Accumulates position deltas.
    /// Returns (steps_correct, steps_total) for this sequence.
    pub fn train_sequence(
        &mut self,
        sequence: &[UnitSequenceEntry],
        context: &ContextMatrix,
        grid: &SpatialGrid,
        memory_snapshot: &MemorySnapshot,
        weights: &ScoringWeights,
        resolver_config: &FineResolverConfig,
        shaping_config: &IntentShapingConfig,
    ) -> (usize, usize) {
        let mut position = sequence[0].position;
        let mut correct = 0usize;

        for i in 0..sequence.len() - 1 {
            let expected = &sequence[i + 1];
            
            // 1. Find candidates near current position
            let candidates = grid.nearby(position, 6.0);
            // ... score candidates via 7D scorer ...
            // ... resolve via FineResolver ...

            // 2. Compare proposed vs expected
            let expected_id = /* lookup unit id for expected.content */;
            if proposed_id == expected_id {
                // Correct: attract expected toward current position
                self.accumulate_attract(expected_id, position, self.config.attract_strength);
                correct += 1;
            } else {
                // Incorrect: attract expected, repel proposed
                self.accumulate_attract(
                    expected_id, position, 
                    self.config.attract_strength + self.config.correct_attract_bonus
                );
                self.accumulate_repel(proposed_id, position, self.config.repel_strength);
            }

            // 3. Teacher forcing: advance to expected position
            position = expected.position;
        }

        (correct, sequence.len().saturating_sub(1))
    }

    /// Apply accumulated deltas with momentum to the spatial grid.
    /// Call after every mini_batch_size sequences.
    pub fn apply_deltas(&mut self, grid: &mut SpatialGrid) {
        for (unit_id, delta) in &self.position_deltas {
            // Apply momentum: momentum_state = momentum * old + (1-momentum) * delta
            let mom = self.momentum.entry(*unit_id).or_insert([0.0; 3]);
            for d in 0..3 {
                mom[d] = self.config.momentum * mom[d] 
                       + (1.0 - self.config.momentum) * delta[d];
            }
            grid.update_position(*unit_id, *mom);
        }
        self.position_deltas.clear();
    }

    /// Evaluate predictive loss on a held-out sequence set.
    pub fn evaluate_loss(
        &self,
        sequences: &[Vec<UnitSequenceEntry>],
        grid: &SpatialGrid,
        memory_snapshot: &MemorySnapshot,
        weights: &ScoringWeights,
    ) -> PredictiveLoss { todo!() }

    fn accumulate_attract(&mut self, unit_id: Uuid, toward: [f32; 3], strength: f32) {
        let delta = self.position_deltas.entry(unit_id).or_insert([0.0; 3]);
        let current = /* get current position from grid */;
        for d in 0..3 {
            delta[d] += (toward[d] - current[d]) * strength;
        }
    }

    fn accumulate_repel(&mut self, unit_id: Uuid, away_from: [f32; 3], strength: f32) {
        let delta = self.position_deltas.entry(unit_id).or_insert([0.0; 3]);
        let current = /* get current position from grid */;
        for d in 0..3 {
            delta[d] -= (away_from[d] - current[d]) * strength;
        }
    }
}
```

**Files created:** `src/predictive/walk_trainer.rs`  
**Files modified:** `src/predictive/mod.rs` (add `pub mod walk_trainer;`)  
**Tests:**
- `train_sequence_correct_prediction_attracts`
- `train_sequence_incorrect_prediction_repels_and_attracts`
- `apply_deltas_with_momentum_smooths_updates`
- `evaluate_loss_decreases_after_training`

### Task 4.2: Wire Predictive Training into Engine (`src/engine.rs`)

```rust
// Add to Engine impl

/// Train the Predictive System independently.
/// Runs autoregressive spatial walk training on unit sequences.
pub fn train_predictive(
    &self,
    train_sequences: &[Vec<UnitSequenceEntry>],
    val_sequences: &[Vec<UnitSequenceEntry>],
) -> PredictiveLoss {
    let mut trainer = SpatialWalkTrainer::new(
        self.config.system_training.predictive.clone()
    );
    let snapshot = self.memory_snapshot.load_full();
    let mut grid = self.spatial_grid.lock().expect("spatial lock");
    
    // Phase 1: Sequence training with attract/repel
    for (i, sequence) in train_sequences.iter().enumerate() {
        let context = /* build context from sequence */ ;
        trainer.train_sequence(
            sequence,
            &context,
            &grid,
            &snapshot,
            &self.config.layer_14_candidate_scoring,
            &self.config.layer_16_fine_resolver,
            &self.config.intent_shaping,
        );
        
        // Apply deltas every mini_batch_size sequences
        if (i + 1) % self.config.system_training.predictive.mini_batch_size == 0 {
            trainer.apply_deltas(&mut grid);
        }
    }
    trainer.apply_deltas(&mut grid); // flush remaining
    
    // Phase 2: Force-directed layout refinement
    let units = snapshot.all_units();
    crate::spatial_index::force_directed_layout(&mut grid, &units, &self.config);
    
    // Phase 3: Evaluate on validation set
    trainer.evaluate_loss(val_sequences, &grid, &snapshot, &self.config.layer_14_candidate_scoring)
}
```

**Files modified:** `src/engine.rs`

### Task 4.3: SpatialGrid Position Update Method (`src/spatial_index.rs`)

Add a method to update a single unit's position (used by attract/repel):

```rust
// Add to SpatialGrid impl in src/spatial_index.rs

/// Update a single unit's position. Removes from old cell, inserts into new cell.
pub fn update_position(&mut self, unit_id: Uuid, delta: [f32; 3]) {
    if let Some(old_pos) = self.positions.get(&unit_id).copied() {
        let old_cell = self.cell_key(old_pos);
        let new_pos = [
            old_pos[0] + delta[0],
            old_pos[1] + delta[1],
            old_pos[2] + delta[2],
        ];
        let new_cell = self.cell_key(new_pos);
        
        // Remove from old cell
        if old_cell != new_cell {
            if let Some(cell) = self.cells.get_mut(&old_cell) {
                cell.retain(|id| *id != unit_id);
            }
            // Insert into new cell
            self.cells.entry(new_cell).or_default().push(unit_id);
        }
        
        // Update stored position
        self.positions.insert(unit_id, new_pos);
    }
}
```

**Files modified:** `src/spatial_index.rs`  
**Tests:** `update_position_moves_unit_between_cells`

---

## 6. Phase 5: Cross-System Consistency

### Task 5.1: Consistency Checker (`src/reasoning/consistency.rs`) — NEW FILE

```rust
// src/reasoning/consistency.rs — NEW FILE

use crate::types::*;
use crate::config::ConsistencyCheckConfig;

pub struct ConsistencyChecker {
    config: ConsistencyCheckConfig,
}

impl ConsistencyChecker {
    pub fn new(config: ConsistencyCheckConfig) -> Self {
        Self { config }
    }

    /// Run consistency check over validation examples.
    /// Returns ConsistencyLoss and suggested config corrections.
    pub fn check(
        &self,
        results: &[ConsistencyCheckResult],
    ) -> (ConsistencyLoss, Vec<ConfigCorrection>) {
        let mut loss = ConsistencyLoss::default();
        loss.sample_count = results.len();

        for result in results {
            // R1: High uncertainty must trigger retrieval
            if result.classification_confidence < 0.40 && !result.reasoning_used_retrieval {
                loss.r1_violations += 1;
            }
            // R2: Social intents should skip reasoning
            if is_social_intent(result.classification_intent) && result.reasoning_steps > 0 {
                loss.r2_violations += 1;
            }
            // R3: Factual intents must preserve anchors
            if is_factual_intent(result.classification_intent) && result.prediction_contradicts_anchor {
                loss.r3_violations += 1;
            }
            // R4: Creative intents must allow drift
            if is_creative_intent(result.classification_intent) && !result.prediction_drift_allowed {
                loss.r4_violations += 1;
            }
        }

        let total_violations = loss.r1_violations + loss.r2_violations 
                             + loss.r3_violations + loss.r4_violations;
        loss.l_consistency = total_violations as f32 / loss.sample_count.max(1) as f32;

        let corrections = self.generate_corrections(&loss);
        (loss, corrections)
    }

    fn generate_corrections(&self, loss: &ConsistencyLoss) -> Vec<ConfigCorrection> {
        let mut corrections = Vec::new();
        if loss.r1_violations > 0 {
            corrections.push(ConfigCorrection {
                section: "reasoning_loop".into(),
                field: "trigger_confidence_floor".into(),
                adjustment: -self.config.r1_threshold_adjustment,
                reason: format!("{} R1 violations: lower retrieval trigger", loss.r1_violations),
            });
        }
        if loss.r3_violations > 0 {
            corrections.push(ConfigCorrection {
                section: "intent_shaping".into(),
                field: "anchor_trust_threshold".into(),
                adjustment: self.config.r3_anchor_adjustment,
                reason: format!("{} R3 violations: increase anchor protection", loss.r3_violations),
            });
        }
        if loss.r4_violations > 0 {
            corrections.push(ConfigCorrection {
                section: "layer_16_fine_resolver".into(),
                field: "min_confidence_floor".into(),
                adjustment: -self.config.r4_confidence_adjustment,
                reason: format!("{} R4 violations: lower confidence floor for creative", loss.r4_violations),
            });
        }
        corrections
    }
}

#[derive(Debug, Clone)]
pub struct ConsistencyCheckResult {
    pub classification_intent: IntentKind,
    pub classification_confidence: f32,
    pub reasoning_used_retrieval: bool,
    pub reasoning_steps: usize,
    pub prediction_contradicts_anchor: bool,
    pub prediction_drift_allowed: bool,
}

#[derive(Debug, Clone)]
pub struct ConfigCorrection {
    pub section: String,
    pub field: String,
    pub adjustment: f32,
    pub reason: String,
}
```

**Files created:** `src/reasoning/consistency.rs`  
**Files modified:** `src/reasoning/mod.rs` (add `pub mod consistency;`)  
**Tests:**
- `check_detects_r1_violations`
- `check_generates_corrections_for_r3`
- `check_zero_violations_gives_zero_loss`

### Task 5.2: Wire Consistency into Engine (`src/engine.rs`)

```rust
/// Run cross-system consistency check.
/// Executes full pipeline on validation examples and checks inter-system agreements.
pub fn run_consistency_check(
    &self,
    validation_examples: &[TrainingExample],
) -> (ConsistencyLoss, Vec<ConfigCorrection>) {
    let mut results = Vec::with_capacity(validation_examples.len());
    
    for example in validation_examples {
        let process_result = self.process(&example.question);
        // Extract consistency check fields from process_result + trace
        results.push(ConsistencyCheckResult {
            classification_intent: process_result.trace.intent,
            classification_confidence: process_result.confidence,
            reasoning_used_retrieval: process_result.used_retrieval,
            reasoning_steps: process_result.trace.reasoning_steps,
            prediction_contradicts_anchor: /* check via trace */,
            prediction_drift_allowed: /* check via trace */,
        });
    }
    
    let checker = ConsistencyChecker::new(self.config.system_training.consistency.clone());
    checker.check(&results)
}
```

**Files modified:** `src/engine.rs`

---

## 7. Phase 6: Dataset Generators

### Task 6.1: Classification Dataset Generator (`src/seed/classification_generator.rs`) — NEW FILE

Generates 50K+ labeled `{text, intent, tone, needs_retrieval}` examples.

**Strategy:**
- Template-based generation with randomized slot filling
- Per intent: 10-20 templates × 100-200 slot variations = ~1,500-3,000 per intent
- Tone variation: each template rendered with 6 tone modifiers
- Retrieval flag: set based on intent type (Unknown, fact-heavy → true; social → false)
- Ambiguity injection: 30% of examples have borderline intent (e.g., "Can you explain how to debug?" → Explain or Debug?)

```rust
// src/seed/classification_generator.rs

pub struct ClassificationDatasetGenerator;

impl ClassificationDatasetGenerator {
    /// Generate full classification training dataset.
    pub fn generate(count: usize) -> Vec<TrainingExample> { todo!() }
    
    /// Generate examples for a specific intent.
    fn generate_for_intent(intent: IntentKind, count: usize) -> Vec<TrainingExample> { todo!() }

    /// Generate ambiguous/boundary examples.
    fn generate_ambiguous(count: usize) -> Vec<TrainingExample> { todo!() }

    /// Generate retrieval-needed examples.
    fn generate_retrieval_examples(count: usize) -> Vec<TrainingExample> { todo!() }
}
```

**Template examples per intent:**

| Intent | Template | Slot Variations |
|--------|----------|-----------------|
| Question | "What is {noun}?", "How does {noun} work?", "Can you tell me about {noun}?" | 200 nouns from knowledge domains |
| Compare | "Compare {noun_a} and {noun_b}", "What's the difference between {noun_a} and {noun_b}?" | 150 noun pairs |
| Plan | "Help me plan {activity}", "Create a plan for {activity}" | 150 activities |
| Greeting | "Hello", "Hi there", "Hey, how's it going?" | 50 greeting variations |
| Debug | "Why is {component} throwing {error}?", "Fix the {error} in {component}" | 200 component × error combos |

**Files created:** `src/seed/classification_generator.rs`  
**Files modified:** `src/seed/mod.rs`  
**Tests:** `generate_produces_correct_count`, `generate_covers_all_intents`

### Task 6.2: Reasoning Dataset Generator (`src/seed/reasoning_generator.rs`) — NEW FILE

Generates 20K+ QA pairs with reasoning traces.

```rust
// src/seed/reasoning_generator.rs

pub struct ReasoningDatasetGenerator;

impl ReasoningDatasetGenerator {
    /// Generate single-hop QA examples (60% of total).
    pub fn generate_single_hop(count: usize) -> Vec<TrainingExample> { todo!() }
    
    /// Generate multi-hop QA with sub-questions and reasoning traces (25%).
    pub fn generate_multi_hop(count: usize) -> Vec<TrainingExample> { todo!() }
    
    /// Generate adversarial examples (15%).
    pub fn generate_adversarial(count: usize) -> Vec<TrainingExample> { todo!() }
    
    /// Generate full reasoning dataset.
    pub fn generate(count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        examples.extend(Self::generate_single_hop((count as f32 * 0.60) as usize));
        examples.extend(Self::generate_multi_hop((count as f32 * 0.25) as usize));
        examples.extend(Self::generate_adversarial((count as f32 * 0.15) as usize));
        examples
    }
}
```

**Multi-hop example structure:**
```json
{
  "question": "Is the capital of France larger than the capital of Germany?",
  "answer": "Paris has a population of ~2.1M, Berlin ~3.6M, so Berlin is larger.",
  "sub_questions": [
    "What is the capital of France?",
    "What is the population of Paris?",
    "What is the capital of Germany?",
    "What is the population of Berlin?",
    "Which is larger?"
  ],
  "reasoning": {
    "steps": [
      {"type": "Premise", "content": "Need to compare capitals of France and Germany"},
      {"type": "Inference", "content": "Capital of France is Paris"},
      {"type": "Inference", "content": "Capital of Germany is Berlin"},
      {"type": "Calculation", "content": "Paris ~2.1M, Berlin ~3.6M"},
      {"type": "Conclusion", "content": "Berlin is larger than Paris"}
    ]
  },
  "intent": "Compare",
  "needs_retrieval": true
}
```

**Files created:** `src/seed/reasoning_generator.rs`  
**Files modified:** `src/seed/mod.rs`  
**Tests:** `generate_multi_hop_has_sub_questions`, `generate_adversarial_has_needs_retrieval`

### Task 6.3: Predictive Sequence Generator (`src/seed/predictive_generator.rs`) — NEW FILE

Generates 100K+ unit sequences from text corpora.

```rust
// src/seed/predictive_generator.rs

pub struct PredictiveSequenceGenerator;

impl PredictiveSequenceGenerator {
    /// Generate unit sequences from raw text.
    /// Uses L2 unit builder to discover units, then extracts ordered sequences.
    pub fn generate_from_text(
        texts: &[String],
        hasher: &SemanticHasher,
        config: &UnitBuilderConfig,
    ) -> Vec<Vec<UnitSequenceEntry>> { todo!() }

    /// Generate from existing memory store (extract sequences from linked units).
    pub fn generate_from_memory(
        memory: &MemorySnapshot,
    ) -> Vec<Vec<UnitSequenceEntry>> { todo!() }
}
```

**Strategy:**
- Take raw text paragraphs → run through L1 (normalize) → L2 (unit discovery) → L3 (hierarchy)
- Extract sequences at Word and Phrase level (skip Char/Subword for training efficiency)
- Each sequence = units in order of appearance within a sentence/paragraph
- Assign initial positions via SemanticHasher
- Target: 100K sequences × 30 average units = 3M training steps

**Files created:** `src/seed/predictive_generator.rs`  
**Files modified:** `src/seed/mod.rs`  
**Tests:** `generate_from_text_produces_ordered_sequences`, `sequences_have_valid_positions`

### Task 6.4: Consistency Dataset Generator (`src/seed/consistency_generator.rs`) — NEW FILE

Generates 5K+ cross-system validation examples with expected behavior per system.

```rust
// src/seed/consistency_generator.rs

pub struct ConsistencyDatasetGenerator;

impl ConsistencyDatasetGenerator {
    /// Generate consistency validation examples.
    /// Each example specifies expected behavior across all three systems.
    pub fn generate(count: usize) -> Vec<TrainingExample> { todo!() }
}
```

**Example categories:**

| Category | Count | Classification Expected | Reasoning Expected | Predictive Expected |
|----------|-------|------------------------|-------------------|---------------------|
| Social short-circuit | 500 | Greeting/Farewell, high confidence | 0 reasoning steps | Direct spatial response |
| Factual with anchors | 1000 | Question/Verify, factual mode | Retrieve if unknown | Anchor protection ON |
| Creative drift | 500 | Brainstorm/Creative | Wide candidate pool | Drift allowed, low floor |
| High uncertainty | 1000 | Unknown/low confidence | Must retrieve | External evidence mode |
| Adversarial (conflict) | 500 | Various | Conflict detection | No anchor contradiction |

**Files created:** `src/seed/consistency_generator.rs`  
**Files modified:** `src/seed/mod.rs`  
**Tests:** `generate_covers_all_categories`, `social_examples_expect_zero_reasoning`

---

## 8. Phase 7: Efficiency Optimizations

### Task 7.1: Two-Phase Classification (`src/classification/calculator.rs`)

```rust
// Modify calculate() in ClassificationCalculator

pub fn calculate(&self, text: &str, memory: &MemoryStore, spatial: &SpatialGrid, config: &ClassificationConfig) -> ClassificationResult {
    let query_sig = ClassificationSignature::compute(text, &self.semantic_hasher);
    let query_fv = query_sig.to_feature_vector();
    
    // Phase 1: Structural-only pre-filter (14 dims)
    let structural = &query_fv[..14];
    let intent_centroids = memory.intent_centroids();
    
    let mut struct_scores: Vec<(IntentKind, f32)> = intent_centroids.iter()
        .map(|(intent, centroid)| {
            let c_struct = &centroid[..14];
            (*intent, raw_cosine_similarity(structural, c_struct))
        })
        .collect();
    struct_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    // If structural margin is decisive (>0.3), skip full comparison
    if let (Some(first), Some(second)) = (struct_scores.first(), struct_scores.get(1)) {
        if first.1 - second.1 > config.two_phase_structural_margin {
            // Fast path: emit result from structural comparison only
            return self.build_result_from_structural(first, &query_fv, memory, config);
        }
    }
    
    // Phase 2: Full 78-dim comparison (existing code)
    // ... existing centroid comparison logic ...
}
```

**Files modified:** `src/classification/calculator.rs`  
**Config addition:** `two_phase_structural_margin: f32` (default: 0.3) in `ClassificationConfig`  
**Tests:** `two_phase_fast_path_triggers_for_clear_intent`, `two_phase_falls_through_for_ambiguous`

### Task 7.2: POS Tag LRU Cache (`src/classification/signature.rs`)

```rust
// Add to src/classification/signature.rs

use lru::LruCache;
use std::sync::Mutex;
use std::num::NonZeroUsize;

/// Global POS tag cache — avoids redundant tagger calls for repeated phrases.
static POS_CACHE: Lazy<Mutex<LruCache<u64, Vec<(String, String)>>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(NonZeroUsize::new(128).unwrap()))
});

fn cached_pos_tag(text: &str) -> Option<Vec<(String, String)>> {
    let key = crate::types::text_fingerprint(text);
    
    // Check cache first
    if let Ok(mut cache) = POS_CACHE.lock() {
        if let Some(cached) = cache.get(&key) {
            return Some(cached.clone());
        }
    }
    
    // Compute POS tags
    if let Some(tagger) = POS_TAGGER.as_ref() {
        let safe = sanitize_for_pos(text);
        if safe.is_empty() { return None; }
        let tags = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tagger.tag(&safe))) {
            Ok(t) => t,
            Err(_) => return None,
        };
        let result: Vec<(String, String)> = tags.iter()
            .map(|t| (t.word.clone(), t.tag.clone()))
            .collect();
        
        // Store in cache
        if let Ok(mut cache) = POS_CACHE.lock() {
            cache.put(key, result.clone());
        }
        Some(result)
    } else {
        None
    }
}
```

Then modify `compute_pos_intent_hash()` and `compute_pos_tone_hash()` to use `cached_pos_tag()`.

**Files modified:** `src/classification/signature.rs`  
**Dependency:** Add `lru = "0.12"` to `Cargo.toml`  
**Tests:** `pos_cache_returns_same_result`, `pos_cache_avoids_redundant_calls`

### Task 7.3: Reasoning Chain Cache (`src/engine.rs`)

```rust
// Add to Engine struct
reasoning_cache: Mutex<LruCache<u64, ReasoningResult>>,

// In execute_reasoning_loop, before the loop:
fn execute_reasoning_loop(&self, query: &str, ...) -> ReasoningResult {
    // Check cache
    let cache_key = {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        intent.primary.hash(&mut hasher);
        query.to_lowercase().hash(&mut hasher);
        hasher.finish()
    };
    if let Ok(mut cache) = self.reasoning_cache.lock() {
        if let Some(cached) = cache.get(&cache_key) {
            return cached.clone();
        }
    }

    // ... existing reasoning loop ...

    // Store result in cache
    if let Ok(mut cache) = self.reasoning_cache.lock() {
        cache.put(cache_key, result.clone());
    }
    result
}
```

**Files modified:** `src/engine.rs`  
**Tests:** `reasoning_cache_returns_cached_result`, `reasoning_cache_misses_for_different_intent`

### Task 7.4: Delta Scoring in Evidence Merge (`src/reasoning/search.rs`)

```rust
// Add to CandidateScorer

/// Re-score candidates updating only the evidence_support dimension.
/// Used after L13 evidence merge to avoid full 7D rescore.
pub fn delta_rescore_evidence(
    scored: &mut [ScoredCandidate],
    old_evidence_support: f32,
    new_evidence_support: f32,
    evidence_weight: f32,
) {
    let delta = (new_evidence_support - old_evidence_support) * evidence_weight;
    for candidate in scored.iter_mut() {
        candidate.score += delta;
    }
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
}
```

**Files modified:** `src/reasoning/search.rs`  
**Tests:** `delta_rescore_preserves_relative_order_except_evidence`

### Task 7.5: Pipeline Short-Circuit for Social Intents (`src/engine.rs`)

```rust
// In process_prompt(), after Classification System block:

// Short-circuit: social intents skip Reasoning entirely
const SOCIAL_INTENTS: &[IntentKind] = &[
    IntentKind::Greeting, IntentKind::Farewell,
    IntentKind::Gratitude, IntentKind::Continue,
];
if intent.confidence > self.config.classification.social_shortcircuit_confidence
    && SOCIAL_INTENTS.contains(&intent.primary) {
    // Route directly to Predictive System
    // ... spatial lookup + resolve + decode ...
    return process_result;
}
```

**Files modified:** `src/engine.rs`  
**Config addition:** `social_shortcircuit_confidence: f32` (default: 0.95) in `ClassificationConfig`  
**Tests:** `social_intent_skips_reasoning`, `low_confidence_social_does_not_shortcircuit`

### Task 7.6: Batch Position Updates in Spatial Grid (`src/spatial_index.rs`)

```rust
// Add to SpatialGrid

/// Apply multiple position updates in batch, rebuilding affected cells once.
pub fn batch_update_positions(&mut self, deltas: &HashMap<Uuid, [f32; 3]>) {
    let mut affected_cells = HashSet::new();
    
    for (unit_id, delta) in deltas {
        if let Some(old_pos) = self.positions.get(unit_id).copied() {
            let old_cell = self.cell_key(old_pos);
            let new_pos = [
                old_pos[0] + delta[0],
                old_pos[1] + delta[1],
                old_pos[2] + delta[2],
            ];
            let new_cell = self.cell_key(new_pos);
            
            affected_cells.insert(old_cell);
            affected_cells.insert(new_cell);
            
            self.positions.insert(*unit_id, new_pos);
        }
    }
    
    // Rebuild only affected cells
    for cell_key in affected_cells {
        self.rebuild_cell(cell_key);
    }
}
```

**Files modified:** `src/spatial_index.rs`  
**Tests:** `batch_update_moves_units_correctly`, `batch_update_rebuilds_only_affected_cells`

### Task 7.7: Neighborhood Pre-computation Cache (`src/spatial_index.rs`)

```rust
// Add to SpatialGrid

/// Cached neighbor results per cell, invalidated on position updates.
neighbor_cache: HashMap<CellKey, Vec<Uuid>>,
cache_generation: u64,

/// Get neighbors with caching. Cache is invalidated when positions change.
pub fn nearby_cached(&self, center: [f32; 3], radius: f32) -> Vec<Uuid> {
    let cell = self.cell_key(center);
    if let Some(cached) = self.neighbor_cache.get(&cell) {
        return cached.clone();
    }
    let result = self.nearby(center, radius);
    // Note: cache insertion deferred to avoid borrow issues
    result
}

/// Invalidate cache for cells affected by position updates.
fn invalidate_cache(&mut self, affected_cells: &HashSet<CellKey>) {
    for cell in affected_cells {
        self.neighbor_cache.remove(cell);
    }
    self.cache_generation += 1;
}
```

**Files modified:** `src/spatial_index.rs`

### Task 7.8: Confidence Calibration in Calculator (`src/classification/calculator.rs`)

```rust
// Add Platt scaling to ClassificationCalculator

/// Apply Platt scaling to raw confidence: calibrated = 1 / (1 + exp(-(a*raw + b)))
fn calibrate_confidence(&self, raw_confidence: f32, config: &ClassificationConfig) -> f32 {
    let a = config.platt_a; // default: 1.0 (no-op)
    let b = config.platt_b; // default: 0.0 (no-op)
    1.0 / (1.0 + (-(a * raw_confidence + b)).exp())
}
```

**Config addition:** `platt_a: f32` (default: 1.0), `platt_b: f32` (default: 0.0) in `ClassificationConfig`  
**Files modified:** `src/classification/calculator.rs`, `src/config/mod.rs`

---

## 9. Phase 8: Training Sweep Harness

### Task 8.1: Per-System Training Sweep Binary (`src/bin/training_sweep.rs`) — NEW FILE

```rust
// src/bin/training_sweep.rs

use clap::Parser;

#[derive(Parser)]
struct Args {
    /// System to sweep: classification, reasoning, predictive, consistency
    #[arg(long)]
    system: String,
    
    /// Comma-separated dimension names to sweep
    #[arg(long)]
    sweep_dims: Option<String>,
    
    /// Validate all systems (for consistency mode)
    #[arg(long)]
    validate_all: bool,
    
    /// Output report path
    #[arg(long, default_value = "benchmarks/training_sweep_report.md")]
    output: String,
}

fn main() {
    let args = Args::parse();
    
    match args.system.as_str() {
        "classification" => run_classification_sweep(&args),
        "reasoning" => run_reasoning_sweep(&args),
        "predictive" => run_predictive_sweep(&args),
        "consistency" => run_consistency_check(&args),
        "all" => {
            run_classification_sweep(&args);
            run_reasoning_sweep(&args);
            run_predictive_sweep(&args);
            run_consistency_check(&args);
        }
        _ => eprintln!("Unknown system: {}", args.system),
    }
}
```

**Files created:** `src/bin/training_sweep.rs`  
**Files modified:** `Cargo.toml` (add `[[bin]]` entry)

---

## 10. Phase 9: Integration Testing

### Task 9.1: System Training Integration Tests (`tests/training_integration.rs`) — NEW FILE

```rust
// tests/training_integration.rs

#[test]
fn classification_training_end_to_end() {
    // 1. Generate 1000 classification examples
    // 2. Split 800 train / 200 val
    // 3. Run train_classification()
    // 4. Assert L_class < 0.5 (better than random)
    // 5. Assert intent_accuracy > 0.6
}

#[test]
fn reasoning_training_end_to_end() {
    // 1. Generate 500 QA examples
    // 2. Train reasoning weights
    // 3. Assert MRR > 0.3
}

#[test]
fn predictive_training_end_to_end() {
    // 1. Generate 1000 unit sequences
    // 2. Train spatial walk
    // 3. Assert next_unit_accuracy > 0.1 (above random for vocab size)
}

#[test]
fn consistency_check_detects_violations() {
    // 1. Generate 200 consistency examples
    // 2. Run full pipeline
    // 3. Assert consistency check finds expected violations
    // 4. Assert corrections are generated
}

#[test]
fn full_training_pipeline_end_to_end() {
    // 1. Generate all datasets (small: 500 each)
    // 2. Run classification → reasoning → predictive → consistency
    // 3. Assert SystemTrainingReport has all fields populated
    // 4. Assert total_duration_ms < 60_000 (1 minute for small datasets)
}

#[test]
fn training_does_not_degrade_inference() {
    // 1. Run inference on probe queries → record baseline latency + quality
    // 2. Run full training pipeline
    // 3. Run same inference → assert latency within 2× baseline
    // 4. Assert no quality regression (confidence > baseline * 0.9)
}
```

**Files created:** `tests/training_integration.rs`

---

## 11. Phase 10: On-the-fly Learning

Implements the minimal runtime learning mechanism described in Architecture §4.9. Zero new files — only extensions to 3 existing files (~40 lines total).

**Architecture Reference:** §4.9 On-the-fly Learning (Runtime Edge Injection)

### Task 10.1: Cold-Start Detection in IntentDetector (`src/classification/intent.rs`)

Add a cold-start check to the existing `assess()` function. When input words have very few outgoing Word Graph edges, set `learning_flag = true` on the retrieval request.

```rust
// Add to IntentDetector::assess() in src/classification/intent.rs

// After existing intent classification, before returning:
let content_words = extract_content_words(&normalized_input);
let mut cold_start = false;
for word in &content_words {
    if let Some(node_id) = word_graph.lookup(word) {
        if word_graph.outgoing_edge_count(node_id)
            < config.runtime_learning.cold_start_edge_threshold
        {
            cold_start = true;
            break;
        }
    } else {
        // Word not in graph at all — definitely cold
        cold_start = true;
        break;
    }
}

// Set learning_flag on the assessment result
assessment.learning_flag = cold_start
    && assessment.confidence < config.retrieval_gate_threshold;
```

**Files modified:** `src/classification/intent.rs`
**Config addition:** `cold_start_edge_threshold: u32` (default: 3) in `RuntimeLearningConfig`
**Tests:**
- `cold_start_detected_for_word_with_no_edges`
- `cold_start_not_detected_for_well_connected_word`
- `learning_flag_false_when_confidence_high`

### Task 10.2: Edge Injection in Evidence Merge (`src/reasoning/merge.rs`)

After the existing evidence merge logic, add an edge injection step when `learning_flag` is set. Tokenizes merged evidence and creates Word Graph edges between consecutive words.

```rust
// Add to EvidenceMerger::merge() in src/reasoning/merge.rs

// After standard evidence merge (existing logic unchanged):
if retrieval_request.learning_flag {
    let context_hash = crate::types::text_fingerprint(&retrieval_request.query);
    for snippet in &merged_evidence {
        let words = crate::classification::input::tokenize_words(&snippet.content);
        for window in words.windows(2) {
            word_graph.create_or_strengthen_edge(
                &window[0],
                &window[1],
                snippet.trust_score * config.runtime_learning.edge_learn_rate,
                context_hash,
            );
        }
    }
}
```

**Files modified:** `src/reasoning/merge.rs`
**Config addition:** `edge_learn_rate: f32` (default: 0.1) in `RuntimeLearningConfig`
**Tests:**
- `edge_injection_creates_edges_from_evidence`
- `edge_injection_skipped_when_learning_flag_false`
- `injected_edges_tagged_with_query_context`
- `trust_score_scales_edge_weight`

### Task 10.3: Implicit Feedback Reinforcement (`src/reasoning/feedback.rs`)

Extend `FeedbackController::learn()` to track edges walked during response generation. On next user message (implicit accept), strengthen those edges. On explicit correction, weaken them.

> **Note:** This initial implementation uses a simple `last_walked_edges` field. **Task 11K** upgrades this to a `FeedbackQueue` with trace-ID matching to fix the concurrent feedback loss race condition.

```rust
// Add to FeedbackController in src/reasoning/feedback.rs

/// Track which edges were walked during the last response.
pub fn track_walked_edges(&mut self, edges: Vec<(WordId, WordId)>) {
    self.last_walked_edges = edges;
}

/// Called on next user message. If not a correction, strengthen walked edges.
pub fn apply_implicit_feedback(
    &mut self,
    is_correction: bool,
    word_graph: &mut WordGraph,
    config: &RuntimeLearningConfig,
) {
    for (from, to) in &self.last_walked_edges {
        if is_correction {
            word_graph.weaken_edge(*from, *to, config.correction_penalty);
        } else {
            word_graph.strengthen_edge(*from, *to, config.implicit_reinforce_delta);
            // Promote to Core if used enough times
            if let Some(edge) = word_graph.get_edge(*from, *to) {
                if edge.frequency >= config.promotion_threshold {
                    word_graph.set_edge_memory_type(*from, *to, MemoryType::Core);
                }
            }
        }
    }
    self.last_walked_edges.clear();
}
```

**Files modified:** `src/reasoning/feedback.rs`
**Config additions:**
- `implicit_reinforce_delta: f32` (default: 0.05)
- `correction_penalty: f32` (default: 0.15)
- `promotion_threshold: u32` (default: 5)

**Tests:**
- `implicit_accept_strengthens_walked_edges`
- `explicit_correction_weakens_walked_edges`
- `edge_promoted_to_core_after_threshold_uses`
- `cleared_after_feedback_applied`

### Task 10.4: RuntimeLearningConfig (`src/config/mod.rs`, `config/config.yaml`)

Add the config struct and YAML defaults:

```rust
// Add to src/config/mod.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeLearningConfig {
    pub cold_start_edge_threshold: u32,     // default: 3
    pub edge_learn_rate: f32,               // default: 0.1
    pub implicit_reinforce_delta: f32,      // default: 0.05
    pub correction_penalty: f32,            // default: 0.15
    pub promotion_threshold: u32,           // default: 5
}
```

```yaml
# Add to config/config.yaml under layer_5_semantic_map
runtime_learning:
  cold_start_edge_threshold: 3
  edge_learn_rate: 0.1
  implicit_reinforce_delta: 0.05
  correction_penalty: 0.15
  promotion_threshold: 5
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`
**Tests:** Config deserialization round-trip, default value assertions.

### Task 10.5: Wire into Engine (`src/engine.rs`)

Minimal wiring in `process_prompt()`:

1. Pass `learning_flag` from Classification assessment to retrieval request (already in the assessment struct from Task 10.1)
2. After L17 output, call `feedback_controller.track_walked_edges(walked_edges)` (L17 already knows which edges were walked)
3. At start of `process_prompt()`, call `feedback_controller.apply_implicit_feedback(is_correction, word_graph, config)` for the *previous* response

```rust
// At start of process_prompt():
let is_correction = self.detect_correction(&normalized_input);
self.feedback_controller.apply_implicit_feedback(
    is_correction,
    &mut word_graph,
    &self.config.runtime_learning,
);

// After L17 assembles output:
self.feedback_controller.track_walked_edges(walked_edges);
```

**Files modified:** `src/engine.rs`
**Tests:**
- `second_query_uses_injected_edges_without_retrieval`
- `cold_start_hi_learns_and_responds_locally_on_retry`

---

## 12. Phase 11: Hardening Improvements

Eleven targeted improvements to harden Classification accuracy, Reasoning consistency, Predictive stability, on-the-fly learning quality, and cross-system evolution. See Architecture §3.9, §4.3.1, §5.9, §5.10, §4.9 Ext 2b, §6.1.

### Task 11A: Semantic Anchor Probes (`src/classification/signature.rs` + new file)

**Objective:** Extend the 78-float Classification signature to 82 floats by adding 4 binary Semantic Category flags (Rhetorical, Epistemic, Pragmatic, Emotional) matched via FNV hash probe against ~500 Semantic Anchors.

#### Step 1: Semantic Anchor Registry (`src/classification/semantic_anchors.rs`) — NEW FILE

```rust
// src/classification/semantic_anchors.rs — NEW FILE

pub struct SemanticAnchorRegistry {
    anchors: Vec<SemanticAnchor>,
    fingerprint_index: HashMap<u64, Vec<usize>>,
}

pub struct SemanticAnchor {
    pub id: u16,
    pub label: String,
    pub probe_phrases: Vec<String>,
    pub probe_fingerprints: Vec<u64>,
    pub category: SemanticCategory,
    pub weight: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticCategory {
    Rhetorical,
    Epistemic,
    Pragmatic,
    Emotional,
}

impl SemanticAnchorRegistry {
    pub fn load_from_yaml(path: &str) -> Result<Self, ...> { ... }
    pub fn probe(&self, unit_fingerprints: &[u64]) -> [bool; 4] { ... }
}
```

**Files created:** `src/classification/semantic_anchors.rs`
**Files modified:** `src/classification/mod.rs` (add `pub mod semantic_anchors;`)

#### Step 2: Extend ClassificationSignature (`src/classification/signature.rs`)

Add 4 semantic flag dimensions (indices 78–81) to the signature computation:

```rust
// After existing 78-float computation:
let semantic_flags = self.semantic_registry.probe(&unit_fingerprints);
signature.features[78] = if semantic_flags[0] { 1.0 } else { 0.0 }; // Rhetorical
signature.features[79] = if semantic_flags[1] { 1.0 } else { 0.0 }; // Epistemic
signature.features[80] = if semantic_flags[2] { 1.0 } else { 0.0 }; // Pragmatic
signature.features[81] = if semantic_flags[3] { 1.0 } else { 0.0 }; // Emotional
```

**Files modified:** `src/classification/signature.rs`

#### Step 3: Update ClassificationCalculator (`src/classification/calculator.rs`)

Extend centroid comparison to include semantic probe weight:

```rust
let existing_score = weighted_cosine(&query[0..78], &centroid[0..78], existing_weights);
let semantic_score = dot(&query[78..82], &centroid[78..82]);
let final_score = (1.0 - config.semantic_probe_weight) * existing_score
                + config.semantic_probe_weight * semantic_score;
```

**Files modified:** `src/classification/calculator.rs`

#### Step 4: Seed Anchor YAML (`config/semantic_anchors.yaml`) — NEW FILE

Create initial ~500 anchor definitions across 4 categories.

**Files created:** `config/semantic_anchors.yaml`

#### Step 5: Config additions

```yaml
classification:
  semantic_probes:
    enabled: true
    semantic_probe_weight: 0.15
    anchor_fuzzy_threshold: 0.25
    semantic_anchor_count: 500
    anchor_file: "config/semantic_anchors.yaml"
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `semantic_probe_detects_sarcasm_flags_rhetorical`
- `semantic_probe_detects_hypothetical_flags_epistemic`
- `signature_82_float_backward_compatible_with_78`
- `calculator_uses_semantic_weight_in_scoring`

---

### Task 11B: Micro-Validator for Logical Consistency (`src/reasoning/merge.rs`)

**Objective:** Add a lightweight post-merge validation step in L13 that detects numerical contradictions, date ordering violations, and entity-property conflicts before candidates are finalized. Includes **lazy gating** (skip validation when candidates are unambiguous and high-trust) and **L12 pre-extraction** (MetadataSummary) to avoid regex on the critical path.

#### Step 1: Add MetadataSummary to L12 normalization (`src/reasoning/retrieval.rs`)

```rust
// Add to src/reasoning/retrieval.rs (L12 normalization output)

pub struct MetadataSummary {
    pub numbers: Vec<(String, f64, String)>,      // (entity, value, unit)
    pub dates: Vec<(String, String, u16)>,         // (entity, relation, year)
    pub properties: Vec<(String, String, String)>, // (entity, property, value)
}
```

When L12 normalizes HTML/JSON/PDF, run regex extraction **once** and attach `MetadataSummary` to the retrieval result. This moves the expensive work out of the scoring hot path.

**Files modified:** `src/reasoning/retrieval.rs`

#### Step 2: Add MicroValidator with lazy gating to merge.rs

```rust
// Add to src/reasoning/merge.rs

pub struct MicroValidatorResult {
    pub contradictions_found: u32,
    pub penalized_candidates: Vec<Uuid>,
    pub verification_steps: Vec<ReasoningStep>,
    pub needs_human_review: bool,
    pub skipped_reason: Option<String>,  // "clear_winner", "high_trust", "disabled"
}

impl EvidenceMerger {
    pub fn validate_consistency(
        &self,
        merged_candidates: &mut Vec<ScoredCandidate>,
        metadata: &[MetadataSummary],
        config: &MicroValidatorConfig,
    ) -> MicroValidatorResult {
        let mut result = MicroValidatorResult::default();
        if !config.micro_validator_enabled { return result; }

        // LAZY GATING: skip if clear winner with high-trust sources
        if config.lazy_validation_enabled {
            let score_gap = merged_candidates[0].score - merged_candidates.get(1)
                .map_or(0.0, |c| c.score);
            let all_high_trust = merged_candidates.iter()
                .all(|c| c.source_trust >= config.validation_trust_floor);
            if score_gap > config.ambiguity_margin && all_high_trust {
                result.skipped_reason = Some("clear_winner".into());
                return result;
            }
        }

        // Read from pre-computed MetadataSummary (NOT raw text regex)
        self.check_numeric_contradictions(metadata, merged_candidates, config, &mut result);
        self.check_date_ordering(metadata, merged_candidates, config, &mut result);
        self.check_entity_property(metadata, merged_candidates, config, &mut result);

        result
    }
}
```

**Files modified:** `src/reasoning/merge.rs`

#### Step 3: Wire into evidence merge pipeline

After `EvidenceMerger::merge()` returns, call `validate_consistency()` before passing candidates to L14. Add verification steps to `ReasoningTrace`.

**Files modified:** `src/engine.rs`

#### Step 4: Config additions

```yaml
layer_13_evidence_merge:
  micro_validator:
    enabled: true
    numeric_contradiction_threshold: 0.15
    contradiction_penalty: 0.30
    contradiction_override_trust: 0.90
    lazy_validation_enabled: true
    ambiguity_margin: 0.05
    validation_trust_floor: 0.50
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `micro_validator_detects_numeric_contradiction`
- `micro_validator_detects_date_ordering_violation`
- `micro_validator_detects_entity_property_conflict`
- `micro_validator_skipped_when_disabled`
- `high_trust_source_overrides_contradiction`
- `lazy_gating_skips_clear_winner_high_trust`
- `lazy_gating_triggers_on_ambiguous_candidates`
- `metadata_summary_extracted_during_l12_normalization`

---

### Task 11C: Adaptive Dimensionality + Anchor Locking Zones (`src/spatial_index.rs`)

**Objective:** Allow spatial grid cells to expand to 4D/5D when layout energy fails to converge, and confine factual anchor nodes to immutable Semantic Zones.

#### Step 1: Adaptive Dimensionality (`src/spatial_index.rs`)

```rust
// Add to src/spatial_index.rs

pub struct CellDimensionality {
    pub cell_id: usize,
    pub dims: u8,                       // 3, 4, or 5
    pub energy: f32,                    // last layout energy
}

impl SpatialGrid {
    pub fn expand_hot_cells(
        &mut self,
        config: &AdaptiveDimsConfig,
    ) -> Vec<CellDimensionality> {
        if !config.adaptive_dim_enabled { return vec![]; }
        // Identify cells where energy > threshold after layout
        // Expand positions, re-run local layout, keep if energy drops
        ...
    }
}
```

**Files modified:** `src/spatial_index.rs`

#### Step 2: Semantic Zones + Anchor Locking (`src/spatial_index.rs` + new file)

```rust
// src/classification/semantic_zones.rs — NEW FILE

pub struct SemanticZone {
    pub id: ZoneId,
    pub label: String,
    pub bounds: ZoneBounds,
    pub node_filter: ZoneFilter,
    pub lock_strength: f32,
    pub allow_internal_layout: bool,
}

pub struct ZoneManager {
    zones: Vec<SemanticZone>,
}

impl ZoneManager {
    pub fn load_from_yaml(path: &str) -> Result<Self, ...> { ... }
    pub fn clamp_position(&self, node: &WordNode, new_pos: &mut [f32]) { ... }
}
```

**Files created:** `src/classification/semantic_zones.rs`
**Files modified:** `src/classification/mod.rs` (add `pub mod semantic_zones;`)

#### Step 3: Wire zone enforcement into force-directed layout

During position updates in `spatial_index.rs`, after computing attract/repel delta, call `ZoneManager::clamp_position()` to enforce zone bounds.

**Files modified:** `src/spatial_index.rs`

#### Step 4: Config + zone definitions

```yaml
layer_5_semantic_map:
  adaptive_dims:
    enabled: false
    energy_threshold: 0.5
    max_spatial_dims: 5
    perturbation: 0.1
  anchor_locking:
    enabled: true
    zone_definitions_path: "config/semantic_zones.yaml"
```

**Files created:** `config/semantic_zones.yaml`
**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `hot_cell_expands_to_4d_when_energy_high`
- `expanded_cell_projects_to_3d_for_global_queries`
- `number_node_clamped_to_numbers_zone`
- `proper_noun_soft_locked_to_zone`
- `function_word_moderate_lock_allows_some_drift`
- `zone_enforcement_applied_during_layout`

---

### Task 11D: Probabilistic Edge Promotion with TTG Lease (`src/predictive/router.rs`)

**Objective:** Add `EdgeStatus` (Probationary → Episodic → Core) lifecycle to WordEdge. Instead of rapid decay (which causes race conditions with L21 maintenance in concurrent environments), use a **Time-To-Graduation (TTG) lease** — Probationary edges are immune to decay during a configurable lease window, then either graduate or are immediately purged.

**Race condition addressed:** In multi-user or parallel-training scenarios, an edge injected by User A could be pruned by the L21 maintenance worker (running on a 30s cycle) before User A's next turn can traverse it. The TTG lease guarantees survival during the originating session.

#### Step 1: Add EdgeStatus to types

```rust
// Add to src/types.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum EdgeStatus {
    Probationary,   // Just injected from retrieval, under TTG lease
    Episodic,       // Graduated, standard decay
    Core,           // Permanent, no decay
}
```

**Files modified:** `src/types.rs`

#### Step 2: Update WordEdge in router.rs

Add `status: EdgeStatus`, `traversal_count: u32`, and `lease_expires_at: Option<u64>` fields to `WordEdge`. Update `create_or_strengthen_edge()`:
- When called from L13 edge injection: set `status = Probationary`, `lease_expires_at = Some(now_epoch_secs + ttg_lease_duration_secs)`
- When called from training: set `status = Episodic`, `lease_expires_at = None`

**Files modified:** `src/predictive/router.rs`

#### Step 3: Update L15 graph walk

When a Probationary edge is walked, increment `traversal_count`. If `traversal_count >= probationary_graduation_count`:
- Promote `status → Episodic`
- Clear `lease_expires_at = None`

**Files modified:** `src/engine.rs` (L15 walk logic)

#### Step 4: Update L21 maintenance — TTG lease enforcement

Replace rapid decay with lease-based lifecycle:
```rust
// In L21 maintenance cycle:
for edge in edges.iter_mut() {
    match edge.status {
        EdgeStatus::Probationary => {
            if let Some(expires) = edge.lease_expires_at {
                if now_epoch_secs < expires {
                    continue; // IMMUNE: skip decay entirely during lease
                }
                // Lease expired
                if edge.traversal_count >= config.probationary_graduation_count {
                    edge.status = EdgeStatus::Episodic; // late graduation
                    edge.lease_expires_at = None;
                } else {
                    edges_to_purge.push(edge.id); // IMMEDIATE PURGE
                }
            }
        }
        EdgeStatus::Episodic => { /* standard decay */ }
        EdgeStatus::Core => { /* no decay */ }
    }
}
```

**Files modified:** `src/predictive/router.rs` (maintenance/decay logic)

#### Step 5: Config additions

```yaml
layer_5_semantic_map:
  runtime_learning:
    ttg_lease_duration_secs: 300
    probationary_graduation_count: 2
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `injected_edge_starts_as_probationary_with_lease`
- `probationary_edge_immune_to_decay_during_lease`
- `probationary_edge_graduates_after_two_traversals`
- `expired_lease_purges_ungraduated_edge`
- `expired_lease_graduates_edge_with_enough_traversals`
- `graduated_edge_follows_normal_decay`
- `concurrent_maintenance_does_not_prune_leased_edge`

---

### Task 11E: Structural Feedback Loop with Anti-Thrashing (`src/reasoning/feedback.rs` + `src/engine.rs`)

**Objective:** Track Tier 3 pathfinding overuse per intent class. When an intent's Tier 3 rate exceeds 40% for N consecutive windows, propose splitting. Includes **hysteresis bands** (20%–40% dead zone), **min-viability windows** (freeze after split), and **data inheritance** (warm-start sub-centroids) to prevent taxonomy thrashing.

#### Step 1: Tier 3 Overuse Tracker with Hysteresis

```rust
// Add to src/reasoning/feedback.rs

pub struct Tier3OveruseTracker {
    intent_tier3_rates: HashMap<IntentKind, (u64, u64)>,
    /// Consecutive windows above split threshold per intent
    consecutive_above: HashMap<IntentKind, u32>,
    /// Consecutive windows below merge threshold per intent
    consecutive_below: HashMap<IntentKind, u32>,
    /// Freeze: sub-intents created after this sweep cannot be merged
    viability_freeze: HashMap<IntentKind, u32>,  // intent → freeze_until_sweep
    window_size: u64,
    split_proposal_threshold: f32,     // default: 0.40
    merge_proposal_threshold: f32,     // default: 0.20 (hysteresis gap)
    consecutive_windows_required: u32, // default: 3
    min_viability_sweeps: u32,         // default: 3
}

pub struct IntentSplitProposal {
    pub intent: IntentKind,
    pub tier3_rate: f32,
    pub sample_count: u64,
    pub proposed_at: DateTime<Utc>,
}

pub struct IntentMergeProposal {
    pub sub_intents: Vec<IntentKind>,
    pub parent_intent: IntentKind,
    pub tier3_rate: f32,
}

impl Tier3OveruseTracker {
    pub fn record(&mut self, intent: IntentKind, used_tier3: bool) { ... }
    pub fn check_split_proposals(&self) -> Vec<IntentSplitProposal> { ... }
    pub fn check_merge_proposals(&self, current_sweep: u32) -> Vec<IntentMergeProposal> {
        // Only propose merge if:
        // 1. consecutive_below >= consecutive_windows_required
        // 2. viability_freeze[intent] <= current_sweep (freeze expired)
        ...
    }
}
```

**Files modified:** `src/reasoning/feedback.rs`

#### Step 2: Wire tracker into L15/L18

After each L15 graph walk step, record whether Tier 3 was used via `Tier3OveruseTracker::record()`. In L18 feedback, periodically call `check_split_proposals()` and `check_merge_proposals()`, log to telemetry.

**Files modified:** `src/engine.rs`

#### Step 3: Training sweep integration with data inheritance

During `training_sweep`, check for pending proposals:

```rust
// Split execution with data inheritance:
for proposal in split_proposals {
    let parent_centroid = get_centroid(proposal.intent);
    let clusters = k_means_cluster(recent_queries[proposal.intent], k=2..4);
    for (i, cluster) in clusters.iter().enumerate() {
        let sub_centroid = parent_centroid + cluster.offset; // NOT zero-init
        create_sub_intent(proposal.intent, i, sub_centroid);
    }
    // Set viability freeze
    tracker.viability_freeze.insert(proposal.intent, current_sweep + min_viability_sweeps);
}
```

**Files modified:** `src/training/pipeline.rs`

#### Step 4: Config additions

```yaml
structural_feedback:
  enabled: false
  split_proposal_threshold: 0.40
  merge_proposal_threshold: 0.20
  min_cluster_separation: 0.30
  auto_split_confidence: 0.85
  split_window_size: 1000
  max_intent_count: 48
  consecutive_windows_required: 3
  min_viability_sweeps: 3
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `tier3_tracker_records_and_computes_rates`
- `split_proposed_when_tier3_rate_exceeds_threshold_for_n_windows`
- `no_proposal_in_hysteresis_dead_zone`
- `merge_proposed_when_below_merge_threshold_for_n_windows`
- `viability_freeze_prevents_premature_merge`
- `data_inheritance_warm_starts_sub_centroids`
- `max_intent_count_prevents_over_splitting`
- `split_proposal_logged_to_telemetry`
- `single_window_spike_does_not_trigger_split`

---

### Task 11F: POS-Weighted Perturbation for SemanticHasher (`src/predictive/router.rs`)

**Objective:** Update the `SemanticHasher` initialization to use POS-based cluster offsets and context-seeded perturbation, reducing polysemy spatial collision at node creation time. See Architecture §5.2 "Domain-Anchor Blending".

#### Step 1: Define POS Cluster Centroids (`config/pos_clusters.yaml`) — NEW FILE

```yaml
# config/pos_clusters.yaml
clusters:
  Noun:     [0.7, 0.8, 0.3]  # Object Cluster
  Verb:     [0.3, 0.3, 0.7]  # Action Cluster
  Adjective: [0.5, 0.5, 0.5] # Modifier Cluster
  Adverb:   [0.5, 0.5, 0.5]  # Modifier Cluster
  Function: [0.5, 0.5, 0.5]  # Center (no offset)
```

**Files created:** `config/pos_clusters.yaml`

#### Step 2: Update SemanticHasher in router.rs

Modify `hash_to_3d()` (or equivalent) to accept POS tag and optional sentence neighbors:

```rust
impl SemanticHasher {
    pub fn hash_to_3d_blended(
        &self,
        word: &str,
        pos_tag: Option<&str>,
        neighbors: Option<(&str, &str)>,
        config: &PosClusterConfig,
    ) -> [f32; 3] {
        let base = self.hash_to_3d(word);

        // POS-based offset
        let offset = match pos_tag {
            Some(tag) => self.pos_cluster_offset(tag, config.pos_offset_strength),
            None => [0.0; 3],
        };

        // Context-seeded perturbation
        let perturbation = match neighbors {
            Some((n1, n2)) => {
                let ctx = self.hash_to_3d(&format!("{}{}", n1, n2));
                [ctx[0] * config.context_seed_strength,
                 ctx[1] * config.context_seed_strength,
                 ctx[2] * config.context_seed_strength]
            }
            None => [0.0; 3],
        };

        [base[0] + offset[0] + perturbation[0],
         base[1] + offset[1] + perturbation[1],
         base[2] + offset[2] + perturbation[2]]
    }
}
```

**Files modified:** `src/predictive/router.rs`

#### Step 3: Wire into vocabulary bootstrap and runtime growth

- Dictionary pre-population (Phase 1): calls `hash_to_3d_blended(word, Some(pos_tag), None, config)` — POS offset only, no context
- Runtime growth (Phase 3): calls `hash_to_3d_blended(word, Some(pos_tag), Some((n1, n2)), config)` — full blending

**Files modified:** `src/predictive/router.rs`, `src/engine.rs`

#### Step 4: Config additions

```yaml
layer_5_semantic_map:
  pos_clusters:
    pos_offset_strength: 0.05
    context_seed_strength: 0.10
    clusters_path: "config/pos_clusters.yaml"
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `blended_hash_differs_from_pure_fnv`
- `noun_offset_shifts_toward_object_cluster`
- `same_word_different_context_gets_different_position`
- `function_words_get_no_pos_offset`
- `dictionary_prepop_uses_pos_only_blend`
- `runtime_growth_uses_full_context_blend`

---

### Task 11G: Function Word Hub Management (`src/predictive/router.rs`)

**Objective:** Prevent Function-word hub saturation by enforcing edge caps with recency-aware aging, adding semantic domain gating on hub edges during Tier 3 pathfinding, and promoting high-connectivity Content words to secondary hub status. See Architecture §5.11.

#### Step 1: Edge caps with recency-aware aging

Add hub pruning logic to L21 maintenance in `router.rs`:

```rust
impl WordGraph {
    pub fn prune_hub_edges(&mut self, config: &HubManagementConfig) {
        for hub in self.nodes.values_mut().filter(|n| n.node_type == NodeType::Function) {
            let edge_count = self.edges_from(hub.id).count();
            if edge_count <= config.max_edges_per_hub as usize { continue; }

            let mut edges: Vec<_> = self.edges_from(hub.id).collect();
            edges.sort_by(|a, b| {
                let score_a = a.weight * recency_factor(a.last_reinforced);
                let score_b = b.weight * recency_factor(b.last_reinforced);
                score_a.partial_cmp(&score_b).unwrap()
            });

            // Prune lowest-scoring edges, keeping min_hub_edges
            // NEVER prune edges to other Function words
            let to_prune = edge_count - config.max_edges_per_hub as usize;
            let mut pruned = 0;
            for edge in &edges {
                if pruned >= to_prune { break; }
                let target = self.node(edge.to);
                if target.node_type == NodeType::Function { continue; }
                self.remove_edge(edge.from, edge.to);
                pruned += 1;
            }
        }
    }
}
```

**Files modified:** `src/predictive/router.rs`

#### Step 2: Domain tagging on edges

Add `domain_tags: SmallVec<[u64; 2]>` field to `WordEdge`. During edge creation/reinforcement, derive domain tags from the target node's context tags. Store as hashed fingerprints.

**Files modified:** `src/predictive/router.rs`

#### Step 3: Semantic gating during Tier 3 A*

In the L15 graph walk, when expanding a Function-word hub during A* pathfinding:
- Retrieve current context domain from Classification + Reasoning
- Filter `hub.edges.filter(|e| e.domain_tags.intersects(current_domain))`
- Only score filtered edges (reduces candidate set ~85%)

**Files modified:** `src/engine.rs` (L15 walk logic)

#### Step 4: Secondary hub promotion

During L21 maintenance, check Content nodes for promotion:

```rust
for node in self.nodes.values_mut() {
    if node.node_type != NodeType::Content { continue; }
    let edge_count = self.edges_from(node.id).count();
    if edge_count > config.secondary_hub_threshold as usize
        && node.frequency > config.secondary_hub_min_frequency {
        node.is_secondary_hub = true;
    } else {
        node.is_secondary_hub = false; // can be demoted
    }
}
```

In L15 A* pathfinding, use `secondary_hub_priority` (default: 0.7) as heuristic weight for secondary hubs, vs 1.0 for primary Function-word hubs and 0.3 for regular nodes.

**Files modified:** `src/predictive/router.rs`, `src/engine.rs`

#### Step 5: Config additions

```yaml
layer_5_semantic_map:
  hub_management:
    max_edges_per_hub: 2000
    min_hub_edges: 500
    hub_prune_threshold: 0.05
    hub_domain_gating_enabled: true
    secondary_hub_threshold: 200
    secondary_hub_min_frequency: 1000
    secondary_hub_priority: 0.7
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `hub_edges_pruned_when_exceeding_max`
- `inter_function_word_edges_never_pruned`
- `domain_gating_filters_irrelevant_hub_edges`
- `domain_gated_astar_reduces_candidate_set`
- `content_node_promoted_to_secondary_hub`
- `secondary_hub_demoted_when_below_threshold`
- `astar_uses_secondary_hub_priority`
- `min_hub_edges_floor_respected`

---

### Task 11H: Bloom-Filtered Context Summaries (`src/predictive/router.rs`)

**Objective:** Replace the fixed-capacity `SmallVec<[u64; 4]>` context tag storage on WordEdge with a **Bloom filter + dominant cluster ID** to support infinite context history in constant memory. See Architecture §5.1 "Context Tag Scalability".

#### Step 1: Add ContextBloom to WordEdge

```rust
// Add to src/predictive/router.rs (WordEdge extension)

/// Per-edge Bloom filter for context gating (replaces SmallVec for mature edges)
pub struct ContextBloom {
    bits: [u8; 32],  // 256-bit Bloom filter, fixed size
}

impl ContextBloom {
    pub fn insert(&mut self, hash: u64) { /* set k=3 bit positions */ }
    pub fn probably_contains(&self, hash: u64) -> bool { /* check k=3 positions */ }
    pub fn from_smallvec(tags: &SmallVec<[u64; 4]>) -> Self {
        let mut bloom = Self::default();
        for tag in tags { bloom.insert(*tag); }
        bloom
    }
}
```

**Files modified:** `src/predictive/router.rs`

#### Step 2: Add migration logic during edge reinforcement

When `context_tags.len() >= context_bloom_migration_threshold` (default: 4):
- Migrate all tags into a new `ContextBloom`
- Set `edge.context_bloom = Some(bloom)`
- Clear `edge.context_tags` (SmallVec no longer used for this edge)

**Files modified:** `src/predictive/router.rs`

#### Step 3: Add dominant cluster tracking in L21 maintenance

During maintenance, for edges with `context_bloom.is_some()`:
- Cluster recent context hashes (tracked via a small ring buffer or sampling)
- If >80% of recent activations belong to one cluster: set `edge.dominant_context_cluster = Some(cluster_id)`
- Fast-path: if `query_cluster == edge.dominant_context_cluster` → force-activate (skip Bloom check)

**Files modified:** `src/predictive/router.rs` (maintenance logic)

#### Step 4: Update context gating in L15 graph walk

```rust
// In L15 edge activation check:
fn is_edge_active(edge: &WordEdge, query_hash: u64, query_cluster: Option<u32>) -> bool {
    // Fast path: dominant cluster match
    if let (Some(qc), Some(dc)) = (query_cluster, edge.dominant_context_cluster) {
        if qc == dc { return true; }
    }
    // Bloom path (mature edges)
    if let Some(ref bloom) = edge.context_bloom {
        return bloom.probably_contains(query_hash);
    }
    // SmallVec path (young edges)
    edge.context_tags.contains(&query_hash) || edge.context_tags.is_empty()
}
```

**Files modified:** `src/engine.rs` (L15 walk logic)

#### Step 5: Config additions

```yaml
layer_5_semantic_map:
  context_bloom:
    bloom_size_bytes: 32
    dominant_cluster_threshold: 0.80
    migration_threshold: 4
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `bloom_filter_insert_and_contains`
- `bloom_false_positive_rate_within_bounds`
- `smallvec_migrates_to_bloom_at_threshold`
- `dominant_cluster_fast_path_activates_edge`
- `young_edge_uses_smallvec_gating`
- `mature_edge_uses_bloom_gating`
- `empty_context_tags_activates_unconditionally`

---

### Task 11I: Silent Training Intent Quarantine (`src/training/pipeline.rs`)

**Objective:** Prevent noisy Silent Training jobs from polluting the global Intent channel by routing new ClassificationPatterns to a session-scoped quarantine buffer with post-job validation. See Architecture §11.7 "Silent Training Intent Quarantine".

#### Step 1: Add SessionIntentBuffer

```rust
// Add to src/training/pipeline.rs

pub struct SessionIntentBuffer {
    patterns: Vec<ClassificationPattern>,
    source_job_id: String,
}

impl SessionIntentBuffer {
    pub fn add_pattern(&mut self, pattern: ClassificationPattern) { ... }
    pub fn validate_and_merge(
        self,
        existing_centroids: &[Centroid],
        config: &QuarantineConfig,
    ) -> QuarantineResult { ... }
}

pub struct QuarantineResult {
    pub merged: Vec<ClassificationPattern>,
    pub purged: Vec<(ClassificationPattern, String)>,  // (pattern, reason)
}
```

**Files modified:** `src/training/pipeline.rs`

#### Step 2: Wire into Silent Training path

During `train_batch()` when `training_mode == Silent`:
- Route new `ClassificationPattern` discoveries to `SessionIntentBuffer` instead of the persistent Intent channel
- Units and memory ingested normally (Main channel)

**Files modified:** `src/training/pipeline.rs`, `src/classification/trainer.rs`

#### Step 3: Post-job validation gate

After Silent Training job completes:
```rust
let result = buffer.validate_and_merge(&existing_centroids, &config.quarantine);
for (pattern, reason) in &result.purged {
    telemetry.log_quarantined_intent(pattern, reason);
}
for pattern in result.merged {
    intent_channel.insert(pattern);
}
```

Validation checks:
- **Consistency:** `max_cosine_similarity(P.centroid, existing) > conflict_threshold` → purge
- **Frequency/Utility:** `P.source_doc_count == 1 AND P.example_count < min_examples` → purge

**Files modified:** `src/training/pipeline.rs`, `src/engine.rs`

#### Step 4: Config additions

```yaml
silent_training:
  quarantine:
    enabled: true
    conflict_threshold: 0.85
    min_examples: 10
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `quarantine_buffers_patterns_during_silent_training`
- `quarantine_purges_conflicting_patterns`
- `quarantine_purges_single_doc_low_count_patterns`
- `quarantine_merges_valid_patterns`
- `quarantine_logs_purged_patterns_to_telemetry`
- `normal_training_bypasses_quarantine`

---

### Task 11J: Dynamic Hub Election + Language-Aware Bootstrap (`src/predictive/router.rs`)

**Objective:** Enable the Word Graph to self-organize routing hubs based on actual usage patterns (betweenness centrality), supporting non-English and domain-specific corpora where standard Function words may be absent. See Architecture §5.11 "Dynamic Hub Election for Low-Resource Scenarios".

#### Step 1: Approximate betweenness centrality in L21 maintenance

```rust
// Add to src/predictive/router.rs (maintenance logic)

impl WordGraph {
    pub fn compute_approximate_centrality(
        &self,
        sample_pairs: u32,
        min_frequency: u32,
    ) -> HashMap<WordId, f32> {
        let eligible: Vec<_> = self.nodes.values()
            .filter(|n| n.frequency > min_frequency)
            .collect();
        let mut centrality: HashMap<WordId, f32> = HashMap::new();

        for _ in 0..sample_pairs {
            let (src, dst) = random_pair(&eligible);
            if let Some(path) = self.shortest_path(src.id, dst.id) {
                for &node_id in &path[1..path.len()-1] {
                    *centrality.entry(node_id).or_default() += 1.0;
                }
            }
        }
        // Normalize by sample count
        let norm = sample_pairs as f32;
        for v in centrality.values_mut() { *v /= norm; }
        centrality
    }
}
```

**Files modified:** `src/predictive/router.rs`

#### Step 2: Extend secondary hub promotion with centrality signal

During L21 maintenance, after computing centrality:
- If Content word C has `centrality(C) > dynamic_hub_centrality_threshold` AND meets existing secondary hub criteria → promote to `is_secondary_hub = true`
- This catches domain-critical words with high routing importance but moderate edge count

**Files modified:** `src/predictive/router.rs`

#### Step 3: Language-aware function word bootstrap

When L1 detects non-English input (character distribution / script detection):
- Load `config/function_words/{language_code}.yaml`
- Seed graph with these words as `NodeType::Function`
- Fallback: if no language list exists, use top-50 highest-frequency words from first training batch as provisional function words

**Files modified:** `src/predictive/router.rs`, `src/classification/input.rs`

#### Step 4: Config additions

```yaml
layer_5_semantic_map:
  hub_management:
    dynamic_hub_centrality_threshold: 0.15
    hub_election_min_frequency: 500
    hub_centrality_sample_pairs: 100
    language_function_words_dir: "config/function_words/"
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Files created:** `config/function_words/en.yaml` (English function words, used as reference)

**Tests:**
- `approximate_centrality_identifies_high_routing_nodes`
- `dynamic_hub_election_promotes_domain_hub`
- `dynamic_hub_demotion_when_centrality_drops`
- `language_aware_bootstrap_loads_function_words`
- `fallback_uses_top_frequency_words`
- `non_english_corpus_self_organizes_hubs`

---

### Task 11K: Trace-ID Tagged Feedback Queue (`src/reasoning/feedback.rs`)

**Objective:** Replace the mutable `last_walked_edges` field on `FeedbackController` with a **trace-ID tagged feedback queue** to correctly attribute implicit feedback in concurrent and multi-turn scenarios. See Architecture §4.9 Extension 3.

#### Step 1: Add FeedbackQueue data structure

```rust
// Add to src/reasoning/feedback.rs

pub struct WalkEvent {
    pub trace_id: u64,
    pub edges: Vec<(WordId, WordId)>,
    pub created_at: u64,  // epoch seconds
}

pub struct FeedbackQueue {
    events: VecDeque<WalkEvent>,
    max_size: usize,
}

impl FeedbackQueue {
    pub fn push(&mut self, event: WalkEvent) {
        if self.events.len() >= self.max_size {
            self.events.pop_front(); // evict oldest
        }
        self.events.push_back(event);
    }
    pub fn find_by_trace(&mut self, trace_id: u64) -> Option<WalkEvent> {
        if let Some(pos) = self.events.iter().position(|e| e.trace_id == trace_id) {
            Some(self.events.remove(pos).unwrap())
        } else { None }
    }
    pub fn expire_before(&mut self, cutoff: u64) {
        self.events.retain(|e| e.created_at > cutoff);
    }
}
```

**Files modified:** `src/reasoning/feedback.rs`

#### Step 2: Replace `last_walked_edges` with queue

Remove the `last_walked_edges: Vec<(WordId, WordId)>` field from `FeedbackController`. Replace with `feedback_queue: FeedbackQueue`.

Update `track_walked_edges()`:
```rust
pub fn track_walked_edges(&mut self, trace_id: u64, edges: Vec<(WordId, WordId)>) {
    self.feedback_queue.push(WalkEvent {
        trace_id,
        edges,
        created_at: now_epoch_secs(),
    });
}
```

**Files modified:** `src/reasoning/feedback.rs`

#### Step 3: Update `apply_implicit_feedback()` with trace matching

```rust
pub fn apply_implicit_feedback(
    &mut self,
    parent_trace_id: u64,
    is_correction: bool,
    word_graph: &mut WordGraph,
    config: &RuntimeLearningConfig,
) {
    // TTL cleanup
    let cutoff = now_epoch_secs() - config.feedback_queue_ttl_secs;
    self.feedback_queue.expire_before(cutoff);

    // Find matching event
    if let Some(event) = self.feedback_queue.find_by_trace(parent_trace_id) {
        for (from, to) in &event.edges {
            if is_correction {
                word_graph.weaken_edge(*from, *to, config.correction_penalty);
            } else {
                word_graph.strengthen_edge(*from, *to, config.implicit_reinforce_delta);
                if let Some(edge) = word_graph.get_edge(*from, *to) {
                    if edge.frequency >= config.promotion_threshold {
                        word_graph.set_edge_memory_type(*from, *to, MemoryType::Core);
                    }
                }
            }
        }
    }
}
```

**Files modified:** `src/reasoning/feedback.rs`

#### Step 4: Wire trace_id through engine

In `process_prompt()`:
- Generate `trace_id` at response start (e.g., `text_fingerprint(&format!("{}{}", session_id, turn_count))`)
- After L17 output: `feedback_controller.track_walked_edges(trace_id, walked_edges)`
- At start of next `process_prompt()`: `feedback_controller.apply_implicit_feedback(parent_trace_id, is_correction, word_graph, config)`
- The `parent_trace_id` comes from the previous assistant message's trace_id (stored in session state)

**Files modified:** `src/engine.rs`

#### Step 5: Config additions

```yaml
layer_5_semantic_map:
  runtime_learning:
    feedback_queue_ttl_secs: 600
    feedback_queue_max_size: 64
```

**Files modified:** `src/config/mod.rs`, `config/config.yaml`

**Tests:**
- `feedback_queue_push_and_find_by_trace`
- `feedback_queue_evicts_oldest_at_max_size`
- `feedback_queue_expires_old_events`
- `trace_matched_feedback_strengthens_correct_edges`
- `concurrent_turns_reinforce_independently`
- `unmatched_trace_is_no_op`
- `correction_weakens_trace_matched_edges`

---

## 13. Config Changes Summary

All new config fields with their defaults:

| Section | Field | Type | Default | Purpose |
|---------|-------|------|---------|---------|
| `system_training.classification` | `centroid_batch_size` | `usize` | 500 | Batch size for centroid construction |
| | `weight_sweep_iterations` | `usize` | 10 | Sweep iterations for weight optimization |
| | `calibration_bins` | `usize` | 10 | Confidence calibration bands |
| | `validation_split` | `f32` | 0.1 | Train/val split ratio |
| | `min_examples_per_intent` | `usize` | 100 | Minimum examples per intent class |
| `system_training.reasoning` | `scoring_sweep_iterations` | `usize` | 20 | 7D weight sweep iterations |
| | `decomposition_min_examples` | `usize` | 50 | Min examples to learn a template |
| | `mrr_target` | `f32` | 0.70 | Target MRR for early stopping |
| | `threshold_sweep_iterations` | `usize` | 10 | Threshold sweep iterations |
| `system_training.predictive` | `attract_strength` | `f32` | 0.02 | Attract force for correct predictions |
| | `repel_strength` | `f32` | 0.01 | Repel force for incorrect predictions |
| | `correct_attract_bonus` | `f32` | 0.01 | Extra attract for correct |
| | `mini_batch_size` | `usize` | 32 | Sequences per position update batch |
| | `max_walk_steps` | `usize` | 50 | Max steps per training walk |
| | `convergence_tolerance` | `f32` | 0.001 | Stop when delta below this |
| | `temperature_anneal_rate` | `f32` | 0.95 | Temperature decay per step |
| | `momentum` | `f32` | 0.9 | EMA momentum for position updates |
| | `walk_sweep_iterations` | `usize` | 10 | Walk parameter sweep iterations |
| `system_training.consistency` | `r1_threshold_adjustment` | `f32` | 0.05 | R1 correction magnitude |
| | `r3_anchor_adjustment` | `f32` | 0.05 | R3 correction magnitude |
| | `r4_confidence_adjustment` | `f32` | 0.05 | R4 correction magnitude |
| | `max_correction_rounds` | `usize` | 3 | Max iterative correction rounds |
| `classification` | `two_phase_structural_margin` | `f32` | 0.3 | Margin for fast-path structural-only classification |
| | `social_shortcircuit_confidence` | `f32` | 0.95 | Min confidence for social intent short-circuit |
| | `platt_a` | `f32` | 1.0 | Platt scaling parameter A |
| | `platt_b` | `f32` | 0.0 | Platt scaling parameter B |
| `layer_5_semantic_map.runtime_learning` | `cold_start_edge_threshold` | `u32` | 3 | Min outgoing edges before word is "cold" |
| | `edge_learn_rate` | `f32` | 0.1 | Trust-to-weight scaling for injected edges |
| | `implicit_reinforce_delta` | `f32` | 0.05 | Edge weight boost on implicit accept |
| | `correction_penalty` | `f32` | 0.15 | Edge weight penalty on explicit correction |
| | `promotion_threshold` | `u32` | 5 | Uses before Episodic edge promotes to Core |
| | `ttg_lease_duration_secs` | `u64` | 300 | TTG lease duration (5 min) for Probationary edges |
| | `probationary_graduation_count` | `u32` | 2 | Traversals needed to graduate to Episodic |
| `classification.semantic_probes` | `enabled` | `bool` | true | Enable semantic anchor probing |
| | `semantic_probe_weight` | `f32` | 0.15 | Weight of semantic flags in scoring |
| | `anchor_fuzzy_threshold` | `f32` | 0.25 | Levenshtein threshold for fuzzy match |
| | `semantic_anchor_count` | `u32` | 500 | Number of semantic anchors |
| | `anchor_file` | `String` | "config/semantic_anchors.yaml" | Path to anchor definitions |
| `layer_13_evidence_merge.micro_validator` | `enabled` | `bool` | true | Enable micro-validator |
| | `numeric_contradiction_threshold` | `f32` | 0.15 | Relative spread before flagging |
| | `contradiction_penalty` | `f32` | 0.30 | Confidence reduction for contradictions |
| | `contradiction_override_trust` | `f32` | 0.90 | Trust level that overrides check |
| | `lazy_validation_enabled` | `bool` | true | Skip validation for clear winners |
| | `ambiguity_margin` | `f32` | 0.05 | Score gap below which validation triggers |
| | `validation_trust_floor` | `f32` | 0.50 | Source trust below which validation always triggers |
| `layer_5_semantic_map.adaptive_dims` | `enabled` | `bool` | false | Enable adaptive dimensionality |
| | `energy_threshold` | `f32` | 0.5 | Per-cell energy to trigger expansion |
| | `max_spatial_dims` | `u8` | 5 | Hard ceiling on dimensions |
| | `perturbation` | `f32` | 0.1 | Initial noise for new dimension |
| `layer_5_semantic_map.anchor_locking` | `enabled` | `bool` | true | Enable anchor zone locking |
| | `zone_definitions_path` | `String` | "config/semantic_zones.yaml" | Path to zone definitions |
| `layer_5_semantic_map.pos_clusters` | `pos_offset_strength` | `f32` | 0.05 | POS-based cluster offset magnitude |
| | `context_seed_strength` | `f32` | 0.10 | Context-seeded perturbation magnitude |
| | `clusters_path` | `String` | "config/pos_clusters.yaml" | Path to POS cluster centroids |
| `layer_5_semantic_map.hub_management` | `max_edges_per_hub` | `u32` | 2000 | Hard cap on Function-word hub edges |
| | `min_hub_edges` | `u32` | 500 | Floor to maintain routing viability |
| | `hub_prune_threshold` | `f32` | 0.05 | Composite score below which edges pruned |
| | `hub_domain_gating_enabled` | `bool` | true | Enable domain-based edge masking |
| | `secondary_hub_threshold` | `u32` | 200 | Edge count for secondary hub promotion |
| | `secondary_hub_min_frequency` | `u32` | 1000 | Min traversal frequency for secondary hub |
| | `secondary_hub_priority` | `f32` | 0.7 | A* priority weight for secondary hubs |
| `structural_feedback` | `enabled` | `bool` | false | Enable structural feedback loop |
| | `split_proposal_threshold` | `f32` | 0.40 | Tier 3 rate triggering split proposal |
| | `merge_proposal_threshold` | `f32` | 0.20 | Tier 3 rate below which merge proposed |
| | `min_cluster_separation` | `f32` | 0.30 | Min cosine distance between sub-intents |
| | `auto_split_confidence` | `f32` | 0.85 | Confidence for auto-applying splits |
| | `split_window_size` | `u64` | 1000 | Rolling window for Tier 3 tracking |
| | `max_intent_count` | `u32` | 48 | Max total intent variants |
| | `consecutive_windows_required` | `u32` | 3 | Windows above/below threshold before action |
| | `min_viability_sweeps` | `u32` | 3 | Freeze period after split (training sweeps) |
| `layer_5_semantic_map.context_bloom` | `bloom_size_bytes` | `u8` | 32 | Per-edge Bloom filter size in bytes |
| | `dominant_cluster_threshold` | `f32` | 0.80 | Activation fraction before setting dominant cluster |
| | `migration_threshold` | `u32` | 4 | SmallVec entries before migrating to Bloom |
| `layer_5_semantic_map.runtime_learning` | `feedback_queue_ttl_secs` | `u64` | 600 | TTL for WalkEvents in feedback queue |
| | `feedback_queue_max_size` | `usize` | 64 | Max WalkEvents in feedback queue |
| `layer_5_semantic_map.hub_management` | `dynamic_hub_centrality_threshold` | `f32` | 0.15 | Betweenness centrality for dynamic hub election |
| | `hub_election_min_frequency` | `u32` | 500 | Min frequency before centrality computed |
| | `hub_centrality_sample_pairs` | `u32` | 100 | Node pairs sampled per maintenance cycle |
| | `language_function_words_dir` | `String` | "config/function_words/" | Per-language function word lists |
| `silent_training.quarantine` | `enabled` | `bool` | true | Enable intent quarantine for silent training |
| | `conflict_threshold` | `f32` | 0.85 | Cosine similarity above which pattern conflicts |
| | `min_examples` | `u32` | 10 | Min examples before pattern is considered valid |

---

## 14. File Manifest

### New Files (16)

| File | Phase | Purpose |
|------|-------|---------|
| `src/reasoning/decomposer.rs` | 3 | Query decomposition for multi-hop reasoning |
| `src/predictive/walk_trainer.rs` | 4 | Word Graph edge formation trainer |
| `src/reasoning/consistency.rs` | 5 | Cross-system consistency checker |
| `src/seed/classification_generator.rs` | 6 | Classification dataset generator (50K+) |
| `src/seed/reasoning_generator.rs` | 6 | Reasoning QA dataset generator (20K+) |
| `src/seed/predictive_generator.rs` | 6 | Predictive sequence generator (100K+) |
| `src/seed/consistency_generator.rs` | 6 | Consistency validation generator (5K+) |
| `src/bin/training_sweep.rs` | 8 | Per-system training sweep binary |
| `tests/training_integration.rs` | 9 | End-to-end training integration tests |
| `docs/CODING_PLAN_TRAINING_ARCHITECTURE.md` | — | This document |
| `src/classification/semantic_anchors.rs` | 11A | Semantic Anchor Registry + probe matching |
| `config/semantic_anchors.yaml` | 11A | ~500 semantic anchor definitions |
| `src/classification/semantic_zones.rs` | 11C | Semantic Zone definitions + ZoneManager |
| `config/semantic_zones.yaml` | 11C | Default zone coordinate definitions |
| `config/pos_clusters.yaml` | 11F | POS cluster centroids for domain-anchor blending |
| `config/function_words/en.yaml` | 11J | English function words reference list |

### Modified Files (~22)

| File | Phase | Changes |
|------|-------|---------|
| `src/types.rs` | 1, 11D | Add loss types, SystemTrainingReport, EdgeStatus enum |
| `src/seed/mod.rs` | 1 | Extend TrainingExample, add new module exports |
| `src/config/mod.rs` | 1, 10, 11 | Add SystemTrainingConfig, RuntimeLearningConfig, SemanticProbeConfig, MicroValidatorConfig, AdaptiveDimsConfig, AnchorLockingConfig, StructuralFeedbackConfig, PosClusterConfig, HubManagementConfig, ContextBloomConfig, QuarantineConfig |
| `config/config.yaml` | 1, 10, 11 | Add system_training, runtime_learning, semantic_probes, micro_validator, adaptive_dims, anchor_locking, structural_feedback, pos_clusters, hub_management, context_bloom, quarantine sections |
| `src/memory/store.rs` | 1 | Add set/update centroid methods |
| `src/classification/trainer.rs` | 2, 11I | Full rewrite: centroid + sweep + calibration; quarantine buffer wiring |
| `src/classification/calculator.rs` | 2, 7, 11A | Add evaluate_loss(), two-phase, calibration, semantic probe scoring |
| `src/classification/signature.rs` | 7, 11A | Add POS tag LRU cache, extend to 82-float with semantic flags |
| `src/classification/mod.rs` | 11A, 11C | Add semantic_anchors, semantic_zones module exports |
| `src/classification/input.rs` | 11J | Add language/script detection for function word bootstrap |
| `src/engine.rs` | 2, 3, 4, 5, 7, 10, 11B, 11D, 11E, 11F, 11G, 11H, 11K | Add train_*() methods, reasoning cache, short-circuit, wire on-the-fly learning, micro-validator, TTG lease, Tier 3 tracking, hub domain gating, context bloom walk gating, trace_id wiring |
| `src/reasoning/mod.rs` | 3, 5 | Add decomposer, consistency module exports |
| `src/predictive/mod.rs` | 4 | Add walk_trainer module export |
| `src/reasoning/search.rs` | 3, 7 | Add evaluate_mrr(), delta_rescore_evidence() |
| `src/spatial_index.rs` | 4, 7, 11C | Add update_position(), batch_update_positions(), neighbor cache, adaptive dims, zone enforcement |
| `src/training/pipeline.rs` | 3, 11E, 11I | Wire train_reasoning(), structural feedback split/merge proposals with data inheritance, SessionIntentBuffer quarantine |
| `src/classification/intent.rs` | 10 | Add cold-start detection to assess() |
| `src/reasoning/retrieval.rs` | 11B | Add MetadataSummary extraction during L12 normalization |
| `src/reasoning/merge.rs` | 10, 11B | Add edge injection after evidence merge, micro-validator with lazy gating |
| `src/reasoning/feedback.rs` | 5, 10, 11E, 11K | Wire consistency corrections, implicit feedback, Tier3OveruseTracker with hysteresis, FeedbackQueue with trace-ID matching |
| `src/predictive/router.rs` | 11D, 11F, 11G, 11H, 11J | Add EdgeStatus + TTG lease lifecycle, POS-weighted blending, hub edge caps + domain tagging + secondary hub promotion, ContextBloom + migration, betweenness centrality + dynamic hub election + language bootstrap |
| `Cargo.toml` | 7, 8 | Add lru dependency, training_sweep binary |

### Execution Order (Critical Path)

```
Week 1: Phase 1 (Foundation) + Phase 6 (Dataset Generators — can parallelize)
Week 2: Phase 2 (Classification Training)
Week 3: Phase 3 (Reasoning Training)
Week 4: Phase 4 (Predictive Training)
Week 5: Phase 5 (Consistency) + Phase 7 (Efficiency — can parallelize)
Week 6: Phase 8 (Sweep Harness) + Phase 9 (Integration Tests)
Week 7: Phase 10 (On-the-fly Learning) — lightweight, can overlap with Week 6
Week 8-9: Phase 11 (Hardening + Robustness + Edge Cases) — sub-tasks can parallelize:
         11A (Semantic Probes) + 11B (Micro-Validator + Lazy Gating) + 11F (POS Blending)
         11C (Adaptive Dims + Anchor Locking) + 11G (Hub Management) + 11H (Context Bloom)
         11D (TTG Lease Edges) + 11E (Structural Feedback + Hysteresis)
         11I (Silent Training Quarantine) + 11J (Dynamic Hub Election) + 11K (Feedback Queue)
```

### Verification Commands

```bash
# Quick compile check after each task
cargo check --no-default-features

# Unit tests for specific module
cargo test --lib --no-default-features -- classification::trainer
cargo test --lib --no-default-features -- reasoning::decomposer
cargo test --lib --no-default-features -- predictive::walk_trainer
cargo test --lib --no-default-features -- reasoning::consistency
cargo test --lib --no-default-features -- classification::semantic_anchors
cargo test --lib --no-default-features -- classification::semantic_zones
cargo test --lib --no-default-features -- reasoning::merge::micro_validator
cargo test --lib --no-default-features -- predictive::router::semantic_hasher
cargo test --lib --no-default-features -- predictive::router::hub_management
cargo test --lib --no-default-features -- predictive::router::context_bloom
cargo test --lib --no-default-features -- training::pipeline::quarantine
cargo test --lib --no-default-features -- predictive::router::centrality
cargo test --lib --no-default-features -- reasoning::feedback::feedback_queue

# Integration tests (after all phases)
cargo test --test training_integration --no-default-features

# Full training sweep
cargo run --bin training_sweep --no-default-features -- --system all

# Benchmark before/after
cargo run --bin benchmark_eval --no-default-features
```
