# SPSE Engine Architecture Documentation v11.1

**Document Version:** 11.1  
**Date:** March 15, 2026  
**Status:** Finalized for MVP Implementation  
**Target Hardware:** Edge Devices (2GB RAM / Dual-Core CPU)  
**Architecture Mode:** Unified Auto+Reasoning (Mode C)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [21-Layer Architecture](#3-21-layer-architecture)
4. [Unified Auto+Reasoning Architecture (Mode C)](#4-unified-autoreasoning-architecture-mode-c)
5. [Core Data Structures](#5-core-data-structures)
6. [Configuration System](#6-configuration-system)
7. [Memory Architecture](#7-memory-architecture)
8. [API Specifications](#8-api-specifications)
9. [Telemetry & Observability](#9-telemetry--observability)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Security Considerations](#11-security-considerations)
12. [Performance Targets](#12-performance-targets)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Risk Mitigation](#14-risk-mitigation)

---

## 1. Executive Summary

The SPSE (Semantic Personal Search Engine) is a privacy-preserving, on-device knowledge engine that combines the capabilities of a personal search system with LLM-like reasoning abilities. Unlike cloud-based solutions, SPSE operates entirely locally, ensuring user data never leaves the device.

### Key Innovations in v11.1

**Unified Auto+Reasoning (Mode C):** The system defaults to a lightweight Auto-Only path (~350MB RAM, <100ms latency), automatically triggering a multi-step reasoning loop only when internal confidence metrics fall below defined thresholds. This achieves optimal resource usage on low-spec hardware while maintaining high capability for complex tasks.

**No User Controls:** All external parameters for `mode`, `temperature`, `reasoning_depth`, and `creative_level` are removed. The engine operates in Auto-Mode exclusively, dynamically inferring intent, tone, and creativity level from query semantics and conversation history.

**Controlled Creative Spark:** A fixed 15% non-greedy sampling rate is enforced globally, subject to factual anchor validation, preventing robotic output while maintaining factual accuracy.

### Target Hardware

- **Minimum:** 2GB RAM, Dual-Core CPU (1.2GHz)
- **Recommended:** 4GB RAM, Quad-Core CPU (2.0GHz)
- **Storage:** 500MB for engine + variable for user data

### Performance Guarantees

| Query Type | RAM Usage | Latency (TTFT) | Percentage |
|------------|-----------|----------------|------------|
| Simple (greeting, lookup, chat) | ~350MB | <100ms | 90% |
| Complex (coding, logic, planning) | ~550MB | 500ms-2s | 10% |

---

## 2. System Overview

### 2.1 Design Philosophy

1. **Privacy First:** All processing occurs on-device. No data leaves the device without explicit user consent.
2. **Config-Driven:** Every numeric threshold, weight, and limit is configurable via YAML.
3. **Layer Separation:** 21 distinct layers with clear responsibilities and no cross-layer logic leakage.
4. **Dynamic Resource Allocation:** Memory allocated on-demand for reasoning, released immediately after.
5. **Auto-Mode Only:** No user toggles for mode, temperature, or creativity. System infers everything.

### 2.2 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   Web UI         │  │  CLI Interface   │  │  OpenAI-Compatible API   │  │
│  │  (Next.js)       │  │  (main.rs)       │  │  (REST/SSE)             │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────────┬─────────────┘  │
└───────────┼─────────────────────┼─────────────────────────┼────────────────┘
            │                     │                         │
            └─────────────────────┼─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ENGINE ORCHESTRATION                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        21-LAYER PIPELINE                              │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │   │
│  │  │ L1-6   │→│ L7-9   │→│ L10-13 │→│ L14-16 │→│ L17-18 │→│ L19-21 │  │   │
│  │  │Input   │ │Intent  │ │Retrieve│ │Resolve │ │Output  │ │Govern  │  │   │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                  │                                           │
│                    ┌─────────────┼─────────────┐                             │
│                    ▼             ▼             ▼                             │
│            ┌───────────┐  ┌───────────┐  ┌───────────┐                       │
│            │  Memory   │  │ Telemetry │  │  Config   │                       │
│            │  Store    │  │  Worker   │  │  Manager  │                       │
│            └───────────┘  └───────────┘  └───────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  SQLite (Core)   │  │  SQLite (Telemetry)│  │  Append-Only Logs       │  │
│  │  Units, Anchors  │  │  Hot Events       │  │  Cold Logs              │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Data Flow

1. **Input Ingestion (L1):** Raw text normalized into `InputPacket`
2. **Unit Building (L2-L3):** Rolling hash discovery, hierarchy organization
3. **Memory Ingestion (L4):** Unit persistence with channel routing
4. **Semantic Routing (L5-L6):** Spatial positioning, context management
5. **Intent Detection (L7-L9):** Intent classification, entropy calculation, retrieval decision
6. **Retrieval (L10-L13):** Query building, external fetching, evidence merging
7. **Resolution (L14-L16):** Candidate scoring, fine resolution with confidence gating
8. **Output (L17-L18):** Answer finalization, feedback collection
9. **Governance (L19-L21):** Trust validation, telemetry emission, memory pruning

---

## 3. 21-Layer Architecture

### 3.1 Layer Overview

| Layer | Name | Module | Config Section | Responsibility |
|-------|------|--------|----------------|----------------|
| L1 | Input Ingestion | `layers/input.rs` | - | Normalize raw text, create `InputPacket` |
| L2 | Unit Builder | `layers/builder.rs` | `layer_2_unit_builder` | Rolling hash discovery, unit activation, scoring |
| L3 | Hierarchy Organizer | `layers/hierarchy.rs` | - | Level grouping, anchor/entity extraction |
| L4 | Memory Ingestion | `memory/store.rs` | `layer_21_memory_governance` | Unit persistence, candidate observation |
| L5 | Semantic Router | `layers/router.rs` | `layer_5_semantic_map` | Spatial routing, neighbor selection, escape |
| L6 | Context Manager | `layers/context.rs` | - | Context matrix, sequence state, task entities |
| L7 | Intent Detector | `layers/intent.rs` | `intent` | Intent classification, entropy calculation |
| L8 | Adaptive Runtime | `config/mod.rs` | `adaptive_behavior` | Profile selection, weight adjustment |
| L9 | Retrieval Decision | `layers/intent.rs` | `layer_9_retrieval_gating` | Entropy/freshness/cost scoring, **Dynamic Reasoning Trigger** |
| L10 | Query Builder | `layers/query.rs` | `layer_10_query_builder` | Safe query construction, PII stripping |
| L11 | Retrieval Pipeline | `layers/retrieval.rs` | `layer_11_retrieval` | External source fetching, caching |
| L12 | Safety Validator | `layers/safety.rs` | `layer_19_trust_heuristics` | Trust assessment, document filtering |
| L13 | Evidence Merger | `layers/merge.rs` | `layer_13_evidence_merge` | Conflict detection, agreement scoring |
| L14 | Candidate Scorer | `layers/search.rs` | `layer_14_candidate_scoring` | 7-dimensional feature scoring |
| L15 | Resolver Mode Selection | `engine.rs` | `adaptive_behavior` | **Stochastic Floor Enforcement (15%)** |
| L16 | Fine Resolver | `layers/resolver.rs` | `layer_16_fine_resolver` | Top-k selection, **Confidence Gating** |
| L17 | Output Decoder | `layers/output.rs` | - | Answer finalization, **Silent Thought Emission** |
| L18 | Feedback Controller | `layers/feedback.rs` | - | Learning events, impact scoring |
| L19 | Trust/Safety | `layers/safety.rs` | `layer_19_trust_heuristics` | Source validation, allowlist management |
| L20 | Telemetry | `telemetry/` | `layer_20_telemetry` | Trace emission, **Reasoning Trace Logging** |
| L21 | Governance | `memory/store.rs` | `layer_21_memory_governance` | Pruning, promotion, maintenance |

### 3.2 Layer Details

#### L1: Input Ingestion

**Responsibility:** Normalize raw text input into structured `InputPacket`.

```rust
pub struct InputPacket {
    pub raw_text: String,
    pub normalized_text: String,
    pub timestamp: u64,
    pub source: InputSource,
    pub session_id: Uuid,
    pub trace_id: Uuid,
}

pub enum InputSource {
    UserQuery,
    SilentThought,  // Internal reasoning loop
    TrainingData,
    ExternalRetrieval,
}
```

**Processing:**
1. Trim whitespace
2. Normalize unicode
3. Detect language
4. Create `InputPacket` with session/trace IDs

---

#### L2: Unit Builder

**Responsibility:** Discover semantic units using rolling hash, activate units, score by frequency/salience.

```rust
pub struct UnitBuilder {
    config: UnitBuilderConfig,
    rolling_hash: RollingHash,
    active_units: Vec<Unit>,
}

pub struct UnitBuilderConfig {
    pub min_frequency_threshold: f32,
    pub max_unit_length: usize,
    pub salience_weight: f32,
}
```

**Processing:**
1. Apply rolling hash to discover phrase boundaries
2. Check frequency against threshold
3. Calculate salience score
4. Activate units meeting criteria

---

#### L3: Hierarchy Organizer

**Responsibility:** Group units into hierarchy levels, extract anchors and entities.

```rust
pub struct HierarchyLevel {
    pub level: usize,
    pub units: Vec<Unit>,
    pub anchors: Vec<Anchor>,
    pub entities: Vec<Entity>,
}

pub struct Anchor {
    pub unit_id: Uuid,
    pub anchor_type: AnchorType,
    pub trust_score: f32,
}

pub enum AnchorType {
    MathematicalConstant,
    UserIdentity,
    VerifiedCodeSyntax,
    FactualStatement,
}
```

---

#### L4: Memory Ingestion

**Responsibility:** Persist units to memory store with channel routing.

```rust
pub struct MemoryStore {
    core_units: HashMap<Uuid, Unit>,
    working_units: HashMap<Uuid, Unit>,
    intent_channel: HashMap<Uuid, Unit>,
    governance: GovernanceConfig,
}

pub enum MemoryChannel {
    Core,      // High-trust, persistent
    Working,   // Session-scoped
    Intent,    // Intent-specific, isolated
}

pub enum MemoryType {
    Core,
    Working,
    Ephemeral,
}
```

**Key Logic - Intent Channel Isolation:**
```rust
let is_intent_channel = memory_channels.contains(&MemoryChannel::Intent);
let promote_to_core = default_memory_type == MemoryType::Core
    && !(self.intent_channel_core_promotion_blocked && is_intent_channel);
```

---

#### L5: Semantic Router

**Responsibility:** Position units in 3D semantic space, route to neighbors, handle escape conditions.

```rust
pub struct SpatialGrid {
    cells: HashMap<GridCell, Vec<Unit>>,
    cell_size: f32,
    bounds: BoundingBox,
}

pub struct Router {
    config: SemanticMapConfig,
    grid: SpatialGrid,
}

pub enum TraversalMode {
    GreedyNearest,      // Default
    StochasticWalk,     // Random walk for brainstorm
    DriftFloor,         // 15% non-greedy sampling
}
```

---

#### L6: Context Manager

**Responsibility:** Maintain context matrix, sequence state, task entities.

```rust
pub struct ContextManager {
    context_matrix: Vec<Vec<f32>>,
    sequence_state: SequenceState,
    task_entities: Vec<Entity>,
    style_state: SessionStyleState,
}

pub struct SessionStyleState {
    pub active_style_anchor_id: Option<Uuid>,
    pub decay_rate: f32,  // 0.0 = permanent for session
}
```

---

#### L7: Intent Detector

**Responsibility:** Classify intent, calculate entropy, **infer tone**.

```rust
pub struct IntentDetector {
    config: IntentConfig,
    tone_inferrer: ToneInferrer,
}

pub enum IntentKind {
    Factual,
    Creative,
    Brainstorm,
    Plan,
    Act,
    Critique,
    ComplexLogic,
}

pub struct ToneInferrer {
    style_anchors: HashMap<ToneKind, StyleAnchor>,
    decay_rate: f32,
}

pub enum ToneKind {
    Empathetic,
    Direct,
    Exploratory,
    Technical,
    NeutralProfessional,
}
```

**Tone Inference Logic:**
```rust
pub fn infer_tone(&self, input: &str, history: &[Unit]) -> ToneKind {
    let urgency = self.detect_urgency(input);
    let sadness = self.detect_sadness(input);
    let technical = self.detect_technical_domain(input);
    
    if urgency > 0.7 { return ToneKind::Direct; }
    if sadness > 0.5 { return ToneKind::Empathetic; }
    if technical > 0.6 { return ToneKind::Technical; }
    ToneKind::NeutralProfessional
}
```

---

#### L9: Retrieval Decision (Dynamic Reasoning Trigger)

**Responsibility:** Decide if external retrieval needed, **trigger dynamic reasoning loop**.

```rust
pub struct RetrievalDecision {
    entropy_score: f32,
    freshness_score: f32,
    cost_score: f32,
    should_retrieve: bool,
}

// NEW: Dynamic Reasoning Trigger
pub fn should_trigger_reasoning(&self, confidence: f32, intent: IntentKind) -> bool {
    confidence < self.config.reasoning_trigger_floor 
        || intent == IntentKind::ComplexLogic
}
```

---

#### L14: Candidate Scorer

**Responsibility:** Score candidates using 7-dimensional features, **style resonance**.

```rust
pub struct CandidateScorer {
    weights: ScoringWeights,
    style_anchor: Option<StyleAnchor>,
}

pub struct ScoringWeights {
    pub w_spatial: f32,
    pub w_context: f32,
    pub w_sequence: f32,
    pub w_transition: f32,
    pub w_utility: f32,
    pub w_confidence: f32,
    pub w_evidence: f32,
}

// Scoring Note:
// Confidence is derived from source corroboration count and recency decay.
// Utility is derived from observed frequency and prediction success.

// NEW: Style Resonance Scoring
fn score_style_resonance(&self, candidate: &Unit, anchor: &StyleAnchor) -> f32 {
    self.semantic_similarity(candidate.embedding, anchor.embedding)
}
```

---

#### L15: Resolver Mode Selection (Stochastic Floor Enforcement)

**Responsibility:** Select resolver mode, **enforce 15% stochastic floor**.

```rust
pub enum ResolverMode {
    Deterministic,
    Balanced,
    Exploratory,
}

// NEW: Stochastic Floor Enforcement
impl Router {
    pub fn select_with_creative_floor(
        &self,
        candidates: &[Candidate],
        config: &CreativeSparkConfig,
    ) -> Candidate {
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

---

#### L16: Fine Resolver (Confidence Gating)

**Responsibility:** Select top-k candidates, **output confidence score for reasoning trigger**.

```rust
pub struct FineResolver {
    config: FineResolverConfig,
}

pub struct ResolutionResult {
    pub candidates: Vec<Candidate>,
    pub confidence_score: f32,
    pub needs_reasoning: bool,  // NEW: Signal to orchestrator
}

// NEW: Anchor Validation Gate
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

---

#### L17: Output Decoder (Silent Thought Emission)

**Responsibility:** Finalize answer, **emit SilentThought for reasoning loop**.

```rust
pub enum OutputType {
    FinalAnswer(String),
    SilentThought(String),  // Internal reasoning step
}

pub struct ThoughtUnit {
    pub content: String,
    pub step: usize,
    pub internal_only: bool,  // true = hidden from user
}
```

---

#### L20: Telemetry (Reasoning Trace Logging)

**Responsibility:** Emit structured events, **log reasoning steps**.

```rust
pub enum TelemetryEvent {
    Calculation { layer: u8, operation: String, duration_ms: u64 },
    DbPush { unit_id: Uuid, memory_type: MemoryType },
    Retrieval { source: String, results: usize },
    ReasoningStep { step: usize, thought: String, confidence: f32 },  // NEW
    ConfidenceTrajectory { steps: Vec<f32> },  // NEW
}
```

---

## 4. Unified Auto+Reasoning Architecture (Mode C)

### 4.1 Mode Comparison

| Feature | **Mode A: Auto-Only** | **Mode B: Reasoning-Only** | **Mode C: Unified Auto+Reasoning** |
| :--- | :--- | :--- | :--- |
| **Logic** | Direct retrieval → Scoring → Output | Detects need → Loops internally → Output | Default is Mode A. If confidence < threshold, switches to Mode B internally. |
| **RAM Usage** | ~350 MB | ~550 MB | ~350 MB (Idle), ~550 MB (Active) |
| **CPU Load** | Low | High | Variable (Low by default; spikes for complex queries) |
| **Latency** | < 100ms (TTFT) | 500ms - 2s | < 100ms (Simple), 500ms+ (Complex) |
| **Code Complexity** | Low | Medium | Medium-High |
| **User Experience** | Fast, but fails on complex logic | User must know when to trigger | Seamless. System "thinks" only when necessary. |
| **Optimality for Low Spec** | ✅ Best Efficiency, ❌ Low Capability | ❌ High Overhead, ✅ High Capability | ✅ **Best Balance** |

### 4.2 Dynamic Reasoning Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DYNAMIC REASONING FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ User Query   │
    └──────┬───────┘
           ▼
    ┌──────────────┐     ┌──────────────────────────────────────┐
    │ Initial Pass │────▶│ Confidence Check (L16)               │
    │ (L14-L17)    │     │ confidence_score < 0.40?             │
    └──────────────┘     │ OR intent == ComplexLogic?           │
                         └──────────────┬───────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │ YES               │                   │ NO
                    ▼                   │                   ▼
    ┌────────────────────────┐         │         ┌──────────────────┐
    │ Enter Reasoning State  │         │         │ Direct Output    │
    │ (Allocate ~550MB)       │         │         │ (~350MB, <100ms)│
    └────────────┬───────────┘         │         └──────────────────┘
                 ▼                     │
    ┌────────────────────────┐         │
    │ Generate SilentThought │         │
    │ (L17)                  │         │
    └────────────┬───────────┘         │
                 ▼                     │
    ┌────────────────────────┐         │
    │ Ingest as Internal Input│        │
    │ (L1, internal_only=true)│        │
    └────────────┬───────────┘         │
                 ▼                     │
    ┌────────────────────────┐         │
    │ Re-assess Confidence   │         │
    │ confidence > 0.60?     │         │
    └────────────┬───────────┘         │
                 │                     │
    ┌────────────┼────────────┐        │
    │ YES        │ NO         │        │
    ▼            ▼            │        │
┌────────┐ ┌─────────────┐    │        │
│Output  │ │ step < max? │    │        │
│        │ │ (max=3)     │    │        │
└────────┘ └──────┬──────┘    │        │
                  │           │        │
         ┌────────┼────────┐  │        │
         │ YES    │ NO     │  │        │
         ▼        ▼        │  │        │
    ┌────────┐ ┌────────┐ │  │        │
    │Loop    │ │Force   │ │  │        │
    │Again   │ │Output  │ │  │        │
    └────────┘ └────────┘ │  │        │
                         │  │        │
                         └──┼────────┘
                            │
                            ▼
                   ┌──────────────────┐
                   │ Release Memory   │
                   │ (Back to ~350MB) │
                   └──────────────────┘
```

### 4.3 Implementation

```rust
// src/engine.rs
impl Engine {
    pub fn resolve_with_dynamic_reasoning(&mut self, query: &str) -> OutputType {
        let initial_confidence = self.assess_confidence(query);
        
        if !self.should_trigger_reasoning(initial_confidence, self.current_intent) {
            return self.resolve_direct(query);  // Fast path: ~350MB, <100ms
        }
        
        // Reasoning path: temporarily allocate ~550MB
        self.memory.allocate_thought_buffer();
        let result = self.execute_reasoning_loop(query, 0);
        self.memory.deallocate_thought_buffer();
        result
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
    
    fn should_trigger_reasoning(&self, confidence: f32, intent: IntentKind) -> bool {
        confidence < self.config.reasoning_trigger_floor 
            || intent == IntentKind::ComplexLogic
    }
}
```

---

## 5. Core Data Structures

### 5.1 Unit

```rust
pub struct Unit {
    pub id: Uuid,
    pub content: String,
    pub embedding: Vec<f32>,
    pub position: Vec3,
    pub memory_type: MemoryType,
    pub memory_channels: Vec<MemoryChannel>,
    pub frequency: f32,
    pub salience: f32,
    pub trust_score: f32,
    pub created_at: u64,
    pub last_accessed: u64,
    pub metadata: HashMap<String, String>,
}

pub enum MemoryType {
    Core,       // High-trust, persistent
    Working,    // Session-scoped
    Ephemeral,  // Temporary
}

pub enum MemoryChannel {
    Core,
    Working,
    Intent,     // Isolated from Core promotion
}
```

### 5.2 InputPacket

```rust
pub struct InputPacket {
    pub raw_text: String,
    pub normalized_text: String,
    pub timestamp: u64,
    pub source: InputSource,
    pub session_id: Uuid,
    pub trace_id: Uuid,
}

pub enum InputSource {
    UserQuery,
    SilentThought,
    TrainingData,
    ExternalRetrieval,
}
```

### 5.3 ThoughtUnit

```rust
pub struct ThoughtUnit {
    pub content: String,
    pub step: usize,
    pub internal_only: bool,  // true = hidden from user
    pub confidence_before: f32,
    pub confidence_after: f32,
}
```

### 5.4 StyleAnchor

```rust
pub struct StyleAnchor {
    pub id: Uuid,
    pub name: String,           // "Empathetic", "Direct", "Technical"
    pub tone_kind: ToneKind,
    pub exemplar_text: String,
    pub style_vector: Vec<f32>,  // Semantic embedding of style
}
```

### 5.5 Anchor

```rust
pub struct Anchor {
    pub unit_id: Uuid,
    pub anchor_type: AnchorType,
    pub trust_score: f32,
    pub content: String,
}

pub enum AnchorType {
    MathematicalConstant,   // Never drift (e.g., "2+2=4")
    UserIdentity,           // Never drift (e.g., user's name)
    VerifiedCodeSyntax,     // Never drift
    FactualStatement,       // High trust, limited drift
}
```

---

## 6. Configuration System

### 6.1 Configuration Schema

```yaml
# config/config.yaml

engine:
  mode: "auto_unified"  # LOCKED. No user override.

# Dynamic Reasoning Configuration
auto_inference:
  reasoning_loop:
    enabled: true
    trigger_confidence_floor: 0.40  # Below this, system starts "thinking"
    max_internal_steps: 3           # Cap loops to save CPU on low-spec
    exit_confidence_threshold: 0.60
    
  tone_inference:
    enabled: true
    style_anchor_decay: 0.0         # Persist tone for session
    
  creative_spark:
    global_stochastic_floor: 0.15   # Enforce 15% drift
    anchor_protection_strictness: 0.95  # Never drift on high-trust anchors

# Memory Configuration
memory:
  dynamic_allocation:
    enable_thought_buffer_on_demand: true
    base_memory_limit_mb: 350
    max_memory_limit_mb: 550

# Layer-Specific Configurations
layer_2_unit_builder:
  min_frequency_threshold: 0.01
  max_unit_length: 100
  salience_weight: 0.5

layer_5_semantic_map:
  cell_size: 4.0
  neighbor_limit: 50
  escape_threshold: 0.3

layer_9_retrieval_gating:
  entropy_threshold: 0.72
  freshness_threshold: 0.65
  cost_threshold: 0.5

layer_14_candidate_scoring:
  w_spatial: 0.15
  w_context: 0.25
  w_sequence: 0.15
  w_transition: 0.10
  w_utility: 0.15
  w_confidence: 0.10
  w_evidence: 0.10

layer_16_fine_resolver:
  evidence_answer_confidence_threshold: 0.22
  min_confidence_floor: 0.22
  top_k: 10

layer_20_telemetry:
  enabled: true
  sample_rate: 1.0
  hot_store_limit: 10000

layer_21_memory_governance:
  prune_utility_threshold: 0.1
  max_candidate_pool: 1000
  intent_channel_core_promotion_blocked: true

# Intent Profiles
adaptive_behavior:
  intent_profiles:
    creative:
      scoring:
        w_spatial: 0.10
        w_context: 0.20
        w_utility: 0.15
      resolver:
        selection_temperature: 0.80
        min_confidence_floor: 0.15
        mode: "exploratory"
    factual:
      scoring:
        w_spatial: 0.20
        w_context: 0.30
        w_utility: 0.10
      resolver:
        selection_temperature: 0.30
        min_confidence_floor: 0.25
        mode: "deterministic"
```

### 6.2 Configuration Principles

1. **All numeric values in config:** No hardcoded thresholds in code
2. **Named config fields:** Every field has descriptive name
3. **Layer-prefixed sections:** `layer_N_name` for layer-specific settings
4. **Documented fields:** Every field has comment explaining purpose

---

## 7. Memory Architecture

### 7.1 Memory Tiers

| Tier | Type | Persistence | Trust Score | Use Case |
|------|------|-------------|-------------|----------|
| Core | Persistent | SQLite | > 0.8 | High-trust facts, anchors |
| Working | Session | In-memory | 0.3-0.8 | Context, recent queries |
| Ephemeral | Temporary | In-memory | < 0.3 | Temporary processing |
| Intent | Isolated | In-memory | Variable | Intent-specific data |

### 7.2 Memory Channels

```rust
pub enum MemoryChannel {
    Core,      // Standard Core memory path
    Working,   // Session-scoped working memory
    Intent,    // Intent-specific, isolated from Core promotion
}
```

**Intent Channel Isolation:** Units in the Intent channel are blocked from promotion to Core memory to prevent pollution.

### 7.3 Dynamic Memory Allocation

```rust
pub struct DynamicMemoryAllocator {
    base_limit_mb: usize,    // 350MB idle
    max_limit_mb: usize,     // 550MB during reasoning
    current_usage: AtomicUsize,
    thought_buffer: Option<ThoughtBuffer>,
}

impl DynamicMemoryAllocator {
    pub fn allocate_thought_buffer(&mut self) -> Option<ThoughtBuffer> {
        if self.current_usage.load() + THOUGHT_BUFFER_SIZE <= self.max_limit_mb {
            self.current_usage.fetch_add(THOUGHT_BUFFER_SIZE);
            Some(ThoughtBuffer::new())
        } else {
            None
        }
    }
    
    pub fn deallocate_thought_buffer(&mut self) {
        self.current_usage.fetch_sub(THOUGHT_BUFFER_SIZE);
        self.thought_buffer = None;
    }
}
```

---

## 8. API Specifications

### 8.1 REST API

**Base URL:** `http://localhost:8080/api/v1`

#### POST /query

Submit a query to the engine.

**Request:**
```json
{
    "query": "What is the capital of France?",
    "context": ["Previous conversation context..."]
}
```

**Response:**
```json
{
    "answer": "The capital of France is Paris.",
    "confidence": 0.85,
    "intent": "Factual",
    "tone": "NeutralProfessional",
    "trace_id": "uuid-here"
}
```

### 8.2 OpenAI-Compatible API

**Base URL:** `http://localhost:8080/v1`

#### POST /chat/completions

OpenAI Chat Completions API compatibility.

**Request:**
```json
{
    "model": "spse-default",
    "messages": [
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false
}
```

**Note:** `temperature`, `top_p`, `max_tokens` are **ignored**. Auto-Mode only.

**Response:**
```json
{
    "id": "chatcmpl-uuid",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "spse-default",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18
    }
}
```

### 8.3 Streaming SSE

**Request:**
```json
{
    "model": "spse-default",
    "messages": [...],
    "stream": true
}
```

**Response (SSE):**
```
data: {"id":"chatcmpl-uuid","choices":[{"delta":{"content":"The"},"index":0}]}

data: {"id":"chatcmpl-uuid","choices":[{"delta":{"content":" capital"},"index":0}]}

data: {"id":"chatcmpl-uuid","choices":[{"delta":{"content":" of"},"index":0}]}

data: [DONE]
```

---

## 9. Telemetry & Observability

### 9.1 Telemetry Events

```rust
pub enum TelemetryEvent {
    // Layer events
    Calculation { layer: u8, operation: String, duration_ms: u64 },
    DbPush { unit_id: Uuid, memory_type: MemoryType },
    Retrieval { source: String, results: usize },
    
    // NEW: Reasoning events
    ReasoningStep { step: usize, thought: String, confidence: f32 },
    ConfidenceTrajectory { steps: Vec<f32> },
    ReasoningTriggered { trigger_reason: String },
    
    // Intent events
    IntentLabel { label: IntentKind, channel: MemoryChannel, score: f32 },
    ToneInferred { tone: ToneKind, signals: HashMap<String, f32> },
}
```

### 9.2 Hot/Cold Storage

| Storage | Type | Retention | Use Case |
|---------|------|-----------|----------|
| Hot | SQLite | Last 10,000 events | Real-time UI, debugging |
| Cold | Append-only file | Indefinite | Long-term analysis |

---

## 10. Deployment Architecture

### 10.1 Single-Process Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                      SINGLE PROCESS                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ API Server  │  │   Engine    │  │  Background Tasks   │  │
│  │ (Axum)      │  │  (Core)     │  │  (Training, Prune)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                          │                                   │
│              ┌───────────┼───────────┐                       │
│              ▼           ▼           ▼                       │
│        ┌─────────┐ ┌─────────┐ ┌─────────┐                   │
│        │ SQLite  │ │ Memory  │ │ Config  │                   │
│        │ (Disk)  │ │ (RAM)   │ │ (YAML)  │                   │
│        └─────────┘ └─────────┘ └─────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

**MVP Communication Note:** The API server, engine core, and background tasks communicate through a shared memory directory and append-only event log. No direct RPC is required for MVP deployment, which keeps coordination simple in the single-process design.

### 10.2 Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 2GB | 4GB |
| CPU | Dual-Core 1.2GHz | Quad-Core 2.0GHz |
| Storage | 500MB + data | 1GB + data |
| OS | Linux, macOS, Windows | Linux, macOS |

---

## 11. Security Considerations

### 11.1 Data Privacy

- **On-device processing:** All data stays on device
- **No cloud dependencies:** No external API calls without user consent
- **Encrypted storage:** SQLite database encrypted at rest (optional)

### 11.2 Input Validation

- **PII stripping:** Layer 10 strips PII from external queries
- **Injection prevention:** Safe query construction prevents SQL/code injection
- **Trust validation:** Layer 12 validates all external sources

### 11.3 Trust Heuristics

```rust
pub struct TrustHeuristics {
    min_source_trust: f32,           // 0.35 minimum
    allowlist: Vec<String>,          // Trusted domains
    blocklist: Vec<String>,          // Blocked domains
    require_https: bool,             // true
}
```

---

## 12. Performance Targets

### 12.1 Latency Targets

| Query Type | TTFT | Total Latency | RAM |
|------------|------|---------------|-----|
| Simple | < 100ms | < 200ms | ~350MB |
| Complex | 500ms-2s | 1-5s | ~550MB |

### 12.2 Throughput Targets

| Metric | Target |
|--------|--------|
| Queries/second (simple) | 10+ |
| Queries/second (complex) | 1-2 |
| Concurrent sessions | 5+ |

### 12.3 Memory Targets

| State | Target |
|-------|--------|
| Idle | ~350MB |
| Reasoning | ~550MB |
| Peak | < 600MB |

### 12.4 Evaluation Assumptions

These targets assume an average workload of roughly 200 queries/day, average query length of 50-100 tokens, and an external retrieval rate of approximately 20%. Memory growth, telemetry volume, and cache behavior should be re-baselined if real-world usage materially exceeds those assumptions.

---

## 13. Implementation Roadmap

### Phase 1: Creative Mode & Hybrid Intent Profiles ✅ COMPLETE

- Intent-specific config profiles
- Hybrid intent score validation
- MemoryChannel::Intent gate validation

### Phase 2: Drill Suite & Pollution Integration ✅ COMPLETE

- Layer-specific drill suite
- Unified stress drill
- Snapshot atomicity & recovery

### Phase 3: LLM-Like Core (HIGHEST PRIORITY) - 4-5 weeks

1. **Dynamic Reasoning Type** - Confidence-gated reasoning loop
2. **Controlled Creative Spark** - 15% stochastic floor with anchor validation
3. **Internal Tone Inference** - Multi-signal tone detection
4. **Auto-Mode Enforcement** - Remove all user toggles
5. **LLM-Like Core Drills** - Comprehensive drill coverage

### Phase 4: Core Infrastructure - 2-3 weeks

1. **Global Logging Engine** - Telemetry with reasoning trace
2. **Latency Monitoring** - p50/p95/p99 tracking
3. **Dynamic Memory Allocation** - On-demand thought buffer

### Phase 5: Retrieval & Optimization - 2-3 weeks

1. **Multi-Engine Consensus** - Aggregate multiple sources
2. **Config Sweeping** - Optimize for low-spec hardware

### Phase 6: User Interface - 2-3 weeks

1. **Web UI** - Auto-Mode interface (no toggles)
2. **OpenAI-Compatible API** - LLM replacement capability

**Total Duration:** 19-24 weeks

---

## 14. Risk Mitigation

| Risk | Mitigation | Fallback |
| :--- | :--- | :--- |
| **Dynamic Reasoning Infinite Loop** | Max 3 steps enforced, confidence threshold check | Force final answer with partial reasoning |
| **Creative Drift Factual Corruption** | Anchor validation gate re-samples greedily on contradiction | Disable drift for mathematical/identity queries |
| **Auto-Mode Tone Misclassification** | Multi-signal tone detection (keywords + intent + history) | Default to NeutralProfessional on ambiguous signals |
| **Memory Exhaustion on Low-Spec** | Dynamic allocation with hard limits (350MB base, 550MB max) | Skip reasoning, return direct answer with lower confidence |
| **Telemetry Performance Overhead** | Async worker with batching, configurable sample rate | Reduce sample rate to 0.1 if needed |
| **OpenAI API Incompatibility** | Comprehensive API compatibility test suite | Compatibility shims for missing behaviors |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Unit** | A semantic unit of text with embedding, position, and metadata |
| **Anchor** | A high-trust unit that should not be modified by creative drift |
| **SilentThought** | An internal reasoning step not shown to the user |
| **MemoryChannel** | Routing path for units (Core, Working, Intent) |
| **StyleAnchor** | An exemplar text used to bias vocabulary selection |
| **TTFT** | Time To First Token |
| **Mode C** | Unified Auto+Reasoning architecture with dynamic switching |

---

## Appendix B: References

- `AGENTS.md` - Architecture compliance guide
- `config/config.yaml` - Main configuration file
- `README.md` - Project overview
- `docs/PRE_PRODUCTION_EXECUTION_PLAN.md` - Implementation roadmap
