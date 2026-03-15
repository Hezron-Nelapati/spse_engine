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
8. [Classification System](#8-classification-system)
9. [GPU Acceleration](#9-gpu-acceleration)
10. [Unified Training Pipeline](#10-unified-training-pipeline)
11. [API Specifications](#11-api-specifications)
12. [Telemetry & Observability](#12-telemetry--observability)
13. [Deployment Architecture](#13-deployment-architecture)
14. [Security Considerations](#14-security-considerations)
15. [Performance Targets](#15-performance-targets)
16. [Implementation Roadmap](#16-implementation-roadmap)
17. [Risk Mitigation](#17-risk-mitigation)

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
pub struct UnitHierarchy {
    pub levels: BTreeMap<String, Vec<ActivatedUnit>>,
    pub anchors: Vec<String>,
    pub entities: Vec<String>,
}

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

pub enum UnitLevel {
    Word,
    Phrase,
    Sentence,
    Paragraph,
}
```

Note: Anchors are tracked as strings (content) in the hierarchy, not as separate structs. Units with `anchor_status: true` are protected from creative drift.

---

#### L4: Memory Ingestion

**Responsibility:** Persist units to memory store with channel routing.

```rust
pub struct MemoryStore {
    cache: HashMap<Uuid, Unit>,
    content_index: HashMap<String, Uuid>,
    channel_index: HashMap<MemoryChannel, HashSet<Uuid>>,
    candidate_cache: HashMap<String, UnitCandidate>,
    sequence_state: SequenceState,
    anchor_reuse_threshold: u64,
    core_promotion_threshold: u64,
    promotion_min_corroborations: u32,
    prune_utility_threshold: f32,
    episodic_decay_days: u64,
    // ... additional config fields
}

pub enum MemoryChannel {
    #[default]
    Main,      // Standard memory path
    Intent,    // Intent-specific, isolated from Core promotion
    Reasoning, // Internal reasoning loop, isolated
}

pub enum MemoryType {
    Episodic,  // Session-scoped, decayed over time
    Core,      // High-trust, persistent
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
pub struct ContextMatrix {
    pub cells: Vec<ContextCell>,
    pub summary: String,
}

pub struct SequenceState {
    pub recent_unit_ids: Vec<Uuid>,
    pub anchor_ids: Vec<Uuid>,
    pub task_entities: Vec<String>,
    pub turn_index: u64,
}

pub struct SessionStyleState {
    pub active_style_anchor: Option<StyleAnchor>,
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

pub struct ToneInferrer {
    style_anchors: HashMap<ToneKind, StyleAnchor>,
    decay_rate: f32,
}

pub enum ToneKind {
    #[default]
    NeutralProfessional,
    Empathetic,
    Direct,
    Technical,
    Casual,
    Formal,
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
    pub spatial: f32,
    pub context: f32,
    pub sequence: f32,
    pub transition: f32,
    pub utility: f32,
    pub confidence: f32,
    pub evidence: f32,
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

pub struct ScoredCandidate {
    pub unit_id: Uuid,
    pub content: String,
    pub score: f32,
    pub breakdown: ScoreBreakdown,
    pub memory_type: MemoryType,
}

pub struct ResolvedCandidate {
    pub unit_id: Uuid,
    pub content: String,
    pub score: f32,
    pub mode: ResolverMode,
    pub used_escape: bool,
}

pub enum ResolverMode {
    Deterministic,
    Balanced,
    Exploratory,
}
```

**Anchor Validation:** Units with `anchor_status: true` are protected from creative drift. The resolver checks for contradictions against high-trust anchors before accepting creative candidates.

---

#### L17: Output Decoder (Silent Thought Emission)

**Responsibility:** Finalize answer, **emit SilentThought for reasoning loop**.

```rust
pub struct DecodedOutput {
    pub text: String,
    pub grounded: bool,
}

pub struct ThoughtUnit {
    pub content: String,
    pub step: usize,
    pub internal_only: bool,  // true = hidden from user
    pub confidence: f32,
    pub created_at: DateTime<Utc>,
}
```

---

#### L20: Telemetry (Reasoning Trace Logging)

**Responsibility:** Emit structured events, **log reasoning steps**.

```rust
pub enum TelemetryEvent {
    /// Calculation event at layer boundary
    Calculation {
        layer: u8,
        operation: String,
        duration_ms: u64,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Database push event
    DbPush {
        unit_id: Uuid,
        memory_type: MemoryType,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Retrieval event
    Retrieval {
        source: String,
        results: usize,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Intent label event with channel tracking
    IntentLabel {
        label: IntentKind,
        channel: MemoryChannel,
        score: f32,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Reasoning step event
    ReasoningStep {
        step: usize,
        thought: String,
        confidence: f32,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Latency spike event
    LatencySpike {
        layer: u8,
        latency_ms: u64,
        threshold_ms: u64,
        session_id: Uuid,
        trace_id: Uuid,
    },
    /// Memory allocation event
    MemoryAllocation {
        allocated_kb: usize,
        total_kb: usize,
        limit_kb: usize,
        session_id: Uuid,
        trace_id: Uuid,
    },
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
    pub normalized: String,
    pub level: UnitLevel,
    pub frequency: u64,
    pub utility_score: f32,
    pub semantic_position: [f32; 3],
    pub anchor_status: bool,
    pub memory_type: MemoryType,
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
    /// GPU shaders confine these to Z = -1.0 subspace with 3x repulsion from content units
    pub is_process_unit: bool,
}

pub enum UnitLevel {
    Word,
    Phrase,
    Sentence,
    Paragraph,
}

pub enum MemoryType {
    Episodic,  // Session-scoped, decayed over time
    Core,      // High-trust, persistent
}

pub enum MemoryChannel {
    #[default]
    Main,      // Standard memory path
    Intent,    // Intent-specific, isolated from Core promotion
    Reasoning, // Internal reasoning loop, isolated
}

/// Memory target for training/ingestion (separate from MemoryType)
pub enum MemoryTarget {
    #[default]
    StagingEpisodic,  // High utility, requires corroboration for Core promotion
    Core,             // Direct to Core (only via consolidate_immediately)
    Episodic,         // Standard episodic memory
}
```

### 5.2 InputPacket

```rust
pub struct InputPacket {
    pub original_text: String,
    pub normalized_text: String,
    pub bytes: Vec<u8>,
    pub training_mode: bool,
    pub timestamp: DateTime<Utc>,
}
```

### 5.3 ThoughtUnit

```rust
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
```

### 5.4 StyleAnchor

```rust
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
```

### 5.5 Link

```rust
pub struct Link {
    pub target_id: Uuid,
    pub edge_type: EdgeType,
    pub weight: f32,
}

pub enum EdgeType {
    Semantic,
    Parent,
    Child,
    Sequence,
    SourceEvidence,
}
```

Note: Units use `anchor_status: bool` flag rather than a separate Anchor struct. High-trust units with `anchor_status: true` are protected from creative drift.

### 5.6 Reasoning Pattern System

The engine implements a reasoning pattern system that isolates process units from core semantic space, enabling iterative reasoning without polluting the knowledge base.

#### Process Units

Process units are reasoning steps stored in the `Reasoning` memory channel, isolated from `Core`:

- **Flag**: `is_process_unit: bool` on `Unit` struct
- **Spatial Isolation**: GPU shaders confine process units to Z = -1.0 subspace
- **Repulsion**: 3x repulsion multiplier between process and content units prevents drift
- **Pruning Protection**: Process anchors registered via `register_process_anchor()` are never pruned

#### Reasoning Types

```rust
pub enum ReasoningType {
    #[default]
    General,        // Step-by-step reasoning
    Mathematical,   // Calculation (GSM8K style)
    Logical,        // Deductive inference (ProofWriter style)
    Explanatory,    // Why/how causal chains
    Planning,       // Multi-step action sequences
    Verification,   // Fact-checking with source tracing
    Debugging,      // Error isolation and hypothesis testing
    MultiHop,       // Multi-hop deduction across entities
}
```

#### Reasoning Trace

```rust
pub struct ReasoningTrace {
    pub steps: Vec<ReasoningStep>,
    pub reasoning_type: ReasoningType,
    pub confidence_trajectory: Vec<f32>,
    pub entities: Vec<String>,
    pub structure_hash: Option<u64>,
}

pub struct ReasoningStep {
    pub content: String,
    pub step_type: ReasoningStepType,
    pub anchor_step: bool,
    pub dependencies: Vec<usize>,
    pub structure_hash: Option<u64>,
}
```

#### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `reasoning_patterns_for_query()` | `memory/store.rs` | Retrieve matching patterns from memory |
| `register_reasoning_pattern()` | `memory/store.rs` | Index a unit as a reasoning pattern |
| `should_prune_unit()` | `memory/store.rs` | Pruning decision with anchor protection |
| `adapt_reasoning_pattern()` | `engine.rs` | Adapt retrieved pattern to current context |
| `ingest_reasoning_trace()` | `training.rs` | Convert trace to process units in memory |

#### GPU Force Layout Integration

The GPU-accelerated force layout handles process units:

1. `GpuPosition` struct includes `is_process_unit: u32` field
2. Shader applies 3x repulsion between process/content unit pairs
3. Process units are constrained to Z = -1.0 subspace after each iteration
4. Anchors (`is_anchor: u32`) remain fixed regardless of type

#### Memory Channel Isolation

Process units are ingested via `MemoryChannel::Reasoning`, not `MemoryChannel::Core`:

```rust
memory.ingest_hierarchy_with_channels(
    &hierarchy,
    SourceKind::UserInput,
    &context,
    MemoryType::Episodic,
    &[MemoryChannel::Reasoning],  // Reasoning channel only
);
```

This prevents reasoning artifacts from polluting the core knowledge base.

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
    enable_thought_buffer_on_demand: true
    
  tone_inference:
    enabled: true
    style_anchor_decay: 0.0         # Persist tone for session
    
  creative_spark:
    global_stochastic_floor: 0.15   # Enforce 15% drift
    anchor_protection_strictness: 0.95  # Never drift on high-trust anchors

  dynamic_memory:
    enabled: true
    base_memory_limit_mb: 350
    max_memory_limit_mb: 550
    thought_buffer_size_kb: 64

# Layer-Specific Configurations
layer_2_unit_builder:
  min_frequency_threshold: 2  # u64 - minimum frequency for unit activation
  initial_utility_score: 0.50
  edit_distance_cluster_threshold: 1
  rolling_hash_window_sizes: [2, 3, 4, 5, 6, 7, 8]
  max_activated_units: 96
  punctuation_ratio_limit: 0.55
  hash_base: 257
  min_fragment_length: 4
  utility_compression_gain_cap: 1.2
  utility_span_weight: 0.45
  utility_frequency_weight: 0.30
  salience_base: 0.18
  confidence_base: 0.22

layer_5_semantic_map:
  preferred_spacing_k: 1.0
  max_displacement_per_iteration: 1.75
  convergence_tolerance: 0.001
  max_layout_iterations: 24
  energy_rollback_threshold: 3
  attractive_force_coefficient: 1.0
  repulsive_force_coefficient: 1.0
  layout_boundary: 128.0
  max_layout_units: 256
  spatial_cell_size: 4.0
  neighbor_radius: 6.0

layer_9_retrieval_gating:
  w_entropy: 0.35
  w_recency: 0.25
  w_disagreement: 0.20
  w_cost: 0.20
  entropy_threshold: 0.72
  freshness_threshold: 0.65
  disagreement_threshold: 0.30
  decision_threshold: 0.50
  cost_penalty: 0.10

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
  selection_temperature: 0.7
  creative_drift_tolerance: 0.25
  factual_corruption_threshold: 0.15
  factual_intent_retrieval_threshold: 0.65

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
| Episodic | Session-scoped | SQLite + decay | Variable | Context, recent queries |

### 7.2 Memory Channels

```rust
pub enum MemoryChannel {
    #[default]
    Main,      // Standard memory path
    Intent,    // Intent-specific, isolated from Core promotion
    Reasoning, // Internal reasoning loop, isolated from Core
}
```

### 7.3 Memory Targets (Training/Ingestion)

```rust
pub enum MemoryTarget {
    #[default]
    StagingEpisodic,  // High utility, requires corroboration for Core promotion
    Core,             // Direct to Core (only via consolidate_immediately)
    Episodic,         // Standard episodic memory
}
```

**Intent Channel Isolation:** Units in the Intent channel are blocked from promotion to Core memory to prevent pollution.

### 7.4 Dynamic Memory Allocation

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

## 8. Classification System

### 8.1 Overview

The classification system provides hybrid retrieval-augmented classification for intent, tone, and resolver mode inference. It combines lightweight feature signatures with spatial pattern retrieval.

### 8.2 Architecture Alignment

| Layer | Classification Component |
|-------|-------------------------|
| L2 | ClassificationSignature as specialized Unit feature |
| L4 | ClassificationPattern storage in Intent memory channel |
| L6 | Spatial query for O(log N) pattern retrieval |
| L14 | Similarity scoring and candidate aggregation |
| L18 | Feedback-driven learning and spatial adjustment |

### 8.3 Core Components

```rust
// src/classification/mod.rs

/// Semantic hash generator for classification signatures
pub struct SemanticHasher;

/// Feature signature for classification queries
pub struct ClassificationSignature {
    pub semantic_hash: u64,
    pub feature_vector: Vec<f32>,
    pub token_count: usize,
}

/// Stored pattern with label for retrieval
pub struct ClassificationPattern {
    pub id: Uuid,
    pub signature: ClassificationSignature,
    pub label: ClassificationLabel,
    pub position: Vec3,
    pub vote_count: u32,
    pub confidence: f32,
}

/// Main classifier using spatial retrieval
pub struct ClassificationCalculator {
    config: ClassificationConfig,
    spatial_index: SpatialIndex,
}

pub enum ClassificationLabel {
    Intent(IntentKind),
    Tone(ToneKind),
    ResolverMode(ResolverMode),
}

pub struct ClassificationResult {
    pub label: ClassificationLabel,
    pub confidence: f32,
    pub method: CalculationMethod,
    pub nearby_patterns: usize,
}
```

### 8.4 Classification Flow

```
Input Text
    │
    ▼
┌─────────────────────────┐
│ SemanticHasher          │
│ - Token normalization   │
│ - Feature extraction    │
│ - Hash generation       │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Spatial Query (L6)      │
│ - k-NN lookup           │
│ - Radius search         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Vote Aggregation        │
│ - Weighted voting       │
│ - Confidence scaling    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ ClassificationResult    │
│ - Intent/Tone/Mode      │
│ - Confidence score      │
└─────────────────────────┘
```

### 8.5 Training Integration

```rust
pub struct ClassificationTrainer {
    config: TrainerConfig,
    patterns: Vec<ClassificationPattern>,
}

pub struct LabeledDialogue {
    pub turns: Vec<LabeledTurn>,
    pub metadata: DialogueMetadata,
}

pub struct TrainingOutcome {
    pub patterns_added: usize,
    pub patterns_updated: usize,
    pub spatial_adjustments: usize,
}
```

---

## 9. GPU Acceleration

### 9.1 Overview

The engine supports optional GPU acceleration via `wgpu` for compute-intensive operations. GPU acceleration is automatically used when available and falls back to CPU when disabled or unavailable.

### 9.2 Feature Flag

```toml
# Cargo.toml
[features]
default = []
gpu = ["wgpu"]
```

### 9.3 GPU-Accelerated Operations

| Operation | Module | Speedup |
|-----------|--------|---------|
| Candidate Scoring | `gpu/compute/candidate_scorer.rs` | 10-50x |
| Intent Scoring | `gpu/compute/intent_scorer.rs` | 5-20x |
| Distance Calculation | `gpu/compute/distance.rs` | 20-100x |
| Force-Directed Layout | `gpu/compute/force_layout.rs` | 10-30x |
| Tone Detection | `gpu/compute/tone_detector.rs` | 5-15x |
| Evidence Merging | `gpu/compute/evidence_merger.rs` | 3-10x |

### 9.4 GPU Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GPU COMPUTE MODULE                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Device Manager                        │ │
│  │  - Device selection (discrete/integrated)               │ │
│  │  - Memory allocation                                     │ │
│  │  - Shader compilation                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│                              │                               │
│              ┌───────────────┼───────────────┐              │
│              ▼               ▼               ▼              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐      │
│  │ Candidate     │ │ Intent        │ │ Distance      │      │
│  │ Scorer        │ │ Scorer        │ │ Calculator    │      │
│  │               │ │               │ │               │      │
│  │ - 7-dim score │ │ - Multi-class │ │ - Euclidean   │      │
│  │ - Batch proc  │ │ - Batch proc  │ │ - Batch proc  │      │
│  └───────────────┘ └───────────────┘ └───────────────┘      │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Shader Pipeline                       │ │
│  │  - WGSL compute shaders                                  │ │
│  │  - Bind groups for input/output                          │ │
│  │  - Dispatch orchestration                                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 9.5 CPU Fallback Pattern

```rust
pub fn score_candidates_gpu_accelerated(
    candidates: &[Unit],
    context: &ContextMatrix,
    // ...
) -> Vec<ScoredCandidate> {
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::compute::candidate_scorer::score_candidates;
        let mut scored = score_candidates(candidates, context, /* ... */);
        
        // Post-processing (GPU doesn't handle this)
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        return scored;
    }
    
    // CPU fallback when GPU feature is disabled
    #[cfg(not(feature = "gpu"))]
    {
        CandidateScorer::score(candidates, context, /* ... */)
    }
}
```

### 9.6 GPU Configuration

```yaml
# config/config.yaml
gpu:
  enabled: true
  preferred_device: "discrete"  # or "integrated"
  max_memory_mb: 512
  batch_size: 256
  timeout_ms: 100
```

---

## 10. Unified Training Pipeline

### 10.1 Overview

The training pipeline provides unified system training through seed data ingestion, pattern learning, and spatial adjustment. Training runs as a background task with progress tracking.

### 10.2 Training Phases

| Phase | Description | Duration |
|-------|-------------|----------|
| Cold Start | Initial pattern seeding | 1-2 min |
| Growth | Active learning from feedback | Ongoing |
| Stable | Maintenance and pruning | Ongoing |

### 10.3 Training Configuration

```yaml
# config/config.yaml
training:
  phases:
    cold_start:
      batch_size: 100
      max_iterations: 10
    growth:
      batch_size: 50
      feedback_threshold: 0.7
    stable:
      prune_threshold: 0.1
      consolidate_interval: 3600  # seconds

  execution_modes:
    - silent_batch    # Background, low priority
    - interactive     # User-initiated, high priority
    - drill           # Stress testing mode
```

### 10.4 Training Data Format

```rust
pub struct LabeledDialogue {
    pub turns: Vec<LabeledTurn>,
    pub metadata: DialogueMetadata,
}

pub struct LabeledTurn {
    pub role: String,
    pub content: String,
    pub expected_entities: Vec<String>,
    pub expected_anchors: Vec<String>,
    pub expected_unit_count: ExpectedUnitCount,
    pub source_quality: Option<f32>,
}

pub struct DialogueMetadata {
    pub domain: Option<String>,
    pub complexity: String,
    pub memory_target: MemoryTarget,
    pub channels: Vec<MemoryChannel>,
    pub corroboration_threshold: f32,
}

pub struct ExpectedUnitCount {
    pub phrase: usize,
    pub sentence: usize,
    pub word: usize,
}
```

### 10.5 Training API

```rust
impl Engine {
    /// Start training with execution mode
    pub fn start_train(&self, mode: TrainingExecutionMode) -> String;
    
    /// Get training job status
    pub fn training_status(&self, job_id: &str) -> Option<TrainingJobStatus>;
    
    /// Run training plan with job ID
    pub async fn run_training_plan_with_id(
        &self,
        job_id: String,
        plan: TrainingPlan,
    ) -> TrainingJobStatus;
    
    /// Train with specific scope
    pub async fn train_with_scope(
        &self,
        execution_mode: TrainingExecutionMode,
        scope: TrainingScope,
    ) -> TrainingJobStatus;
}
```

### 10.6 Training Progress Tracking

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

pub enum JobState {
    Queued,
    Processing,
    Completed,
    Failed,
}
```

---

## 11. API Specifications

### 11.1 REST API

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
