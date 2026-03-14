# SPSE Engine Pre-Production Training Plan

**Version:** 2.0  
**Date:** 2025-01-13  
**Status:** Draft for Review

---

## Executive Summary

This document outlines the pre-production training strategy for the SPSE Engine, a 21-layer cognitive architecture that builds memory through unit discovery, intent classification, and adaptive governance. The training process populates the engine's memory store with **custom-generated targeted high-density datasets** designed specifically for the target domain, ensuring maximum unit discovery efficiency and semantic relevance.

---

## 1. Training Architecture Overview

### 1.1 Layer Involvement in Training

| Layer | Name | Training Role |
|-------|------|---------------|
| L2 | Unit Builder | Discovers and activates units from raw text using rolling hash |
| L4 | Memory Ingestion | Persists units to memory store, observes candidates |
| L8 | Adaptive Runtime | Selects profiles based on intent and database maturity |
| L9 | Retrieval Decision | Gates external retrieval during training (bypassed) |
| L18 | Feedback Controller | Commits learning events, adjusts confidence/utility |
| L21 | Governance | Prunes low-utility units, promotes high-frequency to Core |

### 1.2 Memory Architecture

**Memory Types:**
- **Core**: Long-term, high-frequency, high-trust units (promoted from Episodic)
- **Episodic**: Short-term units with decay (default for training ingestion)

**Memory Channels:**
- **Main**: All units (required channel)
- **Intent**: Intent-dialogue patterns (for intent classification training)
- **Reasoning**: Reasoning traces and procedural patterns

**Channel Routing Rules:**
- `intent_dialogue` category sources → `[Main, Intent]`
- `reasoning` category sources → `[Main, Reasoning]`
- Default sources → `[Main]`
- Intent-channel units blocked from Core promotion when `intent_channel_core_promotion_blocked: true`

### 1.3 Database Maturity Stages

| Stage | Unit Count | Discovery Frequency | Discovery Utility | Candidate Observations |
|-------|------------|---------------------|-------------------|------------------------|
| ColdStart | < 1,000 | 1 | 0.18 | 2 |
| Growth | 1,000 - 10,000 | 2 | 0.28 | 3 |
| Stable | > 10,000 | 3 | 0.42 | 4 |

---

## 2. Training Phases

### 2.1 Phase Sequence

```
DryRun → Bootstrap → Validation → Expansion → Lifelong (Continuous)
```

**Note:** All phases use **custom targeted high-density datasets**, not open-source datasets. Dataset generation specifications are defined in Section 3.

### 2.2 Phase Details

#### Phase 1: DryRun (Pipeline Validation)

**Purpose:** Verify training pipeline functionality with minimal custom data.

**Configuration:**
```yaml
max_memory_delta_mb: 5.0
daily_growth_limit_mb: 50.0
merge_to_core: false
```

**Custom Dataset:**
| Dataset | Type | Size | Density Target | Purpose |
|---------|------|------|----------------|--------|
| `dryrun_intent_core` | StructuredJson | 10MB | 0.85 | Intent classification patterns |
| `dryrun_entity_seed` | EntityJson | 5MB | 0.90 | Core entity definitions |

**Dataset Requirements:**
- Intent patterns covering all 26 intent kinds
- Entity definitions for domain core concepts
- Minimum 500 unique phrases per intent category
- Structured with `question`, `context`, `reasoning` fields

**Success Criteria:**
- Pipeline completes without errors
- Unit discovery efficiency ≥ 0.80
- All 26 intent kinds represented
- No memory budget violations

---

#### Phase 2: Bootstrap (Foundation Building)

**Purpose:** Establish foundational domain knowledge from custom high-density datasets.

**Configuration:**
```yaml
max_memory_delta_mb: 5.0
daily_growth_limit_mb: 50.0
merge_to_core: true
min_unit_discovery_efficiency: 0.85
min_semantic_routing_accuracy: 0.75
```

**Custom Datasets (Curriculum Order):**
| Priority | Dataset | Type | Size | Density | Memory Channels | Purpose |
|----------|---------|------|------|---------|-----------------|--------|
| 120 | `domain_entities_core` | EntityJson | 50MB | 0.95 | Main, Intent | Domain entity definitions |
| 115 | `domain_procedures` | ProcedureJson | 30MB | 0.90 | Main, Reasoning | Procedural workflows |
| 110 | `domain_concepts` | ConceptJson | 40MB | 0.88 | Main | Conceptual knowledge graph |
| 105 | `intent_dialogue_seed` | DialogueJson | 25MB | 0.85 | Main, Intent | Intent classification patterns |
| 100 | `reasoning_chains` | ReasoningJson | 20MB | 0.82 | Main, Reasoning | Logical reasoning traces |

**Dataset Generation Requirements:**
- **Entity Density:** ≥ 50 entities per KB, all with unique normalized forms
- **Concept Linkage:** ≥ 3 outgoing links per concept to related concepts
- **Dialogue Coverage:** ≥ 200 examples per intent kind
- **Reasoning Depth:** 3-5 step reasoning chains with explicit intermediate states

**Expected Outcomes:**
- 8,000 - 20,000 high-quality units discovered
- Core domain entities established with high confidence
- Intent classification patterns seeded for all 26 kinds
- Reasoning patterns for domain procedures
- Database transitions from ColdStart to Growth stage

---

#### Phase 3: Validation (Quality Assurance)

**Purpose:** Validate unit quality, retrieval accuracy, and intent classification with custom benchmark datasets.

**Configuration:**
```yaml
max_memory_delta_mb: 2.0
daily_growth_limit_mb: 10.0
merge_to_core: true
```

**Custom Benchmark Datasets:**
| Priority | Dataset | Type | Size | Density | Purpose |
|----------|---------|------|------|---------|--------|
| 95 | `validation_queries` | QueryJson | 15MB | 0.88 | Query-response validation pairs |
| 92 | `intent_benchmark` | IntentTestJson | 10MB | 0.90 | Intent classification test set |
| 90 | `retrieval_ground_truth` | RetrievalJson | 12MB | 0.85 | Retrieval relevance ground truth |

**Dataset Generation Requirements:**
- **Query Coverage:** Queries spanning all domain concepts and procedures
- **Intent Distribution:** Balanced distribution across all 26 intent kinds
- **Ground Truth:** Human-validated retrieval targets with relevance scores
- **Adversarial:** 15% adversarial/ambiguous queries for robustness testing

**Quality Gates:**
- Unit discovery efficiency ≥ 0.88 (units per MB)
- Semantic routing accuracy ≥ 0.82
- Intent classification accuracy ≥ 0.85
- Retrieval precision@5 ≥ 0.80
- No pollution findings in audit

---

#### Phase 4: Expansion (Scale & Diversity)

**Purpose:** Test system scalability and expand domain coverage with large-scale custom datasets.

**Configuration:**
```yaml
max_memory_delta_mb: 2.0
daily_growth_limit_mb: 15.0
merge_to_core: true
min_unit_discovery_efficiency: 0.90
min_semantic_routing_accuracy: 0.85
```

**Custom Expansion Datasets:**
| Priority | Dataset | Type | Size | Density | Purpose |
|----------|---------|------|------|---------|--------|
| 85 | `domain_variants` | VariantJson | 100MB | 0.75 | Entity/concept paraphrases |
| 80 | `edge_cases` | EdgeCaseJson | 30MB | 0.70 | Rare scenarios and exceptions |
| 75 | `cross_domain_links` | LinkJson | 40MB | 0.80 | Cross-domain concept bridges |
| 70 | `temporal_patterns` | TemporalJson | 25MB | 0.78 | Time-based reasoning patterns |

**Dataset Generation Requirements:**
- **Variant Coverage:** ≥ 5 paraphrases per core entity/concept
- **Edge Case Ratio:** 10-15% of total training data
- **Cross-Domain Links:** Explicit links to adjacent domain concepts
- **Temporal Patterns:** Time-relative expressions and reasoning

**Quality Gates:**
- Unit discovery efficiency ≥ 0.88
- Semantic routing accuracy ≥ 0.85
- Memory budget compliance
- Candidate pool utilization < 48
- No excessive pruning required (< 5% of new units)

---

#### Phase 5: Lifelong (Continuous Domain Learning)

**Purpose:** Enable continuous learning from ongoing domain data generation and user interactions.

**Configuration:**
```yaml
max_memory_delta_mb: 1.0
daily_growth_limit_mb: 5.0
merge_to_core: true
```

**Continuous Data Sources:**
| Priority | Source | Type | Frequency | Purpose |
|----------|--------|------|-----------|--------|
| 100 | `user_interaction_log` | InteractionJson | Daily | Real user query patterns |
| 95 | `domain_updates` | UpdateJson | Weekly | Domain knowledge updates |
| 90 | `feedback_corrections` | FeedbackJson | Real-time | User corrections and ratings |
| 85 | `generated_augmentations` | AugmentJson | Daily | AI-generated variations |

**Data Generation Pipeline:**
```yaml
interaction_capture:
  enabled: true
  anonymize: true
  min_quality_score: 0.60
  
feedback_integration:
  confidence_threshold: 0.70
  utility_boost: 0.10
  
augmentation:
  enabled: true
  max_variations_per_unit: 5
  quality_filter: 0.75
```

**Quality Gates:**
- User interaction units maintain utility ≥ 0.50
- Feedback corrections applied within 24h
- Augmentation quality score ≥ 0.75
- No pollution from generated content

---

## 3. Custom Dataset Catalog

### 3.1 Dataset Type Specifications

All training data uses **custom-generated targeted high-density datasets**. The following dataset types are supported:

**Core Dataset Types:**
| Type | Format | Density Target | Memory Type | Trust Bonus |
|------|--------|----------------|-------------|-------------|
| EntityJson | `{"entity": {..., "links": [...]}` | 0.90-0.95 | Core | 0.25 |
| ConceptJson | `{"concept": {..., "related": [...]}` | 0.85-0.90 | Core | 0.20 |
| ProcedureJson | `{"procedure": {..., "steps": [...]}` | 0.85-0.92 | Core | 0.20 |
| DialogueJson | `{"dialogue": [...], "intent": "..."}` | 0.80-0.88 | Episodic | 0.15 |
| ReasoningJson | `{"premise": ..., "steps": [...], "conclusion": ...}` | 0.75-0.85 | Episodic | 0.15 |

**Supporting Dataset Types:**
| Type | Format | Density Target | Memory Type | Purpose |
|------|--------|----------------|-------------|--------|
| QueryJson | `{"query": ..., "expected_units": [...]}` | 0.85-0.90 | Episodic | Validation benchmarks |
| IntentTestJson | `{"input": ..., "expected_intent": ...}` | 0.88-0.92 | Episodic | Intent testing |
| RetrievalJson | `{"query": ..., "relevant": [...], "scores": [...]}` | 0.80-0.88 | Episodic | Retrieval ground truth |
| VariantJson | `{"canonical": ..., "variants": [...]}` | 0.70-0.80 | Episodic | Paraphrase coverage |
| EdgeCaseJson | `{"scenario": ..., "handling": ...}` | 0.65-0.75 | Episodic | Rare scenarios |
| LinkJson | `{"source": ..., "targets": [...], "weights": [...]}` | 0.75-0.85 | Core | Cross-domain links |
| TemporalJson | `{"pattern": ..., "time_refs": [...]}` | 0.70-0.80 | Episodic | Time reasoning |

### 3.2 Dataset Generation Requirements

**High-Density Criteria:**
- **Unit Density:** Minimum 40-50 discoverable units per KB
- **Unique Normalized Forms:** ≥ 95% unique normalized phrases
- **Link Density:** Average 2-3 outgoing links per entity/concept
- **Intent Coverage:** All 26 intent kinds represented
- **Reasoning Depth:** 3-5 step chains with explicit intermediate states

**Quality Metrics:**
```yaml
dataset_quality:
  min_entity_density: 50        # entities per KB
  min_unique_ratio: 0.95        # unique normalized forms
  min_link_coverage: 0.80       # entities with links
  max_noise_ratio: 0.05         # low-quality fragments
  min_intent_balance: 0.85      # distribution evenness
```

### 3.3 Domain-Specific Dataset Registry

**Bootstrap Phase Datasets:**
| Dataset ID | Type | Size | Density | Description |
|------------|------|------|---------|-------------|
| `domain_entities_core` | EntityJson | 50MB | 0.95 | Core domain entities with definitions |
| `domain_procedures` | ProcedureJson | 30MB | 0.90 | Domain-specific workflows and procedures |
| `domain_concepts` | ConceptJson | 40MB | 0.88 | Conceptual knowledge with relationships |
| `intent_dialogue_seed` | DialogueJson | 25MB | 0.85 | Intent classification training dialogues |
| `reasoning_chains` | ReasoningJson | 20MB | 0.82 | Domain reasoning patterns |

**Validation Phase Datasets:**
| Dataset ID | Type | Size | Density | Description |
|------------|------|------|---------|-------------|
| `validation_queries` | QueryJson | 15MB | 0.88 | Query-response validation pairs |
| `intent_benchmark` | IntentTestJson | 10MB | 0.90 | Intent classification test set |
| `retrieval_ground_truth` | RetrievalJson | 12MB | 0.85 | Retrieval relevance ground truth |

**Expansion Phase Datasets:**
| Dataset ID | Type | Size | Density | Description |
|------------|------|------|---------|-------------|
| `domain_variants` | VariantJson | 100MB | 0.75 | Entity/concept paraphrases |
| `edge_cases` | EdgeCaseJson | 30MB | 0.70 | Rare scenarios and exceptions |
| `cross_domain_links` | LinkJson | 40MB | 0.80 | Cross-domain concept bridges |
| `temporal_patterns` | TemporalJson | 25MB | 0.78 | Time-based reasoning patterns |

### 3.4 Dataset Generation Pipeline

**Generation Workflow:**
```
Domain Analysis → Schema Definition → Content Generation → Quality Filtering → Density Validation
```

**Generation Tools:**
1. **Schema Extractor:** Analyzes domain to extract entity/concept schemas
2. **Content Generator:** Generates high-density content following schemas
3. **Quality Filter:** Removes low-quality fragments, enforces density
4. **Density Validator:** Verifies unit discovery efficiency targets

**Output Format:**
```json
{
  "dataset_id": "domain_entities_core",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "EntityJson",
  "density_score": 0.95,
  "unit_count_estimate": 2500,
  "entities": [
    {
      "id": "entity_001",
      "name": "Example Entity",
      "normalized": "example entity",
      "definition": "...",
      "links": [{"target": "entity_002", "type": "related", "weight": 0.8}]
    }
  ]
}
```

---

## 4. Unit Discovery Process

### 4.1 Rolling Hash Discovery

The Unit Builder (L2) uses rolling hash algorithms to identify recurring patterns:

**Window Sizes:** `[3, 4, 5, 6, 7, 8]` characters

**Discovery Thresholds (Adaptive):**
```rust
struct DiscoveryThresholds {
    min_frequency: u64,      // Based on maturity stage
    min_utility: f32,        // Based on maturity stage
    training_mode: bool,     // Relaxed thresholds during training
}
```

**Unit Levels:**
- `Char` - Single characters (rarely promoted)
- `Subword` - Character sequences
- `Word` - Complete words
- `Phrase` - Multi-word phrases
- `Pattern` - Recurring structural patterns

### 4.2 Unit Scoring

**Utility Score Formula:**
```
utility = base_scale 
        + (compression_gain * span_weight)
        + (frequency * frequency_weight)
        + (density * density_weight)
        + boundary_bonus/penalty
```

**Salience Score Formula:**
```
salience = base 
         + (digit_ratio * digit_weight)
         + (upper_ratio * upper_weight)
         - (reuse_count / reuse_divisor)
```

**Confidence Score Formula:**
```
confidence = base 
           + (length_factor * length_weight)
           + boundary_bonus/penalty
           - (reuse_count / reuse_divisor)
```

### 4.3 Fragment Filtering

Units are rejected if they match:
- Unicode escape sequences (`\uXXXX`, `\xNN`)
- URL/path fragments (`http://`, `file://`, `/`)
- File extensions (`.txt`, `.rs`, etc.)
- Excessive punctuation ratio (> 0.35)
- Minimum length < 5 characters

---

## 5. Memory Governance

### 5.1 Promotion Rules

**Episodic → Core Promotion:**
```yaml
core_promotion_threshold: 6        # Frequency required
promotion_min_corroborations: 3    # External confirmations
anchor_reuse_threshold: 3          # Anchor activation count
anchor_salience_threshold: 0.70    # Salience for anchor status
```

**Intent Channel Blocking:**
- When `intent_channel_core_promotion_blocked: true`
- Intent-channel units remain Episodic
- Prevents noisy intent signals from polluting Core memory

### 5.2 Pruning Rules

**Prune Thresholds by Maturity:**
| Stage | Prune Utility Threshold |
|-------|------------------------|
| ColdStart | 0.08 |
| Growth | 0.12 (default) |
| Stable | 0.18 |

**Pruning Conditions:**
```yaml
conditions:
  - memory_type: Episodic
  - utility_score < prune_threshold
  - stale_hours > 24
  - anchor_status: false
  - anchor_grace_active: false
```

**Protected Units:**
- Anchor units (high salience, reused)
- Units within anchor protection grace period (14 days)
- Core memory units (never pruned)

### 5.3 Pollution Detection

**Pollution Findings:**
- Overlapping content (similarity > 0.70)
- Low-quality fragments (quality margin < 0.08)
- Edge trimming issues (edge trim limit: 3)

**Remediation:**
- Absorb polluted units into canonical units
- Archive polluted units
- Merge scores into canonical

---

## 6. Parallel Training Architecture

### 6.1 Worker Configuration

```yaml
parallel:
  enabled: true
  worker_count: 0                    # 0 = auto-detect
  queue_capacity: 64
  queue_capacity_per_worker: 4
  commit_batch_size: 8
  commit_flush_interval_ms: 250
  total_memory_limit_mb: 2560.0
  non_worker_memory_reserve_mb: 1280.0
  local_shard_soft_limit_mb: 96.0
  local_shard_hard_limit_mb: 128.0
```

### 6.2 Shard Merging Process

1. **Ingestion:** Workers process sources into local shards
2. **Merging:** Main store merges shard units
3. **Deduplication:** Content index prevents duplicates
4. **Link Resolution:** Cross-shard links reconnected
5. **Candidate Promotion:** Validated candidates activated

**Merge Scoring:**
```rust
// Existing unit merge
frequency += incoming.frequency.max(1);
utility = (existing * 0.70) + (incoming * 0.30);
salience = (existing * 0.65) + (incoming * 0.35);
confidence = existing.max(incoming);
```

---

## 7. Feedback and Learning

### 7.1 Feedback Events (L18)

**Event Types:**
| Layer | Event | Impact |
|-------|-------|--------|
| 18 | retrieval_path_observed | 0.6 |
| 18 | internal_path_observed | 0.4 |
| 5 | routing_adjustments_committed | 0.1 - 0.8 |
| 8 | anchor_reuse_detected | 0.5 |
| 21 | maintenance_pruned/promoted | 0.4 |

**Impact Application:**
```rust
confidence = (confidence + total_impact * 0.05).clamp(0.05, 1.0);
utility = (utility + total_impact * 0.03).clamp(0.05, 2.0);
```

### 7.2 Intent Classification Training

**Intent Kinds (26 total):**
- Social: Greeting, Gratitude, Farewell
- Assistance: Help, Clarify, Rewrite, Verify, Continue, Forget
- Information: Question, Summarize, Explain, Compare, Extract, Analyze
- Action: Plan, Act, Recommend, Classify, Translate, Debug
- Creative: Critique, Brainstorm
- Fallback: Unknown

**Intent Profiles (Adaptive):**
| Profile | Temperature | Confidence Floor | Mode |
|---------|-------------|------------------|------|
| factual | 0.10 | 0.30 | Deterministic |
| explanatory | 0.30 | 0.24 | Balanced |
| procedural | 0.22 | 0.28 | Balanced |
| creative | 0.75 | 0.18 | Exploratory |
| brainstorm | 0.90 | 0.12 | Exploratory |

---

## 8. Execution Modes

### 8.1 Development Mode

Used for training runs with relaxed constraints:
```yaml
max_memory_delta_mb: phase_default
daily_growth_limit_mb: phase_default
bypass_retrieval_gate: true
bypass_generation: true
```

### 8.2 User Mode

Used for production inference:
```yaml
max_memory_delta_mb: 0.5
daily_growth_limit_mb: 1.0
bypass_retrieval_gate: false
bypass_generation: false
```

---

## 9. Pre-Production Checklist

### 9.1 Environment Setup

- [ ] Database path configured
- [ ] Memory budgets allocated (ColdStart: 50MB Episodic, 10MB Core)
- [ ] GPU acceleration enabled (if available)
- [ ] Telemetry worker configured
- [ ] Custom dataset storage path configured

### 9.2 Configuration Validation

- [ ] `config.yaml` loaded successfully
- [ ] All layer configurations present
- [ ] Dataset type policies defined for all custom types
- [ ] Intent profiles configured
- [ ] Governance thresholds appropriate for hardware

### 9.3 Dataset Availability

- [ ] Custom datasets generated for all phases
- [ ] Dataset density validated (≥ target density)
- [ ] Dataset quality metrics verified
- [ ] Dataset storage accessible
- [ ] Dataset integrity verified (checksums)

### 9.4 Monitoring Setup

- [ ] Telemetry hot store path writable
- [ ] Observation logging enabled
- [ ] Latency monitoring active
- [ ] Memory usage tracking enabled

---

## 10. Execution Plan

### 10.1 Phase Execution Order

```bash
# Phase 1: DryRun (Pipeline Validation)
cargo run --release -- --train --dataset datasets/dryrun_intent_core.json --mode development
cargo run --release -- --train --dataset datasets/dryrun_entity_seed.json --mode development

# Verify success criteria before proceeding
cargo test --release

# Phase 2: Bootstrap (Foundation Building)
cargo run --release -- --train --dataset datasets/domain_entities_core.json --mode development
cargo run --release -- --train --dataset datasets/domain_procedures.json --mode development
cargo run --release -- --train --dataset datasets/domain_concepts.json --mode development
cargo run --release -- --train --dataset datasets/intent_dialogue_seed.json --mode development
cargo run --release -- --train --dataset datasets/reasoning_chains.json --mode development

# Phase 3: Validation (Quality Assurance)
cargo run --release -- --train --dataset datasets/validation_queries.json --mode development
cargo run --release -- --train --dataset datasets/intent_benchmark.json --mode development
cargo run --release -- --train --dataset datasets/retrieval_ground_truth.json --mode development

# Phase 4: Expansion (Scale & Diversity)
cargo run --release -- --train --dataset datasets/domain_variants.json --mode development
cargo run --release -- --train --dataset datasets/edge_cases.json --mode development
cargo run --release -- --train --dataset datasets/cross_domain_links.json --mode development
cargo run --release -- --train --dataset datasets/temporal_patterns.json --mode development

# Phase 5: Lifelong (Continuous Domain Learning)
# Enable continuous learning mode
cargo run --release -- --enable-continuous-learning --mode production
```

### 10.2 Monitoring Commands

```bash
# Check memory health
cargo run --release -- --status

# Audit pollution
cargo run --release -- --audit-pollution

# Validate channel isolation
cargo run --release -- --validate-channels

# View training progress
tail -f telemetry/cold.log
```

### 10.3 Rollback Procedures

If training produces undesirable results:

```bash
# Restore from snapshot
cargo run --release -- --restore-snapshot <path>

# Archive and reset
cargo run --release -- --archive-all
cargo run --release -- --reset-memory

# Re-run from specific phase
cargo run --release -- --train --scope bootstrap --mode development
```

---

## 11. Success Metrics

### 11.1 Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Unit Discovery Efficiency | ≥ 0.90 | Units per MB ingested |
| Semantic Routing Accuracy | ≥ 0.85 | Correct neighbor selection |
| Intent Classification Accuracy | ≥ 0.85 | Correct intent prediction |
| Retrieval Precision@5 | ≥ 0.80 | Relevant results in top 5 |
| Core Memory Ratio | 20-30% | Core units / Total units |
| Candidate Pool Utilization | < 48 | Active candidates |
| Daily Growth Compliance | 100% | Within daily limit |
| Pollution Findings | 0 | Clean audit |
| Dataset Density Score | ≥ 0.80 | Average across all datasets |

### 11.2 Qualitative Metrics

- Intent classification accuracy (manual spot-check)
- Retrieval relevance (sample query testing)
- Response coherence (end-to-end testing)
- Memory stability over time (no rapid pruning)

---

## 12. Risk Mitigation

### 12.1 Memory Overflow

**Risk:** Exceeding memory budgets during large-scale ingestion.

**Mitigation:**
- Monitor `estimate_memory_kb()` during training
- Enable `cold_archive_threshold_mb` (100MB default)
- Use `daily_growth_limit_mb` to cap ingestion rate

### 12.2 Pollution Contamination

**Risk:** Low-quality units polluting memory.

**Mitigation:**
- Enable `pollution_detection_enabled: true`
- Run `audit_pollution()` after each phase
- Review `pruned_references` in governance report

### 12.3 Channel Isolation Violation

**Risk:** Intent/Reasoning channels leaking into Core memory.

**Mitigation:**
- Enable `intent_channel_core_promotion_blocked: true`
- Run `validate_channel_isolation()` periodically
- Review channel overlap ratios

### 12.4 Candidate Pool Explosion

**Risk:** Candidate pool growing unbounded.

**Mitigation:**
- Set `max_candidate_pool: 48`
- Configure `candidate_activation_utility_threshold: 0.55`
- Monitor `candidate_pool_memory_kb()`

---

## 13. Post-Training Validation

### 13.1 Automated Tests

```bash
# Run integration tests
cargo test --release

# Run config sweep tests
cargo test --release config_sweep_test

# Run GPU fallback tests
cargo test --release gpu_fallback_test
```

### 13.2 Manual Validation

1. Query engine with sample questions
2. Verify intent classification accuracy
3. Check retrieval relevance
4. Assess response quality
5. Review memory summary statistics

### 13.3 Performance Benchmarks

```bash
# Run benchmark suite
cargo run --release --bin benchmark_eval

# Review last report
cat benchmarks/last_report.md
```

---

## 14. Appendices

### A. Configuration Reference

See `config/config.yaml` for complete configuration schema.

### B. Custom Dataset Type Reference

See Section 3 for dataset type specifications and generation requirements.

### C. Layer Architecture Reference

See `AGENTS.md` for complete layer mapping and responsibilities.

### D. Intent Handling Reference

See `docs/intent_handling.md` for intent classification details.

### E. Dataset Generation Guide

See `docs/DATASET_GENERATION_GUIDE.md` (to be created) for detailed instructions on generating custom high-density datasets.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-13 | Cascade | Initial draft with open-source datasets |
| 2.0 | 2025-01-13 | Cascade | Updated to use custom targeted high-density datasets |

---

*This document is a living specification. Update as training progresses and lessons are learned.*
