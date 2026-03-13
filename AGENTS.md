# SPSE Engine Architecture Compliance Guide

This document defines the architectural rules, layer responsibilities, and coding standards for the SPSE Engine. All changes must comply with this guide. Update this document when architectural changes are made.

## 21-Layer Architecture

The engine implements a 21-layer processing pipeline. Each layer has specific responsibilities and must not leak concerns to other layers.

### Layer Mapping

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
| L9 | Retrieval Decision | `layers/intent.rs` | `layer_9_retrieval_gating` | Entropy/freshness/cost scoring |
| L10 | Query Builder | `layers/query.rs` | `layer_10_query_builder` | Safe query construction, PII stripping |
| L11 | Retrieval Pipeline | `layers/retrieval.rs` | `layer_11_retrieval` | External source fetching, caching |
| L12 | Safety Validator | `layers/safety.rs` | `layer_19_trust_heuristics` | Trust assessment, document filtering |
| L13 | Evidence Merger | `layers/merge.rs` | `layer_13_evidence_merge` | Conflict detection, agreement scoring |
| L14 | Candidate Scorer | `layers/search.rs` | `layer_14_candidate_scoring` | 7-dimensional feature scoring |
| L15 | Resolver Mode Selection | `engine.rs` | `adaptive_behavior` | Deterministic/Balanced/Exploratory |
| L16 | Fine Resolver | `layers/resolver.rs` | `layer_16_fine_resolver` | Top-k selection, temperature-based |
| L17 | Output Decoder | `layers/output.rs` | - | Answer finalization, sentence extraction |
| L18 | Feedback Controller | `layers/feedback.rs` | - | Learning events, impact scoring |
| L19 | Trust/Safety | `layers/safety.rs` | `layer_19_trust_heuristics` | Source validation, allowlist management |
| L20 | Telemetry | `telemetry/` | `layer_20_telemetry` | Trace emission, observation logging |
| L21 | Governance | `memory/store.rs` | `layer_21_memory_governance` | Pruning, promotion, maintenance |

## Configuration Principles

### All Configurable Values Must Be in Config

Every numeric threshold, weight, limit, and text value must be:

1. Defined in `src/config/mod.rs` as a struct field
2. Included in `config/config.yaml` with appropriate section
3. Accessible via `self.config.<section>.<field>` in engine code

**Forbidden patterns:**
```rust
// ❌ Hardcoded magic numbers
if score >= 0.24 { ... }

// ❌ Inline string constants
const THRESHOLD: f32 = 0.22;

// ❌ Unnamed constants
if count > 5 { ... }
```

**Required patterns:**
```rust
// ✅ Config-driven thresholds
if score >= self.config.resolver.evidence_answer_confidence_threshold { ... }

// ✅ Named config fields
if count > self.config.governance.max_candidate_pool { ... }
```

### Config Section Naming Convention

Config sections use `layer_N_name` prefix for layer-specific settings:
- `layer_2_unit_builder`
- `layer_5_semantic_map`
- `layer_9_retrieval_gating`
- `layer_10_query_builder`
- `layer_11_retrieval`
- `layer_13_evidence_merge`
- `layer_14_candidate_scoring`
- `layer_16_fine_resolver`
- `layer_19_trust_heuristics`
- `layer_20_telemetry`
- `layer_21_memory_governance`

Cross-cutting concerns use descriptive names:
- `intent`
- `adaptive_behavior`
- `memory_budgets`
- `document`
- `source_policies`
- `silent_training`
- `huggingface_streaming`
- `training_phases`

## Directory Structure

```
spse_engine/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── lib.rs               # Library exports
│   ├── engine.rs            # Core engine orchestration
│   ├── config/
│   │   └── mod.rs           # Configuration structs and defaults
│   ├── layers/
│   │   ├── mod.rs
│   │   ├── input.rs         # L1
│   │   ├── builder.rs       # L2
│   │   ├── hierarchy.rs     # L3
│   │   ├── router.rs        # L5
│   │   ├── context.rs       # L6
│   │   ├── intent.rs        # L7, L9
│   │   ├── query.rs         # L10
│   │   ├── retrieval.rs     # L11
│   │   ├── safety.rs        # L12, L19
│   │   ├── merge.rs         # L13
│   │   ├── search.rs        # L14
│   │   ├── resolver.rs      # L16
│   │   ├── output.rs        # L17
│   │   └── feedback.rs      # L18
│   ├── memory/
│   │   ├── mod.rs
│   │   └── store.rs         # L4, L21
│   ├── telemetry/
│   │   ├── mod.rs
│   │   └── trace.rs         # L20
│   ├── types.rs             # Core type definitions
│   ├── persistence.rs       # SQLite layer
│   ├── spatial_index.rs     # Spatial grid, force layout
│   ├── training.rs          # Training pipeline
│   ├── open_sources.rs      # Source catalog
│   ├── document.rs          # Document processing
│   ├── api.rs               # REST API
│   └── bin/
│       └── test_harness.rs  # Config sweep harness
├── tests/
│   ├── integration.rs       # Integration tests
│   ├── config_harness_assets_test.rs
│   └── config_sweep_test.rs
├── config/
│   ├── config.yaml          # Main configuration file
│   ├── profiles.json        # Profile manifest
│   ├── layer_20_schema.json # Layer 20 JSON schema
│   └── profiles/            # Profile overrides
│       ├── balanced.yaml
│       ├── deterministic.yaml
│       └── ...
├── docs/
│   └── intent_handling.md
├── scripts/
│   ├── test_data_generator.py
│   └── analyze_results.py
├── test_data/
│   └── controlled_story_dataset.json
├── proto/
│   └── ...
├── Cargo.toml
├── README.md
└── AGENTS.md                 # This file
```

## Naming Conventions

### Rust Naming

| Item Type | Convention | Example |
|-----------|------------|---------|
| Structs | PascalCase | `MemoryStore`, `IntentDetector` |
| Enums | PascalCase | `IntentKind`, `ResolverMode` |
| Enum variants | PascalCase | `IntentKind::Factual` |
| Functions | snake_case | `calculate_entropy()`, `route_units()` |
| Methods | snake_case | `self.ingest_hierarchy()` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_LAYOUT_UNITS` (use config instead) |
| Module names | snake_case | `memory_store`, `intent_detector` |
| Config fields | snake_case | `min_confidence_floor` |
| YAML keys | snake_case | `min_frequency_threshold` |

### File Naming

- Source files: `snake_case.rs`
- Config files: `snake_case.yaml`
- Test files: `snake_case_test.rs` (unit tests in same file as source)

### Layer File Mapping

Each layer has a dedicated file in `src/layers/`:
- File name matches primary concern: `intent.rs`, `retrieval.rs`, etc.
- Multiple layers can share a file if tightly coupled (L7/L9 in `intent.rs`, L12/L19 in `safety.rs`)

## Testing Standards

### Test Organization

1. **Unit tests**: In same file as source, under `#[cfg(test)] mod tests`
2. **Integration tests**: In `tests/integration.rs`
3. **Config tests**: In `tests/config_*_test.rs`

### Test Naming

```rust
#[test]
fn <action>_<expected_outcome>() {
    // Example:
    // fn promotes_frequent_salient_phrases_to_entities()
    // fn does_not_promote_low_signal_phrases()
    // fn require_https_blocks_non_https_sources()
}
```

## Threshold Reference

Key thresholds that must remain in config:

| Threshold | Config Path | Default | Purpose |
|-----------|-------------|---------|---------|
| Evidence answer confidence | `resolver.evidence_answer_confidence_threshold` | 0.22 | Minimum confidence for evidence answers |
| Min confidence floor | `resolver.min_confidence_floor` | 0.22 | Minimum candidate score for resolution |
| Intent floor | `intent.intent_floor_threshold` | 0.40 | Minimum intent score |
| Entropy threshold | `retrieval.entropy_threshold` | 0.72 | High entropy triggers retrieval |
| Freshness threshold | `retrieval.freshness_threshold` | 0.65 | Stale context triggers retrieval |
| Min source trust | `trust.min_source_trust` | 0.35 | Minimum trust for external sources |
| Prune utility | `governance.prune_utility_threshold` | varies | Utility floor to avoid pruning |

## Making Architecture Changes

When modifying the architecture:

1. **Update this file**: Document the change in the relevant section
2. **Update README.md**: Reflect changes in architecture mapping
3. **Update config**: Add new configurable values to `config/mod.rs` and `config.yaml`
4. **Update tests**: Add tests for new behavior
5. **Run verification**: `cargo check && cargo test`

### Adding a New Layer

1. Create file in `src/layers/` with appropriate name
2. Add layer to `src/layers/mod.rs` exports
3. Create config struct in `src/config/mod.rs` with `layer_N_name` prefix
4. Add config section to `config/config.yaml`
5. Wire layer into `engine.rs` processing pipeline
6. Update this document's layer mapping table
7. Update README.md architecture section
8. Add unit tests in layer file

### Adding a New Intent Kind

1. Add variant to `IntentKind` enum in `types.rs`
2. Add scoring logic in `layers/intent.rs`
3. Add adaptive profile in `config/mod.rs` under `adaptive_behavior.intent_profiles`
4. Add profile to `config/config.yaml`
5. Update `docs/intent_handling.md`
6. Add test case in `layers/intent.rs` tests

## Code Review Checklist

Before submitting changes:

- [ ] All numeric values are in config, not hardcoded
- [ ] New config fields have YAML entries
- [ ] Layer boundaries respected (no cross-layer logic leakage)
- [ ] Naming conventions followed
- [ ] Tests added for new behavior
- [ ] Documentation updated (AGENTS.md, README.md)
- [ ] `cargo check` passes
- [ ] `cargo test` passes
- [ ] `cargo clippy` warnings addressed

## Forbidden Actions

1. **Hardcoding thresholds**: All numbers must be configurable
2. **Cross-layer imports**: Layers should not import from each other; use `engine.rs` orchestration
3. **Mutable global state**: Use `Arc<Mutex<T>>` through engine struct
4. **Undocumented config fields**: Every field needs a comment explaining purpose
5. **Magic strings**: All string constants should be in config or clearly documented
