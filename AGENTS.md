# SPSE Engine Architecture Compliance Guide

This document defines the architectural rules, layer responsibilities, and coding standards for the SPSE Engine. All changes must comply with this guide. Update this document when architectural changes are made.

## 21-Layer Architecture

The engine implements a 21-layer processing pipeline. Each layer has specific responsibilities and must not leak concerns to other layers.

### Layer Mapping

| Layer | Name | Module | Config Section | Responsibility |
|-------|------|--------|----------------|----------------|
| L1 | Input Ingestion | `classification/input.rs` | - | Normalize raw text, create `InputPacket` |
| L2 | Unit Builder | `classification/builder.rs` | `layer_2_unit_builder` | Rolling hash discovery, unit activation, scoring |
| L3 | Hierarchy Organizer | `classification/hierarchy.rs` | - | Level grouping, anchor/entity extraction |
| L4 | Memory Ingestion | `memory/store.rs` | `layer_21_memory_governance` | Unit persistence, candidate observation |
| L5 | Semantic Router | `predictive/router.rs` | `layer_5_semantic_map` | Spatial routing, neighbor selection, escape |
| L6 | Context Manager | `reasoning/context.rs` | - | Context matrix, sequence state, task entities |
| L7 | Intent Detector | `classification/intent.rs` | `intent` | Intent classification, entropy calculation |
| L8 | Adaptive Runtime | `config/mod.rs` | `adaptive_behavior` | Profile selection, weight adjustment |
| L9 | Retrieval Decision | `classification/intent.rs` | `layer_9_retrieval_gating` | Entropy/freshness/cost scoring |
| L10 | Query Builder | `classification/query.rs` | `layer_10_query_builder` | Safe query construction, PII stripping |
| L11 | Retrieval Pipeline | `reasoning/retrieval.rs` | `layer_11_retrieval` | External source fetching, caching |
| L12 | Safety Validator | `classification/safety.rs` | `layer_19_trust_heuristics` | Trust assessment, document filtering |
| L13 | Evidence Merger | `reasoning/merge.rs` | `layer_13_evidence_merge` | Conflict detection, agreement scoring |
| L14 | Candidate Scorer | `reasoning/search.rs` | `layer_14_candidate_scoring` | 7-dimensional feature scoring |
| L15 | Resolver Mode Selection | `engine.rs` | `adaptive_behavior` | Deterministic/Balanced/Exploratory |
| L16 | Fine Resolver | `predictive/resolver.rs` | `layer_16_fine_resolver` | Top-k selection, temperature-based |
| L17 | Output Decoder | `predictive/output.rs` | - | Answer finalization, sentence extraction |
| L18 | Feedback Controller | `reasoning/feedback.rs` | - | Learning events, impact scoring |
| L19 | Trust/Safety | `classification/safety.rs` | `layer_19_trust_heuristics` | Source validation, allowlist management |
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
│   ├── classification/      # Classification System (L1, L2, L3, L9, L10, L19)
│   │   ├── mod.rs
│   │   ├── input.rs         # L1: Input ingestion
│   │   ├── builder.rs       # L2: Unit builder (rolling hash)
│   │   ├── hierarchy.rs     # L3: Hierarchy organizer
│   │   ├── intent.rs        # L9: Intent & uncertainty detection
│   │   ├── query.rs         # L10: Safe query builder
│   │   ├── safety.rs        # L19: Trust & safety validation
│   │   ├── calculator.rs    # Nearest centroid classifier
│   │   ├── signature.rs     # 78-float feature vector
│   │   ├── pattern.rs       # Classification patterns
│   │   └── trainer.rs       # Classification training pipeline
│   ├── reasoning/           # Reasoning System (L7, L11, L13, L14, L18)
│   │   ├── mod.rs
│   │   ├── context.rs       # L7: Context matrix management
│   │   ├── retrieval.rs     # L11: External retrieval pipeline
│   │   ├── merge.rs         # L13: Evidence merger
│   │   ├── search.rs        # L14: 7D candidate scoring
│   │   └── feedback.rs      # L18: Feedback controller
│   ├── predictive/          # Predictive System (L5, L15, L16, L17)
│   │   ├── mod.rs
│   │   ├── router.rs        # L5: Semantic router (3D map)
│   │   ├── resolver.rs      # L16: Fine resolver
│   │   └── output.rs        # L17: Output decoder
│   ├── memory/              # Shared: Memory Store (L4, L21)
│   │   ├── mod.rs
│   │   ├── store.rs         # Memory store + governance
│   │   └── dynamic.rs       # Dynamic memory allocation
│   ├── training/            # Training infrastructure
│   │   ├── mod.rs
│   │   └── pipeline.rs      # Training pipeline, phases, plans
│   ├── telemetry/           # Telemetry & Observability (L20)
│   │   ├── mod.rs
│   │   └── trace.rs
│   ├── seed/                # Dataset generators
│   │   └── mod.rs
│   ├── gpu/                 # Optional GPU acceleration (wgpu)
│   │   └── mod.rs
│   ├── common/              # Shared utilities
│   │   └── mod.rs
│   ├── types.rs             # Core type definitions
│   ├── persistence.rs       # SQLite layer
│   ├── spatial_index.rs     # Spatial grid, force layout
│   ├── region_index.rs      # Regional spatial index
│   ├── open_sources.rs      # Source catalog
│   ├── document.rs          # Document processing
│   ├── api.rs               # REST API
│   ├── scheduler.rs         # Priority scheduler
│   ├── bloom_filter.rs      # Bloom filter
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

Each layer lives in its parent system directory:
- **Classification System** (`src/classification/`): L1, L2, L3, L9, L10, L19
- **Reasoning System** (`src/reasoning/`): L7, L11, L13, L14, L18
- **Predictive System** (`src/predictive/`): L5, L15, L16, L17
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

### Development Testing Guidelines

During development, use **minimal corpora with high density** for fast verification:

1. **Unit tests only**: `cargo test --lib --no-default-features -- <module_name>`
   - Skips GPU compilation and integration tests
   - Example: `cargo test --lib --no-default-features -- telemetry::trace`

2. **Skip GPU features**: Use `--no-default-features` flag to disable GPU acceleration
   - GPU compilation adds significant build time
   - Most logic tests don't require GPU

3. **Targeted module testing**: Filter tests by module path
   - `cargo test --lib -- telemetry::hot_store` - tests only hot_store module
   - `cargo test --lib -- memory::dynamic` - tests only dynamic memory module

4. **Fast check cycle**: Use `cargo check` before running tests
   - Catches compilation errors quickly
   - Only run full tests after check passes

5. **Minimal test data**: Tests should use small, dense datasets
   - Prefer 10-100 items over 1000+ items
   - Focus on edge cases and boundary conditions
   - Use `tempfile` for isolated test environments

6. **Avoid slow patterns**:
   - Don't use `std::thread::sleep` in tests
   - Mock time-dependent operations
   - Use in-memory SQLite (`:memory:`) instead of file-based

### Verification Commands

| Purpose | Command |
|---------|---------|
| Quick compile check | `cargo check` |
| Unit tests (no GPU) | `cargo test --lib --no-default-features` |
| Single module test | `cargo test --lib --no-default-features -- <module>` |
| Full test suite | `cargo test` |
| Clippy linting | `cargo clippy` |

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

1. Identify which system owns the layer (Classification, Reasoning, or Predictive)
2. Create file in the appropriate system directory (`src/classification/`, `src/reasoning/`, or `src/predictive/`)
3. Add module to the system's `mod.rs` exports
4. Create config struct in `src/config/mod.rs` with `layer_N_name` prefix
5. Add config section to `config/config.yaml`
6. Wire layer into `engine.rs` processing pipeline
7. Update this document's layer mapping table
8. Update README.md architecture section
9. Add unit tests in layer file

### Adding a New Intent Kind

1. Add variant to `IntentKind` enum in `types.rs`
2. Add scoring logic in `classification/intent.rs`
3. Add adaptive profile in `config/mod.rs` under `adaptive_behavior.intent_profiles`
4. Add profile to `config/config.yaml`
5. Update `docs/intent_handling.md`
6. Add test case in `classification/intent.rs` tests

## Code Review Checklist

Before submitting changes:

- [ ] All numeric values are in config, not hardcoded
- [ ] New config fields have YAML entries
- [ ] Layer boundaries respected (no cross-layer logic leakage)
- [ ] Naming conventions followed
- [ ] Tests added for new behavior
- [ ] Documentation updated (AGENTS.md, README.md)

## Forbidden Actions

1. **Hardcoding thresholds**: All numbers must be configurable
2. **Cross-layer imports**: Layers should not import from each other; use `engine.rs` orchestration
3. **Mutable global state**: Use `Arc<Mutex<T>>` through engine struct
4. **Undocumented config fields**: Every field needs a comment explaining purpose
5. **Magic strings**: All string constants should be in config or clearly documented
