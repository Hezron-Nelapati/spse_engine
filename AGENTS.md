# SPSE Engine Architecture Compliance Guide

This file defines the repository-level rules for keeping SPSE aligned with the V14.2 architecture and the repo's current simplified training/testing shape.

## Source Of Truth

The architecture target lives in [docs/SPSE_ARCHITECTURE_V14.2.md](docs/SPSE_ARCHITECTURE_V14.2.md).

Use this file to enforce:
- layer ownership
- config discipline
- the single supported training path
- the current test and verification policy

If architecture changes, update:
1. `docs/SPSE_ARCHITECTURE_V14.2.md`
2. `AGENTS.md`
3. `README.md`
4. `src/config/mod.rs`
5. `config/config.yaml`

## 21-Layer Ownership

The target processing model remains the 21-layer SPSE pipeline. Layer ownership is:

| Layer | Name | Module | Config Section | Responsibility |
|-------|------|--------|----------------|----------------|
| L1 | Input Ingestion | `classification/input.rs` | - | Normalize raw text, build `InputPacket`, apply V14.2 input shaping |
| L2 | Unit Builder | `classification/builder.rs` | `layer_2_unit_builder` | Rolling discovery, activation, unit scoring |
| L3 | Hierarchy Organizer | `classification/hierarchy.rs` | - | Group units, extract anchors/entities |
| L4 | Memory Ingestion | `memory/store.rs` | `layer_21_memory_governance` | Persist units and observations |
| L5 | Word Graph Manager | `predictive/router.rs` | `layer_5_semantic_map` | Word graph, edges, highways, layout |
| L6 | Spatial + Path Index | `reasoning/context.rs` + `spatial_index.rs` | - | Near/far lookup and spatial retrieval support |
| L7 | Intent Detector | `classification/intent.rs` | `intent` | Intent classification and uncertainty shaping |
| L8 | Adaptive Runtime | `config/mod.rs` | `adaptive_behavior` | Runtime profile selection and weighting |
| L9 | Retrieval Decision | `classification/intent.rs` | `layer_9_retrieval_gating` | Entropy/freshness/cost retrieval gating |
| L10 | Query Builder | `classification/query.rs` | `layer_10_query_builder` | Safe query construction and stripping |
| L11 | Retrieval Pipeline | `reasoning/retrieval.rs` | `layer_11_retrieval` | External retrieval, normalization, caching |
| L12 | Safety Validator | `classification/safety.rs` | `layer_19_trust_heuristics` | Trust assessment and document filtering |
| L13 | Evidence Merger | `reasoning/merge.rs` | `layer_13_evidence_merge` | Agreement/conflict merge |
| L14 | Candidate Scorer | `reasoning/search.rs` | `layer_14_candidate_scoring` | 7-dimensional candidate scoring |
| L15 | Graph Walker | `engine.rs` | `adaptive_behavior` | Runtime graph walk orchestration |
| L16 | Step Resolver | `predictive/resolver.rs` | `layer_16_fine_resolver` | Candidate resolution and mode selection |
| L17 | Sequence Assembler | `predictive/output.rs` | - | Output assembly and grounding |
| L18 | Feedback Controller | `reasoning/feedback.rs` | - | Learning events and impact propagation |
| L19 | Trust/Safety | `classification/safety.rs` | `layer_19_trust_heuristics` | Source validation and trust policy |
| L20 | Telemetry | `telemetry/` | `layer_20_telemetry` | Trace emission and observation logging |
| L21 | Governance | `memory/store.rs` | `layer_21_memory_governance` | Pruning, promotion, maintenance |

## Single Training Path

There is only one supported training pipeline:
- `src/training/system_training.rs`

Supporting rules:
- `system_training.rs` is the only training entrypoint that should generate training outputs and persist them through the engine/database flow.
- `src/training/consistency.rs` may support consistency checks, but it is not a separate training pipeline.
- Training is invoked directly from the engine and CLI. Do not add queued, job-based, or status-polled training flows.
- Do not add training job IDs, training request/status DTOs, or `/api/v1/train*` endpoints back into the codebase.
- The benchmark path for training quality/performance is `src/bin/training_benchmark.rs`.

## Configuration Rules

Every threshold, weight, limit, and behavior toggle must be config-driven.

Required:
1. Define the field in `src/config/mod.rs`
2. Add the value to `config/config.yaml`
3. Read it through engine/config structs

Forbidden:

```rust
// Hardcoded thresholds
if score >= 0.24 { ... }

// Inline config-like constants
const THRESHOLD: f32 = 0.22;

// Anonymous limits
if count > 5 { ... }
```

Required:

```rust
if score >= self.config.resolver.evidence_answer_confidence_threshold { ... }
if count > self.config.governance.max_candidate_pool { ... }
```

Layer-specific config sections use `layer_N_name` naming, for example:
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

Cross-cutting sections use descriptive names such as:
- `intent`
- `adaptive_behavior`
- `memory_budgets`
- `document`
- `ingestion_policies`
- `silent_training`
- `huggingface_streaming`
- `training_phases`

## Directory Map

The current repo shape that architectural changes must respect:

```text
spse_engine/
├── src/
│   ├── main.rs
│   ├── lib.rs
│   ├── engine.rs
│   ├── api.rs
│   ├── types.rs
│   ├── persistence.rs
│   ├── spatial_index.rs
│   ├── region_index.rs
│   ├── document.rs
│   ├── scheduler.rs
│   ├── bloom_filter.rs
│   ├── classification/
│   ├── reasoning/
│   ├── predictive/
│   ├── memory/
│   ├── telemetry/
│   ├── training/
│   │   ├── mod.rs
│   │   ├── consistency.rs
│   │   └── system_training.rs
│   ├── seed/
│   ├── common/
│   ├── gpu/
│   └── bin/
│       └── training_benchmark.rs
├── tests/
│   └── v14_2_architecture_validation_test.rs
├── config/
│   ├── config.yaml
│   ├── semantic_anchors.yaml
│   ├── profiles.json
│   └── profiles/
├── docs/
│   └── SPSE_ARCHITECTURE_V14.2.md
└── AGENTS.md
```

Notes:
- The `src/bin/` directory contains additional operational binaries, but `training_benchmark.rs` is the benchmark binary relevant to the core training path.
- `src/training/pipeline.rs` and `src/bin/test_harness.rs` are intentionally removed and must not be reintroduced without an explicit architecture change.
- The old multi-test layout under `tests/` is intentionally removed for now.

## Naming Conventions

| Item Type | Convention | Example |
|-----------|------------|---------|
| Structs | PascalCase | `MemoryStore`, `IntentDetector` |
| Enums | PascalCase | `IntentKind`, `ResolverMode` |
| Enum variants | PascalCase | `IntentKind::Question` |
| Functions | snake_case | `calculate_entropy`, `route_candidate_units` |
| Methods | snake_case | `self.ingest_hierarchy()` |
| Modules | snake_case | `memory`, `predictive`, `reasoning` |
| Config fields | snake_case | `min_confidence_floor` |
| YAML keys | snake_case | `min_frequency_threshold` |

Use config instead of SCREAMING_SNAKE_CASE thresholds wherever the value is architectural or tunable.

## Layer Boundary Rules

- Keep layer logic in its owning module.
- Use `engine.rs` to orchestrate across systems.
- Do not move training orchestration into ad hoc helpers outside `system_training.rs` unless the architecture docs are updated first.
- Do not hide architecture logic in bins, scripts, or tests.
- Prefer shared utilities in `src/common/` only when the logic is genuinely cross-cutting and not layer-specific.

## Testing Policy

Current repository policy:
- The only retained test file is `tests/v14_2_architecture_validation_test.rs`.
- Inline `#[cfg(test)]` modules are intentionally removed for now.
- Do not add new tests unless explicitly requested.
- If architecture contracts change, update the retained validation suite instead of creating parallel test tracks by default.

Preferred verification commands:

| Purpose | Command |
|---------|---------|
| Format | `cargo fmt` |
| Quick compile check | `cargo check --no-default-features` |
| Architecture validation | `cargo test --test v14_2_architecture_validation_test --no-default-features` |
| Optional linting | `cargo clippy --no-default-features` |

## Change Checklist

Before closing architectural work:

- [ ] V14.2 layer ownership still makes sense
- [ ] No extra training pipeline or job-based training flow was added
- [ ] No hardcoded thresholds or weights were introduced
- [ ] New config fields exist in both `src/config/mod.rs` and `config/config.yaml`
- [ ] `AGENTS.md`, `README.md`, and `docs/SPSE_ARCHITECTURE_V14.2.md` were updated if architecture changed
- [ ] `cargo fmt` succeeds
- [ ] `cargo check --no-default-features` succeeds
- [ ] `cargo test --test v14_2_architecture_validation_test --no-default-features` succeeds

## Forbidden Actions

1. Hardcoding thresholds, limits, or routing weights in engine logic
2. Reintroducing training job IDs, queued training orchestration, or training status polling
3. Reintroducing `src/training/pipeline.rs` or `src/bin/test_harness.rs`
4. Adding `/api/v1/train*` endpoints without an explicit architecture change
5. Reintroducing broad test scaffolding without explicit approval
6. Moving layer-owned logic into unrelated systems or binaries
7. Using undocumented config fields or magic strings for architecture behavior
