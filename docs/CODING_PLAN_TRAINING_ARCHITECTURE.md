# SPSE Architecture Implementation Plan

**Source of Truth:** [SPSE_ARCHITECTURE_V14.2.md](./SPSE_ARCHITECTURE_V14.2.md)  
**Updated:** March 17, 2026  
**Purpose:** Translate the standalone architecture into an implementation sequence that matches the current repository state.

## 1. How To Use This Plan

This plan is **derivative**. The architecture document is the standalone build contract. Use this file to decide:

1. what the current repository already provides,
2. what is still missing for architecture parity,
3. what order to implement the missing pieces in.

If this plan and the architecture document ever disagree, the architecture document wins.

## 2. Current Repository Snapshot

The repository already contains the baseline SPSE skeleton:

| Area | Status | Evidence | Notes |
|------|--------|----------|-------|
| Core module layout | Present | `src/classification/`, `src/reasoning/`, `src/predictive/`, `src/memory/`, `src/training/`, `src/telemetry/` | Matches the three-system structure at a high level |
| Engine orchestration | Present | `src/engine.rs` | Main orchestration exists |
| Memory/governance substrate | Present | `src/memory/store.rs`, `src/memory/dynamic.rs` | Dual-memory foundation exists |
| API surface | Present | `src/api.rs`, `src/api/openai_compat.rs` | REST + OpenAI-compatible endpoints exist |
| Training pipeline scaffold | Present | `src/training/pipeline.rs` | Planning and dry-run infrastructure exists |
| Dry run support | Present | `src/engine.rs`, `src/types.rs` (`DryRunReport`) | Phase-0 style validation exists |
| Classification dataset generator | Present | `src/seed/classification_generator.rs` | One of the architecture-required generators already exists |
| Validation benchmark suite | Present and green | `tests/v14_2_architecture_validation_test.rs` | Current benchmark: `116/116`, real-world slice `45/45` |

The main architecture gaps are still open:

| Area | Status | Evidence | Gap |
|------|--------|----------|-----|
| Dataset catalog naming | Drift | `src/open_sources.rs` exists, `src/dataset_catalog.rs` does not | Architecture uses `dataset_catalog.rs` as canonical name |
| Reasoning decomposer | Missing | `src/reasoning/decomposer.rs` absent | Architecture requires decomposition for multi-hop reasoning |
| Consistency checker module | Missing | `src/reasoning/consistency.rs` absent | Architecture defines an explicit cross-system consistency loop |
| Predictive walk trainer | Missing | `src/predictive/walk_trainer.rs` absent | Predictive training pipeline is not yet isolated in code |
| Seed generators | Partial | `classification_generator.rs` exists; `reasoning_generator.rs`, `predictive_generator.rs`, `consistency_generator.rs` absent | Training data generation is incomplete |
| Training sweep binary | Missing | `src/bin/training_sweep.rs` absent | Architecture requires per-system sweep tooling |
| Semantic anchor config | Missing | `config/semantic_anchors.yaml` absent | Required by semantic probe architecture |
| POS cluster config | Missing | `config/pos_clusters.yaml` absent | Required by Predictive bootstrap / perturbation design |
| Language function-word bootstrap | Missing | `config/function_words/` absent | Required by multilingual bootstrap design |
| Advanced architecture types | Missing | No source hits for `SemanticAnchor`, `MetadataSummary`, `FeedbackQueue`, `EdgeStatus`, `Tier3OveruseTracker`, `RuntimeLearningConfig` | Major V14.2 architectural features are not implemented yet |

## 3. Planning Rules

All implementation work derived from this plan must preserve these rules:

1. **Architecture before optimization.** Do not optimize paths that do not yet satisfy the architecture contract.
2. **Inference before broad training.** Close the Classification → Reasoning → Predictive inference path before expanding training breadth.
3. **Validation-first changes.** Extend or update [tests/v14_2_architecture_validation_test.rs](../tests/v14_2_architecture_validation_test.rs) whenever a core principle changes.
4. **No hardcoded thresholds.** New weights, limits, thresholds, and routing values must be added to `src/config/mod.rs` and `config/config.yaml`.
5. **System boundaries stay explicit.** Classification decides task and retrieval need; Reasoning decides answer strategy; Predictive realizes output.
6. **Answer-level verification is required.** End-to-end work is not complete until the pipeline produces validated answer text, not just a valid routing trace.

## 4. Recommended Execution Order

The shortest path to architecture parity is:

1. repository and naming alignment,
2. Classification closure,
3. Reasoning closure,
4. Predictive closure,
5. training/data closure,
6. integration and performance closure.

This order matches the dependency chain in the architecture document and minimizes rework.

## 5. Phase 0: Repository Alignment

**Goal:** Align the repo surface with the canonical architecture so future work lands in the right places.

### Work

- Normalize naming drift between `src/open_sources.rs` and the architecture's `src/dataset_catalog.rs`.
- Reconcile module exports in `src/lib.rs` with the architecture document.
- Ensure config section names in code match the architecture naming in §8.
- Check README and AGENTS layer/file naming against the canonical architecture layout after code changes land.

### Deliverables

- Canonical dataset catalog module path.
- Clean module map with no ambiguous aliases.
- Updated architecture-adjacent docs if file/module names change.

### Exit Gate

- The repository tree can be mapped one-to-one to the architecture document without relying on historical naming context.

## 6. Phase 1: Classification Closure

**Goal:** Bring the Classification System up to the V14.2 architecture contract.

### Current State

- Core classifier and trainer scaffolding exist.
- Real-world validation already proves the benchmark contract for:
  - confidence guard,
  - social token normalization behavior,
  - `creative_cue`,
  - recommendation softness detection.
- Several architecture features are still missing in production code.

### Work

1. Implement the `best_sim < epsilon -> 0.0` confidence guard in the production calculator path.
2. Ensure punctuated social tokens are normalized before social-word lookup.
3. Preserve and wire the `creative_cue` feature as a first-class structural signal.
4. Add advisory-family arbitration for `Recommend`, `Critique`, `Rewrite`, and `Plan`.
5. Add semantic probe support:
   - `src/classification/semantic_anchors.rs`
   - `config/semantic_anchors.yaml`
   - config-backed semantic probe weights and thresholds
6. Upgrade classification training to support:
   - centroid construction,
   - weight sweep,
   - confidence calibration,
   - storage/publication of centroids.

### Primary Files

- `src/classification/signature.rs`
- `src/classification/calculator.rs`
- `src/classification/trainer.rs`
- `src/classification/intent.rs`
- `src/config/mod.rs`
- `config/config.yaml`
- `config/semantic_anchors.yaml`

### Exit Gate

- Classification satisfies architecture §3.5, §3.9, and §3.10.
- Recommendation prompts no longer remain a low-margin edge in production behavior.
- Validation suite remains green and classification-specific tests cover arbitration and probe behavior.

## 7. Phase 2: Reasoning Closure

**Goal:** Bring Reasoning in line with the architecture's retrieval, consistency, and decomposition model.

### Current State

- Core context, retrieval, merge, search, feedback, and memory plumbing exist.
- Dry-run support exists.
- Architecture-critical reasoning features remain absent from source.

### Work

1. Remove the reasoning-loop retrieval dead band by checking retrieval on loop entry.
2. Add disagreement-based reasoning triggers and local-sufficiency suppressors.
3. Add `src/reasoning/decomposer.rs` for intent-aware decomposition templates.
4. Add `MetadataSummary` extraction during retrieval normalization.
5. Implement the Micro-Validator with lazy gating in evidence merge.
6. Add contradiction-aware penalties before final ranking.
7. Add `src/reasoning/consistency.rs` for R1-R7 evaluation and correction interfaces.
8. Expose standalone reasoning training / consistency entrypoints instead of keeping everything implicit inside unified ingestion.

### Primary Files

- `src/engine.rs`
- `src/reasoning/retrieval.rs`
- `src/reasoning/merge.rs`
- `src/reasoning/search.rs`
- `src/reasoning/feedback.rs`
- `src/reasoning/decomposer.rs`
- `src/reasoning/consistency.rs`
- `src/config/mod.rs`
- `config/config.yaml`

### Exit Gate

- Reasoning satisfies architecture §4.3, §4.5, §4.6, and §11.5.
- Borderline-uncertain factual prompts no longer loop without retrieval.
- Contradictory evidence is penalized upstream of generation.

## 8. Phase 3: Predictive Closure

**Goal:** Implement the Word Graph behaviors defined by V14.2 rather than leaving Predictive as a baseline router.

### Current State

- Router, resolver, output, and basic graph tests exist.
- The advanced Predictive mechanisms described in the architecture are still missing in code.

### Work

1. Add `src/predictive/walk_trainer.rs` for predictive training as a first-class module.
2. Implement runtime learning support:
   - `RuntimeLearningConfig`
   - cold-start edge injection hooks
   - reinforcement controls
3. Add `EdgeStatus` and TTG lease behavior for probationary edges.
4. Implement `FeedbackQueue` to replace mutable last-walked-edge style coupling.
5. Add Bloom-filtered context summaries plus dominant-cluster fast paths.
6. Implement hub management:
   - edge caps,
   - domain gating,
   - secondary hubs,
   - dynamic hub election.
7. Add adaptive dimensionality and anchor locking zones.
8. Implement POS-cluster / context perturbation bootstrap support.
9. Enforce predictive numerical constraints in production:
   - beam search in log-space,
   - capped A* falls back instead of emitting partial output.

### Primary Files

- `src/predictive/router.rs`
- `src/predictive/resolver.rs`
- `src/predictive/output.rs`
- `src/spatial_index.rs`
- `src/region_index.rs`
- `src/reasoning/feedback.rs`
- `src/config/mod.rs`
- `config/config.yaml`
- `config/pos_clusters.yaml`
- `config/function_words/`

### Exit Gate

- Predictive satisfies architecture §5.1 through §5.12.
- Runtime learning is bounded, explainable, and does not pollute the graph.
- Long walks and sparse graph cases follow the benchmarked numerical behavior.

## 9. Phase 4: Training and Dataset Closure

**Goal:** Complete the architecture-defined training inputs and per-system training interfaces.

### Current State

- `src/training/pipeline.rs` exists.
- Dry-run planning exists.
- Classification generator exists.
- Architecture-required dataset generators and sweep tools are still missing.

### Work

1. Add missing seed generators:
   - `src/seed/reasoning_generator.rs`
   - `src/seed/predictive_generator.rs`
   - `src/seed/consistency_generator.rs`
2. Expose per-system entrypoints:
   - `train_classification`
   - `train_reasoning`
   - `train_predictive`
   - `run_consistency_check`
3. Add `src/bin/training_sweep.rs`.
4. Wire silent-training intent quarantine into `src/training/pipeline.rs`.
5. Finalize dataset catalog naming and registration flow.
6. Ensure training plan construction reflects the architecture's phase ordering and dataset expectations.

### Primary Files

- `src/training/pipeline.rs`
- `src/engine.rs`
- `src/seed/mod.rs`
- `src/seed/reasoning_generator.rs`
- `src/seed/predictive_generator.rs`
- `src/seed/consistency_generator.rs`
- `src/bin/training_sweep.rs`
- `src/open_sources.rs` or `src/dataset_catalog.rs`

### Exit Gate

- Architecture §11.2 through §11.10 can be traced to concrete source files and executable entrypoints.
- Training can be run per system, not only through monolithic or implicit flows.

## 10. Phase 5: Integration, Verification, and Performance Closure

**Goal:** Turn architecture parity into a stable development loop.

### Work

1. Keep [tests/v14_2_architecture_validation_test.rs](../tests/v14_2_architecture_validation_test.rs) as the core benchmark contract.
2. Add integration tests for:
   - end-to-end training phases,
   - consistency correction,
   - runtime learning,
   - quarantine behavior,
   - dataset generator validity.
3. Add or refresh sweep tooling:
   - config sweep
   - pollution sweep
   - training sweep
4. Track performance against architecture budgets:
   - classification latency,
   - reasoning loop latency,
   - predictive walk latency,
   - full no-retrieval inference latency.
5. Ensure docs stay synchronized:
   - `AGENTS.md`
   - `README.md`
   - architecture doc

### Exit Gate

- Architecture benchmarks remain green.
- Integration tests cover all new modules.
- The repo has a repeatable validation loop for architecture changes.

## 11. What To Build Next

If implementation starts now, the highest-value order is:

1. `dataset_catalog` alignment,
2. Classification closure,
3. Reasoning dead-band fix + Micro-Validator,
4. Predictive runtime-learning safety features,
5. missing generators and sweep tooling.

This order gives the best chance of improving real user-visible behavior early while keeping the architecture benchmark meaningful.

## 12. Explicit Gap Checklist

The following items are currently absent from the repository and should be treated as implementation backlog until added:

- `src/dataset_catalog.rs`
- `src/reasoning/decomposer.rs`
- `src/reasoning/consistency.rs`
- `src/predictive/walk_trainer.rs`
- `src/seed/reasoning_generator.rs`
- `src/seed/predictive_generator.rs`
- `src/seed/consistency_generator.rs`
- `src/bin/training_sweep.rs`
- `config/semantic_anchors.yaml`
- `config/pos_clusters.yaml`
- `config/function_words/`
- production `SemanticAnchor`
- production `MetadataSummary`
- production `FeedbackQueue`
- production `EdgeStatus`
- production `Tier3OveruseTracker`
- production `RuntimeLearningConfig`

## 13. Verification Commands

Use these commands as the default validation loop while implementing the plan:

```bash
cargo check
cargo test --test v14_2_architecture_validation_test --no-default-features
cargo test --test v14_2_architecture_validation_test --no-default-features rw_
cargo test --test v14_2_architecture_validation_test --no-default-features rw_pipeline_
```

For targeted work, prefer focused runs first:

```bash
cargo test --lib --no-default-features -- <module_name>
```

## 14. Success Definition

This plan is complete when:

1. the repository structure matches the architecture document cleanly,
2. the three systems implement the V14.2 core principles rather than approximations,
3. the training/dataset/tooling story matches the architecture document,
4. the validation suite remains green while exercising system-by-system and end-to-end behavior,
5. future contributors can follow the architecture doc alone and use this plan only as an execution checklist.
