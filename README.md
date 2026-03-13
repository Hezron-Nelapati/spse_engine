# SPSE Engine

Structured Predictive Search Engine implementing the full 21-layer architecture described in the accompanying design document.

## Product Surface

- Interactive inference loop with explainable traces
- Direct query mode for one-shot requests
- Single phased silent-training pipeline over the integrated source catalog
- Native local `.docx` / `.pdf` / `.txt` / `.md` ingestion
- Direct local document paths in conversation prompts
- Session-scoped active document workspace for follow-up questions
- Core-memory promotion for user interactions and inserted documents
- Expanded operational intent model with ambiguity fallback, verification routing, and natural-language session reset
- Training job status lookup
- SQLite-backed memory with snapshots and background maintenance
- Retrieval gating, privacy-aware query building, trust/safety validation, evidence merge, feedback, and memory governance

## CLI

Run interactive mode:

```bash
cargo run
```

Run a one-shot query:

```bash
cargo run -- query "Summarize the SPSE architecture"
```

Show human debug metadata:

```bash
cargo run -- query "Summarize the SPSE architecture" --debug
```

The debug view prints the structured per-step trace in `trace.debug_steps`, including input normalization, intent resolution, retrieval gating, reasoning merge, resolution, output, feedback, and memory state.

Return full JSON including Layer 20 trace output:

```bash
cargo run -- query "Summarize the SPSE architecture" --json
```

Start the full phased training pipeline:

```bash
cargo run -- train
cargo run --bin spse_engine -- train --execution-mode development
```

Check training status:

```bash
cargo run -- train-status <job_id>
```

Use a local document directly in a question:

```bash
cargo run -- query "\"/absolute/path/to/file.docx\" what does it say about memory?"
```

In interactive mode:

```text
User > "/absolute/path/to/file.docx"
User > what does it say about memory?
User > continue
User > clear the document context
User > /clear
User > /train
User > "/absolute/path/to/file.docx" summarize the architecture
User > /debug explain the anchor memory design
User > /trace explain the anchor memory design
```

When you paste only a local document path in interactive mode, SPSE loads it into the current session. You can then ask follow-up questions without repeating the path until you load a different document or use `/clear`.

## Training Pipeline

The product no longer exposes direct source or ad-hoc batch training. `train` runs one internal phased plan derived from the `open_sources` catalog, and `POST /api/v1/train/batch` only accepts `mode: "silent"` to start that same release pipeline.

Training is now local-first:

1. Run `cargo run --bin spse_engine -- prepare --scope full` to download and cache the catalog artifacts up front.
2. Run `cargo run --bin spse_engine -- train --scope full` after preparation completes.

The catalog keeps canonical upstream URLs in code, but training localizes named sources through the prepared-source manifest in `.spse_cache/prepared_sources.json`. If a required source has not been prepared yet, training fails fast instead of downloading mid-job.

There is also a separate streaming Hugging Face scope:

1. Run `cargo run --bin spse_engine -- train --scope huggingface --execution-mode development`
2. This scope streams rows directly from the Hugging Face dataset viewer API instead of downloading parquet shards into the local cache.
3. Rows are processed incrementally in adaptive logical pulls. The runtime groups API calls based on the configured `huggingface_streaming.initial_rows_per_pull`, while each individual Hugging Face API request stays within the viewer API row limit.

Current Hugging Face scope sources:

1. `hf_tb_smoltalk2`
2. `hf_h4_ultrachat_200k`
3. `hf_openai_gsm8k`
4. `hf_fw_fineweb_edu`
5. `hf_tb_cosmopedia`
6. `hf_karpathy_climbmix`
7. `hf_karpathy_fineweb_edu`
8. `hf_karpathy_tinystories`

Current silent-training phases:

1. Bootstrap: `wikidata`, `gsm8k_train`, `wikipedia`, `dolly_15k`
2. Validation: `natural_questions`, `squad_v2`
3. Expansion: `public_openapi_specs`, `proofwriter`, `common_crawl`

Phase behavior is metric-gated rather than epoch-gated. Expansion only runs after bootstrap metrics reach the configured thresholds for `unit_discovery_efficiency` and `semantic_routing_accuracy`. Silent training bypasses Layers 14-17 and keeps retrieval gating out of the ingest path.

Cold-start training now pre-seeds the empty database with a small bootstrap vocabulary (characters, common words, reasoning connectors, numeric spans) so Layer 2 does not start from a fully blank store. Discovery and pruning thresholds are maturity-aware: early training keeps more candidate structure, while larger mature stores tighten discovery frequency/utility floors and prune episodic noise more aggressively.

SPSE now has a second memory axis in addition to `core` vs `episodic`: `main`, `intent`, and `reasoning` channels. All trained units remain in `main`; intent and reasoning corpora can also be mirrored into their specialized channels. Runtime intent resolution consults intent memory first and falls back to the older heuristic classifier only when channel evidence is weak. Prompt processing similarly pulls reasoning support from reasoning memory before falling back to the prior generalized reasoning heuristics. Every answer path, including direct-response and document-workspace shortcuts, now emits a structured trace so you can inspect how the engine resolved intent, whether retrieval was skipped, what reasoning support was merged, how the answer was resolved, and what memory state was left behind.

Training execution modes:

- `user`: runtime-governed budgets, targeting `<1 MB/day` growth
- `development`: relaxed bootstrap budgets by phase (`5 MB`/`50 MB` for bootstrap, `2 MB`/`10 MB` for validation, `0.5 MB`/`1 MB` for expansion)

When a training job reaches its per-job or daily growth ceiling, Layer 21 prunes low-utility episodic units and the job continues. Training no longer halts on budget pressure unless a fatal source or parsing error occurs.

Training status now exposes database-health and efficiency telemetry alongside source/chunk progress: current maturity stage, core/episodic/anchor counts, units discovered per KB, pruning ratio, and anchor density.

Integrated open-source coverage now includes:

- Core KB: Wikidata, Wikipedia, DBpedia
- Corpora: Project Gutenberg, Common Crawl WET / OpenWebText
- Reasoning: GSM8K, ProofWriter, RuleTaker, StrategyQA, LogiQA, ReClor catalog support
- Intent/dialogue: OpenAssistant, Dolly, OpenOrca, MultiWOZ catalog support
- Action/procedure: public OpenAPI specs, open-license repositories, WikiHow, government SOP/manual catalog support
- Structured datasets: SQuAD 2.0 and generalized structured JSON ingestion for Natural Questions, TriviaQA, OpenTriviaQA, MS MARCO, reasoning corpora, and dialogue datasets
- Live retrieval: Wikimedia, Wikidata, PubMed Central, and OpenStreetMap Nominatim with allowlisted trust bonuses

## Architecture Mapping

- Layers 1-3: raw ingestion, dynamic unit discovery, hierarchical organization
- Layers 4-8: dual memory type (`core`/`episodic`) plus specialized channels (`main`/`intent`/`reasoning`), 3D routing, context matrix, sequence and anchors
- Layers 9-13: retrieval gating, safe query building, retrieval, normalization, evidence merge
- Layers 14-17: adaptive scoring, candidate routing with escape, fine resolution, output decoding
- Layers 18-21: feedback, trust/safety, explainability trace, governance and archival

## Intent Model

Intent handling is documented in [docs/intent_handling.md](/Volumes/SSD/Github/spse_engine/docs/intent_handling.md). It is now documentation only; runtime behavior comes from the engine logic and trained `intent` memory rather than startup ingestion of a prose guide. The runtime supports ambiguity-aware fallback, verification routing, continuation/reset intents, briefness weighting, temporal/domain/preference cues, and silent-training telemetry via per-job intent distributions.

## Verification

```bash
cargo check
cargo test
```

## Configuration Sweep Harness

The repo now includes a configuration-only optimization harness for SPS v11 Appendix B tuning. It does not modify core engine logic; it spins up isolated engines with different config profiles, ingests a controlled story via silent training, runs a fixed 20-question suite, and records Layer 20-style observations for later analysis.

Artifacts:

- Dataset generator: [scripts/test_data_generator.py](/Volumes/SSD/Github/spse_engine/scripts/test_data_generator.py)
- Generated controlled dataset: [test_data/controlled_story_dataset.json](/Volumes/SSD/Github/spse_engine/test_data/controlled_story_dataset.json)
- Profile manifest: [config/profiles.json](/Volumes/SSD/Github/spse_engine/config/profiles.json)
- Profile overrides: [config/profiles](/Volumes/SSD/Github/spse_engine/config/profiles)
- Harness binary: [src/bin/test_harness.rs](/Volumes/SSD/Github/spse_engine/src/bin/test_harness.rs)
- Analysis script: [scripts/analyze_results.py](/Volumes/SSD/Github/spse_engine/scripts/analyze_results.py)
- Layer 20 schema: [config/layer_20_schema.json](/Volumes/SSD/Github/spse_engine/config/layer_20_schema.json)

Typical run:

```bash
python3 scripts/test_data_generator.py
cargo run --bin test_harness -- \
  --profiles config/profiles.json \
  --dataset test_data/controlled_story_dataset.json \
  --output-dir results/config_sweep
python3 scripts/analyze_results.py --results results/config_sweep/config_sweep_results.json
```

Useful fast smoke run:

```bash
cargo run --bin test_harness -- \
  --profiles config/profiles.json \
  --dataset test_data/controlled_story_dataset.json \
  --output-dir results/config_sweep_smoke \
  --limit-profiles 2 \
  --limit-questions 4
```

Outputs:

- Raw run report: `results/.../config_sweep_results.json`
- Per-profile query logs and merged configs: `results/.../<profile>/`
- Summary CSV: `results/.../config_sweep_results.csv`
- Pareto plot: `results/.../config_sweep_frontier.svg`
- Recommendations: `results/.../config_recommendations.json`

The harness isolates profiles by giving each run its own SQLite store and observation log. Story ingestion uses `merge_to_core=false` and the existing silent-training batch path, which keeps the experiment inside the same training model as the product.
