# Configuration Tuning Guide

## Overview
Runtime and training tunables are loaded from [config/config.yaml](/Volumes/SSD/Github/spse_engine/config/config.yaml). Observation logs can be emitted as JSONL without recompiling by setting `layer_20_telemetry.observation_log_path`.

## Quick Start
1. Run the sweep tests:
```bash
cargo test --test config_sweep_test
```
2. Generate JSONL observations into `test_logs/` or a temp path by setting `layer_20_telemetry.observation_log_path`.
3. Analyze the logs:
```bash
python3 scripts/analyze_observations.py test_logs
```
4. Update [config/config.yaml](/Volumes/SSD/Github/spse_engine/config/config.yaml) with the recommended values and rerun the sweep.

## Key Parameters
- Better accuracy: increase `layer_14_candidate_scoring.evidence`, lower `layer_9_retrieval_gating.entropy_threshold`, raise `layer_19_trust_heuristics.min_corroborating_sources`.
- Better latency: raise `layer_9_retrieval_gating.entropy_threshold`, lower `layer_11_retrieval.max_retrieval_results`, increase retrieval cost weighting.
- Better memory efficiency: raise `layer_2_unit_builder.min_frequency_threshold`, lower `layer_21_memory_governance.daily_growth_limit_mb`, lower `layer_21_memory_governance.episodic_decay_days`.

## Observation Metrics
- `total_latency_ms`: end-to-end latency.
- `retrieval_triggered`: external lookup rate.
- `final_answer_confidence`: answer confidence after resolution.
- `units_discovered` / `new_units_created`: learning pressure from the prompt.
- `memory_delta_kb`: memory growth pressure.
- `score_breakdown`: top-candidate score composition when enabled.

## Notes
- Config is loaded at engine startup. Restart the process to pick up edits.
- Observation logging is opt-in through `layer_20_telemetry.observation_log_path`.
- Sweep tests use temporary DBs so they do not mutate your working memory store.
