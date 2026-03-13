#!/usr/bin/env python3
"""
Analyze JSONL observation logs and summarize config impact.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dig(record: dict, path: str):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return numerator / (denom_x * denom_y)


def analyze(rows: list[dict]) -> dict:
    latencies = [row.get("total_latency_ms", 0) for row in rows]
    confidences = [row.get("final_answer_confidence", 0.0) for row in rows]
    retrieval_rate = [1.0 if row.get("retrieval_triggered") else 0.0 for row in rows]
    entropy_values = [
        dig(row, "config_values_used.entropy_threshold") or 0.0 for row in rows
    ]
    w_spatial = [dig(row, "config_values_used.w_spatial") or 0.0 for row in rows]
    w_evidence = [dig(row, "config_values_used.w_evidence") or 0.0 for row in rows]
    units = [row.get("units_discovered", 0) for row in rows]
    new_units = [row.get("new_units_created", 0) for row in rows]

    return {
        "avg_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
        "avg_confidence": statistics.fmean(confidences) if confidences else 0.0,
        "avg_units_discovered": statistics.fmean(units) if units else 0.0,
        "avg_new_units": statistics.fmean(new_units) if new_units else 0.0,
        "entropy_vs_retrieval": corr(entropy_values, retrieval_rate),
        "spatial_vs_confidence": corr(w_spatial, confidences),
        "evidence_vs_confidence": corr(w_evidence, confidences),
    }


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil((pct / 100) * len(ordered)) - 1))
    return float(ordered[index])


def recommend(rows: list[dict]) -> dict:
    ranked = sorted(
        rows,
        key=lambda row: (
            row.get("final_answer_confidence", 0.0)
            / (row.get("total_latency_ms", 0) + 1)
        ),
        reverse=True,
    )[:10]
    if not ranked:
        return {}

    def avg(path: str) -> float:
        values = [dig(row, path) or 0.0 for row in ranked]
        return statistics.fmean(values) if values else 0.0

    return {
        "optimal_entropy_threshold": avg("config_values_used.entropy_threshold"),
        "optimal_w_spatial": avg("config_values_used.w_spatial"),
        "optimal_w_context": avg("config_values_used.w_context"),
        "optimal_w_evidence": avg("config_values_used.w_evidence"),
    }


def main() -> int:
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("test_logs")
    if not log_dir.exists():
        print(f"log directory not found: {log_dir}")
        return 1

    for log_file in sorted(log_dir.glob("*.jsonl")):
        print(f"\n=== Analyzing {log_file.name} ===")
        rows = load_jsonl(log_file)
        impact = analyze(rows)
        print("Config Impact Analysis:")
        for key, value in impact.items():
            print(f"  {key}: {value:.4f}")
        recommendations = recommend(rows)
        if recommendations:
            print("\nRecommended Config Values:")
            for key, value in recommendations.items():
                print(f"  {key}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
