#!/usr/bin/env python3
"""Analyze SPS config sweep results and generate Pareto recommendations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_profiles(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for result in report["results"]:
        metrics = result["metrics"]
        row = {
            "profile": result["name"],
            "goal": result["goal"],
            "accuracy_score": float(metrics["accuracy_score"]),
            "exact_match_rate": float(metrics["exact_match_rate"]),
            "semantic_similarity_avg": float(metrics["semantic_similarity_avg"]),
            "retrieval_correct_rate": float(metrics["retrieval_correct_rate"]),
            "search_precision": float(metrics["search_precision"]),
            "avg_latency_ms": float(metrics["avg_latency_ms"]),
            "p95_latency_ms": int(metrics["p95_latency_ms"]),
            "latency_budget_pass_rate": float(metrics["latency_budget_pass_rate"]),
            "memory_delta_kb": int(metrics["memory_delta_kb"]),
            "training_memory_delta_kb": int(metrics["training_memory_delta_kb"]),
            "query_memory_delta_kb": int(metrics["query_memory_delta_kb"]),
            "timeout_count": int(metrics["timeout_count"]),
            "error_count": int(metrics["error_count"]),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_dominated(candidate: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    for other in others:
        if other["profile"] == candidate["profile"]:
            continue
        if (
            other["accuracy_score"] >= candidate["accuracy_score"]
            and other["avg_latency_ms"] <= candidate["avg_latency_ms"]
            and other["memory_delta_kb"] <= candidate["memory_delta_kb"]
            and (
                other["accuracy_score"] > candidate["accuracy_score"]
                or other["avg_latency_ms"] < candidate["avg_latency_ms"]
                or other["memory_delta_kb"] < candidate["memory_delta_kb"]
            )
        ):
            return True
    return False


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not is_dominated(row, rows)]


def normalize_metric(rows: list[dict[str, Any]], key: str, invert: bool = False) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    min_value = min(values)
    max_value = max(values)
    spread = max(max_value - min_value, 1e-9)
    normalized = {}
    for row in rows:
        value = (float(row[key]) - min_value) / spread
        normalized[row["profile"]] = 1.0 - value if invert else value
    return normalized


def recommend(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frontier = pareto_frontier(rows)
    best_accuracy = max(
        rows,
        key=lambda row: (row["accuracy_score"], -row["avg_latency_ms"], -row["memory_delta_kb"]),
    )
    best_latency = min(
        rows,
        key=lambda row: (row["avg_latency_ms"], row["memory_delta_kb"], -row["accuracy_score"]),
    )

    accuracy_norm = normalize_metric(frontier, "accuracy_score")
    latency_norm = normalize_metric(frontier, "avg_latency_ms", invert=True)
    memory_norm = normalize_metric(frontier, "memory_delta_kb", invert=True)

    def balanced_score(row: dict[str, Any]) -> float:
        profile = row["profile"]
        return (
            0.5 * accuracy_norm[profile]
            + 0.3 * latency_norm[profile]
            + 0.2 * memory_norm[profile]
        )

    best_balanced = max(frontier, key=balanced_score)
    return {
        "pareto_frontier": frontier,
        "best_accuracy": best_accuracy,
        "best_latency": best_latency,
        "best_balanced": {
            **best_balanced,
            "balanced_score": balanced_score(best_balanced),
        },
    }


def write_svg_scatter(path: Path, rows: list[dict[str, Any]], frontier: list[dict[str, Any]]) -> None:
    width = 900
    height = 560
    padding = 70
    plot_width = width - (padding * 2)
    plot_height = height - (padding * 2)

    latencies = [row["avg_latency_ms"] for row in rows]
    accuracies = [row["accuracy_score"] for row in rows]
    memories = [row["memory_delta_kb"] for row in rows]

    min_latency = min(latencies)
    max_latency = max(latencies)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    min_memory = min(memories)
    max_memory = max(memories)

    latency_spread = max(max_latency - min_latency, 1e-6)
    accuracy_spread = max(max_accuracy - min_accuracy, 1e-6)
    memory_spread = max(max_memory - min_memory, 1e-6)
    frontier_profiles = {row["profile"] for row in frontier}

    def x_pos(latency: float) -> float:
        return padding + ((latency - min_latency) / latency_spread) * plot_width

    def y_pos(accuracy: float) -> float:
        return height - padding - ((accuracy - min_accuracy) / accuracy_spread) * plot_height

    def radius(memory_kb: int) -> float:
        return 6 + ((memory_kb - min_memory) / memory_spread) * 16

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Helvetica,Arial,sans-serif;font-size:12px} .axis{stroke:#444;stroke-width:1.5} .frontier{fill:#c2410c;stroke:#7c2d12;stroke-width:2} .point{fill:#2563eb;stroke:#1e3a8a;stroke-width:1.5} .label{fill:#111827}</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line class="axis" x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}"/>',
        f'<line class="axis" x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" class="label">Average latency (ms)</text>',
        f'<text x="20" y="{height/2}" transform="rotate(-90 20 {height/2})" text-anchor="middle" class="label">Accuracy score</text>',
        '<text x="70" y="32" class="label">Configuration sweep Pareto frontier</text>',
    ]

    for row in rows:
        cx = x_pos(row["avg_latency_ms"])
        cy = y_pos(row["accuracy_score"])
        r = radius(row["memory_delta_kb"])
        css_class = "frontier" if row["profile"] in frontier_profiles else "point"
        elements.append(f'<circle class="{css_class}" cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}"/>')
        elements.append(
            f'<text x="{cx + r + 6:.2f}" y="{cy + 4:.2f}" class="label">{row["profile"]}</text>'
        )

    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="results/config_sweep/config_sweep_results.json",
        help="Path to the raw JSON output from the Rust sweep harness",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV, SVG, and recommendation outputs. Defaults to the results file directory.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report = load_results(results_path)
    rows = summarize_profiles(report)
    if not rows:
        raise SystemExit("No profile rows found in results file.")

    recommendations = recommend(rows)
    write_csv(output_dir / "config_sweep_results.csv", rows)
    write_svg_scatter(
        output_dir / "config_sweep_frontier.svg",
        rows,
        recommendations["pareto_frontier"],
    )
    (output_dir / "config_recommendations.json").write_text(
        json.dumps(recommendations, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Best accuracy:", recommendations["best_accuracy"]["profile"])
    print("Best latency:", recommendations["best_latency"]["profile"])
    print("Best balanced:", recommendations["best_balanced"]["profile"])


if __name__ == "__main__":
    main()
