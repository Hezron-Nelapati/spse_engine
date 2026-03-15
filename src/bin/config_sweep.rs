//! Config Sweep Benchmark Tool
//!
//! Sweeps key configuration values across a range, runs a fixed set of probe
//! queries against each configuration, and reports which settings produce the
//! best aggregate score. Results are written to `benchmarks/config_sweep_report.md`.
//!
//! Usage:
//!   cargo run --bin config_sweep [-- --iterations N]

use spse_engine::config::EngineConfig;
use spse_engine::engine::Engine;
use std::collections::BTreeMap;
use std::fs;
use uuid::Uuid;

/// A single config dimension to sweep.
struct SweepDimension {
    name: &'static str,
    config_path: &'static str,
    values: Vec<f64>,
    apply: fn(&mut EngineConfig, f64),
}

/// Result of a single sweep trial.
#[allow(dead_code)]
struct TrialResult {
    dimension: String,
    value: f64,
    avg_confidence: f32,
    avg_keyword_score: f32,
    retrieval_accuracy: f32,
    overall_score: f32,
    queries_run: usize,
}

/// Probe query with expected keywords for scoring.
struct ProbeQuery {
    prompt: &'static str,
    expected_keywords: &'static [&'static str],
    expect_retrieval: bool,
}

const PROBES: &[ProbeQuery] = &[
    ProbeQuery {
        prompt: "What is 2 * 2?",
        expected_keywords: &["4"],
        expect_retrieval: false,
    },
    ProbeQuery {
        prompt: "hi",
        expected_keywords: &["hello", "hi", "help"],
        expect_retrieval: false,
    },
    ProbeQuery {
        prompt: "Remember this: The project codename is Aurora.",
        expected_keywords: &["aurora", "remember", "noted"],
        expect_retrieval: false,
    },
    ProbeQuery {
        prompt: "What is the project codename?",
        expected_keywords: &["aurora"],
        expect_retrieval: false,
    },
    ProbeQuery {
        prompt: "Explain how a CPU works.",
        expected_keywords: &["cpu", "processor", "instruction", "execute"],
        expect_retrieval: false,
    },
    ProbeQuery {
        prompt: "Plan a weekend trip to the mountains.",
        expected_keywords: &["trip", "mountain", "plan", "travel"],
        expect_retrieval: false,
    },
];

fn build_dimensions() -> Vec<SweepDimension> {
    vec![
        SweepDimension {
            name: "evidence_answer_confidence_threshold",
            config_path: "resolver.evidence_answer_confidence_threshold",
            values: vec![0.10, 0.15, 0.20, 0.22, 0.25, 0.30, 0.40],
            apply: |cfg, v| cfg.resolver.evidence_answer_confidence_threshold = v as f32,
        },
        SweepDimension {
            name: "min_confidence_floor",
            config_path: "resolver.min_confidence_floor",
            values: vec![0.10, 0.15, 0.20, 0.22, 0.25, 0.30],
            apply: |cfg, v| cfg.resolver.min_confidence_floor = v as f32,
        },
        SweepDimension {
            name: "intent_floor_threshold",
            config_path: "intent.intent_floor_threshold",
            values: vec![0.20, 0.30, 0.40, 0.50, 0.60],
            apply: |cfg, v| cfg.intent.intent_floor_threshold = v as f32,
        },
        SweepDimension {
            name: "entropy_threshold",
            config_path: "adaptive_behavior.retrieval.entropy_threshold",
            values: vec![0.50, 0.60, 0.72, 0.85, 0.95],
            apply: |cfg, v| cfg.retrieval.entropy_threshold = v as f32,
        },
        SweepDimension {
            name: "freshness_threshold",
            config_path: "adaptive_behavior.retrieval.freshness_threshold",
            values: vec![0.40, 0.55, 0.65, 0.75, 0.85],
            apply: |cfg, v| cfg.retrieval.freshness_threshold = v as f32,
        },
        SweepDimension {
            name: "selection_temperature",
            config_path: "resolver.selection_temperature",
            values: vec![0.05, 0.10, 0.15, 0.20, 0.30],
            apply: |cfg, v| cfg.resolver.selection_temperature = v as f32,
        },
        SweepDimension {
            name: "prune_utility_threshold_stable",
            config_path: "governance.stable_prune_utility_threshold",
            values: vec![0.05, 0.08, 0.12, 0.18, 0.25],
            apply: |cfg, v| cfg.governance.stable_prune_utility_threshold = v as f32,
        },
        SweepDimension {
            name: "reasoning_trigger_confidence_floor",
            config_path: "auto_inference.reasoning_loop.trigger_confidence_floor",
            values: vec![0.20, 0.30, 0.40, 0.50, 0.60],
            apply: |cfg, v| cfg.auto_inference.reasoning_loop.trigger_confidence_floor = v as f32,
        },
        SweepDimension {
            name: "reasoning_exit_confidence",
            config_path: "auto_inference.reasoning_loop.exit_confidence_threshold",
            values: vec![0.40, 0.50, 0.60, 0.70, 0.80],
            apply: |cfg, v| cfg.auto_inference.reasoning_loop.exit_confidence_threshold = v as f32,
        },
    ]
}

#[tokio::main]
async fn main() {
    let iterations: usize = std::env::args()
        .skip_while(|a| a != "--iterations")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    println!("config_sweep: starting with {} iteration(s) per trial", iterations);

    let dimensions = build_dimensions();
    let mut all_results: Vec<TrialResult> = Vec::new();
    let mut best_per_dimension: BTreeMap<String, (f64, f32)> = BTreeMap::new();

    // Baseline run
    println!("Running baseline...");
    let baseline = run_trial(&EngineConfig::default(), iterations).await;
    println!(
        "  baseline: confidence={:.3} keyword={:.3} overall={:.3}",
        baseline.avg_confidence, baseline.avg_keyword_score, baseline.overall_score
    );
    let baseline_score = baseline.overall_score;
    all_results.push(TrialResult {
        dimension: "baseline".to_string(),
        value: 0.0,
        ..baseline
    });

    // Sweep each dimension
    for dim in &dimensions {
        println!("Sweeping {}...", dim.name);
        for &value in &dim.values {
            let mut config = EngineConfig::default();
            (dim.apply)(&mut config, value);

            let result = run_trial(&config, iterations).await;
            println!(
                "  {}={:.3}: confidence={:.3} keyword={:.3} overall={:.3}",
                dim.name, value, result.avg_confidence, result.avg_keyword_score, result.overall_score
            );

            let entry = best_per_dimension
                .entry(dim.name.to_string())
                .or_insert((value, result.overall_score));
            if result.overall_score > entry.1 {
                *entry = (value, result.overall_score);
            }

            all_results.push(TrialResult {
                dimension: dim.name.to_string(),
                value,
                ..result
            });
        }
    }

    // Generate report
    let report = render_report(baseline_score, &all_results, &best_per_dimension, &dimensions);
    fs::create_dir_all("benchmarks").expect("create benchmarks dir");
    fs::write("benchmarks/config_sweep_report.md", &report).expect("write report");

    println!("\n=== Config Sweep Complete ===");
    println!("Report: benchmarks/config_sweep_report.md");
    println!("\nOptimal values found:");
    for (name, (value, score)) in &best_per_dimension {
        let delta = score - baseline_score;
        println!(
            "  {}: {:.3} (score={:.3}, delta={:+.3})",
            name, value, score, delta
        );
    }
}

async fn run_trial(config: &EngineConfig, iterations: usize) -> TrialResult {
    let mut total_confidence = 0.0f32;
    let mut total_keyword = 0.0f32;
    let mut retrieval_correct = 0usize;
    let queries_run = PROBES.len() * iterations;

    for _ in 0..iterations {
        let db_path = temp_db_path();
        let engine = Engine::new_with_config_and_db_path(config.clone(), &db_path);

        for probe in PROBES {
            let result = engine.process(probe.prompt).await;
            total_confidence += result.confidence;

            let answer_lower = result.predicted_text.to_lowercase();
            let keyword_hits = probe
                .expected_keywords
                .iter()
                .filter(|kw| answer_lower.contains(**kw))
                .count();
            let keyword_score = if probe.expected_keywords.is_empty() {
                1.0
            } else {
                keyword_hits as f32 / probe.expected_keywords.len() as f32
            };
            total_keyword += keyword_score;

            if result.used_retrieval == probe.expect_retrieval {
                retrieval_correct += 1;
            }
        }

        let _ = fs::remove_file(&db_path);
    }

    let n = queries_run.max(1) as f32;
    let avg_confidence = total_confidence / n;
    let avg_keyword_score = total_keyword / n;
    let retrieval_accuracy = retrieval_correct as f32 / n;
    let overall_score = (0.40 * avg_keyword_score) + (0.30 * avg_confidence) + (0.30 * retrieval_accuracy);

    TrialResult {
        dimension: String::new(),
        value: 0.0,
        avg_confidence,
        avg_keyword_score,
        retrieval_accuracy,
        overall_score,
        queries_run,
    }
}

fn temp_db_path() -> String {
    let file = format!("spse_sweep_{}.db", Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}

fn render_report(
    baseline_score: f32,
    results: &[TrialResult],
    best: &BTreeMap<String, (f64, f32)>,
    dimensions: &[SweepDimension],
) -> String {
    let mut report = String::new();
    report.push_str("# Config Sweep Benchmark Report\n\n");
    report.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    report.push_str(&format!("Baseline overall score: **{:.3}**\n\n", baseline_score));

    report.push_str("## Optimal Values\n\n");
    report.push_str("| Config Parameter | Optimal Value | Score | Delta vs Baseline |\n");
    report.push_str("| --- | --- | --- | --- |\n");
    for (name, (value, score)) in best {
        let delta = score - baseline_score;
        report.push_str(&format!(
            "| `{}` | {:.3} | {:.3} | {:+.3} |\n",
            name, value, score, delta
        ));
    }

    report.push_str("\n## Sweep Details\n\n");
    for dim in dimensions {
        report.push_str(&format!("### `{}`\n\n", dim.name));
        report.push_str(&format!("Config path: `{}`\n\n", dim.config_path));
        report.push_str("| Value | Confidence | Keyword | Retrieval | Overall |\n");
        report.push_str("| --- | --- | --- | --- | --- |\n");
        for result in results.iter().filter(|r| r.dimension == dim.name) {
            report.push_str(&format!(
                "| {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |\n",
                result.value,
                result.avg_confidence,
                result.avg_keyword_score,
                result.retrieval_accuracy,
                result.overall_score
            ));
        }
        report.push('\n');
    }

    report
}
