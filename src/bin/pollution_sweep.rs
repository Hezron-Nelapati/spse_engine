//! Pollution Control Sweep Tool
//!
//! Sweeps pollution-related configuration values, ingests test content into
//! memory, then audits the database for polluted content under each setting.
//! Reports which pollution config values best prevent contamination while
//! preserving valid content.
//!
//! Usage:
//!   cargo run --bin pollution_sweep

use spse_engine::config::{EngineConfig, GovernanceConfig};
use spse_engine::engine::Engine;
use std::collections::BTreeMap;
use std::fs;
use uuid::Uuid;

/// A pollution config dimension to sweep.
struct PollutionDimension {
    name: &'static str,
    config_path: &'static str,
    values: Vec<f64>,
    apply: fn(&mut GovernanceConfig, f64),
}

/// Result of a single pollution trial.
#[allow(dead_code)]
struct PollutionTrialResult {
    dimension: String,
    value: f64,
    total_units: u64,
    pollution_findings: u64,
    pollution_rate: f32,
    valid_units_preserved: u64,
    preservation_rate: f32,
    overall_score: f32,
}

/// Test content: mix of valid and intentionally polluted content.
const VALID_CONTENT: &[&str] = &[
    "The Rust programming language emphasizes safety and performance through its ownership system.",
    "Machine learning models require training data to learn patterns and make predictions.",
    "HTTP status code 200 means the request was successful and the response body contains the result.",
    "A binary search tree maintains sorted order with O(log n) lookup time complexity.",
    "Remember this: The project deadline is March 15th and the lead engineer is Sarah.",
    "The TCP three-way handshake establishes a connection using SYN, SYN-ACK, and ACK packets.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "Database normalization reduces data redundancy by organizing tables into related structures.",
    "The observer pattern allows objects to subscribe to events and receive notifications of changes.",
    "Git uses a directed acyclic graph to track commits, branches, and merge history.",
];

const POLLUTED_CONTENT: &[&str] = &[
    "the the the the the the the the the the",
    "",
    "   ",
    "aaaaaaaaaaaaaaaaaaaaaaaa",
    "123,456,789.00,123,456,789.00",
    "<script>alert('xss')</script>cookie accept",
    "click here click here click here click here click here",
    "ab",
    "javascript:void(0) cookie accept terms",
    "lorem lorem lorem ipsum ipsum ipsum dolor dolor dolor sit sit sit",
    "buy now buy now buy now free free free discount discount discount",
    "...   ...   ...   ...   ...   ...",
];

fn build_pollution_dimensions() -> Vec<PollutionDimension> {
    vec![
        PollutionDimension {
            name: "pollution_detection_enabled",
            config_path: "governance.pollution_detection_enabled",
            values: vec![0.0, 1.0], // 0=false, 1=true
            apply: |cfg, v| cfg.pollution_detection_enabled = v > 0.5,
        },
        PollutionDimension {
            name: "pollution_min_length",
            config_path: "governance.pollution_min_length",
            values: vec![2.0, 3.0, 5.0, 8.0, 12.0],
            apply: |cfg, v| cfg.pollution_min_length = v as usize,
        },
        PollutionDimension {
            name: "pollution_similarity_threshold",
            config_path: "governance.pollution_similarity_threshold",
            values: vec![0.20, 0.35, 0.50, 0.65, 0.80],
            apply: |cfg, v| cfg.pollution_similarity_threshold = v as f32,
        },
        PollutionDimension {
            name: "pollution_overlap_threshold",
            config_path: "governance.pollution_overlap_threshold",
            values: vec![0.40, 0.55, 0.65, 0.75, 0.90],
            apply: |cfg, v| cfg.pollution_overlap_threshold = v as f32,
        },
        PollutionDimension {
            name: "pollution_quality_margin",
            config_path: "governance.pollution_quality_margin",
            values: vec![0.01, 0.03, 0.05, 0.08, 0.15],
            apply: |cfg, v| cfg.pollution_quality_margin = v as f32,
        },
        PollutionDimension {
            name: "pollution_penalty_factor",
            config_path: "governance.pollution_penalty_factor",
            values: vec![0.05, 0.10, 0.20, 0.35, 0.50],
            apply: |cfg, v| cfg.pollution_penalty_factor = v as f32,
        },
        PollutionDimension {
            name: "pollution_edge_trim_limit",
            config_path: "governance.pollution_edge_trim_limit",
            values: vec![1.0, 2.0, 3.0, 5.0, 8.0],
            apply: |cfg, v| cfg.pollution_edge_trim_limit = v as usize,
        },
    ]
}

#[tokio::main]
async fn main() {
    println!("pollution_sweep: starting pollution config sweep");

    let dimensions = build_pollution_dimensions();
    let mut all_results: Vec<PollutionTrialResult> = Vec::new();
    let mut best_per_dimension: BTreeMap<String, (f64, f32)> = BTreeMap::new();

    // Baseline run (default config)
    println!("Running baseline...");
    let baseline = run_pollution_trial(&EngineConfig::default()).await;
    println!(
        "  baseline: units={} findings={} rate={:.3} preserved={} overall={:.3}",
        baseline.total_units, baseline.pollution_findings, baseline.pollution_rate,
        baseline.valid_units_preserved, baseline.overall_score
    );
    let baseline_score = baseline.overall_score;
    all_results.push(PollutionTrialResult {
        dimension: "baseline".to_string(),
        value: 0.0,
        ..baseline
    });

    // Sweep each dimension
    for dim in &dimensions {
        println!("Sweeping {}...", dim.name);
        for &value in &dim.values {
            let mut config = EngineConfig::default();
            (dim.apply)(&mut config.governance, value);

            let result = run_pollution_trial(&config).await;
            println!(
                "  {}={:.2}: units={} findings={} rate={:.3} preserved={} overall={:.3}",
                dim.name, value, result.total_units, result.pollution_findings,
                result.pollution_rate, result.valid_units_preserved, result.overall_score
            );

            let entry = best_per_dimension
                .entry(dim.name.to_string())
                .or_insert((value, result.overall_score));
            if result.overall_score > entry.1 {
                *entry = (value, result.overall_score);
            }

            all_results.push(PollutionTrialResult {
                dimension: dim.name.to_string(),
                value,
                ..result
            });
        }
    }

    // Generate report
    let report = render_report(baseline_score, &all_results, &best_per_dimension, &dimensions);
    fs::create_dir_all("benchmarks").expect("create benchmarks dir");
    fs::write("benchmarks/pollution_sweep_report.md", &report).expect("write report");

    println!("\n=== Pollution Sweep Complete ===");
    println!("Report: benchmarks/pollution_sweep_report.md");
    println!("\nOptimal pollution config values:");
    for (name, (value, score)) in &best_per_dimension {
        let delta = score - baseline_score;
        println!(
            "  {}: {:.3} (score={:.3}, delta={:+.3})",
            name, value, score, delta
        );
    }
}

async fn run_pollution_trial(config: &EngineConfig) -> PollutionTrialResult {
    let db_path = temp_db_path();
    let engine = Engine::new_with_config_and_db_path(config.clone(), &db_path);

    // Ingest valid content
    for content in VALID_CONTENT {
        let _ = engine.process(content).await;
    }

    // Ingest polluted content
    for content in POLLUTED_CONTENT {
        if !content.trim().is_empty() {
            let _ = engine.process(content).await;
        }
    }

    // Run maintenance to trigger pollution detection
    let governance_report = engine.run_maintenance();

    // Audit remaining pollution
    let pollution_findings = engine.audit_pollution(100);
    let findings_count = pollution_findings.len() as u64
        + governance_report.as_ref().map(|r| r.purged_polluted_units).unwrap_or(0);

    // Check how many valid probes are still answerable
    let mut valid_preserved = 0u64;
    let valid_probes = [
        ("What is the project deadline?", &["march", "15"] as &[&str]),
        ("What does HTTP 200 mean?", &["successful", "200"]),
        ("What is binary search tree complexity?", &["log", "sorted"]),
    ];
    for (prompt, keywords) in &valid_probes {
        let result = engine.process(prompt).await;
        let answer_lower = result.predicted_text.to_lowercase();
        if keywords.iter().any(|kw| answer_lower.contains(kw)) {
            valid_preserved += 1;
        }
    }

    let total_units = engine.memory_unit_count();
    let pollution_rate = if total_units > 0 {
        findings_count as f32 / total_units as f32
    } else {
        0.0
    };
    let preservation_rate = valid_preserved as f32 / valid_probes.len().max(1) as f32;

    let _ = fs::remove_file(&db_path);

    // Score: maximize preservation, minimize pollution
    let overall_score = (0.50 * preservation_rate)
        + (0.50 * (1.0 - pollution_rate).max(0.0));

    PollutionTrialResult {
        dimension: String::new(),
        value: 0.0,
        total_units,
        pollution_findings: findings_count,
        pollution_rate,
        valid_units_preserved: valid_preserved,
        preservation_rate,
        overall_score,
    }
}

fn temp_db_path() -> String {
    let file = format!("spse_pollution_{}.db", Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}

fn render_report(
    baseline_score: f32,
    results: &[PollutionTrialResult],
    best: &BTreeMap<String, (f64, f32)>,
    dimensions: &[PollutionDimension],
) -> String {
    let mut report = String::new();
    report.push_str("# Pollution Control Sweep Report\n\n");
    report.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    ));
    report.push_str(&format!("Baseline overall score: **{:.3}**\n\n", baseline_score));

    report.push_str("## Optimal Pollution Config Values\n\n");
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
        report.push_str("| Value | Units | Findings | Pollution Rate | Preserved | Overall |\n");
        report.push_str("| --- | --- | --- | --- | --- | --- |\n");
        for result in results.iter().filter(|r| r.dimension == dim.name) {
            report.push_str(&format!(
                "| {:.3} | {} | {} | {:.3} | {} | {:.3} |\n",
                result.value,
                result.total_units,
                result.pollution_findings,
                result.pollution_rate,
                result.valid_units_preserved,
                result.overall_score
            ));
        }
        report.push('\n');
    }

    report
}
