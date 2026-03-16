//! Zero-Shot Test Harness for SPSE Engine
//!
//! Validates retrieval gating behavior in untrained scenarios.
//! Tests Layers 9-13: Intent Detection, Query Sanitization, Retrieval, Evidence Merge, Safety.
//!
//! Usage:
//!   cargo run --bin zero_shot_harness -- [options]
//!
//! Options:
//!   --scenarios <PATH>   Path to test scenarios JSON (default: test_data/zero_shot_scenarios.json)
//!   --output <PATH>      Output report path (default: benchmarks/zero_shot_report.json)
//!   --entropy <VALUE>    Override entropy threshold (default: 0.5 for zero-shot)
//!   --reset-db           Reset memory databases before running tests

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use spse_engine::config::EngineConfig;
use spse_engine::engine::Engine;
use spse_engine::types::IntentKind;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestScenario {
    id: String,
    category: String,
    prompt: String,
    expected_retrieval_triggered: bool,
    expected_intent: String,
    expected_keywords: Vec<String>,
    #[serde(default)]
    expected_fallback_mode: String,
    #[serde(default)]
    notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioFile {
    metadata: ScenarioMetadata,
    scenarios: Vec<TestScenario>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScenarioMetadata {
    version: String,
    description: String,
    total_scenarios: u32,
    categories: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestResult {
    scenario_id: String,
    passed: bool,
    retrieval_triggered: bool,
    expected_retrieval: bool,
    intent_detected: String,
    expected_intent: String,
    fallback_mode: String,
    expected_fallback: String,
    keywords_found: Vec<String>,
    expected_keywords: Vec<String>,
    message: String,
    details: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CategoryStats {
    total: u32,
    passed: u32,
    failed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HarnessReport {
    total_tests: u32,
    passed: u32,
    failed: u32,
    skipped: u32,
    pass_rate: f64,
    results: Vec<TestResult>,
    category_stats: HashMap<String, CategoryStats>,
    start_time: String,
    end_time: String,
    duration_seconds: f64,
    config: HashMap<String, serde_json::Value>,
}

fn temp_db_path(name: &str) -> String {
    let file = format!("{}_{}.db", name, Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}

fn load_scenarios(path: &str) -> Result<ScenarioFile, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let scenarios: ScenarioFile = serde_json::from_str(&content)?;
    Ok(scenarios)
}

fn parse_intent(intent_str: &str) -> IntentKind {
    match intent_str {
        "Question" => IntentKind::Question,
        "Verify" => IntentKind::Verify,
        "Compare" => IntentKind::Compare,
        "Explain" => IntentKind::Explain,
        "Summarize" => IntentKind::Summarize,
        "Greeting" => IntentKind::Greeting,
        "Gratitude" => IntentKind::Gratitude,
        "Farewell" => IntentKind::Farewell,
        "Help" => IntentKind::Help,
        "Translate" => IntentKind::Translate,
        "Continue" => IntentKind::Continue,
        "Rewrite" => IntentKind::Rewrite,
        "Brainstorm" => IntentKind::Brainstorm,
        "Plan" => IntentKind::Plan,
        "Act" => IntentKind::Act,
        "Critique" => IntentKind::Critique,
        "Advisory" => IntentKind::Recommend,
        "Casual" => IntentKind::Unknown,
        _ => IntentKind::Unknown,
    }
}

fn check_keywords_in_response(response_text: &str, keywords: &[String]) -> Vec<String> {
    let response_lower = response_text.to_lowercase();
    keywords
        .iter()
        .filter(|kw| response_lower.contains(&kw.to_lowercase()))
        .cloned()
        .collect()
}

async fn run_scenario(engine: &Engine, scenario: &TestScenario) -> TestResult {
    let result = engine.process(&scenario.prompt).await;

    // Extract Layer 9 decision from intent profile
    let fallback_mode = format!("{:?}", result.trace.intent_profile.fallback_mode);
    let retrieval_triggered = result.used_retrieval
        || result.trace.intent_profile.fallback_mode
            == spse_engine::types::IntentFallbackMode::RetrieveUnknown;

    // Extract intent
    let intent_detected = format!("{:?}", result.trace.intent_profile.primary);

    // Check keywords
    let keywords_found =
        check_keywords_in_response(&result.predicted_text, &scenario.expected_keywords);

    // Build details
    let mut details = HashMap::new();
    details.insert(
        "fallback_mode".to_string(),
        serde_json::to_value(&fallback_mode).unwrap_or_default(),
    );
    details.insert(
        "used_retrieval".to_string(),
        serde_json::to_value(result.used_retrieval).unwrap_or_default(),
    );
    details.insert(
        "confidence".to_string(),
        serde_json::to_value(result.confidence).unwrap_or_default(),
    );
    details.insert(
        "evidence_sources".to_string(),
        serde_json::to_value(&result.trace.evidence_sources).unwrap_or_default(),
    );
    details.insert(
        "predicted_text".to_string(),
        serde_json::to_value(&result.predicted_text).unwrap_or_default(),
    );

    // Determine pass/fail
    let mut issues = Vec::new();

    // Check retrieval decision
    if retrieval_triggered != scenario.expected_retrieval_triggered {
        issues.push(format!(
            "Retrieval mismatch: got {}, expected {}",
            retrieval_triggered, scenario.expected_retrieval_triggered
        ));
    }

    // Check intent (with flexibility)
    if scenario.expected_intent != "Unknown" {
        let expected = parse_intent(&scenario.expected_intent);
        let detected = result.trace.intent_profile.primary;
        if expected != detected {
            // Allow compatible intents
            let compatible = match expected {
                IntentKind::Question => [
                    IntentKind::Question,
                    IntentKind::Verify,
                    IntentKind::Explain,
                ]
                .contains(&detected),
                IntentKind::Verify => {
                    [IntentKind::Verify, IntentKind::Question].contains(&detected)
                }
                IntentKind::Compare => {
                    [IntentKind::Compare, IntentKind::Question].contains(&detected)
                }
                IntentKind::Explain => {
                    [IntentKind::Explain, IntentKind::Question].contains(&detected)
                }
                _ => expected == detected,
            };
            if !compatible {
                issues.push(format!(
                    "Intent mismatch: got {:?}, expected {:?}",
                    detected, expected
                ));
            }
        }
    }

    // Check fallback mode
    if scenario.expected_fallback_mode != "None"
        && !scenario.expected_fallback_mode.is_empty()
        && fallback_mode != scenario.expected_fallback_mode
    {
        issues.push(format!(
            "Fallback mismatch: got {}, expected {}",
            fallback_mode, scenario.expected_fallback_mode
        ));
    }

    // Check keywords for non-empty expected lists
    if !scenario.expected_keywords.is_empty() && keywords_found.is_empty() && retrieval_triggered {
        issues.push("No expected keywords found in response".to_string());
    }

    let passed = issues.is_empty();
    let message = if passed {
        "PASSED".to_string()
    } else {
        issues.join("; ")
    };

    TestResult {
        scenario_id: scenario.id.clone(),
        passed,
        retrieval_triggered,
        expected_retrieval: scenario.expected_retrieval_triggered,
        intent_detected,
        expected_intent: scenario.expected_intent.clone(),
        fallback_mode,
        expected_fallback: scenario.expected_fallback_mode.clone(),
        keywords_found,
        expected_keywords: scenario.expected_keywords.clone(),
        message,
        details,
    }
}

async fn run_harness(
    scenarios_path: &str,
    output_path: &str,
    entropy_threshold: f32,
) -> HarnessReport {
    println!("Zero-Shot Test Harness for SPSE Engine");
    println!("======================================");
    println!();

    // Load scenarios
    let scenario_file = match load_scenarios(scenarios_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error loading scenarios: {}", e);
            std::process::exit(1);
        }
    };

    let scenarios = scenario_file.scenarios;
    let total_tests = scenarios.len() as u32;

    println!("Loaded {} test scenarios", total_tests);
    println!("Entropy threshold override: {}", entropy_threshold);
    println!();

    // Create engine with zero-shot config
    let db_path = temp_db_path("zero_shot_harness");
    let mut config = EngineConfig::default();
    config.retrieval.entropy_threshold = entropy_threshold;
    config.retrieval.freshness_threshold = 0.5; // Lower for zero-shot
    config.retrieval.decision_threshold = 0.8; // Lower for zero-shot

    let engine = Engine::new_with_config_and_db_path(config, &db_path);

    // Initialize report
    let start_instant = Instant::now();
    let start_time: DateTime<Utc> = Utc::now();

    let mut results = Vec::new();
    let mut category_stats: HashMap<String, CategoryStats> = HashMap::new();

    // Initialize category stats
    for scenario in &scenarios {
        let category = &scenario.category;
        category_stats
            .entry(category.clone())
            .or_insert(CategoryStats {
                total: 0,
                passed: 0,
                failed: 0,
            });
    }

    println!("Running tests...");
    println!();

    // Run each scenario
    for (i, scenario) in scenarios.iter().enumerate() {
        let scenario_id = &scenario.id;
        let category = &scenario.category;

        print!(
            "[{}/{}] {} ({})... ",
            i + 1,
            total_tests,
            scenario_id,
            category
        );

        let result = run_scenario(&engine, scenario).await;
        results.push(result.clone());

        // Update category stats
        if let Some(stats) = category_stats.get_mut(category) {
            stats.total += 1;
            if result.passed {
                stats.passed += 1;
            } else {
                stats.failed += 1;
            }
        }

        if result.passed {
            println!("✓ PASSED");
        } else {
            println!("✗ FAILED: {}", result.message);
        }
    }

    let end_time: DateTime<Utc> = Utc::now();
    let duration_seconds = start_instant.elapsed().as_secs_f64();

    let passed = results.iter().filter(|r| r.passed).count() as u32;
    let failed = results.iter().filter(|r| !r.passed).count() as u32;
    let pass_rate = if total_tests > 0 {
        (passed as f64 / total_tests as f64) * 100.0
    } else {
        0.0
    };

    // Build config map
    let mut config_map = HashMap::new();
    config_map.insert(
        "scenarios_path".to_string(),
        serde_json::to_value(scenarios_path).unwrap_or_default(),
    );
    config_map.insert(
        "entropy_threshold".to_string(),
        serde_json::to_value(entropy_threshold).unwrap_or_default(),
    );
    config_map.insert(
        "freshness_threshold".to_string(),
        serde_json::to_value(0.5).unwrap_or_default(),
    );
    config_map.insert(
        "decision_threshold".to_string(),
        serde_json::to_value(0.8).unwrap_or_default(),
    );

    let report = HarnessReport {
        total_tests,
        passed,
        failed,
        skipped: 0,
        pass_rate,
        results,
        category_stats,
        start_time: start_time.to_rfc3339(),
        end_time: end_time.to_rfc3339(),
        duration_seconds,
        config: config_map,
    };

    // Print summary
    println!();
    println!("==============================================");
    println!("ZERO-SHOT TEST HARNESS SUMMARY");
    println!("==============================================");
    println!("Total Tests:  {}", report.total_tests);
    println!("Passed:       {}", report.passed);
    println!("Failed:       {}", report.failed);
    println!("Pass Rate:    {:.1}%", report.pass_rate);
    println!("Duration:     {:.1}s", report.duration_seconds);
    println!();

    // Category breakdown
    println!("Category Breakdown:");
    let mut categories: Vec<_> = report.category_stats.iter().collect();
    categories.sort_by_key(|(k, _)| k.as_str());
    for (category, stats) in categories {
        let cat_rate = if stats.total > 0 {
            (stats.passed as f64 / stats.total as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:20}: {:3}/{:3} ({:5.1}%)",
            category, stats.passed, stats.total, cat_rate
        );
    }
    println!();

    // Failed tests
    if report.failed > 0 {
        println!("Failed Tests:");
        for result in &report.results {
            if !result.passed {
                println!("  {}: {}", result.scenario_id, result.message);
            }
        }
        println!();
    }

    // Pass rate assessment
    if report.pass_rate >= 90.0 {
        println!("✓ PASS RATE >= 90% - Zero-shot behavior validated!");
    } else {
        println!("✗ PASS RATE < 90% - Logic adjustments needed");
        println!("  Review failed tests and adjust retrieval gating logic");
    }

    // Save report
    if let Some(parent) = PathBuf::from(output_path).parent() {
        let _ = fs::create_dir_all(parent);
    }

    match fs::write(
        output_path,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    ) {
        Ok(_) => println!("\nReport saved to: {}", output_path),
        Err(e) => eprintln!("\nError saving report: {}", e),
    }

    report
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut scenarios_path = "test_data/zero_shot_scenarios.json".to_string();
    let mut output_path = "benchmarks/zero_shot_report.json".to_string();
    let mut entropy_threshold = 0.5;
    let mut show_help = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                show_help = true;
            }
            "--scenarios" => {
                if i + 1 < args.len() {
                    scenarios_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--output" => {
                if i + 1 < args.len() {
                    output_path = args[i + 1].clone();
                    i += 1;
                }
            }
            "--entropy" => {
                if i + 1 < args.len() {
                    entropy_threshold = args[i + 1].parse().unwrap_or(0.5);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    if show_help {
        println!("Zero-Shot Test Harness for SPSE Engine");
        println!();
        println!("Usage: cargo run --bin zero_shot_harness -- [options]");
        println!();
        println!("Options:");
        println!("  --scenarios <PATH>   Path to test scenarios JSON");
        println!("  --output <PATH>      Output report path");
        println!("  --entropy <VALUE>    Override entropy threshold (default: 0.5)");
        println!("  --help               Show this help message");
        println!();
        println!("This harness validates retrieval gating behavior in untrained scenarios.");
        println!("Tests Layers 9-13: Intent Detection, Query Sanitization, Retrieval, Evidence Merge, Safety.");
        return;
    }

    // Resolve paths - CARGO_MANIFEST_DIR is the project root where Cargo.toml is
    let project_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let scenarios_full = if PathBuf::from(&scenarios_path).is_absolute() {
        scenarios_path
    } else {
        project_root.join(&scenarios_path).display().to_string()
    };

    let output_full = if PathBuf::from(&output_path).is_absolute() {
        output_path
    } else {
        project_root.join(&output_path).display().to_string()
    };

    // Run harness
    let report = run_harness(&scenarios_full, &output_full, entropy_threshold).await;

    // Exit with appropriate code
    if report.pass_rate >= 90.0 {
        std::process::exit(0);
    } else {
        std::process::exit(1);
    }
}
