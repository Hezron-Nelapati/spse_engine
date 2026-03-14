//! Pollution Development Module
//!
//! This module tackles content pollution in the SPSE Engine's byte-window logic.
//! It generates test content with controlled pollution patterns, sweeps config parameters,
//! and finds optimal settings that minimize/eliminate pollution from entering the core DB.
//!
//! Usage:
//!   cargo run --bin pollution_dev -- [options]
//!
//! Options:
//!   --sweep        Run config parameter sweep to find optimal settings
//!   --dry-run      Generate sample DB with current config (small corpus)
//!   --large        Use large corpus (70MB+) for testing
//!   --report       Analyze pollution in existing DB
//!   --fix-logic    Suggest logic fixes based on pollution patterns

use std::path::PathBuf;
use uuid::Uuid;

mod pollution_dev_lib;

use pollution_dev_lib::{
    config_sweep, generate_test_content, load_large_corpus, pollution_report, PollutionConfig, PollutionResult,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let mode = if args.contains(&"--sweep-large".to_string()) {
        "sweep-large"
    } else if args.contains(&"--sweep".to_string()) {
        "sweep"
    } else if args.contains(&"--report".to_string()) {
        "report"
    } else if args.contains(&"--fix-logic".to_string()) {
        "fix-logic"
    } else if args.contains(&"--large".to_string()) {
        "large"
    } else {
        "dry-run"
    };

    println!("=== SPSE Engine Pollution Dev Module ===");
    println!("Mode: {}", mode);
    println!();

    let result = match mode {
        "sweep" => run_config_sweep(),
        "sweep-large" => run_config_sweep_large(),
        "report" => run_pollution_report(),
        "fix-logic" => run_logic_analysis(),
        "large" => run_large_corpus_test(),
        _ => run_dry_run(),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_dry_run() -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating test content with controlled pollution patterns...");
    
    let test_content = generate_test_content();
    println!("Generated {} test documents", test_content.documents.len());
    
    // Create temp DB
    let db_path = std::env::temp_dir().join(format!("pollution_dev_{}.db", Uuid::new_v4()));
    let config = PollutionConfig::default();
    
    println!("Running dry-run with current config...");
    println!("DB path: {}", db_path.display());
    
    let result = pollution_dev_lib::run_pollution_test(&test_content, &config, &db_path)?;
    
    print_pollution_summary(&result);
    
    // Cleanup
    let _ = std::fs::remove_file(&db_path);
    
    Ok(())
}

fn run_large_corpus_test() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading large corpus (70MB+)...");
    
    let test_content = load_large_corpus()?;
    println!();
    
    // Create temp DB
    let db_path = std::env::temp_dir().join(format!("pollution_dev_large_{}.db", Uuid::new_v4()));
    let config = PollutionConfig::default();
    
    println!("Running pollution test on large corpus...");
    println!("DB path: {}", db_path.display());
    println!("This may take several minutes...");
    println!();
    
    let start = std::time::Instant::now();
    let result = pollution_dev_lib::run_pollution_test(&test_content, &config, &db_path)?;
    let elapsed = start.elapsed();
    
    println!("\nProcessing time: {:.2?}", elapsed);
    print_pollution_summary(&result);
    
    // Cleanup
    let _ = std::fs::remove_file(&db_path);
    
    Ok(())
}

fn run_config_sweep() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running config parameter sweep...");
    println!("This will test multiple config combinations to find optimal anti-pollution settings.\n");
    
    let test_content = generate_test_content();
    let sweep_results = config_sweep(&test_content)?;
    
    println!("\n=== SWEEP RESULTS ===\n");
    
    // Sort by pollution score (lower is better)
    let mut sorted: Vec<_> = sweep_results.iter().collect();
    sorted.sort_by(|a, b| a.1.pollution_score.partial_cmp(&b.1.pollution_score).unwrap());
    
    println!("Top 5 configurations (lowest pollution):\n");
    for (i, (config_name, result)) in sorted.iter().take(5).enumerate() {
        println!("{}. {}", i + 1, config_name);
        println!("   Pollution score: {:.4}", result.pollution_score);
        println!("   Total units: {}", result.total_units);
        println!("   Polluted units: {}", result.polluted_units);
        println!("   Clean units: {}", result.clean_units);
        println!("   Config: {:?}", result.config_summary);
        println!();
    }
    
    if let Some((best_name, best_result)) = sorted.first() {
        println!("\n=== RECOMMENDED CONFIG ===");
        println!("Best performing config: {}", best_name);
        println!("Pollution reduction: {:.1}%", 
            (1.0 - best_result.pollution_score) * 100.0);
        
        println!("\nSuggested config.yaml additions:");
        println!("{}", best_result.config_yaml_patch);
    }
    
    Ok(())
}

fn run_config_sweep_large() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading large corpus for config sweep...");
    let test_content = load_large_corpus()?;
    println!();
    
    println!("Running config parameter sweep on large corpus...");
    println!("This will test multiple config combinations (this may take 30+ minutes).\n");
    
    let sweep_results = config_sweep(&test_content)?;
    
    println!("\n=== SWEEP RESULTS (Large Corpus) ===\n");
    
    // Sort by pollution score (lower is better)
    let mut sorted: Vec<_> = sweep_results.iter().collect();
    sorted.sort_by(|a, b| a.1.pollution_score.partial_cmp(&b.1.pollution_score).unwrap());
    
    println!("Top 5 configurations (lowest pollution):\n");
    for (i, (config_name, result)) in sorted.iter().take(5).enumerate() {
        println!("{}. {}", i + 1, config_name);
        println!("   Pollution score: {:.4}", result.pollution_score);
        println!("   Total units: {}", result.total_units);
        println!("   Polluted units: {}", result.polluted_units);
        println!("   Clean units: {}", result.clean_units);
        println!("   Config: {:?}", result.config_summary);
        println!();
    }
    
    if let Some((best_name, best_result)) = sorted.first() {
        println!("\n=== RECOMMENDED CONFIG ===");
        println!("Best performing config: {}", best_name);
        println!("Pollution reduction: {:.1}%", 
            (1.0 - best_result.pollution_score) * 100.0);
        
        println!("\nSuggested config.yaml additions:");
        println!("{}", best_result.config_yaml_patch);
    }
    
    Ok(())
}

fn run_pollution_report() -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing pollution in existing database...\n");
    
    // Look for existing DB
    let db_path = find_existing_db()?;
    println!("Found DB at: {}", db_path.display());
    
    let report = pollution_report(&db_path)?;
    
    println!("\n=== POLLUTION REPORT ===\n");
    println!("Total units: {}", report.total_units);
    println!("Polluted units: {}", report.polluted_units);
    println!("Pollution ratio: {:.2}%", report.pollution_ratio * 100.0);
    
    println!("\nPollution by category:");
    for (category, count) in &report.pollution_by_category {
        println!("  {}: {}", category, count);
    }
    
    println!("\nTop polluted units:");
    for unit in report.top_polluted.iter().take(20) {
        println!("  '{}' (score: {:.3}, reasons: {})", 
            unit.content, unit.score, unit.reasons.join(", "));
    }
    
    println!("\nRecommendations:");
    for rec in &report.recommendations {
        println!("  - {}", rec);
    }
    
    Ok(())
}

fn run_logic_analysis() -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing pollution patterns to suggest logic fixes...\n");
    
    let test_content = generate_test_content();
    let analysis = pollution_dev_lib::analyze_pollution_patterns(&test_content)?;
    
    println!("=== POLLUTION PATTERN ANALYSIS ===\n");
    
    println!("Detected pollution patterns:\n");
    for pattern in &analysis.patterns {
        println!("Pattern: {}", pattern.name);
        println!("  Frequency: {} occurrences", pattern.frequency);
        println!("  Example: '{}'", pattern.example);
        println!("  Root cause: {}", pattern.root_cause);
        println!("  Suggested fix: {}", pattern.suggested_fix);
        println!();
    }
    
    println!("=== SUGGESTED LOGIC FIXES ===\n");
    for fix in &analysis.suggested_fixes {
        println!("Fix: {}", fix.description);
        println!("  Location: {}", fix.location);
        println!("  Priority: {}", fix.priority);
        println!("  Implementation:\n{}", fix.implementation_hint);
        println!();
    }
    
    Ok(())
}

fn print_pollution_summary(result: &PollutionResult) {
    println!("\n=== POLLUTION TEST RESULTS ===\n");
    println!("Total units discovered: {}", result.total_units);
    println!("Clean units: {}", result.clean_units);
    println!("Polluted units: {}", result.polluted_units);
    println!("Pollution score: {:.4}", result.pollution_score);
    
    println!("\nPollution by category:");
    for (category, count) in &result.pollution_by_category {
        println!("  {}: {}", category, count);
    }
    
    if !result.top_polluted.is_empty() {
        println!("\nTop polluted units:");
        for unit in result.top_polluted.iter().take(10) {
            println!("  '{}' (score: {:.3}, reasons: {})", 
                unit.content, unit.score, unit.reasons.join(", "));
        }
    }
    
    println!("\nConfig used:");
    println!("  min_frequency_threshold: {}", result.config_summary.min_frequency);
    println!("  rolling_hash_window_sizes: {:?}", result.config_summary.window_sizes);
    println!("  min_fragment_length: {}", result.config_summary.min_fragment_length);
    println!("  punctuation_ratio_limit: {}", result.config_summary.punctuation_ratio);
}

fn find_existing_db() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Check common locations
    let candidates = vec![
        PathBuf::from("spse_engine.db"),
        PathBuf::from("data/spse_engine.db"),
        std::env::current_dir()?.join("spse_engine.db"),
    ];
    
    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }
    
    Err("No existing database found. Run with --dry-run first.".into())
}
