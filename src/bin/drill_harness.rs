//! Drill Harness for Layer-Specific Testing
//!
//! Targeted corpus subsets to trigger specific failure modes across all 21 layers.
//! Migrates pollution check logic into unified drill framework.
//!
//! Usage:
//!   cargo run --bin drill_harness -- [options]
//!
//! Options:
//!   --mode <MODE>      Run specific drill mode (see DrillMode enum)
//!   --category <CAT>   Run drills in specific category (happy, edge, failure, stress)
//!   --all              Run all drills
//!   --list             List available drill modes

use std::collections::HashMap;
use std::time::Instant;

use spse_engine::drill_lib::{
    DrillMode, DrillCategory, DrillResult, DrillReport,
    run_drill, generate_drill_corpus,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let mode = if args.contains(&"--all".to_string()) {
        "all"
    } else if args.contains(&"--list".to_string()) {
        "list"
    } else if let Some(pos) = args.iter().position(|arg| arg == "--mode") {
        args.get(pos + 1).map(|s| s.as_str()).unwrap_or("help")
    } else if let Some(pos) = args.iter().position(|arg| arg == "--category") {
        args.get(pos + 1).map(|s| s.as_str()).unwrap_or("help")
    } else {
        "help"
    };

    match mode {
        "list" => list_drills(),
        "all" => run_all_drills(),
        "help" => print_help(),
        _ => run_specific_drill(mode),
    }
}

fn print_help() {
    println!("=== SPSE Engine Drill Harness ===");
    println!();
    println!("Usage: cargo run --bin drill_harness -- [options]");
    println!();
    println!("Options:");
    println!("  --mode <MODE>      Run specific drill mode");
    println!("  --category <CAT>   Run drills in category (happy, edge, failure, stress)");
    println!("  --all              Run all drills");
    println!("  --list             List available drill modes");
    println!();
    println!("Drill Categories:");
    println!("  happy    - Expected normal behavior");
    println!("  edge     - Boundary conditions, unusual inputs");
    println!("  failure  - Known failure scenarios, error handling");
    println!("  stress   - High load, resource exhaustion");
    println!();
    println!("Examples:");
    println!("  cargo run --bin drill_harness -- --mode IntentClassify");
    println!("  cargo run --bin drill_harness -- --category edge");
    println!("  cargo run --bin drill_harness -- --all");
}

fn list_drills() {
    println!("=== Available Drill Modes ===");
    println!();
    
    println!("=== Input Layer Drills ===");
    for mode in &[DrillMode::Garbage, DrillMode::UnitActivation] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Spatial Layer Drills ===");
    for mode in &[DrillMode::Collisions, DrillMode::RoutingEscape] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Context Layer Drills ===");
    for mode in &[DrillMode::AnchorLoss, DrillMode::ContextMatrix] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Intent-Driven Input Drills ===");
    for mode in &[DrillMode::IntentClassify, DrillMode::IntentBlend, 
                   DrillMode::RetrievalGate, DrillMode::IntentMemoryGate] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Safety Layer Drills ===");
    for mode in &[DrillMode::Poison, DrillMode::TrustHeuristics] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Memory Layer Drills ===");
    for mode in &[DrillMode::Maintenance, DrillMode::Promotion, DrillMode::ChannelIsolation] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Intent-Driven Output Drills ===");
    for mode in &[DrillMode::OutputDecode, DrillMode::CreativeDrift, DrillMode::IntentShaping] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Pollution Drills ===");
    for mode in &[DrillMode::PollutionCeiling, DrillMode::PollutionPurge] {
        println!("  {:?}", mode);
    }
    
    println!();
    println!("=== Phase 5: Retrieval & Optimization Drills ===");
    for mode in &[DrillMode::MultiEngineConsensus, DrillMode::MultiEngineDisagreement,
                   DrillMode::MultiEngineUnavailable, DrillMode::StructuredParsing,
                   DrillMode::ConfigSweepPareto, DrillMode::ConfigSweepNoOptimal] {
        println!("  {:?}", mode);
    }
}

fn run_all_drills() {
    println!("=== Running All Drills ===");
    println!();
    
    let all_modes = vec![
        // Input Layer
        DrillMode::Garbage,
        DrillMode::UnitActivation,
        // Spatial Layer
        DrillMode::Collisions,
        DrillMode::RoutingEscape,
        // Context Layer
        DrillMode::AnchorLoss,
        DrillMode::ContextMatrix,
        // Intent-Driven Input
        DrillMode::IntentClassify,
        DrillMode::IntentBlend,
        DrillMode::RetrievalGate,
        DrillMode::IntentMemoryGate,
        // Safety Layer
        DrillMode::Poison,
        DrillMode::TrustHeuristics,
        // Memory Layer
        DrillMode::Maintenance,
        DrillMode::Promotion,
        DrillMode::ChannelIsolation,
        // Intent-Driven Output
        DrillMode::OutputDecode,
        DrillMode::CreativeDrift,
        DrillMode::IntentShaping,
        // Pollution
        DrillMode::PollutionCeiling,
        DrillMode::PollutionPurge,
        // Phase 5: Retrieval & Optimization
        DrillMode::MultiEngineConsensus,
        DrillMode::MultiEngineDisagreement,
        DrillMode::MultiEngineUnavailable,
        DrillMode::StructuredParsing,
        DrillMode::ConfigSweepPareto,
        DrillMode::ConfigSweepNoOptimal,
    ];
    
    let mut report = DrillReport::default();
    let start = Instant::now();
    
    for mode in all_modes {
        println!("Running {:?}...", mode);
        let result = run_drill(&mode, DrillCategory::HappyPath);
        report.add_result(result);
    }
    
    report.total_duration_ms = start.elapsed().as_millis() as u64;
    
    println!();
    println!("=== Drill Summary ===");
    println!("Total drills: {}", report.total_drills());
    println!("Passed: {}", report.passed());
    println!("Failed: {}", report.failed());
    println!("Duration: {}ms", report.total_duration_ms);
    
    if report.failed() > 0 {
        println!();
        println!("=== Failed Drills ===");
        for result in &report.results {
            if !result.passed {
                println!("  {:?}: {}", result.mode, result.message);
            }
        }
        std::process::exit(1);
    }
}

fn run_specific_drill(mode_str: &str) {
    let mode = parse_drill_mode(mode_str);
    
    println!("=== Running {:?} Drill ===", mode);
    println!();
    
    // Run all categories for the specific mode
    let categories = [
        DrillCategory::HappyPath,
        DrillCategory::EdgeCase,
        DrillCategory::FailureMode,
        DrillCategory::Stress,
    ];
    
    let mut passed = 0;
    let mut failed = 0;
    
    for category in &categories {
        println!("Category: {:?}", category);
        let result = run_drill(&mode, category.clone());
        
        if result.passed {
            println!("  ✓ PASSED: {}", result.message);
            passed += 1;
        } else {
            println!("  ✗ FAILED: {}", result.message);
            failed += 1;
        }
        
        if !result.details.is_empty() {
            println!("  Details: {}", result.details);
        }
        println!();
    }
    
    println!("Summary: {} passed, {} failed", passed, failed);
    
    if failed > 0 {
        std::process::exit(1);
    }
}

fn parse_drill_mode(s: &str) -> DrillMode {
    match s {
        // Input Layer
        "Garbage" | "garbage" => DrillMode::Garbage,
        "UnitActivation" | "unit_activation" => DrillMode::UnitActivation,
        // Spatial Layer
        "Collisions" | "collisions" => DrillMode::Collisions,
        "RoutingEscape" | "routing_escape" => DrillMode::RoutingEscape,
        // Context Layer
        "AnchorLoss" | "anchor_loss" => DrillMode::AnchorLoss,
        "ContextMatrix" | "context_matrix" => DrillMode::ContextMatrix,
        // Intent-Driven Input
        "IntentClassify" | "intent_classify" => DrillMode::IntentClassify,
        "IntentBlend" | "intent_blend" => DrillMode::IntentBlend,
        "RetrievalGate" | "retrieval_gate" => DrillMode::RetrievalGate,
        "IntentMemoryGate" | "intent_memory_gate" => DrillMode::IntentMemoryGate,
        // Safety Layer
        "Poison" | "poison" => DrillMode::Poison,
        "TrustHeuristics" | "trust_heuristics" => DrillMode::TrustHeuristics,
        // Memory Layer
        "Maintenance" | "maintenance" => DrillMode::Maintenance,
        "Promotion" | "promotion" => DrillMode::Promotion,
        "ChannelIsolation" | "channel_isolation" => DrillMode::ChannelIsolation,
        // Intent-Driven Output
        "OutputDecode" | "output_decode" => DrillMode::OutputDecode,
        "CreativeDrift" | "creative_drift" => DrillMode::CreativeDrift,
        "IntentShaping" | "intent_shaping" => DrillMode::IntentShaping,
        // Pollution
        "PollutionCeiling" | "pollution_ceiling" => DrillMode::PollutionCeiling,
        "PollutionPurge" | "pollution_purge" => DrillMode::PollutionPurge,
        // Phase 5: Retrieval & Optimization
        "MultiEngineConsensus" | "multi_engine_consensus" => DrillMode::MultiEngineConsensus,
        "MultiEngineDisagreement" | "multi_engine_disagreement" => DrillMode::MultiEngineDisagreement,
        "MultiEngineUnavailable" | "multi_engine_unavailable" => DrillMode::MultiEngineUnavailable,
        "StructuredParsing" | "structured_parsing" => DrillMode::StructuredParsing,
        "ConfigSweepPareto" | "config_sweep_pareto" => DrillMode::ConfigSweepPareto,
        "ConfigSweepNoOptimal" | "config_sweep_no_optimal" => DrillMode::ConfigSweepNoOptimal,
        _ => {
            eprintln!("Unknown drill mode: {}", s);
            eprintln!("Use --list to see available modes");
            std::process::exit(1);
        }
    }
}
