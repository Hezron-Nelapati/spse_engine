//! Stress Drill - End-to-end 7MB heterogeneous ingestion testing
//!
//! Validates Layer 12 normalization, latency monitoring, pollution ceilings,
//! and snapshot consistency under mixed-source load.
//!
//! Usage:
//!   cargo run --bin stress_drill -- [options]
//!
//! Options:
//!   --corpus-size <MB>     Corpus size in MB (default: 7)
//!   --query-interval <MS>  Query interval in milliseconds (default: 100)
//!   --maintenance <SEC>    Maintenance interval in seconds (default: 60)
//!   --max-latency <MS>     Max latency spike threshold (default: 500)

use spse_engine::stress_drill_lib::{
    StressDrillConfig,
    run_stress_drill,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    let config = parse_args(&args);
    
    println!("=== SPSE Engine Stress Drill ===");
    println!("Corpus size: {} MB", config.corpus_size_mb);
    println!("Query interval: {} ms", config.query_interval_ms);
    println!("Maintenance interval: {} sec", config.maintenance_interval_sec);
    println!("Max latency spike: {} ms", config.max_latency_spike_ms);
    println!();
    
    let result = run_stress_drill(&config);
    
    println!();
    println!("=== Stress Drill Results ===");
    println!("Total documents: {}", result.total_documents);
    println!("Total queries: {}", result.total_queries);
    println!("Total duration: {:.2}s", result.duration_sec);
    println!("Throughput: {:.2} docs/sec", result.docs_per_sec);
    println!();
    
    println!("=== Latency Report ===");
    println!("Avg latency: {:.2}ms", result.latency.avg_ms);
    println!("Max latency: {:.2}ms", result.latency.max_ms);
    println!("P99 latency: {:.2}ms", result.latency.p99_ms);
    println!("Spikes detected: {}", result.latency.spike_count);
    println!();
    
    println!("=== Pollution Report ===");
    println!("Pollution ratio: {:.4}%", result.pollution_ratio * 100.0);
    println!("Pollution ceiling: {:.2}%", config.pollution_ceiling_percent);
    println!("Ceiling maintained: {}", result.pollution_ratio < config.pollution_ceiling_percent / 100.0);
    println!();
    
    println!("=== Snapshot Consistency ===");
    println!("Snapshots created: {}", result.snapshots_created);
    println!("Snapshots verified: {}", result.snapshots_verified);
    println!("Consistency errors: {}", result.consistency_errors);
    
    if result.passed {
        println!();
        println!("✓ STRESS DRILL PASSED");
        std::process::exit(0);
    } else {
        println!();
        println!("✗ STRESS DRILL FAILED");
        for failure in &result.failures {
            println!("  - {}", failure);
        }
        std::process::exit(1);
    }
}

fn parse_args(args: &[String]) -> StressDrillConfig {
    let mut config = StressDrillConfig::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus-size" => {
                if i + 1 < args.len() {
                    config.corpus_size_mb = args[i + 1].parse().unwrap_or(7);
                    i += 1;
                }
            }
            "--query-interval" => {
                if i + 1 < args.len() {
                    config.query_interval_ms = args[i + 1].parse().unwrap_or(100);
                    i += 1;
                }
            }
            "--maintenance" => {
                if i + 1 < args.len() {
                    config.maintenance_interval_sec = args[i + 1].parse().unwrap_or(60);
                    i += 1;
                }
            }
            "--max-latency" => {
                if i + 1 < args.len() {
                    config.max_latency_spike_ms = args[i + 1].parse().unwrap_or(500);
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }
    
    config
}

fn print_help() {
    println!("Stress Drill - End-to-end heterogeneous ingestion testing");
    println!();
    println!("Usage: cargo run --bin stress_drill -- [options]");
    println!();
    println!("Options:");
    println!("  --corpus-size <MB>     Corpus size in MB (default: 7)");
    println!("  --query-interval <MS>  Query interval in milliseconds (default: 100)");
    println!("  --maintenance <SEC>    Maintenance interval in seconds (default: 60)");
    println!("  --max-latency <MS>      Max latency spike threshold (default: 500)");
    println!("  --help, -h              Show this help message");
}
