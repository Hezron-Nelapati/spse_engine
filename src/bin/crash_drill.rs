//! Crash Drill - Snapshot atomicity and recovery testing
//!
//! Simulates crashes during Layer 5 map updates or Layer 21 compaction.
//! Verifies ArcSwap rollback mechanisms and snapshot consistency.
//!
//! Usage:
//!   cargo run --bin crash_drill -- [options]
//!
//! Options:
//!   --crash-point <POINT>  Crash simulation point (default: all)
//!   --iterations <N>      Number of iterations per crash point (default: 10)

use spse_engine::crash_drill_lib::{run_crash_drill, CrashDrillConfig, CrashPoint};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let config = parse_args(&args);

    println!("=== SPSE Engine Crash Drill ===");
    println!("Crash points: {:?}", config.crash_points);
    println!("Iterations per point: {}", config.iterations);
    println!();

    let result = run_crash_drill(&config);

    println!();
    println!("=== Crash Drill Results ===");
    println!("Total simulations: {}", result.total_simulations);
    println!("Successful recoveries: {}", result.successful_recoveries);
    println!("Failed recoveries: {}", result.failed_recoveries);
    println!("Consistency errors: {}", result.consistency_errors);
    println!();

    println!("=== Per-Point Results ===");
    for (point, stats) in &result.per_point_stats {
        println!(
            "  {:?}: {} passed, {} failed",
            point, stats.passed, stats.failed
        );
    }

    if result.passed {
        println!();
        println!("✓ CRASH DRILL PASSED");
        std::process::exit(0);
    } else {
        println!();
        println!("✗ CRASH DRILL FAILED");
        for failure in &result.failures {
            println!("  - {}", failure);
        }
        std::process::exit(1);
    }
}

fn parse_args(args: &[String]) -> CrashDrillConfig {
    let mut config = CrashDrillConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--crash-point" => {
                if i + 1 < args.len() {
                    let point = parse_crash_point(&args[i + 1]);
                    config.crash_points = vec![point];
                    i += 1;
                }
            }
            "--iterations" => {
                if i + 1 < args.len() {
                    config.iterations = args[i + 1].parse().unwrap_or(10);
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

fn parse_crash_point(s: &str) -> CrashPoint {
    match s.to_lowercase().as_str() {
        "force_directed" | "forcedirectedmiditeration" => CrashPoint::ForceDirectedMidIteration,
        "spatial_grid" | "spatialgridrebuild" => CrashPoint::SpatialGridRebuild,
        "lfu_pruning" | "lfupruning" => CrashPoint::LfuPruning,
        "candidate_promotion" | "candidatepromotion" => CrashPoint::CandidatePromotion,
        _ => CrashPoint::ForceDirectedMidIteration,
    }
}

fn print_help() {
    println!("Crash Drill - Snapshot atomicity and recovery testing");
    println!();
    println!("Usage: cargo run --bin crash_drill -- [options]");
    println!();
    println!("Options:");
    println!("  --crash-point <POINT>  Crash simulation point:");
    println!("                         - force_directed");
    println!("                         - spatial_grid");
    println!("                         - lfu_pruning");
    println!("                         - candidate_promotion");
    println!("  --iterations <N>       Number of iterations per crash point (default: 10)");
    println!("  --help, -h             Show this help message");
}
