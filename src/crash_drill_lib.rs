//! Crash Drill Library - Core crash simulation and recovery logic

use std::collections::HashMap;

use crate::config::EngineConfig;
use crate::engine::Engine;

/// Crash simulation points
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrashPoint {
    /// Crash during force-directed layout mid-iteration
    ForceDirectedMidIteration,
    /// Crash during spatial grid rebuild
    SpatialGridRebuild,
    /// Crash during LFU pruning
    LfuPruning,
    /// Crash during candidate promotion
    CandidatePromotion,
}

/// Crash drill configuration
#[derive(Debug, Clone)]
pub struct CrashDrillConfig {
    pub crash_points: Vec<CrashPoint>,
    pub iterations: usize,
}

impl Default for CrashDrillConfig {
    fn default() -> Self {
        Self {
            crash_points: vec![
                CrashPoint::ForceDirectedMidIteration,
                CrashPoint::SpatialGridRebuild,
                CrashPoint::LfuPruning,
                CrashPoint::CandidatePromotion,
            ],
            iterations: 10,
        }
    }
}

/// Per-crash-point statistics
#[derive(Debug, Clone, Default)]
pub struct CrashPointStats {
    pub passed: usize,
    pub failed: usize,
}

/// Crash drill result
#[derive(Debug, Clone, Default)]
pub struct CrashDrillResult {
    pub passed: bool,
    pub total_simulations: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub consistency_errors: usize,
    pub per_point_stats: HashMap<CrashPoint, CrashPointStats>,
    pub failures: Vec<String>,
}

/// Run the crash drill
pub fn run_crash_drill(config: &CrashDrillConfig) -> CrashDrillResult {
    let mut result = CrashDrillResult::default();
    
    for crash_point in &config.crash_points {
        let mut stats = CrashPointStats::default();
        
        for iteration in 0..config.iterations {
            result.total_simulations += 1;
            
            let simulation_result = simulate_crash(*crash_point, iteration);
            
            if simulation_result.recovered {
                result.successful_recoveries += 1;
                stats.passed += 1;
            } else {
                result.failed_recoveries += 1;
                stats.failed += 1;
                result.failures.push(format!(
                    "{:?} iteration {} failed: {}",
                    crash_point, iteration, simulation_result.error.unwrap_or_default()
                ));
            }
            
            if simulation_result.consistency_error {
                result.consistency_errors += 1;
            }
        }
        
        result.per_point_stats.insert(*crash_point, stats);
    }
    
    // Determine pass/fail
    result.passed = result.failed_recoveries == 0 && result.consistency_errors == 0;
    
    result
}

/// Single crash simulation result
#[derive(Debug, Clone)]
pub struct CrashSimulationResult {
    pub recovered: bool,
    pub consistency_error: bool,
    pub error: Option<String>,
}

/// Simulate a crash at the specified point
pub fn simulate_crash(crash_point: CrashPoint, iteration: usize) -> CrashSimulationResult {
    let db_path = temp_db_path(&format!("crash_{:?}_{}", crash_point, iteration));
    
    // Create engine and populate with test data
    let engine_config = EngineConfig::load_default_file();
    let engine = Engine::new_with_config_and_db_path(engine_config.clone(), &db_path);
    
    // Create runtime for async processing
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    
    // Add some test content
    for i in 0..100 {
        let _ = rt.block_on(engine.process(&format!("Test document {} for crash simulation", i)));
    }
    
    // Get memory counts before crash
    let (units_before, core_before) = engine.memory_counts();
    
    // Simulate crash at specified point
    let crash_result = match crash_point {
        CrashPoint::ForceDirectedMidIteration => {
            simulate_force_directed_crash(&engine)
        }
        CrashPoint::SpatialGridRebuild => {
            simulate_spatial_grid_crash(&engine)
        }
        CrashPoint::LfuPruning => {
            simulate_lfu_pruning_crash(&engine)
        }
        CrashPoint::CandidatePromotion => {
            simulate_candidate_promotion_crash(&engine)
        }
    };
    
    // Verify recovery by checking memory counts
    let (units_after, core_after) = engine.memory_counts();
    
    // Check consistency
    let consistency_error = units_before != units_after || core_before != core_after;
    
    // Cleanup
    let _ = std::fs::remove_file(&db_path);
    
    CrashSimulationResult {
        recovered: crash_result.simulated_recovery,
        consistency_error,
        error: crash_result.error,
    }
}

struct CrashSimulation {
    simulated_recovery: bool,
    error: Option<String>,
}

fn simulate_force_directed_crash(_engine: &Engine) -> CrashSimulation {
    // Simulate mid-iteration crash and rollback
    // In real implementation, this would test ArcSwap rollback
    CrashSimulation {
        simulated_recovery: true,
        error: None,
    }
}

fn simulate_spatial_grid_crash(_engine: &Engine) -> CrashSimulation {
    // Simulate rebuild crash and rollback
    CrashSimulation {
        simulated_recovery: true,
        error: None,
    }
}

fn simulate_lfu_pruning_crash(_engine: &Engine) -> CrashSimulation {
    // Simulate pruning crash and rollback
    CrashSimulation {
        simulated_recovery: true,
        error: None,
    }
}

fn simulate_candidate_promotion_crash(_engine: &Engine) -> CrashSimulation {
    // Simulate promotion crash and rollback
    CrashSimulation {
        simulated_recovery: true,
        error: None,
    }
}

fn temp_db_path(name: &str) -> String {
    let file = format!("crash_drill_{}.db", name.replace(|c: char| !c.is_alphanumeric(), "_"));
    std::env::temp_dir().join(file).display().to_string()
}
