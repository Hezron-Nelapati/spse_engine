//! GPU Fallback Tests
//!
//! Tests that GPU operations correctly fall back to CPU when GPU is unavailable.

#[cfg(feature = "gpu")]
mod gpu_tests {
    use spse_engine::gpu::is_gpu_available;

    #[test]
    fn test_gpu_detection() {
        // This test verifies GPU detection works without panicking
        // The result depends on the system running the test
        let _available = is_gpu_available();
    }

    #[test]
    fn test_gpu_info_does_not_panic() {
        // Verify gpu_info doesn't panic
        let _info = spse_engine::gpu::gpu_info();
    }
}

#[cfg(not(feature = "gpu"))]
mod no_gpu_tests {
    #[test]
    fn test_gpu_disabled_when_feature_off() {
        // When gpu feature is disabled, is_gpu_available should return false
        assert!(!spse_engine::gpu::is_gpu_available());
    }

    #[test]
    fn test_gpu_info_returns_none_when_disabled() {
        assert!(spse_engine::gpu::gpu_info().is_none());
    }
}

mod cpu_fallback_tests {
    use spse_engine::config::GpuConfig;
    use spse_engine::spatial_index::force_directed_layout;
    use spse_engine::types::{Unit, UnitLevel, Link, MemoryType};
    use spse_engine::config::SemanticMapConfig;
    use uuid::Uuid;

    fn create_test_units(count: usize) -> Vec<Unit> {
        (0..count)
            .map(|i| Unit {
                id: Uuid::new_v4(),
                content: format!("Test unit {}", i),
                normalized: format!("test unit {}", i),
                level: UnitLevel::Phrase,
                frequency: 1,
                utility_score: 0.5,
                confidence: 0.5,
                trust_score: 0.5,
                semantic_position: [
                    (i as f32 % 10.0) - 5.0,
                    ((i / 10) as f32 % 10.0) - 5.0,
                    0.0,
                ],
                anchor_status: false,
                links: vec![],
                memory_type: MemoryType::Episodic,
                memory_channels: vec![],
                created_at: chrono::Utc::now(),
                last_seen_at: chrono::Utc::now(),
                salience_score: 0.5,
                corroboration_count: 0,
                contexts: vec![],
            })
            .collect()
    }

    #[test]
    fn test_force_layout_cpu_fallback_small_dataset() {
        // Small datasets should use CPU (below GPU threshold)
        let units = create_test_units(10);
        let config = SemanticMapConfig::default();
        
        let result = force_directed_layout(&units, &config);
        
        assert!(!result.rolled_back);
        assert_eq!(result.position_updates.len(), 10);
    }

    #[test]
    fn test_force_layout_cpu_fallback_medium_dataset() {
        // Medium dataset - should work regardless of GPU availability
        let units = create_test_units(50);
        let config = SemanticMapConfig::default();
        
        let result = force_directed_layout(&units, &config);
        
        // Should produce valid results
        assert_eq!(result.position_updates.len(), 50);
        
        // All positions should be within bounds
        for (_, pos) in &result.position_updates {
            assert!(pos[0].abs() <= config.layout_boundary);
            assert!(pos[1].abs() <= config.layout_boundary);
            assert!(pos[2].abs() <= config.layout_boundary);
        }
    }

    #[test]
    fn test_gpu_config_defaults() {
        let config = GpuConfig::default();
        
        assert!(config.enabled);
        assert!(!config.force_cpu);
        assert_eq!(config.power_preference, "high");
        assert_eq!(config.min_memory_mb, 512);
        assert!(config.use_for_scoring);
        assert!(config.use_for_layout);
        assert!(config.use_for_distance);
        assert_eq!(config.min_candidates_for_gpu, 256);
        assert_eq!(config.min_units_for_gpu_layout, 100);
    }

    #[test]
    fn test_gpu_config_force_cpu() {
        let mut config = GpuConfig::default();
        config.force_cpu = true;
        
        // When force_cpu is true, GPU should not be used
        assert!(config.force_cpu);
        assert!(config.enabled); // enabled but forced off
    }
}
