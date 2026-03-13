use serde_json::Value;
use spse_engine::config::EngineConfig;
use spse_engine::engine::Engine;
use std::fs;
use uuid::Uuid;

fn temp_db_path(name: &str) -> String {
    let file = format!("{}_{}.db", name, Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}

fn temp_log_path(name: &str) -> String {
    let file = format!("{}_{}.jsonl", name, Uuid::new_v4());
    std::env::temp_dir()
        .join("spse_test_logs")
        .join(file)
        .display()
        .to_string()
}

fn parse_lines(path: &str) -> Vec<Value> {
    fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect()
}

#[tokio::test]
async fn config_yaml_can_drive_observer_logging() {
    let mut config = EngineConfig::load_from_file("config/config.yaml").unwrap();
    let log_path = temp_log_path("config_yaml_observer");
    config.telemetry.observation_log_path = Some(log_path.clone());
    config.telemetry.telemetry_sample_rate = 1.0;

    let engine = Engine::new_with_config_and_db_path(config, &temp_db_path("config_yaml"));
    let _ = engine.process("What is the capital of France?").await;
    let _ = engine.process("Explain semantic routing.").await;

    let observations = parse_lines(&log_path);
    assert!(observations.len() >= 2);
    let first = &observations[0];
    assert!(first["total_latency_ms"].as_u64().unwrap() > 0);
    let expected_entropy = EngineConfig::load_from_file("config/config.yaml")
        .unwrap()
        .retrieval
        .entropy_threshold as f64;
    let observed_entropy = first["config_values_used"]["entropy_threshold"]
        .as_f64()
        .unwrap();
    assert!((observed_entropy - expected_entropy).abs() < 1e-6);
    assert!(first["score_breakdown"].is_object() || first["score_breakdown"].is_null());
}

#[tokio::test]
async fn parameter_sweep_logs_config_snapshots() {
    let entropy_values = [0.70_f32, 0.95_f32];

    for entropy in entropy_values {
        let mut config = EngineConfig::load_from_file("config/config.yaml").unwrap();
        let log_path = temp_log_path(&format!("entropy_{entropy:.2}"));
        config.telemetry.observation_log_path = Some(log_path.clone());
        config.telemetry.telemetry_sample_rate = 1.0;
        config.retrieval.entropy_threshold = entropy;

        let engine = Engine::new_with_config_and_db_path(config, &temp_db_path("parameter_sweep"));
        let _ = engine.process("Who leads India right now?").await;

        let observations = parse_lines(&log_path);
        assert!(!observations.is_empty());
        let logged = observations[0]["config_values_used"]["entropy_threshold"]
            .as_f64()
            .unwrap() as f32;
        assert!((logged - entropy).abs() < 0.0001);
    }
}
