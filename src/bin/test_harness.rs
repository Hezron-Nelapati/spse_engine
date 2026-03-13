use chrono::Utc;
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_yaml::Value as YamlValue;
use spse_engine::config::EngineConfig;
use spse_engine::engine::Engine;
use spse_engine::telemetry::test_observer::{ScoreBreakdown, TestObservation};
use spse_engine::types::{
    MemoryChannel, MemoryType, TrainBatchRequest, TrainingExecutionMode, TrainingOptions,
    TrainingSource, TrainingSourceType, TrainingStreamConfig,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Parser)]
#[command(
    name = "test_harness",
    about = "Run SPS v11 config sweeps against a controlled story corpus."
)]
struct Args {
    #[arg(long, default_value = "config/profiles.json")]
    profiles: PathBuf,
    #[arg(long, default_value = "test_data/controlled_story_dataset.json")]
    dataset: PathBuf,
    #[arg(long, default_value = "results/config_sweep")]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 5)]
    timeout_secs: u64,
    #[arg(long)]
    limit_profiles: Option<usize>,
    #[arg(long)]
    limit_questions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct ProfileManifest {
    schema_version: u32,
    base_config_path: PathBuf,
    profiles: Vec<ProfileDefinition>,
}

#[derive(Debug, Deserialize)]
struct ProfileDefinition {
    name: String,
    description: String,
    goal: String,
    config_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct ControlledDataset {
    dataset_name: String,
    schema_version: u32,
    story_word_count: usize,
    story: String,
    questions: Vec<ControlledQuestion>,
}

#[derive(Debug, Deserialize)]
struct ControlledQuestion {
    id: String,
    category: String,
    text: String,
    expected: ExpectedAnswer,
}

#[derive(Debug, Deserialize)]
struct ExpectedAnswer {
    exact_match: String,
    semantic_allowlist: Vec<String>,
    should_retrieve: bool,
    max_latency_ms: u64,
}

#[derive(Debug, Serialize)]
struct SweepReport {
    generated_at: String,
    dataset_path: String,
    profiles_manifest_path: String,
    dataset_name: String,
    schema_version: u32,
    results: Vec<ProfileRunResult>,
}

#[derive(Debug, Serialize)]
struct ProfileRunResult {
    name: String,
    description: String,
    goal: String,
    config_path: String,
    merged_config_path: String,
    run_directory: String,
    observation_log_path: String,
    training: TrainingRunSummary,
    metrics: ProfileMetrics,
    questions: Vec<QueryEvaluation>,
}

#[derive(Debug, Serialize)]
struct TrainingRunSummary {
    status: String,
    warnings: Vec<String>,
    new_units_discovered: u64,
    units_pruned: u64,
    memory_delta_kb: i64,
    anchors_protected: u64,
    map_adjustments: u64,
    intent_distribution: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct ProfileMetrics {
    accuracy_score: f32,
    exact_match_rate: f32,
    semantic_similarity_avg: f32,
    retrieval_correct_rate: f32,
    search_precision: f32,
    avg_latency_ms: f32,
    p95_latency_ms: u64,
    latency_budget_pass_rate: f32,
    memory_delta_kb: i64,
    training_memory_delta_kb: i64,
    query_memory_delta_kb: i64,
    timeout_count: u32,
    error_count: u32,
}

#[derive(Debug, Serialize)]
struct QueryEvaluation {
    question_id: String,
    category: String,
    query: String,
    predicted_text: String,
    confidence: f32,
    used_retrieval: bool,
    expected_retrieval: bool,
    retrieval_correct: bool,
    latency_ms: u64,
    max_latency_ms: u64,
    within_latency_budget: bool,
    exact_match_score: f32,
    semantic_similarity: f32,
    accuracy_score: f32,
    memory_delta_kb: i64,
    units_discovered: u32,
    units_activated: u32,
    new_units_created: u32,
    sources_consulted: Vec<String>,
    retrieval_reason: Option<String>,
    evidence_merged: u32,
    top_candidate_score: f32,
    score_breakdown: Option<ScoreBreakdown>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct CsvSummaryRow<'a> {
    profile: &'a str,
    goal: &'a str,
    accuracy_score: f32,
    exact_match_rate: f32,
    semantic_similarity_avg: f32,
    retrieval_correct_rate: f32,
    search_precision: f32,
    avg_latency_ms: f32,
    p95_latency_ms: u64,
    latency_budget_pass_rate: f32,
    memory_delta_kb: i64,
    training_memory_delta_kb: i64,
    query_memory_delta_kb: i64,
    timeout_count: u32,
    error_count: u32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let dataset = load_dataset(&args.dataset)?;
    let manifest = load_manifest(&args.profiles)?;

    let mut selected_profiles = manifest.profiles;
    if let Some(limit) = args.limit_profiles {
        selected_profiles.truncate(limit);
    }

    let mut questions = dataset.questions;
    if let Some(limit) = args.limit_questions {
        questions.truncate(limit);
    }

    fs::create_dir_all(&args.output_dir)?;
    let base_config_value = load_yaml_value(&manifest.base_config_path)?;
    let mut results = Vec::new();

    for profile in selected_profiles {
        let run_dir = args.output_dir.join(&profile.name);
        if run_dir.exists() {
            fs::remove_dir_all(&run_dir)?;
        }
        fs::create_dir_all(&run_dir)?;

        let merged_config = merge_profile_config(
            &base_config_value,
            &manifest.base_config_path,
            &profile.config_path,
        )?;
        let merged_config_path = run_dir.join("merged_config.yaml");
        fs::write(&merged_config_path, serde_yaml::to_string(&merged_config)?)?;

        let observation_log_path = run_dir.join("observations.jsonl");
        let db_path = run_dir.join("memory.db");
        let mut config = merged_config.clone();
        config.telemetry.observation_log_path = Some(observation_log_path.display().to_string());
        config.telemetry.telemetry_sample_rate = 1.0;
        config.telemetry.log_score_breakdowns = true;

        let engine =
            Engine::new_with_config_and_db_path(config.clone(), &db_path.display().to_string());
        let training = train_story(&engine, &dataset.story, &config).await;

        let mut question_results = Vec::new();
        let mut seen_observations = 0usize;
        for question in &questions {
            let result = run_question(
                &engine,
                question,
                &observation_log_path,
                &mut seen_observations,
                Duration::from_secs(args.timeout_secs),
            )
            .await;
            question_results.push(result);
        }

        let metrics = summarize_profile_metrics(&question_results, &training);
        write_profile_outputs(&run_dir, &profile, &metrics, &question_results)?;

        results.push(ProfileRunResult {
            name: profile.name,
            description: profile.description,
            goal: profile.goal,
            config_path: profile.config_path.display().to_string(),
            merged_config_path: merged_config_path.display().to_string(),
            run_directory: run_dir.display().to_string(),
            observation_log_path: observation_log_path.display().to_string(),
            training,
            metrics,
            questions: question_results,
        });
    }

    let report = SweepReport {
        generated_at: Utc::now().to_rfc3339(),
        dataset_path: args.dataset.display().to_string(),
        profiles_manifest_path: args.profiles.display().to_string(),
        dataset_name: dataset.dataset_name,
        schema_version: dataset.schema_version,
        results,
    };

    let raw_results_path = args.output_dir.join("config_sweep_results.json");
    fs::write(&raw_results_path, serde_json::to_string_pretty(&report)?)?;
    write_csv_summary(&args.output_dir.join("config_sweep_results.csv"), &report)?;

    println!(
        "Wrote raw results to {}",
        raw_results_path.as_path().display()
    );
    Ok(())
}

fn load_dataset(path: &Path) -> Result<ControlledDataset, Box<dyn std::error::Error>> {
    let dataset: ControlledDataset = serde_json::from_str(&fs::read_to_string(path)?)?;
    let actual_word_count = dataset.story.split_whitespace().count();
    if actual_word_count != dataset.story_word_count {
        return Err(format!(
            "story_word_count mismatch in {}: declared {}, actual {}",
            path.display(),
            dataset.story_word_count,
            actual_word_count
        )
        .into());
    }
    Ok(dataset)
}

fn load_manifest(path: &Path) -> Result<ProfileManifest, Box<dyn std::error::Error>> {
    let manifest: ProfileManifest = serde_json::from_str(&fs::read_to_string(path)?)?;
    if manifest.schema_version != 1 {
        return Err(format!(
            "unsupported profile manifest schema_version: {}",
            manifest.schema_version
        )
        .into());
    }
    Ok(manifest)
}

fn load_yaml_value(path: &Path) -> Result<YamlValue, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    Ok(serde_yaml::from_str(&raw)?)
}

fn merge_profile_config(
    base_config_value: &YamlValue,
    base_config_path: &Path,
    profile_path: &Path,
) -> Result<EngineConfig, Box<dyn std::error::Error>> {
    let mut base = base_config_value.clone();
    let profile_full_path = if profile_path.is_absolute() {
        profile_path.to_path_buf()
    } else {
        let base_dir = base_config_path.parent().unwrap_or_else(|| Path::new("."));
        base_dir
            .parent()
            .unwrap_or(base_dir)
            .join(profile_path)
            .to_path_buf()
    };
    let override_value = load_yaml_value(&profile_full_path)?;
    merge_yaml(&mut base, override_value);
    let config: EngineConfig = serde_yaml::from_value(base)?;
    config.validate()?;
    Ok(config)
}

fn merge_yaml(base: &mut YamlValue, overlay: YamlValue) {
    match (base, overlay) {
        (YamlValue::Mapping(base_map), YamlValue::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge_yaml(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (slot, value) => {
            *slot = value;
        }
    }
}

async fn train_story(engine: &Engine, story: &str, config: &EngineConfig) -> TrainingRunSummary {
    let request = TrainBatchRequest {
        mode: "silent".to_string(),
        sources: vec![TrainingSource {
            source_type: TrainingSourceType::Document,
            name: None,
            value: None,
            mime: Some("text/plain".to_string()),
            content: Some(story.to_string()),
            target_memory: Some(MemoryType::Episodic),
            memory_channels: Some(vec![MemoryChannel::Main]),
            stream: TrainingStreamConfig::default(),
        }],
        options: TrainingOptions {
            consolidate_immediately: true,
            max_memory_delta_mb: config.training_phases.bootstrap.max_memory_delta_mb,
            progress_interval_sec: config.silent_training.progress_interval_sec,
            tag_intent: true,
            merge_to_core: false,
            bypass_retrieval_gate: true,
            bypass_generation: true,
            daily_growth_limit_mb: Some(config.training_phases.bootstrap.daily_growth_limit_mb),
            execution_mode: TrainingExecutionMode::Development,
        },
    };
    let status = engine.train_batch(request).await;
    TrainingRunSummary {
        status: format!("{:?}", status.status).to_ascii_lowercase(),
        warnings: status.warnings,
        new_units_discovered: status.learning_metrics.new_units_discovered,
        units_pruned: status.learning_metrics.units_pruned,
        memory_delta_kb: status.learning_metrics.memory_delta_kb,
        anchors_protected: status.learning_metrics.anchors_protected,
        map_adjustments: status.learning_metrics.map_adjustments,
        intent_distribution: status.intent_distribution,
    }
}

async fn run_question(
    engine: &Engine,
    question: &ControlledQuestion,
    observation_log_path: &Path,
    seen_observations: &mut usize,
    timeout: Duration,
) -> QueryEvaluation {
    match tokio::time::timeout(timeout, engine.process(&question.text)).await {
        Ok(result) => {
            let observation = load_new_observation(observation_log_path, seen_observations);
            let exact_match_score =
                exact_match_score(&result.predicted_text, &question.expected.exact_match);
            let semantic_similarity = semantic_similarity(
                &result.predicted_text,
                &question.expected.semantic_allowlist,
            );
            let retrieval_correct = result.used_retrieval == question.expected.should_retrieve;
            let accuracy_score =
                question_accuracy(exact_match_score, semantic_similarity, retrieval_correct);
            let latency_ms = observation
                .as_ref()
                .map(|entry| entry.total_latency_ms)
                .unwrap_or(0);

            QueryEvaluation {
                question_id: question.id.clone(),
                category: question.category.clone(),
                query: question.text.clone(),
                predicted_text: result.predicted_text,
                confidence: result.confidence,
                used_retrieval: result.used_retrieval,
                expected_retrieval: question.expected.should_retrieve,
                retrieval_correct,
                latency_ms,
                max_latency_ms: question.expected.max_latency_ms,
                within_latency_budget: latency_ms <= question.expected.max_latency_ms,
                exact_match_score,
                semantic_similarity,
                accuracy_score,
                memory_delta_kb: observation
                    .as_ref()
                    .map(|entry| entry.memory_delta_kb)
                    .unwrap_or_default(),
                units_discovered: observation
                    .as_ref()
                    .map(|entry| entry.units_discovered)
                    .unwrap_or_default(),
                units_activated: observation
                    .as_ref()
                    .map(|entry| entry.units_activated)
                    .unwrap_or_default(),
                new_units_created: observation
                    .as_ref()
                    .map(|entry| entry.new_units_created)
                    .unwrap_or_default(),
                sources_consulted: observation
                    .as_ref()
                    .map(|entry| entry.sources_consulted.clone())
                    .unwrap_or_default(),
                retrieval_reason: observation
                    .as_ref()
                    .and_then(|entry| entry.retrieval_reason.clone()),
                evidence_merged: observation
                    .as_ref()
                    .map(|entry| entry.evidence_merged)
                    .unwrap_or_default(),
                top_candidate_score: observation
                    .as_ref()
                    .map(|entry| entry.top_candidate_score)
                    .unwrap_or_default(),
                score_breakdown: observation.and_then(|entry| entry.score_breakdown),
                error: None,
            }
        }
        Err(_) => QueryEvaluation {
            question_id: question.id.clone(),
            category: question.category.clone(),
            query: question.text.clone(),
            predicted_text: String::new(),
            confidence: 0.0,
            used_retrieval: false,
            expected_retrieval: question.expected.should_retrieve,
            retrieval_correct: false,
            latency_ms: timeout.as_millis() as u64,
            max_latency_ms: question.expected.max_latency_ms,
            within_latency_budget: false,
            exact_match_score: 0.0,
            semantic_similarity: 0.0,
            accuracy_score: 0.0,
            memory_delta_kb: 0,
            units_discovered: 0,
            units_activated: 0,
            new_units_created: 0,
            sources_consulted: Vec::new(),
            retrieval_reason: None,
            evidence_merged: 0,
            top_candidate_score: 0.0,
            score_breakdown: None,
            error: Some("timeout".to_string()),
        },
    }
}

fn load_new_observation(path: &Path, seen: &mut usize) -> Option<TestObservation> {
    if !path.exists() {
        return None;
    }
    let raw = fs::read_to_string(path).ok()?;
    let observations = raw
        .lines()
        .filter_map(|line| serde_json::from_str::<TestObservation>(line).ok())
        .collect::<Vec<_>>();
    if observations.len() <= *seen {
        return observations.last().cloned();
    }
    let next = observations.get(*seen).cloned();
    *seen = observations.len();
    next
}

fn exact_match_score(answer: &str, exact_match: &str) -> f32 {
    let normalized_answer = normalize(answer);
    let normalized_exact = normalize(exact_match);
    if normalized_answer.is_empty() || normalized_exact.is_empty() {
        return 0.0;
    }
    if normalized_answer == normalized_exact
        || normalized_answer.contains(&normalized_exact)
        || normalized_exact.contains(&normalized_answer)
    {
        1.0
    } else {
        0.0
    }
}

fn semantic_similarity(answer: &str, allowlist: &[String]) -> f32 {
    let normalized_answer = normalize(answer);
    if normalized_answer.is_empty() || allowlist.is_empty() {
        return 0.0;
    }

    let hits = allowlist
        .iter()
        .filter(|phrase| normalized_answer.contains(&normalize(phrase)))
        .count() as f32;
    let allowlist_score = hits / allowlist.len() as f32;

    let answer_tokens = token_set(&normalized_answer);
    let best_overlap = allowlist
        .iter()
        .map(|phrase| {
            let phrase_tokens = token_set(&normalize(phrase));
            jaccard(&answer_tokens, &phrase_tokens)
        })
        .fold(0.0, f32::max);

    allowlist_score.max(best_overlap)
}

fn question_accuracy(exact_match: f32, semantic_similarity: f32, retrieval_correct: bool) -> f32 {
    let retrieval_score = if retrieval_correct { 1.0 } else { 0.0 };
    (0.45 * exact_match + 0.35 * semantic_similarity + 0.20 * retrieval_score).clamp(0.0, 1.0)
}

fn summarize_profile_metrics(
    questions: &[QueryEvaluation],
    training: &TrainingRunSummary,
) -> ProfileMetrics {
    let count = questions.len().max(1) as f32;
    let accuracy_score = questions.iter().map(|q| q.accuracy_score).sum::<f32>() / count;
    let exact_match_rate = questions.iter().map(|q| q.exact_match_score).sum::<f32>() / count;
    let semantic_similarity_avg =
        questions.iter().map(|q| q.semantic_similarity).sum::<f32>() / count;
    let retrieval_correct_rate =
        questions.iter().filter(|q| q.retrieval_correct).count() as f32 / count;
    let search_precision = retrieval_correct_rate;
    let avg_latency_ms = questions.iter().map(|q| q.latency_ms as f32).sum::<f32>() / count;
    let latency_budget_pass_rate =
        questions.iter().filter(|q| q.within_latency_budget).count() as f32 / count;

    let mut latencies = questions.iter().map(|q| q.latency_ms).collect::<Vec<_>>();
    latencies.sort_unstable();
    let p95_latency_ms = if latencies.is_empty() {
        0
    } else {
        let index = ((latencies.len() - 1) as f32 * 0.95).round() as usize;
        latencies[index.min(latencies.len() - 1)]
    };

    let timeout_count = questions
        .iter()
        .filter(|q| q.error.as_deref() == Some("timeout"))
        .count() as u32;
    let error_count = questions.iter().filter(|q| q.error.is_some()).count() as u32;
    let query_memory_delta_kb = questions
        .iter()
        .map(|q| q.memory_delta_kb.max(0))
        .sum::<i64>();

    ProfileMetrics {
        accuracy_score,
        exact_match_rate,
        semantic_similarity_avg,
        retrieval_correct_rate,
        search_precision,
        avg_latency_ms,
        p95_latency_ms,
        latency_budget_pass_rate,
        memory_delta_kb: training.memory_delta_kb + query_memory_delta_kb,
        training_memory_delta_kb: training.memory_delta_kb,
        query_memory_delta_kb,
        timeout_count,
        error_count,
    }
}

fn write_profile_outputs(
    run_dir: &Path,
    profile: &ProfileDefinition,
    metrics: &ProfileMetrics,
    questions: &[QueryEvaluation],
) -> Result<(), Box<dyn std::error::Error>> {
    let summary_path = run_dir.join("profile_summary.json");
    let query_results_path = run_dir.join("query_results.jsonl");
    fs::write(
        summary_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "profile": profile.name,
            "goal": profile.goal,
            "metrics": metrics,
        }))?,
    )?;

    let mut lines = String::new();
    for result in questions {
        lines.push_str(&serde_json::to_string(result)?);
        lines.push('\n');
    }
    fs::write(query_results_path, lines)?;
    Ok(())
}

fn write_csv_summary(
    output_path: &Path,
    report: &SweepReport,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut csv = String::from(
        "profile,goal,accuracy_score,exact_match_rate,semantic_similarity_avg,retrieval_correct_rate,search_precision,avg_latency_ms,p95_latency_ms,latency_budget_pass_rate,memory_delta_kb,training_memory_delta_kb,query_memory_delta_kb,timeout_count,error_count\n",
    );
    for result in &report.results {
        let row = CsvSummaryRow {
            profile: &result.name,
            goal: &result.goal,
            accuracy_score: result.metrics.accuracy_score,
            exact_match_rate: result.metrics.exact_match_rate,
            semantic_similarity_avg: result.metrics.semantic_similarity_avg,
            retrieval_correct_rate: result.metrics.retrieval_correct_rate,
            search_precision: result.metrics.search_precision,
            avg_latency_ms: result.metrics.avg_latency_ms,
            p95_latency_ms: result.metrics.p95_latency_ms,
            latency_budget_pass_rate: result.metrics.latency_budget_pass_rate,
            memory_delta_kb: result.metrics.memory_delta_kb,
            training_memory_delta_kb: result.metrics.training_memory_delta_kb,
            query_memory_delta_kb: result.metrics.query_memory_delta_kb,
            timeout_count: result.metrics.timeout_count,
            error_count: result.metrics.error_count,
        };
        csv.push_str(&format!(
            "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{},{:.4},{},{},{},{},{}\n",
            escape_csv(row.profile),
            escape_csv(row.goal),
            row.accuracy_score,
            row.exact_match_rate,
            row.semantic_similarity_avg,
            row.retrieval_correct_rate,
            row.search_precision,
            row.avg_latency_ms,
            row.p95_latency_ms,
            row.latency_budget_pass_rate,
            row.memory_delta_kb,
            row.training_memory_delta_kb,
            row.query_memory_delta_kb,
            row.timeout_count,
            row.error_count
        ));
    }
    fs::write(output_path, csv)?;
    Ok(())
}

fn escape_csv(value: &str) -> String {
    if value.contains(',') || value.contains('"') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn token_set(text: &str) -> Vec<String> {
    let mut tokens = text
        .split_whitespace()
        .map(|token| token.to_string())
        .collect::<Vec<_>>();
    tokens.sort();
    tokens.dedup();
    tokens
}

fn jaccard(left: &[String], right: &[String]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let left_set = left.iter().collect::<std::collections::BTreeSet<_>>();
    let right_set = right.iter().collect::<std::collections::BTreeSet<_>>();
    let intersection = left_set.intersection(&right_set).count() as f32;
    let union = left_set.union(&right_set).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        intersection / union
    }
}
