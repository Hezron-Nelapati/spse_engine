use clap::{Parser, Subcommand};
use spse_engine::api;
use spse_engine::engine::Engine;
use spse_engine::seed::{generate_dryrun_datasets, DryRunDatasetConfig};
use spse_engine::training::TrainingScope;
use spse_engine::types::{DryRunReport, JobState, TrainingExecutionMode, TrainingJobStatus};
use std::io::{self, Write};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "spse_engine")]
#[command(about = "Structured Predictive Search Engine", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    Interactive,
    Query {
        text: String,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        debug: bool,
    },
    Train {
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = "development")]
        execution_mode: String,
        #[arg(long, default_value = "full")]
        scope: String,
        #[arg(long, help = "Train only a single source by name, e.g. seed_entities")]
        source: Option<String>,
    },
    TrainStatus {
        job_id: String,
    },
    AuditPollution {
        #[arg(long, default_value_t = 50)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Serve {
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },
    GenerateSeed {
        #[arg(long, default_value = "dryrun")]
        phase: String,
        #[arg(long, default_value = "datasets/dryrun")]
        output_dir: String,
        #[arg(long)]
        json: bool,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Interactive) {
        Commands::Interactive => {
            let engine = Arc::new(Engine::new());
            run_interactive(engine.as_ref()).await
        }
        Commands::Query { text, json, debug } => {
            let engine = Engine::new();
            let result = engine.process(&text).await;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&result).expect("serialize result")
                );
            } else if debug {
                print_debug_result(&result, None);
            } else {
                println!("{}", result.predicted_text);
            }
        }
        Commands::Train {
            json,
            execution_mode,
            scope,
            source,
        } => {
            let execution_mode = parse_execution_mode(&execution_mode);
            let scope = parse_training_scope(&scope);
            if matches!(scope, TrainingScope::DryRun) {
                let dry_run_db =
                    std::env::temp_dir().join(format!("spse_phase0_dry_run_{}.db", Uuid::new_v4()));
                let dry_run_db = dry_run_db.display().to_string();
                let engine = Arc::new(Engine::new_with_db_path(&dry_run_db));
                if json {
                    let report = engine.run_phase0_dry_run(execution_mode).await;
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report).expect("serialize dry run report")
                    );
                } else {
                    let job_id = engine
                        .clone()
                        .start_train_with_scope(execution_mode, TrainingScope::DryRun);
                    let status = wait_for_training_completion(&engine, &job_id).await;
                    let report = engine.finalize_phase0_dry_run(status).await;
                    print_dry_run_report(&report);
                }
            } else {
                let engine = Arc::new(Engine::new());
                let status = engine
                    .train_with_scope(execution_mode, scope, source.as_deref())
                    .await;
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&status).expect("serialize training status")
                    );
                } else {
                    print_training_status(&status);
                }
            }
        }
        Commands::TrainStatus { job_id } => {
            let engine = Engine::new();
            let status = engine.training_status(&job_id);
            match status {
                Some(status) => println!(
                    "{}",
                    serde_json::to_string_pretty(&status).expect("serialize training status")
                ),
                None => {
                    eprintln!("training job not found: {job_id}");
                    std::process::exit(1);
                }
            }
        }
        Commands::AuditPollution { limit, json } => {
            let engine = Engine::new();
            let findings = engine.audit_pollution(limit);
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&findings).expect("serialize pollution findings")
                );
            } else if findings.is_empty() {
                println!("No likely pollution findings.");
            } else {
                for finding in findings {
                    println!(
                        "polluted={} ({:?}) -> canonical={} ({:?}) overlap={:.2} delta={:.2} reason={}",
                        finding.polluted_normalized,
                        finding.polluted_level,
                        finding.canonical_normalized,
                        finding.canonical_level,
                        finding.overlap_ratio,
                        finding.quality_delta,
                        finding.reason
                    );
                }
            }
        }
        Commands::Serve { port } => {
            let engine = Arc::new(Engine::new());
            if let Err(err) = api::serve(engine.clone(), port).await {
                eprintln!("{err}");
                std::process::exit(1);
            }
        }
        Commands::GenerateSeed {
            phase,
            output_dir,
            json,
        } => match phase.as_str() {
            "dryrun" => {
                let config = DryRunDatasetConfig {
                    output_dir,
                    ..Default::default()
                };
                let result = generate_dryrun_datasets(&config);
                if json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&result).expect("serialize result")
                    );
                } else {
                    println!("Generated DryRun datasets:");
                    println!(
                        "  Intent dataset: {} ({} dialogues)",
                        result.intent_dataset_path, result.intent_dialogue_count
                    );
                    println!(
                        "  Entity dataset: {} ({} entities)",
                        result.entity_dataset_path, result.entity_count
                    );
                    println!("  Intents covered: {}", result.intents_covered.len());
                    if result.quality_passed {
                        println!("  Quality: PASSED");
                    } else {
                        println!("  Quality: NEEDS ATTENTION");
                    }
                    if !result.warnings.is_empty() {
                        println!("  Warnings:");
                        for w in &result.warnings {
                            println!("    - {}", w);
                        }
                    }
                }
                if !result.quality_passed {
                    std::process::exit(1);
                }
            }
            other => {
                eprintln!("Unknown phase: {}. Supported phases: dryrun", other);
                std::process::exit(1);
            }
        },
    }
}

async fn run_interactive(engine: &Engine) {
    println!("--- SPSE Engine (Full Architecture) ---");
    println!("Type a question, paste a local .docx/.pdf path, use '/train', '/clear', '/debug <text>', '/trace <text>', or 'exit'.");

    loop {
        print!("User > ");
        io::stdout().flush().expect("stdout flush");

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("failed to read line");

        let input = input.trim();
        if input.eq_ignore_ascii_case("exit") {
            break;
        }
        if input.is_empty() {
            continue;
        }

        if input == "/train" {
            let status = engine
                .train_with_execution_mode(TrainingExecutionMode::Development)
                .await;
            println!(
                "System > Training status={:?}, phases={}, sources={}, new_units={}",
                status.status,
                status.phase_statuses.len(),
                status.progress.sources_total,
                status.learning_metrics.new_units_discovered
            );
            for phase in &status.phase_statuses {
                println!(
                    "System >   {:?}: status={:?}, sources={}/{}, unit_efficiency={:.3}, routing_accuracy={:.3}",
                    phase.phase,
                    phase.status,
                    phase.sources_processed,
                    phase.sources_total,
                    phase.metrics.unit_discovery_efficiency,
                    phase.metrics.semantic_routing_accuracy
                );
            }
            continue;
        }

        if input.starts_with("/train ") {
            println!(
                "System > Direct source training is disabled. Use '/train' to run the full ordered training pipeline."
            );
            continue;
        }

        if input == "/clear" {
            let cleared = engine.clear_session_documents();
            if cleared == 0 {
                println!("System > No active document session.");
            } else {
                println!("System > Cleared {} active document(s).", cleared);
            }
            continue;
        }

        let trace_mode = input.starts_with("/trace ");
        let debug_mode = input.starts_with("/debug ");
        let text = if trace_mode {
            input.strip_prefix("/trace ").unwrap_or(input)
        } else if debug_mode {
            input.strip_prefix("/debug ").unwrap_or(input)
        } else {
            input
        };
        let result = engine.process(text).await;
        if trace_mode {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("serialize interactive result")
            );
        } else if debug_mode {
            print_debug_result(&result, Some("System > "));
        } else {
            println!("System > {}", result.predicted_text);
        }
    }
}

fn print_debug_result(result: &spse_engine::types::ProcessResult, prefix: Option<&str>) {
    let prefix = prefix.unwrap_or("");
    println!("{prefix}{}", result.predicted_text);
    println!(
        "{prefix}intent={:?} ({:.3})",
        result.trace.intent_profile.primary, result.trace.intent_profile.confidence
    );
    println!("{prefix}confidence={:.3}", result.confidence);
    println!("{prefix}retrieval={}", result.used_retrieval);
    if !result.trace.evidence_sources.is_empty() {
        println!(
            "{prefix}sources={}",
            result.trace.evidence_sources.join(", ")
        );
    }
    if !result.trace.memory_summary.is_empty() {
        println!("{prefix}memory={}", result.trace.memory_summary);
    }
    for step in &result.trace.debug_steps {
        println!("{prefix}[L{}:{}] {}", step.layer, step.stage, step.summary);
        for (key, value) in &step.details {
            if value.is_empty() {
                continue;
            }
            println!("{prefix}  {key}={value}");
        }
    }
}

async fn wait_for_training_completion(engine: &Arc<Engine>, job_id: &str) -> TrainingJobStatus {
    let mut last_render = String::new();
    loop {
        if let Some(status) = engine.training_status(job_id) {
            let render = render_live_training_progress(&status);
            if render != last_render {
                println!("{render}");
                last_render = render;
            }
            if !matches!(status.status, JobState::Queued | JobState::Processing) {
                return status;
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

fn render_live_training_progress(status: &TrainingJobStatus) -> String {
    let state = match status.status {
        JobState::Queued => "queued",
        JobState::Processing => "processing",
        JobState::Completed => "completed",
        JobState::Failed => "failed",
    };
    let phase = status
        .active_phase
        .map(|phase| format!("{phase:?}").to_lowercase())
        .unwrap_or_else(|| "idle".to_string());
    let source = status
        .progress
        .active_source
        .clone()
        .unwrap_or_else(|| "-".to_string());
    format!(
        "live status={} phase={} source={}/{} active_source={} chunks={} bytes={} workers={}/{} queued={} prepared={} batches={} new_units={} mem_delta_kb={}",
        state,
        phase,
        status.progress.sources_processed,
        status.progress.sources_total,
        source,
        status.progress.chunks_processed,
        status.progress.bytes_processed,
        status.progress.active_workers,
        status.progress.worker_count,
        status.progress.queued_chunks,
        status.progress.prepared_chunks,
        status.progress.committed_batches,
        status.learning_metrics.new_units_discovered,
        status.learning_metrics.memory_delta_kb,
    )
}

fn print_training_status(status: &TrainingJobStatus) {
    println!(
        "training status={}, phases={}, sources={}, new_units={}",
        match status.status {
            JobState::Queued => "queued",
            JobState::Processing => "processing",
            JobState::Completed => "completed",
            JobState::Failed => "failed",
        },
        status.phase_statuses.len(),
        status.progress.sources_total,
        status.learning_metrics.new_units_discovered
    );
    if let Some(active_source) = status.progress.active_source.as_deref() {
        if !active_source.is_empty() {
            println!(
                "  progress active_source={} chunks={} bytes={} workers={}/{} queued={} prepared={} batches={}",
                active_source,
                status.progress.chunks_processed,
                status.progress.bytes_processed,
                status.progress.active_workers,
                status.progress.worker_count,
                status.progress.queued_chunks,
                status.progress.prepared_chunks,
                status.progress.committed_batches
            );
        }
    }
    println!(
        "  health stage={:?} total_units={} core={} episodic={} anchors={} efficiency(units_per_kb={:.2}, pruned={:.2}, anchor_density={:.2})",
        status.learning_metrics.database_health.maturity_stage,
        status.learning_metrics.database_health.total_units,
        status.learning_metrics.database_health.core_units,
        status.learning_metrics.database_health.episodic_units,
        status.learning_metrics.database_health.anchor_units,
        status.learning_metrics.efficiency.units_discovered_per_kb,
        status.learning_metrics.efficiency.pruned_units_percent,
        status.learning_metrics.efficiency.anchor_density,
    );
    for phase in &status.phase_statuses {
        println!(
            "  {:?}: status={:?} sources={}/{} batches={}/{} unit_efficiency={:.3} routing_accuracy={:.3}",
            phase.phase,
            phase.status,
            phase.sources_processed,
            phase.sources_total,
            phase.batches_completed,
            phase.batches_target,
            phase.metrics.unit_discovery_efficiency,
            phase.metrics.semantic_routing_accuracy
        );
    }
    if let Some(event) = status.learning_metrics.pruning_events.last() {
        println!(
            "  pruning trigger={} pruned_units={} pruned_candidates={} polluted_units={} polluted_candidates={} reasons={}",
            event.trigger,
            event.pruned_units,
            event.pruned_candidates,
            event.purged_polluted_units,
            event.purged_polluted_candidates,
            if event.reasons.is_empty() {
                "none".to_string()
            } else {
                event.reasons.join("|")
            }
        );
        for finding in event.pollution_findings.iter().take(3) {
            println!(
                "    pollution polluted={} -> canonical={} overlap={:.2} delta={:.2} reason={}",
                finding.polluted_normalized,
                finding.canonical_normalized,
                finding.overlap_ratio,
                finding.quality_delta,
                finding.reason
            );
        }
        for reference in event.pruned_references.iter().take(5) {
            println!(
                "    pruned normalized={} level={:?} utility={:.3} freq={} reason={}",
                reference.normalized,
                reference.level,
                reference.utility_score,
                reference.frequency,
                reference.reason
            );
        }
    }
    if !status.warnings.is_empty() {
        println!("  warnings={}", status.warnings.join(", "));
    }
}

fn print_dry_run_report(report: &DryRunReport) {
    print_training_status(&report.status);
    println!(
        "  dry_run snapshot_readable={} map_stable={} inference_ok={} latency_ms={} latency_per_token_ms={}",
        report.snapshot_readable,
        report.map_stable,
        report.inference_ok,
        report.inference_latency_ms,
        report.latency_per_token_ms,
    );
}

fn parse_execution_mode(value: &str) -> TrainingExecutionMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "user" => TrainingExecutionMode::User,
        "development" | "dev" => TrainingExecutionMode::Development,
        other => panic!("invalid execution mode `{other}`; expected `user` or `development`"),
    }
}

fn parse_training_scope(value: &str) -> TrainingScope {
    match value.trim().to_ascii_lowercase().as_str() {
        "full" => TrainingScope::Full,
        "bootstrap" => TrainingScope::Bootstrap,
        "dry-run" | "dry_run" | "dryrun" => TrainingScope::DryRun,
        "huggingface" | "hf" => TrainingScope::HuggingFace,
        other => {
            panic!(
                "invalid training scope `{other}`; expected `full`, `bootstrap`, `dry-run`, or `huggingface`"
            )
        }
    }
}
