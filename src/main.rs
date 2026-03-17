use clap::{Parser, Subcommand};
use spse_engine::api;
use spse_engine::engine::Engine;
use spse_engine::types::TrainingExecutionMode;
use std::io::{self, Write};
use std::sync::Arc;

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
        } => {
            let engine = Engine::new();
            let metrics = engine
                .train_with_execution_mode(parse_execution_mode(&execution_mode))
                .await;
            match metrics {
                Ok(metrics) if json => println!(
                    "{}",
                    serde_json::to_string_pretty(&metrics).expect("serialize training metrics")
                ),
                Ok(metrics) => print_training_metrics(&metrics),
                Err(error) => {
                    eprintln!("training failed: {error}");
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
    }
}

async fn run_interactive(engine: &Engine) {
    println!("--- SPSE Engine ---");
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
            match engine
                .train_with_execution_mode(TrainingExecutionMode::Development)
                .await
            {
                Ok(metrics) => {
                    print!("System > ");
                    io::stdout().flush().expect("stdout flush");
                    print_training_metrics(&metrics);
                }
                Err(error) => println!("System > training failed: {error}"),
            }
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

fn print_training_metrics(metrics: &spse_engine::types::TrainingMetrics) {
    println!(
        "training examples={} units={} efficiency={:.3} routing={:.3} error={:.3}",
        metrics.examples_ingested,
        metrics.units_created,
        metrics.unit_discovery_efficiency,
        metrics.semantic_routing_accuracy,
        metrics.prediction_error
    );
}

fn parse_execution_mode(value: &str) -> TrainingExecutionMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "user" => TrainingExecutionMode::User,
        "development" | "dev" | "" => TrainingExecutionMode::Development,
        _ => TrainingExecutionMode::Development,
    }
}
