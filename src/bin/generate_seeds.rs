//! Seed data generation runner.
//!
//! Generates ~1GB JSONL files for each seed category:
//! - Intelligence (reasoning chains, retrieval triggers, confidence gating)
//! - Entities (definitions, attributes, relationships)
//! - Dialogues (multi-turn conversations, social patterns)
//! - Classification (intent/tone/resolver patterns for spatial index)
//!
//! Usage:
//!   cargo run --bin generate_seeds [--output-dir <dir>] [--target-gb <float>] [--seed <u64>] [--threads <n>] [--validate]

use spse_engine::seed::bulk_generator::parallel_generate;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let output_dir = parse_arg(&args, "--output-dir")
        .unwrap_or_else(|| "datasets/seeds".to_string());
    let target_gb: f64 = parse_arg(&args, "--target-gb")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let rng_seed: u64 = parse_arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let threads: usize = parse_arg(&args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0); // 0 = auto-detect
    let validate = args.iter().any(|a| a == "--validate");

    // Configure rayon thread pool
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }

    let target_bytes = (target_gb * 1_073_741_824.0) as u64;
    let output_path = PathBuf::from(&output_dir);

    std::fs::create_dir_all(&output_path).expect("create output directory");

    let num_threads = rayon::current_num_threads();
    eprintln!("=== SPSE Seed Data Generator ===");
    eprintln!("Output directory: {}", output_path.display());
    eprintln!("Target per category: {:.2} GB ({} bytes)", target_gb, target_bytes);
    eprintln!("RNG seed: {}", rng_seed);
    eprintln!("Parallelism: {} threads", num_threads);
    eprintln!();

    let overall_start = Instant::now();
    let mut total_examples: u64 = 0;
    let mut total_bytes: u64 = 0;

    // Generator dispatch table: (name, filename, seed_offset, generator_fn)
    let generators: Vec<(&str, &str, u64, fn(&std::path::Path, u64, u64) -> (u64, u64))> = vec![
        ("intelligence", "intelligence.jsonl", 0, spse_engine::seed::generate_bulk_intelligence),
        ("entity", "entities.jsonl", 1, spse_engine::seed::generate_bulk_entities),
        ("dialogue", "dialogues.jsonl", 2, spse_engine::seed::generate_bulk_dialogues),
        ("classification", "classification.jsonl", 3, spse_engine::seed::generate_bulk_classification),
    ];

    for (idx, (name, filename, seed_offset, gen_fn)) in generators.iter().enumerate() {
        eprintln!("[{}/4] Generating {} seeds...", idx + 1, name);
        let start = Instant::now();
        let path = output_path.join(filename);
        let (count, bytes) = parallel_generate(
            &path,
            target_bytes,
            rng_seed + seed_offset,
            gen_fn,
        );
        let elapsed = start.elapsed();
        eprintln!(
            "  Done: {} examples, {} in {:.1}s ({:.0} examples/sec)",
            count,
            human_bytes(bytes),
            elapsed.as_secs_f64(),
            count as f64 / elapsed.as_secs_f64(),
        );
        total_examples += count;
        total_bytes += bytes;
    }

    let overall_elapsed = overall_start.elapsed();
    eprintln!();
    eprintln!("=== Generation Complete ===");
    eprintln!("Total examples: {}", total_examples);
    eprintln!("Total data: {}", human_bytes(total_bytes));
    eprintln!("Total time: {:.1}s", overall_elapsed.as_secs_f64());
    eprintln!("Output: {}/", output_path.display());
    eprintln!("  intelligence.jsonl  - reasoning, retrieval triggers, confidence gating");
    eprintln!("  entities.jsonl      - entity definitions, attributes, relationships");
    eprintln!("  dialogues.jsonl     - multi-turn conversations, social patterns");
    eprintln!("  classification.jsonl - intent/tone/resolver classification patterns");

    if validate {
        eprintln!();
        eprintln!("=== Validation ===");
        let files = [
            "intelligence.jsonl", "entities.jsonl",
            "dialogues.jsonl", "classification.jsonl",
        ];
        let mut all_ok = true;
        for filename in &files {
            let path = output_path.join(filename);
            match validate_jsonl(&path) {
                Ok(report) => {
                    let status = if report.has_issues() { "ISSUES" } else { "OK" };
                    eprintln!("  {} [{}]: {} examples", filename, status, report.total);
                    if report.has_issues() {
                        all_ok = false;
                        report.print_issues();
                    }
                }
                Err(e) => {
                    all_ok = false;
                    eprintln!("  {} [ERROR]: {}", filename, e);
                }
            }
        }
        if all_ok {
            eprintln!("All files passed validation.");
        } else {
            eprintln!("Validation found issues.");
            std::process::exit(1);
        }
    }
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

struct ValidationReport {
    total: u64,
    bad_json: u64,
    empty_question: u64,
    empty_answer: u64,
    missing_intent: u64,
    bad_math: u64,
}

impl ValidationReport {
    fn has_issues(&self) -> bool {
        self.bad_json > 0 || self.empty_question > 0 || self.empty_answer > 0
            || self.missing_intent > 0 || self.bad_math > 0
    }

    fn print_issues(&self) {
        if self.bad_json > 0 { eprintln!("    bad_json: {}", self.bad_json); }
        if self.empty_question > 0 { eprintln!("    empty_question: {}", self.empty_question); }
        if self.empty_answer > 0 { eprintln!("    empty_answer: {}", self.empty_answer); }
        if self.missing_intent > 0 { eprintln!("    missing_intent: {}", self.missing_intent); }
        if self.bad_math > 0 { eprintln!("    bad_math: {}", self.bad_math); }
    }
}

fn validate_jsonl(path: &std::path::Path) -> std::io::Result<ValidationReport> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut report = ValidationReport {
        total: 0, bad_json: 0, empty_question: 0, empty_answer: 0,
        missing_intent: 0, bad_math: 0,
    };

    let add_re = regex::Regex::new(r"(\d+) \+ (\d+) = (\d+)").unwrap();

    for line in reader.lines() {
        let line = line?;
        report.total += 1;

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&line);
        let obj = match parsed {
            Ok(v) => v,
            Err(_) => { report.bad_json += 1; continue; }
        };

        let q = obj.get("question").and_then(|v| v.as_str()).unwrap_or("");
        let a = obj.get("answer").and_then(|v| v.as_str()).unwrap_or("");

        if q.trim().is_empty() { report.empty_question += 1; }
        if a.trim().is_empty() { report.empty_answer += 1; }
        if obj.get("intent").and_then(|v| v.as_str()).map_or(true, |s| s.is_empty()) {
            report.missing_intent += 1;
        }

        // Check math in reasoning steps
        if let Some(reasoning) = obj.get("reasoning") {
            if let Some(steps) = reasoning.get("steps").and_then(|s| s.as_array()) {
                for step in steps {
                    let content = step.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    for cap in add_re.captures_iter(content) {
                        let a_val: i64 = cap[1].parse().unwrap_or(0);
                        let b_val: i64 = cap[2].parse().unwrap_or(0);
                        let result: i64 = cap[3].parse().unwrap_or(0);
                        if a_val + b_val != result {
                            report.bad_math += 1;
                        }
                    }
                }
            }
        }
    }

    Ok(report)
}

fn human_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}
