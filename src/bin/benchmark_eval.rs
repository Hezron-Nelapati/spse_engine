use spse_engine::engine::Engine;
use spse_engine::types::{IntentKind, ProcessResult};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RetrievalExpectation {
    Yes,
    No,
    Conditional,
}

impl RetrievalExpectation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Yes => "retrieval_triggered",
            Self::No => "no_retrieval",
            Self::Conditional => "conditional",
        }
    }
}

#[derive(Clone)]
struct BenchmarkCase {
    id: String,
    domain: &'static str,
    scenario: &'static str,
    setup_documents: Vec<String>,
    setup_turns: Vec<String>,
    prompt: String,
    reference: String,
    expected_intent: IntentKind,
    expected_retrieval: RetrievalExpectation,
    expected_source: &'static str,
    isolated: bool,
}

struct BenchmarkOutcome {
    case: BenchmarkCase,
    product_answer: String,
    actual_intent: IntentKind,
    used_retrieval: bool,
    keyword_score: f32,
    lexical_score: f32,
    intent_score: f32,
    retrieval_score: f32,
    overall_score: f32,
    passed: bool,
    evidence_sources: Vec<String>,
}

#[tokio::main]
async fn main() {
    let default_doc =
        "/Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx";
    let doc_path = env::var("SPSE_BENCH_DOC").unwrap_or_else(|_| default_doc.to_string());
    if !Path::new(&doc_path).exists() {
        eprintln!("benchmark document not found: {doc_path}");
        std::process::exit(1);
    }

    let report_path =
        env::var("SPSE_BENCH_REPORT").unwrap_or_else(|_| "benchmarks/last_report.md".to_string());
    let scenario_doc_path = env::var("SPSE_BENCH_SCENARIO_DOC")
        .unwrap_or_else(|_| "docs/benchmark_scenarios.md".to_string());

    let cases = build_cases(&doc_path);
    if cases.len() < 30 {
        eprintln!("expected at least 30 cases, found {}", cases.len());
        std::process::exit(1);
    }

    fs::create_dir_all("benchmarks").expect("create benchmark directory");
    fs::create_dir_all("docs").expect("create docs directory");
    fs::write(&scenario_doc_path, render_catalog(&doc_path, &cases))
        .expect("write benchmark catalog");

    let general_db_path = temp_db_path("spse_bench_general");
    let general_engine = Engine::new_with_db_path(&general_db_path);
    let doc_db_path = temp_db_path("spse_bench_doc");
    let doc_engine = Engine::new_with_db_path(&doc_db_path);
    let _ = doc_engine.process(&format!("\"{doc_path}\"")).await;

    let mut outcomes = Vec::new();
    for case in cases {
        let result = run_case(&case, &general_engine, &doc_engine).await;
        outcomes.push(evaluate(case, result));
    }

    let _ = fs::remove_file(&general_db_path);
    let _ = fs::remove_file(&doc_db_path);

    outcomes.sort_by(|lhs, rhs| lhs.overall_score.total_cmp(&rhs.overall_score));
    fs::write(&report_path, render_report(&doc_path, &outcomes)).expect("write benchmark report");

    let passed = outcomes.iter().filter(|outcome| outcome.passed).count();
    let avg = outcomes
        .iter()
        .map(|outcome| outcome.overall_score)
        .sum::<f32>()
        / outcomes.len() as f32;
    println!(
        "benchmark_cases={} passed={} failed={} average_score={:.3} report={} scenario_doc={}",
        outcomes.len(),
        passed,
        outcomes.len() - passed,
        avg,
        report_path,
        scenario_doc_path
    );
}

async fn run_case(
    case: &BenchmarkCase,
    general_engine: &Engine,
    doc_engine: &Engine,
) -> ProcessResult {
    if case.isolated {
        let db_path = temp_db_path(&case.id);
        let engine = Engine::new_with_db_path(&db_path);
        for path in &case.setup_documents {
            let _ = engine.process(&format!("\"{path}\"")).await;
        }
        for turn in &case.setup_turns {
            let _ = engine.process(turn).await;
        }
        let result = engine.process(&case.prompt).await;
        let _ = fs::remove_file(&db_path);
        return result;
    }

    if !case.setup_documents.is_empty() {
        doc_engine.reset_session_query_state();
        return doc_engine.process(&case.prompt).await;
    }

    general_engine.clear_session_documents();
    general_engine.reset_session_query_state();
    for turn in &case.setup_turns {
        let _ = general_engine.process(turn).await;
    }
    general_engine.process(&case.prompt).await
}

fn temp_db_path(name: &str) -> String {
    let file = format!("{}_{}.db", name, Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}

fn evaluate(case: BenchmarkCase, result: ProcessResult) -> BenchmarkOutcome {
    let product_answer = result.predicted_text.trim().to_string();
    let reference_terms = reference_terms(&case.reference);
    let answer_terms = normalize_terms(&product_answer);
    let keyword_hits = reference_terms
        .iter()
        .filter(|term| answer_terms.contains(*term))
        .count();
    let keyword_score = if reference_terms.is_empty() {
        1.0
    } else {
        keyword_hits as f32 / reference_terms.len() as f32
    };
    let lexical_score = jaccard(&reference_terms, &answer_terms);
    let intent_score = if matches!(case.expected_intent, IntentKind::Unknown)
        || result.trace.intent_profile.primary == case.expected_intent
    {
        1.0
    } else {
        0.0
    };
    let retrieval_score = match case.expected_retrieval {
        RetrievalExpectation::Yes => result.used_retrieval as u8 as f32,
        RetrievalExpectation::No => (!result.used_retrieval) as u8 as f32,
        RetrievalExpectation::Conditional => 1.0,
    };
    let overall_score = (0.50 * keyword_score)
        + (0.20 * lexical_score)
        + (0.15 * intent_score)
        + (0.15 * retrieval_score);
    let passed = overall_score >= 0.67;

    BenchmarkOutcome {
        case,
        product_answer,
        actual_intent: result.trace.intent_profile.primary,
        used_retrieval: result.used_retrieval,
        keyword_score,
        lexical_score,
        intent_score,
        retrieval_score,
        overall_score,
        passed,
        evidence_sources: result.trace.evidence_sources,
    }
}

fn build_cases(doc_path: &str) -> Vec<BenchmarkCase> {
    let mut cases = Vec::new();
    let doc_setup = vec![doc_path.to_string()];

    let social_cases = vec![
        case(
            "general_hi",
            "social",
            "Basic greeting",
            "hi",
            "Hi. Ask me a question or give me a document path.",
            IntentKind::Greeting,
            RetrievalExpectation::No,
            "Direct social short-circuit",
        ),
        case(
            "general_help",
            "social",
            "Capability help request",
            "help",
            "Give me a question, a document path, or use train to persist a document.",
            IntentKind::Help,
            RetrievalExpectation::No,
            "Direct help short-circuit",
        ),
        case(
            "general_thanks",
            "social",
            "Gratitude acknowledgement",
            "thanks",
            "You're welcome.",
            IntentKind::Gratitude,
            RetrievalExpectation::No,
            "Direct social short-circuit",
        ),
        case(
            "general_bye",
            "social",
            "Conversation close",
            "bye",
            "Goodbye.",
            IntentKind::Farewell,
            RetrievalExpectation::No,
            "Direct social short-circuit",
        ),
        case(
            "general_continue_empty",
            "social",
            "Continue without active context",
            "continue",
            "There is no active document session. Ask a question or load a document first.",
            IntentKind::Continue,
            RetrievalExpectation::No,
            "Session control",
        ),
        case(
            "general_clear_empty",
            "social",
            "Clear without active context",
            "clear the document context",
            "No active document session to clear.",
            IntentKind::Forget,
            RetrievalExpectation::No,
            "Session control",
        ),
    ];
    cases.extend(social_cases);

    let reasoning_cases = vec![
        case(
            "calc_2x2",
            "reasoning",
            "Static multiplication",
            "What is 2 * 2?",
            "4",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_average",
            "reasoning",
            "Grouped arithmetic",
            "What is (3 + 5) / 2 ?",
            "4",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_subtract",
            "reasoning",
            "Subtraction command",
            "calculate 12 - 7",
            "5",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_multiply",
            "reasoning",
            "Bare multiplication",
            "7 * 6",
            "42",
            IntentKind::Analyze,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_nested",
            "reasoning",
            "Nested arithmetic",
            "what is (9 - 3) * 2?",
            "12",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_divide",
            "reasoning",
            "Division",
            "18 / 3",
            "6",
            IntentKind::Analyze,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_add",
            "reasoning",
            "Simple addition",
            "What is 15 + 27?",
            "42",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
        case(
            "calc_mixed",
            "reasoning",
            "Mixed arithmetic",
            "what is 5 * (2 + 4) - 3?",
            "27",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal symbolic short-circuit",
        ),
    ];
    cases.extend(reasoning_cases);

    let open_world_cases = vec![
        isolated_case(
            "world_president_india",
            "open_world",
            "Temporal fact lookup",
            Vec::new(),
            Vec::new(),
            "Who is the President of India?",
            "Droupadi Murmu is the current president of India.",
            IntentKind::Question,
            RetrievalExpectation::Yes,
            "External search (web)",
        ),
        isolated_case(
            "world_verify_president_india",
            "open_world",
            "Verification with freshness pressure",
            Vec::new(),
            Vec::new(),
            "verify whether the president of india is current",
            "Droupadi Murmu is the current president of India.",
            IntentKind::Verify,
            RetrievalExpectation::Yes,
            "External search (web)",
        ),
        isolated_case(
            "world_capital_france",
            "open_world",
            "Static fact from external retrieval",
            Vec::new(),
            Vec::new(),
            "What is the capital of France?",
            "Paris is the capital of France.",
            IntentKind::Question,
            RetrievalExpectation::Yes,
            "External search (web)",
        ),
        isolated_case(
            "world_cars_overview",
            "open_world",
            "Broad exploration query",
            Vec::new(),
            Vec::new(),
            "Tell me about cars.",
            "Cars are motor vehicles used for road transportation.",
            IntentKind::Explain,
            RetrievalExpectation::Conditional,
            "Hybrid (internal map plus web)",
        ),
        isolated_case(
            "world_ferrari_list",
            "open_world",
            "Specific extraction list",
            Vec::new(),
            Vec::new(),
            "List all cars by Ferrari.",
            "Ferrari cars include Ferrari 250, Ferrari F40, and LaFerrari.",
            IntentKind::Extract,
            RetrievalExpectation::Yes,
            "External search (web)",
        ),
        isolated_case(
            "world_car_definition",
            "open_world",
            "Definition query",
            Vec::new(),
            Vec::new(),
            "What is a car?",
            "A car is a wheeled motor vehicle used for transportation.",
            IntentKind::Question,
            RetrievalExpectation::Conditional,
            "Hybrid (internal map plus web)",
        ),
        isolated_case(
            "world_rust_language",
            "open_world",
            "Technology concept lookup",
            Vec::new(),
            Vec::new(),
            "What is Rust programming language?",
            "Rust is a systems programming language focused on safety and performance.",
            IntentKind::Question,
            RetrievalExpectation::Conditional,
            "Hybrid (internal map plus web)",
        ),
        isolated_case(
            "world_photosynthesis",
            "open_world",
            "Science explanation",
            Vec::new(),
            Vec::new(),
            "Explain photosynthesis.",
            "Photosynthesis is the process plants use to convert light into chemical energy.",
            IntentKind::Explain,
            RetrievalExpectation::Conditional,
            "Hybrid (internal map plus web)",
        ),
        isolated_case(
            "world_microsoft_founder",
            "open_world",
            "Historical founder lookup",
            Vec::new(),
            Vec::new(),
            "Who founded Microsoft?",
            "Bill Gates and Paul Allen founded Microsoft.",
            IntentKind::Question,
            RetrievalExpectation::Yes,
            "External search (web)",
        ),
        isolated_case(
            "world_tcp_usage",
            "open_world",
            "Technical purpose question",
            Vec::new(),
            Vec::new(),
            "What is TCP used for?",
            "TCP is used for reliable ordered data delivery over networks.",
            IntentKind::Question,
            RetrievalExpectation::Conditional,
            "Hybrid (internal map plus web)",
        ),
    ];
    cases.extend(open_world_cases);

    let episodic_cases = vec![
        isolated_case(
            "episodic_hotel",
            "episodic",
            "Personal trip memory",
            Vec::new(),
            vec!["Remember this: The hotel from my last trip was Lotus Grand.".to_string()],
            "What was the hotel name from my last trip?",
            "Lotus Grand",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_dentist",
            "episodic",
            "Appointment recall",
            Vec::new(),
            vec!["Remember this: My dentist appointment is on 14 April at 3 PM.".to_string()],
            "When is my dentist appointment?",
            "14 April 3 PM",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_theme",
            "episodic",
            "Preference recall",
            Vec::new(),
            vec!["Remember this: My preferred editor theme is Solarized Dark.".to_string()],
            "What is my preferred editor theme?",
            "Solarized Dark",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_codename",
            "episodic",
            "Project codename recall",
            Vec::new(),
            vec![
                "Remember this: The codename for the migration project is Harbor Lift.".to_string(),
            ],
            "What is the codename for the migration project?",
            "Harbor Lift",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_flight",
            "episodic",
            "Flight number recall",
            Vec::new(),
            vec!["Remember this: My return flight number is AI 287.".to_string()],
            "What is my return flight number?",
            "AI 287",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_book",
            "episodic",
            "Reading-plan recall",
            Vec::new(),
            vec![
                "Remember this: The book I want to read next is The Pragmatic Programmer."
                    .to_string(),
            ],
            "What book do I want to read next?",
            "The Pragmatic Programmer",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_contact",
            "episodic",
            "Emergency-contact recall",
            Vec::new(),
            vec!["Remember this: My emergency contact is Riya Patel.".to_string()],
            "Who is my emergency contact?",
            "Riya Patel",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
        isolated_case(
            "episodic_wifi",
            "episodic",
            "Short code recall",
            Vec::new(),
            vec!["Remember this: The guest Wi-Fi code is Cedar-48.".to_string()],
            "What is the guest Wi-Fi code?",
            "Cedar-48",
            IntentKind::Question,
            RetrievalExpectation::No,
            "Internal episodic memory",
        ),
    ];
    cases.extend(episodic_cases);

    cases.push(isolated_case(
        "doc_continue_active",
        "document_workflow",
        "Continue with active document session",
        doc_setup.clone(),
        Vec::new(),
        "continue",
        "Continuing with the active document. Ask the next question.",
        IntentKind::Continue,
        RetrievalExpectation::No,
        "Session document memory",
    ));
    cases.push(isolated_case(
        "doc_clear_active",
        "document_workflow",
        "Clear active document session",
        doc_setup.clone(),
        Vec::new(),
        "clear the document context",
        "Cleared 1 active document.",
        IntentKind::Forget,
        RetrievalExpectation::No,
        "Session document memory",
    ));

    let document_cases = vec![
        ("doc_architecture_class", "Document architecture class", "what is the architecture class?", "Tokenizer-free, CPU-friendly personal intelligence architecture.", IntentKind::Question),
        ("doc_core_mechanism", "Document core mechanism", "what is the core mechanism?", "Dynamic unit discovery, 3D semantic routing, local candidate search, and optional web retrieval.", IntentKind::Question),
        ("doc_knowledge_model", "Document knowledge model", "what is the knowledge model?", "Dual memory stores with trust-aware ingestion and anchored sequence memory.", IntentKind::Question),
        ("doc_deployment_target", "Document deployment target", "what is the primary deployment target?", "Edge and CPU-constrained systems with lifelong adaptation.", IntentKind::Question),
        ("doc_revision_status", "Document revision status", "what is the revision status?", "Final publication-ready architecture documentation.", IntentKind::Question),
        ("doc_abstract_summary", "Abstract summary", "summarize the abstract", "The abstract presents a tokenizer-free architecture with dynamic unit discovery, 3D routing, anchor memory, and optional retrieval.", IntentKind::Summarize),
        ("doc_architecture_summary", "Architecture summary", "summarize the architecture", "The architecture combines dynamic units, 3D semantic routing, anchor memory, trust-aware ingestion, and optional retrieval.", IntentKind::Summarize),
        ("doc_memory_question", "Memory-specific lookup", "what does it say about memory?", "The document describes dual memory stores, anchored sequence memory, consolidation, and episodic to core transitions.", IntentKind::Question),
        ("doc_reader_question", "Audience guidance", "who should focus on the runtime and training flow?", "Engineering teams should focus on the runtime and training flow, interfaces, and performance assumptions.", IntentKind::Question),
        ("doc_dual_memory_compare", "Compare memory tiers", "compare core memory and episodic memory", "Core memory stores stable knowledge, while episodic memory holds recent or newly learned material before consolidation.", IntentKind::Compare),
        ("doc_layer21", "Memory governance layer", "what is the function of layer 21?", "Layer 21 prevents unbounded growth and fragmentation through pruning, compaction, and archival.", IntentKind::Question),
        ("doc_layer9", "Retrieval gate layer", "what is the function of layer 9?", "Layer 9 decides whether internal knowledge is sufficient or search is needed.", IntentKind::Question),
    ];
    for (id, scenario, prompt, reference, intent) in document_cases {
        cases.push(doc_case(
            id,
            "document",
            scenario,
            doc_setup.clone(),
            prompt,
            reference,
            intent,
        ));
    }

    cases
}

fn case(
    id: &str,
    domain: &'static str,
    scenario: &'static str,
    prompt: &str,
    reference: &str,
    expected_intent: IntentKind,
    expected_retrieval: RetrievalExpectation,
    expected_source: &'static str,
) -> BenchmarkCase {
    BenchmarkCase {
        id: id.to_string(),
        domain,
        scenario,
        setup_documents: Vec::new(),
        setup_turns: Vec::new(),
        prompt: prompt.to_string(),
        reference: reference.to_string(),
        expected_intent,
        expected_retrieval,
        expected_source,
        isolated: false,
    }
}

fn isolated_case(
    id: &str,
    domain: &'static str,
    scenario: &'static str,
    setup_documents: Vec<String>,
    setup_turns: Vec<String>,
    prompt: &str,
    reference: &str,
    expected_intent: IntentKind,
    expected_retrieval: RetrievalExpectation,
    expected_source: &'static str,
) -> BenchmarkCase {
    BenchmarkCase {
        id: id.to_string(),
        domain,
        scenario,
        setup_documents,
        setup_turns,
        prompt: prompt.to_string(),
        reference: reference.to_string(),
        expected_intent,
        expected_retrieval,
        expected_source,
        isolated: true,
    }
}

fn doc_case(
    id: &str,
    domain: &'static str,
    scenario: &'static str,
    setup_documents: Vec<String>,
    prompt: &str,
    reference: &str,
    expected_intent: IntentKind,
) -> BenchmarkCase {
    BenchmarkCase {
        id: id.to_string(),
        domain,
        scenario,
        setup_documents,
        setup_turns: Vec::new(),
        prompt: prompt.to_string(),
        reference: reference.to_string(),
        expected_intent,
        expected_retrieval: RetrievalExpectation::No,
        expected_source: "Session document memory",
        isolated: false,
    }
}

fn render_catalog(doc_path: &str, cases: &[BenchmarkCase]) -> String {
    let mut by_domain = BTreeMap::new();
    let mut by_intent = BTreeMap::new();
    let mut by_retrieval = BTreeMap::new();
    let mut by_source = BTreeMap::new();
    for case in cases {
        *by_domain.entry(case.domain).or_insert(0usize) += 1;
        *by_intent
            .entry(format!("{:?}", case.expected_intent))
            .or_insert(0usize) += 1;
        *by_retrieval
            .entry(case.expected_retrieval.as_str())
            .or_insert(0usize) += 1;
        *by_source.entry(case.expected_source).or_insert(0usize) += 1;
    }

    let mut doc = String::new();
    doc.push_str("# SPSE Scenario Benchmark Catalog\n\n");
    doc.push_str(
        "This catalog is the practical scenario matrix used by `src/bin/benchmark_eval.rs`. It focuses on real product behavior across social interaction, symbolic reasoning, open-world retrieval, personal memory recall, and document-grounded Q&A. Each row includes the expected query, intent, retrieval decision, data source, and answer target used during scoring.\n\n",
    );
    doc.push_str(&format!("Primary architecture document: `{doc_path}`\n\n"));
    doc.push_str(&format!("Total scenarios: {}\n\n", cases.len()));

    doc.push_str("## Distribution By Domain\n\n");
    for (domain, count) in by_domain {
        doc.push_str(&format!("- `{domain}`: {count}\n"));
    }

    doc.push_str("\n## Distribution By Expected Intent\n\n");
    for (intent, count) in by_intent {
        doc.push_str(&format!("- `{intent}`: {count}\n"));
    }

    doc.push_str("\n## Distribution By Retrieval Decision\n\n");
    for (retrieval, count) in by_retrieval {
        doc.push_str(&format!("- `{retrieval}`: {count}\n"));
    }

    doc.push_str("\n## Distribution By Expected Data Source\n\n");
    for (source, count) in by_source {
        doc.push_str(&format!("- {}: {count}\n", escape_md(source)));
    }

    let mut grouped = BTreeMap::new();
    for case in cases {
        grouped
            .entry(case.domain)
            .or_insert_with(Vec::new)
            .push(case);
    }

    for (domain, group) in grouped {
        doc.push_str(&format!("\n## {}\n\n", to_title(domain)));
        doc.push_str("| ID | Scenario | Setup | Query | Expected Intent | Retrieval | Data Source | Expected Answer |\n");
        doc.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for case in group {
            doc.push_str(&format!(
                "| `{}` | {} | {} | {} | `{:?}` | `{}` | {} | {} |\n",
                case.id,
                escape_md(case.scenario),
                escape_md(&setup_summary(case)),
                escape_md(&case.prompt),
                case.expected_intent,
                case.expected_retrieval.as_str(),
                escape_md(case.expected_source),
                escape_md(&case.reference),
            ));
        }
    }

    doc
}

fn render_report(doc_path: &str, outcomes: &[BenchmarkOutcome]) -> String {
    let total = outcomes.len();
    let passed = outcomes.iter().filter(|outcome| outcome.passed).count();
    let avg = outcomes
        .iter()
        .map(|outcome| outcome.overall_score)
        .sum::<f32>()
        / total as f32;

    let mut by_domain = BTreeMap::new();
    for outcome in outcomes {
        let entry = by_domain
            .entry(outcome.case.domain)
            .or_insert((0usize, 0usize, 0.0f32));
        entry.0 += 1;
        if outcome.passed {
            entry.1 += 1;
        }
        entry.2 += outcome.overall_score;
    }

    let mut report = String::new();
    report.push_str("# SPSE Benchmark Report\n\n");
    report.push_str(&format!("Document: `{doc_path}`\n\n"));
    report.push_str(&format!(
        "Total cases: {total}\nPassed: {passed}\nFailed: {}\nAverage score: {:.3}\n\n",
        total - passed,
        avg
    ));

    report.push_str("## Domain Summary\n\n");
    for (domain, (count, passed_count, total_score)) in by_domain {
        report.push_str(&format!(
            "- `{domain}`: passed {passed_count}/{count}, average {:.3}\n",
            total_score / count as f32
        ));
    }

    report.push_str("\n## Worst 25 Cases\n\n");
    for outcome in outcomes.iter().take(25) {
        report.push_str(&format!(
            "### {}\n\n- Domain: `{}`\n- Scenario: {}\n- Score: {:.3}\n- Expected intent: `{:?}`\n- Actual intent: `{:?}`\n- Expected retrieval: `{}`\n- Actual retrieval: `{}`\n- Expected source: {}\n- Prompt: {}\n- Reference: {}\n- Product: {}\n- Sources: {}\n\n",
            outcome.case.id,
            outcome.case.domain,
            outcome.case.scenario,
            outcome.overall_score,
            outcome.case.expected_intent,
            outcome.actual_intent,
            outcome.case.expected_retrieval.as_str(),
            outcome.used_retrieval,
            outcome.case.expected_source,
            outcome.case.prompt,
            outcome.case.reference,
            outcome.product_answer,
            if outcome.evidence_sources.is_empty() {
                "none".to_string()
            } else {
                outcome.evidence_sources.join(", ")
            }
        ));
    }

    report.push_str("## Full Results\n\n");
    for outcome in outcomes {
        report.push_str(&format!(
            "- `{}` | domain=`{}` | pass=`{}` | overall={:.3} | keyword={:.3} | lexical={:.3} | intent={:.3} | retrieval={:.3}\n",
            outcome.case.id,
            outcome.case.domain,
            outcome.passed,
            outcome.overall_score,
            outcome.keyword_score,
            outcome.lexical_score,
            outcome.intent_score,
            outcome.retrieval_score
        ));
    }

    report
}

fn setup_summary(case: &BenchmarkCase) -> String {
    let mut parts = Vec::new();
    if !case.setup_documents.is_empty() {
        parts.push(format!("documents={}", case.setup_documents.join(", ")));
    }
    if !case.setup_turns.is_empty() {
        parts.push(format!("turns={}", case.setup_turns.join(" || ")));
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join("; ")
    }
}

fn escape_md(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

fn to_title(text: &str) -> String {
    text.split('_')
        .map(|segment| {
            let mut chars = segment.chars();
            let first = chars
                .next()
                .map(|ch| ch.to_uppercase().to_string())
                .unwrap_or_default();
            let rest = chars.as_str().to_lowercase();
            format!("{first}{rest}")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn reference_terms(reference: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for term in normalize_terms(reference) {
        if !terms.contains(&term) {
            terms.push(term);
        }
        if terms.len() >= 4 {
            break;
        }
    }
    if terms.is_empty() {
        terms.push("spse".to_string());
    }
    terms
}

fn normalize_terms(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric())
                .trim_end_matches("'s")
                .to_string()
        })
        .filter(|token| token.len() > 2 || token.chars().any(|ch| ch.is_ascii_digit()))
        .filter(|token| !is_stopword(token))
        .map(|token| singularize(&token))
        .collect()
}

fn singularize(token: &str) -> String {
    if token.len() > 4 && token.ends_with('s') && !token.ends_with("ss") {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "this"
            | "that"
            | "with"
            | "from"
            | "into"
            | "when"
            | "what"
            | "does"
            | "main"
            | "more"
            | "they"
            | "only"
            | "through"
            | "while"
            | "than"
            | "then"
            | "after"
            | "before"
            | "their"
            | "about"
            | "which"
            | "where"
            | "should"
            | "would"
            | "could"
            | "briefly"
            | "supposed"
            | "active"
            | "document"
    )
}

fn jaccard(lhs: &[String], rhs: &[String]) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }

    let lhs_set = lhs.iter().cloned().collect::<Vec<_>>();
    let rhs_set = rhs.iter().cloned().collect::<Vec<_>>();
    let intersection = lhs_set.iter().filter(|term| rhs_set.contains(term)).count() as f32;
    let union = lhs_set.len() as f32 + rhs_set.len() as f32 - intersection;
    if union <= f32::EPSILON {
        0.0
    } else {
        intersection / union
    }
}
