use crate::types::{
    MemoryChannel, MemoryType, TrainingSource, TrainingSourceType, TrainingStreamConfig,
};

#[derive(Debug, Clone, Copy)]
pub struct OpenSourceDefinition {
    pub id: &'static str,
    pub label: &'static str,
    pub category: &'static str,
    pub license: &'static str,
    pub integration: &'static str,
    pub summary: &'static str,
    pub default_type: TrainingSourceType,
    pub default_value: Option<&'static str>,
    pub default_memory: MemoryType,
    pub default_item_limit: Option<usize>,
    pub default_batch_size: Option<usize>,
    pub default_chunk_char_limit: Option<usize>,
}

const OPEN_SOURCES: &[OpenSourceDefinition] = &[
    OpenSourceDefinition {
        id: "wikidata",
        label: "Wikidata",
        category: "core_kb",
        license: "CC0",
        integration: "wikidata_truthy,wikidata_search",
        summary: "Structured entity graph for factual grounding and routing anchors.",
        default_type: TrainingSourceType::WikidataTruthy,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: None,
        default_batch_size: Some(96),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "wikipedia",
        label: "Wikipedia",
        category: "core_kb",
        license: "CC-BY-SA",
        integration: "wikipedia_dump,url",
        summary: "Broad explanatory context and multilingual article coverage.",
        default_type: TrainingSourceType::WikipediaDump,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: None,
        default_batch_size: None,
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "dbpedia",
        label: "DBpedia",
        category: "core_kb",
        license: "CC-BY-SA / ODbL",
        integration: "dbpedia_dump,url",
        summary: "Typed relation extracts from Wikipedia for corroborated evidence merges.",
        default_type: TrainingSourceType::DbpediaDump,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: None,
        default_batch_size: Some(96),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "project_gutenberg",
        label: "Project Gutenberg",
        category: "corpus",
        license: "Public Domain (US)",
        integration: "project_gutenberg",
        summary: "Long-form public-domain text suited to silent corpus training.",
        default_type: TrainingSourceType::ProjectGutenberg,
        default_value: Some("https://www.gutenberg.org/cache/epub/11/pg11.txt"),
        default_memory: MemoryType::Core,
        default_item_limit: Some(4),
        default_batch_size: None,
        default_chunk_char_limit: Some(5_000),
    },
    OpenSourceDefinition {
        id: "common_crawl",
        label: "Common Crawl",
        category: "corpus",
        license: "Mixed / filter required",
        integration: "common_crawl_wet,open_web_text",
        summary: "Broad web coverage through filtered WET or parquet-based text snapshots.",
        default_type: TrainingSourceType::CommonCrawlWet,
        default_value: Some("https://commoncrawl.org/latest-crawl"),
        default_memory: MemoryType::Core,
        default_item_limit: Some(64),
        default_batch_size: None,
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "openwebtext",
        label: "OpenWebText",
        category: "corpus",
        license: "Mixed / filtered snapshot",
        integration: "open_web_text",
        summary: "Parquet-based web text snapshot with built-in shard discovery.",
        default_type: TrainingSourceType::OpenWebText,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(1_000),
        default_batch_size: Some(8),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "hf_fw_fineweb_edu",
        label: "HuggingFaceFW FineWeb-Edu",
        category: "corpus",
        license: "ODC-By",
        integration: "huggingface_dataset",
        summary: "Large educational web-text corpus hosted on Hugging Face for broad language and knowledge coverage.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://HuggingFaceFW/fineweb-edu?subset=default&split=train&row_mode=plain_text&text_fields=text",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(12_000),
        default_batch_size: Some(250),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "hf_karpathy_fineweb_edu",
        label: "karpathy FineWeb-Edu 100B Shuffle",
        category: "corpus",
        license: "ODC-By",
        integration: "huggingface_dataset",
        summary: "Karpathy-hosted shuffled educational web corpus for broad text coverage with different document ordering.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://karpathy/fineweb-edu-100b-shuffle?subset=default&split=train&row_mode=plain_text&text_fields=text",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(8_000),
        default_batch_size: Some(250),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "hf_karpathy_tinystories",
        label: "karpathy TinyStories GPT4 Clean",
        category: "corpus",
        license: "Apache-2.0",
        integration: "huggingface_dataset",
        summary: "Small clean story corpus for lightweight narrative and simple compositional text patterns.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://karpathy/tinystories-gpt4-clean?subset=default&split=train&row_mode=plain_text&text_fields=text",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(6_000),
        default_batch_size: Some(200),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "hf_tb_cosmopedia",
        label: "HuggingFaceTB Cosmopedia",
        category: "corpus",
        license: "Apache-2.0",
        integration: "huggingface_dataset",
        summary: "Synthetic textbook-style explanations suited to explanatory and reasoning-oriented memory formation.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://HuggingFaceTB/cosmopedia?subset=auto_math_text&split=train&row_mode=plain_text&text_fields=text",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(8_000),
        default_batch_size: Some(220),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "hf_tb_smoltalk2",
        label: "HuggingFaceTB SmolTalk2",
        category: "intent_dialogue",
        license: "Apache-2.0",
        integration: "huggingface_dataset",
        summary: "Instruction and dialogue data for intent handling, multi-turn structure, and assistant-style behavior.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://HuggingFaceTB/smoltalk2?subset=SFT&split=OpenHermes_2.5_no_think&row_mode=structured_json",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(5_000),
        default_batch_size: Some(180),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "hf_h4_ultrachat_200k",
        label: "HuggingFaceH4 UltraChat 200K",
        category: "intent_dialogue",
        license: "MIT",
        integration: "huggingface_dataset",
        summary: "Multi-turn assistant conversations useful for intent memory and dialogue continuity.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://HuggingFaceH4/ultrachat_200k?subset=default&split=train_sft&row_mode=structured_json",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(5_000),
        default_batch_size: Some(180),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "hf_openai_gsm8k",
        label: "openai GSM8K",
        category: "reasoning",
        license: "MIT",
        integration: "huggingface_dataset",
        summary: "Structured math reasoning examples for stepwise problem solving and reasoning memory.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://openai/gsm8k?subset=main&split=train&row_mode=plain_text&text_fields=question,answer",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(4_000),
        default_batch_size: Some(160),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "hf_karpathy_climbmix",
        label: "karpathy ClimbMix 400B Shuffle",
        category: "corpus",
        license: "ODC-By",
        integration: "huggingface_dataset",
        summary: "Large shuffled text corpus for broad text coverage and incremental language pattern formation.",
        default_type: TrainingSourceType::HuggingFaceDataset,
        default_value: Some(
            "hf://karpathy/climbmix-400b-shuffle?subset=default&split=train&row_mode=plain_text&text_fields=text",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(8_000),
        default_batch_size: Some(250),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "squad_v2",
        label: "SQuAD 2.0",
        category: "qa_dataset",
        license: "CC-BY-SA",
        integration: "structured_json",
        summary: "Question-answer supervision for uncertainty and answerability behavior.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(512),
        default_batch_size: Some(32),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "natural_questions",
        label: "Natural Questions",
        category: "qa_dataset",
        license: "CC-BY-SA",
        integration: "structured_json",
        summary: "Real-world question distribution useful for retrieval gating evaluation.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.train.jsonl",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "triviaqa",
        label: "TriviaQA",
        category: "qa_dataset",
        license: "Apache-2.0",
        integration: "structured_json",
        summary: "Evidence-backed QA examples for multi-source answer validation.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "opentriviaqa",
        label: "OpenTriviaQA",
        category: "qa_dataset",
        license: "CC-BY-SA 4.0",
        integration: "structured_json",
        summary: "Lightweight multiple-choice QA set for edge validation loops.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(128),
        default_batch_size: Some(16),
        default_chunk_char_limit: Some(5_000),
    },
    OpenSourceDefinition {
        id: "ms_marco",
        label: "MS MARCO",
        category: "qa_dataset",
        license: "MIT",
        integration: "structured_json",
        summary: "Query-passage data for retrieval-focused training and evaluation.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(512),
        default_batch_size: Some(32),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "gsm8k_train",
        label: "GSM8K (Train)",
        category: "reasoning",
        license: "MIT",
        integration: "structured_json",
        summary: "Stepwise grade-school math reasoning for procedural chain discovery.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(512),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "proofwriter",
        label: "ProofWriter",
        category: "reasoning",
        license: "CC-BY 4.0",
        integration: "structured_json",
        summary: "Logical deduction traces for implication and entailment pathways.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "ruletaker",
        label: "RuleTaker",
        category: "reasoning",
        license: "CC-BY 4.0",
        integration: "structured_json",
        summary: "Synthetic rule application tasks for deduction and contradiction patterns.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "strategyqa_train",
        label: "StrategyQA (Train)",
        category: "reasoning",
        license: "Apache-2.0",
        integration: "structured_json",
        summary: "Multi-hop yes-no reasoning to train decomposition and insufficiency detection.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://raw.githubusercontent.com/eladsegal/strategyqa/master/data/strategyqa/train.json",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "logiqa",
        label: "LogiQA",
        category: "reasoning",
        license: "MIT",
        integration: "structured_json",
        summary: "Logic problem contexts with answer options and explanatory patterns.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "reclor",
        label: "ReClor",
        category: "reasoning",
        license: "Custom / review required",
        integration: "structured_json",
        summary: "Logical reasoning benchmark; cataloged but excluded from automatic presets.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "oasst1",
        label: "OpenAssistant (OASST1)",
        category: "intent_dialogue",
        license: "Apache-2.0",
        integration: "structured_json",
        summary: "Conversation trees for turn-taking, instruction following, and response ranking.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(512),
        default_batch_size: Some(32),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "dryrun_intent_core",
        label: "DryRun Intent Core Dataset",
        category: "intent_dialogue",
        license: "Internal",
        integration: "structured_json",
        summary: "High-density intent classification dialogues with reasoning chains, multi-turn conversations, and tone/resolver labels for all 24 intent kinds.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/dryrun/dryrun_intent_core.jsonl"),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(100_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "dryrun_entity_seed",
        label: "DryRun Entity Seed Dataset",
        category: "core_kb",
        license: "Internal",
        integration: "structured_json",
        summary: "High-density entity definitions with rich attributes, cross-references, and domain contexts for core knowledge base seeding.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/dryrun/dryrun_entity_seed.jsonl"),
        default_memory: MemoryType::Core,
        default_item_limit: Some(50_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "dolly_15k",
        label: "Databricks Dolly 15K",
        category: "intent_dialogue",
        license: "CC-BY-SA 3.0",
        integration: "structured_json",
        summary: "Instruction-following examples across brainstorming, extraction, and generation tasks.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(512),
        default_batch_size: Some(32),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "openorca",
        label: "OpenOrca",
        category: "intent_dialogue",
        license: "MIT / ODC-By",
        integration: "structured_json",
        summary: "Instruction-response traces with rationale useful for evidence-backed answering.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(512),
        default_batch_size: Some(32),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "multiwoz_2_2_train",
        label: "MultiWOZ 2.2 (Train)",
        category: "intent_dialogue",
        license: "Apache-2.0",
        integration: "structured_json",
        summary: "Task-oriented multi-turn dialogue for booking, lookup, and slot-filling behavior.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some(
            "https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/train/dialogues_001.json",
        ),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "public_openapi_specs",
        label: "Public OpenAPI Specs",
        category: "action_procedure",
        license: "Mixed permissive / review required",
        integration: "open_api_spec",
        summary: "Structured API contracts for parameter, endpoint, and procedure learning.",
        default_type: TrainingSourceType::OpenApiSpec,
        default_value: Some(
            "https://github.com/dlt-hub/openapi-specs/archive/refs/heads/main.zip",
        ),
        default_memory: MemoryType::Core,
        default_item_limit: Some(128),
        default_batch_size: Some(12),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "open_license_repo",
        label: "Open License Repository",
        category: "action_procedure",
        license: "MIT / Apache / permissive review",
        integration: "code_repository",
        summary: "Code repositories with explicit logic and tool-use patterns.",
        default_type: TrainingSourceType::CodeRepository,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(12),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "wikihow",
        label: "WikiHow",
        category: "action_procedure",
        license: "CC-BY-NC / review required",
        integration: "structured_json,url",
        summary: "Step-by-step procedures; cataloged with a commercial-use license warning.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(256),
        default_batch_size: Some(24),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "gov_sops",
        label: "Government SOPs / Manuals",
        category: "action_procedure",
        license: "Public Domain (US Gov)",
        integration: "document,url,structured_json",
        summary: "High-trust procedural manuals suitable for core procedural memory.",
        default_type: TrainingSourceType::Document,
        default_value: None,
        default_memory: MemoryType::Core,
        default_item_limit: Some(128),
        default_batch_size: Some(16),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "wikimedia_rest",
        label: "Wikimedia REST API",
        category: "live_web",
        license: "CC-BY-SA",
        integration: "retrieval",
        summary: "Official summary and page endpoints for trusted live lookups.",
        default_type: TrainingSourceType::Url,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: None,
        default_batch_size: None,
        default_chunk_char_limit: None,
    },
    OpenSourceDefinition {
        id: "pubmed_central",
        label: "PubMed Central OA",
        category: "live_web",
        license: "Open access subset",
        integration: "retrieval",
        summary: "Academic retrieval path for scientific and medical questions.",
        default_type: TrainingSourceType::Url,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: None,
        default_batch_size: None,
        default_chunk_char_limit: None,
    },
    OpenSourceDefinition {
        id: "openstreetmap_nominatim",
        label: "OpenStreetMap Nominatim",
        category: "live_web",
        license: "ODbL",
        integration: "retrieval",
        summary: "Geospatial retrieval path for location-centric prompts.",
        default_type: TrainingSourceType::Url,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: None,
        default_batch_size: None,
        default_chunk_char_limit: None,
    },
    OpenSourceDefinition {
        id: "internet_archive",
        label: "Internet Archive",
        category: "live_web",
        license: "Mixed / filter required",
        integration: "trust_allowlist",
        summary: "Allowlisted archival domain for future retrieval and user-supplied URLs.",
        default_type: TrainingSourceType::Url,
        default_value: None,
        default_memory: MemoryType::Episodic,
        default_item_limit: None,
        default_batch_size: None,
        default_chunk_char_limit: None,
    },
];

pub fn catalog() -> &'static [OpenSourceDefinition] {
    OPEN_SOURCES
}

pub fn render_catalog_table() -> String {
    let mut lines = vec![
        "id | label | category | license | integration | reference".to_string(),
        "--- | --- | --- | --- | --- | ---".to_string(),
    ];
    for source in OPEN_SOURCES {
        lines.push(format!(
            "{} | {} | {} | {} | {} | {}",
            source.id,
            source.label,
            source.category,
            source.license,
            source.integration,
            reference_url(source.id).unwrap_or("-")
        ));
    }
    lines.join("\n")
}

pub fn reference_url(id: &str) -> Option<&'static str> {
    if id.eq_ignore_ascii_case("wikidata") {
        Some("https://www.wikidata.org/wiki/Wikidata:Database_download")
    } else if id.eq_ignore_ascii_case("wikipedia") {
        Some("https://dumps.wikimedia.org/enwiki/")
    } else if id.eq_ignore_ascii_case("dbpedia") {
        Some("https://downloads.dbpedia.org/")
    } else if id.eq_ignore_ascii_case("project_gutenberg") {
        Some("https://huggingface.co/datasets/Despina/project_gutenberg")
    } else if id.eq_ignore_ascii_case("common_crawl") {
        Some("https://commoncrawl.org/latest-crawl")
    } else if id.eq_ignore_ascii_case("openwebtext") {
        Some("https://huggingface.co/datasets/Skylion007/openwebtext")
    } else if id.eq_ignore_ascii_case("hf_fw_fineweb_edu") {
        Some("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu")
    } else if id.eq_ignore_ascii_case("hf_karpathy_fineweb_edu") {
        Some("https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle")
    } else if id.eq_ignore_ascii_case("hf_karpathy_tinystories") {
        Some("https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean")
    } else if id.eq_ignore_ascii_case("hf_tb_cosmopedia") {
        Some("https://huggingface.co/datasets/HuggingFaceTB/cosmopedia")
    } else if id.eq_ignore_ascii_case("hf_tb_smoltalk2") {
        Some("https://huggingface.co/datasets/HuggingFaceTB/smoltalk2")
    } else if id.eq_ignore_ascii_case("hf_h4_ultrachat_200k") {
        Some("https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k")
    } else if id.eq_ignore_ascii_case("hf_openai_gsm8k") {
        Some("https://huggingface.co/datasets/openai/gsm8k")
    } else if id.eq_ignore_ascii_case("hf_karpathy_climbmix") {
        Some("https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle")
    } else if id.eq_ignore_ascii_case("squad_v2") {
        Some("https://rajpurkar.github.io/SQuAD-explorer/")
    } else if id.eq_ignore_ascii_case("natural_questions") {
        Some("https://github.com/google-research-datasets/natural-questions")
    } else if id.eq_ignore_ascii_case("triviaqa") {
        Some("https://huggingface.co/datasets/mandarjoshi/trivia_qa")
    } else if id.eq_ignore_ascii_case("opentriviaqa") {
        Some("https://github.com/uberspot/OpenTriviaQA")
    } else if id.eq_ignore_ascii_case("ms_marco") {
        Some("https://github.com/microsoft/msmarco/blob/master/Datasets.md")
    } else if id.eq_ignore_ascii_case("gsm8k_train") {
        Some("https://huggingface.co/datasets/openai/gsm8k")
    } else if id.eq_ignore_ascii_case("proofwriter") {
        Some("https://opendatalab.com/OpenDataLab/ProofWriter/download")
    } else if id.eq_ignore_ascii_case("ruletaker") {
        Some("https://github.com/allenai/ruletaker")
    } else if id.eq_ignore_ascii_case("strategyqa_train") {
        Some("https://github.com/eladsegal/strategyqa")
    } else if id.eq_ignore_ascii_case("logiqa") {
        Some("https://github.com/lgw863/LogiQA-dataset")
    } else if id.eq_ignore_ascii_case("reclor") {
        Some("https://whyu.me/reclor/")
    } else if id.eq_ignore_ascii_case("oasst1") {
        Some("https://huggingface.co/datasets/OpenAssistant/oasst1")
    } else if id.eq_ignore_ascii_case("dryrun_intent_core") {
        Some("datasets/dryrun/dryrun_intent_core.jsonl")
    } else if id.eq_ignore_ascii_case("dryrun_entity_seed") {
        Some("datasets/dryrun/dryrun_entity_seed.jsonl")
    } else if id.eq_ignore_ascii_case("dolly_15k") {
        Some("https://huggingface.co/datasets/databricks/databricks-dolly-15k")
    } else if id.eq_ignore_ascii_case("openorca") {
        Some("https://huggingface.co/datasets/Open-Orca/OpenOrca")
    } else if id.eq_ignore_ascii_case("multiwoz_2_2_train") {
        Some("https://github.com/budzianowski/multiwoz")
    } else if id.eq_ignore_ascii_case("public_openapi_specs") {
        Some("https://github.com/dlt-hub/openapi-specs")
    } else if id.eq_ignore_ascii_case("open_license_repo") {
        Some("https://github.com")
    } else if id.eq_ignore_ascii_case("wikihow") {
        Some("https://github.com/mahnazkoupaee/WikiHow-Dataset")
    } else if id.eq_ignore_ascii_case("gov_sops") {
        Some("https://www.usa.gov/")
    } else if id.eq_ignore_ascii_case("wikimedia_rest") {
        Some("https://en.wikipedia.org/api/rest_v1/")
    } else if id.eq_ignore_ascii_case("pubmed_central") {
        Some("https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/")
    } else if id.eq_ignore_ascii_case("openstreetmap_nominatim") {
        Some("https://nominatim.openstreetmap.org/")
    } else if id.eq_ignore_ascii_case("internet_archive") {
        Some("https://scholar.archive.org/")
    } else {
        None
    }
}

pub fn resolve_training_source(source: &TrainingSource) -> Result<TrainingSource, String> {
    if matches!(source.source_type, TrainingSourceType::Dataset) && source.name.is_none() {
        return Err("dataset_source_requires_name".to_string());
    }

    let Some(definition) = source
        .name
        .as_deref()
        .map(|name| find_definition(name).ok_or_else(|| format!("unknown_open_source:{name}")))
        .transpose()?
    else {
        return Ok(source.clone());
    };

    let mut resolved = source.clone();
    if matches!(resolved.source_type, TrainingSourceType::Dataset) {
        resolved.source_type = definition.default_type;
    }
    if resolved.value.is_none() {
        resolved.value = definition.default_value.map(str::to_string);
    }
    if resolved.target_memory.is_none() {
        resolved.target_memory = Some(definition.default_memory);
    }
    if resolved.memory_channels.is_none() {
        resolved.memory_channels = Some(default_channels_for_definition(definition));
    }
    apply_stream_defaults(&mut resolved.stream, definition);
    if resolved.value.is_none()
        && resolved.content.is_none()
        && !matches!(
            resolved.source_type,
            TrainingSourceType::WikipediaDump
                | TrainingSourceType::WikidataTruthy
                | TrainingSourceType::OpenWebText
        )
    {
        return Err(format!("open_source_requires_value:{}", definition.id));
    }
    Ok(resolved)
}

fn apply_stream_defaults(stream: &mut TrainingStreamConfig, definition: OpenSourceDefinition) {
    if stream.item_limit.is_none() {
        stream.item_limit = definition.default_item_limit;
    }
    if stream.batch_size.is_none() {
        stream.batch_size = definition.default_batch_size;
    }
    if stream.chunk_char_limit.is_none() {
        stream.chunk_char_limit = definition.default_chunk_char_limit;
    }
}

fn find_definition(id: &str) -> Option<OpenSourceDefinition> {
    OPEN_SOURCES
        .iter()
        .find(|definition| definition.id.eq_ignore_ascii_case(id))
        .copied()
}

pub fn category_for_source(id: &str) -> Option<&'static str> {
    find_definition(id).map(|definition| definition.category)
}

fn default_channels_for_definition(definition: OpenSourceDefinition) -> Vec<MemoryChannel> {
    match definition.category {
        "intent_dialogue" => vec![MemoryChannel::Main, MemoryChannel::Intent],
        "reasoning" => vec![MemoryChannel::Main, MemoryChannel::Reasoning],
        _ => vec![MemoryChannel::Main],
    }
}

pub fn catalog_source(name: &str) -> TrainingSource {
    TrainingSource {
        source_type: TrainingSourceType::Dataset,
        name: Some(name.to_string()),
        value: None,
        mime: None,
        content: None,
        target_memory: None,
        memory_channels: None,
        stream: TrainingStreamConfig::default(),
    }
}
