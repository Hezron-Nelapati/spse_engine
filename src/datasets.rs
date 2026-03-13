use crate::config::{EngineConfig, SourcePolicyConfig};
use crate::layers::input;
use crate::open_sources;
use crate::types::{MemoryChannel, TrainingSource, TrainingSourceType, TrainingStreamConfig};
use bzip2::read::MultiBzDecoder;
use chrono::{DateTime, Utc};
use flate2::read::MultiGzDecoder;
use futures_util::StreamExt;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};
use quick_xml::events::Event;
use quick_xml::Reader;
use regex::Regex;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_LENGTH, RANGE, RETRY_AFTER};
use reqwest::{Client, StatusCode};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tokio::time::{sleep, Duration};
use zip::ZipArchive;

const DEFAULT_WIKIPEDIA_URL: &str =
    "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2";
const DEFAULT_WIKIDATA_TRUTHY_URL: &str =
    "https://dumps.wikimedia.org/wikidatawiki/entities/latest-truthy.nt.bz2";
const DEFAULT_OPENWEBTEXT_MANIFEST_URL: &str =
    "https://huggingface.co/api/datasets/Skylion007/openwebtext/parquet";
const HUGGINGFACE_ROWS_API_URL: &str = "https://datasets-server.huggingface.co/rows";
const HUGGINGFACE_SPLITS_API_URL: &str = "https://datasets-server.huggingface.co/splits";
const HUGGINGFACE_ROWS_API_MAX_LENGTH: usize = 100;
const DEFAULT_CACHE_DIR: &str = ".spse_cache/open_datasets";
const DEFAULT_PREPARED_MANIFEST: &str = ".spse_cache/prepared_sources.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetSinkControl {
    Continue,
    Halt,
}

#[derive(Debug, Clone)]
pub struct DatasetTextChunk {
    pub context_label: String,
    pub content: String,
}

#[derive(Debug, Clone, Default)]
pub struct DatasetStreamStats {
    pub emitted_chunks: u64,
    pub items_seen: u64,
    pub items_ingested: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedSourceRecord {
    pub key: String,
    pub source_name: Option<String>,
    pub source_type: TrainingSourceType,
    pub remote_value: Option<String>,
    pub local_value: String,
    #[serde(default)]
    pub shard_paths: Vec<String>,
    pub prepared_at: DateTime<Utc>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PreparedSourceManifest {
    #[serde(default)]
    sources: Vec<PreparedSourceRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreparationReport {
    pub scope: String,
    pub sources: Vec<PreparedSourceRecord>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
struct ActiveSourcePolicy {
    key: String,
    source: SourcePolicyConfig,
    block_patterns: Vec<Regex>,
    allow_patterns: Vec<Regex>,
    min_unique_words_ratio: f32,
    max_boilerplate_ratio: f32,
    value_only_output: bool,
}

impl ActiveSourcePolicy {
    fn allows_field(&self, key: &str) -> bool {
        if self.skips_field(key) {
            return false;
        }
        if self.source.fields_to_extract.is_empty() {
            return true;
        }
        self.source
            .fields_to_extract
            .iter()
            .any(|field| field.eq_ignore_ascii_case(key))
    }

    fn skips_field(&self, key: &str) -> bool {
        self.source
            .fields_to_skip
            .iter()
            .chain(self.source.skip.iter())
            .any(|field| field.eq_ignore_ascii_case(key))
    }

    fn allows_response_fields(&self) -> bool {
        [
            "answer",
            "answers",
            "wellFormedAnswers",
            "output",
            "completion",
            "target",
            "label",
            "final",
            "final_answer",
            "response",
        ]
        .iter()
        .any(|field| self.allows_field(field))
    }

    fn keep_text(&self, text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return false;
        }
        if self
            .block_patterns
            .iter()
            .any(|pattern| pattern.is_match(trimmed))
        {
            return false;
        }
        if !self.allow_patterns.is_empty()
            && !self
                .allow_patterns
                .iter()
                .any(|pattern| pattern.is_match(trimmed))
        {
            return false;
        }
        unique_word_ratio(trimmed) >= self.min_unique_words_ratio
    }
}

fn resolve_source_policy(config: &EngineConfig, source: &TrainingSource) -> ActiveSourcePolicy {
    let key = source_policy_key(source);
    let mut policy = match key.as_str() {
        "html" => config.source_policies.html.clone(),
        "qa_json" => config.source_policies.qa_json.clone(),
        "structured_json" => config.source_policies.structured_json.clone(),
        "plain_text" => config.source_policies.plain_text.clone(),
        "code" => config.source_policies.code.clone(),
        "wikipedia_xml" => config.source_policies.wikipedia_xml.clone(),
        "wikidata_truthy" => config.source_policies.wikidata_truthy.clone(),
        "openapi_spec" => config.source_policies.openapi_spec.clone(),
        "common_crawl_wet" => config.source_policies.common_crawl_wet.clone(),
        _ => config.source_policies.plain_text.clone(),
    };

    if let Some(rule) = config.silent_training.format_ingestion_rules.get(&key) {
        if !rule.ingest_fields.is_empty() {
            policy.fields_to_extract =
                merge_unique_strings(policy.fields_to_extract, rule.ingest_fields.clone());
        }
        if !rule.skip_fields.is_empty() {
            policy.fields_to_skip =
                merge_unique_strings(policy.fields_to_skip, rule.skip_fields.clone());
        }
        if !rule.extract.is_empty() {
            policy.extract = merge_unique_strings(policy.extract, rule.extract.clone());
        }
        if !rule.skip.is_empty() {
            policy.skip = merge_unique_strings(policy.skip, rule.skip.clone());
        }
        if policy.usage_tag.is_none() {
            policy.usage_tag = rule.use_as.clone();
        }
    }

    if let Some(placement) = config.silent_training.memory_placement.get(&key) {
        if placement.memory_type.is_some() {
            policy.memory_type = placement.memory_type;
        }
        if placement.decay_days.is_some() {
            policy.decay_days = placement.decay_days;
        }
        if placement.min_corroborations.is_some() {
            policy.min_corroborations = placement.min_corroborations;
        }
    }

    let block_patterns = config
        .trust
        .promotion_rules
        .block_patterns
        .iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect::<Vec<_>>();
    let allow_patterns = config
        .trust
        .promotion_rules
        .allow_patterns
        .iter()
        .filter_map(|pattern| Regex::new(pattern).ok())
        .collect::<Vec<_>>();

    ActiveSourcePolicy {
        key,
        source: policy,
        block_patterns,
        allow_patterns,
        min_unique_words_ratio: config
            .trust
            .content_quality_thresholds
            .min_unique_words_ratio,
        max_boilerplate_ratio: config
            .trust
            .content_quality_thresholds
            .max_boilerplate_ratio,
        value_only_output: matches!(source.source_type, TrainingSourceType::HuggingFaceDataset),
    }
}

fn source_policy_key(source: &TrainingSource) -> String {
    match source.source_type {
        TrainingSourceType::WikipediaDump => "wikipedia_xml".to_string(),
        TrainingSourceType::WikidataTruthy => "wikidata_truthy".to_string(),
        TrainingSourceType::OpenApiSpec => "openapi_spec".to_string(),
        TrainingSourceType::CodeRepository => "code".to_string(),
        TrainingSourceType::CommonCrawlWet => "common_crawl_wet".to_string(),
        TrainingSourceType::QaJson => "qa_json".to_string(),
        TrainingSourceType::HuggingFaceDataset => huggingface_policy_key(source),
        TrainingSourceType::ProjectGutenberg
        | TrainingSourceType::OpenWebText
        | TrainingSourceType::DbpediaDump => "plain_text".to_string(),
        TrainingSourceType::Document | TrainingSourceType::Url => classify_mime_policy(source),
        TrainingSourceType::StructuredJson => structured_json_policy_key(source),
        TrainingSourceType::Dataset => "plain_text".to_string(),
    }
}

fn huggingface_policy_key(source: &TrainingSource) -> String {
    if let Some(value) = source.value.as_deref() {
        if value.starts_with("hf://") {
            if let Ok(locator) = parse_huggingface_locator(value) {
                return match locator.row_mode {
                    HuggingFaceRowMode::PlainText => "plain_text".to_string(),
                    HuggingFaceRowMode::StructuredJson => structured_json_policy_key(source),
                };
            }
        } else if let Some(path) = existing_path(value) {
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(manifest) = serde_json::from_str::<HuggingFacePreparedManifest>(&raw) {
                    return match manifest.row_mode {
                        HuggingFaceRowMode::PlainText => "plain_text".to_string(),
                        HuggingFaceRowMode::StructuredJson => structured_json_policy_key(source),
                    };
                }
            }
        }
    }
    structured_json_policy_key(source)
}

fn structured_json_policy_key(source: &TrainingSource) -> String {
    if source
        .memory_channels
        .as_ref()
        .is_some_and(|channels| channels.contains(&MemoryChannel::Intent))
    {
        return "qa_json".to_string();
    }
    if let Some(name) = source.name.as_deref() {
        if let Some(category) = open_sources::category_for_source(name) {
            return match category {
                "qa_dataset" | "reasoning" | "intent_dialogue" => "qa_json".to_string(),
                "action_procedure" => "structured_json".to_string(),
                _ => "structured_json".to_string(),
            };
        }
    }
    "structured_json".to_string()
}

fn classify_mime_policy(source: &TrainingSource) -> String {
    if source
        .mime
        .as_deref()
        .is_some_and(|mime| mime.contains("html"))
        || source
            .value
            .as_deref()
            .is_some_and(|value| value.ends_with(".html") || value.ends_with(".htm"))
    {
        return "html".to_string();
    }
    "plain_text".to_string()
}

fn merge_unique_strings(mut base: Vec<String>, additions: Vec<String>) -> Vec<String> {
    for value in additions {
        if !base
            .iter()
            .any(|existing| existing.eq_ignore_ascii_case(&value))
        {
            base.push(value);
        }
    }
    base
}

pub async fn stream_training_source<F>(
    source: &TrainingSource,
    config: &EngineConfig,
    mut sink: F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let policy = resolve_source_policy(config, source);
    match source.source_type {
        TrainingSourceType::StructuredJson => {
            stream_structured_json(source, &policy, &mut sink).await
        }
        TrainingSourceType::HuggingFaceDataset => {
            stream_huggingface_dataset(source, config, &policy, &mut sink).await
        }
        TrainingSourceType::OpenApiSpec => stream_openapi_spec(source, &mut sink).await,
        TrainingSourceType::CodeRepository => {
            stream_code_repository(source, &policy, &mut sink).await
        }
        TrainingSourceType::WikipediaDump => stream_wikipedia_dump(source, &mut sink).await,
        TrainingSourceType::WikidataTruthy => stream_wikidata_truthy(source, &mut sink).await,
        TrainingSourceType::OpenWebText => stream_openwebtext(source, &mut sink).await,
        TrainingSourceType::DbpediaDump => stream_dbpedia_dump(source, &mut sink).await,
        TrainingSourceType::ProjectGutenberg => stream_project_gutenberg(source, &mut sink).await,
        TrainingSourceType::CommonCrawlWet => {
            stream_common_crawl_wet(source, &policy, &mut sink).await
        }
        TrainingSourceType::QaJson => stream_qa_json(source, &policy, &mut sink).await,
        other => Err(format!("unsupported dataset source type: {other:?}")),
    }
}

pub fn effective_memory_type(
    source: &TrainingSource,
    config: &EngineConfig,
) -> Option<crate::types::MemoryType> {
    resolve_source_policy(config, source).source.memory_type
}

pub async fn prepare_training_source(
    source: &TrainingSource,
) -> Result<PreparedSourceRecord, String> {
    if matches!(source.source_type, TrainingSourceType::HuggingFaceDataset)
        && source
            .value
            .as_deref()
            .is_some_and(|value| value.starts_with("hf://"))
    {
        let client = dataset_client()?;
        let remote = resolve_huggingface_remote_dataset(source, &client).await?;
        let record = PreparedSourceRecord {
            key: prepared_source_key(source),
            source_name: source.name.clone(),
            source_type: source.source_type,
            remote_value: source.value.clone(),
            local_value: source
                .value
                .clone()
                .unwrap_or_else(|| format!("hf://{}", remote.repo_id)),
            shard_paths: Vec::new(),
            prepared_at: Utc::now(),
            size_bytes: 0,
        };
        persist_prepared_source_record(source, record.clone())?;
        return Ok(record);
    }

    let artifact = materialize_prepared_artifact(source).await?;
    let record = PreparedSourceRecord {
        key: prepared_source_key(source),
        source_name: source.name.clone(),
        source_type: source.source_type,
        remote_value: source.value.clone(),
        local_value: artifact.local_value.display().to_string(),
        shard_paths: artifact
            .shard_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        prepared_at: Utc::now(),
        size_bytes: total_size_bytes(&artifact.local_value, &artifact.shard_paths),
    };
    persist_prepared_source_record(source, record.clone())?;
    Ok(record)
}

pub fn localize_prepared_source(source: &TrainingSource) -> Result<TrainingSource, String> {
    if source.value.as_deref().and_then(existing_path).is_some() || source.content.is_some() {
        return Ok(source.clone());
    }

    let Some(record) = load_prepared_source_record(source)? else {
        return Err(format!(
            "training_source_not_prepared:{}",
            source
                .name
                .as_deref()
                .unwrap_or_else(|| source.value.as_deref().unwrap_or("unnamed"))
        ));
    };
    let local_path = PathBuf::from(&record.local_value);
    if !local_path.exists() {
        return Err(format!(
            "prepared_source_missing:{}:{}",
            source
                .name
                .as_deref()
                .unwrap_or_else(|| source.value.as_deref().unwrap_or("unnamed")),
            local_path.display()
        ));
    }

    let mut localized = source.clone();
    localized.value = Some(record.local_value);
    Ok(localized)
}

pub fn localize_remote_capable_source(source: &TrainingSource) -> Result<TrainingSource, String> {
    if source.value.as_deref().and_then(existing_path).is_some() || source.content.is_some() {
        return Ok(source.clone());
    }

    if let Some(record) = load_prepared_source_record(source)? {
        let local_path = PathBuf::from(&record.local_value);
        if local_path.exists() {
            let mut localized = source.clone();
            localized.value = Some(record.local_value);
            return Ok(localized);
        }
    }

    match source.source_type {
        TrainingSourceType::HuggingFaceDataset => Ok(source.clone()),
        _ => localize_prepared_source(source),
    }
}

#[derive(Debug, Clone)]
struct PreparedArtifact {
    local_value: PathBuf,
    shard_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum HuggingFaceRowMode {
    PlainText,
    StructuredJson,
}

#[derive(Debug, Clone)]
struct HuggingFaceDatasetLocator {
    repo_id: String,
    subset: Option<String>,
    split: Option<String>,
    row_mode: HuggingFaceRowMode,
    text_fields: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HuggingFacePreparedManifest {
    repo_id: String,
    subset: Option<String>,
    split: String,
    row_mode: HuggingFaceRowMode,
    text_fields: Vec<String>,
    shard_urls: Vec<String>,
}

#[derive(Debug, Clone)]
struct HuggingFaceRemoteDataset {
    repo_id: String,
    config: String,
    split: String,
    row_mode: HuggingFaceRowMode,
    text_fields: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSplitsResponse {
    #[serde(default)]
    splits: Vec<HuggingFaceSplitEntry>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceSplitEntry {
    config: String,
    split: String,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRowsResponse {
    #[serde(default)]
    rows: Vec<HuggingFaceRowsEntry>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceRowsEntry {
    row: Value,
}

async fn stream_wikipedia_dump<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_local_file(
        source,
        DEFAULT_WIKIPEDIA_URL,
        "wikipedia_dump",
        source.stream.cache_dir.as_deref(),
    )
    .await?;
    let max_items = source.stream.item_limit;
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let reader = open_text_reader(&path)?;
    let mut xml = Reader::from_reader(reader);
    xml.trim_text(true);

    let mut buf = Vec::new();
    let mut path_stack: Vec<Vec<u8>> = Vec::new();
    let mut title = String::new();
    let mut namespace = String::new();
    let mut body = String::new();
    let mut redirect = false;

    loop {
        match xml.read_event_into(&mut buf) {
            Ok(Event::Start(event)) => {
                let name = event.name().as_ref().to_vec();
                if name.as_slice() == b"page" {
                    title.clear();
                    namespace.clear();
                    body.clear();
                    redirect = false;
                }
                if name.as_slice() == b"redirect"
                    && path_stack.last().map(|segment| segment.as_slice()) == Some(b"page")
                {
                    redirect = true;
                }
                path_stack.push(name);
            }
            Ok(Event::Empty(event)) => {
                if event.name().as_ref() == b"redirect"
                    && path_stack.last().map(|segment| segment.as_slice()) == Some(b"page")
                {
                    redirect = true;
                }
            }
            Ok(Event::Text(event)) => {
                let text = event
                    .unescape()
                    .map_err(|err| format!("failed to decode wikipedia text: {err}"))?
                    .into_owned();
                if matches_path(&path_stack, &[b"page", b"title"]) {
                    title.push_str(&text);
                } else if matches_path(&path_stack, &[b"page", b"ns"]) {
                    namespace.push_str(&text);
                } else if matches_path(&path_stack, &[b"page", b"revision", b"text"]) {
                    body.push_str(&text);
                }
            }
            Ok(Event::CData(event)) => {
                if matches_path(&path_stack, &[b"page", b"revision", b"text"]) {
                    body.push_str(&String::from_utf8_lossy(event.as_ref()));
                }
            }
            Ok(Event::End(event)) => {
                if event.name().as_ref() == b"page" {
                    if namespace.trim() == "0" && !redirect {
                        let cleaned = clean_wikipedia_markup(&body);
                        if !cleaned.is_empty() {
                            stats.items_seen += 1;
                            let control = emit_document_chunks(
                                format!("wikipedia:{}", title.trim()),
                                &cleaned,
                                chunk_char_limit,
                                sink,
                                &mut stats,
                            )?;
                            if control == DatasetSinkControl::Halt {
                                break;
                            }
                            if max_items.is_some_and(|limit| stats.items_seen >= limit as u64) {
                                break;
                            }
                        }
                    }
                }
                path_stack.pop();
            }
            Ok(Event::Eof) => break,
            Err(err) => return Err(format!("failed to parse wikipedia xml: {err}")),
            _ => {}
        }
        buf.clear();
    }

    Ok(stats)
}

async fn stream_wikidata_truthy<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_local_file(
        source,
        DEFAULT_WIKIDATA_TRUTHY_URL,
        "wikidata_truthy",
        source.stream.cache_dir.as_deref(),
    )
    .await?;
    let reader = open_text_reader(&path)?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(96).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator =
        TextBatchAccumulator::new("wikidata_truthy".to_string(), batch_size, chunk_char_limit);

    for line in reader.lines() {
        if stats.items_seen as usize >= max_items {
            break;
        }
        let line = line.map_err(|err| format!("failed to read wikidata dump: {err}"))?;
        let Some(sentence) = wikidata_sentence_from_triple(&line) else {
            continue;
        };
        stats.items_seen += 1;
        let control = accumulator.push(sentence, sink, &mut stats)?;
        if control == DatasetSinkControl::Halt {
            return Ok(stats);
        }
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_openwebtext<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let client = dataset_client()?;
    let mut shard_refs = resolve_openwebtext_shards(source, &client).await?;
    if let Some(limit) = source.stream.shard_limit {
        shard_refs.truncate(limit);
    }

    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(8).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(8_000).max(512);
    let cache_dir = resolve_cache_dir(&source.stream);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator =
        TextBatchAccumulator::new("openwebtext".to_string(), batch_size, chunk_char_limit);

    for shard in shard_refs {
        if stats.items_seen as usize >= max_items {
            break;
        }

        let local_path = match shard {
            ShardRef::Local(path) => path,
            ShardRef::Remote(url) => download_to_cache(&client, &url, &cache_dir).await?,
        };

        let file = File::open(&local_path).map_err(|err| {
            format!(
                "failed to open parquet shard {}: {err}",
                local_path.display()
            )
        })?;
        let reader = SerializedFileReader::new(file).map_err(|err| {
            format!(
                "failed to read parquet shard {}: {err}",
                local_path.display()
            )
        })?;
        let rows = reader.get_row_iter(None).map_err(|err| {
            format!(
                "failed to iterate parquet rows {}: {err}",
                local_path.display()
            )
        })?;

        for row in rows {
            if stats.items_seen as usize >= max_items {
                break;
            }
            let row = row.map_err(|err| {
                format!(
                    "failed to decode parquet row from {}: {err}",
                    local_path.display()
                )
            })?;
            let Some(text) = extract_openwebtext_text(&row) else {
                continue;
            };
            let normalized = input::normalize_text(&text);
            if normalized.is_empty() {
                continue;
            }
            stats.items_seen += 1;
            let control = accumulator.push(normalized, sink, &mut stats)?;
            if control == DatasetSinkControl::Halt {
                return Ok(stats);
            }
        }
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_dbpedia_dump<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_required_local_file(source, "dbpedia_dump").await?;
    let reader = open_text_reader(&path)?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(96).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator =
        TextBatchAccumulator::new("dbpedia_dump".to_string(), batch_size, chunk_char_limit);

    for line in reader.lines() {
        if stats.items_seen as usize >= max_items {
            break;
        }
        let line = line.map_err(|err| format!("failed to read dbpedia dump: {err}"))?;
        let Some(sentence) = dbpedia_sentence_from_triple(&line) else {
            continue;
        };
        stats.items_seen += 1;
        let control = accumulator.push(sentence, sink, &mut stats)?;
        if control == DatasetSinkControl::Halt {
            return Ok(stats);
        }
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_project_gutenberg<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_required_local_file(source, "project_gutenberg").await?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let raw = read_text_file(&path)?;
    let cleaned = strip_project_gutenberg_boilerplate(&raw);
    if cleaned.is_empty() {
        stats
            .warnings
            .push(format!("project_gutenberg_empty:{}", path.display()));
        return Ok(stats);
    }
    for (index, chunk) in split_text_blocks(&cleaned, chunk_char_limit)
        .into_iter()
        .enumerate()
    {
        if index >= max_items {
            break;
        }
        stats.items_seen += 1;
        stats.items_ingested += 1;
        stats.emitted_chunks += 1;
        let control = sink(DatasetTextChunk {
            context_label: format!("project_gutenberg:{}", display_label(&path)),
            content: chunk,
        })?;
        if control == DatasetSinkControl::Halt {
            return Ok(stats);
        }
    }
    Ok(stats)
}

async fn stream_common_crawl_wet<F>(
    source: &TrainingSource,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_common_crawl_input(source).await?;
    if is_common_crawl_manifest_path(&path) {
        return stream_common_crawl_manifest(&path, source, policy, sink).await;
    }
    stream_common_crawl_wet_file(
        &path,
        policy,
        source.stream.item_limit,
        source.stream.chunk_char_limit,
        sink,
    )
}

fn stream_common_crawl_wet_file<F>(
    path: &Path,
    policy: &ActiveSourcePolicy,
    item_limit: Option<usize>,
    chunk_char_limit: Option<usize>,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let reader = open_text_reader(&path)?;
    let max_items = item_limit.unwrap_or(usize::MAX);
    let chunk_char_limit = chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut headers = Vec::new();
    let mut body = String::new();
    let mut in_headers = false;
    let mut in_body = false;

    for line in reader.lines() {
        let line = line.map_err(|err| format!("failed to read common crawl wet: {err}"))?;
        if line.trim() == "WARC/1.0" {
            let control = flush_common_crawl_record(
                &headers,
                &body,
                policy,
                chunk_char_limit,
                sink,
                &mut stats,
            )?;
            if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
                return Ok(stats);
            }
            headers.clear();
            body.clear();
            in_headers = true;
            in_body = false;
            continue;
        }

        if in_headers {
            if line.trim().is_empty() {
                in_headers = false;
                in_body = true;
                continue;
            }
            headers.push(line);
            continue;
        }

        if in_body {
            body.push_str(&line);
            body.push('\n');
        }
    }

    let control =
        flush_common_crawl_record(&headers, &body, policy, chunk_char_limit, sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_common_crawl_manifest<F>(
    path: &Path,
    source: &TrainingSource,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let raw = read_text_file(path)?;
    let shard_limit = source.stream.shard_limit.unwrap_or(1).max(1);
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let chunk_char_limit = source.stream.chunk_char_limit;
    let client = dataset_client()?;
    let cache_dir = resolve_cache_dir(&source.stream);
    let mut stats = DatasetStreamStats::default();

    for shard_path in raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(shard_limit)
    {
        if stats.items_seen as usize >= max_items {
            break;
        }

        let url = if looks_like_url(shard_path) {
            shard_path.to_string()
        } else {
            format!("https://data.commoncrawl.org/{shard_path}")
        };
        let local_path = download_to_cache(&client, &url, &cache_dir).await?;
        let remaining_items = max_items.saturating_sub(stats.items_seen as usize);
        let shard_stats = stream_common_crawl_wet_file(
            &local_path,
            policy,
            Some(remaining_items),
            chunk_char_limit,
            sink,
        )?;
        stats.emitted_chunks += shard_stats.emitted_chunks;
        stats.items_seen += shard_stats.items_seen;
        stats.items_ingested += shard_stats.items_ingested;
        stats.warnings.extend(shard_stats.warnings);
    }

    Ok(stats)
}

async fn stream_qa_json<F>(
    source: &TrainingSource,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    stream_structured_json(source, policy, sink).await
}

async fn stream_huggingface_dataset<F>(
    source: &TrainingSource,
    config: &EngineConfig,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let client = dataset_client()?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut rows_per_pull = source
        .stream
        .batch_size
        .unwrap_or(config.huggingface_streaming.initial_rows_per_pull)
        .clamp(
            config.huggingface_streaming.min_rows_per_pull,
            config.huggingface_streaming.max_rows_per_pull,
        );
    let mut stats = DatasetStreamStats::default();
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| "missing_huggingface_dataset_value".to_string())?;
    let mut accumulator = if value.starts_with("hf://") {
        let remote = resolve_huggingface_remote_dataset(source, &client).await?;
        let mut accumulator = TextBatchAccumulator::new(
            format!(
                "huggingface:{}:{}:{}",
                remote.repo_id, remote.config, remote.split
            ),
            rows_per_pull,
            chunk_char_limit,
        );
        stream_huggingface_rows_api(
            source,
            config,
            policy,
            sink,
            &client,
            &remote,
            &mut rows_per_pull,
            &mut accumulator,
            &mut stats,
            max_items,
        )
        .await?;
        accumulator
    } else {
        let manifest = resolve_huggingface_manifest(source, &client).await?;
        let cache_dir = resolve_cache_dir(&source.stream);
        let mut accumulator = TextBatchAccumulator::new(
            format!(
                "huggingface:{}:{}:{}",
                manifest.repo_id,
                manifest.subset.as_deref().unwrap_or("default"),
                manifest.split
            ),
            rows_per_pull,
            chunk_char_limit,
        );

        let shard_iter = manifest
            .shard_urls
            .iter()
            .take(source.stream.shard_limit.unwrap_or(usize::MAX));

        for shard_url in shard_iter {
            if stats.items_seen as usize >= max_items {
                break;
            }

            let started = Instant::now();
            let local_path = if let Some(path) = existing_path(shard_url) {
                path
            } else {
                download_to_cache(&client, shard_url, &cache_dir).await?
            };
            let download_ms = started.elapsed().as_millis() as u64;
            rows_per_pull = adapt_huggingface_rows_per_pull(
                rows_per_pull,
                download_ms,
                &config.huggingface_streaming,
            );
            accumulator.set_batch_size(rows_per_pull);

            let file = File::open(&local_path).map_err(|err| {
                format!(
                    "failed to open huggingface parquet shard {}: {err}",
                    local_path.display()
                )
            })?;
            let reader = SerializedFileReader::new(file).map_err(|err| {
                format!(
                    "failed to read huggingface parquet shard {}: {err}",
                    local_path.display()
                )
            })?;
            let rows = reader.get_row_iter(None).map_err(|err| {
                format!(
                    "failed to iterate huggingface parquet rows {}: {err}",
                    local_path.display()
                )
            })?;

            for row in rows {
                if stats.items_seen as usize >= max_items {
                    break;
                }
                let row = row.map_err(|err| {
                    format!(
                        "failed to decode huggingface parquet row from {}: {err}",
                        local_path.display()
                    )
                })?;
                let examples = huggingface_row_examples(&row, &manifest, policy);
                for example in examples {
                    if stats.items_seen as usize >= max_items {
                        break;
                    }
                    if example.is_empty() {
                        continue;
                    }
                    stats.items_seen += 1;
                    let control = accumulator.push(example, sink, &mut stats)?;
                    if control == DatasetSinkControl::Halt {
                        return Ok(stats);
                    }
                }
            }
        }
        accumulator
    };

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_huggingface_rows_api<F>(
    source: &TrainingSource,
    config: &EngineConfig,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
    client: &Client,
    remote: &HuggingFaceRemoteDataset,
    rows_per_pull: &mut usize,
    accumulator: &mut TextBatchAccumulator,
    stats: &mut DatasetStreamStats,
    max_items: usize,
) -> Result<(), String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let mut offset = 0usize;
    loop {
        if stats.items_seen as usize >= max_items {
            break;
        }

        let started = Instant::now();
        let logical_target = (*rows_per_pull).min(max_items.max(1));
        let mut fetched_rows = 0usize;
        let mut exhausted = false;

        while fetched_rows < logical_target && (stats.items_seen as usize) < max_items {
            let request_len = (logical_target - fetched_rows)
                .min(HUGGINGFACE_ROWS_API_MAX_LENGTH)
                .max(1);
            if fetched_rows > 0 && config.huggingface_streaming.request_delay_ms > 0 {
                sleep(Duration::from_millis(
                    config.huggingface_streaming.request_delay_ms,
                ))
                .await;
            }
            let page = fetch_huggingface_rows_page(
                client,
                remote,
                offset,
                request_len,
                &config.huggingface_streaming,
            )
            .await?;
            if page.rows.is_empty() {
                exhausted = true;
                break;
            }

            let page_count = page.rows.len();
            for entry in page.rows {
                if stats.items_seen as usize >= max_items {
                    break;
                }
                let examples = huggingface_value_examples(
                    &entry.row,
                    remote.row_mode.clone(),
                    &remote.text_fields,
                    policy,
                );
                for example in examples {
                    if stats.items_seen as usize >= max_items {
                        break;
                    }
                    if example.is_empty() {
                        continue;
                    }
                    stats.items_seen += 1;
                    let control = accumulator.push(example, sink, stats)?;
                    if control == DatasetSinkControl::Halt {
                        return Ok(());
                    }
                }
            }

            offset += page_count;
            fetched_rows += page_count;
            if page_count < request_len {
                exhausted = true;
                break;
            }
        }

        let pull_ms = started.elapsed().as_millis() as u64;
        *rows_per_pull =
            adapt_huggingface_rows_per_pull(*rows_per_pull, pull_ms, &config.huggingface_streaming);
        accumulator.set_batch_size(*rows_per_pull);

        if exhausted {
            break;
        }
    }

    let _ = source;
    Ok(())
}

async fn stream_structured_json<F>(
    source: &TrainingSource,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_required_local_file(source, "structured_json").await?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(16).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator = TextBatchAccumulator::new(
        format!("structured_json:{}", display_label(&path)),
        batch_size,
        chunk_char_limit,
    );

    if is_zip_path(&path) {
        let file = File::open(&path).map_err(|err| {
            format!(
                "failed to open structured json zip {}: {err}",
                path.display()
            )
        })?;
        let mut archive = ZipArchive::new(file)
            .map_err(|err| format!("failed to read zip {}: {err}", path.display()))?;
        for index in 0..archive.len() {
            let mut entry = archive
                .by_index(index)
                .map_err(|err| format!("failed to read zip entry {}: {err}", path.display()))?;
            if !entry.is_file() {
                continue;
            }
            let entry_name = entry.name().to_ascii_lowercase();
            if !entry_name.ends_with(".json") && !entry_name.ends_with(".jsonl") {
                continue;
            }

            let mut raw = String::new();
            entry.read_to_string(&mut raw).map_err(|err| {
                format!(
                    "failed to read structured json entry {} from {}: {err}",
                    entry.name(),
                    path.display()
                )
            })?;
            let entry_path = PathBuf::from(entry.name());
            let examples =
                collect_structured_examples_from_text(&entry_path, &raw, &mut stats, policy)?;
            for example in examples {
                if stats.items_seen as usize >= max_items {
                    break;
                }
                if example.is_empty() {
                    continue;
                }
                stats.items_seen += 1;
                let control = accumulator.push(example, sink, &mut stats)?;
                if control == DatasetSinkControl::Halt {
                    return Ok(stats);
                }
            }
            if stats.items_seen as usize >= max_items {
                break;
            }
        }
    } else {
        let raw = read_text_file(&path)?;
        let examples = collect_structured_examples_from_text(&path, &raw, &mut stats, policy)?;
        for example in examples {
            if stats.items_seen as usize >= max_items {
                break;
            }
            if example.is_empty() {
                continue;
            }
            stats.items_seen += 1;
            let control = accumulator.push(example, sink, &mut stats)?;
            if control == DatasetSinkControl::Halt {
                return Ok(stats);
            }
        }
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    Ok(stats)
}

async fn stream_openapi_spec<F>(
    source: &TrainingSource,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_required_local_file(source, "open_api_spec").await?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(12).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator = TextBatchAccumulator::new(
        format!("openapi:{}", display_label(&path)),
        batch_size,
        chunk_char_limit,
    );

    let control = if path.is_dir() {
        stream_openapi_directory(&path, max_items, sink, &mut stats, &mut accumulator)?
    } else if is_zip_path(&path) {
        stream_openapi_zip(&path, max_items, sink, &mut stats, &mut accumulator)?
    } else {
        let raw = read_text_file(&path)?;
        emit_openapi_from_text(
            &display_label(&path),
            &raw,
            max_items,
            sink,
            &mut stats,
            &mut accumulator,
        )?
    };

    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    if stats.items_seen == 0 {
        stats
            .warnings
            .push(format!("openapi_no_operations_found:{}", path.display()));
    }
    Ok(stats)
}

async fn stream_code_repository<F>(
    source: &TrainingSource,
    policy: &ActiveSourcePolicy,
    sink: &mut F,
) -> Result<DatasetStreamStats, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let path = resolve_required_local_file(source, "code_repository").await?;
    let max_items = source.stream.item_limit.unwrap_or(usize::MAX);
    let batch_size = source.stream.batch_size.unwrap_or(12).max(1);
    let chunk_char_limit = source.stream.chunk_char_limit.unwrap_or(6_000).max(512);
    let mut stats = DatasetStreamStats::default();
    let mut accumulator = TextBatchAccumulator::new(
        format!("code_repository:{}", display_label(&path)),
        batch_size,
        chunk_char_limit,
    );

    let control = if path.is_dir() {
        stream_repository_directory(
            &path,
            &path,
            policy,
            max_items,
            sink,
            &mut stats,
            &mut accumulator,
        )?
    } else if is_zip_path(&path) {
        stream_repository_zip(&path, policy, max_items, sink, &mut stats, &mut accumulator)?
    } else {
        let bytes = fs::read(&path)
            .map_err(|err| format!("failed to read repository file {}: {err}", path.display()))?;
        emit_repository_blob(
            &display_label(&path),
            &bytes,
            policy,
            max_items,
            sink,
            &mut stats,
            &mut accumulator,
        )?
    };

    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }

    let control = accumulator.flush(sink, &mut stats)?;
    if control == DatasetSinkControl::Halt {
        return Ok(stats);
    }
    if stats.items_seen == 0 {
        stats.warnings.push(format!(
            "code_repository_no_textual_files:{}",
            path.display()
        ));
    }
    Ok(stats)
}

fn emit_document_chunks<F>(
    context_label: String,
    text: &str,
    chunk_char_limit: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    for chunk in split_text_blocks(text, chunk_char_limit) {
        if chunk.is_empty() {
            continue;
        }
        stats.items_ingested += 1;
        stats.emitted_chunks += 1;
        let control = sink(DatasetTextChunk {
            context_label: context_label.clone(),
            content: chunk,
        })?;
        if control == DatasetSinkControl::Halt {
            return Ok(control);
        }
    }
    Ok(DatasetSinkControl::Continue)
}

struct TextBatchAccumulator {
    context_label: String,
    batch_size: usize,
    chunk_char_limit: usize,
    items: Vec<String>,
    chars: usize,
}

impl TextBatchAccumulator {
    fn new(context_label: String, batch_size: usize, chunk_char_limit: usize) -> Self {
        Self {
            context_label,
            batch_size,
            chunk_char_limit,
            items: Vec::new(),
            chars: 0,
        }
    }

    fn push<F>(
        &mut self,
        text: String,
        sink: &mut F,
        stats: &mut DatasetStreamStats,
    ) -> Result<DatasetSinkControl, String>
    where
        F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
    {
        for chunk in split_text_blocks(&text, self.chunk_char_limit) {
            if chunk.is_empty() {
                continue;
            }
            if !self.items.is_empty()
                && (self.items.len() >= self.batch_size
                    || self.chars + chunk.len() + 2 > self.chunk_char_limit)
            {
                let control = self.flush(sink, stats)?;
                if control == DatasetSinkControl::Halt {
                    return Ok(control);
                }
            }

            self.chars += chunk.len() + 2;
            self.items.push(chunk);
            stats.items_ingested += 1;

            if self.items.len() >= self.batch_size || self.chars >= self.chunk_char_limit {
                let control = self.flush(sink, stats)?;
                if control == DatasetSinkControl::Halt {
                    return Ok(control);
                }
            }
        }

        Ok(DatasetSinkControl::Continue)
    }

    fn flush<F>(
        &mut self,
        sink: &mut F,
        stats: &mut DatasetStreamStats,
    ) -> Result<DatasetSinkControl, String>
    where
        F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
    {
        if self.items.is_empty() {
            return Ok(DatasetSinkControl::Continue);
        }

        let content = self.items.join("\n\n");
        self.items.clear();
        self.chars = 0;
        stats.emitted_chunks += 1;

        sink(DatasetTextChunk {
            context_label: self.context_label.clone(),
            content,
        })
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size.max(1);
    }
}

#[derive(Debug, Clone)]
enum ShardRef {
    Local(PathBuf),
    Remote(String),
}

async fn resolve_huggingface_manifest(
    source: &TrainingSource,
    client: &Client,
) -> Result<HuggingFacePreparedManifest, String> {
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| "missing_huggingface_dataset_value".to_string())?;
    if let Some(path) = existing_path(value) {
        let raw = fs::read_to_string(&path).map_err(|err| {
            format!(
                "failed to read huggingface manifest {}: {err}",
                path.display()
            )
        })?;
        return serde_json::from_str::<HuggingFacePreparedManifest>(&raw).map_err(|err| {
            format!(
                "failed to parse huggingface manifest {}: {err}",
                path.display()
            )
        });
    }

    let locator = parse_huggingface_locator(value)?;
    let parquet_api_url = format!(
        "https://huggingface.co/api/datasets/{}/parquet",
        locator.repo_id
    );
    let value = client
        .get(&parquet_api_url)
        .send()
        .await
        .map_err(|err| {
            format!("failed to fetch huggingface parquet manifest {parquet_api_url}: {err}")
        })?
        .error_for_status()
        .map_err(|err| {
            format!("huggingface parquet manifest request failed {parquet_api_url}: {err}")
        })?
        .json::<Value>()
        .await
        .map_err(|err| {
            format!("failed to parse huggingface parquet manifest {parquet_api_url}: {err}")
        })?;
    build_huggingface_manifest_from_api(locator, value)
}

async fn resolve_huggingface_remote_dataset(
    source: &TrainingSource,
    client: &Client,
) -> Result<HuggingFaceRemoteDataset, String> {
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| "missing_huggingface_dataset_value".to_string())?;
    let locator = parse_huggingface_locator(value)?;
    if let (Some(config), Some(split)) = (locator.subset.clone(), locator.split.clone()) {
        return Ok(HuggingFaceRemoteDataset {
            repo_id: locator.repo_id,
            config,
            split,
            row_mode: locator.row_mode,
            text_fields: locator.text_fields,
        });
    }

    let splits = fetch_huggingface_splits(client, &locator.repo_id).await?;
    let config = locator
        .subset
        .clone()
        .or_else(|| splits.first().map(|entry| entry.config.clone()))
        .ok_or_else(|| {
            format!(
                "huggingface rows api returned no configs for {}",
                locator.repo_id
            )
        })?;

    let split = locator.split.clone().unwrap_or_else(|| {
        splits
            .iter()
            .find(|entry| entry.config == config && entry.split == "train")
            .or_else(|| {
                splits
                    .iter()
                    .find(|entry| entry.config == config && entry.split == "train_sft")
            })
            .or_else(|| splits.iter().find(|entry| entry.config == config))
            .or_else(|| splits.first())
            .map(|entry| entry.split.clone())
            .unwrap_or_else(|| "train".to_string())
    });

    Ok(HuggingFaceRemoteDataset {
        repo_id: locator.repo_id,
        config,
        split,
        row_mode: locator.row_mode,
        text_fields: locator.text_fields,
    })
}

async fn fetch_huggingface_splits(
    client: &Client,
    repo_id: &str,
) -> Result<Vec<HuggingFaceSplitEntry>, String> {
    let response: HuggingFaceSplitsResponse = send_huggingface_json_with_retry(
        || {
            client
                .get(HUGGINGFACE_SPLITS_API_URL)
                .query(&[("dataset", repo_id)])
        },
        &crate::config::HuggingFaceStreamingConfig::default(),
        || format!("huggingface splits request failed for {repo_id}"),
    )
    .await?;
    if response.splits.is_empty() {
        return Err(format!(
            "huggingface splits api returned no splits for {repo_id}"
        ));
    }
    Ok(response.splits)
}

async fn fetch_huggingface_rows_page(
    client: &Client,
    remote: &HuggingFaceRemoteDataset,
    offset: usize,
    length: usize,
    streaming: &crate::config::HuggingFaceStreamingConfig,
) -> Result<HuggingFaceRowsResponse, String> {
    send_huggingface_json_with_retry(
        || {
            client.get(HUGGINGFACE_ROWS_API_URL).query(&[
                ("dataset", remote.repo_id.as_str()),
                ("config", remote.config.as_str()),
                ("split", remote.split.as_str()),
                ("offset", &offset.to_string()),
                (
                    "length",
                    &length.min(HUGGINGFACE_ROWS_API_MAX_LENGTH).to_string(),
                ),
            ])
        },
        streaming,
        || {
            format!(
                "huggingface rows request failed for {}/{}/{} at offset {}",
                remote.repo_id, remote.config, remote.split, offset
            )
        },
    )
    .await
}

async fn send_huggingface_json_with_retry<T, F>(
    request_builder: F,
    streaming: &crate::config::HuggingFaceStreamingConfig,
    context: impl Fn() -> String,
) -> Result<T, String>
where
    T: DeserializeOwned,
    F: Fn() -> reqwest::RequestBuilder,
{
    let mut attempt = 0_u32;
    loop {
        let response = request_builder()
            .send()
            .await
            .map_err(|err| format!("{}: {err}", context()))?;
        let status = response.status();
        if status.is_success() {
            return response
                .json::<T>()
                .await
                .map_err(|err| format!("failed to parse response for {}: {err}", context()));
        }

        let should_retry = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
        if should_retry && attempt < streaming.max_retries {
            let delay_ms = retry_delay_ms(&response, attempt, streaming);
            sleep(Duration::from_millis(delay_ms)).await;
            attempt += 1;
            continue;
        }

        let detail = response
            .text()
            .await
            .unwrap_or_else(|_| "<response body unavailable>".to_string());
        return Err(format!("{}: HTTP {} {}", context(), status, detail));
    }
}

fn retry_delay_ms(
    response: &reqwest::Response,
    attempt: u32,
    streaming: &crate::config::HuggingFaceStreamingConfig,
) -> u64 {
    parse_retry_after_ms(response)
        .unwrap_or_else(|| {
            let multiplier = 2_u64.saturating_pow(attempt);
            streaming
                .retry_base_delay_ms
                .saturating_mul(multiplier.max(1))
        })
        .clamp(streaming.retry_base_delay_ms, streaming.retry_max_delay_ms)
}

fn parse_retry_after_ms(response: &reqwest::Response) -> Option<u64> {
    let value = response.headers().get(RETRY_AFTER)?.to_str().ok()?.trim();
    let seconds = value.parse::<u64>().ok()?;
    Some(seconds.saturating_mul(1_000))
}

fn parse_huggingface_locator(value: &str) -> Result<HuggingFaceDatasetLocator, String> {
    if !value.starts_with("hf://") {
        return Err(format!("invalid_huggingface_locator:{value}"));
    }
    let parsed = reqwest::Url::parse(value)
        .map_err(|err| format!("failed to parse huggingface locator {value}: {err}"))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| format!("huggingface locator missing owner: {value}"))?;
    let repo_path = parsed.path().trim_start_matches('/');
    if repo_path.is_empty() {
        return Err(format!("huggingface locator missing dataset name: {value}"));
    }
    let repo_id = format!("{host}/{repo_path}");
    let mut subset = None;
    let mut split = None;
    let mut row_mode = HuggingFaceRowMode::PlainText;
    let mut text_fields = Vec::new();
    for (key, raw_value) in parsed.query_pairs() {
        match key.as_ref() {
            "subset" => subset = Some(raw_value.into_owned()),
            "split" => split = Some(raw_value.into_owned()),
            "row_mode" => {
                row_mode = match raw_value.as_ref() {
                    "structured_json" => HuggingFaceRowMode::StructuredJson,
                    _ => HuggingFaceRowMode::PlainText,
                }
            }
            "text_fields" => {
                text_fields = raw_value
                    .split(',')
                    .map(str::trim)
                    .filter(|field| !field.is_empty())
                    .map(|field| field.to_string())
                    .collect();
            }
            _ => {}
        }
    }
    Ok(HuggingFaceDatasetLocator {
        repo_id,
        subset,
        split,
        row_mode,
        text_fields,
    })
}

fn build_huggingface_manifest_from_api(
    locator: HuggingFaceDatasetLocator,
    api_value: Value,
) -> Result<HuggingFacePreparedManifest, String> {
    let root = api_value
        .as_object()
        .ok_or_else(|| "huggingface parquet api did not return an object".to_string())?;

    let (subset_name, subset_value) = if let Some(subset) = locator.subset.as_deref() {
        let value = root
            .get(subset)
            .ok_or_else(|| format!("huggingface parquet api missing subset `{subset}`"))?;
        (Some(subset.to_string()), value)
    } else if let Some(default) = root.get("default") {
        (Some("default".to_string()), default)
    } else if let Some((name, value)) = root.iter().next() {
        (Some(name.to_string()), value)
    } else {
        return Err("huggingface parquet api returned no subsets".to_string());
    };

    let (split_name, split_value) = match subset_value {
        Value::Object(map) => {
            if let Some(split) = locator.split.as_deref() {
                let value = map
                    .get(split)
                    .ok_or_else(|| format!("huggingface parquet api missing split `{split}`"))?;
                (split.to_string(), value)
            } else if let Some(value) = map.get("train") {
                ("train".to_string(), value)
            } else if let Some(value) = map.get("train_sft") {
                ("train_sft".to_string(), value)
            } else if let Some((name, value)) = map.iter().next() {
                (name.to_string(), value)
            } else {
                return Err("huggingface parquet api returned an empty split map".to_string());
            }
        }
        Value::Array(_) => (
            locator.split.unwrap_or_else(|| "train".to_string()),
            subset_value,
        ),
        _ => {
            return Err("huggingface parquet api returned an unsupported subset shape".to_string());
        }
    };

    let mut shards = Vec::new();
    collect_huggingface_parquet_urls(split_value, &mut shards);
    if shards.is_empty() {
        return Err("huggingface parquet api did not contain shard urls".to_string());
    }

    Ok(HuggingFacePreparedManifest {
        repo_id: locator.repo_id,
        subset: subset_name,
        split: split_name,
        row_mode: locator.row_mode,
        text_fields: locator.text_fields,
        shard_urls: shards,
    })
}

fn collect_huggingface_parquet_urls(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if text.ends_with(".parquet") {
                output.push(text.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_huggingface_parquet_urls(item, output);
            }
        }
        Value::Object(map) => {
            if let Some(url) = map.get("url").and_then(Value::as_str) {
                if url.ends_with(".parquet") {
                    output.push(url.to_string());
                    return;
                }
            }
            if let Some(url) = map.get("parquet").and_then(Value::as_str) {
                if url.ends_with(".parquet") {
                    output.push(url.to_string());
                    return;
                }
            }
            for nested in map.values() {
                collect_huggingface_parquet_urls(nested, output);
            }
        }
        _ => {}
    }
}

fn adapt_huggingface_rows_per_pull(
    current: usize,
    download_ms: u64,
    config: &crate::config::HuggingFaceStreamingConfig,
) -> usize {
    let adjusted = if download_ms <= config.fast_pull_threshold_ms {
        ((current as f32) * config.increase_factor).round() as usize
    } else if download_ms >= config.slow_pull_threshold_ms {
        ((current as f32) * config.decrease_factor).round() as usize
    } else {
        current
    };
    adjusted.clamp(config.min_rows_per_pull, config.max_rows_per_pull)
}

fn huggingface_row_examples(
    row: &Row,
    manifest: &HuggingFacePreparedManifest,
    policy: &ActiveSourcePolicy,
) -> Vec<String> {
    let value = parquet_row_to_json_value(row);
    match manifest.row_mode {
        HuggingFaceRowMode::PlainText => {
            let direct =
                huggingface_plain_text_examples_from_row(row, &manifest.text_fields, policy);
            if direct.is_empty() {
                huggingface_plain_text_examples(&value, &manifest.text_fields, policy)
            } else {
                direct
            }
        }
        HuggingFaceRowMode::StructuredJson => {
            let mut examples = Vec::new();
            collect_structured_examples(&value, &mut examples, policy);
            examples.retain(|example| policy.keep_text(example));
            dedup_texts(&mut examples);
            examples
        }
    }
}

fn huggingface_value_examples(
    value: &Value,
    row_mode: HuggingFaceRowMode,
    fields: &[String],
    policy: &ActiveSourcePolicy,
) -> Vec<String> {
    match row_mode {
        HuggingFaceRowMode::PlainText => huggingface_plain_text_examples(value, fields, policy),
        HuggingFaceRowMode::StructuredJson => {
            let mut examples = Vec::new();
            collect_structured_examples(value, &mut examples, policy);
            examples.retain(|example| policy.keep_text(example));
            dedup_texts(&mut examples);
            examples
        }
    }
}

fn huggingface_plain_text_examples(
    value: &Value,
    fields: &[String],
    policy: &ActiveSourcePolicy,
) -> Vec<String> {
    let mut texts = Vec::new();
    if let Some(map) = value.as_object() {
        for field in fields {
            if !policy.allows_field(field) {
                continue;
            }
            if let Some(candidate) = map.get(field).and_then(extract_single_text) {
                if let Some(finalized) = finalize_policy_text(candidate, policy) {
                    texts.push(finalized);
                }
            }
        }

        if texts.is_empty() && fields.is_empty() {
            for (key, nested) in map {
                if !policy.allows_field(key) || !is_semantic_text_field(key) {
                    continue;
                }
                if let Some(candidate) = extract_single_text(nested) {
                    if let Some(finalized) = finalize_policy_text(candidate, policy) {
                        texts.push(finalized);
                    }
                }
            }
        }
    }
    dedup_texts(&mut texts);
    texts
}

fn huggingface_plain_text_examples_from_row(
    row: &Row,
    fields: &[String],
    policy: &ActiveSourcePolicy,
) -> Vec<String> {
    let mut texts = Vec::new();
    for field in fields {
        if !policy.allows_field(field) {
            continue;
        }
        if let Some(candidate) = row
            .get_column_iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(field))
            .and_then(|(_, value)| field_to_string(value))
        {
            if let Some(finalized) = finalize_policy_text(candidate, policy) {
                texts.push(finalized);
            }
        }
    }

    if texts.is_empty() && fields.is_empty() {
        for (name, value) in row.get_column_iter() {
            if !policy.allows_field(name) || !is_semantic_text_field(name) {
                continue;
            }
            if let Some(candidate) = field_to_string(value) {
                if let Some(finalized) = finalize_policy_text(candidate, policy) {
                    texts.push(finalized);
                }
            }
        }
    }

    dedup_texts(&mut texts);
    texts
}

fn is_semantic_text_field(field: &str) -> bool {
    matches!(
        field.to_ascii_lowercase().as_str(),
        "text"
            | "content"
            | "body"
            | "summary"
            | "description"
            | "document"
            | "article"
            | "passage"
            | "context"
            | "question"
            | "query"
            | "prompt"
            | "instruction"
            | "input"
            | "reasoning"
            | "rationale"
            | "explanation"
            | "analysis"
            | "solution"
            | "proof"
            | "message"
            | "messages"
            | "dialogue"
            | "conversation"
            | "turns"
    )
}

fn parquet_row_to_json_value(row: &Row) -> Value {
    Value::Object(
        row.get_column_iter()
            .map(|(name, field)| (name.to_string(), parquet_field_to_json_value(field)))
            .collect(),
    )
}

fn parquet_field_to_json_value(field: &Field) -> Value {
    match field {
        Field::Null => Value::Null,
        Field::Bool(value) => Value::Bool(*value),
        Field::Byte(value) => Value::from(*value),
        Field::Short(value) => Value::from(*value),
        Field::Int(value) => Value::from(*value),
        Field::Long(value) => Value::from(*value),
        Field::UByte(value) => Value::from(*value),
        Field::UShort(value) => Value::from(*value),
        Field::UInt(value) => Value::from(*value),
        Field::ULong(value) => Value::from(*value),
        Field::Float16(value) => Value::from(value.to_f32() as f64),
        Field::Float(value) => Value::from(*value as f64),
        Field::Double(value) => Value::from(*value),
        Field::Decimal(value) => Value::String(format!("{value:?}")),
        Field::Str(value) => Value::String(value.clone()),
        Field::Bytes(value) => Value::String(String::from_utf8_lossy(value.data()).into_owned()),
        Field::Date(value) => Value::from(*value),
        Field::TimestampMillis(value) => Value::from(*value),
        Field::TimestampMicros(value) => Value::from(*value),
        Field::Group(value) => parquet_row_to_json_value(value),
        Field::ListInternal(list) => Value::Array(
            list.elements()
                .iter()
                .map(parquet_field_to_json_value)
                .collect(),
        ),
        Field::MapInternal(map) => Value::Array(
            map.entries()
                .iter()
                .map(|(key, value)| {
                    Value::Object(
                        [
                            ("key".to_string(), parquet_field_to_json_value(key)),
                            ("value".to_string(), parquet_field_to_json_value(value)),
                        ]
                        .into_iter()
                        .collect(),
                    )
                })
                .collect(),
        ),
    }
}

async fn resolve_openwebtext_shards(
    source: &TrainingSource,
    client: &Client,
) -> Result<Vec<ShardRef>, String> {
    if let Some(value) = source.value.as_deref() {
        if let Some(path) = existing_path(value) {
            if path.is_dir() {
                let mut files = fs::read_dir(&path)
                    .map_err(|err| {
                        format!("failed to read openwebtext dir {}: {err}", path.display())
                    })?
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("parquet"))
                    .collect::<Vec<_>>();
                files.sort();
                return Ok(files.into_iter().map(ShardRef::Local).collect());
            }
            return resolve_manifest_like_path(&path);
        }

        if value.ends_with(".parquet") {
            return Ok(vec![ShardRef::Remote(value.to_string())]);
        }

        if value.ends_with(".json")
            || value.contains("/parquet")
            || value.contains("openwebtext")
            || value.contains("OpenWebText")
        {
            let manifest_url = normalize_openwebtext_manifest_url(value);
            return resolve_remote_manifest(&manifest_url, client).await;
        }
    }

    resolve_remote_manifest(DEFAULT_OPENWEBTEXT_MANIFEST_URL, client).await
}

fn resolve_manifest_like_path(path: &Path) -> Result<Vec<ShardRef>, String> {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "parquet" => Ok(vec![ShardRef::Local(path.to_path_buf())]),
        "json" => {
            let raw = fs::read_to_string(path)
                .map_err(|err| format!("failed to read manifest {}: {err}", path.display()))?;
            parse_manifest_shards(&raw, Some(path))
        }
        _ => {
            let raw = fs::read_to_string(path)
                .map_err(|err| format!("failed to read manifest {}: {err}", path.display()))?;
            let shards = raw
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(|line| {
                    existing_path(line)
                        .map(ShardRef::Local)
                        .unwrap_or_else(|| ShardRef::Remote(line.to_string()))
                })
                .collect::<Vec<_>>();
            if shards.is_empty() {
                Err(format!(
                    "manifest {} did not contain any shard paths",
                    path.display()
                ))
            } else {
                Ok(shards)
            }
        }
    }
}

async fn resolve_remote_manifest(url: &str, client: &Client) -> Result<Vec<ShardRef>, String> {
    let body = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("failed to fetch openwebtext manifest {url}: {err}"))?
        .error_for_status()
        .map_err(|err| format!("openwebtext manifest request failed {url}: {err}"))?
        .text()
        .await
        .map_err(|err| format!("failed to read openwebtext manifest {url}: {err}"))?;
    parse_manifest_shards(&body, None)
}

fn parse_manifest_shards(raw: &str, manifest_path: Option<&Path>) -> Result<Vec<ShardRef>, String> {
    let value = serde_json::from_str::<Value>(raw)
        .map_err(|err| format!("failed to parse openwebtext manifest json: {err}"))?;
    let mut shards = Vec::new();
    collect_manifest_shards(&value, manifest_path, &mut shards);
    if shards.is_empty() {
        Err("openwebtext manifest did not contain any shard references".to_string())
    } else {
        Ok(shards)
    }
}

fn collect_manifest_shards(
    value: &Value,
    manifest_path: Option<&Path>,
    shards: &mut Vec<ShardRef>,
) {
    match value {
        Value::String(text) => {
            if text.ends_with(".parquet") {
                if let Some(path) = manifest_path {
                    let joined = path
                        .parent()
                        .map(|parent| parent.join(text))
                        .unwrap_or_else(|| PathBuf::from(text));
                    if joined.exists() {
                        shards.push(ShardRef::Local(joined));
                        return;
                    }
                }
                shards.push(ShardRef::Remote(text.to_string()));
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_manifest_shards(item, manifest_path, shards);
            }
        }
        Value::Object(map) => {
            if let Some(url) = map.get("url").and_then(Value::as_str) {
                if url.ends_with(".parquet") {
                    shards.push(ShardRef::Remote(url.to_string()));
                    return;
                }
            }
            if let Some(path) = map.get("path").and_then(Value::as_str) {
                if path.ends_with(".parquet") {
                    if let Some(manifest_path) = manifest_path {
                        let joined = manifest_path
                            .parent()
                            .map(|parent| parent.join(path))
                            .unwrap_or_else(|| PathBuf::from(path));
                        if joined.exists() {
                            shards.push(ShardRef::Local(joined));
                            return;
                        }
                    }
                    shards.push(ShardRef::Remote(path.to_string()));
                    return;
                }
            }
            for nested in map.values() {
                collect_manifest_shards(nested, manifest_path, shards);
            }
        }
        _ => {}
    }
}

fn extract_openwebtext_text(row: &Row) -> Option<String> {
    for (name, field) in row.get_column_iter() {
        if matches!(name.as_str(), "text" | "content" | "body") {
            if let Some(text) = field_to_string(field) {
                return Some(text);
            }
        }
    }

    for (_, field) in row.get_column_iter() {
        if let Some(text) = field_to_string(field) {
            return Some(text);
        }
    }

    None
}

fn field_to_string(field: &Field) -> Option<String> {
    match field {
        Field::Str(value) => Some(value.clone()),
        Field::Bytes(value) => Some(String::from_utf8_lossy(value.data()).into_owned()),
        _ => None,
    }
}

async fn resolve_required_local_file(
    source: &TrainingSource,
    label: &str,
) -> Result<PathBuf, String> {
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| format!("missing_{label}_value"))?;
    if let Some(path) = existing_path(value) {
        return Ok(path);
    }
    if looks_like_url(value) {
        let client = dataset_client()?;
        let cache_dir = resolve_cache_dir(&source.stream);
        return download_to_cache(&client, value, &cache_dir).await;
    }
    Err(format!("unsupported_{label}_value:{value}"))
}

async fn materialize_prepared_artifact(
    source: &TrainingSource,
) -> Result<PreparedArtifact, String> {
    match source.source_type {
        TrainingSourceType::HuggingFaceDataset => materialize_huggingface_manifest(source).await,
        TrainingSourceType::WikipediaDump => {
            let path = resolve_local_file(
                source,
                DEFAULT_WIKIPEDIA_URL,
                "wikipedia_dump",
                source.stream.cache_dir.as_deref(),
            )
            .await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::WikidataTruthy => {
            let path = resolve_local_file(
                source,
                DEFAULT_WIKIDATA_TRUTHY_URL,
                "wikidata_truthy",
                source.stream.cache_dir.as_deref(),
            )
            .await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::CommonCrawlWet => {
            let manifest = resolve_common_crawl_input(source).await?;
            if is_common_crawl_manifest_path(&manifest) {
                materialize_common_crawl_manifest(source, &manifest).await
            } else {
                Ok(PreparedArtifact {
                    local_value: manifest,
                    shard_paths: Vec::new(),
                })
            }
        }
        TrainingSourceType::OpenWebText => materialize_openwebtext_manifest(source).await,
        TrainingSourceType::StructuredJson => {
            let path = resolve_required_local_file(source, "structured_json").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::OpenApiSpec => {
            let path = resolve_required_local_file(source, "open_api_spec").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::CodeRepository => {
            let path = resolve_required_local_file(source, "code_repository").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::DbpediaDump => {
            let path = resolve_required_local_file(source, "dbpedia_dump").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::ProjectGutenberg => {
            let path = resolve_required_local_file(source, "project_gutenberg").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::QaJson => {
            let path = resolve_required_local_file(source, "qa_json").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::Url => {
            let path = resolve_required_local_file(source, "url").await?;
            Ok(PreparedArtifact {
                local_value: path,
                shard_paths: Vec::new(),
            })
        }
        TrainingSourceType::Document => {
            if let Some(value) = source.value.as_deref() {
                if let Some(path) = existing_path(value) {
                    return Ok(PreparedArtifact {
                        local_value: path,
                        shard_paths: Vec::new(),
                    });
                }
                if looks_like_url(value) {
                    let path = resolve_required_local_file(source, "document").await?;
                    return Ok(PreparedArtifact {
                        local_value: path,
                        shard_paths: Vec::new(),
                    });
                }
            }
            if let Some(content) = source.content.as_deref() {
                let cache_dir = resolve_cache_dir(&source.stream);
                fs::create_dir_all(&cache_dir).map_err(|err| {
                    format!("failed to create cache dir {}: {err}", cache_dir.display())
                })?;
                let label = source
                    .name
                    .as_deref()
                    .unwrap_or_else(|| source.mime.as_deref().unwrap_or("inline_document"));
                let extension = source
                    .mime
                    .as_deref()
                    .and_then(default_extension_for_mime)
                    .unwrap_or("txt");
                let path = cache_dir.join(format!(
                    "{}_inline.{}",
                    sanitized_cache_name(label),
                    extension
                ));
                fs::write(&path, content).map_err(|err| {
                    format!(
                        "failed to write prepared document {}: {err}",
                        path.display()
                    )
                })?;
                return Ok(PreparedArtifact {
                    local_value: path,
                    shard_paths: Vec::new(),
                });
            }
            Err("missing_document_value".to_string())
        }
        TrainingSourceType::Dataset => Err("dataset_source_requires_resolution".to_string()),
    }
}

async fn materialize_huggingface_manifest(
    source: &TrainingSource,
) -> Result<PreparedArtifact, String> {
    let client = dataset_client()?;
    let manifest = resolve_huggingface_manifest(source, &client).await?;
    let cache_dir = resolve_cache_dir(&source.stream);
    fs::create_dir_all(&cache_dir)
        .map_err(|err| format!("failed to create cache dir {}: {err}", cache_dir.display()))?;
    let label = source
        .name
        .as_deref()
        .map(sanitized_cache_name)
        .unwrap_or_else(|| sanitized_cache_name(&manifest.repo_id));
    let path = cache_dir.join(format!("huggingface_{label}.manifest.json"));
    let json = serde_json::to_string_pretty(&manifest).map_err(|err| {
        format!(
            "failed to serialize huggingface manifest {}: {err}",
            path.display()
        )
    })?;
    fs::write(&path, json).map_err(|err| {
        format!(
            "failed to write huggingface manifest {}: {err}",
            path.display()
        )
    })?;
    Ok(PreparedArtifact {
        local_value: path,
        shard_paths: Vec::new(),
    })
}

async fn materialize_common_crawl_manifest(
    source: &TrainingSource,
    manifest_path: &Path,
) -> Result<PreparedArtifact, String> {
    let raw = read_text_file(manifest_path)?;
    let shard_limit = source.stream.shard_limit.unwrap_or(1).max(1);
    let cache_dir = resolve_cache_dir(&source.stream);
    let client = dataset_client()?;
    let mut local_shards = Vec::new();

    for shard_path in raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(shard_limit)
    {
        let url = normalize_common_crawl_url(shard_path);
        local_shards.push(download_to_cache(&client, &url, &cache_dir).await?);
    }

    let manifest = write_prepared_manifest_file(&cache_dir, source, "common_crawl", &local_shards)?;
    Ok(PreparedArtifact {
        local_value: manifest,
        shard_paths: local_shards,
    })
}

async fn materialize_openwebtext_manifest(
    source: &TrainingSource,
) -> Result<PreparedArtifact, String> {
    let client = dataset_client()?;
    let mut shards = resolve_openwebtext_shards(source, &client).await?;
    if let Some(limit) = source.stream.shard_limit {
        shards.truncate(limit);
    }

    let cache_dir = resolve_cache_dir(&source.stream);
    let mut local_shards = Vec::new();
    for shard in shards {
        let local = match shard {
            ShardRef::Local(path) => path,
            ShardRef::Remote(url) => download_to_cache(&client, &url, &cache_dir).await?,
        };
        local_shards.push(local);
    }

    let manifest = write_prepared_manifest_file(&cache_dir, source, "openwebtext", &local_shards)?;
    Ok(PreparedArtifact {
        local_value: manifest,
        shard_paths: local_shards,
    })
}

fn write_prepared_manifest_file(
    cache_dir: &Path,
    source: &TrainingSource,
    prefix: &str,
    paths: &[PathBuf],
) -> Result<PathBuf, String> {
    fs::create_dir_all(cache_dir)
        .map_err(|err| format!("failed to create cache dir {}: {err}", cache_dir.display()))?;
    let name = source
        .name
        .as_deref()
        .map(sanitized_cache_name)
        .unwrap_or_else(|| prefix.to_string());
    let path = cache_dir.join(format!("{prefix}_{name}_local.manifest"));
    let body = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&path, body).map_err(|err| {
        format!(
            "failed to write prepared manifest {}: {err}",
            path.display()
        )
    })?;
    Ok(path)
}

fn default_extension_for_mime(mime: &str) -> Option<&'static str> {
    match mime {
        "text/plain" => Some("txt"),
        "application/json" => Some("json"),
        "application/yaml" | "text/yaml" => Some("yaml"),
        "application/xml" | "text/xml" => Some("xml"),
        "application/pdf" => Some("pdf"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("docx"),
        _ => None,
    }
}

fn read_text_file(path: &Path) -> Result<String, String> {
    let mut reader = open_text_reader(path)?;
    let mut text = String::new();
    reader
        .read_to_string(&mut text)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    Ok(text)
}

fn display_label(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
        .unwrap_or_else(|| path.to_string_lossy().to_string())
}

fn is_common_crawl_manifest_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.contains("wet.paths"))
        .unwrap_or(false)
}

async fn resolve_common_crawl_input(source: &TrainingSource) -> Result<PathBuf, String> {
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| "missing_common_crawl_wet_value".to_string())?;
    if let Some(path) = existing_path(value) {
        return Ok(path);
    }
    if !looks_like_url(value) {
        return Err(format!("unsupported_common_crawl_wet_value:{value}"));
    }

    let client = dataset_client()?;
    let cache_dir = resolve_cache_dir(&source.stream);
    let resolved = if is_common_crawl_index_url(value) {
        resolve_common_crawl_manifest_url(value, &client).await?
    } else {
        normalize_common_crawl_url(value)
    };
    download_to_cache(&client, &resolved, &cache_dir).await
}

fn is_common_crawl_index_url(value: &str) -> bool {
    value.contains("commoncrawl.org/latest-crawl") || value.contains("commoncrawl.org/blog/")
}

fn normalize_common_crawl_url(value: &str) -> String {
    if value.starts_with("https://") || value.starts_with("http://") {
        return value.to_string();
    }
    if let Some(path) = value.strip_prefix("s3://commoncrawl/") {
        return format!("https://data.commoncrawl.org/{path}");
    }
    if let Some(path) = value.strip_prefix('/') {
        return format!("https://data.commoncrawl.org/{path}");
    }
    if value.starts_with("crawl-data/") {
        return format!("https://data.commoncrawl.org/{value}");
    }
    value.to_string()
}

async fn resolve_common_crawl_manifest_url(url: &str, client: &Client) -> Result<String, String> {
    let page = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("failed to fetch common crawl index {url}: {err}"))?
        .error_for_status()
        .map_err(|err| format!("common crawl index request failed {url}: {err}"))?
        .text()
        .await
        .map_err(|err| format!("failed to read common crawl index {url}: {err}"))?;
    if let Some(manifest) = extract_common_crawl_manifest_url(&page) {
        return Ok(manifest);
    }

    if let Some(crawl_id) = extract_common_crawl_crawl_id(&page) {
        let manifest = format!("https://data.commoncrawl.org/crawl-data/{crawl_id}/wet.paths.gz");
        let response =
            client.head(&manifest).send().await.map_err(|err| {
                format!("failed to probe common crawl manifest {manifest}: {err}")
            })?;
        if response.status().is_success() {
            return Ok(manifest);
        }
    }

    Err(format!("common_crawl_manifest_not_found:{url}"))
}

fn extract_common_crawl_manifest_url(page: &str) -> Option<String> {
    static FULL_RE: OnceLock<Regex> = OnceLock::new();
    static PATH_RE: OnceLock<Regex> = OnceLock::new();

    let full = FULL_RE.get_or_init(|| {
        Regex::new(r#"https://data\.commoncrawl\.org/[^\s"'<>]+wet\.paths\.gz"#)
            .expect("valid common crawl manifest regex")
    });
    if let Some(found) = full.find(page) {
        return Some(found.as_str().to_string());
    }

    let path = PATH_RE.get_or_init(|| {
        Regex::new(r#"/crawl-data/[^\s"'<>]+wet\.paths\.gz"#)
            .expect("valid common crawl manifest path regex")
    });
    path.find(page)
        .map(|found| format!("https://data.commoncrawl.org{}", found.as_str()))
}

fn extract_common_crawl_crawl_id(page: &str) -> Option<String> {
    static CRAWL_ID_RE: OnceLock<Regex> = OnceLock::new();
    let crawl_id = CRAWL_ID_RE
        .get_or_init(|| Regex::new(r#"CC-MAIN-\d{4}-\d{2}"#).expect("valid common crawl id regex"));
    crawl_id.find(page).map(|found| found.as_str().to_string())
}

fn flush_common_crawl_record<F>(
    headers: &[String],
    body: &str,
    policy: &ActiveSourcePolicy,
    chunk_char_limit: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    if headers.is_empty() || body.trim().is_empty() {
        return Ok(DatasetSinkControl::Continue);
    }

    let warc_type = header_value(headers, "WARC-Type")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !warc_type.is_empty() && warc_type != "conversion" {
        return Ok(DatasetSinkControl::Continue);
    }

    let content_type = header_value(headers, "Content-Type")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !content_type.is_empty() && !content_type.contains("text/plain") {
        return Ok(DatasetSinkControl::Continue);
    }

    let cleaned = clean_common_crawl_payload(body, policy);
    if cleaned.split_whitespace().count() < 12 {
        return Ok(DatasetSinkControl::Continue);
    }

    stats.items_seen += 1;
    let label = header_value(headers, "WARC-Target-URI")
        .map(|uri| format!("common_crawl:{uri}"))
        .unwrap_or_else(|| "common_crawl:record".to_string());
    emit_document_chunks(label, &cleaned, chunk_char_limit, sink, stats)
}

fn header_value<'a>(headers: &'a [String], key: &str) -> Option<&'a str> {
    headers.iter().find_map(|header| {
        let (header_key, value) = header.split_once(':')?;
        (header_key.trim().eq_ignore_ascii_case(key)).then_some(value.trim())
    })
}

fn clean_common_crawl_payload(body: &str, policy: &ActiveSourcePolicy) -> String {
    let normalized = sanitize_text_for_policy(body, policy);
    if normalized.is_empty() || text_quality_score(&normalized) < 0.18 {
        return String::new();
    }
    normalized
}

fn text_quality_score(text: &str) -> f32 {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }

    let alpha_chars = text.chars().filter(|ch| ch.is_alphabetic()).count() as f32;
    let visible_chars = text.chars().filter(|ch| !ch.is_whitespace()).count() as f32;
    let alphabetic_ratio = if visible_chars > 0.0 {
        alpha_chars / visible_chars
    } else {
        0.0
    };

    let mut unique = Vec::new();
    for token in &tokens {
        let normalized = token
            .trim_matches(|ch: char| !ch.is_alphanumeric())
            .to_ascii_lowercase();
        if normalized.is_empty() || unique.contains(&normalized) {
            continue;
        }
        unique.push(normalized);
    }
    let unique_ratio = unique.len() as f32 / tokens.len() as f32;
    let punctuation_hits = text
        .chars()
        .filter(|ch| matches!(ch, '.' | '!' | '?' | ';' | ':'))
        .count() as f32;
    let punctuation_ratio = punctuation_hits / tokens.len() as f32;

    (0.45 * alphabetic_ratio) + (0.40 * unique_ratio) + (0.15 * punctuation_ratio.min(1.0))
}

fn unique_word_ratio(text: &str) -> f32 {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }

    let mut unique = Vec::new();
    for token in tokens {
        let normalized = token
            .trim_matches(|ch: char| !ch.is_alphanumeric())
            .to_ascii_lowercase();
        if normalized.is_empty() || unique.contains(&normalized) {
            continue;
        }
        unique.push(normalized);
    }
    unique.len() as f32 / text.split_whitespace().count().max(1) as f32
}

fn sanitize_text_for_policy(text: &str, policy: &ActiveSourcePolicy) -> String {
    let mut kept_lines = Vec::new();
    let mut blocked_lines = 0usize;
    let mut total_lines = 0usize;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        total_lines += 1;
        if looks_like_standalone_url_line(line) {
            blocked_lines += 1;
            continue;
        }
        if policy
            .block_patterns
            .iter()
            .any(|pattern| pattern.is_match(line))
        {
            blocked_lines += 1;
            continue;
        }
        kept_lines.push(line.to_string());
    }

    if total_lines > 0 {
        let boilerplate_ratio = blocked_lines as f32 / total_lines as f32;
        if boilerplate_ratio > policy.max_boilerplate_ratio {
            return String::new();
        }
    }

    let mut normalized = if policy.source.trim_lines {
        kept_lines.join("\n")
    } else {
        text.to_string()
    };

    normalized = strip_trailing_gsm8k_answer_marker(&normalized);

    if policy.source.skip_internal_ids {
        normalized = internal_id_regex()
            .replace_all(&normalized, "")
            .into_owned();
    }

    if policy.source.normalize_whitespace {
        normalized = input::normalize_text(&normalized);
    }

    normalized.trim().to_string()
}

fn strip_trailing_gsm8k_answer_marker(text: &str) -> String {
    let mut lines = text.lines().map(str::trim_end).collect::<Vec<_>>();
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    if lines
        .last()
        .is_some_and(|line| line.trim_start().starts_with("#### "))
    {
        lines.pop();
    }
    let joined = lines.join("\n");
    trailing_gsm8k_marker_regex()
        .replace(&joined, "")
        .trim()
        .to_string()
}

fn looks_like_standalone_url_line(line: &str) -> bool {
    let normalized = line.trim();
    if normalized.is_empty() {
        return false;
    }
    let lower = normalized.to_ascii_lowercase();
    let is_url =
        lower.starts_with("http://") || lower.starts_with("https://") || lower.starts_with("www.");
    if !is_url {
        return false;
    }
    normalized.split_whitespace().count() <= 3
}

fn trailing_gsm8k_marker_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\s*####\s+[^\n]+$").expect("valid trailing gsm8k marker regex")
    })
}

fn finalize_policy_text(text: String, policy: &ActiveSourcePolicy) -> Option<String> {
    let normalized = sanitize_text_for_policy(&text, policy);
    policy.keep_text(&normalized).then_some(normalized)
}

fn extract_repository_semantic_text(label: &str, raw: &str) -> String {
    let lowered = label.to_ascii_lowercase();
    if lowered.contains("readme")
        || lowered.ends_with(".md")
        || lowered.ends_with(".rst")
        || lowered.ends_with(".txt")
    {
        return raw.to_string();
    }

    let mut extracted = Vec::new();
    let mut in_block = false;
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("/*") || trimmed.starts_with("/**") {
            in_block = true;
            let comment = trimmed
                .trim_start_matches("/*")
                .trim_start_matches('*')
                .trim();
            if !comment.is_empty() {
                extracted.push(comment.to_string());
            }
            continue;
        }
        if in_block {
            let comment = trimmed
                .trim_end_matches("*/")
                .trim_start_matches('*')
                .trim();
            if !comment.is_empty() {
                extracted.push(comment.to_string());
            }
            if trimmed.ends_with("*/") {
                in_block = false;
            }
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("///")
            .or_else(|| trimmed.strip_prefix("//!"))
            .or_else(|| trimmed.strip_prefix("//"))
            .or_else(|| trimmed.strip_prefix('#'))
        {
            let comment = rest.trim();
            if !comment.is_empty() {
                extracted.push(comment.to_string());
            }
            continue;
        }
        if let Some(rest) = trimmed
            .strip_prefix("\"\"\"")
            .or_else(|| trimmed.strip_prefix("'''"))
        {
            let comment = rest.trim();
            if !comment.is_empty() {
                extracted.push(comment.to_string());
            }
        }
    }

    if extracted.is_empty() {
        raw.to_string()
    } else {
        extracted.join("\n")
    }
}

fn strip_project_gutenberg_boilerplate(text: &str) -> String {
    let mut started = false;
    let mut lines = Vec::new();
    for line in text.lines() {
        let upper = line.to_ascii_uppercase();
        if !started
            && (upper.contains("*** START OF THE PROJECT GUTENBERG EBOOK")
                || upper.contains("*** START OF THIS PROJECT GUTENBERG EBOOK"))
        {
            started = true;
            continue;
        }
        if started
            && (upper.contains("*** END OF THE PROJECT GUTENBERG EBOOK")
                || upper.contains("*** END OF THIS PROJECT GUTENBERG EBOOK"))
        {
            break;
        }
        if started {
            lines.push(line);
        }
    }

    let body = if started {
        lines.join("\n")
    } else {
        text.to_string()
    };
    input::normalize_text(&body)
}

fn is_json_lines_source(path: &Path, raw: &str) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("jsonl")
        || raw.lines().take(3).all(|line| {
            let trimmed = line.trim();
            trimmed.is_empty() || trimmed.starts_with('{')
        })
}

fn collect_structured_examples_from_text(
    path: &Path,
    raw: &str,
    stats: &mut DatasetStreamStats,
    policy: &ActiveSourcePolicy,
) -> Result<Vec<String>, String> {
    let mut examples = if is_json_lines_source(path, raw) {
        collect_structured_examples_from_jsonl(raw, stats, policy)?
    } else {
        let value = serde_json::from_str::<Value>(raw)
            .map_err(|err| format!("failed to parse structured json {}: {err}", path.display()))?;
        let mut examples = Vec::new();
        collect_structured_examples(&value, &mut examples, policy);
        examples
    };
    if examples.is_empty() {
        stats.warnings.push(format!(
            "structured_json_no_examples_found:{}",
            path.display()
        ));
    }
    examples.retain(|example| policy.keep_text(example));
    dedup_texts(&mut examples);
    Ok(examples)
}

fn collect_structured_examples_from_jsonl(
    raw: &str,
    stats: &mut DatasetStreamStats,
    policy: &ActiveSourcePolicy,
) -> Result<Vec<String>, String> {
    let mut examples = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = serde_json::from_str::<Value>(trimmed)
            .map_err(|err| format!("failed to parse structured jsonl row: {err}"))?;
        collect_structured_examples(&value, &mut examples, policy);
    }
    if examples.is_empty() {
        stats
            .warnings
            .push("structured_jsonl_no_examples_found".to_string());
    }
    examples.retain(|example| policy.keep_text(example));
    dedup_texts(&mut examples);
    Ok(examples)
}

fn dedup_texts(values: &mut Vec<String>) {
    let mut deduped = Vec::new();
    for value in values.drain(..) {
        if !value.is_empty() && !deduped.contains(&value) {
            deduped.push(value);
        }
    }
    *values = deduped;
}

fn collect_structured_examples(
    value: &Value,
    output: &mut Vec<String>,
    policy: &ActiveSourcePolicy,
) {
    match value {
        Value::Array(items) => {
            for item in items {
                collect_structured_examples(item, output, policy);
            }
        }
        Value::Object(map) => {
            if collect_parallel_qa_examples(map, output, policy) {
                return;
            }
            if collect_contextual_qa_examples(map, output, policy) {
                return;
            }

            let mut emitted = false;
            if let Some(example) = format_task_record(map, policy) {
                output.push(example);
                emitted = true;
            }
            if let Some(example) = format_dialogue_record(map, policy) {
                output.push(example);
                emitted = true;
            }
            if let Some(example) = format_procedure_record(map, policy) {
                output.push(example);
                emitted = true;
            }
            if !emitted {
                if let Some(example) = format_generic_record(map, policy) {
                    output.push(example);
                }
            }

            for nested in map.values() {
                collect_structured_examples(nested, output, policy);
            }
        }
        _ => {}
    }
}

fn collect_parallel_qa_examples(
    map: &serde_json::Map<String, Value>,
    output: &mut Vec<String>,
    policy: &ActiveSourcePolicy,
) -> bool {
    if !policy.allows_field("query") && !policy.allows_field("question") {
        return false;
    }
    let Some(queries) = map.get("query").and_then(Value::as_array) else {
        return false;
    };

    let answers = (!policy.skips_field("answers"))
        .then(|| map.get("answers").and_then(Value::as_array))
        .flatten();
    let well_formed_answers = (!policy.skips_field("wellFormedAnswers"))
        .then(|| map.get("wellFormedAnswers").and_then(Value::as_array))
        .flatten();
    let passage_texts = map
        .get("passages")
        .and_then(Value::as_object)
        .and_then(|passages| passages.get("passage_text"))
        .and_then(Value::as_array);

    for (index, query) in queries.iter().enumerate() {
        let question = extract_single_text(query).unwrap_or_default();
        if question.is_empty() {
            continue;
        }
        let answer = answers
            .and_then(|items| items.get(index))
            .and_then(extract_single_text)
            .or_else(|| {
                well_formed_answers
                    .and_then(|items| items.get(index))
                    .and_then(extract_single_text)
            })
            .unwrap_or_default();
        let context = passage_texts
            .and_then(|items| items.get(index))
            .and_then(extract_single_text);
        if let Some(example) = format_task_example(
            "Question",
            &question,
            "Answer",
            (!answer.is_empty()).then_some(answer.as_str()),
            context.as_deref(),
            None,
            None,
            policy,
        ) {
            output.push(example);
        }
    }

    true
}

fn collect_contextual_qa_examples(
    map: &serde_json::Map<String, Value>,
    output: &mut Vec<String>,
    policy: &ActiveSourcePolicy,
) -> bool {
    if !policy.allows_field("context") || !policy.allows_field("qas") {
        return false;
    }
    let Some(context) = map.get("context").and_then(Value::as_str) else {
        return false;
    };
    let Some(qas) = map.get("qas").and_then(Value::as_array) else {
        return false;
    };

    for qa in qas {
        if let Some(question) = qa.get("question").and_then(Value::as_str) {
            let answers = qa.get("answers").map(extract_text_list).unwrap_or_default();
            let fallback = qa
                .get("plausible_answers")
                .map(extract_text_list)
                .unwrap_or_default();
            let answer = answers
                .into_iter()
                .chain(fallback.into_iter())
                .find(|text| !text.is_empty())
                .unwrap_or_default();
            if let Some(example) = format_task_example(
                "Question",
                question,
                "Answer",
                (!answer.is_empty()).then_some(answer.as_str()),
                Some(context),
                None,
                None,
                policy,
            ) {
                output.push(example);
            }
        }
    }

    true
}

fn format_task_record(
    map: &serde_json::Map<String, Value>,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let prompt = extract_first_text_with_policy(
        map,
        &[
            "question",
            "query",
            "prompt",
            "instruction",
            "input",
            "task",
        ],
        policy,
    )?;
    let response = extract_first_text_with_policy(
        map,
        &[
            "answer",
            "answers",
            "wellFormedAnswers",
            "response",
            "output",
            "completion",
            "target",
            "label",
            "final",
            "final_answer",
        ],
        policy,
    );
    let context = extract_first_text_with_policy(
        map,
        &[
            "context",
            "passage",
            "evidence",
            "document",
            "support",
            "background",
            "article",
        ],
        policy,
    );
    let reasoning = extract_first_text_with_policy(
        map,
        &[
            "reasoning",
            "rationale",
            "explanation",
            "analysis",
            "solution",
            "proof",
        ],
        policy,
    );
    let options =
        extract_first_list_with_policy(map, &["options", "choices", "candidates"], policy);
    let prompt_label = if map.contains_key("instruction") {
        "Instruction"
    } else {
        "Question"
    };
    format_task_example(
        prompt_label,
        &prompt,
        "Answer",
        response.as_deref(),
        context.as_deref(),
        (!options.is_empty()).then_some(options.as_slice()),
        reasoning.as_deref(),
        policy,
    )
}

fn format_dialogue_record(
    map: &serde_json::Map<String, Value>,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let dialogue = extract_first_value_with_policy(
        map,
        &[
            "messages",
            "turns",
            "dialogue",
            "conversation",
            "conversations",
            "utterances",
            "thread",
        ],
        policy,
    )?;
    let lines = collect_dialogue_lines(dialogue);
    if lines.len() < 2 {
        return None;
    }

    let mut output = Vec::with_capacity(lines.len() + 2);
    if let Some(title) =
        extract_first_text_with_policy(map, &["title", "task", "domain", "intent"], policy)
    {
        output.push(title);
    }
    output.extend(lines.into_iter().take(12));
    finalize_policy_text(output.join("\n"), policy)
}

fn format_procedure_record(
    map: &serde_json::Map<String, Value>,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let steps_value = extract_first_value_with_policy(
        map,
        &[
            "steps",
            "instructions",
            "procedure",
            "actions",
            "workflow",
            "plan",
        ],
        policy,
    )?;
    let steps = extract_step_list(steps_value);
    if steps.len() < 2 {
        return None;
    }

    let title = extract_first_text_with_policy(
        map,
        &["title", "name", "task", "goal", "summary", "operationId"],
        policy,
    )
    .unwrap_or_else(|| "Procedure".to_string());
    let description = extract_first_text_with_policy(
        map,
        &["description", "summary", "context", "objective"],
        policy,
    );
    let mut lines = vec![title];
    if let Some(description) = description {
        lines.push(trim_word_window(&description, 80));
    }
    lines.push(steps.into_iter().take(10).collect::<Vec<_>>().join(" | "));
    finalize_policy_text(lines.join("\n"), policy)
}

fn format_generic_record(
    map: &serde_json::Map<String, Value>,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let mut lines = Vec::new();
    for key in [
        "title",
        "name",
        "summary",
        "description",
        "domain",
        "category",
        "intent",
        "tool",
        "method",
        "path",
        "endpoint",
        "operationId",
        "status",
        "result",
    ] {
        if !policy.allows_field(key) {
            continue;
        }
        if let Some(value) = map.get(key).and_then(extract_single_text) {
            let value = trim_word_window(&value, 60);
            if policy.value_only_output {
                lines.push(value);
            } else {
                lines.push(format!("{key}: {value}"));
            }
        }
    }

    if lines.is_empty() {
        for (key, value) in map.iter().take(8) {
            if !policy.allows_field(key) {
                continue;
            }
            if let Some(text) = extract_single_text(value) {
                let text = trim_word_window(&text, 40);
                if policy.value_only_output {
                    lines.push(text);
                } else {
                    lines.push(format!("{key}: {text}"));
                }
            }
        }
    }

    (lines.len() >= 2)
        .then(|| finalize_policy_text(lines.join("\n"), policy))
        .flatten()
}

fn collect_dialogue_lines(value: &Value) -> Vec<String> {
    match value {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                let map = item.as_object()?;
                let role = extract_first_text(map, &["role", "speaker", "from", "author"])
                    .unwrap_or_else(|| "speaker".to_string());
                let content =
                    extract_first_text(map, &["content", "text", "message", "utterance", "value"])?;
                Some(format!("{}: {}", role, trim_word_window(&content, 60)))
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn extract_step_list(value: &Value) -> Vec<String> {
    match value {
        Value::Array(items) => items
            .iter()
            .filter_map(|item| match item {
                Value::String(_) | Value::Number(_) | Value::Bool(_) => extract_single_text(item),
                Value::Object(map) => extract_first_text(
                    map,
                    &[
                        "step",
                        "instruction",
                        "text",
                        "action",
                        "description",
                        "content",
                    ],
                ),
                _ => None,
            })
            .map(|step| trim_word_window(&step, 40))
            .collect(),
        _ => extract_text_list(value)
            .into_iter()
            .map(|step| trim_word_window(&step, 40))
            .collect(),
    }
}

fn format_task_example(
    prompt_label: &str,
    prompt: &str,
    response_label: &str,
    response: Option<&str>,
    context: Option<&str>,
    options: Option<&[String]>,
    reasoning: Option<&str>,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let prompt = input::normalize_text(prompt);
    if prompt.is_empty() {
        return None;
    }

    let plain_segments = policy.value_only_output
        || policy.key == "qa_json"
        || policy.source.extraction_mode == "field_select";
    let mut lines = if plain_segments {
        vec![prompt.clone()]
    } else {
        vec![format!("{prompt_label}: {prompt}")]
    };
    if let Some(response) = response.filter(|_| policy.allows_response_fields()) {
        let response = input::normalize_text(response);
        if !response.is_empty() {
            if plain_segments {
                lines.push(response);
            } else {
                lines.push(format!("{response_label}: {response}"));
            }
        }
    }
    if let Some(options) = options {
        let options = options
            .iter()
            .map(|value| input::normalize_text(value))
            .filter(|value| !value.is_empty())
            .collect::<Vec<_>>();
        if !options.is_empty() {
            if plain_segments {
                lines.push(options.join(" | "));
            } else {
                lines.push(format!("Options: {}", options.join(" | ")));
            }
        }
    }
    if let Some(context) = context {
        let trimmed = trim_word_window(&input::normalize_text(context), 120);
        if !trimmed.is_empty() {
            if plain_segments {
                lines.push(trimmed);
            } else {
                lines.push(format!("Context: {trimmed}"));
            }
        }
    }
    if let Some(reasoning) = reasoning {
        let trimmed = trim_word_window(&input::normalize_text(reasoning), 100);
        if !trimmed.is_empty() {
            if plain_segments {
                lines.push(trimmed);
            } else {
                lines.push(format!("Reasoning: {trimmed}"));
            }
        }
    }
    finalize_policy_text(lines.join("\n"), policy)
}

fn extract_first_text_with_policy(
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    keys.iter()
        .filter(|key| policy.allows_field(key))
        .find_map(|key| map.get(*key))
        .and_then(extract_single_text)
}

fn extract_first_list_with_policy(
    map: &serde_json::Map<String, Value>,
    keys: &[&str],
    policy: &ActiveSourcePolicy,
) -> Vec<String> {
    keys.iter()
        .filter(|key| policy.allows_field(key))
        .find_map(|key| map.get(*key))
        .map(extract_text_list)
        .unwrap_or_default()
}

fn extract_first_value_with_policy<'a>(
    map: &'a serde_json::Map<String, Value>,
    keys: &[&str],
    policy: &ActiveSourcePolicy,
) -> Option<&'a Value> {
    keys.iter()
        .filter(|key| policy.allows_field(key))
        .find_map(|key| map.get(*key))
}

fn extract_first_text(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| map.get(*key))
        .and_then(extract_single_text)
}

fn extract_first_list(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Vec<String> {
    keys.iter()
        .find_map(|key| map.get(*key))
        .map(extract_text_list)
        .unwrap_or_default()
}

fn extract_single_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => {
            let normalized = input::normalize_text(text);
            (!normalized.is_empty()).then_some(normalized)
        }
        Value::Array(items) => items.iter().find_map(extract_single_text),
        Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("answer"))
            .or_else(|| map.get("value"))
            .or_else(|| map.get("content"))
            .and_then(extract_single_text),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(boolean) => Some(boolean.to_string()),
        Value::Null => None,
    }
}

fn extract_text_list(value: &Value) -> Vec<String> {
    match value {
        Value::Array(items) => items.iter().filter_map(extract_single_text).collect(),
        _ => extract_single_text(value).into_iter().collect(),
    }
}

fn trim_word_window(text: &str, max_words: usize) -> String {
    text.split_whitespace()
        .take(max_words)
        .collect::<Vec<_>>()
        .join(" ")
}

fn stream_openapi_directory<F>(
    root: &Path,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    for entry in fs::read_dir(root)
        .map_err(|err| format!("failed to read openapi dir {}: {err}", root.display()))?
    {
        let entry = entry
            .map_err(|err| format!("failed to read openapi entry in {}: {err}", root.display()))?;
        let path = entry.path();
        if path.is_dir() {
            if hidden_path_component(&path) {
                continue;
            }
            let control = stream_openapi_directory(&path, max_items, sink, stats, accumulator)?;
            if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
                return Ok(control);
            }
            continue;
        }
        let raw = match read_text_file(&path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        let label = path
            .strip_prefix(root)
            .ok()
            .map(|value| value.display().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let control = emit_openapi_from_text(&label, &raw, max_items, sink, stats, accumulator)?;
        if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
            return Ok(control);
        }
    }

    Ok(DatasetSinkControl::Continue)
}

fn stream_openapi_zip<F>(
    path: &Path,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let file = File::open(path)
        .map_err(|err| format!("failed to open archive {}: {err}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err| format!("failed to read archive {}: {err}", path.display()))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|err| format!("failed to read archive entry {}: {err}", path.display()))?;
        if !entry.is_file() {
            continue;
        }
        let mut raw = String::new();
        if entry.read_to_string(&mut raw).is_err() {
            continue;
        }
        let label = entry.name().to_string();
        let control = emit_openapi_from_text(&label, &raw, max_items, sink, stats, accumulator)?;
        if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
            return Ok(control);
        }
    }

    Ok(DatasetSinkControl::Continue)
}

fn emit_openapi_from_text<F>(
    label: &str,
    raw: &str,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let Some(spec) = parse_openapi_document(raw) else {
        return Ok(DatasetSinkControl::Continue);
    };
    let title = spec
        .get("info")
        .and_then(Value::as_object)
        .and_then(|info| extract_first_text(info, &["title", "summary", "description"]))
        .unwrap_or_else(|| label.to_string());
    let Some(paths) = spec.get("paths").and_then(Value::as_object) else {
        return Ok(DatasetSinkControl::Continue);
    };

    for (path_name, item) in paths {
        let Some(item_map) = item.as_object() else {
            continue;
        };
        for method in ["get", "post", "put", "patch", "delete", "head", "options"] {
            let Some(operation) = item_map.get(method).and_then(Value::as_object) else {
                continue;
            };
            if stats.items_seen as usize >= max_items {
                return Ok(DatasetSinkControl::Halt);
            }
            let Some(text) = format_openapi_operation(&title, path_name, method, operation) else {
                continue;
            };
            stats.items_seen += 1;
            let control = accumulator.push(text, sink, stats)?;
            if control == DatasetSinkControl::Halt {
                return Ok(control);
            }
        }
    }

    Ok(DatasetSinkControl::Continue)
}

fn parse_openapi_document(raw: &str) -> Option<Value> {
    let parsed_json = serde_json::from_str::<Value>(raw).ok();
    if let Some(value) = parsed_json {
        if is_openapi_document(&value) {
            return Some(value);
        }
    }

    let parsed_yaml = serde_yaml::from_str::<serde_yaml::Value>(raw).ok()?;
    let value = serde_json::to_value(parsed_yaml).ok()?;
    is_openapi_document(&value).then_some(value)
}

fn is_openapi_document(value: &Value) -> bool {
    value.get("openapi").is_some()
        || value.get("swagger").is_some()
        || value
            .get("paths")
            .and_then(Value::as_object)
            .is_some_and(|paths| !paths.is_empty())
}

fn format_openapi_operation(
    title: &str,
    path_name: &str,
    method: &str,
    operation: &serde_json::Map<String, Value>,
) -> Option<String> {
    let summary = extract_first_text(operation, &["summary", "description"]);
    let operation_id = extract_first_text(operation, &["operationId"]);
    let tags = extract_first_list(operation, &["tags"]);
    let parameters = operation
        .get("parameters")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(format_openapi_parameter)
                .take(8)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let request = operation
        .get("requestBody")
        .and_then(format_openapi_request_body);
    let responses = operation
        .get("responses")
        .and_then(Value::as_object)
        .map(|items| {
            items
                .iter()
                .take(6)
                .filter_map(|(code, value)| {
                    let map = value.as_object()?;
                    let description = extract_first_text(map, &["description"])
                        .unwrap_or_else(|| "response".to_string());
                    let schema = map
                        .get("content")
                        .and_then(format_openapi_content_summary)
                        .unwrap_or_default();
                    Some(if schema.is_empty() {
                        format!("{code}: {description}")
                    } else {
                        format!("{code}: {description} ({schema})")
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let mut lines = vec![
        format!("API: {}", input::normalize_text(title)),
        format!("Operation: {} {}", method.to_ascii_uppercase(), path_name),
    ];
    if let Some(operation_id) = operation_id {
        lines.push(format!("Operation ID: {operation_id}"));
    }
    if let Some(summary) = summary {
        lines.push(format!("Summary: {}", trim_word_window(&summary, 80)));
    }
    if !tags.is_empty() {
        lines.push(format!("Tags: {}", tags.join(" | ")));
    }
    if !parameters.is_empty() {
        lines.push(format!("Parameters: {}", parameters.join(" | ")));
    }
    if let Some(request) = request {
        lines.push(format!("Request: {request}"));
    }
    if !responses.is_empty() {
        lines.push(format!("Responses: {}", responses.join(" | ")));
    }

    (lines.len() >= 3).then_some(lines.join("\n"))
}

fn format_openapi_parameter(value: &Value) -> Option<String> {
    let map = value.as_object()?;
    let name = extract_first_text(map, &["name"])?;
    let location = extract_first_text(map, &["in"]).unwrap_or_else(|| "parameter".to_string());
    let description = extract_first_text(map, &["description"]);
    let required = map
        .get("required")
        .and_then(Value::as_bool)
        .map(|required| if required { "required" } else { "optional" })
        .unwrap_or("optional");
    let schema = map
        .get("schema")
        .map(schema_signature)
        .filter(|value| !value.is_empty());
    let mut parts = vec![format!("{name} in {location} ({required})")];
    if let Some(schema) = schema {
        parts.push(schema);
    }
    if let Some(description) = description {
        parts.push(trim_word_window(&description, 20));
    }
    Some(parts.join(" "))
}

fn format_openapi_request_body(value: &Value) -> Option<String> {
    let map = value.as_object()?;
    let description = extract_first_text(map, &["description"]);
    let content = map
        .get("content")
        .and_then(format_openapi_content_summary)
        .unwrap_or_default();
    let mut parts = Vec::new();
    if let Some(description) = description {
        parts.push(trim_word_window(&description, 24));
    }
    if !content.is_empty() {
        parts.push(content);
    }
    (!parts.is_empty()).then_some(parts.join(" "))
}

fn format_openapi_content_summary(value: &Value) -> Option<String> {
    let content = value.as_object()?;
    let mut parts = Vec::new();
    for (mime, body) in content.iter().take(4) {
        let schema = body
            .as_object()
            .and_then(|map| map.get("schema"))
            .map(schema_signature)
            .filter(|value| !value.is_empty())
            .unwrap_or_default();
        if schema.is_empty() {
            parts.push(mime.to_string());
        } else {
            parts.push(format!("{mime} {schema}"));
        }
    }
    (!parts.is_empty()).then_some(parts.join(" | "))
}

fn schema_signature(value: &Value) -> String {
    let Some(map) = value.as_object() else {
        return extract_single_text(value).unwrap_or_default();
    };
    if let Some(reference) = map.get("$ref").and_then(Value::as_str) {
        return reference.to_string();
    }

    let type_name = extract_first_text(map, &["type"]).unwrap_or_else(|| "object".to_string());
    let format = extract_first_text(map, &["format"]);
    let enum_values = map
        .get("enum")
        .map(extract_text_list)
        .unwrap_or_default()
        .into_iter()
        .take(4)
        .collect::<Vec<_>>();
    let property_names = map
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().take(4).cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    let required = map
        .get("required")
        .map(extract_text_list)
        .unwrap_or_default()
        .into_iter()
        .take(4)
        .collect::<Vec<_>>();

    let mut parts = vec![type_name];
    if let Some(format) = format {
        parts.push(format);
    }
    if !enum_values.is_empty() {
        parts.push(format!("enum {}", enum_values.join("/")));
    }
    if !property_names.is_empty() {
        parts.push(format!("fields {}", property_names.join(",")));
    }
    if !required.is_empty() {
        parts.push(format!("required {}", required.join(",")));
    }
    parts.join(" ")
}

fn stream_repository_directory<F>(
    root: &Path,
    current: &Path,
    policy: &ActiveSourcePolicy,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    for entry in fs::read_dir(current)
        .map_err(|err| format!("failed to read repository dir {}: {err}", current.display()))?
    {
        let entry = entry.map_err(|err| {
            format!(
                "failed to read repository entry in {}: {err}",
                current.display()
            )
        })?;
        let path = entry.path();
        if path.is_dir() {
            if hidden_path_component(&path) {
                continue;
            }
            let control = stream_repository_directory(
                root,
                &path,
                policy,
                max_items,
                sink,
                stats,
                accumulator,
            )?;
            if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
                return Ok(control);
            }
            continue;
        }

        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let label = path
            .strip_prefix(root)
            .ok()
            .map(|value| value.display().to_string())
            .unwrap_or_else(|| path.display().to_string());
        let control =
            emit_repository_blob(&label, &bytes, policy, max_items, sink, stats, accumulator)?;
        if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
            return Ok(control);
        }
    }

    Ok(DatasetSinkControl::Continue)
}

fn stream_repository_zip<F>(
    path: &Path,
    policy: &ActiveSourcePolicy,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    let file = File::open(path)
        .map_err(|err| format!("failed to open archive {}: {err}", path.display()))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|err| format!("failed to read archive {}: {err}", path.display()))?;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|err| format!("failed to read archive entry {}: {err}", path.display()))?;
        if !entry.is_file() {
            continue;
        }
        let mut bytes = Vec::new();
        if entry.read_to_end(&mut bytes).is_err() {
            continue;
        }
        let label = entry.name().to_string();
        let control =
            emit_repository_blob(&label, &bytes, policy, max_items, sink, stats, accumulator)?;
        if control == DatasetSinkControl::Halt || stats.items_seen as usize >= max_items {
            return Ok(control);
        }
    }

    Ok(DatasetSinkControl::Continue)
}

fn emit_repository_blob<F>(
    label: &str,
    bytes: &[u8],
    policy: &ActiveSourcePolicy,
    max_items: usize,
    sink: &mut F,
    stats: &mut DatasetStreamStats,
    accumulator: &mut TextBatchAccumulator,
) -> Result<DatasetSinkControl, String>
where
    F: FnMut(DatasetTextChunk) -> Result<DatasetSinkControl, String>,
{
    if stats.items_seen as usize >= max_items {
        return Ok(DatasetSinkControl::Halt);
    }
    let Some(text) = decode_textual_bytes(bytes) else {
        return Ok(DatasetSinkControl::Continue);
    };
    let Some(document) = format_repository_document(label, &text, policy) else {
        return Ok(DatasetSinkControl::Continue);
    };
    stats.items_seen += 1;
    accumulator.push(document, sink, stats)
}

fn decode_textual_bytes(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() || bytes.len() > 256_000 {
        return None;
    }
    if bytes.iter().any(|byte| *byte == 0) {
        return None;
    }

    let sample = &bytes[..bytes.len().min(4_096)];
    let printable = sample
        .iter()
        .filter(|byte| matches!(**byte, b'\n' | b'\r' | b'\t' | 32..=126))
        .count();
    let printable_ratio = printable as f32 / sample.len().max(1) as f32;
    if printable_ratio < 0.82 {
        return None;
    }

    String::from_utf8(bytes.to_vec())
        .ok()
        .or_else(|| Some(String::from_utf8_lossy(bytes).into_owned()))
}

fn format_repository_document(
    label: &str,
    raw: &str,
    policy: &ActiveSourcePolicy,
) -> Option<String> {
    let semantic = if policy.key == "code" {
        extract_repository_semantic_text(label, raw)
    } else {
        raw.to_string()
    };
    let normalized = sanitize_text_for_policy(&semantic, policy);
    let min_words = if policy.key == "code" { 6 } else { 12 };
    if normalized.split_whitespace().count() < min_words {
        return None;
    }
    let content = trim_word_window(&normalized, 320);
    finalize_policy_text(format!("{label}\n{content}"), policy)
}

fn hidden_path_component(path: &Path) -> bool {
    path.components().any(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|segment| segment.starts_with('.'))
    })
}

fn is_zip_path(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("zip")
}

async fn resolve_local_file(
    source: &TrainingSource,
    default_url: &str,
    label: &str,
    cache_dir_override: Option<&str>,
) -> Result<PathBuf, String> {
    if let Some(value) = source.value.as_deref() {
        if let Some(path) = existing_path(value) {
            return Ok(path);
        }
        if looks_like_url(value) {
            let client = dataset_client()?;
            let cache_dir = cache_dir_override
                .map(PathBuf::from)
                .unwrap_or_else(|| resolve_cache_dir(&source.stream));
            return download_to_cache(&client, value, &cache_dir).await;
        }
    }

    let client = dataset_client()?;
    let cache_dir = cache_dir_override
        .map(PathBuf::from)
        .unwrap_or_else(|| resolve_cache_dir(&source.stream));
    download_to_cache(&client, default_url, &cache_dir)
        .await
        .map_err(|err| format!("failed to resolve {label}: {err}"))
}

fn resolve_cache_dir(stream: &TrainingStreamConfig) -> PathBuf {
    stream
        .cache_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR))
}

fn prepared_manifest_path(stream: &TrainingStreamConfig) -> PathBuf {
    if let Some(cache_dir) = stream.cache_dir.as_ref() {
        PathBuf::from(cache_dir).join("prepared_sources.json")
    } else {
        PathBuf::from(DEFAULT_PREPARED_MANIFEST)
    }
}

fn prepared_source_key(source: &TrainingSource) -> String {
    if let Some(name) = source.name.as_deref() {
        format!(
            "named:{}:{:?}:{}",
            name.to_ascii_lowercase(),
            source.source_type,
            source.stream.shard_limit.unwrap_or(0)
        )
    } else {
        format!(
            "adhoc:{:?}:{}:{}",
            source.source_type,
            source.value.as_deref().unwrap_or_default(),
            source.stream.shard_limit.unwrap_or(0)
        )
    }
}

fn load_prepared_source_record(
    source: &TrainingSource,
) -> Result<Option<PreparedSourceRecord>, String> {
    let path = prepared_manifest_path(&source.stream);
    if !path.exists() {
        return Ok(None);
    }
    let raw = fs::read_to_string(&path).map_err(|err| {
        format!(
            "failed to read prepared source manifest {}: {err}",
            path.display()
        )
    })?;
    let manifest = serde_json::from_str::<PreparedSourceManifest>(&raw).map_err(|err| {
        format!(
            "failed to parse prepared source manifest {}: {err}",
            path.display()
        )
    })?;
    let key = prepared_source_key(source);
    Ok(manifest
        .sources
        .into_iter()
        .find(|record| record.key == key))
}

fn persist_prepared_source_record(
    source: &TrainingSource,
    record: PreparedSourceRecord,
) -> Result<(), String> {
    let path = prepared_manifest_path(&source.stream);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create prepared source dir {}: {err}",
                parent.display()
            )
        })?;
    }

    let mut manifest = if path.exists() {
        let raw = fs::read_to_string(&path).map_err(|err| {
            format!(
                "failed to read prepared source manifest {}: {err}",
                path.display()
            )
        })?;
        serde_json::from_str::<PreparedSourceManifest>(&raw).unwrap_or_default()
    } else {
        PreparedSourceManifest::default()
    };

    manifest
        .sources
        .retain(|existing| existing.key != record.key);
    manifest.sources.push(record);
    manifest.sources.sort_by(|lhs, rhs| lhs.key.cmp(&rhs.key));

    let json = serde_json::to_string_pretty(&manifest).map_err(|err| {
        format!(
            "failed to serialize prepared source manifest {}: {err}",
            path.display()
        )
    })?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, json).map_err(|err| {
        format!(
            "failed to write prepared source manifest {}: {err}",
            tmp.display()
        )
    })?;
    fs::rename(&tmp, &path).map_err(|err| {
        format!(
            "failed to finalize prepared source manifest {}: {err}",
            path.display()
        )
    })?;
    Ok(())
}

fn total_size_bytes(primary: &Path, shards: &[PathBuf]) -> u64 {
    let mut total = fs::metadata(primary).map(|meta| meta.len()).unwrap_or(0);
    for shard in shards {
        total += fs::metadata(shard).map(|meta| meta.len()).unwrap_or(0);
    }
    total
}

fn open_text_reader(path: &Path) -> Result<Box<dyn BufRead>, String> {
    let file =
        File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    if path.extension().and_then(|ext| ext.to_str()) == Some("bz2") {
        Ok(Box::new(BufReader::new(MultiBzDecoder::new(file))))
    } else if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

async fn download_to_cache(
    client: &Client,
    url: &str,
    cache_dir: &Path,
) -> Result<PathBuf, String> {
    fs::create_dir_all(cache_dir)
        .map_err(|err| format!("failed to create cache dir {}: {err}", cache_dir.display()))?;
    let file_name = sanitized_cache_name(url);
    let target = cache_dir.join(file_name);
    if target.exists() {
        return Ok(target);
    }

    let partial = target.with_extension("part");
    let mut partial_len = tokio::fs::metadata(&partial)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let remote_len = fetch_remote_content_length(client, url).await?;

    if partial_len > 0 {
        match remote_len {
            Some(length) if partial_len == length => {
                tokio::fs::rename(&partial, &target).await.map_err(|err| {
                    format!(
                        "failed to finalize resumed cache file {}: {err}",
                        target.display()
                    )
                })?;
                return Ok(target);
            }
            Some(length) if partial_len > length => {
                tokio::fs::remove_file(&partial).await.map_err(|err| {
                    format!(
                        "failed to discard oversized partial cache file {}: {err}",
                        partial.display()
                    )
                })?;
                partial_len = 0;
            }
            _ => {}
        }
    }

    let mut response = client
        .get(url)
        .header(RANGE, format!("bytes={partial_len}-"))
        .send()
        .await
        .map_err(|err| format!("failed to download {url}: {err}"))?;

    if response.status() == StatusCode::RANGE_NOT_SATISFIABLE {
        if partial_len > 0 && remote_len.is_some_and(|length| length == partial_len) {
            tokio::fs::rename(&partial, &target).await.map_err(|err| {
                format!(
                    "failed to finalize resumed cache file {}: {err}",
                    target.display()
                )
            })?;
            return Ok(target);
        }

        if partial_len > 0 && partial.exists() {
            tokio::fs::remove_file(&partial).await.map_err(|err| {
                format!(
                    "failed to reset invalid partial cache file {}: {err}",
                    partial.display()
                )
            })?;
        }
        response = client
            .get(url)
            .send()
            .await
            .map_err(|err| format!("failed to restart download {url}: {err}"))?;
    }

    let status = response.status();
    let response = response
        .error_for_status()
        .map_err(|err| format!("download request failed {url}: {err}"))?;

    let append = partial_len > 0 && status == StatusCode::PARTIAL_CONTENT;
    let mut output = if append {
        tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial)
            .await
            .map_err(|err| {
                format!(
                    "failed to open partial cache file {} for resume: {err}",
                    partial.display()
                )
            })?
    } else {
        tokio::fs::File::create(&partial)
            .await
            .map_err(|err| format!("failed to create cache file {}: {err}", partial.display()))?
    };
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|err| format!("failed to read download stream {url}: {err}"))?;
        output
            .write_all(&chunk)
            .await
            .map_err(|err| format!("failed to write cache file {}: {err}", partial.display()))?;
    }
    output
        .flush()
        .await
        .map_err(|err| format!("failed to flush cache file {}: {err}", partial.display()))?;
    drop(output);
    tokio::fs::rename(&partial, &target)
        .await
        .map_err(|err| format!("failed to finalize cache file {}: {err}", target.display()))?;
    Ok(target)
}

async fn fetch_remote_content_length(client: &Client, url: &str) -> Result<Option<u64>, String> {
    let response = client
        .head(url)
        .send()
        .await
        .map_err(|err| format!("failed to inspect remote file {url}: {err}"))?;

    if !response.status().is_success() {
        return Ok(None);
    }

    let length = response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    Ok(length)
}

fn existing_path(value: &str) -> Option<PathBuf> {
    let path = PathBuf::from(value);
    path.exists().then_some(path)
}

fn looks_like_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

fn normalize_openwebtext_manifest_url(value: &str) -> String {
    if value.ends_with(".json") || value.contains("/parquet") {
        value.to_string()
    } else {
        DEFAULT_OPENWEBTEXT_MANIFEST_URL.to_string()
    }
}

fn matches_path(path_stack: &[Vec<u8>], expected: &[&[u8]]) -> bool {
    if path_stack.len() < expected.len() {
        return false;
    }
    path_stack
        .iter()
        .skip(path_stack.len() - expected.len())
        .zip(expected.iter())
        .all(|(actual, expected)| actual.as_slice() == *expected)
}

fn split_text_blocks(text: &str, chunk_char_limit: usize) -> Vec<String> {
    let normalized = input::normalize_text(text);
    if normalized.is_empty() {
        return Vec::new();
    }
    if normalized.len() <= chunk_char_limit {
        return vec![normalized];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in normalized.split('\n') {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        if paragraph.len() > chunk_char_limit {
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
            }
            split_long_text(paragraph, chunk_char_limit, &mut chunks);
            continue;
        }
        let separator = if current.is_empty() { 0 } else { 2 };
        if current.len() + separator + paragraph.len() > chunk_char_limit {
            chunks.push(current.trim().to_string());
            current.clear();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

fn split_long_text(text: &str, chunk_char_limit: usize, chunks: &mut Vec<String>) {
    let mut current = String::new();
    for word in text.split_whitespace() {
        let separator = if current.is_empty() { 0 } else { 1 };
        if current.len() + separator + word.len() > chunk_char_limit {
            if !current.is_empty() {
                chunks.push(current.trim().to_string());
                current.clear();
            }
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
}

fn clean_wikipedia_markup(raw: &str) -> String {
    let mut text = comment_regex().replace_all(raw, " ").into_owned();
    text = ref_block_regex().replace_all(&text, " ").into_owned();
    text = ref_self_closing_regex()
        .replace_all(&text, " ")
        .into_owned();
    text = table_regex().replace_all(&text, " ").into_owned();
    text = remove_balanced_sections(&text, "{{", "}}");
    text = html_tag_regex().replace_all(&text, " ").into_owned();

    let mut lines = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed == "__TOC__"
            || trimmed.starts_with("[[Category:")
            || trimmed.starts_with("[[File:")
            || trimmed.starts_with("[[Image:")
        {
            continue;
        }

        let mut cleaned = heading_regex()
            .replace_all(trimmed, "$1")
            .into_owned()
            .replace("'''", "")
            .replace("''", "");
        cleaned = internal_link_regex()
            .replace_all(&cleaned, |caps: &regex::Captures<'_>| {
                let body = caps.get(1).map(|value| value.as_str()).unwrap_or_default();
                if body.starts_with("Category:")
                    || body.starts_with("File:")
                    || body.starts_with("Image:")
                {
                    String::new()
                } else {
                    body.rsplit('|').next().unwrap_or(body).trim().to_string()
                }
            })
            .into_owned();
        cleaned = external_link_regex()
            .replace_all(&cleaned, |caps: &regex::Captures<'_>| {
                caps.get(1)
                    .map(|value| value.as_str().trim().to_string())
                    .filter(|value| !value.is_empty())
                    .unwrap_or_default()
            })
            .into_owned();
        cleaned = input::normalize_text(&cleaned);
        if !cleaned.is_empty() && !cleaned.starts_with("#redirect") {
            lines.push(cleaned);
        }
    }

    input::normalize_text(&lines.join("\n"))
}

fn remove_balanced_sections(text: &str, open: &str, close: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut depth = 0usize;

    while cursor < text.len() {
        if text[cursor..].starts_with(open) {
            depth += 1;
            cursor += open.len();
            continue;
        }
        if depth > 0 && text[cursor..].starts_with(close) {
            depth = depth.saturating_sub(1);
            cursor += close.len();
            continue;
        }

        if depth == 0 {
            if let Some(ch) = text[cursor..].chars().next() {
                result.push(ch);
                cursor += ch.len_utf8();
            } else {
                break;
            }
        } else if let Some(ch) = text[cursor..].chars().next() {
            cursor += ch.len_utf8();
        } else {
            break;
        }
    }

    result
}

fn wikidata_sentence_from_triple(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || !line.ends_with(" .") {
        return None;
    }

    let subject_end = line.find('>')?;
    let subject = extract_uri_id(&line[1..subject_end])?;
    let predicate_start = line[subject_end + 1..].find('<')? + subject_end + 1;
    let predicate_end = line[predicate_start..].find('>')? + predicate_start;
    let predicate_uri = &line[predicate_start + 1..predicate_end];
    let predicate = wikidata_predicate_label(predicate_uri)?;
    let object_raw = line[predicate_end + 1..line.len() - 2].trim();
    let object = parse_wikidata_object(object_raw)?;
    Some(format!("{subject} {predicate} {object}."))
}

fn dbpedia_sentence_from_triple(line: &str) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || !line.ends_with(" .") {
        return None;
    }

    let subject_end = line.find('>')?;
    let subject = extract_uri_label(&line[1..subject_end])?;
    let predicate_start = line[subject_end + 1..].find('<')? + subject_end + 1;
    let predicate_end = line[predicate_start..].find('>')? + predicate_start;
    let predicate_uri = &line[predicate_start + 1..predicate_end];
    let predicate = dbpedia_predicate_label(predicate_uri)?;
    let object_raw = line[predicate_end + 1..line.len() - 2].trim();
    let object = parse_dbpedia_object(object_raw)?;
    Some(format!("{subject} {predicate} {object}."))
}

fn extract_uri_id(uri: &str) -> Option<String> {
    uri.rsplit(|ch| ch == '/' || ch == '#')
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
}

fn extract_uri_label(uri: &str) -> Option<String> {
    let label = extract_uri_id(uri)?;
    let normalized = input::normalize_text(
        &label
            .replace('_', " ")
            .replace("%20", " ")
            .replace("%28", "(")
            .replace("%29", ")"),
    );
    (!normalized.is_empty()).then_some(normalized)
}

fn wikidata_predicate_label(uri: &str) -> Option<String> {
    let id = extract_uri_id(uri)?;
    let label = match id.as_str() {
        "P31" => "instance of",
        "P17" => "country",
        "P27" => "country of citizenship",
        "P36" => "capital",
        "P47" => "shares border with",
        "P50" => "author",
        "P61" => "discoverer or inventor",
        "P112" => "founded by",
        "P131" => "located in the administrative territorial entity",
        "P159" => "headquarters location",
        "P169" => "chief executive officer",
        "P279" => "subclass of",
        "P361" => "part of",
        "P495" => "country of origin",
        "P569" => "date of birth",
        "P570" => "date of death",
        "P571" => "inception",
        "P625" => "coordinate location",
        "P856" => "official website",
        "P1082" => "population",
        "P2046" => "area",
        "label" => "label",
        "name" => "name",
        "description" => "description",
        _ => return Some(format!("property {id}")),
    };
    Some(label.to_string())
}

fn dbpedia_predicate_label(uri: &str) -> Option<String> {
    let id = extract_uri_id(uri)?;
    let mut label = String::new();
    let mut previous_was_lowercase = false;
    for ch in id.chars() {
        if matches!(ch, '_' | '-') {
            if !label.ends_with(' ') {
                label.push(' ');
            }
            previous_was_lowercase = false;
            continue;
        }
        if ch.is_uppercase() && previous_was_lowercase {
            label.push(' ');
        }
        label.push(ch);
        previous_was_lowercase = ch.is_lowercase();
    }
    let normalized = input::normalize_text(&label.to_ascii_lowercase());
    (!normalized.is_empty()).then_some(normalized)
}

fn parse_wikidata_object(raw: &str) -> Option<String> {
    if raw.starts_with('<') && raw.ends_with('>') {
        return extract_uri_id(&raw[1..raw.len() - 1]);
    }

    if let Some(rest) = raw.strip_prefix('"') {
        let mut literal = String::new();
        let mut escaped = false;
        for ch in rest.chars() {
            if escaped {
                literal.push(match ch {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => break,
                other => literal.push(other),
            }
        }
        let normalized = input::normalize_text(&literal);
        return (!normalized.is_empty()).then_some(normalized);
    }

    (!raw.is_empty()).then_some(raw.to_string())
}

fn parse_dbpedia_object(raw: &str) -> Option<String> {
    if raw.starts_with('<') && raw.ends_with('>') {
        return extract_uri_label(&raw[1..raw.len() - 1]);
    }
    parse_wikidata_object(raw)
}

fn sanitized_cache_name(source: &str) -> String {
    let mut name = source
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch,
            '.' | '-' | '_' => ch,
            _ => '_',
        })
        .collect::<String>();
    if name.len() > 160 {
        name = name[name.len() - 160..].to_string();
    }
    if name.is_empty() {
        "dataset.bin".to_string()
    } else {
        name
    }
}

fn dataset_client() -> Result<Client, String> {
    let mut default_headers = HeaderMap::new();
    if let Some(token) = huggingface_access_token() {
        let bearer = format!("Bearer {token}");
        let header = HeaderValue::from_str(&bearer)
            .map_err(|err| format!("invalid huggingface auth token: {err}"))?;
        default_headers.insert(AUTHORIZATION, header);
    }
    Client::builder()
        .user_agent("spse_engine/0.1")
        .default_headers(default_headers)
        .build()
        .map_err(|err| format!("failed to build dataset client: {err}"))
}

fn huggingface_access_token() -> Option<String> {
    ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"]
        .into_iter()
        .find_map(|key| std::env::var(key).ok())
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn comment_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)<!--.*?-->").expect("valid wikipedia comment regex"))
}

fn ref_block_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?is)<ref\b[^>]*>.*?</ref>").expect("valid wikipedia ref regex")
    })
}

fn ref_self_closing_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?is)<ref\b[^>]*/>").expect("valid wikipedia ref self closing regex")
    })
}

fn table_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?s)\{\|.*?\|\}").expect("valid wikipedia table regex"))
}

fn html_tag_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"(?is)<[^>]+>").expect("valid wikipedia html tag regex"))
}

fn heading_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^=+\s*(.*?)\s*=+$").expect("valid wikipedia heading regex"))
}

fn internal_link_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\[\[([^\]]+)\]\]").expect("valid wikipedia internal link regex")
    })
}

fn external_link_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\[(?:https?://[^\s\]]+)(?:\s+([^\]]+))?\]")
            .expect("valid wikipedia external link regex")
    })
}

fn internal_id_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\b[QP]\d+\b").expect("valid internal id regex"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::types::{MemoryType, TrainingSource, TrainingSourceType, TrainingStreamConfig};
    use std::io::{Read, Write};
    use std::path::Path;
    use std::thread;
    use std::time::Duration;
    use uuid::Uuid;

    #[test]
    fn common_crawl_manifest_extraction_prefers_official_wet_manifest() {
        let page = r#"
            <html>
              <body>
                <a href="https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-08/wet.paths.gz">WET files</a>
              </body>
            </html>
        "#;

        let manifest = extract_common_crawl_manifest_url(page).expect("manifest url");
        assert_eq!(
            manifest,
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-08/wet.paths.gz"
        );
    }

    #[test]
    fn common_crawl_latest_page_crawl_id_can_be_extracted_without_manifest_link() {
        let page = r#"
            <html>
              <body>
                <div>Latest crawl: CC-MAIN-2026-08</div>
                <a href="https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-08/warc.paths.gz">WARC files</a>
              </body>
            </html>
        "#;

        let crawl_id = extract_common_crawl_crawl_id(page).expect("crawl id");
        assert_eq!(crawl_id, "CC-MAIN-2026-08");
        assert!(extract_common_crawl_manifest_url(page).is_none());
    }

    #[test]
    fn common_crawl_relative_shard_paths_are_normalized_to_https() {
        assert_eq!(
            normalize_common_crawl_url(
                "crawl-data/CC-MAIN-2026-08/segments/1770395505396.36/wet/CC-MAIN-20260206181458-20260206211458-00000.warc.wet.gz"
            ),
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-08/segments/1770395505396.36/wet/CC-MAIN-20260206181458-20260206211458-00000.warc.wet.gz"
        );
        assert_eq!(
            normalize_common_crawl_url(
                "/crawl-data/CC-MAIN-2026-08/segments/1770395505396.36/wet/CC-MAIN-20260206181458-20260206211458-00000.warc.wet.gz"
            ),
            "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-08/segments/1770395505396.36/wet/CC-MAIN-20260206181458-20260206211458-00000.warc.wet.gz"
        );
    }

    #[test]
    fn qa_json_policy_drops_answers_and_schema_noise() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::StructuredJson,
            name: Some("squad_v2".to_string()),
            value: Some("fixture.json".to_string()),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let mut stats = DatasetStreamStats::default();
        let raw = r#"{
            "question": "What kind of language is Rust?",
            "answer": "A systems programming language",
            "context": "Rust is a systems programming language focused on safety and concurrency.",
            "id": "abc123",
            "metadata": {"split": "train"}
        }"#;

        let examples = collect_structured_examples_from_text(
            Path::new("fixture.json"),
            raw,
            &mut stats,
            &policy,
        )
        .expect("examples");

        let combined = examples.join("\n");
        assert!(combined.contains("What kind of language is Rust?"));
        assert!(combined
            .contains("Rust is a systems programming language focused on safety and concurrency."));
        assert!(!combined.contains("Answer:"));
        assert!(!combined.contains("Question:"));
        assert!(!combined.contains("Context:"));
        assert!(!combined.contains("metadata"));
        assert!(!combined.contains("id:"));
    }

    #[test]
    fn code_policy_prefers_comment_text_over_syntax() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::CodeRepository,
            name: Some("open_license_repo".to_string()),
            value: Some("repo.rs".to_string()),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let raw = r#"
            /// Adds two numbers safely.
            /// Returns the combined total.
            fn add(left: i32, right: i32) -> i32 { left + right }
        "#;

        let formatted = format_repository_document("src/lib.rs", raw, &policy).expect("formatted");
        assert!(formatted.contains("Adds two numbers safely."));
        assert!(formatted.contains("Returns the combined total."));
        assert!(!formatted.contains("fn add"));
    }

    #[test]
    fn named_reasoning_sources_use_policy_memory_override() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::StructuredJson,
            name: Some("gsm8k_train".to_string()),
            value: Some("fixture.jsonl".to_string()),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };

        assert_eq!(
            effective_memory_type(&source, &config),
            Some(MemoryType::Episodic)
        );
    }

    #[test]
    fn huggingface_locator_parses_repo_and_shape() {
        let locator = parse_huggingface_locator(
            "hf://HuggingFaceFW/fineweb-edu?subset=default&split=train&row_mode=plain_text&text_fields=text,summary",
        )
        .expect("locator");

        assert_eq!(locator.repo_id, "HuggingFaceFW/fineweb-edu");
        assert_eq!(locator.subset.as_deref(), Some("default"));
        assert_eq!(locator.split.as_deref(), Some("train"));
        assert_eq!(locator.text_fields, vec!["text", "summary"]);
        assert!(matches!(locator.row_mode, HuggingFaceRowMode::PlainText));
    }

    #[test]
    fn huggingface_rows_per_pull_adapts_with_latency() {
        let config = crate::config::HuggingFaceStreamingConfig::default();

        let grown = adapt_huggingface_rows_per_pull(250, 800, &config);
        let reduced = adapt_huggingface_rows_per_pull(250, 5_000, &config);

        assert!(grown > 250);
        assert!(reduced < 250);
        assert!(grown <= config.max_rows_per_pull);
        assert!(reduced >= config.min_rows_per_pull);
    }

    #[test]
    fn huggingface_plain_text_question_and_answer_rows_are_extractable_without_labels() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_openai_gsm8k".to_string()),
            value: Some(
                "hf://openai/gsm8k?subset=main&split=train&row_mode=plain_text&text_fields=question,answer"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let row = serde_json::json!({
            "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
            "answer": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether.\n#### 72"
        });

        let examples = huggingface_plain_text_examples(
            &row,
            &["question".to_string(), "answer".to_string()],
            &policy,
        );

        let combined = examples.join("\n");
        assert!(combined.contains("How many clips did Natalia sell altogether in April and May?"));
        assert!(combined.contains("48/2 = 24 clips in May"));
        assert!(combined.contains("72 clips altogether"));
        assert!(!combined.contains("#### 72"));
        assert!(!combined.contains("question:"));
        assert!(!combined.contains("answer:"));
        assert!(!combined.contains("Question:"));
        assert!(!combined.contains("Answer:"));
    }

    #[test]
    fn huggingface_rows_api_plain_text_uses_value_only_fields() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_openai_gsm8k".to_string()),
            value: Some(
                "hf://openai/gsm8k?subset=main&split=train&row_mode=plain_text&text_fields=question,answer"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let row = serde_json::json!({
            "question": "Marta solved 6 problems on Monday and 7 problems on Tuesday. How many problems did she solve altogether?",
            "answer": "6 + 7 = 13.\nMarta solved 13 problems altogether.\n#### 13",
            "id": "row-1"
        });

        let examples = huggingface_value_examples(
            &row,
            HuggingFaceRowMode::PlainText,
            &["question".to_string(), "answer".to_string()],
            &policy,
        );

        let combined = examples.join("\n");
        assert!(combined.contains("How many problems did she solve altogether?"));
        assert!(combined.contains("6 + 7 = 13."));
        assert!(!combined.contains("question:"));
        assert!(!combined.contains("answer:"));
        assert!(!combined.contains("row-1"));
        assert!(!combined.contains("#### 13"));
    }

    #[test]
    fn huggingface_plain_text_does_not_fall_back_to_metadata_when_text_fields_are_set() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_fw_fineweb_edu".to_string()),
            value: Some(
                "hf://HuggingFaceFW/fineweb-edu?subset=default&split=train&row_mode=plain_text&text_fields=text"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let row = serde_json::json!({
            "text": "",
            "url": "http://example.com/article",
            "id": "abc123"
        });

        let examples = huggingface_plain_text_examples(&row, &["text".to_string()], &policy);

        assert!(examples.is_empty());
    }

    #[test]
    fn standalone_url_lines_are_discarded_from_policy_text() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_fw_fineweb_edu".to_string()),
            value: Some(
                "hf://HuggingFaceFW/fineweb-edu?subset=default&split=train&row_mode=plain_text&text_fields=text"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);

        assert!(finalize_policy_text("https://example.com/path".to_string(), &policy).is_none());
        assert!(finalize_policy_text(
            "This is a real sentence about a topic.".to_string(),
            &policy
        )
        .is_some());
    }

    #[test]
    fn huggingface_structured_dialogue_uses_values_without_field_labels() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_tb_smoltalk2".to_string()),
            value: Some(
                "hf://HuggingFaceTB/smoltalk2?subset=SFT&split=OpenHermes_2.5_no_think&row_mode=structured_json"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let raw = r#"{
            "title": "Travel timeline",
            "messages": [
                {"role": "user", "content": "Create a travel timeline."},
                {"role": "assistant", "content": "Sure, here is a simple travel timeline."}
            ]
        }"#;
        let mut stats = DatasetStreamStats::default();

        let examples = collect_structured_examples_from_text(
            Path::new("fixture.json"),
            raw,
            &mut stats,
            &policy,
        )
        .expect("examples");

        let combined = examples.join("\n");
        assert!(combined.contains("user: Create a travel timeline."));
        assert!(combined.contains("assistant: Sure, here is a simple travel timeline."));
        assert!(!combined.contains("Dialogue:"));
        assert!(!combined.contains("Dialogue context:"));
    }

    #[test]
    fn huggingface_structured_task_uses_values_without_question_answer_labels() {
        let config = EngineConfig::default();
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_custom_structured".to_string()),
            value: Some(
                "hf://example/custom?subset=default&split=train&row_mode=structured_json"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig::default(),
        };
        let policy = resolve_source_policy(&config, &source);
        let raw = r#"{
            "prompt": "Summarize the article.",
            "response": "The article explains the core idea.",
            "context": "This article discusses the main concept in simple terms."
        }"#;
        let mut stats = DatasetStreamStats::default();

        let examples = collect_structured_examples_from_text(
            Path::new("fixture.json"),
            raw,
            &mut stats,
            &policy,
        )
        .expect("examples");

        let combined = examples.join("\n");
        assert!(combined.contains("Summarize the article."));
        assert!(combined.contains("The article explains the core idea."));
        assert!(combined.contains("This article discusses the main concept in simple terms."));
        assert!(!combined.contains("Question:"));
        assert!(!combined.contains("Answer:"));
        assert!(!combined.contains("Context:"));
    }

    #[test]
    fn localize_remote_capable_huggingface_source_keeps_remote_locator_when_prepare_is_nonlocal() {
        let source = TrainingSource {
            source_type: TrainingSourceType::HuggingFaceDataset,
            name: Some("hf_tb_smoltalk2".to_string()),
            value: Some(
                "hf://HuggingFaceTB/smoltalk2?subset=SFT&split=OpenHermes_2.5_no_think&row_mode=structured_json"
                    .to_string(),
            ),
            mime: None,
            content: None,
            target_memory: None,
            memory_channels: None,
            stream: TrainingStreamConfig {
                cache_dir: Some(
                    std::env::temp_dir()
                        .join(format!("prepared_hf_{}", Uuid::new_v4()))
                        .display()
                        .to_string(),
                ),
                ..TrainingStreamConfig::default()
            },
        };

        let record = PreparedSourceRecord {
            key: prepared_source_key(&source),
            source_name: source.name.clone(),
            source_type: source.source_type,
            remote_value: source.value.clone(),
            local_value: source.value.clone().unwrap(),
            shard_paths: Vec::new(),
            prepared_at: Utc::now(),
            size_bytes: 0,
        };
        persist_prepared_source_record(&source, record).expect("persist prepared source");

        let localized =
            localize_remote_capable_source(&source).expect("localize remote-capable hf source");
        assert_eq!(localized.value, source.value);
    }

    #[tokio::test]
    async fn download_to_cache_resumes_existing_partial_file() {
        let body = b"resume-me-without-restarting".to_vec();
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("bind test download server");
        let address = listener.local_addr().expect("local server addr");
        let server_body = body.clone();
        let server = thread::spawn(move || {
            for _ in 0..2 {
                let (mut stream, _) = listener.accept().expect("accept");
                stream
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .expect("read timeout");
                let mut buffer = [0_u8; 4096];
                let read = stream.read(&mut buffer).expect("read request");
                let request = String::from_utf8_lossy(&buffer[..read]);
                let is_head = request.starts_with("HEAD ");
                let range_start = request
                    .lines()
                    .find_map(|line| {
                        let lower = line.to_ascii_lowercase();
                        lower
                            .strip_prefix("range: bytes=")
                            .and_then(|value| value.strip_suffix('-'))
                            .and_then(|value| value.trim().parse::<usize>().ok())
                    })
                    .unwrap_or(0);
                let payload = if is_head {
                    Vec::new()
                } else {
                    server_body
                        .get(range_start..)
                        .expect("range within body")
                        .to_vec()
                };
                let status_line = if is_head {
                    "HTTP/1.1 200 OK"
                } else if range_start > 0 {
                    "HTTP/1.1 206 Partial Content"
                } else {
                    "HTTP/1.1 200 OK"
                };
                let response = format!(
                    "{status_line}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    if is_head {
                        server_body.len()
                    } else {
                        payload.len()
                    }
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write headers");
                if !is_head {
                    stream.write_all(&payload).expect("write body");
                }
            }
        });

        let cache_dir = std::env::temp_dir().join(format!("resume_cache_{}", Uuid::new_v4()));
        fs::create_dir_all(&cache_dir).expect("create cache dir");
        let url = format!("http://{address}/dataset.bin");
        let target = cache_dir.join(sanitized_cache_name(&url));
        let partial = target.with_extension("part");
        fs::write(&partial, &body[..8]).expect("seed partial file");

        let downloaded = download_to_cache(&Client::new(), &url, &cache_dir)
            .await
            .expect("resume download");
        let written = fs::read(&downloaded).expect("read downloaded content");

        assert_eq!(downloaded, target);
        assert_eq!(written, body);
        assert!(!partial.exists());

        server.join().expect("join server");
        fs::remove_dir_all(&cache_dir).expect("remove cache dir");
    }
}
