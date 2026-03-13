use parquet::file::reader::{FileReader, SerializedFileReader};
use regex::Regex;
use reqwest::Client;
use serde_json::Value;
use spse_engine::config::EngineConfig;
use spse_engine::datasets::{self, DatasetSinkControl};
use spse_engine::open_sources;
use std::fs::File;
use std::path::PathBuf;
use std::sync::OnceLock;

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        return Err(
            "usage: cargo run --bin verify_huggingface_rows -- <source-id> [<source-id> ...]"
                .to_string(),
        );
    }

    let config = EngineConfig::load_default_file();

    for source_name in args {
        let mut source = open_sources::catalog_source(&source_name);
        source = open_sources::resolve_training_source(&source)?;
        source = datasets::localize_remote_capable_source(&source)?;
        source.stream.item_limit = Some(1);
        source.stream.batch_size = Some(1);
        source.stream.shard_limit = Some(1);
        source.stream.chunk_char_limit = Some(4_000);
        source.stream.cache_dir = Some(
            std::env::temp_dir()
                .join(format!("spse_hf_verify_{}", source_name))
                .display()
                .to_string(),
        );

        let mut first_chunk = None;
        let stats = datasets::stream_training_source(&source, &config, |chunk| {
            first_chunk = Some(chunk);
            Ok(DatasetSinkControl::Halt)
        })
        .await?;

        let Some(chunk) = first_chunk else {
            let (raw, fallback_preview) = debug_first_row(&source).await?;
            if let Some(preview) = fallback_preview {
                println!("source={source_name}");
                println!("context=raw_row_fallback");
                println!("items_seen={}", stats.items_seen.max(1));
                println!("preview={preview}");
                println!("raw_row={raw}");
                println!();
                continue;
            }
            return Err(format!(
                "no parsed training chunk emitted for {source_name} (warnings: {:?}) raw_row={raw}",
                stats.warnings
            ));
        };

        let preview = chunk.content.chars().take(280).collect::<String>();
        println!("source={source_name}");
        println!("context={}", chunk.context_label);
        println!("items_seen={}", stats.items_seen);
        println!("preview={preview}");
        println!();
    }

    Ok(())
}

async fn debug_first_row(
    source: &spse_engine::types::TrainingSource,
) -> Result<(String, Option<String>), String> {
    let value = source
        .value
        .as_deref()
        .ok_or_else(|| "missing_huggingface_value".to_string())?;
    if !value.starts_with("hf://") {
        return Ok(("non_hf_source".to_string(), None));
    }

    let parsed = reqwest::Url::parse(value).map_err(|err| err.to_string())?;
    let owner = parsed
        .host_str()
        .ok_or_else(|| "missing_huggingface_owner".to_string())?;
    let dataset = parsed.path().trim_start_matches('/');
    let repo_id = format!("{owner}/{dataset}");
    let mut subset = None::<String>;
    let mut split = None::<String>;
    let mut text_fields = Vec::<String>::new();
    for (key, raw_value) in parsed.query_pairs() {
        match key.as_ref() {
            "subset" => subset = Some(raw_value.into_owned()),
            "split" => split = Some(raw_value.into_owned()),
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

    let client = Client::builder()
        .user_agent("spse_engine/0.1")
        .build()
        .map_err(|err| err.to_string())?;
    let parquet_api = format!("https://huggingface.co/api/datasets/{repo_id}/parquet");
    let api_value = client
        .get(&parquet_api)
        .send()
        .await
        .map_err(|err| err.to_string())?
        .error_for_status()
        .map_err(|err| err.to_string())?
        .json::<Value>()
        .await
        .map_err(|err| err.to_string())?;
    let root = api_value
        .as_object()
        .ok_or_else(|| "parquet api not an object".to_string())?;
    let subset_name = subset
        .or_else(|| root.get("default").map(|_| "default".to_string()))
        .or_else(|| root.keys().next().cloned())
        .ok_or_else(|| "no parquet subset".to_string())?;
    let subset_value = root
        .get(&subset_name)
        .ok_or_else(|| format!("missing subset {subset_name}"))?;
    let split_name = split
        .or_else(|| {
            subset_value
                .as_object()
                .and_then(|map| map.get("train").map(|_| "train".to_string()))
        })
        .or_else(|| {
            subset_value
                .as_object()
                .and_then(|map| map.keys().next().cloned())
        })
        .ok_or_else(|| "no parquet split".to_string())?;
    let shard_url = subset_value
        .get(&split_name)
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .ok_or_else(|| "no parquet shard".to_string())?;

    let bytes = client
        .get(shard_url)
        .send()
        .await
        .map_err(|err| err.to_string())?
        .error_for_status()
        .map_err(|err| err.to_string())?
        .bytes()
        .await
        .map_err(|err| err.to_string())?;
    let path = temp_parquet_path(&repo_id);
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|err| err.to_string())?;

    let file = File::open(&path).map_err(|err| err.to_string())?;
    let reader = SerializedFileReader::new(file).map_err(|err| err.to_string())?;
    let mut rows = reader.get_row_iter(None).map_err(|err| err.to_string())?;
    let first = rows
        .next()
        .transpose()
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "empty parquet shard".to_string())?;
    let _ = std::fs::remove_file(&path);
    let raw = format!("{first:?}");
    let mut extracted = Vec::new();
    if !text_fields.is_empty() {
        for field in &text_fields {
            if let Some((_, parquet::record::Field::Str(value))) = first
                .get_column_iter()
                .find(|(name, _)| name.eq_ignore_ascii_case(field))
            {
                extracted.push(value.clone());
            }
        }
    }
    if extracted.is_empty() {
        for (_, field) in first.get_column_iter() {
            if let parquet::record::Field::Str(value) = field {
                extracted.push(value.clone());
            }
        }
    }
    let preview = if extracted.is_empty() {
        None
    } else {
        Some(
            strip_trailing_gsm8k_marker(&extracted.join("\n\n"))
                .chars()
                .take(280)
                .collect::<String>(),
        )
    };
    Ok((raw, preview))
}

fn temp_parquet_path(repo_id: &str) -> PathBuf {
    let name = repo_id.replace('/', "_");
    std::env::temp_dir().join(format!("hf_verify_{name}.parquet"))
}

fn strip_trailing_gsm8k_marker(text: &str) -> String {
    trailing_gsm8k_marker_regex()
        .replace(text.trim(), "")
        .trim()
        .to_string()
}

fn trailing_gsm8k_marker_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\s*####\s+[^\n]+$").expect("valid trailing gsm8k marker regex")
    })
}
