use crate::config::{
    EngineConfig, GovernanceConfig, RetrievalIoConfig, TrustConfig, UnitBuilderConfig,
};
use crate::layers::builder::UnitBuilder;
use crate::layers::input;
use crate::layers::safety::TrustSafetyValidator;
use crate::types::{
    DatabaseHealthMetrics, EvidenceState, RetrievedDocument, SanitizedQuery, TrainingSourceType,
};
use chrono::Utc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct CacheEntry {
    documents: Vec<RetrievedDocument>,
    inserted_at: Instant,
}

pub struct RetrievalPipeline {
    client: Client,
    cache: Mutex<HashMap<String, CacheEntry>>,
    builder: UnitBuilderConfig,
    governance: GovernanceConfig,
}

impl RetrievalPipeline {
    pub fn new(config: &EngineConfig) -> Self {
        let client = Client::builder()
            .user_agent("spse_engine/0.1")
            .timeout(std::time::Duration::from_millis(
                config.retrieval_io.retrieval_timeout_ms,
            ))
            .build()
            .expect("failed to build reqwest client");

        Self {
            client,
            cache: Mutex::new(HashMap::new()),
            builder: config.builder.clone(),
            governance: config.governance.clone(),
        }
    }

    pub async fn search(
        &self,
        query: &SanitizedQuery,
        safety: &TrustSafetyValidator,
        retrieval: &RetrievalIoConfig,
        trust: &TrustConfig,
        max_retries: usize,
        database_health: &DatabaseHealthMetrics,
    ) -> EvidenceState {
        if query.sanitized_query.is_empty() {
            return EvidenceState {
                warnings: vec!["sanitized_query_empty".to_string()],
                ..EvidenceState::default()
            };
        }

        if let Some(cached) = self
            .cache
            .lock()
            .expect("retrieval cache mutex poisoned")
            .get(&query.sanitized_query)
            .cloned()
        {
            if cached.inserted_at.elapsed()
                <= Duration::from_secs(retrieval.cache_ttl_seconds.max(1))
            {
                let average_trust = average_trust(&cached.documents);
                return EvidenceState {
                    documents: cached.documents,
                    evidence_units: Vec::new(),
                    warnings: vec!["cache_hit".to_string()],
                    average_trust,
                };
            }
        }

        let mut attempt = 0;
        let mut warnings = Vec::new();
        let mut docs = Vec::new();
        while attempt <= max_retries {
            match self
                .fetch_search_documents(
                    query,
                    retrieval.max_retrieval_results.min(trust.max_results),
                )
                .await
            {
                Ok(found) => {
                    docs = found;
                    break;
                }
                Err(err) => {
                    warnings.push(err);
                    attempt += 1;
                }
            }
        }

        let (accepted_docs, safety_warnings) = safety.filter_documents(docs, trust);
        warnings.extend(safety_warnings);

        let evidence_units = accepted_docs
            .iter()
            .flat_map(|doc| {
                let packet = input::ingest_raw(&doc.normalized_content, false);
                UnitBuilder::ingest_with_governance(
                    &packet,
                    &self.builder,
                    &self.governance,
                    database_health,
                )
                .activated_units
            })
            .take(40)
            .collect::<Vec<_>>();

        let average_trust = average_trust(&accepted_docs);
        self.cache
            .lock()
            .expect("retrieval cache mutex poisoned")
            .insert(
                query.sanitized_query.clone(),
                CacheEntry {
                    documents: accepted_docs.clone(),
                    inserted_at: Instant::now(),
                },
            );

        EvidenceState {
            documents: accepted_docs,
            evidence_units,
            warnings,
            average_trust,
        }
    }

    pub async fn fetch_training_source(
        &self,
        source_type: TrainingSourceType,
        value: &str,
    ) -> Result<String, String> {
        match source_type {
            TrainingSourceType::Url => {
                let response = self
                    .client
                    .get(value)
                    .send()
                    .await
                    .map_err(|err| err.to_string())?;
                let text = response.text().await.map_err(|err| err.to_string())?;
                Ok(self.normalize_content(value, &text))
            }
            TrainingSourceType::Document
            | TrainingSourceType::Dataset
            | TrainingSourceType::HuggingFaceDataset
            | TrainingSourceType::StructuredJson
            | TrainingSourceType::OpenApiSpec
            | TrainingSourceType::CodeRepository
            | TrainingSourceType::WikipediaDump
            | TrainingSourceType::WikidataTruthy
            | TrainingSourceType::OpenWebText
            | TrainingSourceType::DbpediaDump
            | TrainingSourceType::ProjectGutenberg
            | TrainingSourceType::CommonCrawlWet
            | TrainingSourceType::QaJson => Ok(self.normalize_content("document", value)),
        }
    }

    fn normalize_content(&self, source_url: &str, raw: &str) -> String {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            let mut parts = Vec::new();
            collect_json_text(&value, &mut parts);
            let text = parts.join(" ");
            return input::normalize_text(&text);
        }

        let parsed = Html::parse_document(raw);
        let selector = Selector::parse("body").expect("valid selector");
        let mut text = String::new();
        for node in parsed.select(&selector) {
            text.push_str(&node.text().collect::<Vec<_>>().join(" "));
            text.push(' ');
        }

        let normalized = input::normalize_text(if text.is_empty() { raw } else { &text });
        if source_url == "document" {
            normalized
        } else {
            normalized.chars().take(8_000).collect()
        }
    }

    async fn fetch_search_documents(
        &self,
        query: &SanitizedQuery,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let mut docs = Vec::new();
        let mut errors = Vec::new();
        let connector_budget = limit.max(6);

        for variant in query_variants(query) {
            let variant_docs = match self
                .fetch_duckduckgo_documents(&variant, connector_budget.saturating_sub(docs.len()))
                .await
            {
                Ok(found) => found,
                Err(err) => {
                    errors.push(err);
                    Vec::new()
                }
            };
            let variant_grounded = has_grounded_candidate(&variant_docs);
            docs.extend(variant_docs);

            if docs.len() < connector_budget && !variant_grounded {
                match self
                    .fetch_wikipedia_documents(
                        &variant,
                        connector_budget.saturating_sub(docs.len()),
                    )
                    .await
                {
                    Ok(mut found) => docs.append(&mut found),
                    Err(err) => errors.push(err),
                }
            }

            if docs.len() < connector_budget {
                match self
                    .fetch_wikidata_documents(&variant, connector_budget.saturating_sub(docs.len()))
                    .await
                {
                    Ok(mut found) => docs.append(&mut found),
                    Err(err) => errors.push(err),
                }
            }

            if docs.len() < connector_budget {
                match self
                    .fetch_pubmed_central_documents(
                        &variant,
                        connector_budget.saturating_sub(docs.len()),
                    )
                    .await
                {
                    Ok(mut found) => docs.append(&mut found),
                    Err(err) => errors.push(err),
                }
            }

            if docs.len() < connector_budget {
                match self
                    .fetch_nominatim_documents(
                        &variant,
                        connector_budget.saturating_sub(docs.len()),
                    )
                    .await
                {
                    Ok(mut found) => docs.append(&mut found),
                    Err(err) => errors.push(err),
                }
            }

            dedup_documents(&mut docs);
            rank_documents_for_query(&variant, &mut docs);
            docs.truncate(connector_budget);
            if docs.len() >= limit && has_grounded_candidate(&docs) {
                break;
            }
        }

        if docs.is_empty() {
            return Err(errors.join("; "));
        }

        Ok(self.hydrate_documents(docs, limit).await)
    }

    async fn fetch_duckduckgo_documents(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_redirect=1&no_html=1",
            query.replace(' ', "+")
        );
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|err| err.to_string())?;

        let body = response.text().await.map_err(|err| err.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|err| err.to_string())?;
        let mut docs = Vec::new();

        if let Some(abstract_text) = value.get("AbstractText").and_then(Value::as_str) {
            let source_url = value
                .get("AbstractURL")
                .and_then(Value::as_str)
                .filter(|url| !url.is_empty())
                .unwrap_or("https://duckduckgo.com/");
            let normalized = input::normalize_text(abstract_text);
            if !normalized.is_empty() {
                docs.push(RetrievedDocument {
                    source_url: source_url.to_string(),
                    title: value
                        .get("Heading")
                        .and_then(Value::as_str)
                        .unwrap_or("DuckDuckGo Abstract")
                        .to_string(),
                    raw_content: abstract_text.to_string(),
                    normalized_content: normalized,
                    retrieved_at: Utc::now(),
                    trust_score: 0.5,
                    cached: false,
                });
            }
        }

        let has_grounded_abstract = docs
            .first()
            .map(|doc| doc.normalized_content.split_whitespace().count() >= 20)
            .unwrap_or(false);
        if has_grounded_abstract {
            return Ok(docs);
        }

        if let Some(related) = value.get("RelatedTopics").and_then(Value::as_array) {
            for item in related.iter().take(limit.saturating_sub(docs.len())) {
                if let Some(text) = item.get("Text").and_then(Value::as_str) {
                    let url = item
                        .get("FirstURL")
                        .and_then(Value::as_str)
                        .unwrap_or("https://duckduckgo.com/");
                    if url.contains("/c/") {
                        continue;
                    }
                    let normalized = input::normalize_text(text);
                    if !normalized.is_empty() {
                        docs.push(RetrievedDocument {
                            source_url: if url.is_empty() {
                                "https://duckduckgo.com/".to_string()
                            } else {
                                url.to_string()
                            },
                            title: text
                                .split(" - ")
                                .next()
                                .unwrap_or("Related Topic")
                                .to_string(),
                            raw_content: text.to_string(),
                            normalized_content: normalized,
                            retrieved_at: Utc::now(),
                            trust_score: 0.45,
                            cached: false,
                        });
                    }
                } else if let Some(topics) = item.get("Topics").and_then(Value::as_array) {
                    for nested in topics.iter().take(limit.saturating_sub(docs.len())) {
                        if let Some(text) = nested.get("Text").and_then(Value::as_str) {
                            let url = nested
                                .get("FirstURL")
                                .and_then(Value::as_str)
                                .unwrap_or("https://duckduckgo.com/");
                            if url.contains("/c/") {
                                continue;
                            }
                            let normalized = input::normalize_text(text);
                            if !normalized.is_empty() {
                                docs.push(RetrievedDocument {
                                    source_url: if url.is_empty() {
                                        "https://duckduckgo.com/".to_string()
                                    } else {
                                        url.to_string()
                                    },
                                    title: text
                                        .split(" - ")
                                        .next()
                                        .unwrap_or("Related Topic")
                                        .to_string(),
                                    raw_content: text.to_string(),
                                    normalized_content: normalized,
                                    retrieved_at: Utc::now(),
                                    trust_score: 0.45,
                                    cached: false,
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(docs)
    }

    async fn fetch_wikidata_documents(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let limit_string = limit.to_string();
        let url = reqwest::Url::parse_with_params(
            "https://www.wikidata.org/w/api.php",
            &[
                ("action", "wbsearchentities"),
                ("language", "en"),
                ("format", "json"),
                ("type", "item"),
                ("limit", limit_string.as_str()),
                ("search", query),
            ],
        )
        .map_err(|err| err.to_string())?;
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|err| err.to_string())?;
        let body = response.text().await.map_err(|err| err.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|err| err.to_string())?;
        let mut docs = Vec::new();

        if let Some(items) = value.get("search").and_then(Value::as_array) {
            for item in items.iter().take(limit) {
                let label = item
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if label.is_empty() {
                    continue;
                }
                let description = item
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let entity_id = item.get("id").and_then(Value::as_str).unwrap_or("item");
                let raw_content = if description.is_empty() {
                    label.to_string()
                } else {
                    format!("{label}: {description}.")
                };
                let normalized = input::normalize_text(&raw_content);
                if normalized.is_empty() {
                    continue;
                }
                docs.push(RetrievedDocument {
                    source_url: format!("https://www.wikidata.org/wiki/{entity_id}"),
                    title: label.to_string(),
                    raw_content,
                    normalized_content: normalized,
                    retrieved_at: Utc::now(),
                    trust_score: 0.68,
                    cached: false,
                });
            }
        }

        Ok(docs)
    }

    async fn fetch_pubmed_central_documents(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let limit_string = limit.to_string();
        let search_url = reqwest::Url::parse_with_params(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            &[
                ("db", "pmc"),
                ("retmode", "json"),
                ("retmax", limit_string.as_str()),
                ("term", query),
            ],
        )
        .map_err(|err| err.to_string())?;
        let search_body = self
            .client
            .get(search_url)
            .send()
            .await
            .map_err(|err| err.to_string())?
            .text()
            .await
            .map_err(|err| err.to_string())?;
        let search_value =
            serde_json::from_str::<Value>(&search_body).map_err(|err| err.to_string())?;
        let ids = search_value
            .get("esearchresult")
            .and_then(|value| value.get("idlist"))
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .filter_map(Value::as_str)
                    .take(limit)
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        let ids_csv = ids.join(",");
        let summary_url = reqwest::Url::parse_with_params(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            &[("db", "pmc"), ("retmode", "json"), ("id", ids_csv.as_str())],
        )
        .map_err(|err| err.to_string())?;
        let summary_body = self
            .client
            .get(summary_url)
            .send()
            .await
            .map_err(|err| err.to_string())?
            .text()
            .await
            .map_err(|err| err.to_string())?;
        let summary_value =
            serde_json::from_str::<Value>(&summary_body).map_err(|err| err.to_string())?;
        let result = summary_value
            .get("result")
            .and_then(Value::as_object)
            .ok_or_else(|| "invalid pmc summary payload".to_string())?;
        let mut docs = Vec::new();

        for id in ids {
            let Some(item) = result.get(&id) else {
                continue;
            };
            let title = item
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            if title.is_empty() {
                continue;
            }
            let journal = item
                .get("fulljournalname")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let pubdate = item
                .get("pubdate")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim();
            let raw_content = input::normalize_text(&format!(
                "{title}. Journal: {journal}. Published: {pubdate}. PMC article {id}."
            ));
            if raw_content.is_empty() {
                continue;
            }
            docs.push(RetrievedDocument {
                source_url: format!("https://pmc.ncbi.nlm.nih.gov/articles/PMC{id}/"),
                title: title.to_string(),
                raw_content: raw_content.clone(),
                normalized_content: raw_content,
                retrieved_at: Utc::now(),
                trust_score: 0.74,
                cached: false,
            });
        }

        Ok(docs)
    }

    async fn fetch_nominatim_documents(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let limit_string = limit.to_string();
        let url = reqwest::Url::parse_with_params(
            "https://nominatim.openstreetmap.org/search",
            &[
                ("format", "jsonv2"),
                ("limit", limit_string.as_str()),
                ("q", query),
            ],
        )
        .map_err(|err| err.to_string())?;
        let body = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|err| err.to_string())?
            .text()
            .await
            .map_err(|err| err.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|err| err.to_string())?;
        let mut docs = Vec::new();

        if let Some(items) = value.as_array() {
            for item in items.iter().take(limit) {
                let title = item
                    .get("display_name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if title.is_empty() {
                    continue;
                }
                let lat = item.get("lat").and_then(Value::as_str).unwrap_or("");
                let lon = item.get("lon").and_then(Value::as_str).unwrap_or("");
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let class = item
                    .get("class")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                let raw_content = input::normalize_text(&format!(
                    "{title}. Coordinates: {lat}, {lon}. Class: {class}. Type: {kind}."
                ));
                if raw_content.is_empty() {
                    continue;
                }
                docs.push(RetrievedDocument {
                    source_url: "https://nominatim.openstreetmap.org/".to_string(),
                    title: title.to_string(),
                    raw_content: raw_content.clone(),
                    normalized_content: raw_content,
                    retrieved_at: Utc::now(),
                    trust_score: 0.7,
                    cached: false,
                });
            }
        }

        Ok(docs)
    }

    async fn fetch_wikipedia_documents(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let limit_string = limit.to_string();
        let search_url = reqwest::Url::parse_with_params(
            "https://en.wikipedia.org/w/api.php",
            &[
                ("action", "query"),
                ("list", "search"),
                ("format", "json"),
                ("utf8", "1"),
                ("srlimit", limit_string.as_str()),
                ("srsearch", query),
            ],
        )
        .map_err(|err| err.to_string())?;
        let response = self
            .client
            .get(search_url)
            .send()
            .await
            .map_err(|err| err.to_string())?;
        let body = response.text().await.map_err(|err| err.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|err| err.to_string())?;
        let mut docs = Vec::new();

        if let Some(results) = value
            .get("query")
            .and_then(|query| query.get("search"))
            .and_then(Value::as_array)
        {
            for item in results.iter().take(limit) {
                let title = item
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .trim();
                if title.is_empty() {
                    continue;
                }

                let mut summary_url =
                    reqwest::Url::parse("https://en.wikipedia.org/api/rest_v1/page/summary/")
                        .map_err(|err| err.to_string())?;
                let page_title = title.replace(' ', "_");
                let mut segments = summary_url
                    .path_segments_mut()
                    .map_err(|_| "invalid wikipedia summary url".to_string())?;
                segments.pop_if_empty();
                segments.push(&page_title);
                drop(segments);

                let summary_response = match self.client.get(summary_url.clone()).send().await {
                    Ok(response) => response,
                    Err(_) => continue,
                };
                let summary_body = match summary_response.text().await {
                    Ok(body) => body,
                    Err(_) => continue,
                };
                let summary = match serde_json::from_str::<Value>(&summary_body) {
                    Ok(value) => value,
                    Err(_) => continue,
                };
                let extract = summary
                    .get("extract")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("snippet").and_then(Value::as_str))
                    .unwrap_or("");
                let normalized = input::normalize_text(extract);
                if normalized.is_empty() {
                    continue;
                }

                let source_url = summary
                    .get("content_urls")
                    .and_then(|urls| urls.get("desktop"))
                    .and_then(|desktop| desktop.get("page"))
                    .and_then(Value::as_str)
                    .unwrap_or(summary_url.as_str());

                docs.push(RetrievedDocument {
                    source_url: source_url.to_string(),
                    title: title.to_string(),
                    raw_content: extract.to_string(),
                    normalized_content: normalized,
                    retrieved_at: Utc::now(),
                    trust_score: 0.6,
                    cached: false,
                });
            }
        }

        Ok(docs)
    }

    async fn hydrate_documents(
        &self,
        docs: Vec<RetrievedDocument>,
        limit: usize,
    ) -> Vec<RetrievedDocument> {
        let mut hydrated = Vec::new();

        for (index, mut doc) in docs.into_iter().enumerate() {
            let already_substantive = doc.raw_content.split_whitespace().count() >= 40;
            if index < limit.min(3) && doc.source_url.starts_with("http") && !already_substantive {
                if let Ok(response) = self.client.get(&doc.source_url).send().await {
                    if let Ok(body) = response.text().await {
                        let normalized = self.normalize_content(&doc.source_url, &body);
                        if normalized.len() > doc.normalized_content.len().max(120) {
                            let hydrated = merge_document_content(&doc.raw_content, &normalized);
                            doc.raw_content = hydrated.clone();
                            doc.normalized_content = input::normalize_text(&hydrated);
                        }
                    }
                }
            }
            hydrated.push(doc);
        }

        hydrated
    }
}

fn dedup_documents(docs: &mut Vec<RetrievedDocument>) {
    let mut seen = Vec::new();
    docs.retain(|doc| {
        let key = if doc.source_url.is_empty() {
            doc.title.to_lowercase()
        } else {
            doc.source_url.to_lowercase()
        };
        if seen.contains(&key) {
            false
        } else {
            seen.push(key);
            true
        }
    });
}

fn query_variants(query: &SanitizedQuery) -> Vec<String> {
    let mut variants = Vec::new();
    push_query_variant(&mut variants, clean_search_query(&query.sanitized_query));
    push_query_variant(&mut variants, clean_search_query(&query.raw_query));

    for expansion in &query.semantic_expansions {
        let expansion = clean_search_query(expansion);
        if expansion.is_empty() {
            continue;
        }
        if let Some(base) = variants.first() {
            let combined = clean_search_query(&format!("{base} {expansion}"));
            push_query_variant(&mut variants, combined);
        }
        push_query_variant(&mut variants, expansion);
        if variants.len() >= 5 {
            break;
        }
    }

    variants
}

fn push_query_variant(variants: &mut Vec<String>, candidate: String) {
    if candidate.is_empty() || variants.contains(&candidate) {
        return;
    }
    variants.push(candidate);
}

fn clean_search_query(text: &str) -> String {
    let mut tokens = Vec::new();
    for token in text.split_whitespace() {
        let cleaned = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
            .to_lowercase();
        let cleaned = if cleaned.len() > 4 && cleaned.ends_with('s') && !cleaned.ends_with("ss") {
            cleaned[..cleaned.len() - 1].to_string()
        } else {
            cleaned
        };
        if cleaned.is_empty() {
            continue;
        }
        if !tokens.contains(&cleaned) {
            tokens.push(cleaned);
        }
    }
    tokens.join(" ")
}

fn has_grounded_candidate(docs: &[RetrievedDocument]) -> bool {
    docs.iter().any(|doc| {
        doc.raw_content.split_whitespace().count() >= 12
            && doc.trust_score >= 0.45
            && !doc.raw_content.trim().ends_with('?')
    })
}

fn merge_document_content(summary: &str, hydrated_body: &str) -> String {
    let summary = summary.trim();
    let hydrated_excerpt = hydrated_body
        .split_whitespace()
        .take(320)
        .collect::<Vec<_>>()
        .join(" ");

    match (summary.is_empty(), hydrated_excerpt.is_empty()) {
        (true, true) => String::new(),
        (false, true) => summary.to_string(),
        (true, false) => hydrated_excerpt,
        (false, false) => {
            if hydrated_excerpt.starts_with(summary) {
                hydrated_excerpt
            } else {
                format!("{summary}\n\n{hydrated_excerpt}")
            }
        }
    }
}

fn rank_documents_for_query(query: &str, docs: &mut [RetrievedDocument]) {
    let query_terms = normalized_query_terms(query);
    docs.sort_by(|left, right| {
        let right_score = retrieval_relevance_score(&query_terms, right);
        let left_score = retrieval_relevance_score(&query_terms, left);
        right_score.total_cmp(&left_score)
    });
}

fn retrieval_relevance_score(query_terms: &[String], doc: &RetrievedDocument) -> f32 {
    if query_terms.is_empty() {
        return doc.trust_score;
    }

    let title_terms = normalized_query_terms(&doc.title);
    let content_terms = normalized_query_terms(&doc.normalized_content);
    let title_overlap = overlap_ratio(query_terms, &title_terms);
    let content_overlap = overlap_ratio(query_terms, &content_terms);
    let phrase_match =
        if input::normalize_text(&doc.normalized_content).contains(&query_terms.join(" ")) {
            0.12
        } else {
            0.0
        };

    (0.42 * title_overlap) + (0.36 * content_overlap) + (0.22 * doc.trust_score) + phrase_match
}

fn normalized_query_terms(text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for token in input::normalize_text(text).split_whitespace() {
        let token = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
            .to_ascii_lowercase();
        if token.len() < 2 || terms.contains(&token) {
            continue;
        }
        terms.push(token);
    }
    terms
}

fn overlap_ratio(left: &[String], right: &[String]) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let overlap = left.iter().filter(|term| right.contains(term)).count();
    overlap as f32 / left.len().max(1) as f32
}

fn collect_json_text(value: &Value, parts: &mut Vec<String>) {
    match value {
        Value::Null => {}
        Value::Bool(boolean) => parts.push(boolean.to_string()),
        Value::Number(number) => parts.push(number.to_string()),
        Value::String(text) => parts.push(text.clone()),
        Value::Array(array) => {
            for item in array {
                collect_json_text(item, parts);
            }
        }
        Value::Object(map) => {
            for item in map.values() {
                collect_json_text(item, parts);
            }
        }
    }
}

fn average_trust(docs: &[RetrievedDocument]) -> f32 {
    if docs.is_empty() {
        0.0
    } else {
        docs.iter().map(|doc| doc.trust_score).sum::<f32>() / docs.len() as f32
    }
}
