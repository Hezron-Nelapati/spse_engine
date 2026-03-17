use crate::classification::builder::UnitBuilder;
use crate::classification::input;
use crate::classification::safety::TrustSafetyValidator;
use crate::config::{
    ClassificationConfig, EngineConfig, GovernanceConfig, QueryProcessingConfig, RetrievalIoConfig,
    TrustConfig, UnitBuilderConfig,
};
use crate::types::{
    DatabaseHealthMetrics, EvidenceState, MetadataSummary, RetrievedDocument, SanitizedQuery,
    TrainingSourceType,
};
use chrono::Utc;
use futures_util::future::join_all;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

// ============================================================================
// SearXNG Client (V14.2 L11 Architecture)
// ============================================================================

/// SearXNG search categories per V14.2 architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearxCategory {
    General,
    Science,
    IT,
    News,
    Files,
    Images,
    Videos,
    Music,
    Social,
}

impl SearxCategory {
    fn as_str(&self) -> &'static str {
        match self {
            Self::General => "general",
            Self::Science => "science",
            Self::IT => "it",
            Self::News => "news",
            Self::Files => "files",
            Self::Images => "images",
            Self::Videos => "videos",
            Self::Music => "music",
            Self::Social => "social",
        }
    }
}

/// SearXNG search result from API
#[derive(Debug, Clone, Deserialize)]
pub struct SearxResult {
    pub url: String,
    pub title: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub category: String,
    #[serde(default)]
    pub publishedDate: Option<String>,
}

/// SearXNG API response
#[derive(Debug, Clone, Deserialize)]
pub struct SearxResponse {
    pub query: String,
    #[serde(default)]
    pub results: Vec<SearxResult>,
    #[serde(default)]
    pub answers: Vec<String>,
    #[serde(default)]
    pub corrections: Vec<String>,
    #[serde(default)]
    pub infoboxes: Vec<SearxInfobox>,
    #[serde(default)]
    pub suggestions: Vec<String>,
    #[serde(default)]
    pub unresponsive_engines: Vec<Vec<String>>,
}

/// SearXNG infobox (knowledge graph style)
#[derive(Debug, Clone, Deserialize)]
pub struct SearxInfobox {
    #[serde(default)]
    pub infobox: String,
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub urls: Vec<SearxInfoboxUrl>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearxInfoboxUrl {
    pub title: String,
    pub url: String,
}

/// SearXNG client for L11 retrieval
pub struct SearxNGClient {
    client: Client,
    base_url: String,
    timeout_ms: u64,
}

impl SearxNGClient {
    pub fn new(base_url: &str, timeout_ms: u64) -> Self {
        let client = Client::builder()
            .user_agent("spse_engine/0.1 (educational research)")
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .expect("failed to build SearXNG client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            timeout_ms,
        }
    }

    /// Search SearXNG with specified categories
    pub async fn search(
        &self,
        query: &str,
        categories: &[SearxCategory],
        limit: usize,
        language: Option<&str>,
    ) -> Result<SearxResponse, String> {
        let categories_str = categories
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
            .join(",");

        let encoded_query = Self::encode_query(query);
        let lang = language.unwrap_or("en");

        let url = format!(
            "{}/search?q={}&format=json&categories={}&language={}&safesearch=0&pageno=1",
            self.base_url, encoded_query, categories_str, lang
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("SearXNG request failed: {}", e))?;

        if !response.status().is_success() {
            return Err(format!("SearXNG returned status: {}", response.status()));
        }

        let body = response
            .text()
            .await
            .map_err(|e| format!("SearXNG body read failed: {}", e))?;

        let mut searx_response: SearxResponse = serde_json::from_str(&body).map_err(|e| {
            format!(
                "SearXNG JSON parse failed: {} - body: {}",
                e,
                &body[..body.len().min(200)]
            )
        })?;

        // Truncate results to limit
        searx_response.results.truncate(limit);

        Ok(searx_response)
    }

    /// Check if SearXNG instance is available
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/healthz", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => {
                // Try search endpoint as fallback health check
                let test_url = format!("{}/search?q=test&format=json", self.base_url);
                self.client.get(&test_url).send().await.is_ok()
            }
        }
    }

    /// Convert SearXNG results to RetrievedDocuments
    pub fn results_to_documents(&self, response: &SearxResponse) -> Vec<RetrievedDocument> {
        let mut docs = Vec::new();

        // Process direct answers first (highest trust)
        for answer in &response.answers {
            if !answer.is_empty() {
                let normalized = input::normalize_text(answer);
                if !normalized.is_empty() {
                    docs.push(build_retrieved_document(
                        format!("searxng://answer/{}", response.query),
                        "Direct Answer".to_string(),
                        answer.clone(),
                        normalized,
                        0.85, // High trust for direct answers
                    ));
                }
            }
        }

        // Process infoboxes (knowledge graph - high trust)
        for infobox in &response.infoboxes {
            if !infobox.content.is_empty() {
                let normalized = input::normalize_text(&infobox.content);
                if !normalized.is_empty() {
                    let source_url = infobox
                        .urls
                        .first()
                        .map(|u| u.url.clone())
                        .unwrap_or_else(|| format!("searxng://infobox/{}", infobox.id));

                    docs.push(build_retrieved_document(
                        source_url,
                        infobox.infobox.clone(),
                        infobox.content.clone(),
                        normalized,
                        0.80, // High trust for infoboxes
                    ));
                }
            }
        }

        // Process search results
        for result in &response.results {
            if result.url.is_empty() || result.content.is_empty() {
                continue;
            }

            let normalized = input::normalize_text(&result.content);
            if normalized.is_empty() {
                continue;
            }

            // Trust based on engine source
            let base_trust = Self::engine_trust(&result.engine);

            // Boost trust if score is provided and high
            let trust = if result.score > 0.0 {
                (base_trust + (result.score as f32 * 0.1)).min(0.90)
            } else {
                base_trust
            };

            docs.push(build_retrieved_document(
                result.url.clone(),
                result.title.clone(),
                result.content.clone(),
                normalized,
                trust,
            ));
        }

        docs
    }

    fn encode_query(query: &str) -> String {
        query
            .replace(' ', "+")
            .replace('&', "%26")
            .replace('?', "%3F")
            .replace('#', "%23")
            .replace('=', "%3D")
            .replace('/', "%2F")
    }

    fn engine_trust(engine: &str) -> f32 {
        match engine.to_lowercase().as_str() {
            "wikipedia" => 0.75,
            "wikidata" => 0.70,
            "wolfram alpha" | "wolframalpha" => 0.80,
            "duckduckgo" => 0.55,
            "google" => 0.60,
            "bing" => 0.55,
            "qwant" => 0.50,
            "brave" => 0.55,
            "startpage" => 0.55,
            "mojeek" => 0.45,
            "yandex" => 0.45,
            "yahoo" => 0.50,
            "pubmed" => 0.85,
            "arxiv" => 0.80,
            "semantic scholar" => 0.75,
            "stackoverflow" => 0.65,
            "github" => 0.60,
            _ => 0.45,
        }
    }

    /// Determine best categories for a query.
    /// Architecture does not specify keyword-based category detection.
    /// Returns default General category - actual categorization should come from memory-backed pattern matching.
    pub fn categorize_query(_query: &str) -> Vec<SearxCategory> {
        vec![SearxCategory::General]
    }
}

#[cfg(feature = "gpu")]
use once_cell::sync::Lazy;

/// Check if query is medical/scientific in nature.
/// Architecture does not specify keyword-based detection.
/// Returns false - actual detection should come from memory-backed pattern matching.
fn is_medical_query(_query_lower: &str) -> bool {
    false
}

/// Check if query is location/geographic in nature.
/// Architecture does not specify keyword-based detection.
/// Returns false - actual detection should come from memory-backed pattern matching.
fn is_location_query(_query_lower: &str) -> bool {
    false
}

#[derive(Clone)]
struct CacheEntry {
    documents: Vec<RetrievedDocument>,
    inserted_at: Instant,
}

pub struct RetrievalPipeline {
    client: Client,
    searxng: SearxNGClient,
    cache: Mutex<HashMap<String, CacheEntry>>,
    builder: UnitBuilderConfig,
    classification: ClassificationConfig,
    governance: GovernanceConfig,
    query_processing: QueryProcessingConfig,
    searxng_enabled: bool,
}

impl RetrievalPipeline {
    pub fn new(config: &EngineConfig) -> Self {
        let client = Client::builder()
            .user_agent("spse_engine/0.1 (educational research project)")
            .timeout(std::time::Duration::from_millis(
                config.retrieval_io.retrieval_timeout_ms,
            ))
            .build()
            .expect("failed to build reqwest client");

        let searxng = SearxNGClient::new(
            &config.retrieval_io.searxng_url,
            config.retrieval_io.retrieval_timeout_ms,
        );

        Self {
            client,
            searxng,
            cache: Mutex::new(HashMap::new()),
            builder: config.builder.clone(),
            classification: config.classification.clone(),
            governance: config.governance.clone(),
            query_processing: config.query_processing.clone(),
            searxng_enabled: config.retrieval_io.searxng_enabled,
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
                let packet = input::ingest_raw_with_config(
                    &doc.normalized_content,
                    false,
                    &self.classification,
                );
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
            TrainingSourceType::Document
            | TrainingSourceType::StructuredJson
            | TrainingSourceType::QaJson => Ok(self.normalize_content("document", value)),
            unsupported => Err(format!(
                "training source type {:?} is not permitted by SPSE_ARCHITECTURE_V14.2; use internal generated structured sources or local documents",
                unsupported
            )),
        }
    }

    fn normalize_content(&self, source_url: &str, raw: &str) -> String {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            let mut parts = Vec::new();
            collect_json_text(&value, &mut parts);
            let text = parts.join(" ");
            return input::normalize_text_with_config(&text, &self.classification);
        }

        let parsed = Html::parse_document(raw);
        let selector = Selector::parse("body").expect("valid selector");
        let mut text = String::new();
        for node in parsed.select(&selector) {
            text.push_str(&node.text().collect::<Vec<_>>().join(" "));
            text.push(' ');
        }

        let normalized = input::normalize_text_with_config(
            if text.is_empty() { raw } else { &text },
            &self.classification,
        );
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
        self.fetch_search_documents_with_config(query, limit, "http://localhost:8080", true)
            .await
    }

    /// L11 retrieval per V14.2 architecture - SearXNG as primary source
    async fn fetch_search_documents_with_config(
        &self,
        query: &SanitizedQuery,
        limit: usize,
        searxng_url: &str,
        searxng_enabled: bool,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let query_lower = query.raw_query.to_lowercase();
        let is_medical_query = is_medical_query(&query_lower);
        let is_location_query = is_location_query(&query_lower);

        // Fetch from all sources in parallel for each query variant
        let mut all_docs = Vec::new();
        let mut all_errors = Vec::new();

        for variant in query_variants(query, self.query_processing.max_query_variants, self.query_processing.plural_detection_min_length) {
            // Execute all fetches in parallel using boxed futures
            let mut futures_vec: Vec<
                std::pin::Pin<
                    Box<
                        dyn std::future::Future<Output = Result<Vec<RetrievedDocument>, String>>
                            + Send,
                    >,
                >,
            > = Vec::new();

            // V14.2: SearXNG as primary search source (L11)
            if searxng_enabled {
                futures_vec.push(Box::pin(self.fetch_searxng_documents(
                    &variant,
                    searxng_url,
                    5,
                )));
            }

            // Supplementary sources
            futures_vec.push(Box::pin(self.fetch_wikipedia_documents(&variant, 3)));
            futures_vec.push(Box::pin(self.fetch_wikidata_documents(&variant, 2)));

            // Conditional sources
            if is_medical_query {
                futures_vec.push(Box::pin(self.fetch_pubmed_central_documents(&variant, 2)));
            }
            if is_location_query {
                futures_vec.push(Box::pin(self.fetch_nominatim_documents(&variant, 2)));
            }

            // Execute all fetches in parallel
            let results = join_all(futures_vec).await;

            // Collect results
            for result in results {
                match result {
                    Ok(docs) => all_docs.extend(docs),
                    Err(err) => all_errors.push(err),
                }
            }

            // Deduplicate and rank after each variant
            dedup_documents(&mut all_docs);
            rank_documents_for_query(&variant, &mut all_docs, self.query_processing.min_token_length);

            // Check if we have enough grounded content
            if all_docs.len() >= limit && has_grounded_candidate(&all_docs) {
                break;
            }
        }

        all_docs.truncate(limit);

        if all_docs.is_empty() {
            return Err(format!("All sources failed: {}", all_errors.join("; ")));
        }

        Ok(self.hydrate_documents(all_docs, limit).await)
    }

    /// Fetch documents from SearXNG using the SearxNGClient (V14.2 L11)
    async fn fetch_searxng_documents_via_client(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let categories = SearxNGClient::categorize_query(query);
        let response = self
            .searxng
            .search(query, &categories, limit, Some("en"))
            .await?;
        Ok(self.searxng.results_to_documents(&response))
    }

    /// Fetch documents from SearXNG metasearch engine (V14.2 L11) - direct method
    async fn fetch_searxng_documents(
        &self,
        query: &str,
        searxng_url: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        // Use the SearxNGClient for proper structured parsing
        let client = SearxNGClient::new(searxng_url, 2000);
        let categories = SearxNGClient::categorize_query(query);
        let response = client.search(query, &categories, limit, Some("en")).await?;
        Ok(client.results_to_documents(&response))
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
                docs.push(build_retrieved_document(
                    source_url.to_string(),
                    value
                        .get("Heading")
                        .and_then(Value::as_str)
                        .unwrap_or("DuckDuckGo Abstract")
                        .to_string(),
                    abstract_text.to_string(),
                    normalized,
                    0.5,
                ));
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

                    // Filter out disambiguation-style URLs (e.g., "Donald_Trump_(song)")
                    // These are secondary pages, not the main entity
                    let is_disambiguation = url.contains("_(") && url.contains(")");
                    if is_disambiguation {
                        continue;
                    }

                    let normalized = input::normalize_text(text);
                    if !normalized.is_empty() {
                        docs.push(build_retrieved_document(
                            if url.is_empty() {
                                "https://duckduckgo.com/".to_string()
                            } else {
                                url.to_string()
                            },
                            text.split(" - ")
                                .next()
                                .unwrap_or("Related Topic")
                                .to_string(),
                            text.to_string(),
                            normalized,
                            0.45,
                        ));
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

                            // Filter out disambiguation-style URLs
                            let is_disambiguation = url.contains("_(") && url.contains(")");
                            if is_disambiguation {
                                continue;
                            }

                            let normalized = input::normalize_text(text);
                            if !normalized.is_empty() {
                                docs.push(build_retrieved_document(
                                    if url.is_empty() {
                                        "https://duckduckgo.com/".to_string()
                                    } else {
                                        url.to_string()
                                    },
                                    text.split(" - ")
                                        .next()
                                        .unwrap_or("Related Topic")
                                        .to_string(),
                                    text.to_string(),
                                    normalized,
                                    0.45,
                                ));
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
                docs.push(build_retrieved_document(
                    format!("https://www.wikidata.org/wiki/{entity_id}"),
                    label.to_string(),
                    raw_content,
                    normalized,
                    0.68,
                ));
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
            docs.push(build_retrieved_document(
                format!("https://pmc.ncbi.nlm.nih.gov/articles/PMC{id}/"),
                title.to_string(),
                raw_content.clone(),
                raw_content,
                0.74,
            ));
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
                docs.push(build_retrieved_document(
                    "https://nominatim.openstreetmap.org/".to_string(),
                    title.to_string(),
                    raw_content.clone(),
                    raw_content,
                    0.7,
                ));
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
                    Err(_) => {
                        // Add delay on error to avoid rate limiting
                        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        continue;
                    }
                };
                // Add small delay between requests to avoid rate limiting
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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

                docs.push(build_retrieved_document(
                    source_url.to_string(),
                    title.to_string(),
                    extract.to_string(),
                    normalized,
                    0.6,
                ));
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
            let already_substantive =
                doc.raw_content.split_whitespace().count() >= self.query_processing.substantive_word_count;
            if index < limit.min(self.query_processing.max_hydrate_docs)
                && doc.source_url.starts_with("http")
                && !already_substantive
            {
                if let Ok(response) = self.client.get(&doc.source_url).send().await {
                    if let Ok(body) = response.text().await {
                        let normalized = self.normalize_content(&doc.source_url, &body);
                        if normalized.len()
                            > doc.normalized_content.len().max(self.query_processing.min_hydrate_content_length)
                        {
                            let hydrated = merge_document_content(&doc.raw_content, &normalized);
                            doc.raw_content = hydrated.clone();
                            doc.normalized_content = input::normalize_text(&hydrated);
                            doc.metadata_summary =
                                extract_metadata_summary(&doc.title, &doc.normalized_content);
                        }
                    }
                }
            }
            hydrated.push(doc);
        }

        hydrated
    }
}

fn build_retrieved_document(
    source_url: String,
    title: String,
    raw_content: String,
    normalized_content: String,
    trust_score: f32,
) -> RetrievedDocument {
    let metadata_summary = extract_metadata_summary(&title, &normalized_content);
    RetrievedDocument {
        source_url,
        title,
        raw_content,
        normalized_content,
        retrieved_at: Utc::now(),
        trust_score,
        cached: false,
        metadata_summary,
    }
}

fn extract_metadata_summary(title: &str, normalized_content: &str) -> MetadataSummary {
    let entity = if title.trim().is_empty() {
        "document".to_string()
    } else {
        input::normalize_text(title).to_ascii_lowercase()
    };
    let mut summary = MetadataSummary::default();

    for captures in numeric_claim_regex().captures_iter(normalized_content) {
        let Some(value) = captures.name("value") else {
            continue;
        };
        let Ok(parsed_value) = value.as_str().parse::<f64>() else {
            continue;
        };
        let unit = captures
            .name("unit")
            .map(|unit| unit.as_str().to_ascii_lowercase())
            .unwrap_or_else(|| "count".to_string());
        summary.numbers.push((entity.clone(), parsed_value, unit));
    }

    for captures in date_claim_regex().captures_iter(normalized_content) {
        let Some(year_match) = captures.name("year") else {
            continue;
        };
        let Ok(year) = year_match.as_str().parse::<u16>() else {
            continue;
        };
        let relation = captures
            .name("relation")
            .map(|relation| relation.as_str().to_ascii_lowercase())
            .unwrap_or_else(|| "date".to_string());
        summary.dates.push((entity.clone(), relation, year));
    }

    for captures in property_claim_regex().captures_iter(normalized_content) {
        let property = captures
            .name("property")
            .map(|property| property.as_str().trim().to_ascii_lowercase())
            .unwrap_or_default();
        let value = captures
            .name("value")
            .map(|value| {
                value
                    .as_str()
                    .trim()
                    .trim_matches(|character: char| ".!?,".contains(character))
                    .to_ascii_lowercase()
            })
            .unwrap_or_default();
        if property.is_empty() || value.is_empty() {
            continue;
        }
        summary.properties.push((entity.clone(), property, value));
    }

    summary
}

fn numeric_claim_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b
            (?P<value>\d+(?:\.\d+)?)
            \s*
            (?P<unit>%|percent|year|years|km|kilometers|kg|usd|million|billion|people|citizens|articles?)?
            \b",
        )
        .expect("numeric claim regex must compile")
    })
}

fn date_claim_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b
            (?P<relation>founded|established|created|born|died|published|launched|introduced|dissolved)
            \s*
            (?:in|on)?
            \s*
            (?P<year>\d{4})
            \b",
        )
        .expect("date claim regex must compile")
    })
}

fn property_claim_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?ix)
            \b
            (?P<property>capital|population|currency|ceo|founder|headquarters|president|governor)
            \s+
            (?:is|was|are|were)\s+
            (?P<value>[a-z0-9][a-z0-9 ,.-]{1,60})",
        )
        .expect("property claim regex must compile")
    })
}

fn dedup_documents(docs: &mut Vec<RetrievedDocument>) {
    use crate::common::dedup::DedupUtils;

    DedupUtils::dedup_by_primary_field(
        docs,
        |doc| doc.source_url.as_str(),
        |doc| doc.title.as_str(),
    );
}

fn query_variants(query: &SanitizedQuery, max_variants: usize, plural_min_length: usize) -> Vec<String> {
    let mut variants = Vec::new();
    push_query_variant(&mut variants, clean_search_query(&query.sanitized_query, plural_min_length));
    push_query_variant(&mut variants, clean_search_query(&query.raw_query, plural_min_length));

    for expansion in &query.semantic_expansions {
        let expansion = clean_search_query(expansion, plural_min_length);
        if expansion.is_empty() {
            continue;
        }
        if let Some(base) = variants.first() {
            let combined = clean_search_query(&format!("{base} {expansion}"), plural_min_length);
            push_query_variant(&mut variants, combined);
        }
        push_query_variant(&mut variants, expansion);
        if variants.len() >= max_variants {
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

fn clean_search_query(text: &str, plural_min_length: usize) -> String {
    let mut tokens = Vec::new();
    let mut seen = HashSet::new();
    for token in text.split_whitespace() {
        let cleaned = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
            .to_lowercase();
        let cleaned = if cleaned.len() > plural_min_length && cleaned.ends_with('s') && !cleaned.ends_with("ss") {
            cleaned[..cleaned.len() - 1].to_string()
        } else {
            cleaned
        };
        if cleaned.is_empty() || !seen.insert(cleaned.clone()) {
            continue;
        }
        tokens.push(cleaned);
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

fn rank_documents_for_query(query: &str, docs: &mut [RetrievedDocument], min_token_length: usize) {
    let query_terms = normalized_query_terms(query, min_token_length);
    docs.sort_by(|left, right| {
        let right_score = retrieval_relevance_score(&query_terms, right, min_token_length);
        let left_score = retrieval_relevance_score(&query_terms, left, min_token_length);
        right_score.total_cmp(&left_score)
    });
}

fn retrieval_relevance_score(query_terms: &[String], doc: &RetrievedDocument, min_token_length: usize) -> f32 {
    if query_terms.is_empty() {
        return doc.trust_score;
    }

    let title_terms = normalized_query_terms(&doc.title, min_token_length);
    let content_terms = normalized_query_terms(&doc.normalized_content, min_token_length);
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

fn normalized_query_terms(text: &str, min_token_length: usize) -> Vec<String> {
    let mut terms = Vec::new();
    for token in input::normalize_text(text).split_whitespace() {
        let token = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
            .to_ascii_lowercase();
        if token.len() < min_token_length || terms.contains(&token) {
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

// ============================================================================
// Phase 5.1: Multi-Engine Consensus Retrieval
// ============================================================================

use crate::config::MultiEngineConfig;

/// Result from a single search engine
#[derive(Debug, Clone)]
pub struct EngineResult {
    pub engine_name: String,
    pub documents: Vec<RetrievedDocument>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Consensus-scored document from multiple engines
#[derive(Debug, Clone)]
pub struct ConsensusDocument {
    pub document: RetrievedDocument,
    /// Number of engines that returned this document (or similar)
    pub agreement_count: usize,
    /// Weighted consensus score
    pub consensus_score: f32,
    /// Unique engines that contributed
    pub contributing_engines: Vec<String>,
}

/// Multi-engine aggregator for consensus retrieval
pub struct MultiEngineAggregator {
    config: MultiEngineConfig,
    client: Client,
}

impl MultiEngineAggregator {
    pub fn new(config: MultiEngineConfig) -> Self {
        let client = Client::builder()
            .user_agent("spse_engine/0.1")
            .timeout(std::time::Duration::from_millis(config.engine_timeout_ms))
            .build()
            .expect("failed to build reqwest client for multi-engine");

        Self { config, client }
    }

    /// Aggregate results from multiple search engines with consensus scoring
    pub async fn aggregate(&self, query: &str, limit: usize) -> Vec<ConsensusDocument> {
        if !self.config.enabled {
            return Vec::new();
        }

        // Query all engines in parallel
        let engine_results = self.query_all_engines(query, limit).await;

        // Apply consensus scoring
        self.apply_consensus_scoring(engine_results)
    }

    /// Query all configured engines in parallel
    async fn query_all_engines(&self, query: &str, limit: usize) -> Vec<EngineResult> {
        let mut results = Vec::new();
        let query_lower = query.to_lowercase();

        // Determine which engines are relevant based on query content
        let is_medical = is_medical_query(&query_lower);
        let is_location = is_location_query(&query_lower);

        // Always start with Wikipedia for general knowledge (highest priority)
        if results.len() < self.config.max_engines {
            let start = Instant::now();
            match self.query_wikipedia(query, limit).await {
                Ok(docs) => {
                    results.push(EngineResult {
                        engine_name: "wikipedia".to_string(),
                        documents: docs,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(EngineResult {
                        engine_name: "wikipedia".to_string(),
                        documents: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(e),
                    });
                }
            }
        }

        // DuckDuckGo for general web search
        if results.len() < self.config.max_engines {
            let start = Instant::now();
            match self.query_duckduckgo(query, limit).await {
                Ok(docs) => {
                    results.push(EngineResult {
                        engine_name: "duckduckgo".to_string(),
                        documents: docs,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(EngineResult {
                        engine_name: "duckduckgo".to_string(),
                        documents: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(e),
                    });
                }
            }
        }

        // Wikidata for structured data
        if results.len() < self.config.max_engines {
            let start = Instant::now();
            match self.query_wikidata(query, limit).await {
                Ok(docs) => {
                    results.push(EngineResult {
                        engine_name: "wikidata".to_string(),
                        documents: docs,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(EngineResult {
                        engine_name: "wikidata".to_string(),
                        documents: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(e),
                    });
                }
            }
        }

        // PubMed Central - ONLY for medical/scientific queries
        if is_medical && results.len() < self.config.max_engines {
            let start = Instant::now();
            match self.query_pubmed(query, limit).await {
                Ok(docs) => {
                    results.push(EngineResult {
                        engine_name: "pubmed".to_string(),
                        documents: docs,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(EngineResult {
                        engine_name: "pubmed".to_string(),
                        documents: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(e),
                    });
                }
            }
        }

        // Nominatim (OpenStreetMap) - only for location queries
        if is_location && results.len() < self.config.max_engines {
            let start = Instant::now();
            match self.query_nominatim(query, limit).await {
                Ok(docs) => {
                    results.push(EngineResult {
                        engine_name: "nominatim".to_string(),
                        documents: docs,
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(EngineResult {
                        engine_name: "nominatim".to_string(),
                        documents: Vec::new(),
                        latency_ms: start.elapsed().as_millis() as u64,
                        error: Some(e),
                    });
                }
            }
        }

        results
    }

    /// Apply consensus scoring to aggregate results
    fn apply_consensus_scoring(&self, engine_results: Vec<EngineResult>) -> Vec<ConsensusDocument> {
        let mut consensus_docs: Vec<ConsensusDocument> = Vec::new();

        for result in &engine_results {
            if result.error.is_some() {
                continue;
            }

            for doc in &result.documents {
                // Find similar existing documents
                let similar_idx =
                    self.find_similar_document(&doc.normalized_content, &consensus_docs);

                if let Some(idx) = similar_idx {
                    // Increment agreement for similar document
                    consensus_docs[idx].agreement_count += 1;
                    consensus_docs[idx]
                        .contributing_engines
                        .push(result.engine_name.clone());

                    // Update consensus score
                    consensus_docs[idx].consensus_score = self.calculate_consensus_score(
                        &consensus_docs[idx].document,
                        consensus_docs[idx].agreement_count,
                        &consensus_docs[idx].contributing_engines,
                    );
                } else {
                    // New unique document
                    consensus_docs.push(ConsensusDocument {
                        document: doc.clone(),
                        agreement_count: 1,
                        consensus_score: self.calculate_consensus_score(
                            doc,
                            1,
                            &[result.engine_name.clone()],
                        ),
                        contributing_engines: vec![result.engine_name.clone()],
                    });
                }
            }
        }

        // Sort by consensus score descending
        consensus_docs.sort_by(|a, b| b.consensus_score.total_cmp(&a.consensus_score));

        consensus_docs
    }

    /// Find similar document in consensus list
    fn find_similar_document(&self, content: &str, docs: &[ConsensusDocument]) -> Option<usize> {
        let content_lower = content.to_lowercase();
        let content_words: Vec<&str> = content_lower.split_whitespace().take(20).collect();

        for (idx, cd) in docs.iter().enumerate() {
            let doc_lower = cd.document.normalized_content.to_lowercase();
            let doc_words: Vec<&str> = doc_lower.split_whitespace().take(20).collect();

            // Calculate word overlap
            let overlap = content_words
                .iter()
                .filter(|w| doc_words.contains(w))
                .count();

            let similarity = overlap as f32 / content_words.len().max(1) as f32;

            if similarity >= self.config.consensus_threshold {
                return Some(idx);
            }
        }

        None
    }

    /// Calculate consensus score for a document
    fn calculate_consensus_score(
        &self,
        doc: &RetrievedDocument,
        agreement_count: usize,
        engines: &[String],
    ) -> f32 {
        // Trust component
        let trust_score = doc.trust_score * self.config.trust_weight;

        // Agreement component (normalized by max engines)
        let agreement_score = (agreement_count as f32 / self.config.max_engines as f32)
            * self.config.agreement_weight;

        // Diversity component (unique engines)
        let unique_engines = engines
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let diversity_score =
            (unique_engines as f32 / self.config.max_engines as f32) * self.config.diversity_weight;

        (trust_score + agreement_score + diversity_score).clamp(0.0, 1.0)
    }

    // Individual engine query methods (wrappers around existing fetch methods)

    async fn query_duckduckgo(
        &self,
        query: &str,
        _limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_redirect=1&no_html=1",
            query.replace(' ', "+")
        );
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = response.text().await.map_err(|e| e.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|e| e.to_string())?;

        let mut docs = Vec::new();
        if let Some(abstract_text) = value.get("AbstractText").and_then(Value::as_str) {
            let normalized = input::normalize_text(abstract_text);
            if !normalized.is_empty() {
                docs.push(build_retrieved_document(
                    value
                        .get("AbstractURL")
                        .and_then(Value::as_str)
                        .unwrap_or("https://duckduckgo.com/")
                        .to_string(),
                    value
                        .get("Heading")
                        .and_then(Value::as_str)
                        .unwrap_or("DuckDuckGo Abstract")
                        .to_string(),
                    abstract_text.to_string(),
                    normalized,
                    0.5,
                ));
            }
        }
        Ok(docs)
    }

    async fn query_wikipedia(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let limit_str = limit.to_string();
        let url = reqwest::Url::parse_with_params(
            "https://en.wikipedia.org/w/api.php",
            &[
                ("action", "query"),
                ("list", "search"),
                ("format", "json"),
                ("utf8", "1"),
                ("srlimit", limit_str.as_str()),
                ("srsearch", query),
            ],
        )
        .map_err(|e| e.to_string())?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = response.text().await.map_err(|e| e.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|e| e.to_string())?;

        let mut docs = Vec::new();
        if let Some(results) = value
            .get("query")
            .and_then(|q| q.get("search"))
            .and_then(Value::as_array)
        {
            for item in results.iter().take(limit) {
                let title = item.get("title").and_then(Value::as_str).unwrap_or("");
                let snippet = item.get("snippet").and_then(Value::as_str).unwrap_or("");
                let normalized = input::normalize_text(snippet);
                if !normalized.is_empty() {
                    docs.push(build_retrieved_document(
                        format!("https://en.wikipedia.org/wiki/{}", title.replace(' ', "_")),
                        title.to_string(),
                        snippet.to_string(),
                        normalized,
                        0.6,
                    ));
                }
            }
        }
        Ok(docs)
    }

    async fn query_wikidata(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let limit_str = limit.to_string();
        let url = reqwest::Url::parse_with_params(
            "https://www.wikidata.org/w/api.php",
            &[
                ("action", "wbsearchentities"),
                ("language", "en"),
                ("format", "json"),
                ("type", "item"),
                ("limit", limit_str.as_str()),
                ("search", query),
            ],
        )
        .map_err(|e| e.to_string())?;

        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let body = response.text().await.map_err(|e| e.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|e| e.to_string())?;

        let mut docs = Vec::new();
        if let Some(items) = value.get("search").and_then(Value::as_array) {
            for item in items.iter().take(limit) {
                let label = item.get("label").and_then(Value::as_str).unwrap_or("");
                let description = item
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let entity_id = item.get("id").and_then(Value::as_str).unwrap_or("item");
                let raw = if description.is_empty() {
                    label.to_string()
                } else {
                    format!("{}: {}.", label, description)
                };
                let normalized = input::normalize_text(&raw);
                if !normalized.is_empty() {
                    docs.push(build_retrieved_document(
                        format!("https://www.wikidata.org/wiki/{}", entity_id),
                        label.to_string(),
                        raw,
                        normalized,
                        0.68,
                    ));
                }
            }
        }
        Ok(docs)
    }

    async fn query_pubmed(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let limit_str = limit.to_string();
        let search_url = reqwest::Url::parse_with_params(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            &[
                ("db", "pmc"),
                ("retmode", "json"),
                ("retmax", limit_str.as_str()),
                ("term", query),
            ],
        )
        .map_err(|e| e.to_string())?;

        let search_body = self
            .client
            .get(search_url)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .text()
            .await
            .map_err(|e| e.to_string())?;
        let search_value =
            serde_json::from_str::<Value>(&search_body).map_err(|e| e.to_string())?;

        let ids: Vec<String> = search_value
            .get("esearchresult")
            .and_then(|r| r.get("idlist"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .take(limit)
                    .collect()
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
        .map_err(|e| e.to_string())?;

        let summary_body = self
            .client
            .get(summary_url)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .text()
            .await
            .map_err(|e| e.to_string())?;
        let summary_value =
            serde_json::from_str::<Value>(&summary_body).map_err(|e| e.to_string())?;
        let result = summary_value
            .get("result")
            .and_then(Value::as_object)
            .ok_or_else(|| "invalid pmc summary payload".to_string())?;

        let mut docs = Vec::new();
        for id in ids {
            if let Some(item) = result.get(&id) {
                let title = item.get("title").and_then(Value::as_str).unwrap_or("");
                let journal = item
                    .get("fulljournalname")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let raw = input::normalize_text(&format!("{}. Journal: {}.", title, journal));
                if !raw.is_empty() {
                    docs.push(build_retrieved_document(
                        format!("https://pmc.ncbi.nlm.nih.gov/articles/PMC{}/", id),
                        title.to_string(),
                        raw.clone(),
                        raw,
                        0.74,
                    ));
                }
            }
        }
        Ok(docs)
    }

    async fn query_nominatim(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RetrievedDocument>, String> {
        let limit_str = limit.to_string();
        let url = reqwest::Url::parse_with_params(
            "https://nominatim.openstreetmap.org/search",
            &[
                ("format", "jsonv2"),
                ("limit", limit_str.as_str()),
                ("q", query),
            ],
        )
        .map_err(|e| e.to_string())?;

        let body = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .text()
            .await
            .map_err(|e| e.to_string())?;
        let value = serde_json::from_str::<Value>(&body).map_err(|e| e.to_string())?;

        let mut docs = Vec::new();
        if let Some(items) = value.as_array() {
            for item in items.iter().take(limit) {
                let display_name = item
                    .get("display_name")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let lat = item.get("lat").and_then(Value::as_str).unwrap_or("");
                let lon = item.get("lon").and_then(Value::as_str).unwrap_or("");
                let kind = item.get("type").and_then(Value::as_str).unwrap_or("");
                let raw = input::normalize_text(&format!(
                    "{}. Coordinates: {}, {}. Type: {}.",
                    display_name, lat, lon, kind
                ));
                if !raw.is_empty() {
                    docs.push(build_retrieved_document(
                        "https://nominatim.openstreetmap.org/".to_string(),
                        display_name.to_string(),
                        raw.clone(),
                        raw,
                        0.7,
                    ));
                }
            }
        }
        Ok(docs)
    }
}

/// Structured parser for known formats (HuggingFace, Wikipedia, custom sources)
pub struct StructuredParser;

impl StructuredParser {
    /// Parse HuggingFace dataset row format
    pub fn parse_huggingface_row(row: &Value) -> Option<String> {
        let mut parts = Vec::new();

        // Common HuggingFace fields
        for field in [
            "text",
            "content",
            "question",
            "context",
            "instruction",
            "prompt",
            "input",
            "output",
        ] {
            if let Some(text) = row.get(field).and_then(Value::as_str) {
                parts.push(text.to_string());
            }
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    }

    /// Parse Wikipedia XML format
    pub fn parse_wikipedia_xml(text: &str) -> String {
        // Extract article text from Wikipedia XML
        let mut result = String::new();
        let mut in_text = false;

        for line in text.lines() {
            if line.contains("<text") {
                in_text = true;
            }
            if in_text {
                // Strip XML tags
                let stripped = line
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&amp;", "&")
                    .replace("&quot;", "\"");
                result.push_str(&stripped);
                result.push(' ');
            }
            if line.contains("</text>") {
                in_text = false;
            }
        }

        input::normalize_text(&result)
    }

    /// Parse Wikidata truthy format
    pub fn parse_wikidata_truthy(entity: &Value) -> Option<String> {
        let label = entity
            .get("labels")
            .and_then(|l| l.get("en"))
            .and_then(|e| e.get("value"))
            .and_then(Value::as_str)?;

        let description = entity
            .get("descriptions")
            .and_then(|d| d.get("en"))
            .and_then(|e| e.get("value"))
            .and_then(Value::as_str)
            .unwrap_or("");

        let result = if description.is_empty() {
            label.to_string()
        } else {
            format!("{}: {}.", label, description)
        };

        Some(result)
    }

    /// Parse custom JSON source with field extraction
    pub fn parse_custom_json(value: &Value, fields: &[String]) -> String {
        let mut parts = Vec::new();

        for field in fields {
            if let Some(text) = value.get(field).and_then(|v| match v {
                Value::String(s) => Some(s.clone()),
                Value::Number(n) => Some(n.to_string()),
                Value::Bool(b) => Some(b.to_string()),
                Value::Array(arr) => Some(
                    arr.iter()
                        .filter_map(|item| item.as_str())
                        .collect::<Vec<_>>()
                        .join(", "),
                ),
                _ => None,
            }) {
                if !text.is_empty() {
                    parts.push(text);
                }
            }
        }

        parts.join(" ")
    }
}
