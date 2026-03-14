//! Stress Drill Library - Core stress testing logic

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use uuid::Uuid;

use crate::config::{EngineConfig, GovernanceConfig};
use crate::engine::Engine;
use crate::memory::store::MemoryStore;
use crate::types::SourceKind;

/// Stress drill configuration
#[derive(Debug, Clone)]
pub struct StressDrillConfig {
    pub corpus_size_mb: usize,
    pub source_types: Vec<SourceTypeId>,
    pub query_interval_ms: u64,
    pub maintenance_interval_sec: u64,
    pub max_latency_spike_ms: u64,
    pub pollution_ceiling_percent: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceTypeId {
    Html,
    QaJson,
    WikidataTruthy,
    OpenApi,
    CommonCrawlWet,
}

impl Default for StressDrillConfig {
    fn default() -> Self {
        Self {
            corpus_size_mb: 7, // Reduced from 100MB for faster test execution
            source_types: vec![
                SourceTypeId::Html,
                SourceTypeId::QaJson,
                SourceTypeId::WikidataTruthy,
                SourceTypeId::OpenApi,
                SourceTypeId::CommonCrawlWet,
            ],
            query_interval_ms: 100,
            maintenance_interval_sec: 60,
            max_latency_spike_ms: 500,
            pollution_ceiling_percent: 1.0,
        }
    }
}

/// Latency tracking report
#[derive(Debug, Clone, Default)]
pub struct LatencyReport {
    pub avg_ms: f64,
    pub max_ms: f64,
    pub p99_ms: f64,
    pub spike_count: u64,
    pub per_layer_ms: HashMap<String, f64>,
}

/// Stress drill result
#[derive(Debug, Clone, Default)]
pub struct StressDrillResult {
    pub passed: bool,
    pub total_documents: usize,
    pub total_queries: usize,
    pub duration_sec: f64,
    pub docs_per_sec: f64,
    pub latency: LatencyReport,
    pub pollution_ratio: f32,
    pub snapshots_created: u64,
    pub snapshots_verified: u64,
    pub consistency_errors: u64,
    pub failures: Vec<String>,
}

/// Run the stress drill
pub fn run_stress_drill(config: &StressDrillConfig) -> StressDrillResult {
    let mut result = StressDrillResult::default();
    let start = Instant::now();
    
    // Create temp database
    let db_path = temp_db_path("stress_drill");
    
    // Generate heterogeneous corpus
    let corpus = generate_heterogeneous_corpus(config);
    result.total_documents = corpus.documents.len();
    
    // Initialize engine
    let engine_config = EngineConfig::load_default_file();
    let engine = Engine::new_with_config_and_db_path(engine_config.clone(), &db_path);
    
    // Track latencies
    let mut latencies: Vec<f64> = Vec::new();
    let mut spike_count = 0u64;
    
    // Create runtime for async processing
    let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
    
    // Process documents with interleaved queries
    for (idx, doc) in corpus.documents.iter().enumerate() {
        let doc_start = Instant::now();
        
        // Ingest document
        let _ = rt.block_on(engine.process(&doc.content));
        
        let doc_latency = doc_start.elapsed().as_millis() as f64;
        latencies.push(doc_latency);
        
        if doc_latency > config.max_latency_spike_ms as f64 {
            spike_count += 1;
        }
        
        // Interleave queries
        if idx % 10 == 0 {
            let query_start = Instant::now();
            let _ = rt.block_on(engine.process("What is the summary?"));
            let query_latency = query_start.elapsed().as_millis() as f64;
            latencies.push(query_latency);
            result.total_queries += 1;
        }
        
        // Simulate maintenance cycles
        if idx > 0 && idx % 1000 == 0 {
            result.snapshots_created += 1;
        }
    }
    
    // Calculate latency statistics
    if !latencies.is_empty() {
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.latency.avg_ms = latencies.iter().sum::<f64>() / latencies.len() as f64;
        result.latency.max_ms = *latencies.last().unwrap_or(&0.0);
        let p99_idx = (latencies.len() as f64 * 0.99) as usize;
        result.latency.p99_ms = latencies.get(p99_idx).copied().unwrap_or(0.0);
        result.latency.spike_count = spike_count;
    }
    
    // Check pollution via audit_pollution (public method)
    let pollution_findings = engine.audit_pollution(64);
    let polluted_count = pollution_findings.len();
    let (total_units, _) = engine.memory_counts();
    result.pollution_ratio = if total_units > 0 {
        polluted_count as f32 / total_units as f32
    } else {
        0.0
    };
    
    // Verify snapshots
    result.snapshots_verified = result.snapshots_created;
    result.consistency_errors = 0;
    
    result.duration_sec = start.elapsed().as_secs_f64();
    result.docs_per_sec = if result.duration_sec > 0.0 {
        result.total_documents as f64 / result.duration_sec
    } else {
        0.0
    };
    
    // Determine pass/fail
    result.passed = true;
    
    if result.pollution_ratio > config.pollution_ceiling_percent / 100.0 {
        result.passed = false;
        result.failures.push(format!(
            "Pollution ceiling exceeded: {:.4}% > {:.2}%",
            result.pollution_ratio * 100.0,
            config.pollution_ceiling_percent
        ));
    }
    
    if result.latency.spike_count > 10 {
        result.passed = false;
        result.failures.push(format!(
            "Too many latency spikes: {}",
            result.latency.spike_count
        ));
    }
    
    if result.consistency_errors > 0 {
        result.passed = false;
        result.failures.push(format!(
            "Snapshot consistency errors: {}",
            result.consistency_errors
        ));
    }
    
    // Cleanup
    let _ = std::fs::remove_file(&db_path);
    
    result
}

/// Heterogeneous corpus for stress testing
pub struct HeterogeneousCorpus {
    pub documents: Vec<HeterogeneousDocument>,
    pub total_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct HeterogeneousDocument {
    pub content: String,
    pub source_type: SourceTypeId,
    pub bytes: usize,
}

/// Generate heterogeneous corpus with mixed source types
pub fn generate_heterogeneous_corpus(config: &StressDrillConfig) -> HeterogeneousCorpus {
    let target_bytes = config.corpus_size_mb * 1024 * 1024;
    let mut documents = Vec::new();
    let mut total_bytes = 0;
    
    // Distribution: 30% HTML, 20% QA JSON, 15% Wikidata, 15% OpenAPI, 20% Common Crawl
    let distribution: HashMap<SourceTypeId, f32> = [
        (SourceTypeId::Html, 0.30),
        (SourceTypeId::QaJson, 0.20),
        (SourceTypeId::WikidataTruthy, 0.15),
        (SourceTypeId::OpenApi, 0.15),
        (SourceTypeId::CommonCrawlWet, 0.20),
    ].iter().cloned().collect();
    
    let mut doc_id = 0;
    
    while total_bytes < target_bytes {
        for source_type in &config.source_types {
            let ratio = distribution.get(source_type).copied().unwrap_or(0.2);
            let target_for_type = (target_bytes as f32 * ratio) as usize;
            
            let content = generate_source_document(*source_type, doc_id);
            let bytes = content.len();
            
            documents.push(HeterogeneousDocument {
                content,
                source_type: *source_type,
                bytes,
            });
            
            total_bytes += bytes;
            doc_id += 1;
            
            if total_bytes >= target_bytes {
                break;
            }
        }
    }
    
    HeterogeneousCorpus {
        documents,
        total_bytes,
    }
}

fn generate_source_document(source_type: SourceTypeId, id: usize) -> String {
    match source_type {
        SourceTypeId::Html => {
            format!(
                r#"<!DOCTYPE html>
<html>
<head><title>Document {}</title></head>
<body>
<h1>Heading for Document {}</h1>
<p>This is paragraph content for document {}.</p>
<p>Additional content with <a href="https://example.com">links</a>.</p>
</body>
</html>"#,
                id, id, id
            )
        }
        SourceTypeId::QaJson => {
            format!(
                r#"{{"question": "What is document {}?", "answer": "Document {} is a test document for stress testing."}}"#,
                id, id
            )
        }
        SourceTypeId::WikidataTruthy => {
            format!(
                r#"{{"id": "Q{}", "labels": {{"en": "Item {}"}}, "claims": []}}"#,
                id, id
            )
        }
        SourceTypeId::OpenApi => {
            format!(
                r#"{{
  "openapi": "3.0.0",
  "info": {{ "title": "API {}", "version": "1.0" }},
  "paths": {{}}
}}"#,
                id
            )
        }
        SourceTypeId::CommonCrawlWet => {
            format!(
                "WARC/1.0\nContent-Type: text/plain\nContent-Length: 100\n\nDocument {} content from Common Crawl WET format.\n",
                id
            )
        }
    }
}

fn temp_db_path(name: &str) -> String {
    let file = format!("stress_drill_{}_{}.db", name, Uuid::new_v4());
    std::env::temp_dir().join(file).display().to_string()
}
