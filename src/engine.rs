use crate::config::{
    AdaptiveTrustProfile, EngineConfig, EscapeProfile, FineResolverConfig, IntentShapingConfig,
    ReasoningLoopConfig, RetrievalThresholds, ScoringWeights, TrustConfig,
};
use crate::datasets::{self, DatasetSinkControl, DatasetTextChunk, PreparationReport};
use crate::document::{
    answer_question, is_supported_document_path, load_document_bytes,
    load_documents_from_paths_with_config, load_path_content_with_config,
    parse_inline_document_request, validate_document_mime, DocumentAnswer, DocumentQueryState,
};
use crate::layers::builder::UnitBuilder;
use crate::layers::context::ContextManager;
use crate::layers::feedback::FeedbackController;
use crate::layers::hierarchy::HierarchicalUnitOrganizer;
use crate::layers::input;
use crate::layers::intent::IntentDetector;
use crate::layers::merge::EvidenceMerger;
use crate::layers::output::OutputDecoder;
use crate::layers::query::{focus_query_text, SafeQueryBuilder};
use crate::layers::resolver::FineResolver;
use crate::layers::retrieval::RetrievalPipeline;
use crate::layers::router::SemanticRouter;
use crate::layers::safety::TrustSafetyValidator;
use crate::layers::search::{top_unit_ids, CandidateScorer};
use crate::memory::store::{MemorySnapshot, MemoryStore};
use crate::open_sources;
use crate::scheduler::{PriorityScheduler, WorkPriority};
use crate::memory::{DynamicMemoryAllocator, DynamicMemoryConfig, MemoryStats};
use crate::telemetry::test_observer::{
    ObservationMemory, ObservationRetrieval, ObservationScoring, ObservationTimings,
    ObservationUnits, ScoreBreakdown as ObservedScoreBreakdown, TestObserver,
};
use crate::telemetry::{
    HotStore, LatencyMonitor, LatencyMonitorConfig, LatencyTimer, SessionId, TelemetryEvent,
    TelemetryWorker, TelemetryWorkerConfig, TraceContext, TraceId,
};
use crate::training::{self, TrainingPhasePlan, TrainingScope};
use crate::types::{
    ActivatedUnit, ConfidenceStats, ContextMatrix, DatabaseHealthMetrics, DebugStep, DryRunReport, ExplainTrace,
    InputPacket, IntentKind, IntentProfile, JobState, LayerNote, MemoryChannel, MemoryType,
    MergedState, OutputType, ProcessResult, QueueDepths, ReasoningResult, ReasoningState,
    ResolvedCandidate, ResolverMode, RetrievedDocument, RoutingResult, SearchDecision,
    SequenceState, SourceKind, ThoughtUnit, TrainBatchRequest, TrainingExecutionMode,
    TrainingJobStatus, TrainingMetrics, TrainingOptions, TrainingPhaseKind, TrainingPhaseStatus,
    TrainingSource, TrainingSourceType, Unit, UnitHierarchy,
};
use arc_swap::ArcSwap;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Default, Clone)]
struct SessionDocuments {
    documents: Vec<RetrievedDocument>,
    query_state: DocumentQueryState,
}

#[derive(Default)]
struct ChunkIngestionReport {
    chunks_committed: u64,
    activated_units_observed: u64,
    new_units_discovered: u64,
    reused_units_observed: u64,
    candidate_observations: u64,
    candidate_promotions: u64,
    anchors_observed: u64,
    map_adjustments: u64,
    mean_displacement: f32,
    cache_hits: u64,
    cache_lookups: u64,
    intent_counts: BTreeMap<String, u64>,
}

struct PreparedTrainingChunk {
    context_label: String,
    packet_summary: String,
    hierarchy: UnitHierarchy,
    activated_units_observed: u64,
    anchors_observed: u64,
    intent_key: Option<String>,
}

struct ShardTrainingResult {
    shard_index: usize,
    local_report: ChunkIngestionReport,
    segments: Vec<ShardSegmentResult>,
}

struct ShardSegmentResult {
    shard_index: usize,
    segment_index: usize,
    shard_db_path: PathBuf,
    pattern_cache_hits: u64,
    pattern_cache_lookups: u64,
    pruning_report: crate::types::GovernanceReport,
}

#[derive(Default)]
struct PhaseMetricAccumulator {
    activated_units_observed: u64,
    new_units_discovered: u64,
    reused_units_observed: u64,
    candidate_observations: u64,
    candidate_promotions: u64,
    map_adjustments: u64,
    anchors_observed: u64,
    mean_displacement_sum: f32,
    cache_hits: u64,
    cache_lookups: u64,
    chunks_seen: u64,
}

impl PhaseMetricAccumulator {
    fn observe(&mut self, report: &ChunkIngestionReport) {
        self.activated_units_observed += report.activated_units_observed;
        self.new_units_discovered += report.new_units_discovered;
        self.reused_units_observed += report.reused_units_observed;
        self.candidate_observations += report.candidate_observations;
        self.candidate_promotions += report.candidate_promotions;
        self.map_adjustments += report.map_adjustments;
        self.anchors_observed += report.anchors_observed;
        self.mean_displacement_sum += report.mean_displacement;
        self.cache_hits += report.cache_hits;
        self.cache_lookups += report.cache_lookups;
        self.chunks_seen += report.chunks_committed.max(1);
    }

    fn finalize(&self, memory_delta_kb: i64) -> TrainingMetrics {
        let activated = self.activated_units_observed.max(1) as f32;
        let new_unit_rate = (self.new_units_discovered as f32 / activated).clamp(0.0, 1.0);
        let unit_discovery_efficiency =
            (self.reused_units_observed as f32 / activated).clamp(0.0, 1.0);
        let mean_displacement = if self.chunks_seen == 0 {
            0.0
        } else {
            self.mean_displacement_sum / self.chunks_seen as f32
        };
        let semantic_routing_accuracy = (1.0 / (1.0 + mean_displacement)).clamp(0.0, 1.0);

        TrainingMetrics {
            new_unit_rate,
            unit_discovery_efficiency,
            semantic_routing_accuracy,
            prediction_error: new_unit_rate,
            memory_delta_kb,
            search_trigger_precision: None,
        }
    }
}

#[derive(Default)]
struct ReasoningSupport {
    unit_ids: Vec<Uuid>,
    confidence: f32,
    labels: Vec<String>,
}

#[derive(Clone)]
struct AdaptiveRuntimeSettings {
    intent_profile_name: Option<String>,
    trust_signal_name: Option<String>,
    trust_profile_name: Option<String>,
    scoring: ScoringWeights,
    escape: EscapeProfile,
    resolver: FineResolverConfig,
    resolver_mode: Option<ResolverMode>,
    trust: TrustConfig,
    retrieval: RetrievalThresholds,
    additional_cost_penalty: f32,
    shaping: IntentShapingConfig,
}

impl AdaptiveRuntimeSettings {
    fn from_config(config: &EngineConfig) -> Self {
        Self {
            intent_profile_name: None,
            trust_signal_name: None,
            trust_profile_name: None,
            scoring: config.scoring.clone(),
            escape: EscapeProfile {
                stochastic_jump_prob: 0.0,
                beam_width: config.governance.max_candidate_pool.max(1),
            },
            resolver: config.resolver.clone(),
            resolver_mode: None,
            trust: config.trust.clone(),
            retrieval: config.retrieval.clone(),
            additional_cost_penalty: 0.0,
            shaping: IntentShapingConfig::default(),
        }
    }
}

pub struct Engine {
    config: EngineConfig,
    memory: Arc<Mutex<MemoryStore>>,
    memory_snapshot: Arc<ArcSwap<MemorySnapshot>>,
    scheduler: Arc<PriorityScheduler>,
    retriever: RetrievalPipeline,
    merger: EvidenceMerger,
    decoder: OutputDecoder,
    feedback: FeedbackController,
    feedback_tx: Sender<Vec<crate::types::FeedbackEvent>>,
    safety: TrustSafetyValidator,
    jobs: Arc<Mutex<HashMap<String, TrainingJobStatus>>>,
    session_documents: Arc<Mutex<SessionDocuments>>,
    observer: Option<TestObserver>,
    /// Phase 4: Telemetry worker for async event emission
    telemetry_worker: Option<TelemetryWorker>,
    /// Phase 4: Latency monitor for p50/p95/p99 tracking
    latency_monitor: Arc<LatencyMonitor>,
    /// Phase 4: Dynamic memory allocator for reasoning buffers
    dynamic_memory: Arc<DynamicMemoryAllocator>,
    /// Phase 4: Trace context for session/trace ID management
    trace_context: Arc<Mutex<TraceContext>>,
}

impl Engine {
    pub fn new() -> Self {
        Self::new_with_db_path("spse_memory.db")
    }

    pub fn new_with_db_path(db_path: &str) -> Self {
        Self::new_with_config_and_db_path(EngineConfig::load_default_file(), db_path)
    }

    pub fn new_with_config_and_db_path(config: EngineConfig, db_path: &str) -> Self {
        let memory = Arc::new(Mutex::new(MemoryStore::new_with_config(
            db_path,
            &config.governance,
            &config.semantic_map,
        )));
        let memory_snapshot = {
            let memory = memory.lock().expect("memory mutex poisoned");
            Arc::new(ArcSwap::from_pointee(memory.snapshot()))
        };
        let scheduler = Arc::new(PriorityScheduler::new());
        let jobs = Arc::new(Mutex::new(HashMap::new()));
        let session_documents = Arc::new(Mutex::new(SessionDocuments::default()));
        let (feedback_tx, feedback_rx) = bounded(1024);
        let observer = TestObserver::from_config(&config);

        // Phase 4: Initialize telemetry worker
        let telemetry_worker = if config.telemetry.worker.enabled {
            TelemetryWorker::new(TelemetryWorkerConfig {
                enabled: config.telemetry.worker.enabled,
                hot_store_path: config.telemetry.worker.hot_store_path.clone(),
                cold_log_path: config.telemetry.worker.cold_log_path.clone(),
                batch_size: config.telemetry.worker.batch_size,
                flush_interval_ms: config.telemetry.worker.flush_interval_ms,
                channel_capacity: config.telemetry.worker.channel_capacity,
                sample_rate: config.telemetry.worker.sample_rate,
            }).ok()
        } else {
            None
        };

        // Phase 4: Initialize latency monitor
        let latency_monitor = Arc::new(LatencyMonitor::new(LatencyMonitorConfig {
            alert_threshold_ms: config.telemetry.latency_monitor.alert_threshold_ms,
            window_size: config.telemetry.latency_monitor.window_size,
            enabled: config.telemetry.latency_monitor.enabled,
            sample_rate: config.telemetry.latency_monitor.sample_rate,
        }));

        // Phase 4: Initialize dynamic memory allocator
        let dynamic_memory = Arc::new(DynamicMemoryAllocator::new(DynamicMemoryConfig {
            enabled: config.auto_inference.dynamic_memory.enabled,
            base_memory_limit_mb: config.auto_inference.dynamic_memory.base_memory_limit_mb,
            max_memory_limit_mb: config.auto_inference.dynamic_memory.max_memory_limit_mb,
            thought_buffer_size_kb: config.auto_inference.dynamic_memory.thought_buffer_size_kb,
        }));

        // Phase 4: Initialize trace context
        let trace_context = Arc::new(Mutex::new(TraceContext::new()));

        spawn_maintenance(
            memory.clone(),
            memory_snapshot.clone(),
            scheduler.clone(),
            config.clone(),
        );
        spawn_feedback_worker(
            memory.clone(),
            memory_snapshot.clone(),
            scheduler.clone(),
            feedback_rx,
        );

        let engine = Self {
            retriever: RetrievalPipeline::new(&config),
            config,
            memory,
            memory_snapshot,
            scheduler,
            merger: EvidenceMerger,
            decoder: OutputDecoder,
            feedback: FeedbackController,
            feedback_tx,
            safety: TrustSafetyValidator,
            jobs,
            session_documents,
            observer,
            telemetry_worker,
            latency_monitor,
            dynamic_memory,
            trace_context,
        };
        engine
    }

    pub async fn process(&self, text: &str) -> ProcessResult {
        let _permit = self.scheduler.acquire(WorkPriority::Inference);
        let inline_request = parse_inline_document_request(text);
        if !inline_request.paths.is_empty() {
            match load_documents_from_paths_with_config(
                &inline_request.paths,
                &self.config.document,
            ) {
                Ok(documents) => {
                    self.ingest_documents_into_core(&documents);
                    self.replace_session_documents(documents.clone());
                    let sources = documents
                        .iter()
                        .map(|doc| doc.source_url.clone())
                        .collect::<Vec<_>>();
                    if inline_request.prompt.is_empty() {
                        let names = documents
                            .iter()
                            .map(|doc| doc.title.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        return simple_result(
                            format!(
                                "Loaded {}. Ask a question about it. It will stay active for follow-up questions until you load another document or use /clear.",
                                names
                            ),
                            sources,
                            "inline_document_load",
                            text,
                        );
                    }
                    self.ingest_interaction_input(&inline_request.prompt, "inline_document_prompt");
                    let query_state = self.session_snapshot().query_state;
                    let sequence = self.current_sequence_state();
                    let intent_profile = self.resolve_intent_profile(
                        &inline_request.prompt,
                        &ContextMatrix::default(),
                        &sequence,
                        true,
                    );
                    if let Some(answer) = answer_question(
                        &inline_request.prompt,
                        &documents,
                        Some(&query_state),
                        Some(intent_profile.primary),
                    ) {
                        self.remember_document_turn(&inline_request.prompt, &answer);
                        return document_result(
                            &answer,
                            "inline_document_workspace",
                            &inline_request.prompt,
                            intent_profile,
                            self.memory_summary(),
                        );
                    }
                    return self
                        .process_prompt(
                            &inline_request.prompt,
                            Some(documents),
                            inline_request
                                .paths
                                .iter()
                                .map(|path| path.display().to_string())
                                .collect(),
                            true,
                        )
                        .await;
                }
                Err(err) => {
                    return simple_result(err, Vec::new(), "document_load_error", text);
                }
            }
        }

        if let Some(answer) = evaluate_expression_input(text) {
            self.ingest_interaction_input(text, "direct_expression");
            let expression_intent = if text.contains('?')
                || text.to_lowercase().starts_with("what is")
                || text.to_lowercase().starts_with("what's")
                || text.to_lowercase().starts_with("calculate")
            {
                IntentKind::Question
            } else {
                IntentKind::Analyze
            };
            return simple_result_with_intent(
                answer,
                Vec::new(),
                "direct_expression",
                text,
                IntentProfile {
                    primary: expression_intent,
                    confidence: 0.92,
                    reasons: vec!["symbolic_expression_detected".to_string()],
                    ..IntentProfile::default()
                },
                self.memory_summary(),
            );
        }

        let session = self.session_snapshot();
        let active_titles = self.active_session_document_titles();
        let pre_intent = self.resolve_intent_profile(
            text,
            &ContextMatrix::default(),
            &self.current_sequence_state(),
            !session.documents.is_empty(),
        );
        if matches!(pre_intent.primary, IntentKind::Forget) {
            self.ingest_interaction_input(text, "direct_interaction");
            let cleared = self.clear_session_documents();
            let message = if cleared == 0 {
                "No active document session to clear.".to_string()
            } else {
                format!("Cleared {} active document(s).", cleared)
            };
            return simple_result_with_intent(
                message,
                Vec::new(),
                "direct_interaction_forget",
                text,
                pre_intent,
                self.memory_summary(),
            );
        }
        if matches!(pre_intent.primary, IntentKind::Continue) {
            self.ingest_interaction_input(text, "direct_interaction");
            let message = if active_titles.is_empty() {
                "There is no active document session. Ask a question or load a document first."
                    .to_string()
            } else if active_titles.len() == 1 {
                format!(
                    "Continuing with {}. Ask the next question.",
                    active_titles[0]
                )
            } else {
                format!(
                    "Continuing with {} active documents. Ask the next question.",
                    active_titles.len()
                )
            };
            return simple_result_with_intent(
                message,
                Vec::new(),
                "direct_interaction_continue",
                text,
                pre_intent,
                self.memory_summary(),
            );
        }
        if matches!(
            pre_intent.fallback_mode,
            crate::types::IntentFallbackMode::ClarifyHelp
        ) {
            self.ingest_interaction_input(text, "direct_interaction");
            return simple_result_with_intent(
                "Please clarify what you want me to do. You can ask a question, reference the active document, or load a file.".to_string(),
                Vec::new(),
                "direct_interaction_clarify",
                text,
                pre_intent,
                self.memory_summary(),
            );
        }
        if self.config.intent.social_intent_shortcircuit {
            if let Some(reply) = casual_reply(&pre_intent, &active_titles) {
                self.ingest_interaction_input(text, "direct_interaction");
                return simple_result_with_intent(
                    reply,
                    Vec::new(),
                    "direct_interaction_social",
                    text,
                    pre_intent,
                    self.memory_summary(),
                );
            }
        }

        if !session.documents.is_empty() {
            if let Some(answer) = answer_question(
                text,
                &session.documents,
                Some(&session.query_state),
                Some(pre_intent.primary),
            ) {
                if answer.confidence >= self.config.resolver.evidence_answer_confidence_threshold {
                    self.ingest_interaction_input(text, "session_document_prompt");
                    self.remember_document_turn(text, &answer);
                    return document_result(
                        &answer,
                        "session_document_workspace",
                        text,
                        pre_intent,
                        self.memory_summary(),
                    );
                }
            }
        }

        if matches!(
            pre_intent.primary,
            IntentKind::Question | IntentKind::Verify | IntentKind::Extract | IntentKind::Explain
        ) {
            if let Some(answer) = self.answer_from_personal_memory(text, pre_intent.primary) {
                self.ingest_interaction_input(text, "personal_memory_prompt");
                return document_result(
                    &answer,
                    "personal_memory_workspace",
                    text,
                    pre_intent,
                    self.memory_summary(),
                );
            }
        }

        self.process_prompt(text, None, Vec::new(), true).await
    }

    pub fn clear_session_documents(&self) -> usize {
        let mut session = self
            .session_documents
            .lock()
            .expect("session documents mutex poisoned");
        let cleared = session.documents.len();
        *session = SessionDocuments::default();
        cleared
    }

    pub fn reset_session_query_state(&self) {
        let mut session = self
            .session_documents
            .lock()
            .expect("session documents mutex poisoned");
        session.query_state = DocumentQueryState::default();
    }

    pub fn active_session_document_titles(&self) -> Vec<String> {
        let session = self
            .session_documents
            .lock()
            .expect("session documents mutex poisoned");
        unique_strings(
            session
                .documents
                .iter()
                .map(|doc| doc.title.clone())
                .collect(),
        )
    }

    pub fn memory_counts(&self) -> (usize, usize) {
        let snapshot = self.load_memory_snapshot();
        let units = snapshot.unit_count();
        let core = snapshot
            .all_units()
            .into_iter()
            .filter(|unit| unit.memory_type == MemoryType::Core)
            .count();
        (units, core)
    }

    pub fn memory_summary(&self) -> String {
        self.load_memory_snapshot().memory_summary()
    }

    /// Phase 4: Emit a telemetry event
    pub fn emit_telemetry(&self, event: TelemetryEvent) {
        if let Some(worker) = &self.telemetry_worker {
            let _ = worker.emit(event);
        }
    }

    /// Phase 4: Get latency summary
    pub fn latency_summary(&self) -> crate::telemetry::LatencySummary {
        self.latency_monitor.summary()
    }

    /// Phase 4: Get dynamic memory stats
    pub fn dynamic_memory_stats(&self) -> MemoryStats {
        self.dynamic_memory.stats()
    }

    /// Phase 4: Get current trace context
    pub fn trace_context(&self) -> (SessionId, TraceId) {
        if let Ok(ctx) = self.trace_context.lock() {
            (ctx.session_id, ctx.trace_id)
        } else {
            (SessionId::new(), TraceId::new())
        }
    }

    /// Phase 4: Start a new trace for a query
    pub fn start_new_trace(&self) {
        if let Ok(mut ctx) = self.trace_context.lock() {
            ctx.start_new_trace();
        }
    }

    /// Phase 4: Create a latency timer for a layer
    pub fn latency_timer(&self, layer: u8) -> LatencyTimer {
        LatencyTimer::new(&self.latency_monitor, layer)
    }

    pub fn audit_pollution(&self, limit: usize) -> Vec<crate::types::PollutionFinding> {
        let memory = self.memory.lock().expect("memory mutex poisoned");
        memory.audit_pollution(limit)
    }

    fn load_memory_snapshot(&self) -> Arc<MemorySnapshot> {
        self.memory_snapshot.load_full()
    }

    fn publish_memory_snapshot(&self) {
        let snapshot = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            Arc::new(memory.snapshot())
        };
        self.memory_snapshot.store(snapshot);
    }

    pub async fn train(&self) -> TrainingJobStatus {
        self.train_with_execution_mode(TrainingExecutionMode::User)
            .await
    }

    pub async fn train_with_execution_mode(
        &self,
        execution_mode: TrainingExecutionMode,
    ) -> TrainingJobStatus {
        self.train_with_scope(execution_mode, TrainingScope::Full)
            .await
    }

    pub async fn train_with_scope(
        &self,
        execution_mode: TrainingExecutionMode,
        scope: TrainingScope,
    ) -> TrainingJobStatus {
        let _permit = self.scheduler.acquire(WorkPriority::SilentBatch);
        let job_id = format!("train_{}", Uuid::new_v4().simple());
        let plan = match self.resolve_training_plan_sources(
            training::build_training_plan_with_config(&self.config, execution_mode, scope),
            scope,
        ) {
            Ok(plan) => plan,
            Err(err) => {
                let mut status = TrainingJobStatus::queued(job_id, 0);
                status.status = JobState::Failed;
                status.warnings.push(format!("fatal:{err}"));
                self.store_job(status.clone());
                return status;
            }
        };
        self.run_training_plan_with_id(job_id, plan).await
    }

    pub async fn prepare_training_sources(
        &self,
        execution_mode: TrainingExecutionMode,
        scope: TrainingScope,
    ) -> Result<PreparationReport, String> {
        let _permit = self.scheduler.acquire(WorkPriority::SilentBatch);
        let plan = training::build_training_plan_with_config(&self.config, execution_mode, scope);
        let mut report = PreparationReport {
            scope: format!("{scope:?}").to_ascii_lowercase(),
            ..PreparationReport::default()
        };
        let mut seen = std::collections::HashSet::new();

        for source in plan.phases.iter().flat_map(|phase| phase.sources.iter()) {
            let resolved = open_sources::resolve_training_source(source)?;
            let record = datasets::prepare_training_source(&resolved).await?;
            if seen.insert(record.key.clone()) {
                report.sources.push(record);
            }
        }

        report
            .sources
            .sort_by(|lhs, rhs| lhs.source_name.cmp(&rhs.source_name));
        Ok(report)
    }

    pub async fn run_phase0_dry_run(&self, execution_mode: TrainingExecutionMode) -> DryRunReport {
        let status = self
            .train_with_scope(execution_mode, TrainingScope::DryRun)
            .await;
        self.finalize_phase0_dry_run(status).await
    }

    pub async fn finalize_phase0_dry_run(&self, status: TrainingJobStatus) -> DryRunReport {
        let (snapshot_path, snapshot_readable) = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            let db = memory.db();
            let path = db.snapshot_path();
            let readable = db.load_snapshot().ok().flatten().is_some();
            (path.display().to_string(), readable)
        };

        let probe_documents = {
            let raw_content = "High-intensity interval training (HIIT) stands for high-intensity interval training, a workout style built around short bursts of intense exercise separated by brief recovery periods. It alternates effort and recovery to keep sessions efficient while maintaining high exertion.";
            vec![RetrievedDocument {
                source_url: "memory://phase0_probe".to_string(),
                title: "phase0_probe".to_string(),
                raw_content: raw_content.to_string(),
                normalized_content: input::normalize_text(raw_content),
                retrieved_at: chrono::Utc::now(),
                trust_score: 0.99,
                cached: true,
            }]
        };

        // Warm the post-training inference path before measuring latency.
        let _ = self
            .process_prompt(
                "What does HIIT stand for?",
                Some(probe_documents.clone()),
                Vec::new(),
                true,
            )
            .await;

        let probe = "What does HIIT stand for?";
        let started = Instant::now();
        let result = self
            .process_prompt(probe, Some(probe_documents), Vec::new(), false)
            .await;
        let inference_latency_ms = started.elapsed().as_millis();
        // SPS is tokenizer-free, so use a byte-span token-equivalent instead of whitespace words.
        let token_count = usize::max(
            result.predicted_text.split_whitespace().count(),
            result.predicted_text.len().div_ceil(4),
        )
        .max(1) as u128;
        let latency_per_token_ms = inference_latency_ms / token_count;
        let map_stable = !status
            .warnings
            .iter()
            .any(|warning| warning.starts_with("layout_rollback:"));

        DryRunReport {
            status,
            snapshot_path,
            snapshot_readable,
            map_stable,
            inference_ok: !result.predicted_text.trim().is_empty() && latency_per_token_ms < 200,
            inference_latency_ms,
            latency_per_token_ms,
            query_result: result.predicted_text,
            memory_summary: self.memory_summary(),
        }
    }

    fn resolve_training_plan_sources(
        &self,
        mut plan: training::TrainingPlan,
        scope: TrainingScope,
    ) -> Result<training::TrainingPlan, String> {
        for phase in &mut plan.phases {
            for source in &mut phase.sources {
                let resolved = open_sources::resolve_training_source(source)?;
                *source = if matches!(scope, TrainingScope::HuggingFace) {
                    datasets::localize_remote_capable_source(&resolved)?
                } else {
                    datasets::localize_prepared_source(&resolved)?
                };
            }
        }
        Ok(plan)
    }

    async fn process_prompt(
        &self,
        text: &str,
        local_documents: Option<Vec<RetrievedDocument>>,
        mut preset_sources: Vec<String>,
        allow_retrieval: bool,
    ) -> ProcessResult {
        let total_start = Instant::now();
        let memory_before_kb = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.estimate_memory_kb()
        };

        let layer_2_start = Instant::now();
        let packet = input::ingest_raw(text, false);
        let build_output = self.build_units(&packet);
        let layer_2_time_ms = layer_2_start.elapsed().as_millis() as u64;
        let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &self.config.builder);
        let initial_context = summarize_packet(&packet);

        let active_ids = {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            memory.ingest_hierarchy(&hierarchy, SourceKind::UserInput, &initial_context)
        };
        self.publish_memory_snapshot();

        let snapshot = self.load_memory_snapshot();
        let active_units = snapshot.get_units(&active_ids);
        let all_units = snapshot.all_units();

        let layer_5_start = Instant::now();
        let routing = route_units(&active_units, &all_units, &self.config.semantic_map);
        let layer_5_routing_time_ms = layer_5_start.elapsed().as_millis() as u64;

        {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            memory.update_positions(&routing.position_updates);
            for active_id in &active_ids {
                memory.connect_units(*active_id, &routing.neighbor_ids, 0.35);
            }
        }
        self.publish_memory_snapshot();

        let (context_matrix, sequence_state, all_units) =
            self.prepare_context_and_candidates(&hierarchy, &routing);
        let has_document_context = local_documents.is_some() || !preset_sources.is_empty();
        let intent_profile = self.resolve_intent_profile(
            &packet.original_text,
            &context_matrix,
            &sequence_state,
            has_document_context,
        );
        let decision_queue_depths = self.scheduler.depths();
        let adaptive =
            self.resolve_adaptive_runtime(&intent_profile, decision_queue_depths.clone());
        let candidate_route = route_candidate_units(
            &routing,
            &all_units,
            self.config.governance.max_candidate_pool,
            &self.config.semantic_map,
            &adaptive.escape,
        );

        let reasoning_support = self.reasoning_support_from_memory(&packet.original_text);
        let mut candidate_ids = candidate_route.candidate_ids.clone();
        candidate_ids.extend(reasoning_support.unit_ids.iter().copied());
        candidate_ids.sort();
        candidate_ids.dedup();
        let candidate_units = {
            let snapshot = self.load_memory_snapshot();
            let mut units = snapshot.get_units(&candidate_ids);
            if units.is_empty() {
                units = snapshot.top_units(self.config.governance.max_candidate_pool);
            }
            units
        };
        let mut merged = MergedState {
            candidate_ids: candidate_ids.clone(),
            evidence_support: (reasoning_support.confidence * 0.20).clamp(0.0, 1.0),
            ..MergedState::default()
        };
        let initial_scored = CandidateScorer::score(
            &candidate_units,
            &context_matrix,
            &sequence_state,
            &merged,
            &adaptive.scoring,
            Some(intent_profile.primary),
            Some(&packet.original_text),
        );
        let confidence_stats = CandidateScorer::confidence_stats(&initial_scored);
        let layer_9_start = Instant::now();
        let decision = IntentDetector::assess(
            &context_matrix,
            &sequence_state,
            &confidence_stats,
            &initial_scored,
            &adaptive.retrieval,
            &packet.original_text,
            &intent_profile,
        );
        let layer_9_decision_time_ms = layer_9_start.elapsed().as_millis() as u64;

        let mut retrieval_query = None;
        let mut safety_warnings = Vec::new();
        let mut used_retrieval = false;
        let mut layer_11_retrieval_time_ms = 0u64;
        let mut layer_13_merge_time_ms = 0u64;

        if let Some(documents) = local_documents {
            let snapshot = self.load_memory_snapshot();
            let evidence = local_evidence_state(
                &documents,
                &self.config.builder,
                &self.config.governance,
                &self.current_database_health(),
                Some(snapshot.as_ref()),
            );
            preset_sources.extend(documents.iter().map(|doc| doc.source_url.clone()));
            let merged_input_units = snapshot.get_units(&candidate_route.candidate_ids);
            let layer_13_start = Instant::now();
            merged = self.merger.merge(
                &merged_input_units,
                &context_matrix,
                evidence,
                &self.config.evidence_merge,
            );
            layer_13_merge_time_ms = layer_13_start.elapsed().as_millis() as u64;
        } else if allow_retrieval
            && self.config.retrieval_io.enable_retrieval
            && decision.should_retrieve
        {
            let query = SafeQueryBuilder::build(
                &packet.original_text,
                &context_matrix,
                &sequence_state,
                &self.config.query,
                &adaptive.trust,
            );
            let layer_11_start = Instant::now();
            let evidence = self
                .retriever
                .search(
                    &query,
                    &self.safety,
                    &self.config.retrieval_io,
                    &adaptive.trust,
                    self.config.retrieval_io.max_retries,
                    &self.current_database_health(),
                )
                .await;
            layer_11_retrieval_time_ms = layer_11_start.elapsed().as_millis() as u64;
            safety_warnings.extend(evidence.warnings.clone());
            if !evidence.documents.is_empty() {
                self.ingest_web_evidence_into_memory(
                    &evidence.documents,
                    &query.sanitized_query,
                    &intent_profile,
                    &adaptive.trust,
                );
            }
            let evidence_hierarchy = hierarchy_from_activations(&evidence.evidence_units);
            let evidence_ids = {
                let mut memory = self.memory.lock().expect("memory mutex poisoned");
                memory.ingest_hierarchy(
                    &evidence_hierarchy,
                    SourceKind::Retrieval,
                    &query.sanitized_query,
                )
            };
            self.publish_memory_snapshot();

            let merged_input_units = {
                let snapshot = self.load_memory_snapshot();
                let mut units = snapshot.get_units(&candidate_ids);
                units.extend(snapshot.get_units(&evidence_ids));
                units
            };

            let layer_13_start = Instant::now();
            merged = self.merger.merge(
                &merged_input_units,
                &context_matrix,
                evidence,
                &self.config.evidence_merge,
            );
            layer_13_merge_time_ms = layer_13_start.elapsed().as_millis() as u64;
            merged.candidate_ids.extend(evidence_ids);
            merged.candidate_ids.sort();
            merged.candidate_ids.dedup();
            retrieval_query = Some(query);
            used_retrieval = true;
        }

        let final_candidates = {
            let snapshot = self.load_memory_snapshot();
            let all_units = snapshot.all_units();
            let mut ids = candidate_ids.clone();
            ids.extend(active_ids.iter().copied());
            ids.extend(merged.candidate_ids.iter().copied());
            ids.sort();
            ids.dedup();
            let mut units = snapshot.get_units(&ids);
            if units.is_empty() {
                units = all_units
                    .into_iter()
                    .take(self.config.governance.max_candidate_pool)
                    .collect();
            }
            if units
                .iter()
                .any(|unit| unit.level != crate::types::UnitLevel::Char)
            {
                units.retain(|unit| unit.level != crate::types::UnitLevel::Char);
            }
            units
        };

        let layer_14_start = Instant::now();
        let scored = CandidateScorer::score(
            &final_candidates,
            &context_matrix,
            &sequence_state,
            &merged,
            &adaptive.scoring,
            Some(intent_profile.primary),
            Some(&packet.original_text),
        );
        let layer_14_scoring_time_ms = layer_14_start.elapsed().as_millis() as u64;

        let resolver_mode = adaptive.resolver_mode.unwrap_or_else(|| {
            if used_retrieval || intent_profile.certainty_bias < 0.0 || intent_profile.ambiguous {
                ResolverMode::Balanced
            } else {
                ResolverMode::Deterministic
            }
        });
        let layer_16_start = Instant::now();
        
        // Collect anchor units for shaping (high-trust factual anchors)
        let anchor_units: Vec<&Unit> = final_candidates
            .iter()
            .filter(|unit| unit.trust_score >= adaptive.shaping.anchor_trust_threshold)
            .collect();
        
        let resolved = FineResolver::select_with_shaping(
            &scored,
            resolver_mode,
            candidate_route.used_escape,
            &adaptive.resolver,
            &adaptive.shaping,
            &anchor_units,
        )
        .or_else(|| {
            final_candidates
                .first()
                .map(|unit| crate::types::ResolvedCandidate {
                    unit_id: unit.id,
                    content: unit.content.clone(),
                    score: unit.utility_score,
                    mode: ResolverMode::Deterministic,
                    used_escape: false,
                })
        });
        let layer_16_resolution_time_ms = layer_16_start.elapsed().as_millis() as u64;

        // Check if retrieval was triggered but returned no documents
        let retrieval_failed = used_retrieval && merged.evidence.documents.is_empty();

        let mut decoded = resolved
            .as_ref()
            .map(|resolved| {
                self.decoder
                    .decode(&packet.original_text, resolved, &context_matrix, &merged)
            })
            .unwrap_or_else(|| crate::types::DecodedOutput {
                text: "No candidate could be resolved.".to_string(),
                grounded: false,
            });

        // If retrieval failed, provide a helpful message instead of random memory content
        if retrieval_failed && !decoded.grounded {
            let prompt_lower = packet.original_text.to_lowercase();
            let is_realtime_query = prompt_lower.contains("current") 
                || prompt_lower.contains("latest") 
                || prompt_lower.contains("recent")
                || prompt_lower.contains("today")
                || prompt_lower.contains("now")
                || prompt_lower.contains("price")
                || prompt_lower.contains("stock")
                || prompt_lower.contains("won")
                || prompt_lower.contains("winner")
                || prompt_lower.contains("2026")
                || prompt_lower.contains("2025");
            
            if is_realtime_query {
                decoded.text = format!(
                    "I don't have access to current real-time information about {}. My knowledge is based on general knowledge sources and may not reflect recent events or live data.",
                    packet.original_text
                );
            } else {
                decoded.text = format!(
                    "I searched multiple sources but couldn't find specific information about {}. Please try rephrasing your query or providing more context.",
                    packet.original_text
                );
            }
        } else if !merged.evidence.documents.is_empty() {
            if matches!(intent_profile.primary, IntentKind::Extract) {
                if let Some(list_answer) =
                    list_evidence_answer(&packet.original_text, &merged.evidence.documents)
                {
                    decoded.text = list_answer;
                } else if let Some(evidence_answer) =
                    grounded_evidence_answer(&packet.original_text, &merged.evidence.documents)
                {
                    decoded.text = evidence_answer;
                } else if let Some(evidence_answer) = answer_question(
                    &packet.original_text,
                    &merged.evidence.documents,
                    None,
                    Some(intent_profile.primary),
                ) {
                    let normalized_prompt = input::normalize_text(&packet.original_text);
                    let normalized_answer = input::normalize_text(&evidence_answer.text);
                    let confidence_threshold = self.config.resolver.evidence_answer_confidence_threshold;
                    if !normalized_answer.is_empty()
                        && normalized_answer != normalized_prompt
                        && evidence_answer.confidence >= confidence_threshold
                    {
                        decoded.text = evidence_answer.text;
                    }
                }
            } else if let Some(evidence_answer) =
                grounded_evidence_answer(&packet.original_text, &merged.evidence.documents)
            {
                decoded.text = evidence_answer;
            } else if let Some(evidence_answer) = answer_question(
                &packet.original_text,
                &merged.evidence.documents,
                None,
                Some(intent_profile.primary),
            ) {
                let normalized_prompt = input::normalize_text(&packet.original_text);
                let normalized_answer = input::normalize_text(&evidence_answer.text);
                let confidence_threshold = self.config.resolver.evidence_answer_confidence_threshold;
                if !normalized_answer.is_empty()
                    && normalized_answer != normalized_prompt
                    && evidence_answer.confidence >= confidence_threshold
                {
                    decoded.text = evidence_answer.text;
                }
            }
        }
        let (reshaped_text, output_strategy) = reshape_output_for_intent(
            &packet.original_text,
            intent_profile.primary,
            &decoded.text,
            &merged.evidence.documents,
            &[],
            &scored,
        );
        decoded.text = reshaped_text;

        let governance = {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            let recent_ids = top_unit_ids(&scored, 6);
            memory.update_sequence_state(
                recent_ids.clone(),
                sequence_state.anchor_ids.clone(),
                sequence_state.task_entities.clone(),
            );

            if memory.estimate_memory_kb() > self.config.governance.train_memory_ceiling_kb
                || sequence_state.turn_index % 5 == 0
            {
                Some(memory.run_maintenance(&self.config.governance))
            } else {
                None
            }
        };
        self.publish_memory_snapshot();

        let predicted_text = decoded.text.clone();
        let decoded_grounded = decoded.grounded;
        let resolved_candidate = resolved.as_ref().cloned();
        let mut result = ProcessResult {
            predicted_text,
            confidence: resolved
                .as_ref()
                .map(|candidate| candidate.score.clamp(0.0, 1.0))
                .unwrap_or(0.0),
            used_retrieval,
            trace: ExplainTrace::default(),
        };

        let feedback = self.feedback.learn(
            &decision,
            &result,
            &routing,
            &sequence_state,
            governance.as_ref(),
        );

        self.enqueue_feedback(feedback.clone());

        let memory_summary = self.load_memory_snapshot().memory_summary();
        let queue_depths = self.scheduler.depths();
        let trace_sources = preset_sources.clone();

        result.trace = build_trace(
            &packet,
            &routing,
            &context_matrix,
            &decision,
            retrieval_query,
            &merged,
            &scored,
            resolved_candidate,
            result.predicted_text.clone(),
            decoded_grounded,
            confidence_stats,
            intent_profile,
            &adaptive,
            output_strategy,
            reasoning_support.labels,
            Vec::new(),
            feedback,
            memory_summary,
            trace_sources,
            queue_depths,
        );

        self.record_test_observation(
            &packet.original_text,
            memory_before_kb,
            &build_output,
            governance.as_ref(),
            used_retrieval,
            &decision,
            &merged,
            &scored,
            result.confidence,
            ObservationTimings {
                total_latency_ms: total_start.elapsed().as_millis() as u64,
                layer_2_time_ms,
                layer_5_routing_time_ms,
                layer_9_decision_time_ms,
                layer_11_retrieval_time_ms,
                layer_13_merge_time_ms,
                layer_14_scoring_time_ms,
                layer_16_resolution_time_ms,
            },
            preset_sources,
        );

        result
    }

    // ========================================================================
    // Phase 3: Dynamic Reasoning Loop
    // ========================================================================

    /// Phase 3.1: Execute dynamic reasoning loop if triggered by confidence gating.
    /// Returns reasoning result with thoughts and final output.
    fn execute_reasoning_loop(
        &self,
        query: &str,
        initial_confidence: f32,
        intent: &IntentProfile,
        config: &ReasoningLoopConfig,
    ) -> ReasoningResult {
        let mut state = ReasoningState::default();
        state.active = true;
        state.confidence_trajectory.push(initial_confidence);

        // Check if reasoning should be triggered
        if !IntentDetector::should_trigger_reasoning(intent, config) {
            return ReasoningResult {
                output: OutputType::FinalAnswer(String::new()),
                steps_taken: 0,
                final_confidence: initial_confidence,
                reasoning_triggered: false,
                thoughts: Vec::new(),
            };
        }

        // Execute reasoning steps
        for step in 0..config.max_internal_steps {
            state.current_step = step;

            // Generate thought unit for this step
            let thought = self.generate_thought_unit(query, step, &state);
            let thought_confidence = thought.confidence;

            // Ingest silent thought into memory (Intent channel, not Core)
            self.ingest_silent_thought(&thought);

            state.thoughts.push(thought);
            state.confidence_trajectory.push(thought_confidence);

            // Check exit condition
            if thought_confidence >= config.exit_confidence_threshold {
                break;
            }
        }

        state.max_steps_reached = state.current_step >= config.max_internal_steps - 1;

        // Build final output
        let final_confidence = *state.confidence_trajectory.last().unwrap_or(&initial_confidence);
        let output = OutputType::FinalAnswer(String::new());

        ReasoningResult {
            output,
            steps_taken: state.thoughts.len(),
            final_confidence,
            reasoning_triggered: true,
            thoughts: state.thoughts,
        }
    }

    /// Generate a thought unit for the reasoning loop.
    /// This is a placeholder implementation - actual thought generation would use
    /// the candidate pool and context to generate internal reasoning steps.
    fn generate_thought_unit(&self, query: &str, step: usize, state: &ReasoningState) -> ThoughtUnit {
        // Calculate confidence improvement based on step and query complexity
        let previous_confidence = state.confidence_trajectory.last().copied().unwrap_or(0.0);
        let improvement = 0.1 * (1.0 - previous_confidence).min(0.3);
        let new_confidence = (previous_confidence + improvement).clamp(0.0, 1.0);

        // Generate thought content (placeholder - actual implementation would analyze candidates)
        let content = format!(
            "Reasoning step {}: Analyzing '{}' with {} previous thoughts",
            step + 1,
            query.chars().take(50).collect::<String>(),
            state.thoughts.len()
        );

        ThoughtUnit::new(content, step, new_confidence)
    }

    /// Ingest a silent thought into memory using the Reasoning channel.
    /// This prevents pollution of Core memory with reasoning artifacts.
    fn ingest_silent_thought(&self, thought: &ThoughtUnit) {
        let mut memory = self.memory.lock().expect("memory mutex poisoned");

        // Create an activated unit from the thought for hierarchy ingestion
        let activation = ActivatedUnit {
            normalized: thought.content.clone(),
            content: thought.content.clone(),
            level: crate::types::UnitLevel::Phrase,
            utility_score: thought.confidence,
            confidence: thought.confidence,
            frequency: 1,
            salience: 0.5,
            context_hint: format!("reasoning_step_{}", thought.step),
        };

        // Build a minimal hierarchy for the thought
        let mut hierarchy = UnitHierarchy::default();
        hierarchy.levels.insert(
            "Phrase".to_string(),
            vec![activation],
        );

        // Ingest into Reasoning channel, not Core
        memory.ingest_hierarchy_with_channels(
            &hierarchy,
            SourceKind::UserInput,
            &format!("reasoning_step_{}", thought.step),
            MemoryType::Episodic,
            &[MemoryChannel::Reasoning],
        );
    }

    /// Assess confidence for dynamic reasoning trigger.
    fn assess_confidence(&self, query: &str, intent: &IntentProfile) -> f32 {
        // Use intent confidence as primary signal
        let base_confidence = intent.confidence;

        // Adjust based on query characteristics
        let query_lower = query.to_lowercase();
        let has_question_words = query_lower.contains("what")
            || query_lower.contains("how")
            || query_lower.contains("why")
            || query_lower.contains("when")
            || query_lower.contains("where");

        // Questions have slightly lower confidence (need more reasoning)
        if has_question_words {
            (base_confidence * 0.9).clamp(0.0, 1.0)
        } else {
            base_confidence
        }
    }

    #[doc(hidden)]
    pub async fn train_batch(&self, request: TrainBatchRequest) -> TrainingJobStatus {
        let _permit = self.scheduler.acquire(WorkPriority::SilentBatch);
        let job_id = format!("train_{}", Uuid::new_v4().simple());
        self.run_train_batch_with_id(job_id, request).await
    }

    pub fn start_train(self: Arc<Self>, execution_mode: TrainingExecutionMode) -> String {
        self.start_train_with_scope(execution_mode, TrainingScope::Full)
    }

    pub fn start_train_with_scope(
        self: Arc<Self>,
        execution_mode: TrainingExecutionMode,
        scope: TrainingScope,
    ) -> String {
        let job_id = format!("train_{}", Uuid::new_v4().simple());
        let plan = training::build_training_plan_with_config(&self.config, execution_mode, scope);
        let total_sources = plan.phases.iter().map(|phase| phase.sources.len()).sum();
        self.store_job(TrainingJobStatus::queued(job_id.clone(), total_sources));

        let engine = self.clone();
        let spawned_job_id = job_id.clone();
        tokio::spawn(async move {
            let _permit = engine.scheduler.acquire(WorkPriority::SilentBatch);
            let _ = engine.run_training_plan_with_id(spawned_job_id, plan).await;
        });

        job_id
    }

    async fn run_training_plan_with_id(
        &self,
        job_id: String,
        plan: training::TrainingPlan,
    ) -> TrainingJobStatus {
        let total_sources = plan.phases.iter().map(|phase| phase.sources.len()).sum();
        let mut status = TrainingJobStatus::queued(job_id.clone(), total_sources);
        status.status = JobState::Processing;
        status.phase_statuses = plan
            .phases
            .iter()
            .map(|phase| {
                TrainingPhaseStatus::new(phase.phase, phase.batches_target, phase.sources.len())
            })
            .collect();
        self.store_job(status.clone());

        let mut total_elapsed = 0u128;
        let mut completed_sources = 0usize;

        for phase in &plan.phases {
            if !self.phase_ready(phase, &status) {
                if let Some(phase_status) = status
                    .phase_statuses
                    .iter_mut()
                    .find(|entry| entry.phase == phase.phase)
                {
                    phase_status.status = JobState::Completed;
                }
                status.warnings.push(format!(
                    "phase_skipped:{}:bootstrap_metrics_below_threshold",
                    phase_label(phase.phase)
                ));
                self.store_job(status.clone());
                continue;
            }

            status.active_phase = Some(phase.phase);
            if let Some(phase_status) = status
                .phase_statuses
                .iter_mut()
                .find(|entry| entry.phase == phase.phase)
            {
                phase_status.status = JobState::Processing;
            }
            self.store_job(status.clone());

            let started = Instant::now();
            self.run_training_phase(phase, &mut status).await;
            total_elapsed += started.elapsed().as_millis();

            if let Some(phase_status) = status
                .phase_statuses
                .iter()
                .find(|entry| entry.phase == phase.phase)
            {
                completed_sources += phase_status.sources_processed;
            }
            status.progress.sources_processed = completed_sources;
            status.progress.percent_complete =
                (completed_sources as f32 / total_sources.max(1) as f32).clamp(0.0, 1.0);
            self.store_job(status.clone());
        }

        status.active_phase = None;
        status.status = if status
            .warnings
            .iter()
            .any(|warning| warning.starts_with("fatal:"))
        {
            JobState::Failed
        } else {
            JobState::Completed
        };
        status.performance.avg_ms_per_source = if total_sources == 0 {
            0
        } else {
            (total_elapsed / total_sources as u128) as u64
        };
        self.store_job(status.clone());
        status
    }

    async fn run_train_batch_with_id(
        &self,
        job_id: String,
        request: TrainBatchRequest,
    ) -> TrainingJobStatus {
        let mut status = TrainingJobStatus::queued(job_id.clone(), request.sources.len());
        status.status = JobState::Processing;
        self.store_job(status.clone());

        let memory_before = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.estimate_memory_kb()
        };

        let mut total_elapsed = 0u128;
        let mut last_memory_kb = memory_before;
        let mut phase_metrics = PhaseMetricAccumulator::default();
        for (index, source) in request.sources.iter().enumerate() {
            let started = Instant::now();
            let should_halt = match self
                .process_training_source(
                    source,
                    &request.options,
                    &mut status,
                    memory_before,
                    &mut last_memory_kb,
                    &mut phase_metrics,
                )
                .await
            {
                Ok(should_halt) => should_halt,
                Err(err) if is_skippable_training_error(&err) => {
                    status.warnings.push(format!(
                        "skipped_source:{}:{}",
                        training_source_label(source),
                        err.trim_start_matches("fatal:")
                    ));
                    false
                }
                Err(err) => {
                    status.warnings.push(err);
                    false
                }
            };

            total_elapsed += started.elapsed().as_millis();
            status.progress.sources_processed = index + 1;
            status.progress.percent_complete = (status.progress.sources_processed as f32
                / status.progress.sources_total.max(1) as f32)
                .clamp(0.0, 1.0);
            self.store_job(status.clone());
            if should_halt {
                break;
            }
        }

        if request.options.consolidate_immediately {
            let report = {
                let mut memory = self.memory.lock().expect("memory mutex poisoned");
                memory.run_maintenance(&self.config.governance)
            };
            self.publish_memory_snapshot();
            status.learning_metrics.units_pruned += report.pruned_units;
            status.learning_metrics.anchors_protected += report.anchors_protected;
            status.learning_metrics.map_adjustments += report.layout_adjustments;
            record_pruning_event(
                &mut status.learning_metrics,
                &report,
                "batch_consolidation_maintenance",
            );
            if report.layout_rolled_back {
                status.warnings.push("layout_rollback:batch".to_string());
            }
        }

        status.status = if status.warnings.iter().any(|warning| {
            warning.starts_with("fatal:") || warning == "daily_memory_limit_exceeded"
        }) {
            JobState::Failed
        } else {
            JobState::Completed
        };
        status.performance.avg_ms_per_source = if request.sources.is_empty() {
            0
        } else {
            (total_elapsed / request.sources.len() as u128) as u64
        };
        self.store_job(status.clone());
        status
    }

    async fn run_training_phase(&self, phase: &TrainingPhasePlan, status: &mut TrainingJobStatus) {
        let phase_governance =
            self.effective_phase_governance(phase.phase, phase.options.execution_mode);
        self.apply_phase_governance(phase.phase, phase.options.execution_mode);
        let memory_before = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.estimate_memory_kb()
        };
        let mut last_memory_kb = memory_before;
        let mut metrics_accumulator = PhaseMetricAccumulator::default();
        let mut processed_sources = 0usize;

        for source in &phase.sources {
            let should_halt = match self
                .process_training_source(
                    source,
                    &phase.options,
                    status,
                    memory_before,
                    &mut last_memory_kb,
                    &mut metrics_accumulator,
                )
                .await
            {
                Ok(should_halt) => should_halt,
                Err(err) if is_skippable_training_error(&err) => {
                    status.warnings.push(format!(
                        "skipped_source:{}:{}",
                        training_source_label(source),
                        err.trim_start_matches("fatal:")
                    ));
                    false
                }
                Err(err) => {
                    status.warnings.push(err);
                    false
                }
            };
            processed_sources += 1;
            if let Some(phase_status) = status
                .phase_statuses
                .iter_mut()
                .find(|entry| entry.phase == phase.phase)
            {
                phase_status.sources_processed = processed_sources;
            }
            let completed_before_phase = status
                .phase_statuses
                .iter()
                .filter(|entry| entry.phase != phase.phase)
                .map(|entry| entry.sources_processed)
                .sum::<usize>();
            status.progress.sources_processed = completed_before_phase + processed_sources;
            status.progress.percent_complete = (status.progress.sources_processed as f32
                / status.progress.sources_total.max(1) as f32)
                .clamp(0.0, 1.0);
            status.progress.active_source = None;
            status.progress.chunks_processed = 0;
            status.progress.bytes_processed = 0;
            status.progress.worker_count = 0;
            status.progress.active_workers = 0;
            status.progress.queued_chunks = 0;
            status.progress.prepared_chunks = 0;
            status.progress.committed_batches = 0;
            self.store_job(status.clone());
            if should_halt {
                break;
            }
        }

        if phase.options.consolidate_immediately {
            let report = {
                let mut memory = self.memory.lock().expect("memory mutex poisoned");
                memory.run_maintenance(&phase_governance)
            };
            self.publish_memory_snapshot();
            status.learning_metrics.units_pruned += report.pruned_units;
            status.learning_metrics.anchors_protected += report.anchors_protected;
            status.learning_metrics.map_adjustments += report.layout_adjustments;
            record_pruning_event(
                &mut status.learning_metrics,
                &report,
                &format!("phase_consolidation:{}", phase_label(phase.phase)),
            );
            if report.layout_rolled_back {
                status
                    .warnings
                    .push(format!("layout_rollback:{}", phase_label(phase.phase)));
            }
        }

        let memory_after = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.estimate_memory_kb()
        };
        let mut phase_metrics = metrics_accumulator.finalize(memory_after - memory_before);
        if matches!(phase.phase, TrainingPhaseKind::Validation) {
            phase_metrics.search_trigger_precision = Some(
                phase_metrics
                    .unit_discovery_efficiency
                    .min(phase_metrics.semantic_routing_accuracy),
            );
        }

        if let Some(phase_status) = status
            .phase_statuses
            .iter_mut()
            .find(|entry| entry.phase == phase.phase)
        {
            phase_status.batches_completed = 1;
            phase_status.metrics = phase_metrics;
            phase_status.status = JobState::Completed;
        }
        status.progress.active_source = None;
        status.progress.chunks_processed = 0;
        status.progress.bytes_processed = 0;
        status.progress.worker_count = 0;
        status.progress.active_workers = 0;
        status.progress.queued_chunks = 0;
        status.progress.prepared_chunks = 0;
        status.progress.committed_batches = 0;
        self.restore_runtime_governance();
    }

    fn phase_ready(&self, phase: &TrainingPhasePlan, status: &TrainingJobStatus) -> bool {
        if !matches!(phase.phase, TrainingPhaseKind::Expansion) {
            return true;
        }
        let Some(bootstrap) = status
            .phase_statuses
            .iter()
            .find(|entry| entry.phase == TrainingPhaseKind::Bootstrap)
        else {
            return false;
        };
        let min_unit_discovery_efficiency = phase.min_unit_discovery_efficiency.unwrap_or(0.0);
        let min_semantic_routing_accuracy = phase.min_semantic_routing_accuracy.unwrap_or(0.0);
        bootstrap.metrics.unit_discovery_efficiency >= min_unit_discovery_efficiency
            && bootstrap.metrics.semantic_routing_accuracy >= min_semantic_routing_accuracy
    }

    pub fn training_status(&self, job_id: &str) -> Option<TrainingJobStatus> {
        if let Some(status) = self
            .jobs
            .lock()
            .expect("jobs mutex poisoned")
            .get(job_id)
            .cloned()
        {
            return Some(status);
        }

        let db = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.db()
        };
        let statuses = db.load_training_jobs().ok()?;
        let status = statuses
            .into_iter()
            .find(|status| status.job_id == job_id)?;
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.insert(job_id.to_string(), status.clone());
        }
        Some(status)
    }

    /// Phase 3.4: Expose config for auto-mode enforcement
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    fn apply_phase_governance(
        &self,
        phase: TrainingPhaseKind,
        execution_mode: TrainingExecutionMode,
    ) {
        let governance = self.effective_phase_governance(phase, execution_mode);
        if let Ok(mut memory) = self.memory.lock() {
            memory.apply_governance(&governance);
        }
    }

    fn restore_runtime_governance(&self) {
        if let Ok(mut memory) = self.memory.lock() {
            memory.apply_governance(&self.config.governance);
        }
    }

    fn effective_phase_governance(
        &self,
        phase: TrainingPhaseKind,
        execution_mode: TrainingExecutionMode,
    ) -> crate::config::GovernanceConfig {
        let mut config = self.config.clone();
        if matches!(execution_mode, TrainingExecutionMode::Development)
            && matches!(
                phase,
                TrainingPhaseKind::DryRun | TrainingPhaseKind::Bootstrap
            )
        {
            config = config.with_bootstrap_overrides();
        }
        config.governance
    }

    fn resolve_adaptive_runtime(
        &self,
        intent_profile: &IntentProfile,
        queue_depths: QueueDepths,
    ) -> AdaptiveRuntimeSettings {
        let mut settings = AdaptiveRuntimeSettings::from_config(&self.config);
        let profile_blend = dynamic_intent_profile_blend(intent_profile);
        settings.intent_profile_name =
            dominant_behavior_profile_name(&profile_blend).map(str::to_string);
        settings.scoring = blend_scoring_weights(&self.config, &profile_blend);
        settings.escape = blend_escape_profile(&self.config, &profile_blend);
        let (resolver, resolver_mode) = blend_resolver_profile(&self.config, &profile_blend);
        settings.resolver = resolver;
        settings.resolver_mode = resolver_mode;
        settings.shaping = blend_shaping_config(&self.config, &profile_blend);

        let trust_signal = dynamic_trust_signal(intent_profile);
        if trust_signal.harden {
            settings.trust_signal_name = Some(trust_signal.reason.clone());
            if let Some(trust_profile) = self.config.adaptive_behavior.trust_profile("high_stakes")
            {
                settings.trust_profile_name = Some("high_stakes".to_string());
                apply_adaptive_trust_profile(&mut settings.trust, trust_profile);
            }
        }

        let additive_penalty = self.compute_load_cost_penalty(queue_depths);
        settings.additional_cost_penalty = additive_penalty;
        settings.retrieval.cost_penalty =
            (settings.retrieval.cost_penalty + additive_penalty).clamp(0.0, 2.0);

        settings
    }

    fn compute_load_cost_penalty(&self, queue_depths: QueueDepths) -> f32 {
        let config = &self.config.adaptive_behavior.load_cost;
        let saturation = config.queue_saturation_depth.max(1.0);
        let normalized = |depth: usize| ((depth as f32) / saturation).clamp(0.0, 1.0);
        let penalty = (normalized(queue_depths.inference) * config.inference_queue_weight)
            + (normalized(queue_depths.interactive_training) * config.interactive_training_weight)
            + (normalized(queue_depths.silent_batch) * config.silent_batch_weight)
            + (normalized(queue_depths.maintenance) * config.maintenance_weight);
        penalty.clamp(0.0, config.max_additive_penalty)
    }

    fn prepare_context_and_candidates(
        &self,
        hierarchy: &UnitHierarchy,
        routing: &RoutingResult,
    ) -> (ContextMatrix, SequenceState, Vec<Unit>) {
        let snapshot = self.load_memory_snapshot();
        let prior_state = snapshot.sequence_state();
        let active_units = snapshot.get_units(
            &routing
                .position_updates
                .iter()
                .map(|(id, _)| *id)
                .collect::<Vec<_>>(),
        );
        let neighbor_units = snapshot.get_units(&routing.neighbor_ids);
        let all_units = snapshot.all_units();

        let (context_matrix, sequence_state) = ContextManager::update(
            routing,
            hierarchy,
            &prior_state,
            &active_units,
            &neighbor_units,
        );
        (context_matrix, sequence_state, all_units)
    }

    async fn process_training_source(
        &self,
        source: &TrainingSource,
        options: &TrainingOptions,
        status: &mut TrainingJobStatus,
        memory_before: i64,
        last_memory_kb: &mut i64,
        phase_metrics: &mut PhaseMetricAccumulator,
    ) -> Result<bool, String> {
        let resolved_source =
            open_sources::resolve_training_source(source).map_err(|err| format!("fatal:{err}"))?;
        let target_memory = datasets::effective_memory_type(&resolved_source, &self.config)
            .or(resolved_source.target_memory)
            .unwrap_or_else(|| default_memory_for_training_source(resolved_source.source_type));
        let memory_channels = resolved_source
            .memory_channels
            .clone()
            .unwrap_or_else(|| vec![MemoryChannel::Main]);
        status.progress.active_source = Some(training_source_label(source));
        status.progress.chunks_processed = 0;
        status.progress.bytes_processed = 0;
        status.progress.worker_count = 0;
        status.progress.active_workers = 0;
        status.progress.queued_chunks = 0;
        status.progress.prepared_chunks = 0;
        status.progress.committed_batches = 0;
        self.store_job(status.clone());

        if matches!(
            resolved_source.source_type,
            TrainingSourceType::HuggingFaceDataset
                | TrainingSourceType::StructuredJson
                | TrainingSourceType::OpenApiSpec
                | TrainingSourceType::CodeRepository
                | TrainingSourceType::WikipediaDump
                | TrainingSourceType::WikidataTruthy
                | TrainingSourceType::OpenWebText
                | TrainingSourceType::DbpediaDump
                | TrainingSourceType::ProjectGutenberg
                | TrainingSourceType::CommonCrawlWet
                | TrainingSourceType::QaJson
        ) {
            let (should_halt, warnings) =
                if self.parallel_training_enabled_for_source(&resolved_source) {
                    self.process_parallel_streaming_training_source(
                        &resolved_source,
                        options,
                        status,
                        memory_before,
                        last_memory_kb,
                        phase_metrics,
                        target_memory,
                        &memory_channels,
                    )
                    .await?
                } else {
                    self.process_serial_streaming_training_source(
                        &resolved_source,
                        options,
                        status,
                        memory_before,
                        last_memory_kb,
                        phase_metrics,
                        target_memory,
                        &memory_channels,
                    )
                    .await?
                };
            status.warnings.extend(warnings);
            return Ok(should_halt);
        }

        let content = match resolved_source.source_type {
            TrainingSourceType::Url => {
                let url = resolved_source
                    .value
                    .as_deref()
                    .ok_or_else(|| "fatal:missing_url_value".to_string())?;
                let content = self
                    .retriever
                    .fetch_training_source(TrainingSourceType::Url, url)
                    .await?;
                let assessment = self.safety.assess(url, &content, &self.config.trust);
                if !assessment.accepted {
                    return Err(format!(
                        "discarded_untrusted_source:{url}:{}",
                        assessment.warnings.join("|")
                    ));
                }
                content
            }
            TrainingSourceType::Document => {
                if let Some(value) = resolved_source.value.as_deref() {
                    let path = Path::new(value);
                    if is_supported_document_path(path) {
                        if let Some(mime) = resolved_source.mime.as_deref() {
                            validate_document_mime(path, mime)
                                .map_err(|err| format!("fatal:{err}"))?;
                        }
                        load_path_content_with_config(path, &self.config.document)
                            .map_err(|err| format!("fatal:{err}"))?
                    } else {
                        let decoded = decode_document_bytes(value, resolved_source.mime.as_deref());
                        load_document_bytes(
                            &decoded,
                            resolved_source.mime.as_deref(),
                            &self.config.document,
                        )
                        .map_err(|err| format!("fatal:{err}"))?
                    }
                } else {
                    let content = resolved_source
                        .content
                        .as_deref()
                        .ok_or_else(|| "fatal:missing_document_content".to_string())?;
                    let decoded = decode_document_bytes(content, resolved_source.mime.as_deref());
                    load_document_bytes(
                        &decoded,
                        resolved_source.mime.as_deref(),
                        &self.config.document,
                    )
                    .map_err(|err| format!("fatal:{err}"))?
                }
            }
            TrainingSourceType::Dataset
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
            | TrainingSourceType::QaJson => unreachable!("dataset sources handled above"),
        };

        let source_kind = match resolved_source.source_type {
            TrainingSourceType::Url => SourceKind::TrainingUrl,
            TrainingSourceType::Document => SourceKind::TrainingDocument,
            TrainingSourceType::Dataset
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
            | TrainingSourceType::QaJson => unreachable!("dataset sources handled above"),
        };
        let context_label = resolved_source
            .value
            .as_deref()
            .map(|value| format!("training_source:{value}"))
            .or_else(|| {
                resolved_source
                    .name
                    .as_deref()
                    .map(|name| format!("training_source:{name}"))
            })
            .unwrap_or_else(|| "training_source:inline".to_string());
        let report = self.ingest_training_text_chunk(
            &content,
            source_kind,
            target_memory,
            &memory_channels,
            &context_label,
            options.tag_intent,
            options.merge_to_core,
        )?;
        phase_metrics.observe(&report);
        status.progress.chunks_processed = 1;
        status.progress.bytes_processed = content.len() as u64;
        let should_halt = self.apply_training_chunk_report(
            report,
            options,
            status,
            memory_before,
            last_memory_kb,
        );
        self.store_job(status.clone());
        Ok(should_halt)
    }

    fn parallel_training_enabled_for_source(&self, source: &TrainingSource) -> bool {
        self.config.silent_training.parallel.enabled
            && self.parallel_training_worker_count() > 1
            && matches!(
                source.source_type,
                TrainingSourceType::HuggingFaceDataset
                    | TrainingSourceType::StructuredJson
                    | TrainingSourceType::OpenApiSpec
                    | TrainingSourceType::CodeRepository
                    | TrainingSourceType::WikipediaDump
                    | TrainingSourceType::WikidataTruthy
                    | TrainingSourceType::OpenWebText
                    | TrainingSourceType::DbpediaDump
                    | TrainingSourceType::ProjectGutenberg
                    | TrainingSourceType::CommonCrawlWet
                    | TrainingSourceType::QaJson
            )
    }

    fn parallel_training_worker_count(&self) -> usize {
        let configured = self.config.silent_training.parallel.worker_count;
        let cpu_based = std::thread::available_parallelism()
            .map(|parallelism| parallelism.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let ram_based = self.parallel_training_worker_count_for_budget();
        let auto = cpu_based.min(ram_based).max(1);
        if configured > 0 {
            configured.min(auto).max(1)
        } else {
            auto
        }
    }

    fn parallel_training_worker_count_for_budget(&self) -> usize {
        let parallel = &self.config.silent_training.parallel;
        let total_kb = (parallel.total_memory_limit_mb.max(0.0) * 1024.0) as i64;
        let reserve_kb = (parallel.non_worker_memory_reserve_mb.max(0.0) * 1024.0) as i64;
        let per_worker_kb = (parallel.local_shard_hard_limit_mb.max(1.0) * 1024.0) as i64;
        if total_kb <= 0 || per_worker_kb <= 0 {
            return 1;
        }
        let available_kb = (total_kb - reserve_kb).max(per_worker_kb);
        ((available_kb / per_worker_kb).max(1)) as usize
    }

    fn parallel_training_queue_capacity(&self, worker_count: usize) -> usize {
        let parallel = &self.config.silent_training.parallel;
        let requested = parallel.queue_capacity.max(worker_count);
        let per_worker_cap = parallel
            .queue_capacity_per_worker
            .max(1)
            .saturating_mul(worker_count.max(1));
        requested.min(per_worker_cap).max(worker_count)
    }

    fn local_shard_soft_limit_kb(&self) -> i64 {
        (self
            .config
            .silent_training
            .parallel
            .local_shard_soft_limit_mb
            .max(1.0)
            * 1024.0) as i64
    }

    fn local_shard_hard_limit_kb(&self) -> i64 {
        (self
            .config
            .silent_training
            .parallel
            .local_shard_hard_limit_mb
            .max(self.config.silent_training.parallel.local_shard_soft_limit_mb.max(1.0))
            * 1024.0) as i64
    }

    async fn process_serial_streaming_training_source(
        &self,
        source: &TrainingSource,
        options: &TrainingOptions,
        status: &mut TrainingJobStatus,
        memory_before: i64,
        last_memory_kb: &mut i64,
        phase_metrics: &mut PhaseMetricAccumulator,
        target_memory: MemoryType,
        memory_channels: &[MemoryChannel],
    ) -> Result<(bool, Vec<String>), String> {
        let mut should_halt = false;
        let max_input_bytes = source.stream.max_input_bytes;
        let mut streamed_bytes = 0usize;
        let mut streamed_chunks = 0u64;
        let mut last_progress_flush = Instant::now();
        status.progress.worker_count = 0;
        status.progress.active_workers = 0;
        status.progress.queued_chunks = 0;
        status.progress.prepared_chunks = 0;
        status.progress.committed_batches = 0;

        let stream_result = datasets::stream_training_source(source, &self.config, |chunk| {
            streamed_bytes += chunk.content.len();
            streamed_chunks += 1;
            let report = self.ingest_training_text_chunk(
                &chunk.content,
                SourceKind::TrainingDocument,
                target_memory,
                memory_channels,
                &chunk.context_label,
                options.tag_intent,
                options.merge_to_core,
            )?;
            phase_metrics.observe(&report);
            should_halt = self.apply_training_chunk_report(
                report,
                options,
                status,
                memory_before,
                last_memory_kb,
            ) || max_input_bytes
                .map(|limit| streamed_bytes >= limit)
                .unwrap_or(false);
            status.progress.chunks_processed = streamed_chunks;
            status.progress.bytes_processed = streamed_bytes as u64;
            if should_halt
                || last_progress_flush.elapsed().as_secs() >= options.progress_interval_sec
            {
                self.store_job(status.clone());
                last_progress_flush = Instant::now();
            }
            Ok(if should_halt {
                DatasetSinkControl::Halt
            } else {
                DatasetSinkControl::Continue
            })
        })
        .await
        .map_err(|err| format!("fatal:{err}"))?;

        status.progress.worker_count = 0;
        status.progress.active_workers = 0;
        status.progress.queued_chunks = 0;
        Ok((should_halt, stream_result.warnings))
    }

    async fn process_parallel_streaming_training_source(
        &self,
        source: &TrainingSource,
        options: &TrainingOptions,
        status: &mut TrainingJobStatus,
        memory_before: i64,
        last_memory_kb: &mut i64,
        phase_metrics: &mut PhaseMetricAccumulator,
        target_memory: MemoryType,
        memory_channels: &[MemoryChannel],
    ) -> Result<(bool, Vec<String>), String> {
        let worker_count = self.parallel_training_worker_count();
        let queue_capacity = self.parallel_training_queue_capacity(worker_count);
        let max_input_bytes = source.stream.max_input_bytes;
        let local_shard_soft_limit_kb = self.local_shard_soft_limit_kb();
        let local_shard_hard_limit_kb = self.local_shard_hard_limit_kb();
        let tag_intent = options.tag_intent;
        let config = Arc::new(self.config.clone());
        let halt = Arc::new(AtomicBool::new(false));
        let active_workers = Arc::new(AtomicUsize::new(0));
        let processed_chunks = Arc::new(AtomicU64::new(0));
        let committed_batches = Arc::new(AtomicU64::new(0));
        let streamed_bytes = Arc::new(AtomicU64::new(0));

        let source_name = source
            .name
            .clone()
            .or_else(|| source.value.clone())
            .unwrap_or_else(|| "training_source".to_string());
        let (raw_tx, raw_rx) = crossbeam_channel::bounded::<DatasetTextChunk>(queue_capacity);

        let mut worker_handles = Vec::with_capacity(worker_count);
        for shard_index in 0..worker_count {
            let raw_rx = raw_rx.clone();
            let config = config.clone();
            let halt = halt.clone();
            let active_workers = active_workers.clone();
            let processed_chunks = processed_chunks.clone();
            let source_name = source_name.clone();
            let target_memory = target_memory;
            let memory_channels = memory_channels.to_vec();
            let merge_to_core = options.merge_to_core;
            worker_handles.push(thread::spawn(move || -> Result<ShardTrainingResult, String> {
                let mut segment_index = 0usize;
                let mut shard_db_path =
                    training_shard_db_path(&source_name, shard_index, segment_index);
                let mut shard_db_path_str = shard_db_path.to_string_lossy().to_string();
                let mut local_store = MemoryStore::new_training_shard(
                    &shard_db_path_str,
                    &config.governance,
                    &config.semantic_map,
                );
                let mut local_snapshot = local_store.snapshot();
                let mut local_report = ChunkIngestionReport::default();
                let mut segments = Vec::new();
                while let Ok(chunk) = raw_rx.recv() {
                    if halt.load(Ordering::Relaxed) {
                        break;
                    }
                    active_workers.fetch_add(1, Ordering::SeqCst);
                    let database_health = local_store.database_health();
                    let prepared = prepare_training_chunk_for_ingest(
                        &chunk,
                        &config,
                        &database_health,
                        Some(&local_snapshot),
                        tag_intent,
                    );
                    let outcome = prepared.and_then(|prepared| {
                        ingest_prepared_training_batch_into_store(
                            &mut local_store,
                            &[prepared],
                            SourceKind::TrainingDocument,
                            target_memory,
                            &memory_channels,
                            merge_to_core,
                        )
                    });
                    active_workers.fetch_sub(1, Ordering::SeqCst);
                    match outcome {
                        Ok((report, _active_ids)) => {
                            accumulate_chunk_ingestion_report(&mut local_report, &report);
                            processed_chunks.fetch_add(report.chunks_committed.max(1), Ordering::SeqCst);
                            local_snapshot = local_store.snapshot();
                            let estimated_kb = local_store.estimate_memory_kb();
                            if estimated_kb >= local_shard_soft_limit_kb
                                || estimated_kb >= local_shard_hard_limit_kb
                            {
                                let pruning_report =
                                    local_store.run_shard_pruning(&config.governance);
                                let (pattern_cache_hits, pattern_cache_lookups) =
                                    local_store.pattern_cache_stats();
                                let has_payload = {
                                    let health = local_store.database_health();
                                    health.total_units > 0 || health.candidate_units > 0
                                };
                                drop(local_store);
                                if has_payload {
                                    segments.push(ShardSegmentResult {
                                        shard_index,
                                        segment_index,
                                        shard_db_path: shard_db_path.clone(),
                                        pattern_cache_hits,
                                        pattern_cache_lookups,
                                        pruning_report,
                                    });
                                } else {
                                    cleanup_training_shard_db(&shard_db_path);
                                }

                                segment_index += 1;
                                shard_db_path =
                                    training_shard_db_path(&source_name, shard_index, segment_index);
                                shard_db_path_str = shard_db_path.to_string_lossy().to_string();
                                local_store = MemoryStore::new_training_shard(
                                    &shard_db_path_str,
                                    &config.governance,
                                    &config.semantic_map,
                                );
                                local_snapshot = local_store.snapshot();
                            }
                        }
                        Err(err) => {
                            cleanup_training_shard_db(&shard_db_path);
                            return Err(err);
                        }
                    }
                }
                let pruning_report = local_store.run_shard_pruning(&config.governance);
                let (pattern_cache_hits, pattern_cache_lookups) = local_store.pattern_cache_stats();
                let has_payload = {
                    let health = local_store.database_health();
                    health.total_units > 0 || health.candidate_units > 0
                };
                drop(local_store);
                if has_payload {
                    segments.push(ShardSegmentResult {
                        shard_index,
                        segment_index,
                        shard_db_path,
                        pattern_cache_hits,
                        pattern_cache_lookups,
                        pruning_report,
                    });
                } else {
                    cleanup_training_shard_db(&shard_db_path);
                }
                Ok(ShardTrainingResult {
                    shard_index,
                    local_report,
                    segments,
                })
            }));
        }
        drop(raw_rx);

        let mut sent_chunks = 0u64;
        let mut should_halt = false;
        let mut warnings = Vec::new();
        let mut last_progress_flush = Instant::now();
        apply_parallel_training_progress(
            status,
            worker_count,
            active_workers.as_ref(),
            raw_tx.len(),
            processed_chunks.as_ref(),
            committed_batches.as_ref(),
            0,
            streamed_bytes.load(Ordering::Relaxed),
        );
        self.store_job(status.clone());

        let stream_result = datasets::stream_training_source(source, &self.config, |chunk| {
            if should_halt || halt.load(Ordering::Relaxed) {
                return Ok(DatasetSinkControl::Halt);
            }

            let chunk_len = chunk.content.len() as u64;
            let mut pending = Some(chunk);
            loop {
                match raw_tx.try_send(pending.take().expect("pending chunk")) {
                    Ok(()) => {
                        sent_chunks += 1;
                        let total_bytes =
                            streamed_bytes.fetch_add(chunk_len, Ordering::SeqCst) + chunk_len;
                        if max_input_bytes
                            .map(|limit| total_bytes >= limit as u64)
                            .unwrap_or(false)
                        {
                            should_halt = true;
                        }
                        apply_parallel_training_progress(
                            status,
                            worker_count,
                            active_workers.as_ref(),
                            raw_tx.len(),
                            processed_chunks.as_ref(),
                            committed_batches.as_ref(),
                            0,
                            streamed_bytes.load(Ordering::Relaxed),
                        );
                        if last_progress_flush.elapsed().as_secs() >= options.progress_interval_sec
                        {
                            self.store_job(status.clone());
                            last_progress_flush = Instant::now();
                        }
                        break;
                    }
                    Err(TrySendError::Full(chunk)) => {
                        pending = Some(chunk);
                        thread::sleep(Duration::from_millis(5));
                        apply_parallel_training_progress(
                            status,
                            worker_count,
                            active_workers.as_ref(),
                            raw_tx.len(),
                            processed_chunks.as_ref(),
                            committed_batches.as_ref(),
                            0,
                            streamed_bytes.load(Ordering::Relaxed),
                        );
                        if should_halt {
                            halt.store(true, Ordering::SeqCst);
                            return Ok(DatasetSinkControl::Halt);
                        }
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        return Err("parallel_training_channel_closed".to_string());
                    }
                }
            }

            if should_halt {
                halt.store(true, Ordering::SeqCst);
                Ok(DatasetSinkControl::Halt)
            } else {
                Ok(DatasetSinkControl::Continue)
            }
        })
        .await
        .map_err(|err| format!("fatal:{err}"));

        if should_halt || stream_result.is_err() {
            halt.store(true, Ordering::SeqCst);
        }
        drop(raw_tx);

        let mut shard_reports = Vec::with_capacity(worker_count);
        for handle in worker_handles {
            match handle.join() {
                Ok(Ok(result)) => shard_reports.push(result),
                Ok(Err(err)) => return Err(format!("fatal:{err}")),
                Err(_) => return Err("fatal:parallel_training_worker_panicked".to_string()),
            }
        }
        shard_reports.sort_by_key(|shard| shard.shard_index);

        let mut aggregate_report = ChunkIngestionReport {
            chunks_committed: processed_chunks.load(Ordering::Relaxed),
            ..ChunkIngestionReport::default()
        };
        let mut merged_active_ids = Vec::new();

        for shard in shard_reports {
            aggregate_report.activated_units_observed += shard.local_report.activated_units_observed;
            aggregate_report.anchors_observed += shard.local_report.anchors_observed;
            for (intent_key, count) in &shard.local_report.intent_counts {
                *aggregate_report
                    .intent_counts
                    .entry(intent_key.clone())
                    .or_insert(0) += count;
            }
            let batch_size = self
                .config
                .silent_training
                .parallel
                .commit_batch_size
                .max(1);
            for segment in shard.segments {
                let local_removed = removed_count_from_governance(&segment.pruning_report);
                if local_removed > 0 {
                    status.learning_metrics.units_pruned += local_removed;
                    status.learning_metrics.anchors_protected +=
                        segment.pruning_report.anchors_protected;
                }
                record_pruning_event(
                    &mut status.learning_metrics,
                    &segment.pruning_report,
                    &format!("local_shard:{}:{}", segment.shard_index, segment.segment_index),
                );

                let merge_result = {
                    let mut memory = self.memory.lock().expect("memory mutex poisoned");
                    memory.merge_training_shard_db(&segment.shard_db_path, batch_size)
                };
                cleanup_training_shard_db(&segment.shard_db_path);
                let merge_report = merge_result?;
                aggregate_report.new_units_discovered +=
                    merge_report.new_units + merge_report.candidate_observations;
                aggregate_report.reused_units_observed += merge_report.reused_units;
                aggregate_report.candidate_observations += merge_report.candidate_observations;
                aggregate_report.candidate_promotions += merge_report.candidate_promotions;
                aggregate_report.cache_hits +=
                    merge_report.cache_hits + segment.pattern_cache_hits;
                aggregate_report.cache_lookups +=
                    merge_report.cache_lookups + segment.pattern_cache_lookups;
                merged_active_ids.extend(merge_report.active_ids);

                committed_batches.fetch_add(1, Ordering::SeqCst);
                apply_parallel_training_progress(
                    status,
                    worker_count,
                    active_workers.as_ref(),
                    0,
                    processed_chunks.as_ref(),
                    committed_batches.as_ref(),
                    processed_chunks.load(Ordering::Relaxed),
                    streamed_bytes.load(Ordering::Relaxed),
                );
                self.store_job(status.clone());
            }
        }

        if !merged_active_ids.is_empty() {
            merged_active_ids.sort();
            merged_active_ids.dedup();
            let (active_units, all_units) = {
                let memory = self.memory.lock().expect("memory mutex poisoned");
                (memory.get_units(&merged_active_ids), memory.all_units())
            };
            let routing = route_units(&active_units, &all_units, &self.config.semantic_map);
            aggregate_report.map_adjustments = routing.map_adjustments as u64;
            aggregate_report.mean_displacement =
                mean_displacement_for_training_routing(&active_units, &routing.position_updates);
            {
                let mut memory = self.memory.lock().expect("memory mutex poisoned");
                memory.update_positions(&routing.position_updates);
                let governance_report = memory.run_maintenance(&self.config.governance);
                let global_removed = removed_count_from_governance(&governance_report);
                if global_removed > 0 {
                    status.learning_metrics.units_pruned += global_removed;
                    status.learning_metrics.anchors_protected += governance_report.anchors_protected;
                }
                record_pruning_event(
                    &mut status.learning_metrics,
                    &governance_report,
                    "global_shard_merge",
                );
                if governance_report.layout_rolled_back {
                    status
                        .warnings
                        .push("layout_rollback:global_shard_merge".to_string());
                }
            }
        }

        phase_metrics.observe(&aggregate_report);
        should_halt = self.apply_training_chunk_report(
            aggregate_report,
            options,
            status,
            memory_before,
            last_memory_kb,
        ) || should_halt;

        apply_parallel_training_progress(
            status,
            worker_count,
            active_workers.as_ref(),
            0,
            processed_chunks.as_ref(),
            committed_batches.as_ref(),
            processed_chunks.load(Ordering::Relaxed),
            streamed_bytes.load(Ordering::Relaxed),
        );
        status.progress.active_workers = 0;
        status.progress.queued_chunks = 0;
        self.store_job(status.clone());

        let stream_result = stream_result?;
        warnings.extend(stream_result.warnings);
        Ok((should_halt, warnings))
    }

    fn ingest_training_text_chunk(
        &self,
        content: &str,
        source_kind: SourceKind,
        target_memory: MemoryType,
        memory_channels: &[MemoryChannel],
        context_label: &str,
        collect_intent_telemetry: bool,
        merge_to_core: bool,
    ) -> Result<ChunkIngestionReport, String> {
        let normalized = input::normalize_text(content);
        if normalized.is_empty() {
            return Ok(ChunkIngestionReport::default());
        }

        let intent_key = if collect_intent_telemetry {
            let telemetry_window = normalized
                .split_whitespace()
                .take(96)
                .collect::<Vec<_>>()
                .join(" ");
            let intent_profile = IntentDetector::classify(
                &telemetry_window,
                &ContextMatrix::default(),
                &SequenceState::default(),
                false,
                &self.config.intent,
            );
            Some(format!("{:?}", intent_profile.primary).to_lowercase())
        } else {
            None
        };

        let packet = input::ingest_raw(&normalized, true);
        let build_output = self.build_units(&packet);
        let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &self.config.builder);
        let effective_memory = if merge_to_core {
            target_memory
        } else {
            MemoryType::Episodic
        };
        let packet_summary = summarize_packet(&packet);
        let mut context_segments = Vec::new();
        if !context_label.is_empty() {
            context_segments.push(context_label.to_string());
        }
        context_segments.push(packet_summary);
        if memory_channels.contains(&MemoryChannel::Intent) {
            if let Some(intent_key) = intent_key.as_deref() {
                context_segments.push(format!("intent_label:{intent_key}"));
            }
        }
        let context_summary = context_segments.join(" | ");

        let active_ids = {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            let ingest_report = memory.ingest_hierarchy_with_channels_report(
                &hierarchy,
                source_kind,
                &context_summary,
                effective_memory,
                memory_channels,
            );
            (
                ingest_report.active_ids,
                ingest_report.new_units,
                ingest_report.reused_units,
                ingest_report.candidate_observations,
                ingest_report.candidate_promotions,
                ingest_report.cache_hits,
                ingest_report.cache_lookups,
            )
        };

        let (
            active_ids,
            new_units_discovered,
            reused_units_observed,
            candidate_observations,
            candidate_promotions,
            cache_hits,
            cache_lookups,
        ) = active_ids;

        let (active_units, all_units) = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            (memory.get_units(&active_ids), memory.all_units())
        };

        let routing = route_units(&active_units, &all_units, &self.config.semantic_map);

        let displacement_mean = if routing.position_updates.is_empty() {
            let positions = active_units
                .iter()
                .map(|unit| unit.semantic_position)
                .collect::<Vec<_>>();
            if positions.is_empty() {
                0.0
            } else {
                let mut total = [0.0, 0.0, 0.0];
                for position in &positions {
                    total[0] += position[0];
                    total[1] += position[1];
                    total[2] += position[2];
                }
                let count = positions.len() as f32;
                let center = [total[0] / count, total[1] / count, total[2] / count];
                positions
                    .iter()
                    .map(|position| euclidean_distance(*position, center))
                    .sum::<f32>()
                    / positions.len() as f32
            }
        } else {
            let previous_positions = active_units
                .iter()
                .map(|unit| (unit.id, unit.semantic_position))
                .collect::<HashMap<_, _>>();
            let total = routing
                .position_updates
                .iter()
                .map(|(id, next)| {
                    previous_positions
                        .get(id)
                        .map(|prev| euclidean_distance(*prev, *next))
                        .unwrap_or(0.0)
                })
                .sum::<f32>();
            total / routing.position_updates.len() as f32
        };

        {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            memory.update_positions(&routing.position_updates);
        }
        self.publish_memory_snapshot();

        Ok(ChunkIngestionReport {
            chunks_committed: 1,
            activated_units_observed: build_output.activated_units.len() as u64,
            new_units_discovered: new_units_discovered + candidate_observations,
            reused_units_observed,
            candidate_observations,
            candidate_promotions,
            anchors_observed: hierarchy.anchors.len() as u64,
            map_adjustments: routing.map_adjustments as u64,
            mean_displacement: displacement_mean,
            cache_hits,
            cache_lookups,
            intent_counts: intent_key
                .into_iter()
                .map(|key| (key, 1))
                .collect::<BTreeMap<_, _>>(),
        })
    }

    fn apply_training_chunk_report(
        &self,
        report: ChunkIngestionReport,
        options: &TrainingOptions,
        status: &mut TrainingJobStatus,
        memory_before: i64,
        last_memory_kb: &mut i64,
    ) -> bool {
        status.learning_metrics.new_units_discovered += report.new_units_discovered;
        status.learning_metrics.map_adjustments += report.map_adjustments;
        status.learning_metrics.anchors_protected += report.anchors_observed;
        for (intent_key, count) in report.intent_counts {
            *status.intent_distribution.entry(intent_key).or_insert(0) += count;
        }

        let (
            current_memory,
            database_health,
            bloom_false_positive_rate,
            cache_hit_rate,
            pruned_candidates_total,
            candidate_promotions_total,
            memory_breakdown,
            daily_growth_mb,
        ) = {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            let stage = memory.database_health().maturity_stage;
            let stage_budget = self.memory_budget_for(stage);
            let current = memory.estimate_memory_kb();
            let max_memory_kb = (options.max_memory_delta_mb.max(0.0) * 1024.0) as i64;
            let daily_growth_limit_mb = match options.execution_mode {
                TrainingExecutionMode::User => options
                    .daily_growth_limit_mb
                    .unwrap_or(self.config.governance.daily_growth_limit_mb),
                TrainingExecutionMode::Development => options
                    .daily_growth_limit_mb
                    .unwrap_or(stage_budget.daily_growth_limit_mb)
                    .min(stage_budget.daily_growth_limit_mb),
            }
            .max(0.0);
            let current_daily_growth_kb = memory.growth_for_today();
            let remaining_daily_growth_kb =
                ((daily_growth_limit_mb * 1024.0) - current_daily_growth_kb).max(0.0) as i64;
            let job_budget_target_kb = memory_before + max_memory_kb;
            let daily_budget_target_kb = *last_memory_kb + remaining_daily_growth_kb;
            let target_memory_kb = job_budget_target_kb.min(daily_budget_target_kb);
            let job_budget_bound = job_budget_target_kb <= daily_budget_target_kb;
            let daily_budget_bound = daily_budget_target_kb <= job_budget_target_kb;

            if current > target_memory_kb {
                let pruned_candidates_before = memory.pruned_candidates_total();
                let report = memory.prune_to_memory_budget(target_memory_kb.max(memory_before));
                let pruned_candidate_delta =
                    memory.pruned_candidates_total() - pruned_candidates_before;
                status.learning_metrics.units_pruned +=
                    report.pruned_units + pruned_candidate_delta;
                status.learning_metrics.anchors_protected += report.anchors_protected;
                let mut trigger_parts = Vec::new();
                if job_budget_bound {
                    status
                        .warnings
                        .push("memory_budget_triggered_pruning".to_string());
                    trigger_parts.push("memory_budget_limit");
                }
                if daily_budget_bound {
                    status
                        .warnings
                        .push("daily_memory_limit_triggered_pruning".to_string());
                    trigger_parts.push("daily_memory_limit");
                }
                record_pruning_event(
                    &mut status.learning_metrics,
                    &report,
                    &if trigger_parts.is_empty() {
                        "memory_budget_limit".to_string()
                    } else {
                        trigger_parts.join("+")
                    },
                );
            }
            let estimated = memory.estimate_memory_kb();
            let delta_since_last = (estimated - *last_memory_kb).max(0) as f32;
            if delta_since_last > 0.0 {
                let _ = memory.record_daily_growth(delta_since_last);
            }
            status.learning_metrics.memory_delta_kb = estimated - memory_before;
            *last_memory_kb = estimated;
            let database_health = memory.database_health();
            let bloom = memory.bloom_stats();
            let (cache_hits, cache_lookups) = memory.pattern_cache_stats();
            let memory_breakdown = memory.memory_usage_breakdown_mb();
            let bloom_false_positive_rate = if bloom.maybe_hits == 0 {
                0.0
            } else {
                bloom.false_positives as f32 / bloom.maybe_hits as f32
            };
            let cache_hit_rate = if cache_lookups == 0 {
                0.0
            } else {
                cache_hits as f32 / cache_lookups as f32
            };
            (
                estimated,
                database_health,
                bloom_false_positive_rate,
                cache_hit_rate,
                memory.pruned_candidates_total(),
                memory.candidate_promotions_total(),
                memory_breakdown,
                memory.growth_for_today() / 1024.0,
            )
        };
        self.publish_memory_snapshot();

        status.learning_metrics.memory_delta_kb = current_memory - memory_before;
        status.learning_metrics.database_health = database_health.clone();
        let effective_delta_kb = status.learning_metrics.memory_delta_kb.max(1) as f32;
        let total_observed =
            status.learning_metrics.new_units_discovered + status.learning_metrics.units_pruned;
        status.learning_metrics.efficiency.units_discovered_per_kb =
            status.learning_metrics.new_units_discovered as f32 / effective_delta_kb;
        status.learning_metrics.efficiency.units_discovered_per_mb =
            status.learning_metrics.new_units_discovered as f32
                / (effective_delta_kb / 1024.0).max(0.001);
        status.learning_metrics.efficiency.pruned_units_percent =
            status.learning_metrics.units_pruned as f32 / total_observed.max(1) as f32;
        status.learning_metrics.efficiency.anchor_density =
            database_health.anchor_units as f32 / database_health.total_units.max(1) as f32;
        status.learning_metrics.efficiency.candidate_to_active_ratio =
            database_health.candidate_units as f32 / database_health.active_units.max(1) as f32;
        status.learning_metrics.efficiency.pruned_candidates_percent = pruned_candidates_total
            as f32
            / (pruned_candidates_total
                + candidate_promotions_total
                + database_health.candidate_units)
                .max(1) as f32;
        status
            .learning_metrics
            .efficiency
            .avg_observations_to_promotion = if database_health.active_candidates == 0 {
            0.0
        } else {
            (report
                .candidate_observations
                .max(report.candidate_promotions)) as f32
                / database_health.active_candidates as f32
        };
        status.learning_metrics.efficiency.map_rebuild_frequency =
            status.learning_metrics.map_adjustments as f32
                / status.progress.chunks_processed.max(1) as f32;
        status.learning_metrics.efficiency.cache_hit_rate = cache_hit_rate;
        status
            .learning_metrics
            .efficiency
            .bloom_filter_false_positive_rate = bloom_false_positive_rate;
        status.learning_metrics.memory_governance.episodic_memory_mb = memory_breakdown.0;
        status.learning_metrics.memory_governance.core_memory_mb = memory_breakdown.1;
        status.learning_metrics.memory_governance.candidate_pool_mb = memory_breakdown.2;
        status.learning_metrics.memory_governance.daily_growth_mb = daily_growth_mb;
        status
            .learning_metrics
            .memory_governance
            .pruning_rate_per_hour = status.learning_metrics.units_pruned as f32;
        false
    }

    fn store_job(&self, status: TrainingJobStatus) {
        let statuses = {
            let mut jobs = self.jobs.lock().expect("jobs mutex poisoned");
            let mut status = status;
            status.performance.queue_depths = self.scheduler.depths();
            jobs.insert(status.job_id.clone(), status);
            jobs.values().cloned().collect::<Vec<_>>()
        };

        let db = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.db()
        };
        let _ = db.save_training_jobs(&statuses);
    }

    fn session_snapshot(&self) -> SessionDocuments {
        self.session_documents
            .lock()
            .expect("session documents mutex poisoned")
            .clone()
    }

    fn replace_session_documents(&self, documents: Vec<RetrievedDocument>) {
        let mut session = self
            .session_documents
            .lock()
            .expect("session documents mutex poisoned");
        session.documents = documents;
        session.query_state = DocumentQueryState::default();
    }

    fn remember_document_turn(&self, prompt: &str, answer: &DocumentAnswer) {
        let mut session = self
            .session_documents
            .lock()
            .expect("session documents mutex poisoned");
        if session.documents.is_empty() {
            return;
        }
        session.query_state.previous_prompt = Some(prompt.to_string());
        session.query_state.carry_terms = answer.carry_terms.clone();
    }

    fn current_sequence_state(&self) -> SequenceState {
        self.load_memory_snapshot().sequence_state()
    }

    fn resolve_intent_profile(
        &self,
        raw_input: &str,
        context: &ContextMatrix,
        sequence: &SequenceState,
        has_active_document_context: bool,
    ) -> IntentProfile {
        let heuristic = IntentDetector::classify(
            raw_input,
            context,
            sequence,
            has_active_document_context,
            &self.config.intent,
        );
        let memory_scores = self.intent_scores_from_memory(raw_input);
        if memory_scores.is_empty() {
            let mut fallback = heuristic;
            fallback
                .reasons
                .push("heuristic_intent_fallback".to_string());
            return fallback;
        }

        let mut combined_scores = heuristic.scores.clone();
        for item in &mut combined_scores {
            let memory_score = memory_scores.get(&item.intent).copied().unwrap_or(0.0);
            item.score = (memory_score * 0.65) + (item.score * 0.35);
        }
        combined_scores.sort_by(|lhs, rhs| rhs.score.total_cmp(&lhs.score));

        let top = combined_scores
            .first()
            .map(|entry| entry.score)
            .unwrap_or(0.0);
        let second = combined_scores
            .get(1)
            .map(|entry| entry.score)
            .unwrap_or(0.0);
        let ambiguous = (top - second) < self.config.intent.intent_ambiguity_margin;
        let confidence = profile_confidence(
            top,
            second,
            self.config.intent.intent_floor_threshold,
            self.config.intent.intent_ambiguity_margin,
            heuristic.certainty_bias,
            self.config.intent.certainty_softener_weight,
        );
        if top < self.config.intent.intent_floor_threshold || ambiguous {
            let mut fallback = heuristic;
            fallback
                .reasons
                .push("intent_memory_low_confidence_fallback".to_string());
            return fallback;
        }

        let primary = combined_scores
            .first()
            .map(|entry| entry.intent)
            .unwrap_or(heuristic.primary);
        let mut reasons = heuristic.reasons.clone();
        reasons.push("intent_memory_primary".to_string());
        reasons.push(format!(
            "intent_memory_top={}",
            combined_scores
                .iter()
                .take(3)
                .map(|score| format!("{:?}:{:.2}", score.intent, score.score))
                .collect::<Vec<_>>()
                .join("|")
        ));

        IntentProfile {
            primary,
            confidence,
            top_score: top,
            second_score: second,
            ambiguous,
            wants_brief: heuristic.wants_brief,
            references_document_context: heuristic.references_document_context,
            certainty_bias: heuristic.certainty_bias,
            fallback_mode: crate::types::IntentFallbackMode::None,
            scores: combined_scores,
            reasons,
        }
    }

    fn intent_scores_from_memory(&self, raw_input: &str) -> HashMap<IntentKind, f32> {
        let normalized = input::normalize_text(raw_input);
        if normalized.is_empty() {
            return HashMap::new();
        }

        let matched_units = {
            let snapshot = self.load_memory_snapshot();
            snapshot.top_channel_matches(
                MemoryChannel::Intent,
                &normalized,
                self.config.governance.max_candidate_pool,
            )
        };

        let mut scores = HashMap::new();
        for unit in matched_units {
            let labels = intent_labels_from_contexts(&unit.contexts);
            if labels.is_empty() {
                continue;
            }
            let unit_score = ((unit.utility_score.clamp(0.0, 1.0)
                + unit.confidence.clamp(0.0, 1.0)
                + unit.trust_score.clamp(0.0, 1.0))
                / 3.0)
                * (1.0 + (unit.frequency as f32).ln_1p() * 0.08);
            for label in labels {
                *scores.entry(label).or_insert(0.0) += unit_score;
            }
        }

        let max_score = scores.values().copied().fold(0.0, f32::max);
        if max_score > 0.0 {
            for score in scores.values_mut() {
                *score = (*score / max_score).clamp(0.0, 1.0);
            }
        }
        scores
    }

    fn reasoning_support_from_memory(&self, raw_input: &str) -> ReasoningSupport {
        let normalized = input::normalize_text(raw_input);
        if normalized.is_empty() {
            return ReasoningSupport::default();
        }

        let matched_units = {
            let snapshot = self.load_memory_snapshot();
            snapshot.top_channel_matches(MemoryChannel::Reasoning, &normalized, 12)
        };
        if matched_units.is_empty() {
            return ReasoningSupport::default();
        }

        let mut confidence = 0.0;
        let mut labels = Vec::new();
        let mut unit_ids = Vec::new();
        for unit in matched_units.iter().take(8) {
            unit_ids.push(unit.id);
            confidence += (unit.utility_score.clamp(0.0, 1.0)
                + unit.confidence.clamp(0.0, 1.0)
                + unit.trust_score.clamp(0.0, 1.0))
                / 3.0;
            labels.push(
                unit.content
                    .split_whitespace()
                    .take(6)
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }

        ReasoningSupport {
            unit_ids,
            confidence: (confidence / matched_units.len().min(8) as f32).clamp(0.0, 1.0),
            labels,
        }
    }

    fn answer_from_personal_memory(
        &self,
        prompt: &str,
        intent_hint: IntentKind,
    ) -> Option<DocumentAnswer> {
        let prompt_terms = normalized_terms(prompt);
        if prompt_terms.is_empty() {
            return None;
        }
        let focus = infer_evidence_focus(prompt);
        let contexts = {
            let snapshot = self.load_memory_snapshot();
            let mut scored_contexts = BTreeMap::new();

            for unit in snapshot
                .all_units()
                .into_iter()
                .filter(|unit| unit.level != crate::types::UnitLevel::Char)
            {
                let unit_terms = normalized_terms(&unit.content);
                let unit_overlap = token_overlap(&prompt_terms, &unit_terms);
                let base_score = (0.44 * unit_overlap)
                    + (0.12 * unit.utility_score.clamp(0.0, 1.0))
                    + (0.10 * unit.confidence.clamp(0.0, 1.0))
                    + (0.08 * unit.salience_score.clamp(0.0, 1.0));
                let candidate_contexts = if unit.contexts.is_empty() {
                    vec![unit.content.clone()]
                } else {
                    unit.contexts.clone()
                };

                for context in candidate_contexts {
                    let cleaned = clean_memory_context(&context);
                    if cleaned.split_whitespace().count() < 2 {
                        continue;
                    }
                    let declarative_score = declarative_fact_score(&cleaned);
                    if cleaned.trim_end().ends_with('?')
                        || input::normalize_text(&cleaned)
                            .eq_ignore_ascii_case(&input::normalize_text(prompt))
                        || declarative_score < 0.18
                    {
                        continue;
                    }

                    let context_terms = normalized_terms(&cleaned);
                    let context_overlap = token_overlap(&prompt_terms, &context_terms);
                    if context_overlap < 0.14 && unit_overlap < 0.12 {
                        continue;
                    }

                    let personal_reference = first_person_reference_score(&cleaned);
                    let focus_signal = evidence_focus_score(focus, &cleaned)
                        .max(definition_sentence_score(&cleaned));
                    let value_signal = answer_candidate_signal(&cleaned);
                    let score = (0.42 * context_overlap)
                        + (0.24 * unit_overlap)
                        + (0.16 * base_score)
                        + (0.08 * personal_reference)
                        + (0.12 * declarative_score)
                        + (0.10 * focus_signal)
                        + (0.08 * value_signal)
                        + (0.08 * sentence_shape_score(&cleaned));
                    if score < 0.28 {
                        continue;
                    }

                    match scored_contexts.get_mut(&cleaned) {
                        Some(existing) if *existing >= score => {}
                        Some(existing) => *existing = score,
                        None => {
                            scored_contexts.insert(cleaned, score);
                        }
                    }
                }
            }

            let mut ranked = scored_contexts
                .into_iter()
                .map(|(context, score)| (score, context))
                .collect::<Vec<_>>();
            ranked.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));
            ranked
        };

        let top_score = contexts.first().map(|(score, _)| *score).unwrap_or(0.0);
        if top_score < 0.34 {
            return None;
        }

        for (score, context) in contexts.iter().take(3) {
            if let Some(answer_text) = extract_memory_answer(prompt, context, focus) {
                return Some(DocumentAnswer {
                    text: answer_text,
                    confidence: (0.54 + (score * 0.32)).clamp(0.42, 0.96),
                    supporting_sources: vec!["memory://personal".to_string()],
                    supporting_passages: vec![context.clone()],
                    carry_terms: normalized_terms(context).into_iter().take(8).collect(),
                });
            }
        }

        let snippets = contexts
            .into_iter()
            .map(|(_, context)| context)
            .take(6)
            .collect::<Vec<_>>();
        if snippets.is_empty() {
            return None;
        }

        let raw_content = snippets.join(". ");
        let documents = vec![RetrievedDocument {
            source_url: "memory://personal".to_string(),
            title: "personal_memory".to_string(),
            normalized_content: input::normalize_text(&raw_content),
            raw_content,
            retrieved_at: chrono::Utc::now(),
            trust_score: 0.99,
            cached: true,
        }];

        if let Some(answer_text) = grounded_evidence_answer(prompt, &documents) {
            let normalized_answer = input::normalize_text(&answer_text);
            let normalized_prompt = input::normalize_text(prompt);
            if !normalized_answer.is_empty() && normalized_answer != normalized_prompt {
                return Some(DocumentAnswer {
                    text: answer_text,
                    confidence: 0.48,
                    supporting_sources: vec!["memory://personal".to_string()],
                    supporting_passages: snippets.iter().take(2).cloned().collect(),
                    carry_terms: normalized_terms(&documents[0].raw_content)
                        .into_iter()
                        .take(8)
                        .collect(),
                });
            }
        }

        let answer = answer_question(prompt, &documents, None, Some(intent_hint))?;
        let normalized_answer = input::normalize_text(&answer.text);
        let normalized_prompt = input::normalize_text(prompt);
        if answer.confidence >= 0.18
            && !normalized_answer.is_empty()
            && normalized_answer != normalized_prompt
        {
            Some(answer)
        } else {
            None
        }
    }

    fn ingest_documents_into_core(&self, documents: &[RetrievedDocument]) {
        for document in documents {
            let context = format!("document {} {}", document.title, document.source_url);
            self.ingest_text_into_memory(
                &document.raw_content,
                SourceKind::TrainingDocument,
                &context,
            );
        }
    }

    fn ingest_web_evidence_into_memory(
        &self,
        documents: &[RetrievedDocument],
        query: &str,
        intent_profile: &IntentProfile,
        trust_config: &TrustConfig,
    ) {
        let corroborated = documents
            .iter()
            .filter(|document| {
                document.trust_score
                    >= trust_config.min_source_trust + trust_config.corroboration_bonus
            })
            .count();
        let min_corroborating_sources = trust_config.min_corroborating_sources.max(1);
        let freshness_sensitive = intent_profile
            .reasons
            .iter()
            .any(|reason| reason.starts_with("temporal_cues="))
            || matches!(intent_profile.primary, IntentKind::Verify);
        for document in documents {
            let context = format!("web {} {}", query, document.source_url);
            let promote_to_core = if freshness_sensitive {
                corroborated >= min_corroborating_sources
                    && document.trust_score
                        >= trust_config.min_source_trust + trust_config.corroboration_bonus
            } else {
                corroborated >= min_corroborating_sources || document.trust_score >= 0.78
            };
            let source_kind = if promote_to_core {
                SourceKind::TrainingUrl
            } else {
                SourceKind::Retrieval
            };
            self.ingest_text_into_memory(&document.raw_content, source_kind, &context);
        }
    }

    fn ingest_interaction_input(&self, text: &str, context: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.ingest_text_into_memory(text, SourceKind::UserInput, context);
    }

    fn ingest_text_into_memory(&self, text: &str, source_kind: SourceKind, context: &str) {
        if text.trim().is_empty() {
            return;
        }

        let packet = input::ingest_raw(text, !matches!(source_kind, SourceKind::UserInput));
        let build_output = self.build_units(&packet);
        let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &self.config.builder);
        let context_summary = if context.is_empty() {
            summarize_packet(&packet)
        } else {
            context.to_string()
        };

        let active_ids = {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            memory.ingest_hierarchy(&hierarchy, source_kind, &context_summary)
        };

        let (active_units, all_units) = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            (memory.get_units(&active_ids), memory.all_units())
        };

        let routing = route_units(&active_units, &all_units, &self.config.semantic_map);

        {
            let mut memory = self.memory.lock().expect("memory mutex poisoned");
            memory.update_positions(&routing.position_updates);
        }
        self.publish_memory_snapshot();
    }

    fn current_database_health(&self) -> DatabaseHealthMetrics {
        let memory = self.memory.lock().expect("memory mutex poisoned");
        memory.database_health()
    }

    fn memory_budget_for(
        &self,
        stage: crate::types::DatabaseMaturityStage,
    ) -> &crate::config::MemoryBudgetTier {
        match stage {
            crate::types::DatabaseMaturityStage::ColdStart => {
                &self.config.memory_budgets.cold_start
            }
            crate::types::DatabaseMaturityStage::Growth => &self.config.memory_budgets.growth_phase,
            crate::types::DatabaseMaturityStage::Stable => &self.config.memory_budgets.stable_phase,
        }
    }

    fn record_test_observation(
        &self,
        query: &str,
        memory_before_kb: i64,
        build_output: &crate::types::BuildOutput,
        governance: Option<&crate::types::GovernanceReport>,
        retrieval_triggered: bool,
        decision: &SearchDecision,
        merged: &MergedState,
        scored: &[crate::types::ScoredCandidate],
        final_answer_confidence: f32,
        timings: ObservationTimings,
        mut sources_consulted: Vec<String>,
    ) {
        let Some(observer) = &self.observer else {
            return;
        };

        let database_health = self.current_database_health();
        let memory_after_kb = {
            let memory = self.memory.lock().expect("memory mutex poisoned");
            memory.estimate_memory_kb()
        };
        sources_consulted.extend(
            merged
                .evidence
                .documents
                .iter()
                .map(|doc| doc.source_url.clone()),
        );
        let retrieval_reason = if decision.reasons.is_empty() {
            None
        } else {
            Some(decision.reasons.join("; "))
        };
        let scoring = scored.first().map(|candidate| ObservationScoring {
            candidates_scored: scored.len() as u32,
            top_candidate_score: candidate.score,
            score_breakdown: if self.config.telemetry.log_score_breakdowns {
                Some(ObservedScoreBreakdown::from(&candidate.breakdown))
            } else {
                None
            },
        });
        let observation = observer.build_observation(
            Uuid::new_v4().to_string(),
            query,
            &self.config,
            timings,
            ObservationUnits {
                units_discovered: build_output.new_units.len() as u32,
                units_activated: build_output.activated_units.len() as u32,
                new_units_created: build_output.new_units.len() as u32,
                units_pruned: governance
                    .map(|report| report.pruned_units as u32)
                    .unwrap_or(0),
            },
            ObservationMemory {
                memory_delta_kb: memory_after_kb - memory_before_kb,
                episodic_count: database_health.episodic_units as u32,
                core_count: database_health.core_units as u32,
                anchors_protected: governance
                    .map(|report| report.anchors_protected as u32)
                    .unwrap_or(database_health.anchor_units as u32),
            },
            ObservationRetrieval {
                retrieval_triggered,
                retrieval_reason,
                sources_consulted: unique_strings(sources_consulted),
                evidence_merged: merged.evidence.documents.len() as u32,
            },
            scoring.unwrap_or_default(),
            final_answer_confidence,
        );
        let _ = observer.log_observation(&observation);
    }

    fn build_units(&self, packet: &InputPacket) -> crate::types::BuildOutput {
        let database_health = self.current_database_health();
        let snapshot = self.load_memory_snapshot();
        UnitBuilder::ingest_with_governance_snapshot(
            packet,
            &self.config.builder,
            &self.config.governance,
            &database_health,
            Some(snapshot.as_ref()),
        )
    }

    fn enqueue_feedback(&self, feedback: Vec<crate::types::FeedbackEvent>) {
        if feedback.is_empty() {
            return;
        }
        let _ = self.feedback_tx.try_send(feedback);
    }
}

fn record_pruning_event(
    metrics: &mut crate::types::LearningMetrics,
    report: &crate::types::GovernanceReport,
    trigger: &str,
) {
    if report.pruned_units == 0
        && report.pruned_candidates == 0
        && report.pruned_references.is_empty()
    {
        return;
    }

    metrics.pruning_events.push(crate::types::PruningEventLog {
        trigger: trigger.to_string(),
        pruned_units: report.pruned_units,
        pruned_candidates: report.pruned_candidates,
        purged_polluted_units: report.purged_polluted_units,
        purged_polluted_candidates: report.purged_polluted_candidates,
        anchors_protected: report.anchors_protected,
        snapshot_path: report.snapshot_path.clone(),
        reasons: report.pruning_reasons.clone(),
        pruned_references: report.pruned_references.clone(),
        pollution_findings: report.pollution_findings.clone(),
    });
    if metrics.pruning_events.len() > 16 {
        let overflow = metrics.pruning_events.len() - 16;
        metrics.pruning_events.drain(0..overflow);
    }
}

fn hierarchy_from_activations(activations: &[crate::types::ActivatedUnit]) -> UnitHierarchy {
    let mut levels = BTreeMap::new();
    for activation in activations {
        let key = format!("{:?}", activation.level).to_lowercase();
        levels
            .entry(key)
            .or_insert_with(Vec::new)
            .push(activation.clone());
    }
    UnitHierarchy {
        levels,
        anchors: activations
            .iter()
            .filter(|activation| activation.salience >= 0.7)
            .map(|activation| activation.content.clone())
            .collect(),
        entities: activations
            .iter()
            .filter(|activation| activation.level == crate::types::UnitLevel::Word)
            .take(8)
            .map(|activation| activation.content.clone())
            .collect(),
    }
}

fn build_trace(
    packet: &InputPacket,
    routing: &RoutingResult,
    context: &ContextMatrix,
    decision: &SearchDecision,
    retrieval_query: Option<crate::types::SanitizedQuery>,
    merged: &MergedState,
    scored: &[crate::types::ScoredCandidate],
    resolved_candidate: Option<crate::types::ResolvedCandidate>,
    output_text: String,
    grounded: bool,
    confidence_stats: ConfidenceStats,
    intent_profile: IntentProfile,
    adaptive: &AdaptiveRuntimeSettings,
    output_strategy: &'static str,
    reasoning_labels: Vec<String>,
    safety_warnings: Vec<String>,
    feedback: Vec<crate::types::FeedbackEvent>,
    memory_summary: String,
    preset_sources: Vec<String>,
    queue_depths: QueueDepths,
) -> ExplainTrace {
    let mut layer_notes = vec![
        LayerNote {
            layer: 1,
            note: format!("normalized_input={}", packet.normalized_text),
        },
        LayerNote {
            layer: 6,
            note: format!("active_regions={}", routing.active_regions.join(",")),
        },
        LayerNote {
            layer: 9,
            note: format!(
                "retrieve={} reasons={} intent={:?} intent_confidence={:.2} fallback={:?} ambiguous={} top_intents={} adaptive_intent_profile={} trust_signal={} trust_profile={}",
                decision.should_retrieve,
                decision.reasons.join(","),
                intent_profile.primary,
                intent_profile.confidence,
                intent_profile.fallback_mode,
                intent_profile.ambiguous,
                intent_profile
                    .scores
                    .iter()
                    .take(3)
                    .map(|score| format!("{:?}:{:.2}", score.intent, score.score))
                    .collect::<Vec<_>>()
                    .join("|"),
                adaptive
                    .intent_profile_name
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                adaptive
                    .trust_signal_name
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                adaptive
                    .trust_profile_name
                    .clone()
                    .unwrap_or_else(|| "default".to_string())
            ),
        },
        LayerNote {
            layer: 13,
            note: format!(
                "conflicts={} reasoning_support={}",
                merged.conflict_records.len(),
                reasoning_labels.join("|")
            ),
        },
        LayerNote {
            layer: 20,
            note: format!("context_summary={}", context.summary),
        },
    ];
    if retrieval_query.is_none() {
        layer_notes.push(LayerNote {
            layer: 10,
            note: "query_builder_skipped".to_string(),
        });
    }

    let top_candidates = scored
        .iter()
        .take(5)
        .map(|candidate| format!("{}:{:.2}", candidate.content, candidate.score))
        .collect::<Vec<_>>()
        .join(" | ");
    let intent_resolution = intent_resolution_source(&intent_profile);
    let mut debug_steps = vec![
        debug_step(
            1,
            "input",
            "Input normalized and packetized.",
            [
                ("normalized_input", packet.normalized_text.clone()),
                ("training_mode", packet.training_mode.to_string()),
                ("timestamp", packet.timestamp.to_rfc3339()),
            ],
        ),
        debug_step(
            5,
            "routing",
            "Semantic routing activated candidate regions.",
            [
                ("active_regions", routing.active_regions.join(",")),
                ("neighbor_count", routing.neighbor_ids.len().to_string()),
                ("map_adjustments", routing.map_adjustments.to_string()),
            ],
        ),
        debug_step(
            7,
            "context",
            "Context matrix and sequence state prepared.",
            [
                ("context_summary", context.summary.clone()),
                ("context_cells", context.cells.len().to_string()),
            ],
        ),
        debug_step(
            9,
            "intent_resolution",
            "Intent resolved from specialized memory with fallback support.",
            [
                ("primary", format!("{:?}", intent_profile.primary)),
                ("confidence", format!("{:.3}", intent_profile.confidence)),
                ("ambiguous", intent_profile.ambiguous.to_string()),
                (
                    "fallback_mode",
                    format!("{:?}", intent_profile.fallback_mode),
                ),
                ("resolver", intent_resolution.to_string()),
                (
                    "top_intents",
                    intent_profile
                        .scores
                        .iter()
                        .take(5)
                        .map(|score| format!("{:?}:{:.2}", score.intent, score.score))
                        .collect::<Vec<_>>()
                        .join(" | "),
                ),
                ("reasons", intent_profile.reasons.join(",")),
            ],
        ),
        debug_step(
            9,
            "retrieval_gate",
            "Retrieval decision scored from entropy, freshness, and disagreement.",
            [
                ("should_retrieve", decision.should_retrieve.to_string()),
                ("score", format!("{:.3}", decision.score)),
                ("entropy", format!("{:.3}", decision.entropy)),
                ("freshness_need", format!("{:.3}", decision.freshness_need)),
                ("disagreement", format!("{:.3}", decision.disagreement)),
                ("cost_penalty", format!("{:.3}", decision.cost_penalty)),
                ("reasons", decision.reasons.join(",")),
            ],
        ),
        debug_step(
            9,
            "adaptive_policy",
            "Live intent state tuned scoring, escape, trust, and resolver behavior.",
            [
                (
                    "intent_profile",
                    adaptive
                        .intent_profile_name
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                ),
                (
                    "trust_signal",
                    adaptive
                        .trust_signal_name
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                ),
                (
                    "trust_profile",
                    adaptive
                        .trust_profile_name
                        .clone()
                        .unwrap_or_else(|| "default".to_string()),
                ),
                (
                    "scoring_weights",
                    format!(
                        "spatial={:.2},context={:.2},sequence={:.2},transition={:.2},utility={:.2},confidence={:.2},evidence={:.2}",
                        adaptive.scoring.spatial,
                        adaptive.scoring.context,
                        adaptive.scoring.sequence,
                        adaptive.scoring.transition,
                        adaptive.scoring.utility,
                        adaptive.scoring.confidence,
                        adaptive.scoring.evidence,
                    ),
                ),
                (
                    "escape",
                    format!(
                        "jump_prob={:.2},beam_width={}",
                        adaptive.escape.stochastic_jump_prob, adaptive.escape.beam_width
                    ),
                ),
                (
                    "resolver",
                    format!(
                        "temperature={:.2},min_confidence_floor={:.2},mode={}",
                        adaptive.resolver.selection_temperature,
                        adaptive.resolver.min_confidence_floor,
                        adaptive
                            .resolver_mode
                            .map(|mode| format!("{mode:?}"))
                            .unwrap_or_else(|| "derived".to_string()),
                    ),
                ),
                (
                    "trust",
                    format!(
                        "default_source_trust={:.2},min_corroborating_sources={},require_https={}",
                        adaptive.trust.default_source_trust,
                        adaptive.trust.min_corroborating_sources,
                        adaptive.trust.require_https,
                    ),
                ),
                (
                    "cost_penalty",
                    format!(
                        "base_plus_load={:.3} (+{:.3})",
                        adaptive.retrieval.cost_penalty, adaptive.additional_cost_penalty
                    ),
                ),
            ],
        ),
        debug_step(
            20,
            "scheduler",
            "Current queue depths captured for traceability.",
            [
                ("inference", queue_depths.inference.to_string()),
                (
                    "interactive_training",
                    queue_depths.interactive_training.to_string(),
                ),
                ("silent_batch", queue_depths.silent_batch.to_string()),
                ("maintenance", queue_depths.maintenance.to_string()),
            ],
        ),
    ];

    if let Some(query) = retrieval_query.as_ref() {
        debug_steps.push(debug_step(
            10,
            "query_builder",
            "External retrieval query sanitized and expanded.",
            [
                ("raw_query", query.raw_query.clone()),
                ("sanitized_query", query.sanitized_query.clone()),
                ("semantic_expansions", query.semantic_expansions.join(" | ")),
                ("removed_tokens", query.removed_tokens.join(" | ")),
                ("pii_redacted", query.pii_redacted.to_string()),
            ],
        ));
    } else {
        debug_steps.push(debug_step(
            10,
            "query_builder",
            "External query builder skipped.",
            [("status", "skipped".to_string())],
        ));
    }

    debug_steps.push(debug_step(
        13,
        "reasoning_merge",
        "Reasoning support and evidence merge prepared candidate inputs.",
        [
            ("reasoning_support", reasoning_labels.join(" | ")),
            ("candidate_ids", merged.candidate_ids.len().to_string()),
            (
                "evidence_support",
                format!("{:.3}", merged.evidence_support),
            ),
            ("conflicts", merged.conflict_records.len().to_string()),
            (
                "evidence_sources",
                merged.evidence.documents.len().to_string(),
            ),
        ],
    ));
    debug_steps.push(debug_step(
        15,
        "candidate_scoring",
        "Candidates scored across spatial, context, sequence, confidence, and evidence factors.",
        [
            (
                "candidate_count",
                confidence_stats.candidate_count.to_string(),
            ),
            (
                "mean_confidence",
                format!("{:.3}", confidence_stats.mean_confidence),
            ),
            (
                "disagreement",
                format!("{:.3}", confidence_stats.disagreement),
            ),
            ("top_candidates", top_candidates),
        ],
    ));
    debug_steps.push(debug_step(
        16,
        "resolution",
        "Fine resolver selected the output anchor.",
        [
            (
                "selected_unit",
                resolved_candidate
                    .as_ref()
                    .map(|candidate| candidate.content.clone())
                    .unwrap_or_default(),
            ),
            (
                "resolver_mode",
                resolved_candidate
                    .as_ref()
                    .map(|candidate| format!("{:?}", candidate.mode))
                    .unwrap_or_else(|| "none".to_string()),
            ),
            (
                "used_escape",
                resolved_candidate
                    .as_ref()
                    .map(|candidate| candidate.used_escape.to_string())
                    .unwrap_or_else(|| "false".to_string()),
            ),
            (
                "selected_score",
                resolved_candidate
                    .as_ref()
                    .map(|candidate| format!("{:.3}", candidate.score))
                    .unwrap_or_else(|| "0.000".to_string()),
            ),
        ],
    ));
    debug_steps.push(debug_step(
        17,
        "output",
        "Final answer decoded from the resolved candidate set.",
        [
            ("grounded", grounded.to_string()),
            ("used_retrieval", decision.should_retrieve.to_string()),
            ("strategy", output_strategy.to_string()),
            ("predicted_text", output_text.clone()),
        ],
    ));
    debug_steps.push(debug_step(
        18,
        "feedback",
        "Feedback events captured for post-answer learning.",
        [
            ("feedback_events", feedback.len().to_string()),
            (
                "events",
                feedback
                    .iter()
                    .map(|event| format!("L{}:{}:{:.2}", event.layer, event.event, event.impact))
                    .collect::<Vec<_>>()
                    .join(" | "),
            ),
        ],
    ));
    debug_steps.push(debug_step(
        20,
        "memory",
        "Memory state recorded after answer generation.",
        [("memory_summary", memory_summary.clone())],
    ));

    ExplainTrace {
        intent_profile,
        layer_notes,
        debug_steps,
        active_regions: routing.active_regions.clone(),
        retrieval_query,
        evidence_sources: if preset_sources.is_empty() {
            merged
                .evidence
                .documents
                .iter()
                .map(|doc| doc.source_url.clone())
                .collect()
        } else {
            preset_sources
        },
        score_breakdowns: scored.iter().take(8).cloned().collect(),
        selected_unit: resolved_candidate.map(|candidate| candidate.content),
        safety_warnings,
        feedback_events: feedback,
        memory_summary,
    }
}

fn debug_step<const N: usize>(
    layer: u8,
    stage: &str,
    summary: &str,
    details: [(&str, String); N],
) -> DebugStep {
    DebugStep {
        layer,
        stage: stage.to_string(),
        summary: summary.to_string(),
        details: details
            .into_iter()
            .map(|(key, value)| (key.to_string(), value))
            .collect(),
    }
}

fn intent_resolution_source(intent_profile: &IntentProfile) -> &'static str {
    if intent_profile
        .reasons
        .iter()
        .any(|reason| reason == "intent_memory_primary")
    {
        "memory_guided"
    } else if intent_profile
        .reasons
        .iter()
        .any(|reason| reason == "heuristic_intent_fallback")
    {
        "heuristic_fallback"
    } else if intent_profile.primary == IntentKind::Unknown && intent_profile.reasons.is_empty() {
        "not_applicable"
    } else {
        "direct_assignment"
    }
}

fn summarize_packet(packet: &InputPacket) -> String {
    packet
        .normalized_text
        .split_whitespace()
        .take(12)
        .collect::<Vec<_>>()
        .join(" ")
}

fn prepare_training_chunk_for_ingest(
    chunk: &DatasetTextChunk,
    config: &EngineConfig,
    database_health: &DatabaseHealthMetrics,
    snapshot: Option<&MemorySnapshot>,
    collect_intent_telemetry: bool,
) -> Result<PreparedTrainingChunk, String> {
    let normalized = input::normalize_text(&chunk.content);
    if normalized.is_empty() {
        return Ok(PreparedTrainingChunk {
            context_label: chunk.context_label.clone(),
            packet_summary: String::new(),
            hierarchy: UnitHierarchy::default(),
            activated_units_observed: 0,
            anchors_observed: 0,
            intent_key: None,
        });
    }

    let intent_key = if collect_intent_telemetry {
        let telemetry_window = normalized
            .split_whitespace()
            .take(96)
            .collect::<Vec<_>>()
            .join(" ");
        let intent_profile = IntentDetector::classify(
            &telemetry_window,
            &ContextMatrix::default(),
            &SequenceState::default(),
            false,
            &config.intent,
        );
        Some(format!("{:?}", intent_profile.primary).to_lowercase())
    } else {
        None
    };

    let packet = input::ingest_raw(&normalized, true);
    let build_output = UnitBuilder::ingest_with_governance_snapshot(
        &packet,
        &config.builder,
        &config.governance,
        database_health,
        snapshot,
    );
    let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &config.builder);
    let anchors_observed = hierarchy.anchors.len() as u64;

    Ok(PreparedTrainingChunk {
        context_label: chunk.context_label.clone(),
        packet_summary: summarize_packet(&packet),
        hierarchy,
        activated_units_observed: build_output.activated_units.len() as u64,
        anchors_observed,
        intent_key,
    })
}

fn ingest_prepared_training_batch_into_store(
    store: &mut MemoryStore,
    batch: &[PreparedTrainingChunk],
    source_kind: SourceKind,
    target_memory: MemoryType,
    memory_channels: &[MemoryChannel],
    merge_to_core: bool,
) -> Result<(ChunkIngestionReport, Vec<Uuid>), String> {
    if batch.is_empty() {
        return Ok((ChunkIngestionReport::default(), Vec::new()));
    }

    let effective_memory = if merge_to_core {
        target_memory
    } else {
        MemoryType::Episodic
    };
    let mut report = ChunkIngestionReport {
        chunks_committed: batch.len() as u64,
        ..ChunkIngestionReport::default()
    };
    let mut active_ids = Vec::new();

    for prepared in batch {
        let mut context_segments = Vec::new();
        if !prepared.context_label.is_empty() {
            context_segments.push(prepared.context_label.clone());
        }
        context_segments.push(prepared.packet_summary.clone());
        if memory_channels.contains(&MemoryChannel::Intent) {
            if let Some(intent_key) = prepared.intent_key.as_deref() {
                context_segments.push(format!("intent_label:{intent_key}"));
            }
        }
        let context_summary = context_segments.join(" | ");
        let ingest_report = store.ingest_hierarchy_with_channels_report(
            &prepared.hierarchy,
            source_kind,
            &context_summary,
            effective_memory,
            memory_channels,
        );
        report.activated_units_observed += prepared.activated_units_observed;
        report.new_units_discovered +=
            ingest_report.new_units + ingest_report.candidate_observations;
        report.reused_units_observed += ingest_report.reused_units;
        report.candidate_observations += ingest_report.candidate_observations;
        report.candidate_promotions += ingest_report.candidate_promotions;
        report.anchors_observed += prepared.anchors_observed;
        report.cache_hits += ingest_report.cache_hits;
        report.cache_lookups += ingest_report.cache_lookups;
        if let Some(intent_key) = prepared.intent_key.as_ref() {
            *report.intent_counts.entry(intent_key.clone()).or_insert(0) += 1;
        }
        active_ids.extend(ingest_report.active_ids);
    }

    Ok((report, active_ids))
}

fn accumulate_chunk_ingestion_report(
    target: &mut ChunkIngestionReport,
    report: &ChunkIngestionReport,
) {
    target.chunks_committed += report.chunks_committed;
    target.activated_units_observed += report.activated_units_observed;
    target.new_units_discovered += report.new_units_discovered;
    target.reused_units_observed += report.reused_units_observed;
    target.candidate_observations += report.candidate_observations;
    target.candidate_promotions += report.candidate_promotions;
    target.anchors_observed += report.anchors_observed;
    target.map_adjustments += report.map_adjustments;
    target.mean_displacement += report.mean_displacement;
    target.cache_hits += report.cache_hits;
    target.cache_lookups += report.cache_lookups;
    for (intent_key, count) in &report.intent_counts {
        *target.intent_counts.entry(intent_key.clone()).or_insert(0) += count;
    }
}

fn removed_count_from_governance(report: &crate::types::GovernanceReport) -> u64 {
    report.pruned_units
        + report.pruned_candidates
        + report.purged_polluted_units
        + report.purged_polluted_candidates
}

fn mean_displacement_for_training_routing(
    active_units: &[Unit],
    position_updates: &[(Uuid, [f32; 3])],
) -> f32 {
    if position_updates.is_empty() {
        let positions = active_units
            .iter()
            .map(|unit| unit.semantic_position)
            .collect::<Vec<_>>();
        if positions.is_empty() {
            return 0.0;
        }
        let mut total = [0.0, 0.0, 0.0];
        for position in &positions {
            total[0] += position[0];
            total[1] += position[1];
            total[2] += position[2];
        }
        let count = positions.len() as f32;
        let center = [total[0] / count, total[1] / count, total[2] / count];
        return positions
            .iter()
            .map(|position| euclidean_distance(*position, center))
            .sum::<f32>()
            / positions.len() as f32;
    }

    let previous_positions = active_units
        .iter()
        .map(|unit| (unit.id, unit.semantic_position))
        .collect::<HashMap<_, _>>();
    let total = position_updates
        .iter()
        .map(|(id, next)| {
            previous_positions
                .get(id)
                .map(|prev| euclidean_distance(*prev, *next))
                .unwrap_or(0.0)
        })
        .sum::<f32>();
    total / position_updates.len() as f32
}

fn apply_parallel_training_progress(
    status: &mut TrainingJobStatus,
    worker_count: usize,
    active_workers: &AtomicUsize,
    queued_chunks: usize,
    prepared_chunks: &AtomicU64,
    committed_batches: &AtomicU64,
    committed_chunks: u64,
    bytes_processed: u64,
) {
    status.progress.worker_count = worker_count;
    status.progress.active_workers = active_workers.load(Ordering::Relaxed);
    status.progress.queued_chunks = queued_chunks;
    status.progress.prepared_chunks = prepared_chunks.load(Ordering::Relaxed);
    status.progress.committed_batches = committed_batches.load(Ordering::Relaxed);
    status.progress.chunks_processed = committed_chunks;
    status.progress.bytes_processed = bytes_processed;
}

fn sanitize_training_shard_name(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>();
    sanitized.trim_matches('_').to_string()
}

fn training_shard_db_path(
    source_name: &str,
    shard_index: usize,
    segment_index: usize,
) -> std::path::PathBuf {
    let directory = format!(
        "spse_training_shard_{}_{}_{}_{}",
        sanitize_training_shard_name(source_name),
        shard_index,
        segment_index,
        Uuid::new_v4()
    );
    let dir = std::env::temp_dir().join(directory);
    let _ = fs::create_dir_all(&dir);
    dir.join("shard.db")
}

fn cleanup_training_shard_db(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir_all(parent);
    } else {
        let _ = fs::remove_file(path);
    }
}

fn decode_document_bytes(content: &str, mime: Option<&str>) -> Vec<u8> {
    let trimmed = content.trim();
    let prefers_binary = mime
        .map(|value| !value.trim().to_ascii_lowercase().starts_with("text/"))
        .unwrap_or(false);

    if let Ok(bytes) = BASE64.decode(trimmed) {
        let binary_signature = bytes.starts_with(b"%PDF-") || bytes.starts_with(b"PK\x03\x04");
        if prefers_binary || binary_signature || looks_like_base64_blob(trimmed) {
            return bytes;
        }
    }

    content.as_bytes().to_vec()
}

fn local_evidence_state(
    documents: &[crate::types::RetrievedDocument],
    builder: &crate::config::UnitBuilderConfig,
    governance: &crate::config::GovernanceConfig,
    database_health: &crate::types::DatabaseHealthMetrics,
    snapshot: Option<&crate::memory::store::MemorySnapshot>,
) -> crate::types::EvidenceState {
    let evidence_units = documents
        .iter()
        .flat_map(|doc| {
            let packet = input::ingest_raw(&doc.normalized_content, false);
            UnitBuilder::ingest_with_governance_snapshot(
                &packet,
                builder,
                governance,
                database_health,
                snapshot,
            )
            .activated_units
        })
        .take(64)
        .collect::<Vec<_>>();
    let average_trust = if documents.is_empty() {
        0.0
    } else {
        documents.iter().map(|doc| doc.trust_score).sum::<f32>() / documents.len() as f32
    };

    crate::types::EvidenceState {
        documents: documents.to_vec(),
        evidence_units,
        warnings: Vec::new(),
        average_trust,
    }
}

fn simple_result(
    text: String,
    evidence_sources: Vec<String>,
    route: &str,
    request_input: &str,
) -> ProcessResult {
    simple_result_with_intent(
        text,
        evidence_sources,
        route,
        request_input,
        IntentProfile::default(),
        String::new(),
    )
}

fn simple_result_with_intent(
    text: String,
    evidence_sources: Vec<String>,
    route: &str,
    request_input: &str,
    intent_profile: IntentProfile,
    memory_summary: String,
) -> ProcessResult {
    let evidence_sources = unique_strings(evidence_sources);
    let predicted_text = text.clone();
    let intent_primary = format!("{:?}", intent_profile.primary);
    let normalized_input = input::normalize_text(request_input);
    let intent_resolution = intent_resolution_source(&intent_profile).to_string();
    let intent_confidence = format!("{:.3}", intent_profile.confidence);
    let intent_reasons = intent_profile.reasons.join(",");
    ProcessResult {
        predicted_text,
        confidence: 1.0,
        used_retrieval: false,
        trace: ExplainTrace {
            intent_profile,
            memory_summary: memory_summary.clone(),
            evidence_sources,
            layer_notes: vec![LayerNote {
                layer: 20,
                note: route.to_string(),
            }],
            debug_steps: vec![
                debug_step(
                    1,
                    "input",
                    "Input normalized for the direct response path.",
                    [
                        ("normalized_input", normalized_input),
                        ("route", route.to_string()),
                    ],
                ),
                debug_step(
                    9,
                    "intent_resolution",
                    "Direct response path resolved intent before answer generation.",
                    [
                        ("primary", intent_primary),
                        ("confidence", intent_confidence),
                        ("resolution_source", intent_resolution),
                        ("reasons", intent_reasons),
                    ],
                ),
                debug_step(
                    17,
                    "output",
                    "Direct response generated.",
                    [
                        ("route", route.to_string()),
                        ("predicted_text", text.clone()),
                    ],
                ),
            ],
            active_regions: vec![],
            retrieval_query: None,
            score_breakdowns: vec![],
            selected_unit: None,
            safety_warnings: vec![],
            feedback_events: vec![],
        },
    }
}

fn simple_result_with_safety(
    text: String,
    evidence_sources: Vec<String>,
    route: &str,
    request_input: &str,
    safety_warnings: Vec<String>,
) -> ProcessResult {
    let evidence_sources = unique_strings(evidence_sources);
    let predicted_text = text.clone();
    let normalized_input = input::normalize_text(request_input);
    let warnings_count = safety_warnings.len().to_string();
    ProcessResult {
        predicted_text,
        confidence: 1.0,
        used_retrieval: false,
        trace: ExplainTrace {
            intent_profile: IntentProfile::default(),
            memory_summary: String::new(),
            evidence_sources,
            layer_notes: vec![LayerNote {
                layer: 19,
                note: route.to_string(),
            }],
            debug_steps: vec![
                debug_step(
                    1,
                    "input",
                    "Input received but blocked by safety layer.",
                    [
                        ("normalized_input", normalized_input),
                        ("route", route.to_string()),
                    ],
                ),
                debug_step(
                    19,
                    "safety",
                    "Request blocked due to safety concerns.",
                    [
                        ("blocked", "true".to_string()),
                        ("warnings_count", warnings_count),
                    ],
                ),
            ],
            active_regions: vec![],
            retrieval_query: None,
            score_breakdowns: vec![],
            selected_unit: None,
            safety_warnings,
            feedback_events: vec![],
        },
    }
}

fn document_result(
    answer: &DocumentAnswer,
    route: &str,
    request_input: &str,
    intent_profile: IntentProfile,
    memory_summary: String,
) -> ProcessResult {
    let (reshaped_text, output_strategy) = reshape_output_for_intent(
        request_input,
        intent_profile.primary,
        &answer.text,
        &[],
        &answer.supporting_passages,
        &[],
    );
    let intent_primary = format!("{:?}", intent_profile.primary);
    let normalized_input = input::normalize_text(request_input);
    let intent_resolution = intent_resolution_source(&intent_profile).to_string();
    let intent_confidence = format!("{:.3}", intent_profile.confidence);
    let intent_reasons = intent_profile.reasons.join(",");
    let memory_snapshot = memory_summary.clone();
    ProcessResult {
        predicted_text: reshaped_text.clone(),
        confidence: answer.confidence,
        used_retrieval: false,
        trace: ExplainTrace {
            intent_profile,
            memory_summary,
            evidence_sources: unique_strings(answer.supporting_sources.clone()),
            layer_notes: vec![
                LayerNote {
                    layer: 17,
                    note: format!("document_passages={}", answer.supporting_passages.len()),
                },
                LayerNote {
                    layer: 20,
                    note: route.to_string(),
                },
            ],
            debug_steps: vec![
                debug_step(
                    1,
                    "input",
                    "Input normalized for document-scoped answering.",
                    [
                        ("normalized_input", normalized_input),
                        ("route", route.to_string()),
                    ],
                ),
                debug_step(
                    7,
                    "context",
                    "Document workspace and supporting passages activated.",
                    [
                        ("route", route.to_string()),
                        (
                            "supporting_sources",
                            answer.supporting_sources.len().to_string(),
                        ),
                    ],
                ),
                debug_step(
                    9,
                    "intent_resolution",
                    "Intent carried into the document answer path.",
                    [
                        ("primary", intent_primary),
                        ("confidence", intent_confidence),
                        ("resolver", intent_resolution),
                        ("reasons", intent_reasons),
                        ("route", route.to_string()),
                    ],
                ),
                debug_step(
                    9,
                    "retrieval_gate",
                    "External retrieval bypassed in favor of document-scoped evidence.",
                    [
                        ("should_retrieve", "false".to_string()),
                        ("status", "document_workspace".to_string()),
                        ("route", route.to_string()),
                    ],
                ),
                debug_step(
                    13,
                    "reasoning_merge",
                    "Evidence merge operated on the active document workspace.",
                    [
                        ("route", route.to_string()),
                        (
                            "supporting_passages",
                            answer.supporting_passages.len().to_string(),
                        ),
                    ],
                ),
                debug_step(
                    16,
                    "resolution",
                    "Document workspace selected the final answer span.",
                    [
                        ("route", route.to_string()),
                        ("confidence", format!("{:.3}", answer.confidence)),
                    ],
                ),
                debug_step(
                    17,
                    "document_answer",
                    "Answer resolved directly from loaded document context.",
                    [
                        ("route", route.to_string()),
                        ("confidence", format!("{:.3}", answer.confidence)),
                        ("strategy", output_strategy.to_string()),
                        (
                            "supporting_passages",
                            answer.supporting_passages.len().to_string(),
                        ),
                        ("predicted_text", reshaped_text),
                    ],
                ),
                debug_step(
                    20,
                    "memory",
                    "Memory snapshot recorded after document answer.",
                    [("memory_summary", memory_snapshot)],
                ),
            ],
            ..ExplainTrace::default()
        },
    }
}

fn reshape_output_for_intent(
    prompt: &str,
    intent: IntentKind,
    base_text: &str,
    evidence_documents: &[RetrievedDocument],
    supporting_passages: &[String],
    scored: &[crate::types::ScoredCandidate],
) -> (String, &'static str) {
    let fallback = finalize_output_text(base_text);
    let snippets = output_support_snippets(
        prompt,
        base_text,
        evidence_documents,
        supporting_passages,
        scored,
    );

    let shaped = match intent {
        IntentKind::Plan => format_numbered_steps("Plan", &snippets, 3),
        IntentKind::Act => format_action_output(&snippets),
        IntentKind::Recommend => format_recommendation_output(&snippets),
        IntentKind::Classify => format_classification_output(&snippets),
        IntentKind::Translate => format_translation_output(&snippets),
        IntentKind::Debug => format_debug_output(&snippets),
        IntentKind::Critique => format_critique_output(&snippets),
        IntentKind::Brainstorm => format_brainstorm_output(&snippets),
        _ => None,
    };

    match shaped {
        Some((text, strategy)) if !text.trim().is_empty() => (text, strategy),
        _ => (fallback, "default"),
    }
}

fn output_support_snippets(
    prompt: &str,
    base_text: &str,
    evidence_documents: &[RetrievedDocument],
    supporting_passages: &[String],
    scored: &[crate::types::ScoredCandidate],
) -> Vec<String> {
    let mut snippets = Vec::new();
    let normalized_prompt = input::normalize_text(prompt);

    push_output_sentences(&mut snippets, base_text, &normalized_prompt, 3);
    for passage in supporting_passages {
        push_output_sentences(&mut snippets, passage, &normalized_prompt, 2);
    }
    for document in evidence_documents.iter().take(3) {
        for sentence in simple_sentences(&document.raw_content).into_iter().take(3) {
            push_output_snippet(&mut snippets, &sentence, &normalized_prompt);
        }
    }
    for candidate in scored.iter().take(6) {
        push_output_snippet(&mut snippets, &candidate.content, &normalized_prompt);
    }

    snippets
}

fn push_output_sentences(
    target: &mut Vec<String>,
    raw: &str,
    normalized_prompt: &str,
    limit: usize,
) {
    let sentences = simple_sentences(raw);
    if sentences.is_empty() {
        push_output_snippet(target, raw, normalized_prompt);
        return;
    }

    for sentence in sentences.into_iter().take(limit) {
        push_output_snippet(target, &sentence, normalized_prompt);
    }
}

fn push_output_snippet(target: &mut Vec<String>, raw: &str, normalized_prompt: &str) {
    let cleaned = clean_output_snippet(raw);
    if cleaned.is_empty() {
        return;
    }
    let normalized = input::normalize_text(&cleaned);
    if normalized.is_empty() || normalized == normalized_prompt {
        return;
    }
    if !looks_like_semantic_output(&cleaned) {
        return;
    }
    if !target.iter().any(|existing| {
        input::normalize_text(existing) == normalized || existing.eq_ignore_ascii_case(&cleaned)
    }) {
        target.push(cleaned);
    }
}

fn clean_output_snippet(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .trim_matches(|ch: char| matches!(ch, '"' | '\''))
        .to_string()
}

fn looks_like_semantic_output(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    let alnum = trimmed.chars().filter(|ch| ch.is_alphanumeric()).count();
    let total = trimmed
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .count()
        .max(1);
    if (alnum as f32 / total as f32) < 0.45 {
        return false;
    }
    let word_count = trimmed.split_whitespace().count();
    if word_count == 0 {
        return false;
    }
    !trimmed.contains("://")
}

fn format_numbered_steps(
    heading: &str,
    snippets: &[String],
    limit: usize,
) -> Option<(String, &'static str)> {
    let steps = snippets
        .iter()
        .filter_map(|snippet| concise_sentence(snippet))
        .take(limit)
        .collect::<Vec<_>>();
    if steps.len() < 2 {
        return None;
    }

    let body = steps
        .iter()
        .enumerate()
        .map(|(index, step)| format!("{}. {}", index + 1, strip_terminal_punctuation(step)))
        .collect::<Vec<_>>()
        .join("\n");
    Some((format!("{heading}:\n{body}"), "numbered_steps"))
}

fn format_action_output(snippets: &[String]) -> Option<(String, &'static str)> {
    let first = snippets
        .first()
        .and_then(|snippet| concise_sentence(snippet))?;
    let mut lines = vec![format!(
        "Next action: {}",
        strip_terminal_punctuation(&first)
    )];
    if let Some(follow_up) = snippets
        .get(1)
        .and_then(|snippet| concise_sentence(snippet))
    {
        lines.push(format!("Then: {}", strip_terminal_punctuation(&follow_up)));
    }
    Some((lines.join("\n"), "action_guidance"))
}

fn format_recommendation_output(snippets: &[String]) -> Option<(String, &'static str)> {
    let primary = snippets
        .first()
        .and_then(|snippet| concise_sentence(snippet))?;
    let mut lines = vec![format!(
        "Recommendation: {}.",
        strip_terminal_punctuation(&primary)
    )];
    if let Some(reason) = snippets
        .get(1)
        .and_then(|snippet| concise_sentence(snippet))
    {
        lines.push(format!("Why: {}.", strip_terminal_punctuation(&reason)));
    }
    Some((lines.join("\n"), "recommendation"))
}

fn format_classification_output(snippets: &[String]) -> Option<(String, &'static str)> {
    for snippet in snippets {
        if let Some(label) = compact_label(snippet) {
            return Some((label, "classification_label"));
        }
    }
    None
}

fn format_translation_output(snippets: &[String]) -> Option<(String, &'static str)> {
    for snippet in snippets {
        let surface = strip_translation_label(snippet);
        if surface.split_whitespace().count() <= 12 && looks_like_semantic_output(&surface) {
            return Some((surface, "translation_surface"));
        }
    }
    None
}

fn format_debug_output(snippets: &[String]) -> Option<(String, &'static str)> {
    let issue = snippets
        .first()
        .and_then(|snippet| concise_sentence(snippet))?;
    let mut lines = vec![format!("Issue: {}.", strip_terminal_punctuation(&issue))];
    if let Some(cause) = snippets
        .get(1)
        .and_then(|snippet| concise_sentence(snippet))
    {
        lines.push(format!(
            "Likely cause: {}.",
            strip_terminal_punctuation(&cause)
        ));
    }
    if let Some(next_check) = snippets
        .get(2)
        .and_then(|snippet| concise_sentence(snippet))
    {
        lines.push(format!(
            "Next check: {}.",
            strip_terminal_punctuation(&next_check)
        ));
    }
    Some((lines.join("\n"), "debug_triage"))
}

fn format_critique_output(snippets: &[String]) -> Option<(String, &'static str)> {
    let concern = snippets
        .first()
        .and_then(|snippet| concise_sentence(snippet))?;
    let mut lines = vec![format!(
        "Main concern: {}.",
        strip_terminal_punctuation(&concern)
    )];
    if let Some(improvement) = snippets
        .get(1)
        .and_then(|snippet| concise_sentence(snippet))
    {
        lines.push(format!(
            "Improve by: {}.",
            strip_terminal_punctuation(&improvement)
        ));
    }
    Some((lines.join("\n"), "critique_summary"))
}

fn format_brainstorm_output(snippets: &[String]) -> Option<(String, &'static str)> {
    let ideas = snippets
        .iter()
        .filter_map(|snippet| concise_sentence(snippet))
        .filter(|idea| idea.split_whitespace().count() >= 4)
        .take(3)
        .collect::<Vec<_>>();
    if ideas.len() < 2 {
        return None;
    }

    let body = ideas
        .iter()
        .enumerate()
        .map(|(index, idea)| format!("{}. {}", index + 1, strip_terminal_punctuation(idea)))
        .collect::<Vec<_>>()
        .join("\n");
    Some((format!("Ideas:\n{body}"), "brainstorm_list"))
}

fn concise_sentence(text: &str) -> Option<String> {
    let sentence = simple_sentences(text)
        .into_iter()
        .next()
        .unwrap_or_else(|| text.trim().to_string());
    let cleaned = clean_output_snippet(&sentence);
    if cleaned.split_whitespace().count() < 2 {
        return None;
    }
    Some(cleaned)
}

fn compact_label(text: &str) -> Option<String> {
    let candidate = extract_predicate_complement(text, EvidenceFocus::Definition)
        .or_else(|| Some(strip_translation_label(text)))
        .map(|value| finalize_answer_fragment(&value))
        .filter(|value| !value.is_empty())?;
    let short = candidate
        .split(|ch| matches!(ch, ',' | ';' | ':' | '.'))
        .next()
        .unwrap_or(candidate.as_str())
        .trim()
        .to_string();
    if short.split_whitespace().count() <= 10 {
        Some(finalize_output_text(&short))
    } else {
        None
    }
}

fn strip_translation_label(text: &str) -> String {
    let trimmed = clean_output_snippet(text);
    let lowered = trimmed.to_ascii_lowercase();
    for label in ["translation:", "translated:", "answer:"] {
        if lowered.starts_with(label) {
            return trimmed[label.len()..].trim().to_string();
        }
    }
    trimmed
}

fn strip_terminal_punctuation(text: &str) -> String {
    text.trim()
        .trim_end_matches(|ch: char| matches!(ch, '.' | '!' | '?'))
        .trim()
        .to_string()
}

fn finalize_output_text(text: &str) -> String {
    let cleaned = text.trim();
    if cleaned.is_empty() {
        return "I don't have enough signal yet.".to_string();
    }
    let mut chars = cleaned.chars();
    let first = chars
        .next()
        .map(|ch| ch.to_uppercase().to_string())
        .unwrap_or_default();
    let rest = chars.collect::<String>();
    let sentence = format!("{first}{rest}");
    if sentence.ends_with('.') || sentence.ends_with('!') || sentence.ends_with('?') {
        sentence
    } else {
        format!("{sentence}.")
    }
}

fn casual_reply(intent_profile: &IntentProfile, active_titles: &[String]) -> Option<String> {
    if intent_profile.confidence < 0.4 {
        return None;
    }

    let active_note = if active_titles.is_empty() {
        None
    } else if active_titles.len() == 1 {
        Some(format!(
            "{} is active. Ask a follow-up question or use /clear.",
            active_titles[0]
        ))
    } else {
        Some(format!(
            "{} documents are active. Ask a follow-up question or use /clear.",
            active_titles.len()
        ))
    };
    match intent_profile.primary {
        IntentKind::Greeting => Some(match active_note {
            Some(note) => format!("Hi. {note}"),
            None => {
                "Hi. Ask me a question, give me a local document path, or use /train <path> to learn a file."
                    .to_string()
            }
        }),
        IntentKind::Gratitude => Some("You're welcome.".to_string()),
        IntentKind::Farewell => Some("Goodbye.".to_string()),
        IntentKind::Help => Some(match active_note {
            Some(note) => format!(
                "Give me a question about the active document session. {note} You can also load another local .docx/.pdf path or use /train <path> to persist it."
            ),
            None => "Give me a question, paste a local .docx/.pdf path in the prompt, or use /train <path> to persist a document.".to_string(),
        }),
        _ => None,
    }
}

fn looks_like_base64_blob(content: &str) -> bool {
    let compact = content
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<String>();
    compact.len() >= 32
        && compact.len() % 4 == 0
        && compact
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '+' | '/' | '='))
}

fn first_person_reference_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let references = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "i" | "me" | "my" | "mine" | "we" | "us" | "our" | "ours"
            )
        })
        .count();
    (references.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn declarative_fact_score(text: &str) -> f32 {
    if text.trim_end().ends_with('?') {
        return 0.0;
    }

    let clause = clause_marker_score(text);
    let entity_ratio = capitalized_token_ratio(text);
    let numeric_ratio = numeric_token_ratio(text);
    let shape = sentence_shape_score(text);
    ((0.42 * clause) + (0.22 * entity_ratio) + (0.18 * numeric_ratio) + (0.18 * shape))
        .clamp(0.0, 1.0)
}

fn answer_candidate_signal(text: &str) -> f32 {
    let entity_ratio = capitalized_token_ratio(text);
    let numeric_ratio = numeric_token_ratio(text);
    let punctuation = text
        .chars()
        .filter(|ch| matches!(ch, ':' | ',' | ';' | '-' | '/'))
        .count() as f32;
    ((0.45 * entity_ratio) + (0.35 * numeric_ratio) + (0.20 * (punctuation.min(3.0) / 3.0)))
        .clamp(0.0, 1.0)
}

fn clean_memory_context(text: &str) -> String {
    let trimmed = text.trim();
    let lowered = trimmed.to_lowercase();
    let cleaned = if lowered.starts_with("remember this:") {
        trimmed
            .split_once(':')
            .map(|(_, rest)| rest.trim())
            .unwrap_or(trimmed)
    } else {
        trimmed
    };
    cleaned
        .trim()
        .trim_matches('"')
        .trim_matches('\'')
        .to_string()
}

fn extract_memory_answer(prompt: &str, context: &str, focus: EvidenceFocus) -> Option<String> {
    let cleaned = clean_memory_context(context);
    if cleaned.is_empty() {
        return None;
    }

    if let Some(value) = extract_predicate_complement(&cleaned, focus) {
        return Some(value);
    }

    match focus {
        EvidenceFocus::Time => extract_temporal_phrase(&cleaned),
        EvidenceFocus::Person => {
            leading_proper_name(&cleaned).or_else(|| extract_named_suffix(&cleaned))
        }
        EvidenceFocus::Place => extract_locative_phrase(&cleaned),
        EvidenceFocus::Quantity => extract_numeric_phrase(&cleaned),
        EvidenceFocus::Definition | EvidenceFocus::Unknown => extract_named_suffix(&cleaned),
    }
    .or_else(|| {
        let prompt_terms = normalized_terms(prompt);
        let context_terms = normalized_terms(&cleaned);
        if token_overlap(&prompt_terms, &context_terms) >= 0.45 {
            Some(finalize_answer_fragment(&cleaned))
        } else {
            None
        }
    })
}

fn extract_predicate_complement(sentence: &str, focus: EvidenceFocus) -> Option<String> {
    let connectors = match focus {
        EvidenceFocus::Time => vec![
            " is on ", " was on ", " is at ", " was at ", " on ", " at ", " is ", " was ",
        ],
        EvidenceFocus::Place => vec![
            " is in ", " was in ", " is at ", " was at ", " from ", " in ", " at ", " is ", " was ",
        ],
        _ => vec![
            " is called ",
            " was called ",
            " is named ",
            " was named ",
            " is on ",
            " was on ",
            " refers to ",
            " means ",
            " is ",
            " was ",
            " are ",
            " were ",
        ],
    };

    for connector in connectors {
        if let Some((_, suffix)) = split_once_case_insensitive(sentence, connector) {
            let value = finalize_answer_fragment(suffix);
            if plausible_memory_value(&value, focus) {
                return Some(value);
            }
        }
    }

    None
}

fn split_once_case_insensitive<'a>(text: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    let lowered = text.to_lowercase();
    let index = lowered.find(needle)?;
    let suffix = text.get(index + needle.len()..)?;
    Some((&text[..index], suffix))
}

fn plausible_memory_value(value: &str, focus: EvidenceFocus) -> bool {
    if value.is_empty() {
        return false;
    }
    let word_count = value.split_whitespace().count();
    match focus {
        EvidenceFocus::Person => (1..=6).contains(&word_count),
        EvidenceFocus::Time => (1..=8).contains(&word_count),
        EvidenceFocus::Place => (1..=8).contains(&word_count),
        EvidenceFocus::Quantity => value.chars().any(|ch| ch.is_ascii_digit()),
        EvidenceFocus::Definition | EvidenceFocus::Unknown => (1..=12).contains(&word_count),
    }
}

fn finalize_answer_fragment(text: &str) -> String {
    text.trim()
        .trim_matches(|ch: char| matches!(ch, '.' | '!' | '?' | ',' | ';' | ':' | '"' | '\''))
        .trim()
        .to_string()
}

fn extract_named_suffix(sentence: &str) -> Option<String> {
    let suffix = extract_predicate_complement(sentence, EvidenceFocus::Definition)?;
    if suffix.is_empty() {
        None
    } else {
        Some(suffix)
    }
}

fn extract_temporal_phrase(sentence: &str) -> Option<String> {
    let lowered = sentence.to_lowercase();
    if lowered.contains(" on ") {
        return extract_predicate_complement(sentence, EvidenceFocus::Time);
    }
    let month_like = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .iter()
    .any(|month| lowered.contains(month));
    if month_like || sentence.chars().any(|ch| ch.is_ascii_digit()) {
        Some(finalize_answer_fragment(sentence))
    } else {
        None
    }
}

fn extract_locative_phrase(sentence: &str) -> Option<String> {
    for connector in [" in ", " at ", " from "] {
        if let Some((_, suffix)) = split_once_case_insensitive(sentence, connector) {
            let value = finalize_answer_fragment(suffix);
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

fn extract_numeric_phrase(sentence: &str) -> Option<String> {
    let value = finalize_answer_fragment(sentence);
    if value.chars().any(|ch| ch.is_ascii_digit()) {
        Some(value)
    } else {
        None
    }
}

fn grounded_evidence_answer(prompt: &str, documents: &[RetrievedDocument]) -> Option<String> {
    let focused_prompt = focus_query_text(prompt);
    let prompt_terms = normalized_terms(&focused_prompt);
    let focus = infer_evidence_focus(prompt);
    let status_requested = status_requested(prompt);
    let raw_prompt = input::normalize_text(prompt);
    let mut best: Option<(f32, String)> = None;
    let mut best_focus_match: Option<(f32, String)> = None;

    for doc in documents {
        let title_terms = normalized_terms(&doc.title);
        let title_overlap = token_overlap(&prompt_terms, &title_terms);
        
        // Exact match bonus: prioritize documents whose title exactly matches query
        let title_lower = doc.title.to_lowercase();
        let prompt_lower = prompt.to_lowercase();
        let exact_title_match = title_lower.trim() == prompt_lower.trim()
            || title_lower.starts_with(&format!("{} ", prompt_lower)) == false 
                && title_lower.contains(&prompt_lower) == false;
        
        // Penalize titles that are supersets (e.g., "Donald Trump (song)" when querying "Donald Trump")
        let is_superset_title = title_lower.contains(&prompt_lower) 
            && title_lower != prompt_lower
            && !title_lower.starts_with(&prompt_lower);
        let exact_match_bonus = if title_lower.trim() == prompt_lower.trim() {
            0.35 // Strong bonus for exact match
        } else if is_superset_title {
            -0.20 // Penalize superset titles (disambiguation pages, etc.)
        } else {
            0.0
        };
        
        let doc_body_terms = normalized_terms(&doc.normalized_content)
            .into_iter()
            .take(180)
            .collect::<Vec<_>>();
        let doc_overlap = token_overlap(&prompt_terms, &doc_body_terms);
        let ambiguity_penalty =
            evidence_ambiguity_penalty(&raw_prompt, &doc.title, &doc.raw_content);
        let doc_score = ((0.42 * title_overlap) + (0.33 * doc_overlap) + (0.20 * doc.trust_score)
            - (0.18 * ambiguity_penalty) + exact_match_bonus)
            .clamp(0.0, 1.0);

        for sentence in simple_sentences(&doc.raw_content) {
            if sentence.trim().ends_with('?') {
                continue;
            }
            let sentence_terms = normalized_terms(&sentence);
            if sentence_terms.is_empty() {
                continue;
            }
            let lexical = token_overlap(&prompt_terms, &sentence_terms);
            let focus_score = evidence_focus_score(focus, &sentence);
            let status_score = if status_requested {
                status_sentence_score(&sentence)
            } else {
                0.0
            };
            let relation_score = relation_sentence_score(&raw_prompt, &sentence);
            let generic_shape = sentence_shape_score(&sentence);
            let score = (0.28 * lexical)
                + (0.36 * focus_score)
                + (0.14 * status_score)
                + (0.12 * relation_score)
                + (0.10 * generic_shape)
                + (0.10 * doc_score);
            if focus_score >= 0.55 || status_score >= 0.55 || relation_score >= 0.65 {
                let focused_score = (0.42 * focus_score)
                    + (0.20 * status_score)
                    + (0.20 * relation_score)
                    + (0.12 * lexical)
                    + (0.12 * doc_score);
                match &best_focus_match {
                    Some((best_score, _)) if *best_score >= focused_score => {}
                    _ => best_focus_match = Some((focused_score, sentence.clone())),
                }
            }
            match &best {
                Some((best_score, _)) if *best_score >= score => {}
                _ => best = Some((score, sentence)),
            }
        }
    }

    let (_, sentence) = best_focus_match.or(best)?;
    Some(finalize_evidence_answer(&sentence, focus))
}

fn list_evidence_answer(prompt: &str, documents: &[RetrievedDocument]) -> Option<String> {
    if documents.is_empty() {
        return None;
    }

    let focused_prompt = focus_query_text(prompt);
    let subject_terms = normalized_terms(&focused_prompt);
    let mut items = Vec::new();

    for doc in documents {
        let title = normalize_list_title(&doc.title);
        if title.is_empty() {
            continue;
        }
        let title_terms = normalized_terms(&title);
        let overlap = token_overlap(&subject_terms, &title_terms);
        let title_lower = title.to_lowercase();
        let generic = matches!(
            title_lower.as_str(),
            "wikipedia" | "duckduckgo abstract" | "ferrari" | "cars" | "car" | "automobile"
        ) || title_lower.starts_with("list of ");
        if overlap < 0.1 && generic {
            continue;
        }
        if !items.iter().any(|existing| existing == &title) {
            items.push(title);
        }
        if items.len() >= 5 {
            break;
        }
    }

    if items.len() < 2 {
        return None;
    }

    let preview = join_display_items(&items[..items.len().min(4)]);
    let subject =
        subject_label_from_prompt(&focused_prompt).unwrap_or_else(|| "Relevant items".to_string());
    Some(format!("{subject} include {preview}."))
}

fn normalize_list_title(title: &str) -> String {
    title
        .split(" - ")
        .next()
        .unwrap_or(title)
        .split(" | ")
        .next()
        .unwrap_or(title)
        .trim()
        .trim_matches(|ch: char| ch == '"' || ch == '\'')
        .to_string()
}

fn subject_label_from_prompt(prompt: &str) -> Option<String> {
    if let Some((items, owner)) = prompt.split_once(" by ") {
        let item_label = cleaned_subject_terms(items);
        let owner_label = cleaned_subject_terms(owner);
        if !item_label.is_empty() && !owner_label.is_empty() {
            return Some(format!(
                "{} {}",
                to_title_case(&owner_label),
                to_title_case(&item_label)
            ));
        }
    }

    let cleaned = cleaned_subject_terms(prompt);
    if cleaned.is_empty() {
        None
    } else {
        Some(to_title_case(&cleaned))
    }
}

fn cleaned_subject_terms(text: &str) -> String {
    text.split_whitespace()
        .filter(|term| {
            !matches!(
                *term,
                "list"
                    | "all"
                    | "show"
                    | "me"
                    | "give"
                    | "tell"
                    | "about"
                    | "the"
                    | "a"
                    | "an"
                    | "of"
                    | "by"
            )
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn join_display_items(items: &[String]) -> String {
    match items {
        [] => String::new(),
        [only] => only.clone(),
        [first, second] => format!("{first} and {second}"),
        _ => {
            let mut parts = items.to_vec();
            let last = parts.pop().unwrap_or_default();
            format!("{}, and {}", parts.join(", "), last)
        }
    }
}

fn to_title_case(text: &str) -> String {
    text.split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
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

fn token_overlap(lhs: &[String], rhs: &[String]) -> f32 {
    if lhs.is_empty() || rhs.is_empty() {
        return 0.0;
    }

    let mut intersection = 0usize;
    for token in lhs {
        if rhs.contains(token) {
            intersection += 1;
        }
    }

    intersection as f32 / lhs.len().max(rhs.len()) as f32
}

fn simple_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.split_whitespace().collect::<Vec<_>>().join(" ");
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trailing = current.split_whitespace().collect::<Vec<_>>().join(" ");
    if !trailing.is_empty() {
        sentences.push(trailing);
    }
    sentences
}

fn normalized_terms(text: &str) -> Vec<String> {
    input::normalize_text(text)
        .split_whitespace()
        .map(|term| {
            term.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-')
                .to_lowercase()
        })
        .filter(|term| term.len() > 2)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "the"
                    | "and"
                    | "for"
                    | "with"
                    | "this"
                    | "that"
                    | "from"
                    | "about"
                    | "what"
                    | "who"
                    | "when"
                    | "where"
                    | "which"
            )
        })
        .map(|term| singularize_term(&term))
        .collect()
}

fn intent_labels_from_contexts(contexts: &[String]) -> Vec<IntentKind> {
    let mut labels = Vec::new();
    for context in contexts {
        for segment in context.split('|') {
            let trimmed = segment.trim();
            let Some(label) = trimmed.strip_prefix("intent_label:") else {
                continue;
            };
            if let Some(kind) = intent_kind_from_label(label) {
                if !labels.contains(&kind) {
                    labels.push(kind);
                }
            }
        }
    }
    labels
}

fn intent_kind_from_label(label: &str) -> Option<IntentKind> {
    match label.trim().to_ascii_lowercase().as_str() {
        "greeting" => Some(IntentKind::Greeting),
        "gratitude" => Some(IntentKind::Gratitude),
        "farewell" => Some(IntentKind::Farewell),
        "help" => Some(IntentKind::Help),
        "clarify" => Some(IntentKind::Clarify),
        "rewrite" => Some(IntentKind::Rewrite),
        "verify" => Some(IntentKind::Verify),
        "continue" => Some(IntentKind::Continue),
        "forget" => Some(IntentKind::Forget),
        "question" => Some(IntentKind::Question),
        "summarize" => Some(IntentKind::Summarize),
        "explain" => Some(IntentKind::Explain),
        "compare" => Some(IntentKind::Compare),
        "extract" => Some(IntentKind::Extract),
        "analyze" => Some(IntentKind::Analyze),
        "plan" => Some(IntentKind::Plan),
        "act" => Some(IntentKind::Act),
        "recommend" => Some(IntentKind::Recommend),
        "classify" => Some(IntentKind::Classify),
        "translate" => Some(IntentKind::Translate),
        "debug" => Some(IntentKind::Debug),
        "critique" => Some(IntentKind::Critique),
        "brainstorm" => Some(IntentKind::Brainstorm),
        "unknown" => Some(IntentKind::Unknown),
        _ => None,
    }
}

fn profile_confidence(
    top: f32,
    second: f32,
    floor_threshold: f32,
    ambiguity_margin: f32,
    certainty_bias: f32,
    certainty_softener_weight: f32,
) -> f32 {
    let top_gap = (top - second).max(0.0);
    let floor_score = (top / floor_threshold.max(0.01)).clamp(0.0, 1.0);
    let gap_score = (top_gap / ambiguity_margin.max(0.01)).clamp(0.0, 1.0);
    (0.55 * floor_score + 0.45 * gap_score
        - certainty_bias.min(0.0).abs() * certainty_softener_weight)
        .clamp(0.0, 1.0)
}

fn singularize_term(term: &str) -> String {
    if term.len() > 4 && term.ends_with('s') && !term.ends_with("ss") {
        term[..term.len() - 1].to_string()
    } else {
        term.to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvidenceFocus {
    Person,
    Time,
    Place,
    Quantity,
    Definition,
    Unknown,
}

fn infer_evidence_focus(prompt: &str) -> EvidenceFocus {
    let lowered = input::normalize_text(prompt).to_lowercase();
    let first = lowered.split_whitespace().next().unwrap_or_default();
    match first {
        "who" => EvidenceFocus::Person,
        "when" => EvidenceFocus::Time,
        "where" => EvidenceFocus::Place,
        "what" => EvidenceFocus::Definition,
        "which" => EvidenceFocus::Definition,
        "explain" | "describe" => EvidenceFocus::Definition,
        "how" if lowered.contains("how many") || lowered.contains("how much") => {
            EvidenceFocus::Quantity
        }
        _ if lowered.starts_with("tell me about ") => EvidenceFocus::Definition,
        _ if role_entity_query(&lowered) => EvidenceFocus::Person,
        _ => EvidenceFocus::Unknown,
    }
}

fn role_entity_query(text: &str) -> bool {
    let has_role = text.split_whitespace().any(|token| {
        matches!(
            token,
            "president"
                | "prime"
                | "minister"
                | "ceo"
                | "cfo"
                | "coo"
                | "founder"
                | "owner"
                | "head"
                | "leader"
                | "governor"
                | "mayor"
                | "director"
                | "chair"
                | "chairman"
                | "chairperson"
                | "commissioner"
                | "secretary"
                | "principal"
                | "dean"
                | "captain"
                | "coach"
        )
    });
    has_role
        && (status_requested(text)
            || text.contains(" of ")
            || text.contains(" for ")
            || text.contains(" at "))
}

fn relation_sentence_score(prompt: &str, sentence: &str) -> f32 {
    let prompt_terms = normalized_terms(&focus_query_text(prompt));
    let sentence_terms = normalized_terms(sentence);
    if prompt_terms.is_empty() || sentence_terms.is_empty() {
        return 0.0;
    }

    let coverage = token_overlap(&prompt_terms, &sentence_terms);
    let cohesion = matched_term_cohesion_score(&prompt_terms, sentence);
    let clause = clause_marker_score(sentence);
    let value_signal = answer_candidate_signal(sentence);

    ((0.40 * coverage) + (0.28 * cohesion) + (0.20 * clause) + (0.12 * value_signal))
        .clamp(0.0, 1.0)
}

fn evidence_ambiguity_penalty(prompt: &str, title: &str, content: &str) -> f32 {
    let prompt_terms = normalized_terms(prompt);
    if prompt_terms.is_empty() {
        return 0.0;
    }

    let title_terms = normalized_terms(title);
    let content_terms = normalized_terms(content)
        .into_iter()
        .take(80)
        .collect::<Vec<_>>();
    let best_overlap = token_overlap(&prompt_terms, &title_terms)
        .max(token_overlap(&prompt_terms, &content_terms));
    let qualifier_penalty = if title.contains('(') { 0.25 } else { 0.0 };
    let divergence_penalty = (1.0 - best_overlap).clamp(0.0, 1.0) * 0.55;
    (qualifier_penalty + divergence_penalty).clamp(0.0, 1.0)
}

fn evidence_focus_score(focus: EvidenceFocus, sentence: &str) -> f32 {
    match focus {
        EvidenceFocus::Person => person_sentence_score(sentence),
        EvidenceFocus::Time => time_sentence_score(sentence),
        EvidenceFocus::Place => place_sentence_score(sentence),
        EvidenceFocus::Quantity => quantity_sentence_score(sentence),
        EvidenceFocus::Definition => definition_sentence_score(sentence),
        EvidenceFocus::Unknown => 0.0,
    }
}

fn person_sentence_score(sentence: &str) -> f32 {
    let longest_run = longest_capitalized_run(sentence);
    let clause = clause_marker_score(sentence);
    let entity_ratio = capitalized_token_ratio(sentence);
    match longest_run {
        0 => 0.0,
        1 => ((0.24_f32) + (0.16 * clause) + (0.20 * entity_ratio)).clamp(0.0, 1.0),
        _ => ((0.62_f32) + (0.16 * clause) + (0.22 * entity_ratio)).clamp(0.0, 1.0),
    }
}

fn time_sentence_score(sentence: &str) -> f32 {
    ((0.68 * temporal_expression_score(sentence))
        + (0.16 * numeric_token_ratio(sentence))
        + (0.16 * clause_marker_score(sentence)))
    .clamp(0.0, 1.0)
}

fn place_sentence_score(sentence: &str) -> f32 {
    let locative = locative_signal_score(sentence);
    let entity_ratio = capitalized_token_ratio(sentence);
    let clause = clause_marker_score(sentence);
    ((0.44 * locative) + (0.36 * entity_ratio) + (0.20 * clause)).clamp(0.0, 1.0)
}

fn quantity_sentence_score(sentence: &str) -> f32 {
    let numeric = numeric_token_ratio(sentence);
    let symbol_signal = sentence
        .chars()
        .filter(|ch| matches!(ch, '%' | '$' | ',' | '.'))
        .count() as f32;
    ((0.74 * numeric)
        + (0.18 * (symbol_signal.min(4.0) / 4.0))
        + (0.08 * clause_marker_score(sentence)))
    .clamp(0.0, 1.0)
}

fn definition_sentence_score(sentence: &str) -> f32 {
    let clause = clause_marker_score(sentence);
    let shape = sentence_shape_score(sentence);
    let temporal_penalty = temporal_expression_score(sentence) * 0.18;
    let numeric_penalty = numeric_token_ratio(sentence) * 0.14;
    ((0.58 * clause) + (0.32 * shape) - temporal_penalty - numeric_penalty).clamp(0.0, 1.0)
}

fn status_requested(text: &str) -> bool {
    let lowered = text.to_lowercase();
    [
        "current",
        "currently",
        "latest",
        "recent",
        "recently",
        "incumbent",
        "now",
        "today",
        "present",
    ]
    .iter()
    .any(|cue| lowered.contains(cue))
}

fn status_sentence_score(sentence: &str) -> f32 {
    let continuity = continuity_signal_score(sentence);
    let present = present_tense_score(sentence);
    let temporal = temporal_expression_score(sentence);
    let past = past_tense_score(sentence);
    let termination = termination_signal_score(sentence);
    ((0.32 * continuity)
        + (0.28 * present)
        + (0.22 * temporal)
        + (0.10 * clause_marker_score(sentence))
        - (0.14 * past)
        - (0.18 * termination))
        .clamp(0.0, 1.0)
}

fn sentence_shape_score(sentence: &str) -> f32 {
    let length = sentence.trim().len();
    if (24..=180).contains(&length) {
        1.0
    } else if (12..=240).contains(&length) {
        0.72
    } else {
        0.3
    }
}

fn lexical_tokens(text: &str) -> Vec<String> {
    input::normalize_text(text)
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
                .to_ascii_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect()
}

fn clause_marker_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let auxiliaries = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "is" | "are"
                    | "was"
                    | "were"
                    | "be"
                    | "been"
                    | "being"
                    | "has"
                    | "have"
                    | "had"
                    | "do"
                    | "does"
                    | "did"
            )
        })
        .count();
    let prepositions = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "in" | "on"
                    | "at"
                    | "from"
                    | "to"
                    | "for"
                    | "by"
                    | "of"
                    | "with"
                    | "about"
                    | "since"
                    | "during"
            )
        })
        .count();
    let separators = text
        .chars()
        .filter(|ch| matches!(ch, ',' | ';' | ':' | '-' | '(' | ')'))
        .count();

    ((0.48 * (auxiliaries.min(3) as f32 / 3.0))
        + (0.34 * (prepositions.min(3) as f32 / 3.0))
        + (0.18 * (separators.min(3) as f32 / 3.0)))
        .clamp(0.0, 1.0)
}

fn capitalized_token_ratio(text: &str) -> f32 {
    let tokens = text
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'))
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    if tokens.is_empty() {
        return 0.0;
    }

    let capitalized = tokens
        .iter()
        .filter(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_uppercase())
                .unwrap_or(false)
                && token.chars().skip(1).any(|ch| ch.is_lowercase())
        })
        .count();
    (capitalized as f32 / tokens.len() as f32).clamp(0.0, 1.0)
}

fn numeric_token_ratio(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let numeric = tokens
        .iter()
        .filter(|token| token.chars().any(|ch| ch.is_ascii_digit()))
        .count();
    (numeric as f32 / tokens.len() as f32).clamp(0.0, 1.0)
}

fn longest_capitalized_run(text: &str) -> usize {
    let tokens = text
        .split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-'))
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut longest = 0usize;
    let mut current = 0usize;

    for token in tokens {
        let lowered = token.to_ascii_lowercase();
        let is_name_like = token
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && token.chars().skip(1).any(|ch| ch.is_lowercase())
            && !matches!(lowered.as_str(), "the" | "a" | "an");
        if is_name_like {
            current += 1;
            longest = longest.max(current);
        } else {
            current = 0;
        }
    }

    longest
}

fn temporal_expression_score(text: &str) -> f32 {
    let lowered = text.to_ascii_lowercase();
    let month_like = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    .iter()
    .filter(|month| lowered.contains(**month))
    .count();
    let temporal_tokens = lexical_tokens(text)
        .into_iter()
        .filter(|token| {
            token.chars().any(|ch| ch.is_ascii_digit())
                || matches!(
                    token.as_str(),
                    "today"
                        | "now"
                        | "current"
                        | "currently"
                        | "present"
                        | "recent"
                        | "since"
                        | "during"
                        | "after"
                        | "before"
                )
        })
        .count();
    ((0.52 * (month_like.min(2) as f32 / 2.0)) + (0.48 * (temporal_tokens.min(4) as f32 / 4.0)))
        .clamp(0.0, 1.0)
}

fn locative_signal_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let locatives = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "in" | "at" | "from" | "near" | "inside" | "outside" | "within" | "across"
            )
        })
        .count();
    (locatives.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn present_tense_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let present = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "is" | "are"
                    | "has"
                    | "have"
                    | "does"
                    | "do"
                    | "remains"
                    | "continues"
                    | "holds"
                    | "leads"
                    | "serves"
            )
        })
        .count();
    (present.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn past_tense_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    if tokens.is_empty() {
        return 0.0;
    }

    let past = tokens
        .iter()
        .filter(|token| {
            matches!(token.as_str(), "was" | "were" | "had" | "did") || token.ends_with("ed")
        })
        .count();
    (past.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn continuity_signal_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    let matches = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "current"
                    | "currently"
                    | "now"
                    | "today"
                    | "present"
                    | "ongoing"
                    | "active"
                    | "since"
                    | "still"
                    | "remain"
                    | "remains"
                    | "continues"
            )
        })
        .count();
    (matches.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn termination_signal_score(text: &str) -> f32 {
    let tokens = lexical_tokens(text);
    let matches = tokens
        .iter()
        .filter(|token| {
            matches!(
                token.as_str(),
                "former"
                    | "previous"
                    | "formerly"
                    | "retired"
                    | "ended"
                    | "ceased"
                    | "defunct"
                    | "past"
            )
        })
        .count();
    (matches.min(3) as f32 / 3.0).clamp(0.0, 1.0)
}

fn matched_term_cohesion_score(query_terms: &[String], text: &str) -> f32 {
    if query_terms.is_empty() {
        return 0.0;
    }

    let tokens = lexical_tokens(text);
    let mut positions = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if query_terms.contains(token) {
            positions.push(index);
        }
    }
    if positions.is_empty() {
        return 0.0;
    }

    let coverage = positions.len() as f32 / query_terms.len() as f32;
    if positions.len() == 1 {
        return (0.55 * coverage).clamp(0.0, 1.0);
    }

    let span = positions.last().copied().unwrap_or(0) - positions[0] + 1;
    let compactness = positions.len() as f32 / span.max(1) as f32;
    ((0.55 * coverage) + (0.45 * compactness)).clamp(0.0, 1.0)
}

fn finalize_evidence_answer(sentence: &str, focus: EvidenceFocus) -> String {
    let trimmed = sentence.trim();
    if trimmed.is_empty() {
        return "I don't have enough signal yet.".to_string();
    }

    if matches!(focus, EvidenceFocus::Person) {
        if let Some(name) = leading_proper_name(trimmed) {
            return format!("{name}.");
        }
    }

    let mut chars = trimmed.chars();
    let first = chars
        .next()
        .map(|ch| ch.to_uppercase().to_string())
        .unwrap_or_default();
    let rest = chars.collect::<String>();
    let normalized = format!("{first}{rest}");
    if normalized.ends_with('.') || normalized.ends_with('!') || normalized.ends_with('?') {
        normalized
    } else {
        format!("{normalized}.")
    }
}

fn leading_proper_name(sentence: &str) -> Option<String> {
    let mut name_parts = Vec::new();
    for token in sentence.split_whitespace() {
        let cleaned = token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-');
        if cleaned.is_empty() {
            continue;
        }
        let lowered = cleaned.to_lowercase();
        let name_like = cleaned
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && cleaned.chars().skip(1).any(|ch| ch.is_lowercase())
            && !matches!(lowered.as_str(), "the" | "a" | "an");
        if name_like {
            name_parts.push(cleaned.to_string());
            continue;
        }
        if !name_parts.is_empty() {
            break;
        }
    }
    if name_parts.len() >= 2 {
        Some(name_parts.join(" "))
    } else {
        None
    }
}

fn evaluate_expression_input(text: &str) -> Option<String> {
    let normalized = normalize_expression_input(text)?;
    let value = evaluate_expression(&normalized)?;
    Some(render_numeric_result(value))
}

fn normalize_expression_input(text: &str) -> Option<String> {
    let trimmed = text.trim().trim_end_matches('?').trim();
    if trimmed.is_empty() {
        return None;
    }

    let lowered = trimmed.to_lowercase();
    let candidate = lowered
        .strip_prefix("what is ")
        .or_else(|| lowered.strip_prefix("calculate "))
        .or_else(|| lowered.strip_prefix("compute "))
        .or_else(|| lowered.strip_prefix("evaluate "))
        .unwrap_or(&lowered)
        .trim();

    if candidate.is_empty() {
        return None;
    }

    if !candidate.chars().all(|ch| {
        ch.is_ascii_digit()
            || matches!(
                ch,
                ' ' | '\t' | '.' | '+' | '-' | '*' | '/' | '%' | '(' | ')'
            )
    }) {
        return None;
    }
    if !candidate
        .chars()
        .any(|ch| matches!(ch, '+' | '-' | '*' | '/' | '%'))
    {
        return None;
    }

    Some(candidate.to_string())
}

fn evaluate_expression(expression: &str) -> Option<f64> {
    let mut parser = ExpressionParser::new(expression);
    let value = parser.parse_expression()?;
    parser.skip_whitespace();
    if parser.is_exhausted() {
        Some(value)
    } else {
        None
    }
}

fn render_numeric_result(value: f64) -> String {
    if (value.fract()).abs() < 1e-9 {
        format!("{}", value.round() as i64)
    } else {
        let rendered = format!("{value:.6}");
        rendered
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

struct ExpressionParser<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> ExpressionParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            position: 0,
        }
    }

    fn parse_expression(&mut self) -> Option<f64> {
        let mut value = self.parse_term()?;
        loop {
            self.skip_whitespace();
            match self.peek_char() {
                Some('+') => {
                    self.position += 1;
                    value += self.parse_term()?;
                }
                Some('-') => {
                    self.position += 1;
                    value -= self.parse_term()?;
                }
                _ => break,
            }
        }
        Some(value)
    }

    fn parse_term(&mut self) -> Option<f64> {
        let mut value = self.parse_factor()?;
        loop {
            self.skip_whitespace();
            match self.peek_char() {
                Some('+') | Some('-') => {
                    break;
                }
                Some('*') => {
                    self.position += 1;
                    value *= self.parse_factor()?;
                }
                Some('/') => {
                    self.position += 1;
                    let divisor = self.parse_factor()?;
                    if divisor.abs() < f64::EPSILON {
                        return None;
                    }
                    value /= divisor;
                }
                Some('%') => {
                    self.position += 1;
                    let divisor = self.parse_factor()?;
                    if divisor.abs() < f64::EPSILON {
                        return None;
                    }
                    value %= divisor;
                }
                _ => break,
            }
        }
        Some(value)
    }

    fn parse_factor(&mut self) -> Option<f64> {
        self.skip_whitespace();
        match self.peek_char()? {
            '+' => {
                self.position += 1;
                self.parse_factor()
            }
            '-' => {
                self.position += 1;
                self.parse_factor().map(|value| -value)
            }
            '(' => {
                self.position += 1;
                let value = self.parse_expression()?;
                self.skip_whitespace();
                if self.peek_char()? != ')' {
                    return None;
                }
                self.position += 1;
                Some(value)
            }
            _ => self.parse_number(),
        }
    }

    fn parse_number(&mut self) -> Option<f64> {
        self.skip_whitespace();
        let start = self.position;
        let mut seen_digit = false;
        let mut seen_dot = false;
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_digit() {
                seen_digit = true;
                self.position += 1;
            } else if ch == '.' && !seen_dot {
                seen_dot = true;
                self.position += 1;
            } else {
                break;
            }
        }
        if !seen_digit {
            return None;
        }
        std::str::from_utf8(&self.input[start..self.position])
            .ok()
            .and_then(|slice| slice.parse::<f64>().ok())
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_ascii_whitespace() {
                self.position += 1;
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input.get(self.position).map(|byte| *byte as char)
    }

    fn is_exhausted(&self) -> bool {
        self.position >= self.input.len()
    }
}

fn unique_strings(values: Vec<String>) -> Vec<String> {
    let mut unique = Vec::new();
    for value in values {
        if !unique.contains(&value) {
            unique.push(value);
        }
    }
    unique
}

fn route_units(
    active_units: &[Unit],
    all_units: &[Unit],
    semantic_map: &crate::config::SemanticMapConfig,
) -> RoutingResult {
    let mut router = SemanticRouter::new(semantic_map);
    router.route(active_units, all_units)
}

fn route_candidate_units(
    routing: &RoutingResult,
    all_units: &[Unit],
    max_candidates: usize,
    semantic_map: &crate::config::SemanticMapConfig,
    escape: &EscapeProfile,
) -> crate::types::CandidateRoute {
    SemanticRouter::new(semantic_map).route_candidates(routing, all_units, max_candidates, escape)
}

fn apply_adaptive_trust_profile(config: &mut TrustConfig, profile: &AdaptiveTrustProfile) {
    config.default_source_trust = profile.default_source_trust;
    config.min_corroborating_sources = profile.min_corroborating_sources;
    config.require_https = profile.require_https;
}

#[derive(Clone, Copy, Default)]
struct BehaviorProfileBlend {
    factual: f32,
    explanatory: f32,
    procedural: f32,
    creative: f32,
    advisory: f32,
    casual: f32,
}

#[derive(Clone)]
struct TrustSignalDecision {
    harden: bool,
    reason: String,
}

fn dynamic_intent_profile_blend(intent_profile: &IntentProfile) -> BehaviorProfileBlend {
    let score_for = |intent: IntentKind| -> f32 {
        intent_profile
            .scores
            .iter()
            .find(|score| score.intent == intent)
            .map(|score| score.score.max(0.0))
            .unwrap_or(0.0)
    };

    let mut factual = score_for(IntentKind::Question)
        + score_for(IntentKind::Verify)
        + score_for(IntentKind::Extract)
        + score_for(IntentKind::Classify);
    let mut explanatory = score_for(IntentKind::Explain)
        + score_for(IntentKind::Summarize)
        + score_for(IntentKind::Analyze)
        + score_for(IntentKind::Compare)
        + score_for(IntentKind::Critique);
    let mut procedural =
        score_for(IntentKind::Plan) + score_for(IntentKind::Act) + score_for(IntentKind::Debug);
    let mut creative = score_for(IntentKind::Brainstorm)
        + score_for(IntentKind::Rewrite)
        + score_for(IntentKind::Translate);
    let mut advisory = score_for(IntentKind::Recommend);
    let mut casual = score_for(IntentKind::Greeting)
        + score_for(IntentKind::Gratitude)
        + score_for(IntentKind::Farewell)
        + score_for(IntentKind::Help)
        + score_for(IntentKind::Clarify)
        + score_for(IntentKind::Continue)
        + score_for(IntentKind::Forget)
        + score_for(IntentKind::Unknown);

    match intent_profile.primary {
        IntentKind::Question | IntentKind::Verify | IntentKind::Extract | IntentKind::Classify => {
            factual += 0.20
        }
        IntentKind::Explain
        | IntentKind::Summarize
        | IntentKind::Analyze
        | IntentKind::Compare
        | IntentKind::Critique => explanatory += 0.20,
        IntentKind::Plan | IntentKind::Act | IntentKind::Debug => procedural += 0.20,
        IntentKind::Rewrite | IntentKind::Translate | IntentKind::Brainstorm => creative += 0.20,
        IntentKind::Recommend => advisory += 0.20,
        _ => casual += 0.20,
    }

    if intent_profile.ambiguous {
        casual += 0.12;
    }
    if intent_profile.wants_brief {
        factual += 0.04;
    }
    if intent_profile.references_document_context {
        explanatory += 0.05;
        procedural += 0.03;
    }
    if intent_profile.certainty_bias < 0.0 {
        advisory += 0.03;
    }

    let total = factual + explanatory + procedural + creative + advisory + casual;
    if total <= f32::EPSILON {
        return BehaviorProfileBlend {
            factual: 0.22,
            explanatory: 0.18,
            procedural: 0.16,
            creative: 0.14,
            advisory: 0.10,
            casual: 0.20,
        };
    }

    BehaviorProfileBlend {
        factual: factual / total,
        explanatory: explanatory / total,
        procedural: procedural / total,
        creative: creative / total,
        advisory: advisory / total,
        casual: casual / total,
    }
}

fn dominant_behavior_profile_name(blend: &BehaviorProfileBlend) -> Option<&'static str> {
    [
        ("factual", blend.factual),
        ("explanatory", blend.explanatory),
        ("procedural", blend.procedural),
        ("creative", blend.creative),
        ("advisory", blend.advisory),
        ("casual", blend.casual),
    ]
    .into_iter()
    .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
    .map(|(name, _)| name)
}

fn blend_scoring_weights(config: &EngineConfig, blend: &BehaviorProfileBlend) -> ScoringWeights {
    let factual = config
        .adaptive_behavior
        .intent_profile("factual")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());
    let explanatory = config
        .adaptive_behavior
        .intent_profile("explanatory")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());
    let procedural = config
        .adaptive_behavior
        .intent_profile("procedural")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());
    let creative = config
        .adaptive_behavior
        .intent_profile("creative")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());
    let advisory = config
        .adaptive_behavior
        .intent_profile("advisory")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());
    let casual = config
        .adaptive_behavior
        .intent_profile("casual")
        .map(|profile| profile.scoring.clone())
        .unwrap_or_else(|| config.scoring.clone());

    ScoringWeights {
        spatial: weighted_sum(
            blend,
            factual.spatial,
            explanatory.spatial,
            procedural.spatial,
            creative.spatial,
            advisory.spatial,
            casual.spatial,
            config.scoring.spatial,
        ),
        context: weighted_sum(
            blend,
            factual.context,
            explanatory.context,
            procedural.context,
            creative.context,
            advisory.context,
            casual.context,
            config.scoring.context,
        ),
        sequence: weighted_sum(
            blend,
            factual.sequence,
            explanatory.sequence,
            procedural.sequence,
            creative.sequence,
            advisory.sequence,
            casual.sequence,
            config.scoring.sequence,
        ),
        transition: weighted_sum(
            blend,
            factual.transition,
            explanatory.transition,
            procedural.transition,
            creative.transition,
            advisory.transition,
            casual.transition,
            config.scoring.transition,
        ),
        utility: weighted_sum(
            blend,
            factual.utility,
            explanatory.utility,
            procedural.utility,
            creative.utility,
            advisory.utility,
            casual.utility,
            config.scoring.utility,
        ),
        confidence: weighted_sum(
            blend,
            factual.confidence,
            explanatory.confidence,
            procedural.confidence,
            creative.confidence,
            advisory.confidence,
            casual.confidence,
            config.scoring.confidence,
        ),
        evidence: weighted_sum(
            blend,
            factual.evidence,
            explanatory.evidence,
            procedural.evidence,
            creative.evidence,
            advisory.evidence,
            casual.evidence,
            config.scoring.evidence,
        ),
    }
}

fn blend_shaping_config(config: &EngineConfig, blend: &BehaviorProfileBlend) -> IntentShapingConfig {
    let factual = config
        .adaptive_behavior
        .intent_profile("factual")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();
    let explanatory = config
        .adaptive_behavior
        .intent_profile("explanatory")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();
    let procedural = config
        .adaptive_behavior
        .intent_profile("procedural")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();
    let creative = config
        .adaptive_behavior
        .intent_profile("creative")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();
    let advisory = config
        .adaptive_behavior
        .intent_profile("advisory")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();
    let casual = config
        .adaptive_behavior
        .intent_profile("casual")
        .map(|profile| profile.shaping.clone())
        .unwrap_or_default();

    // For shaping, use dominant profile's settings rather than blending
    // since boolean flags like allow_semantic_drift shouldn't be averaged
    if let Some(name) = dominant_behavior_profile_name(blend) {
        if let Some(profile) = config.adaptive_behavior.intent_profile(name) {
            return profile.shaping.clone();
        }
    }

    // Fallback: if any profile has allow_semantic_drift=true, use creative settings
    if blend.creative > 0.3 || blend.casual > 0.5 {
        return creative;
    }

    // Default to factual (conservative) shaping
    factual
}

fn blend_escape_profile(config: &EngineConfig, blend: &BehaviorProfileBlend) -> EscapeProfile {
    let factual = config
        .adaptive_behavior
        .intent_profile("factual")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| EscapeProfile {
            stochastic_jump_prob: 0.0,
            beam_width: config.governance.max_candidate_pool.max(1),
        });
    let explanatory = config
        .adaptive_behavior
        .intent_profile("explanatory")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| factual.clone());
    let procedural = config
        .adaptive_behavior
        .intent_profile("procedural")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| factual.clone());
    let creative = config
        .adaptive_behavior
        .intent_profile("creative")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| factual.clone());
    let advisory = config
        .adaptive_behavior
        .intent_profile("advisory")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| factual.clone());
    let casual = config
        .adaptive_behavior
        .intent_profile("casual")
        .map(|profile| profile.escape.clone())
        .unwrap_or_else(|| factual.clone());

    EscapeProfile {
        stochastic_jump_prob: weighted_sum(
            blend,
            factual.stochastic_jump_prob,
            explanatory.stochastic_jump_prob,
            procedural.stochastic_jump_prob,
            creative.stochastic_jump_prob,
            advisory.stochastic_jump_prob,
            casual.stochastic_jump_prob,
            0.0,
        )
        .clamp(0.0, 1.0),
        beam_width: weighted_sum(
            blend,
            factual.beam_width as f32,
            explanatory.beam_width as f32,
            procedural.beam_width as f32,
            creative.beam_width as f32,
            advisory.beam_width as f32,
            casual.beam_width as f32,
            config.governance.max_candidate_pool.max(1) as f32,
        )
        .round()
        .clamp(1.0, config.governance.max_candidate_pool.max(1) as f32)
            as usize,
    }
}

fn blend_resolver_profile(
    config: &EngineConfig,
    blend: &BehaviorProfileBlend,
) -> (FineResolverConfig, Option<ResolverMode>) {
    let factual = config
        .adaptive_behavior
        .intent_profile("factual")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();
    let explanatory = config
        .adaptive_behavior
        .intent_profile("explanatory")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();
    let procedural = config
        .adaptive_behavior
        .intent_profile("procedural")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();
    let creative = config
        .adaptive_behavior
        .intent_profile("creative")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();
    let advisory = config
        .adaptive_behavior
        .intent_profile("advisory")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();
    let casual = config
        .adaptive_behavior
        .intent_profile("casual")
        .map(|profile| profile.resolver.clone())
        .unwrap_or_default();

    let resolver = FineResolverConfig {
        selection_temperature: weighted_sum(
            blend,
            factual.selection_temperature,
            explanatory.selection_temperature,
            procedural.selection_temperature,
            creative.selection_temperature,
            advisory.selection_temperature,
            casual.selection_temperature,
            config.resolver.selection_temperature,
        )
        .clamp(0.0, 2.0),
        min_confidence_floor: weighted_sum(
            blend,
            factual.min_confidence_floor,
            explanatory.min_confidence_floor,
            procedural.min_confidence_floor,
            creative.min_confidence_floor,
            advisory.min_confidence_floor,
            casual.min_confidence_floor,
            config.resolver.min_confidence_floor,
        )
        .clamp(0.0, 1.0),
        evidence_answer_confidence_threshold: config.resolver.evidence_answer_confidence_threshold,
        creative_drift_tolerance: config.resolver.creative_drift_tolerance,
        factual_corruption_threshold: config.resolver.factual_corruption_threshold,
    };

    let mode = dominant_behavior_profile_name(blend).and_then(|name| match name {
        "factual" => factual.mode,
        "explanatory" => explanatory.mode,
        "procedural" => procedural.mode,
        "creative" => creative.mode,
        "advisory" => advisory.mode,
        "casual" => casual.mode,
        _ => None,
    });

    (resolver, mode)
}

fn dynamic_trust_signal(intent_profile: &IntentProfile) -> TrustSignalDecision {
    let verify_score = intent_profile
        .scores
        .iter()
        .find(|score| score.intent == IntentKind::Verify)
        .map(|score| score.score)
        .unwrap_or(0.0);
    let factual_score = intent_profile
        .scores
        .iter()
        .filter(|score| {
            matches!(
                score.intent,
                IntentKind::Question | IntentKind::Verify | IntentKind::Extract
            )
        })
        .map(|score| score.score)
        .sum::<f32>();
    let freshness_sensitive = intent_profile
        .reasons
        .iter()
        .any(|reason| reason.starts_with("temporal_cues=") || reason == "context_marks_freshness");
    let low_confidence = intent_profile.confidence < 0.55
        || intent_profile.ambiguous
        || intent_profile.certainty_bias < 0.0;
    let harden = matches!(intent_profile.primary, IntentKind::Verify)
        || verify_score >= 0.35
        || (freshness_sensitive && factual_score >= 0.45)
        || (low_confidence && factual_score >= 0.60);

    let reason = if matches!(intent_profile.primary, IntentKind::Verify) {
        "verification_request".to_string()
    } else if verify_score >= 0.35 {
        "verification_pressure".to_string()
    } else if freshness_sensitive && factual_score >= 0.45 {
        "freshness_sensitive".to_string()
    } else if low_confidence && factual_score >= 0.60 {
        "uncertain_factual_request".to_string()
    } else {
        "default".to_string()
    };

    TrustSignalDecision { harden, reason }
}

fn weighted_sum(
    blend: &BehaviorProfileBlend,
    factual: f32,
    explanatory: f32,
    procedural: f32,
    creative: f32,
    advisory: f32,
    casual: f32,
    fallback: f32,
) -> f32 {
    let total = blend.factual
        + blend.explanatory
        + blend.procedural
        + blend.creative
        + blend.advisory
        + blend.casual;
    if total <= f32::EPSILON {
        return fallback;
    }
    (blend.factual * factual)
        + (blend.explanatory * explanatory)
        + (blend.procedural * procedural)
        + (blend.creative * creative)
        + (blend.advisory * advisory)
        + (blend.casual * casual)
}

fn phase_label(phase: TrainingPhaseKind) -> &'static str {
    match phase {
        TrainingPhaseKind::DryRun => "dry_run",
        TrainingPhaseKind::Bootstrap => "bootstrap",
        TrainingPhaseKind::Validation => "validation",
        TrainingPhaseKind::Expansion => "expansion",
        TrainingPhaseKind::Lifelong => "lifelong",
    }
}

fn training_source_label(source: &TrainingSource) -> String {
    source
        .name
        .clone()
        .or_else(|| source.value.clone())
        .unwrap_or_else(|| format!("{:?}", source.source_type))
}

fn is_skippable_training_error(err: &str) -> bool {
    err.starts_with("fatal:open_source_requires_value:") || is_huggingface_rate_limit_error(err)
}

fn is_huggingface_rate_limit_error(err: &str) -> bool {
    err.contains("huggingface rows request failed") && err.contains("HTTP 429")
}

fn euclidean_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn default_memory_for_training_source(source_type: TrainingSourceType) -> MemoryType {
    match source_type {
        TrainingSourceType::Url
        | TrainingSourceType::Document
        | TrainingSourceType::Dataset
        | TrainingSourceType::HuggingFaceDataset
        | TrainingSourceType::StructuredJson
        | TrainingSourceType::QaJson => MemoryType::Episodic,
        TrainingSourceType::OpenApiSpec | TrainingSourceType::CodeRepository => MemoryType::Core,
        TrainingSourceType::WikipediaDump
        | TrainingSourceType::WikidataTruthy
        | TrainingSourceType::OpenWebText
        | TrainingSourceType::DbpediaDump
        | TrainingSourceType::ProjectGutenberg
        | TrainingSourceType::CommonCrawlWet => MemoryType::Core,
    }
}

fn spawn_maintenance(
    memory: Arc<Mutex<MemoryStore>>,
    memory_snapshot: Arc<ArcSwap<MemorySnapshot>>,
    scheduler: Arc<PriorityScheduler>,
    config: EngineConfig,
) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(
            config.governance.maintenance_interval_secs,
        ));
        let _permit = scheduler.acquire(WorkPriority::Maintenance);
        if let Ok(mut memory) = memory.lock() {
            memory.run_maintenance(&config.governance);
            memory_snapshot.store(Arc::new(memory.snapshot()));
        }
    });
}

fn spawn_feedback_worker(
    memory: Arc<Mutex<MemoryStore>>,
    memory_snapshot: Arc<ArcSwap<MemorySnapshot>>,
    scheduler: Arc<PriorityScheduler>,
    feedback_rx: Receiver<Vec<crate::types::FeedbackEvent>>,
) {
    std::thread::spawn(move || {
        let ticker = crossbeam_channel::tick(Duration::from_millis(50));
        let mut pending = Vec::new();
        loop {
            crossbeam_channel::select! {
                recv(feedback_rx) -> message => match message {
                    Ok(events) => {
                        pending.extend(events);
                        if pending.len() >= 64 {
                            flush_feedback_batch(&memory, &memory_snapshot, &scheduler, &mut pending);
                        }
                    }
                    Err(_) => break,
                },
                recv(ticker) -> _ => {
                    if !pending.is_empty() {
                        flush_feedback_batch(&memory, &memory_snapshot, &scheduler, &mut pending);
                    }
                }
            }
        }
        if !pending.is_empty() {
            flush_feedback_batch(&memory, &memory_snapshot, &scheduler, &mut pending);
        }
    });
}

fn flush_feedback_batch(
    memory: &Arc<Mutex<MemoryStore>>,
    memory_snapshot: &Arc<ArcSwap<MemorySnapshot>>,
    scheduler: &Arc<PriorityScheduler>,
    pending: &mut Vec<crate::types::FeedbackEvent>,
) {
    let _permit = scheduler.acquire(WorkPriority::InteractiveTraining);
    let batch = std::mem::take(pending);
    if let Ok(mut memory) = memory.lock() {
        memory.apply_feedback(&batch);
        memory_snapshot.store(Arc::new(memory.snapshot()));
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dominant_behavior_profile_name, dynamic_intent_profile_blend, dynamic_trust_signal,
        evaluate_expression_input, grounded_evidence_answer, is_skippable_training_error,
        reshape_output_for_intent,
    };
    use crate::types::{
        IntentKind, IntentProfile, IntentScore, MemoryType, RetrievedDocument, ScoreBreakdown,
        ScoredCandidate,
    };
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn expression_evaluator_handles_basic_math() {
        assert_eq!(evaluate_expression_input("2 * 2").as_deref(), Some("4"));
        assert_eq!(
            evaluate_expression_input("what is (3 + 5) / 2 ?").as_deref(),
            Some("4")
        );
        assert_eq!(evaluate_expression_input("2 / 0"), None);
    }

    #[test]
    fn grounded_evidence_prefers_direct_person_answer() {
        let docs = vec![RetrievedDocument {
            source_url: "https://example.com/president".to_string(),
            title: "President of India".to_string(),
            raw_content: "The president of India is the head of state of the Republic of India. Droupadi Murmu is the 15th and current president, having taken office on 25 July 2022.".to_string(),
            normalized_content: "the president of india is the head of state of the republic of india droupadi murmu is the 15th and current president having taken office on 25 july 2022".to_string(),
            retrieved_at: Utc::now(),
            trust_score: 0.82,
            cached: false,
        }];

        let answer = grounded_evidence_answer("Who is president of india?", &docs)
            .expect("expected grounded answer");
        assert!(answer.contains("Droupadi Murmu"));
    }

    #[test]
    fn grounded_evidence_verification_prefers_status_bearing_sentence() {
        let docs = vec![RetrievedDocument {
            source_url: "https://example.com/president".to_string(),
            title: "President of India".to_string(),
            raw_content: "The president of India is the head of state of the Republic of India. The president is referred to as the first citizen of India. Droupadi Murmu is the current president, having taken office on 25 July 2022.".to_string(),
            normalized_content: "the president of india is the head of state of the republic of india the president is referred to as the first citizen of india droupadi murmu is the current president having taken office on 25 july 2022".to_string(),
            retrieved_at: Utc::now(),
            trust_score: 0.82,
            cached: false,
        }];

        let answer =
            grounded_evidence_answer("verify whether the president of india is current", &docs)
                .expect("expected grounded answer");
        assert!(answer.contains("Droupadi Murmu"));
    }

    #[test]
    fn adaptive_behavior_blend_favors_factual_profile_for_factual_queries() {
        let profile = IntentProfile {
            primary: IntentKind::Question,
            confidence: 0.88,
            scores: vec![
                IntentScore {
                    intent: IntentKind::Question,
                    score: 0.81,
                },
                IntentScore {
                    intent: IntentKind::Verify,
                    score: 0.34,
                },
                IntentScore {
                    intent: IntentKind::Explain,
                    score: 0.12,
                },
            ],
            ..IntentProfile::default()
        };

        let blend = dynamic_intent_profile_blend(&profile);
        assert_eq!(dominant_behavior_profile_name(&blend), Some("factual"));
    }

    #[test]
    fn adaptive_trust_signal_hardens_for_verification_requests() {
        let profile = IntentProfile {
            primary: IntentKind::Verify,
            confidence: 0.49,
            ambiguous: true,
            scores: vec![IntentScore {
                intent: IntentKind::Verify,
                score: 0.62,
            }],
            ..IntentProfile::default()
        };

        let signal = dynamic_trust_signal(&profile);
        assert!(signal.harden);
    }

    #[test]
    fn huggingface_rate_limit_errors_are_skippable() {
        let err = "fatal:huggingface rows request failed for HuggingFaceTB/smoltalk2/SFT/train at offset 0: HTTP 429 Too Many Requests";
        assert!(is_skippable_training_error(err));
    }

    #[test]
    fn reshape_output_formats_plan_as_numbered_steps() {
        let (text, strategy) = reshape_output_for_intent(
            "Plan the rollout",
            IntentKind::Plan,
            "Draft the rollout scope. Align stakeholders on milestones. Track launch readiness weekly.",
            &[],
            &[],
            &[],
        );

        assert_eq!(strategy, "numbered_steps");
        assert!(text.contains("1. Draft the rollout scope"));
        assert!(text.contains("2. Align stakeholders on milestones"));
    }

    #[test]
    fn reshape_output_formats_recommendation_with_reason() {
        let (text, strategy) = reshape_output_for_intent(
            "Recommend a database",
            IntentKind::Recommend,
            "Choose PostgreSQL for the primary store.",
            &[],
            &["It offers strong transactional guarantees.".to_string()],
            &[],
        );

        assert_eq!(strategy, "recommendation");
        assert!(text.contains("Recommendation: Choose PostgreSQL for the primary store."));
        assert!(text.contains("Why: It offers strong transactional guarantees."));
    }

    #[test]
    fn reshape_output_formats_debug_as_triage() {
        let (text, strategy) = reshape_output_for_intent(
            "Debug the failing login flow",
            IntentKind::Debug,
            "The login request is timing out at the auth proxy.",
            &[],
            &[
                "The auth proxy cannot reach the upstream identity service.".to_string(),
                "Inspect the proxy timeout and upstream health checks.".to_string(),
            ],
            &[],
        );

        assert_eq!(strategy, "debug_triage");
        assert!(text.contains("Issue: The login request is timing out at the auth proxy."));
        assert!(text
            .contains("Likely cause: The auth proxy cannot reach the upstream identity service."));
        assert!(text.contains("Next check: Inspect the proxy timeout and upstream health checks."));
    }

    #[test]
    fn reshape_output_uses_scored_candidates_for_brainstorming() {
        let scored = vec![
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "Offer a guided onboarding checklist for new users".to_string(),
                score: 0.9,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "Add milestone celebrations after each completed setup step".to_string(),
                score: 0.8,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
            ScoredCandidate {
                unit_id: Uuid::new_v4(),
                content: "Provide sample projects that users can clone and adapt".to_string(),
                score: 0.7,
                breakdown: ScoreBreakdown::default(),
                memory_type: MemoryType::Episodic,
            },
        ];

        let (text, strategy) = reshape_output_for_intent(
            "Brainstorm activation ideas",
            IntentKind::Brainstorm,
            "Activation ideas",
            &[],
            &[],
            &scored,
        );

        assert_eq!(strategy, "brainstorm_list");
        assert!(text.contains("1. Offer a guided onboarding checklist for new users"));
        assert!(text.contains("2. Add milestone celebrations after each completed setup step"));
    }
}
