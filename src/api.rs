use crate::config::AutoModeConfig;
use crate::engine::Engine;
use crate::proto::api as proto_api;
use crate::types::{
    JobState, QueueDepths, TrainRequest, TrainingExecutionMode, TrainingJobStatus,
    TrainingPhaseKind,
};
use axum::extract::{Path, State};
use axum::http::header::{ACCEPT, CONTENT_TYPE};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{body::Bytes, http::HeaderMap, response::IntoResponse, Json, Router};
use prost::Message;
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone)]
struct ApiState {
    engine: Arc<Engine>,
    auto_mode_config: AutoModeConfig,
}

#[derive(Debug, Serialize)]
struct AcceptedJob {
    job_id: String,
}

#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
}

const PROTOBUF_MIME: &str = "application/x-protobuf";

pub fn router(engine: Arc<Engine>) -> Router {
    let auto_mode_config = engine.config().auto_inference.auto_mode.clone();
    let state = ApiState { engine, auto_mode_config };
    Router::new()
        .route("/api/v1/train/batch", post(train_batch))
        .route("/api/v1/train/status/:job_id", get(training_status))
        .route("/api/v1/status", get(auto_mode_status))
        .with_state(state)
}

pub async fn serve(engine: Arc<Engine>, port: u16) -> Result<(), String> {
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .map_err(|err| format!("failed to bind server on port {port}: {err}"))?;
    axum::serve(listener, router(engine))
        .await
        .map_err(|err| format!("server error: {err}"))
}

async fn train_batch(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    let request = decode_train_request(&headers, &body).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            encode_error(&headers, ApiError { error }),
        )
    })?;
    if !request.mode.eq_ignore_ascii_case("silent") {
        return Err((
            StatusCode::BAD_REQUEST,
            encode_error(
                &headers,
                ApiError {
                    error: format!("unsupported training mode: {}", request.mode),
                },
            ),
        ));
    }
    let execution_mode = request.execution_mode;
    let job_id = state.engine.clone().start_train(execution_mode);
    Ok((
        StatusCode::ACCEPTED,
        encode_accepted(&headers, AcceptedJob { job_id }),
    ))
}

async fn training_status(
    Path(job_id): Path<String>,
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<impl IntoResponse, (StatusCode, impl IntoResponse)> {
    state
        .engine
        .training_status(&job_id)
        .map(|status| encode_status(&headers, status))
        .ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                encode_error(
                    &headers,
                    ApiError {
                        error: format!("training job not found: {job_id}"),
                    },
                ),
            )
        })
}

/// Phase 3.4: Auto-Mode Status Endpoint
/// Returns the auto-mode indicator label to confirm engine is locked to auto mode.
async fn auto_mode_status(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    #[derive(Debug, Serialize)]
    struct AutoModeStatus {
        mode: String,
        locked: bool,
        indicator: String,
    }

    let status = AutoModeStatus {
        mode: "auto".to_string(),
        locked: state.auto_mode_config.locked,
        indicator: state.auto_mode_config.indicator_label.clone(),
    };

    if wants_protobuf(&headers, ACCEPT) {
        // For protobuf, return JSON since we don't have a proto definition for this
        Json(status).into_response()
    } else {
        Json(status).into_response()
    }
}

fn decode_train_request(headers: &HeaderMap, body: &[u8]) -> Result<TrainRequest, String> {
    if body.is_empty() {
        return Ok(TrainRequest::default());
    }

    if wants_protobuf(headers, CONTENT_TYPE) {
        let request = proto_api::TrainRequest::decode(body).map_err(|err| err.to_string())?;
        return Ok(TrainRequest {
            mode: if request.mode.is_empty() {
                "silent".to_string()
            } else {
                request.mode
            },
            execution_mode: parse_execution_mode(&request.execution_mode),
        });
    }

    serde_json::from_slice(body).map_err(|err| err.to_string())
}

fn encode_accepted(headers: &HeaderMap, payload: AcceptedJob) -> impl IntoResponse {
    if wants_protobuf(headers, ACCEPT) {
        let proto = proto_api::AcceptedJob {
            job_id: payload.job_id,
        };
        return ([(CONTENT_TYPE, PROTOBUF_MIME)], proto.encode_to_vec()).into_response();
    }

    Json(payload).into_response()
}

fn encode_error(headers: &HeaderMap, payload: ApiError) -> impl IntoResponse {
    if wants_protobuf(headers, ACCEPT) {
        let proto = proto_api::ApiError {
            error: payload.error,
        };
        return ([(CONTENT_TYPE, PROTOBUF_MIME)], proto.encode_to_vec()).into_response();
    }

    Json(payload).into_response()
}

fn encode_status(headers: &HeaderMap, status: TrainingJobStatus) -> impl IntoResponse {
    if wants_protobuf(headers, ACCEPT) {
        let proto = proto_api::TrainingJobStatus {
            job_id: status.job_id,
            status: job_state_label(status.status).to_string(),
            active_phase: status
                .active_phase
                .map(phase_label)
                .unwrap_or_default()
                .to_string(),
            phase_statuses: status
                .phase_statuses
                .into_iter()
                .map(|phase| proto_api::TrainingPhaseStatus {
                    phase: phase_label(phase.phase).to_string(),
                    status: job_state_label(phase.status).to_string(),
                    batches_completed: phase.batches_completed as u32,
                    batches_target: phase.batches_target as u32,
                    sources_processed: phase.sources_processed as u32,
                    sources_total: phase.sources_total as u32,
                    metrics: Some(proto_api::TrainingMetrics {
                        new_unit_rate: phase.metrics.new_unit_rate,
                        unit_discovery_efficiency: phase.metrics.unit_discovery_efficiency,
                        semantic_routing_accuracy: phase.metrics.semantic_routing_accuracy,
                        prediction_error: phase.metrics.prediction_error,
                        memory_delta_kb: phase.metrics.memory_delta_kb,
                        search_trigger_precision: phase
                            .metrics
                            .search_trigger_precision
                            .unwrap_or_default(),
                    }),
                })
                .collect(),
            progress: Some(proto_api::TrainingProgress {
                percent_complete: status.progress.percent_complete,
                sources_processed: status.progress.sources_processed as u32,
                sources_total: status.progress.sources_total as u32,
                active_source: status.progress.active_source.unwrap_or_default(),
                chunks_processed: status.progress.chunks_processed,
                bytes_processed: status.progress.bytes_processed,
            }),
            learning_metrics: Some(proto_api::LearningMetrics {
                new_units_discovered: status.learning_metrics.new_units_discovered,
                units_pruned: status.learning_metrics.units_pruned,
                memory_delta_kb: status.learning_metrics.memory_delta_kb,
                map_adjustments: status.learning_metrics.map_adjustments,
                anchors_protected: status.learning_metrics.anchors_protected,
                efficiency: Some(proto_api::TrainingEfficiencyMetrics {
                    units_discovered_per_mb: status
                        .learning_metrics
                        .efficiency
                        .units_discovered_per_mb,
                    units_discovered_per_kb: status
                        .learning_metrics
                        .efficiency
                        .units_discovered_per_kb,
                    pruned_units_percent: status.learning_metrics.efficiency.pruned_units_percent,
                    anchor_density: status.learning_metrics.efficiency.anchor_density,
                    candidate_to_active_ratio: status
                        .learning_metrics
                        .efficiency
                        .candidate_to_active_ratio,
                    pruned_candidates_percent: status
                        .learning_metrics
                        .efficiency
                        .pruned_candidates_percent,
                    avg_observations_to_promotion: status
                        .learning_metrics
                        .efficiency
                        .avg_observations_to_promotion,
                    map_rebuild_frequency: status.learning_metrics.efficiency.map_rebuild_frequency,
                    cache_hit_rate: status.learning_metrics.efficiency.cache_hit_rate,
                    bloom_filter_false_positive_rate: status
                        .learning_metrics
                        .efficiency
                        .bloom_filter_false_positive_rate,
                }),
                database_health: Some(proto_api::DatabaseHealthMetrics {
                    total_units: status.learning_metrics.database_health.total_units,
                    core_units: status.learning_metrics.database_health.core_units,
                    episodic_units: status.learning_metrics.database_health.episodic_units,
                    anchor_units: status.learning_metrics.database_health.anchor_units,
                    intent_units: status.learning_metrics.database_health.intent_units,
                    reasoning_units: status.learning_metrics.database_health.reasoning_units,
                    maturity_stage: maturity_stage_label(
                        status.learning_metrics.database_health.maturity_stage,
                    )
                    .to_string(),
                    active_units: status.learning_metrics.database_health.active_units,
                    candidate_units: status.learning_metrics.database_health.candidate_units,
                    validated_candidates: status
                        .learning_metrics
                        .database_health
                        .validated_candidates,
                    active_candidates: status.learning_metrics.database_health.active_candidates,
                    rejected_candidates: status
                        .learning_metrics
                        .database_health
                        .rejected_candidates,
                    pruned_units: status.learning_metrics.database_health.pruned_units,
                    wal_size_mb: status.learning_metrics.database_health.wal_size_mb,
                    index_fragmentation: status
                        .learning_metrics
                        .database_health
                        .index_fragmentation,
                }),
                memory_governance: Some(proto_api::MemoryGovernanceMetrics {
                    episodic_memory_mb: status
                        .learning_metrics
                        .memory_governance
                        .episodic_memory_mb,
                    core_memory_mb: status.learning_metrics.memory_governance.core_memory_mb,
                    candidate_pool_mb: status.learning_metrics.memory_governance.candidate_pool_mb,
                    daily_growth_mb: status.learning_metrics.memory_governance.daily_growth_mb,
                    pruning_rate_per_hour: status
                        .learning_metrics
                        .memory_governance
                        .pruning_rate_per_hour,
                }),
            }),
            performance: Some(proto_api::PerformanceMetrics {
                avg_ms_per_source: status.performance.avg_ms_per_source,
                queue_depths: Some(queue_depths_to_proto(&status.performance.queue_depths)),
            }),
            intent_distribution: status
                .intent_distribution
                .into_iter()
                .map(|(key, value)| proto_api::IntentCount { key, value })
                .collect(),
            warnings: status.warnings,
        };
        return ([(CONTENT_TYPE, PROTOBUF_MIME)], proto.encode_to_vec()).into_response();
    }

    Json(status).into_response()
}

fn wants_protobuf(headers: &HeaderMap, header: axum::http::header::HeaderName) -> bool {
    headers
        .get(header)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.contains(PROTOBUF_MIME))
        .unwrap_or(false)
}

fn parse_execution_mode(value: &str) -> TrainingExecutionMode {
    match value.trim().to_ascii_lowercase().as_str() {
        "user" => TrainingExecutionMode::User,
        "development" | "dev" | "" => TrainingExecutionMode::Development,
        _ => TrainingExecutionMode::Development,
    }
}

fn job_state_label(state: JobState) -> &'static str {
    match state {
        JobState::Queued => "queued",
        JobState::Processing => "processing",
        JobState::Completed => "completed",
        JobState::Failed => "failed",
    }
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

fn maturity_stage_label(stage: crate::types::DatabaseMaturityStage) -> &'static str {
    match stage {
        crate::types::DatabaseMaturityStage::ColdStart => "cold_start",
        crate::types::DatabaseMaturityStage::Growth => "growth",
        crate::types::DatabaseMaturityStage::Stable => "stable",
    }
}

fn queue_depths_to_proto(depths: &QueueDepths) -> proto_api::QueueDepths {
    proto_api::QueueDepths {
        inference: depths.inference as u32,
        interactive_training: depths.interactive_training as u32,
        silent_batch: depths.silent_batch as u32,
        maintenance: depths.maintenance as u32,
    }
}
