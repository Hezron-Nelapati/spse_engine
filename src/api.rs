use crate::config::AutoModeConfig;
use crate::engine::Engine;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{response::IntoResponse, Json, Router};
use serde::Serialize;
use std::sync::Arc;

pub mod openai_compat;

#[derive(Clone)]
pub struct ApiState {
    pub engine: Arc<Engine>,
    pub auto_mode_config: AutoModeConfig,
}

pub fn router(engine: Arc<Engine>) -> Router {
    let auto_mode_config = engine.config().auto_inference.auto_mode.clone();
    let state = ApiState {
        engine,
        auto_mode_config,
    };

    Router::new()
        .route("/api/v1/status", get(auto_mode_status))
        .route(
            "/v1/chat/completions",
            post(openai_compat::chat_completions),
        )
        .route("/v1/models", get(openai_compat::list_models))
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

async fn auto_mode_status(State(state): State<ApiState>) -> impl IntoResponse {
    #[derive(Debug, Serialize)]
    struct AutoModeStatus {
        mode: String,
        locked: bool,
        indicator: String,
    }

    Json(AutoModeStatus {
        mode: "auto".to_string(),
        locked: state.auto_mode_config.locked,
        indicator: state.auto_mode_config.indicator_label.clone(),
    })
}
