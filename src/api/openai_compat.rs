//! OpenAI-Compatible API Layer
//!
//! Phase 6.2: Full OpenAI Chat Completions API compatibility for LLM replacement.
//! Model selection maps to SPSE profiles (ignored in Auto-Mode).
//! System prompt handling via L6 Context Manager.
//! Streaming SSE output for token-by-token responses.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::Engine;

// ============================================================================
// Request/Response Types (OpenAI-compatible)
// ============================================================================

/// OpenAI Chat Completion Request
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model name - ignored in Auto-Mode, maps to SPSE profiles
    pub model: Option<String>,
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Temperature - ignored in Auto-Mode
    pub temperature: Option<f32>,
    /// Max tokens for response
    pub max_tokens: Option<usize>,
    /// Enable streaming SSE output
    pub stream: Option<bool>,
    /// Top-p sampling - ignored in Auto-Mode
    pub top_p: Option<f32>,
    /// Frequency penalty - ignored in Auto-Mode
    pub frequency_penalty: Option<f32>,
    /// Presence penalty - ignored in Auto-Mode
    pub presence_penalty: Option<f32>,
    /// Stop sequences - ignored in Auto-Mode
    pub stop: Option<Vec<String>>,
}

/// OpenAI Message
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI Chat Completion Response
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    /// SPSE-specific: Intent classification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// SPSE-specific: Confidence score
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// SPSE-specific: Inferred tone
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tone: Option<String>,
}

/// OpenAI Choice
#[derive(Debug, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
}

/// OpenAI Usage
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// OpenAI Streaming Delta
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

/// OpenAI Stream Choice
#[derive(Debug, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// OpenAI Delta (streaming)
#[derive(Debug, Serialize, Deserialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// OpenAI Models List Response
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// OpenAI Model Info
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

// ============================================================================
// API State
// ============================================================================

#[derive(Clone)]
pub struct OpenAiApiState {
    pub engine: Arc<Engine>,
}

// ============================================================================
// API Handlers
// ============================================================================

/// POST /v1/chat/completions
/// 
/// OpenAI-compatible chat completions endpoint.
/// All parameters except messages are ignored in Auto-Mode.
pub async fn chat_completions(
    State(state): State<OpenAiApiState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // Validate messages
    if request.messages.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "messages is required and must be non-empty".to_string(),
                    r#type: "invalid_request_error".to_string(),
                    code: "invalid_messages".to_string(),
                },
            }),
        ));
    }

    // Extract the last user message as the query
    let last_message = request.messages.last().unwrap();
    let query = &last_message.content;

    // Check if streaming requested
    let stream = request.stream.unwrap_or(false);

    if stream {
        // Return SSE stream
        let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let model = request.model.clone().unwrap_or_else(|| "spse-auto".to_string());

        let stream = create_stream(response_id, created, model, query.clone());
        
        Ok(Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(15))
                .text("ping"),
        ).into_response())
    } else {
        // Non-streaming response
        let response = process_chat_completion(&state.engine, &request);
        Ok(Json(response).into_response())
    }
}

/// GET /v1/models
/// 
/// List available models (all map to SPSE Auto-Mode).
pub async fn list_models() -> impl IntoResponse {
    let models = ModelsResponse {
        object: "list".to_string(),
        data: vec![
            ModelInfo {
                id: "spse-auto".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "spse-engine".to_string(),
            },
            ModelInfo {
                id: "spse-creative".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "spse-engine".to_string(),
            },
            ModelInfo {
                id: "spse-precise".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "spse-engine".to_string(),
            },
        ],
    };
    Json(models)
}

// ============================================================================
// Internal Processing
// ============================================================================

/// Process a chat completion request (non-streaming)
fn process_chat_completion(
    engine: &Engine,
    request: &ChatCompletionRequest,
) -> ChatCompletionResponse {
    // Extract the last user message as the query
    let last_message = request.messages.last().unwrap();
    let query = &last_message.content;

    // Process through engine
    // Note: In Auto-Mode, temperature, top_p, etc. are ignored
    let result = engine.query(query);

    // Build response
    let response_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let model = request.model.clone().unwrap_or_else(|| "spse-auto".to_string());

    // Estimate token usage (rough approximation)
    let prompt_tokens: usize = request.messages.iter()
        .map(|m| m.content.split_whitespace().count())
        .sum();
    let completion_tokens = result.answer.split_whitespace().count();

    ChatCompletionResponse {
        id: response_id,
        object: "chat.completion".to_string(),
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: result.answer,
                name: None,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
        intent: Some(format!("{:?}", result.intent)),
        confidence: Some(result.confidence),
        tone: Some(format!("{:?}", result.tone)),
    }
}

/// Create SSE stream for streaming response
fn create_stream(
    response_id: String,
    created: u64,
    model: String,
    content: String,
) -> impl Stream<Item = Result<Event, axum::Error>> {
    // Split content into chunks for streaming simulation
    let words: Vec<&str> = content.split_whitespace().collect();
    let chunk_size = (words.len() / 10).max(1);
    
    let chunks: Vec<String> = words
        .chunks(chunk_size)
        .map(|chunk| chunk.join(" "))
        .collect();

    let stream = futures::stream::iter(chunks.into_iter().enumerate().map(move |(i, chunk)| {
        let delta = if i == 0 {
            Delta {
                role: Some("assistant".to_string()),
                content: Some(chunk + " "),
            }
        } else {
            Delta {
                role: None,
                content: Some(chunk + " "),
            }
        };

        let finish_reason = if i == chunks.len() - 1 {
            Some("stop".to_string())
        } else {
            None
        };

        let chunk = ChatCompletionChunk {
            id: response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        };

        Ok(Event::default().json_data(chunk).unwrap())
    }));

    // Add [DONE] marker at end
    let done_stream = futures::stream::once(async { Ok(Event::default().data("[DONE]")) });
    
    futures::stream::select(stream, done_stream)
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

// ============================================================================
// Router
// ============================================================================

/// Create OpenAI-compatible API router
pub fn openai_router(engine: Arc<Engine>) -> axum::Router<OpenAiApiState> {
    let state = OpenAiApiState { engine };
    
    axum::Router::new()
        .route("/v1/chat/completions", axum::routing::post(chat_completions))
        .route("/v1/models", axum::routing::get(list_models))
        .with_state(state)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completion_request_deserialization() {
        let json = r#"{
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7
        }"#;

        let request: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.model, Some("gpt-4".to_string()));
        assert_eq!(request.messages.len(), 1);
        assert_eq!(request.temperature, Some(0.7));
    }

    #[test]
    fn test_chat_completion_response_serialization() {
        let response = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion".to_string(),
            created: 1700000000,
            model: "spse-auto".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: "Hello!".to_string(),
                    name: None,
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            intent: Some("Factual".to_string()),
            confidence: Some(0.95),
            tone: Some("NeutralProfessional".to_string()),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("chatcmpl-123"));
        assert!(json.contains("Hello!"));
        assert!(json.contains("Factual"));
    }

    #[test]
    fn test_models_response() {
        let response = ModelsResponse {
            object: "list".to_string(),
            data: vec![ModelInfo {
                id: "spse-auto".to_string(),
                object: "model".to_string(),
                created: 1700000000,
                owned_by: "spse-engine".to_string(),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("spse-auto"));
    }

    #[test]
    fn test_streaming_chunk() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1700000000,
            model: "spse-auto".to_string(),
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: Some("Hello".to_string()),
                },
                finish_reason: None,
            }],
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
        assert!(json.contains("delta"));
    }
}
