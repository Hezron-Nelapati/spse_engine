use crate::types::InputPacket;
use chrono::Utc;

pub fn ingest_raw(text: &str, training_mode: bool) -> InputPacket {
    let normalized = normalize_text(text);
    InputPacket {
        original_text: text.to_string(),
        bytes: normalized.as_bytes().to_vec(),
        normalized_text: normalized,
        training_mode,
        timestamp: Utc::now(),
    }
}

pub fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .map(|segment| segment.trim_matches(|ch: char| ch.is_control()))
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
