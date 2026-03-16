//! Ingestion Helpers
//!
//! Shared text ingestion pipeline used by both Engine and Training systems.
//! Extracts embedded logic to avoid duplication.

use crate::classification::{input, HierarchicalUnitOrganizer, UnitBuilder};
use crate::config::UnitBuilderConfig;
use crate::memory::store::MemoryStore;
use crate::types::{InputPacket, SourceKind, UnitHierarchy};

/// Ingest raw text through the full pipeline: normalize → build → organize → store
/// Returns the number of active unit IDs created.
pub fn ingest_text(
    memory: &mut MemoryStore,
    text: &str,
    source_kind: SourceKind,
    context: &str,
    config: &UnitBuilderConfig,
) -> Vec<uuid::Uuid> {
    if text.trim().is_empty() {
        return vec![];
    }

    let packet = input::ingest_raw(text, !matches!(source_kind, SourceKind::UserInput));
    let build_output = UnitBuilder::build_units_static(&packet, config);
    let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, config);
    
    let context_summary = if context.is_empty() {
        summarize_packet(&packet)
    } else {
        context.to_string()
    };

    memory.ingest_hierarchy(&hierarchy, source_kind, &context_summary)
}

/// Ingest text in training mode (skips candidate staging, defers persistence)
pub fn ingest_text_training(
    memory: &mut MemoryStore,
    text: &str,
    source_kind: SourceKind,
    context: &str,
    config: &UnitBuilderConfig,
) -> Vec<uuid::Uuid> {
    memory.set_training_mode(true);
    let ids = ingest_text(memory, text, source_kind, context, config);
    memory.set_training_mode(false);
    ids
}

/// Batch ingest multiple texts in training mode
pub fn ingest_texts_batch(
    memory: &mut MemoryStore,
    texts: &[&str],
    source_kind: SourceKind,
    context: &str,
    config: &UnitBuilderConfig,
) -> usize {
    memory.set_training_mode(true);
    
    let mut total = 0;
    for text in texts {
        let ids = ingest_text(memory, text, source_kind, context, config);
        total += ids.len();
    }
    
    memory.set_training_mode(false);
    total
}

/// Build units from text without storing (for validation/testing)
pub fn build_hierarchy(
    text: &str,
    config: &UnitBuilderConfig,
) -> UnitHierarchy {
    let packet = input::ingest_raw(text, true);
    let build_output = UnitBuilder::build_units_static(&packet, config);
    HierarchicalUnitOrganizer::organize(&build_output, config)
}

/// Summarize an input packet for context
fn summarize_packet(packet: &InputPacket) -> String {
    // Use normalized_text, truncated to reasonable length
    packet.normalized_text.chars().take(100).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_hierarchy() {
        let config = UnitBuilderConfig::default();
        let hierarchy = build_hierarchy("The capital of France is Paris.", &config);
        assert!(!hierarchy.levels.is_empty());
    }
}
