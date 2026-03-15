use crate::types::{
    MemoryChannel, MemoryType, TrainingSource, TrainingSourceType, TrainingStreamConfig,
};

#[derive(Debug, Clone, Copy)]
pub struct OpenSourceDefinition {
    pub id: &'static str,
    pub label: &'static str,
    pub category: &'static str,
    pub license: &'static str,
    pub integration: &'static str,
    pub summary: &'static str,
    pub default_type: TrainingSourceType,
    pub default_value: Option<&'static str>,
    pub default_memory: MemoryType,
    pub default_item_limit: Option<usize>,
    pub default_batch_size: Option<usize>,
    pub default_chunk_char_limit: Option<usize>,
}

/// Internal-only seed datasets. No external/open-source datasets are used for training or seeding.
/// The engine acquires knowledge at runtime via web retrieval triggered during reasoning,
/// not through pre-baked open-source corpora.
const OPEN_SOURCES: &[OpenSourceDefinition] = &[
    OpenSourceDefinition {
        id: "seed_intelligence",
        label: "Seed Intelligence Dataset",
        category: "reasoning",
        license: "Internal",
        integration: "structured_json",
        summary: "Reasoning chains, retrieval triggers, confidence gating, multi-hop reasoning, self-correction, and multi-step web retrieval seeds.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/seeds/intelligence.jsonl"),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(200_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "seed_entities",
        label: "Seed Entity Dataset",
        category: "core_kb",
        license: "Internal",
        integration: "structured_json",
        summary: "High-density entity definitions with rich attributes, cross-references, and domain contexts for core knowledge base seeding.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/seeds/entities.jsonl"),
        default_memory: MemoryType::Core,
        default_item_limit: Some(200_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(6_000),
    },
    OpenSourceDefinition {
        id: "seed_dialogues",
        label: "Seed Dialogue Dataset",
        category: "intent_dialogue",
        license: "Internal",
        integration: "structured_json",
        summary: "Multi-turn knowledge dialogues, clarification chains, and social patterns with domain-contextualized follow-ups.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/seeds/dialogues.jsonl"),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(200_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(8_000),
    },
    OpenSourceDefinition {
        id: "seed_classification",
        label: "Seed Classification Dataset",
        category: "intent_dialogue",
        license: "Internal",
        integration: "structured_json",
        summary: "Intent, tone, and resolver mode classification patterns across all 24 intent kinds with domain-specific examples.",
        default_type: TrainingSourceType::StructuredJson,
        default_value: Some("datasets/seeds/classification.jsonl"),
        default_memory: MemoryType::Episodic,
        default_item_limit: Some(200_000),
        default_batch_size: Some(500),
        default_chunk_char_limit: Some(8_000),
    },
];

pub fn catalog() -> &'static [OpenSourceDefinition] {
    OPEN_SOURCES
}

pub fn render_catalog_table() -> String {
    let mut lines = vec![
        "id | label | category | license | integration | reference".to_string(),
        "--- | --- | --- | --- | --- | ---".to_string(),
    ];
    for source in OPEN_SOURCES {
        lines.push(format!(
            "{} | {} | {} | {} | {} | {}",
            source.id,
            source.label,
            source.category,
            source.license,
            source.integration,
            reference_url(source.id).unwrap_or("-")
        ));
    }
    lines.join("\n")
}

pub fn reference_url(id: &str) -> Option<&'static str> {
    find_definition(id).and_then(|d| d.default_value)
}

pub fn resolve_training_source(source: &TrainingSource) -> Result<TrainingSource, String> {
    if matches!(source.source_type, TrainingSourceType::Dataset) && source.name.is_none() {
        return Err("dataset_source_requires_name".to_string());
    }

    let Some(definition) = source
        .name
        .as_deref()
        .map(|name| find_definition(name).ok_or_else(|| format!("unknown_open_source:{name}")))
        .transpose()?
    else {
        return Ok(source.clone());
    };

    let mut resolved = source.clone();
    if matches!(resolved.source_type, TrainingSourceType::Dataset) {
        resolved.source_type = definition.default_type;
    }
    if resolved.value.is_none() {
        resolved.value = definition.default_value.map(str::to_string);
    }
    if resolved.target_memory.is_none() {
        resolved.target_memory = Some(definition.default_memory);
    }
    if resolved.memory_channels.is_none() {
        resolved.memory_channels = Some(default_channels_for_definition(definition));
    }
    apply_stream_defaults(&mut resolved.stream, definition);
    if resolved.value.is_none() && resolved.content.is_none() {
        return Err(format!("source_requires_value:{}", definition.id));
    }
    Ok(resolved)
}

fn apply_stream_defaults(stream: &mut TrainingStreamConfig, definition: OpenSourceDefinition) {
    if stream.item_limit.is_none() {
        stream.item_limit = definition.default_item_limit;
    }
    if stream.batch_size.is_none() {
        stream.batch_size = definition.default_batch_size;
    }
    if stream.chunk_char_limit.is_none() {
        stream.chunk_char_limit = definition.default_chunk_char_limit;
    }
}

fn find_definition(id: &str) -> Option<OpenSourceDefinition> {
    OPEN_SOURCES
        .iter()
        .find(|definition| definition.id.eq_ignore_ascii_case(id))
        .copied()
}

pub fn category_for_source(id: &str) -> Option<&'static str> {
    find_definition(id).map(|definition| definition.category)
}

fn default_channels_for_definition(definition: OpenSourceDefinition) -> Vec<MemoryChannel> {
    match definition.category {
        "intent_dialogue" => vec![MemoryChannel::Main, MemoryChannel::Intent],
        "reasoning" => vec![MemoryChannel::Main, MemoryChannel::Reasoning],
        _ => vec![MemoryChannel::Main],
    }
}

pub fn catalog_source(name: &str) -> TrainingSource {
    TrainingSource {
        source_type: TrainingSourceType::Dataset,
        name: Some(name.to_string()),
        value: None,
        mime: None,
        content: None,
        target_memory: None,
        memory_channels: None,
        stream: TrainingStreamConfig::default(),
    }
}
