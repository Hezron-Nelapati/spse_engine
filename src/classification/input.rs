use crate::config::ClassificationConfig;
use crate::types::InputPacket;
use chrono::Utc;
use once_cell::sync::Lazy;
use postagger::PerceptronTagger;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

pub fn ingest_raw(text: &str, training_mode: bool) -> InputPacket {
    ingest_raw_with_config(text, training_mode, &ClassificationConfig::default())
}

pub fn ingest_raw_with_config(
    text: &str,
    training_mode: bool,
    classification: &ClassificationConfig,
) -> InputPacket {
    let normalized = normalize_text_with_config(text, classification);
    InputPacket {
        original_text: text.to_string(),
        bytes: normalized.as_bytes().to_vec(),
        normalized_text: normalized,
        training_mode,
        timestamp: Utc::now(),
    }
}

pub fn normalize_text(text: &str) -> String {
    normalize_text_with_config(text, &ClassificationConfig::default())
}

pub fn normalize_text_with_config(text: &str, classification: &ClassificationConfig) -> String {
    let merged = merge_compound_nouns(text, classification.compound_noun_min_frequency);
    merged
        .split_whitespace()
        .map(|segment| segment.trim_matches(|ch: char| ch.is_control()))
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Default)]
struct CompoundRegistry {
    observed_counts: HashMap<String, u32>,
    known_compounds: HashSet<String>,
}

static COMPOUND_REGISTRY: Lazy<Mutex<CompoundRegistry>> =
    Lazy::new(|| Mutex::new(CompoundRegistry::default()));

static POS_TAGGER: Lazy<Option<PerceptronTagger>> = Lazy::new(|| {
    let paths = [(
        "config/pos_tagger/weights.json",
        "config/pos_tagger/classes.txt",
        "config/pos_tagger/tags.json",
    )];
    for (weights, classes, tags) in &paths {
        if std::path::Path::new(weights).exists() {
            return Some(PerceptronTagger::new(weights, classes, tags));
        }
    }
    None
});

fn merge_compound_nouns(text: &str, min_frequency: u32) -> String {
    let segments = text
        .split_whitespace()
        .map(|segment| segment.trim_matches(|ch: char| ch.is_control()).to_string())
        .filter(|segment| !segment.is_empty())
        .collect::<Vec<_>>();
    if segments.len() < 2 {
        return segments.join(" ");
    }

    let normalized_tokens = segments
        .iter()
        .map(|segment| token_core(segment))
        .collect::<Vec<_>>();
    let tags = tag_tokens(&normalized_tokens);
    if tags.len() != segments.len() {
        return segments.join(" ");
    }

    let mut registry = COMPOUND_REGISTRY
        .lock()
        .expect("compound registry mutex poisoned");
    let mut merged = Vec::with_capacity(segments.len());
    let mut index = 0usize;

    while index < segments.len() {
        if index + 1 < segments.len()
            && should_merge_compound(
                &normalized_tokens[index],
                &normalized_tokens[index + 1],
                &tags[index],
                &tags[index + 1],
                min_frequency,
                &mut registry,
            )
        {
            merged.push(render_compound(
                &segments[index],
                &segments[index + 1],
                &normalized_tokens[index],
                &normalized_tokens[index + 1],
            ));
            index += 2;
            continue;
        }

        merged.push(segments[index].clone());
        index += 1;
    }

    merged.join(" ")
}

fn should_merge_compound(
    left: &str,
    right: &str,
    left_tag: &str,
    right_tag: &str,
    min_frequency: u32,
    registry: &mut CompoundRegistry,
) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }

    let key = format!("{}_{}", left.to_lowercase(), right.to_lowercase());
    let proper_noun_pair = left_tag.starts_with("NNP") && right_tag.starts_with("NNP");
    let common_noun_pair = left_tag.starts_with("NN") && right_tag.starts_with("NN");

    if proper_noun_pair {
        registry.known_compounds.insert(key);
        return true;
    }

    if registry.known_compounds.contains(&key) {
        return true;
    }

    if common_noun_pair {
        let count = registry.observed_counts.entry(key.clone()).or_insert(0);
        *count += 1;
        if *count >= min_frequency.max(1) {
            registry.known_compounds.insert(key);
            return true;
        }
    }

    false
}

fn render_compound(left_raw: &str, right_raw: &str, left: &str, right: &str) -> String {
    let left_leading = leading_punctuation(left_raw);
    let right_trailing = trailing_punctuation(right_raw);
    let merged = if has_case_signal(left_raw) || has_case_signal(right_raw) {
        format!(
            "{}_{}",
            preserve_case(left_raw, left),
            preserve_case(right_raw, right)
        )
    } else {
        format!("{}_{}", left, right)
    };
    format!("{left_leading}{merged}{right_trailing}")
}

fn token_core(token: &str) -> String {
    token
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '-' || *ch == '_')
        .collect()
}

fn preserve_case(raw: &str, normalized: &str) -> String {
    let trimmed = raw.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_');
    if trimmed.is_empty() {
        normalized.to_string()
    } else {
        trimmed.to_string()
    }
}

fn leading_punctuation(token: &str) -> String {
    token
        .chars()
        .take_while(|ch| !ch.is_alphanumeric() && *ch != '-' && *ch != '_')
        .collect()
}

fn trailing_punctuation(token: &str) -> String {
    token
        .chars()
        .rev()
        .take_while(|ch| !ch.is_alphanumeric() && *ch != '-' && *ch != '_')
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

fn has_case_signal(token: &str) -> bool {
    token.chars().any(|ch| ch.is_uppercase())
}

fn tag_tokens(tokens: &[String]) -> Vec<String> {
    let Some(tagger) = POS_TAGGER.as_ref() else {
        return Vec::new();
    };

    let safe_tokens = tokens
        .iter()
        .map(|token| sanitize_for_pos(token))
        .collect::<Vec<_>>();
    if safe_tokens.iter().any(|token| token.is_empty()) {
        return Vec::new();
    }

    let joined = safe_tokens.join(" ");
    let tags = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tagger.tag(&joined)))
    {
        Ok(tags) => tags,
        Err(_) => return Vec::new(),
    };
    tags.into_iter().map(|tag| tag.tag).collect()
}

fn sanitize_for_pos(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == ' ' {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
