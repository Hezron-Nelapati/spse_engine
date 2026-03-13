use crate::config::{QueryBuilderConfig, TrustConfig};
use crate::types::{ContextMatrix, SanitizedQuery, SequenceState};
use std::collections::BTreeSet;

pub struct SafeQueryBuilder;

impl SafeQueryBuilder {
    pub fn build(
        raw_input: &str,
        context: &ContextMatrix,
        sequence: &SequenceState,
        query: &QueryBuilderConfig,
        trust: &TrustConfig,
    ) -> SanitizedQuery {
        let focused_input = focus_query_text(raw_input);
        let mut removed_tokens = Vec::new();
        let mut kept = Vec::new();
        let mut seen_terms = BTreeSet::new();
        let mut pii_redacted = false;

        for token in focused_input.split_whitespace() {
            let sanitized =
                token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_');
            if sanitized.is_empty() {
                continue;
            }

            let looks_like_email = sanitized.contains('@');
            let digit_count = sanitized.chars().filter(|ch| ch.is_ascii_digit()).count();
            let looks_like_long_number = match query.pii_stripping_aggressiveness.as_str() {
                "low" => false,
                "high" => digit_count >= 4,
                _ => digit_count >= 6,
            };
            if looks_like_email || looks_like_long_number {
                removed_tokens.push(token.to_string());
                pii_redacted = true;
                continue;
            }

            let lowered = sanitized.to_lowercase();
            if is_query_scaffold_term(&lowered) {
                removed_tokens.push(token.to_string());
                continue;
            }
            if seen_terms.insert(lowered.clone()) {
                kept.push(lowered);
            }
        }

        let mut semantic_expansions = Vec::new();
        for entity in &sequence.task_entities {
            if semantic_expansions.len() >= query.max_query_expansion_terms {
                break;
            }
            if let Some(expansion) = candidate_expansion(entity, &seen_terms) {
                for token in normalized_tokens(&expansion) {
                    seen_terms.insert(token);
                }
                semantic_expansions.push(expansion);
            }
        }
        for cell in context
            .cells
            .iter()
            .take(query.max_query_expansion_terms.max(1))
        {
            if semantic_expansions.len() >= query.max_query_expansion_terms {
                break;
            }
            if let Some(expansion) = candidate_expansion(&cell.content, &seen_terms) {
                if !semantic_expansions.contains(&expansion) {
                    for token in normalized_tokens(&expansion) {
                        seen_terms.insert(token);
                    }
                    semantic_expansions.push(expansion);
                }
            }
        }

        let mut query_terms = kept;
        if query_terms.is_empty() {
            query_terms = normalized_tokens(&focused_input)
                .into_iter()
                .filter(|term| !is_query_scaffold_term(term))
                .collect();
        }
        query_terms.truncate(trust.max_query_terms);

        SanitizedQuery {
            raw_query: raw_input.to_string(),
            sanitized_query: query_terms.join(" "),
            semantic_expansions,
            removed_tokens,
            pii_redacted,
        }
    }
}

fn candidate_expansion(text: &str, known_terms: &BTreeSet<String>) -> Option<String> {
    let tokens = normalized_tokens(text);
    if tokens.len() < 2 || tokens.len() > 6 {
        return None;
    }

    let novel_tokens = tokens
        .iter()
        .filter(|token| !known_terms.contains(*token) && !is_low_value_term(token))
        .count();
    if novel_tokens == 0 {
        return None;
    }

    Some(tokens.join(" "))
}

fn normalized_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut seen = BTreeSet::new();
    for token in text.split_whitespace() {
        let lowered = token
            .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '-' && ch != '_')
            .to_lowercase();
        if !lowered.is_empty() && seen.insert(lowered.clone()) {
            tokens.push(lowered);
        }
    }
    tokens
}

fn is_low_value_term(token: &str) -> bool {
    matches!(
        token,
        "the" | "and" | "for" | "with" | "that" | "this" | "from" | "about" | "into" | "onto"
    )
}

fn is_query_scaffold_term(token: &str) -> bool {
    matches!(
        token,
        "tell"
            | "about"
            | "explain"
            | "describe"
            | "show"
            | "give"
            | "list"
            | "summarize"
            | "summarise"
            | "overview"
            | "please"
            | "could"
            | "would"
            | "me"
            | "all"
    )
}

pub(crate) fn focus_query_text(raw_input: &str) -> String {
    let lowered = raw_input.trim().to_lowercase();
    let candidate = lowered
        .strip_prefix("verify whether ")
        .or_else(|| lowered.strip_prefix("verify if "))
        .or_else(|| lowered.strip_prefix("check whether "))
        .or_else(|| lowered.strip_prefix("check if "))
        .or_else(|| lowered.strip_prefix("confirm whether "))
        .or_else(|| lowered.strip_prefix("confirm if "))
        .or_else(|| lowered.strip_prefix("tell me whether "))
        .or_else(|| lowered.strip_prefix("tell me if "))
        .or_else(|| lowered.strip_prefix("is it true that "))
        .or_else(|| lowered.strip_prefix("what can you tell me about "))
        .or_else(|| lowered.strip_prefix("tell me more about "))
        .or_else(|| lowered.strip_prefix("tell me about "))
        .or_else(|| lowered.strip_prefix("explain the "))
        .or_else(|| lowered.strip_prefix("explain "))
        .or_else(|| lowered.strip_prefix("describe the "))
        .or_else(|| lowered.strip_prefix("describe "))
        .or_else(|| lowered.strip_prefix("give me an overview of "))
        .or_else(|| lowered.strip_prefix("give me a summary of "))
        .or_else(|| lowered.strip_prefix("overview of "))
        .or_else(|| lowered.strip_prefix("summary of "))
        .or_else(|| lowered.strip_prefix("list all "))
        .or_else(|| lowered.strip_prefix("list "))
        .or_else(|| lowered.strip_prefix("show me "))
        .or_else(|| lowered.strip_prefix("show "))
        .or_else(|| lowered.strip_prefix("what is the "))
        .or_else(|| lowered.strip_prefix("what is "))
        .or_else(|| lowered.strip_prefix("what are the "))
        .or_else(|| lowered.strip_prefix("what are "))
        .or_else(|| lowered.strip_prefix("who is the "))
        .or_else(|| lowered.strip_prefix("who is "))
        .unwrap_or(&lowered)
        .trim();

    if candidate.is_empty() {
        raw_input.to_string()
    } else {
        canonicalize_claim_text(candidate)
    }
}

fn canonicalize_claim_text(candidate: &str) -> String {
    let candidate = candidate.trim();
    if candidate.is_empty() {
        return String::new();
    }

    if let Some(subject) = candidate.strip_suffix(" is current") {
        return format!("current {}", strip_leading_article(subject));
    }
    if let Some(subject) = candidate.strip_suffix(" is currently") {
        return format!("current {}", strip_leading_article(subject));
    }
    if let Some(subject) = candidate.strip_suffix(" currently") {
        return format!("current {}", strip_leading_article(subject));
    }

    strip_leading_article(candidate).to_string()
}

fn strip_leading_article(text: &str) -> &str {
    let trimmed = text.trim();
    trimmed
        .strip_prefix("the ")
        .or_else(|| trimmed.strip_prefix("a "))
        .or_else(|| trimmed.strip_prefix("an "))
        .unwrap_or(trimmed)
}
