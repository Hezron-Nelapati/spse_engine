use crate::types::{ContextMatrix, DecodedOutput, MergedState, ResolvedCandidate};

pub struct OutputDecoder;

impl OutputDecoder {
    pub fn decode(
        &self,
        prompt: &str,
        resolved: &ResolvedCandidate,
        context: &ContextMatrix,
        merged: &MergedState,
    ) -> DecodedOutput {
        if let Some(doc) = merged.evidence.documents.first() {
            let sentence = best_evidence_sentence(prompt, &resolved.content, merged)
                .unwrap_or_else(|| doc.normalized_content.clone());

            return DecodedOutput {
                text: finalize_answer(&sentence),
                grounded: true,
            };
        }

        if !resolved.content.is_empty() {
            return DecodedOutput {
                text: finalize_answer(&resolved.content),
                grounded: false,
            };
        }

        DecodedOutput {
            text: finalize_answer(&context.summary),
            grounded: false,
        }
    }
}

fn best_evidence_sentence(prompt: &str, resolved: &str, merged: &MergedState) -> Option<String> {
    let prompt_terms = normalized_terms(prompt);
    let resolved_terms = normalized_terms(resolved);
    let mut best: Option<(i32, String)> = None;

    for document in &merged.evidence.documents {
        for sentence in split_sentences(&document.normalized_content) {
            let terms = normalized_terms(&sentence);
            let overlap = overlap_score(&terms, &prompt_terms);
            let resolved_overlap = overlap_score(&terms, &resolved_terms);
            let trust_bonus = (document.trust_score * 10.0) as i32;
            let score =
                overlap * 4 + resolved_overlap * 3 + trust_bonus - sentence.len() as i32 / 80;
            match &best {
                Some((best_score, _)) if *best_score >= score => {}
                _ => best = Some((score, sentence)),
            }
        }
    }

    best.map(|(_, sentence)| sentence)
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(|ch| matches!(ch, '.' | '!' | '?' | '\n'))
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_string)
        .collect()
}

fn normalized_terms(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|token| token.trim_matches(|ch: char| !ch.is_alphanumeric()))
        .map(str::to_lowercase)
        .filter(|token| token.len() > 2)
        .filter(|token| {
            !matches!(
                token.as_str(),
                "the" | "and" | "for" | "with" | "this" | "that" | "from" | "about" | "what"
            )
        })
        .collect()
}

fn overlap_score(lhs: &[String], rhs: &[String]) -> i32 {
    lhs.iter().filter(|token| rhs.contains(token)).count() as i32
}

fn finalize_answer(text: &str) -> String {
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
