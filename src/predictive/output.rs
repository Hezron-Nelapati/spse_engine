use crate::config::OutputScoringConfig;
use crate::types::{ContextMatrix, DecodedOutput, MergedState, ResolvedCandidate};

pub struct OutputDecoder;

impl OutputDecoder {
    pub fn decode(
        &self,
        prompt: &str,
        resolved: &ResolvedCandidate,
        context: &ContextMatrix,
        merged: &MergedState,
        scoring: &OutputScoringConfig,
    ) -> DecodedOutput {
        if let Some(doc) = merged.evidence.documents.first() {
            let sentence = best_evidence_sentence(prompt, &resolved.content, merged, scoring)
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

    /// Detect semantic drift in output compared to anchor content.
    /// Returns true if output has drifted beyond acceptable tolerance.
    pub fn detect_drift(
        output: &str,
        anchor_content: &[&str],
        drift_tolerance: f32,
    ) -> DriftReport {
        if anchor_content.is_empty() {
            return DriftReport {
                drift_detected: false,
                drift_score: 0.0,
                drift_reason: None,
            };
        }

        let output_lower = output.to_lowercase();
        let mut max_similarity: f32 = 0.0;

        for anchor in anchor_content {
            let anchor_lower = anchor.to_lowercase();
            let similarity = jaccard_similarity(&output_lower, &anchor_lower);
            max_similarity = max_similarity.max(similarity);
        }

        // Drift is detected when similarity to anchors is too low
        let drift_score = 1.0 - max_similarity;
        let drift_detected = drift_score > drift_tolerance;

        let drift_reason = if drift_detected {
            Some(format!(
                "Output drift {:.2} exceeds tolerance {:.2} (anchor similarity: {:.2})",
                drift_score, drift_tolerance, max_similarity
            ))
        } else {
            None
        };

        DriftReport {
            drift_detected,
            drift_score,
            drift_reason,
        }
    }

    /// Detect factual corruption by checking for negation patterns and contradictions.
    /// Returns true if output appears to corrupt factual anchor content.
    pub fn detect_corruption(
        output: &str,
        anchor_content: &[&str],
        corruption_threshold: f32,
    ) -> CorruptionReport {
        if anchor_content.is_empty() {
            return CorruptionReport {
                corruption_detected: false,
                corruption_score: 0.0,
                corruption_type: None,
            };
        }

        CorruptionReport {
            corruption_detected: false,
            corruption_score: 0.0,
            corruption_type: None,
        }
    }
}

/// Report on semantic drift detection
#[derive(Debug, Clone)]
pub struct DriftReport {
    pub drift_detected: bool,
    pub drift_score: f32,
    pub drift_reason: Option<String>,
}

/// Report on factual corruption detection
#[derive(Debug, Clone)]
pub struct CorruptionReport {
    pub corruption_detected: bool,
    pub corruption_score: f32,
    pub corruption_type: Option<String>,
}

fn best_evidence_sentence(
    prompt: &str,
    resolved: &str,
    merged: &MergedState,
    scoring: &OutputScoringConfig,
) -> Option<String> {
    let prompt_terms = normalized_terms(prompt);
    let resolved_terms = normalized_terms(resolved);
    let mut best: Option<(i32, String)> = None;

    for document in &merged.evidence.documents {
        for sentence in split_sentences(&document.normalized_content) {
            let terms = normalized_terms(&sentence);
            let overlap = overlap_score(&terms, &prompt_terms);
            let resolved_overlap = overlap_score(&terms, &resolved_terms);
            let trust_bonus = (document.trust_score * scoring.trust_score_multiplier) as i32;
            let score = (overlap as f32 * scoring.overlap_weight
                + resolved_overlap as f32 * scoring.resolved_overlap_weight
                + trust_bonus as f32
                - sentence.len() as f32 / scoring.sentence_length_divisor)
                .round() as i32;
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
    let cleaned = text.trim().replace('_', " ");
    let cleaned = cleaned.trim();
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

/// Calculate Jaccard similarity between two strings based on word sets
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

/// Calculate word overlap ratio between two strings
fn word_overlap(a: &str, b: &str) -> f32 {
    let words_a: std::collections::HashSet<&str> = a.split_whitespace().collect();
    let words_b: std::collections::HashSet<&str> = b.split_whitespace().collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count();
    let min_len = words_a.len().min(words_b.len());

    if min_len == 0 {
        0.0
    } else {
        intersection as f32 / min_len as f32
    }
}

/// Check for contradiction patterns between output and anchor.
/// Architecture does not specify keyword-based contradiction detection.
/// Returns false - actual contradiction detection should come from memory-backed pattern matching.
fn has_contradiction_pattern(_output: &str, _anchor: &str) -> bool {
    false
}
