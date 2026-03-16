//! Similarity Utilities
//!
//! Common text similarity and overlap calculations.

use std::collections::HashSet;

/// Similarity calculation utilities
pub struct SimilarityUtils;

impl SimilarityUtils {
    /// Calculate Jaccard similarity (word overlap) between two strings
    pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let words_a: HashSet<&str> = a_lower.split_whitespace().collect();
        let words_b: HashSet<&str> = b_lower.split_whitespace().collect();

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

    /// Calculate word overlap ratio (intersection / min length)
    pub fn word_overlap_ratio(a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        let words_a: HashSet<&str> = a_lower.split_whitespace().collect();
        let words_b: HashSet<&str> = b_lower.split_whitespace().collect();

        if words_a.is_empty() || words_b.is_empty() {
            return 0.0;
        }

        let intersection = words_a.intersection(&words_b).count();
        let min_len = words_a.len().min(words_b.len());

        intersection as f32 / min_len as f32
    }

    /// Check if two strings are semantically equivalent (simplified)
    pub fn semantically_equivalent(a: &str, b: &str) -> bool {
        let a_normalized = a.trim().to_lowercase();
        let b_normalized = b.trim().to_lowercase();

        // Direct match
        if a_normalized == b_normalized {
            return true;
        }

        // Token match
        let tokens_a: Vec<&str> = a_normalized.split_whitespace().collect();
        let tokens_b: Vec<&str> = b_normalized.split_whitespace().collect();

        if tokens_a.len() == tokens_b.len() {
            let all_match = tokens_a
                .iter()
                .zip(tokens_b.iter())
                .all(|(ta, tb)| ta == tb);
            if all_match {
                return true;
            }
        }

        false
    }

    /// Calculate containment ratio (how much of b is in a)
    pub fn containment_ratio(haystack: &str, needle: &str) -> f32 {
        let haystack_lower = haystack.to_lowercase();
        let needle_lower = needle.to_lowercase();
        let words_haystack: HashSet<&str> = haystack_lower.split_whitespace().collect();
        let words_needle: HashSet<&str> = needle_lower.split_whitespace().collect();

        if words_needle.is_empty() {
            return 1.0;
        }

        let contained = words_needle.intersection(&words_haystack).count();
        contained as f32 / words_needle.len() as f32
    }

    /// Check if needle is fully contained in haystack
    pub fn is_contained(haystack: &str, needle: &str) -> bool {
        let haystack_lower = haystack.to_lowercase();
        let needle_lower = needle.to_lowercase();

        haystack_lower.contains(&needle_lower)
    }

    /// Calculate exact match bonus for entity disambiguation
    pub fn exact_match_score(candidate: &str, query_terms: &[String]) -> f32 {
        if query_terms.is_empty() {
            return 0.0;
        }

        let candidate_terms: Vec<&str> = candidate.split_whitespace().collect();
        let candidate_set: HashSet<&str> = candidate_terms.iter().copied().collect();
        let query_set: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        // Exact match
        if candidate_terms.len() == query_terms.len() {
            let all_match = candidate_terms
                .iter()
                .zip(query_terms.iter())
                .all(|(c, q)| c.to_lowercase() == q.to_lowercase());
            if all_match {
                return 0.85;
            }
        }

        // Superset penalty (candidate has extra terms)
        let query_in_candidate = query_terms
            .iter()
            .all(|q| candidate.contains(&q.to_lowercase()));
        let candidate_has_extra = candidate_terms.len() > query_terms.len();

        if query_in_candidate && candidate_has_extra {
            return -0.25;
        }

        // Partial overlap
        let overlap = query_set.intersection(&candidate_set).count();
        let overlap_ratio = overlap as f32 / query_terms.len().max(1) as f32;

        if overlap_ratio >= 0.8 {
            overlap_ratio * 0.3
        } else if overlap_ratio >= 0.5 {
            overlap_ratio * 0.15
        } else {
            0.0
        }
    }

    /// Calculate n-gram similarity
    pub fn ngram_similarity(a: &str, b: &str, n: usize) -> f32 {
        let ngrams_a: HashSet<String> = Self::ngrams(&a.to_lowercase(), n);
        let ngrams_b: HashSet<String> = Self::ngrams(&b.to_lowercase(), n);

        if ngrams_a.is_empty() || ngrams_b.is_empty() {
            return 0.0;
        }

        let intersection = ngrams_a.intersection(&ngrams_b).count();
        let union = ngrams_a.union(&ngrams_b).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Generate n-grams from string
    fn ngrams(text: &str, n: usize) -> HashSet<String> {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return HashSet::new();
        }

        (0..=chars.len() - n)
            .map(|i| chars[i..i + n].iter().collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_similarity() {
        let sim = SimilarityUtils::jaccard_similarity("hello world", "hello there");
        assert!((sim - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_semantically_equivalent() {
        assert!(SimilarityUtils::semantically_equivalent(
            "hello world",
            "hello world"
        ));
        assert!(!SimilarityUtils::semantically_equivalent(
            "hello world",
            "hello there"
        ));
    }

    #[test]
    fn test_exact_match_score() {
        let query = vec!["donald".to_string(), "trump".to_string()];
        let score = SimilarityUtils::exact_match_score("donald trump", &query);
        assert!((score - 0.85).abs() < 0.01);
    }
}
