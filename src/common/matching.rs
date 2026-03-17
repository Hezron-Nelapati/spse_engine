//! Keyword Matching Utilities
//!
//! Provides efficient keyword matching with caching for use across layers.

use once_cell::sync::Lazy;
use std::collections::HashSet;

/// Keyword matcher with caching support
pub struct KeywordMatcher;

impl KeywordMatcher {
    /// Check if any keyword exists in text (case-insensitive)
    #[inline]
    pub fn contains_any(text_lower: &str, keywords: &[&str]) -> bool {
        keywords.iter().any(|kw| text_lower.contains(kw))
    }

    /// Check if all keywords exist in text (case-insensitive)
    #[inline]
    pub fn contains_all(text_lower: &str, keywords: &[&str]) -> bool {
        keywords.iter().all(|kw| text_lower.contains(kw))
    }

    /// Count matching keywords in text
    pub fn count_matches(text_lower: &str, keywords: &[&str]) -> usize {
        keywords
            .iter()
            .filter(|kw| text_lower.contains(*kw))
            .count()
    }

    /// Score based on keyword matches with per-keyword increment
    pub fn score_matches(text_lower: &str, keywords: &[&str], increment: f32, max: f32) -> f32 {
        let count = Self::count_matches(text_lower, keywords) as f32;
        (count * increment).min(max)
    }

    /// Extract matching keywords from text
    pub fn extract_matches<'a>(text_lower: &str, keywords: &'a [&str]) -> Vec<&'a str> {
        keywords
            .iter()
            .filter(|kw| text_lower.contains(*kw))
            .copied()
            .collect()
    }

    /// Check if text matches any keyword from a cached set
    pub fn matches_cached(text_lower: &str, cached_keywords: &Lazy<HashSet<&'static str>>) -> bool {
        cached_keywords.iter().any(|kw| text_lower.contains(kw))
    }

    /// Build a cached keyword set for O(1) lookup
    pub fn build_keyword_set(keywords: &[&'static str]) -> HashSet<&'static str> {
        keywords.iter().copied().collect()
    }
}
