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

/// Pre-defined keyword categories for common use
pub mod categories {
    use once_cell::sync::Lazy;

    /// Urgency indicators
    pub static URGENCY_KEYWORDS: Lazy<[&str; 11]> = Lazy::new(|| {
        [
            "urgent",
            "emergency",
            "asap",
            "immediately",
            "now",
            "critical",
            "important",
            "quickly",
            "hurry",
            "fast",
            "right now",
        ]
    });

    /// Sadness/empathy indicators
    pub static SADNESS_KEYWORDS: Lazy<[&str; 17]> = Lazy::new(|| {
        [
            "sad",
            "upset",
            "worried",
            "anxious",
            "depressed",
            "lonely",
            "hurt",
            "pain",
            "suffering",
            "struggling",
            "difficult",
            "hard",
            "lost",
            "grief",
            "cry",
            "tears",
            "hopeless",
        ]
    });

    /// Technical domain indicators
    pub static TECHNICAL_KEYWORDS: Lazy<[&str; 21]> = Lazy::new(|| {
        [
            "code",
            "function",
            "api",
            "algorithm",
            "debug",
            "error",
            "variable",
            "method",
            "class",
            "system",
            "architecture",
            "implementation",
            "database",
            "server",
            "client",
            "protocol",
            "interface",
            "module",
            "component",
            "service",
            "endpoint",
        ]
    });

    /// Casual tone indicators
    pub static CASUAL_KEYWORDS: Lazy<[&str; 14]> = Lazy::new(|| {
        [
            "hey", "yo", "sup", "cool", "awesome", "yeah", "nah", "gonna", "wanna", "kinda",
            "sorta", "lol", "haha", "ok",
        ]
    });

    /// Formal tone indicators
    pub static FORMAL_KEYWORDS: Lazy<[&str; 10]> = Lazy::new(|| {
        [
            "please",
            "thank you",
            "regarding",
            "sincerely",
            "respectfully",
            "would you",
            "could you",
            "may i",
            "i would like",
            "kindly",
        ]
    });

    /// Negation patterns for contradiction detection
    pub static NEGATION_PATTERNS: Lazy<[&str; 8]> = Lazy::new(|| {
        [
            "not ", "never ", "isn't ", "aren't ", "wasn't ", "weren't ", "doesn't ", "don't ",
        ]
    });

    /// Mathematical/identity patterns
    pub static MATH_PATTERNS: Lazy<[&str; 6]> = Lazy::new(|| ["=", "+", "-", "*", "/", " is "]);
}
