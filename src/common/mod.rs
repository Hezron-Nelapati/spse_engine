//! Common Utilities Module
//!
//! Provides shared functionality across engine layers to reduce code duplication
//! and ensure consistent behavior.

pub mod scoring;
pub mod matching;
pub mod selection;
pub mod similarity;
pub mod dedup;

pub use scoring::ScoreUtils;
pub use matching::KeywordMatcher;
pub use selection::TopKSelector;
pub use similarity::SimilarityUtils;
pub use dedup::DedupUtils;
