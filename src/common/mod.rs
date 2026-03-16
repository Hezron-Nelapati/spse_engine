//! Common Utilities Module
//!
//! Provides shared functionality across engine layers to reduce code duplication
//! and ensure consistent behavior.

pub mod dedup;
pub mod ingestion;
pub mod matching;
pub mod scoring;
pub mod selection;
pub mod similarity;

pub use dedup::DedupUtils;
pub use ingestion::{build_hierarchy, ingest_text, ingest_text_training, ingest_texts_batch};
pub use matching::KeywordMatcher;
pub use scoring::ScoreUtils;
pub use selection::TopKSelector;
pub use similarity::SimilarityUtils;
