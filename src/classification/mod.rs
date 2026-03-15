//! Classification module for calculation-based intent, tone, and resolver mode inference.
//!
//! This module implements a hybrid retrieval-augmented classification system that:
//! - Computes lightweight feature signatures from input text
//! - Queries nearby patterns using Layer 6 Spatial Index
//! - Aggregates votes from retrieved patterns
//! - Learns from labeled seed data via Layer 18 feedback
//!
//! Architecture alignment:
//! - Layer 2: ClassificationSignature as specialized Unit feature
//! - Layer 4: ClassificationPattern storage in Intent memory channel
//! - Layer 6: Spatial query for O(log N) pattern retrieval
//! - Layer 14: Similarity scoring and candidate aggregation
//! - Layer 18: Feedback-driven learning and spatial adjustment

mod signature;
mod pattern;
mod calculator;
pub mod trainer;

pub use signature::{ClassificationSignature, SemanticHasher};
pub use pattern::ClassificationPattern;
pub use calculator::ClassificationCalculator;
pub use trainer::{ClassificationTrainer, TrainingOutcome, IterationReport, FinalReport, LabeledDialogue, LabeledTurn, DialogueMetadata, ExpectedUnitCount, MemoryTarget};

// Re-export types from crate::types for convenience
pub use crate::types::{ClassificationResult, CalculationMethod};
