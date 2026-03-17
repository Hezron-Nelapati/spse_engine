//! Classification System (L1, L2, L3, L9, L10, L19)
//!
//! Ingests raw text/sentences to identify intent, tone, uncertainty, and semantic
//! category. Acts as the gatekeeper that determines what the input means and
//! whether external help is needed.
//!
//! Core Mechanism: Nearest Centroid Classifier using 82-float POS-based feature
//! vectors (78 base + 4 semantic probe flags) with configurable feature weights.

pub mod builder;
pub mod hierarchy;
pub mod input;
pub mod intent;
pub mod query;
pub mod safety;

mod calculator;
mod pattern;
mod signature;
pub mod trainer;

pub use calculator::ClassificationCalculator;
pub use pattern::{parse_intent_kind, parse_tone_kind, ClassificationPattern};
pub use signature::{ClassificationSignature, SemanticHasher};
pub use trainer::{
    ClassificationTrainer, DialogueMetadata, ExpectedUnitCount, FinalReport, IterationReport,
    LabeledDialogue, LabeledTurn, MemoryTarget, TrainingOutcome,
};

pub use builder::UnitBuilder;
pub use hierarchy::HierarchicalUnitOrganizer;
pub use intent::IntentDetector;
pub use query::SafeQueryBuilder;
pub use safety::TrustSafetyValidator;

// Re-export types from crate::types for convenience
pub use crate::types::{CalculationMethod, ClassificationResult};
