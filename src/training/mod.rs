//! Training infrastructure for the SPS engine.
//!
//! Decoupled Training, Coupled Inference: each system is trained independently
//! with its own loss function, data pipeline, and optimization loop.

pub mod consistency;
mod system_training;

pub use consistency::{
    apply_consistency_corrections, run_consistency_check, ConsistencyCorrection, ConsistencyReport,
    ConsistencyRule,
};
pub use system_training::{
    train_classification, train_full_pipeline, train_predictive, train_reasoning,
};
