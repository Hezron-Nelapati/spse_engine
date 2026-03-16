//! Training infrastructure for the SPS engine.
//!
//! Decoupled Training, Coupled Inference: each system is trained independently
//! with its own loss function, data pipeline, and optimization loop.

pub mod pipeline;

pub use pipeline::*;
