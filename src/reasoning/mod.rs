//! Reasoning System (L4, L7, L8, L11, L12, L13, L14, L18, L21)
//!
//! Takes classified input and current context to reason on the response strategy.
//! Manages memory retrieval, evidence merging, conflict resolution, and the
//! logical assembly of the answer before final word selection.

pub mod context;
pub mod feedback;
pub mod merge;
pub mod retrieval;
pub mod search;

pub use context::ContextManager;
pub use feedback::FeedbackController;
pub use merge::EvidenceMerger;
pub use search::{CandidateScorer, score_candidates_gpu_accelerated, top_unit_ids};
