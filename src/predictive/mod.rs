//! Predictive System (L5, L6, L15, L16, L17)
//!
//! Creates and navigates a spatial map of words/units. Concepts likely to appear
//! side-by-side in a sentence are physically nearer in 3D space, enabling efficient
//! local search for the next token/unit rather than a global softmax computation.

pub mod output;
pub mod resolver;
pub mod router;

pub use output::OutputDecoder;
pub use resolver::FineResolver;
pub use router::SemanticRouter;
