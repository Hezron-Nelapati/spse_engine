use crate::config::FineResolverConfig;
use crate::types::{ResolvedCandidate, ResolverMode, ScoredCandidate};
use rand::seq::SliceRandom;

pub struct FineResolver;

impl FineResolver {
    pub fn select(
        scored: &[ScoredCandidate],
        mode: ResolverMode,
        used_escape: bool,
        config: &FineResolverConfig,
    ) -> Option<ResolvedCandidate> {
        let meaningful = scored
            .iter()
            .filter(|candidate| candidate.content.len() > 1)
            .collect::<Vec<_>>();
        let preferred = if meaningful.is_empty() {
            scored.iter().collect::<Vec<_>>()
        } else {
            meaningful
        }
        .into_iter()
        .filter(|candidate| candidate.score >= config.min_confidence_floor)
        .collect::<Vec<_>>();

        if preferred.is_empty() {
            return None;
        }

        match mode {
            ResolverMode::Deterministic => preferred.first().map(|candidate| ResolvedCandidate {
                unit_id: candidate.unit_id,
                content: candidate.content.clone(),
                score: candidate.score,
                mode,
                used_escape,
            }),
            ResolverMode::Balanced => {
                let top_k = if config.selection_temperature <= 0.4 {
                    1
                } else if config.selection_temperature <= 1.0 {
                    3
                } else {
                    5
                };
                let top = preferred.iter().take(top_k).copied().collect::<Vec<_>>();
                top.first().map(|candidate| ResolvedCandidate {
                    unit_id: candidate.unit_id,
                    content: candidate.content.clone(),
                    score: candidate.score,
                    mode,
                    used_escape,
                })
            }
            ResolverMode::Exploratory => {
                let top_k = if config.selection_temperature <= 0.7 {
                    3
                } else if config.selection_temperature <= 1.25 {
                    5
                } else {
                    7
                };
                let mut options = preferred.iter().take(top_k).copied().collect::<Vec<_>>();
                let mut rng = rand::thread_rng();
                options.shuffle(&mut rng);
                options.first().map(|candidate| ResolvedCandidate {
                    unit_id: candidate.unit_id,
                    content: candidate.content.clone(),
                    score: candidate.score,
                    mode,
                    used_escape,
                })
            }
        }
    }
}
