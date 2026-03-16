//! Scoring Utilities
//!
//! Common scoring operations used across layers.

use std::cmp::Ordering;

/// Common scoring utilities
pub struct ScoreUtils;

impl ScoreUtils {
    /// Clamp score to valid range [0.0, 1.0]
    #[inline]
    pub fn clamp_score(score: f32) -> f32 {
        score.clamp(0.0, 1.0)
    }

    /// Clamp score with custom min/max
    #[inline]
    pub fn clamp_score_range(score: f32, min: f32, max: f32) -> f32 {
        score.clamp(min, max)
    }

    /// Compare two floats for sorting (descending order)
    #[inline]
    pub fn compare_desc(a: f32, b: f32) -> Ordering {
        b.partial_cmp(&a).unwrap_or(Ordering::Equal)
    }

    /// Compare two floats for sorting (ascending order)
    #[inline]
    pub fn compare_asc(a: f32, b: f32) -> Ordering {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }

    /// Calculate mean of scores
    pub fn mean(scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        scores.iter().sum::<f32>() / scores.len() as f32
    }

    /// Calculate variance of scores
    pub fn variance(scores: &[f32]) -> f32 {
        if scores.is_empty() {
            return 0.0;
        }
        let mean = Self::mean(scores);
        scores
            .iter()
            .map(|&s| {
                let diff = s - mean;
                diff * diff
            })
            .sum::<f32>()
            / scores.len() as f32
    }

    /// Calculate standard deviation of scores
    pub fn std_dev(scores: &[f32]) -> f32 {
        Self::variance(scores).sqrt()
    }

    /// Softmax normalization for weighted selection
    pub fn softmax(scores: &[f32], temperature: f32) -> Vec<f32> {
        if scores.is_empty() || temperature <= 0.0 {
            return vec![1.0 / scores.len().max(1) as f32; scores.len()];
        }

        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s / temperature).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();

        exp_scores.into_iter().map(|e| e / sum).collect()
    }

    /// Accumulate score score with per-keyword increment
    pub fn accumulate_keyword_score(
        input: &str,
        keywords: &[&str],
        increment: f32,
        max: f32,
    ) -> f32 {
        let input_lower = input.to_lowercase();
        let mut score = 0.0;

        for keyword in keywords {
            if input_lower.contains(keyword) {
                score += increment;
            }
        }

        score.min(max)
    }

    /// Calculate confidence stats from scored items
    pub fn confidence_stats<T, F>(items: &[T], score_fn: F) -> (f32, f32, usize)
    where
        F: Fn(&T) -> f32,
    {
        if items.is_empty() {
            return (0.0, 0.0, 0);
        }

        let scores: Vec<f32> = items.iter().map(score_fn).collect();
        let mean = Self::mean(&scores);
        let disagreement = Self::std_dev(&scores).clamp(0.0, 1.0);

        (mean, disagreement, items.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_score() {
        assert_eq!(ScoreUtils::clamp_score(0.5), 0.5);
        assert_eq!(ScoreUtils::clamp_score(1.5), 1.0);
        assert_eq!(ScoreUtils::clamp_score(-0.5), 0.0);
    }

    #[test]
    fn test_mean() {
        assert_eq!(ScoreUtils::mean(&[0.5, 0.5]), 0.5);
        assert_eq!(ScoreUtils::mean(&[0.0, 1.0]), 0.5);
        assert_eq!(ScoreUtils::mean(&[]), 0.0);
    }

    #[test]
    fn test_softmax() {
        let result = ScoreUtils::softmax(&[1.0, 2.0, 3.0], 1.0);
        assert!(!result.is_empty());
        assert!((result.iter().sum::<f32>() - 1.0).abs() < 0.001);
    }
}
