//! Selection Utilities
//!
//! Common candidate selection patterns used across layers.

use rand::Rng;
use std::cmp::Ordering;

/// Top-K selection utilities
pub struct TopKSelector;

impl TopKSelector {
    /// Select top-k items by score (descending)
    pub fn top_k<T, F>(items: &[T], k: usize, score_fn: F) -> Vec<&T>
    where
        F: Fn(&T) -> f32,
    {
        let mut indexed: Vec<(usize, f32)> = items
            .iter()
            .enumerate()
            .map(|(i, item)| (i, score_fn(item)))
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        indexed.into_iter()
            .take(k)
            .filter_map(|(i, _)| items.get(i))
            .collect()
    }

    /// Select top-k items by score with minimum threshold
    pub fn top_k_filtered<T, F>(items: &[T], k: usize, min_score: f32, score_fn: F) -> Vec<&T>
    where
        F: Fn(&T) -> f32,
    {
        let mut indexed: Vec<(usize, f32)> = items
            .iter()
            .enumerate()
            .map(|(i, item)| (i, score_fn(item)))
            .filter(|(_, score)| *score >= min_score)
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        indexed.into_iter()
            .take(k)
            .filter_map(|(i, _)| items.get(i))
            .collect()
    }

    /// Weighted random selection from top-k using softmax
    pub fn weighted_random<'a, T, F, R>(
        items: &'a [T],
        k: usize,
        temperature: f32,
        rng: &mut R,
        score_fn: F,
    ) -> Option<&'a T>
    where
        F: Fn(&T) -> f32,
        R: Rng,
    {
        let top_k: Vec<&T> = Self::top_k(items, k, &score_fn);
        if top_k.is_empty() {
            return None;
        }

        let scores: Vec<f32> = top_k.iter().map(|item| score_fn(item)).collect();
        let weights = Self::softmax_weights(&scores, temperature);

        let threshold: f32 = rng.gen();
        let mut cumulative = 0.0;

        for (item, weight) in top_k.iter().zip(weights.iter()) {
            cumulative += weight;
            if cumulative >= threshold {
                return Some(*item);
            }
        }

        top_k.last().copied()
    }

    /// Stochastic selection with probability floor
    pub fn stochastic_or_greedy<'a, T, F, R>(
        items: &'a [T],
        stochastic_prob: f32,
        temperature: f32,
        rng: &mut R,
        score_fn: F,
    ) -> Option<&'a T>
    where
        F: Fn(&T) -> f32 + Copy,
        R: Rng,
    {
        if items.is_empty() {
            return None;
        }

        let sample: f32 = rng.gen();

        if sample < stochastic_prob {
            // Stochastic selection from top-5
            Self::weighted_random(items, 5, temperature, rng, score_fn)
        } else {
            // Greedy: return highest scored
            items.iter().max_by(|a, b| {
                score_fn(a).partial_cmp(&score_fn(b)).unwrap_or(Ordering::Equal)
            })
        }
    }

    /// Calculate softmax weights
    fn softmax_weights(scores: &[f32], temperature: f32) -> Vec<f32> {
        if scores.is_empty() || temperature <= 0.0 {
            return vec![1.0 / scores.len().max(1) as f32; scores.len()];
        }

        let exp_scores: Vec<f32> = scores.iter()
            .map(|&s| (s / temperature).exp())
            .collect();
        let sum: f32 = exp_scores.iter().sum();

        exp_scores.into_iter().map(|e| e / sum).collect()
    }

    /// Determine beam width based on temperature
    pub fn beam_width(temperature: f32, base: usize, max_width: usize) -> usize {
        (temperature * base as f32).clamp(3.0, max_width as f32) as usize
    }

    /// Select top-k with temperature-based k adjustment
    pub fn top_k_temperature<T, F>(items: &[T], temperature: f32, score_fn: F) -> Vec<&T>
    where
        F: Fn(&T) -> f32,
    {
        let k = if temperature <= 0.4 {
            1
        } else if temperature <= 1.0 {
            3
        } else if temperature <= 1.25 {
            5
        } else {
            7
        };

        Self::top_k(items, k, score_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k() {
        let items = vec![5.0f32, 3.0, 8.0, 1.0, 4.0];
        let top = TopKSelector::top_k(&items, 3, |x| *x);
        assert_eq!(top.len(), 3);
        assert_eq!(*top[0], 8.0);
    }

    #[test]
    fn test_beam_width() {
        assert_eq!(TopKSelector::beam_width(0.5, 8, 20), 4);
        assert_eq!(TopKSelector::beam_width(2.0, 8, 20), 16);
    }
}
