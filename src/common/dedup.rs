//! Deduplication Utilities
//!
//! Common deduplication patterns used across layers.

use std::collections::HashSet;
use std::hash::Hash;

/// Deduplication utilities
pub struct DedupUtils;

impl DedupUtils {
    /// Deduplicate items by identity (Hash + Eq)
    pub fn dedup_by_identity<T: Hash + Eq + Clone>(items: Vec<T>) -> Vec<T> {
        let mut seen = HashSet::new();
        items
            .into_iter()
            .filter(|item| seen.insert(item.clone()))
            .collect()
    }

    /// Deduplicate items in place by identity
    pub fn dedup_by_identity_in_place<T: Hash + Eq + Clone>(items: &mut Vec<T>) {
        let mut seen = HashSet::new();
        items.retain(|item| seen.insert(item.clone()));
    }

    /// Deduplicate items by extracted key
    pub fn dedup_by_key<T, K, F>(items: Vec<T>, key_fn: F) -> Vec<T>
    where
        K: Hash + Eq,
        F: Fn(&T) -> K,
    {
        let mut seen = HashSet::new();
        items
            .into_iter()
            .filter(|item| seen.insert(key_fn(item)))
            .collect()
    }

    /// Deduplicate items in place by extracted key
    pub fn dedup_by_key_in_place<T, K, F>(items: &mut Vec<T>, key_fn: F)
    where
        K: Hash + Eq,
        F: Fn(&T) -> K,
    {
        let mut seen = HashSet::new();
        items.retain(|item| seen.insert(key_fn(item)));
    }

    /// Deduplicate strings by lowercase value
    pub fn dedup_strings_lowercase(items: &mut Vec<String>) {
        let mut seen = HashSet::new();
        items.retain(|item| seen.insert(item.to_lowercase()));
    }

    /// Deduplicate by first non-empty field (e.g., URL or title)
    pub fn dedup_by_primary_field<T, F1, F2>(items: &mut Vec<T>, primary_fn: F1, fallback_fn: F2)
    where
        F1: Fn(&T) -> &str,
        F2: Fn(&T) -> &str,
    {
        let mut seen = HashSet::new();
        items.retain(|item| {
            let key = if primary_fn(item).is_empty() {
                fallback_fn(item).to_lowercase()
            } else {
                primary_fn(item).to_lowercase()
            };
            seen.insert(key)
        });
    }

    /// Count unique items by key
    pub fn count_unique<T, K, F>(items: &[T], key_fn: F) -> usize
    where
        K: Hash + Eq,
        F: Fn(&T) -> K,
    {
        let mut seen = HashSet::new();
        for item in items {
            seen.insert(key_fn(item));
        }
        seen.len()
    }

    /// Check if item would be a duplicate
    pub fn is_duplicate<T, K, F>(items: &[T], item: &T, key_fn: F) -> bool
    where
        K: Hash + Eq,
        F: Fn(&T) -> K,
    {
        let item_key = key_fn(item);
        items.iter().any(|existing| key_fn(existing) == item_key)
    }

    /// Merge two vectors, deduplicating by key
    pub fn merge_dedup<T, K, F>(a: Vec<T>, b: Vec<T>, key_fn: F) -> Vec<T>
    where
        K: Hash + Eq,
        F: Fn(&T) -> K + Copy,
    {
        let mut seen = HashSet::new();
        let mut result = Vec::new();

        for item in a.into_iter().chain(b.into_iter()) {
            if seen.insert(key_fn(&item)) {
                result.push(item);
            }
        }

        result
    }
}
