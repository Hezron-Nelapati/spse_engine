use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Default)]
pub struct BloomStats {
    pub queries: u64,
    pub maybe_hits: u64,
    pub false_positives: u64,
}

#[derive(Debug, Clone)]
pub struct UnitBloomFilter {
    bits: Vec<u64>,
    bit_count: usize,
    hash_functions: u64,
    stats: BloomStats,
}

impl UnitBloomFilter {
    pub fn new(expected_items: usize) -> Self {
        let expected_items = expected_items.max(1);
        let bit_count = (expected_items * 10).next_power_of_two().max(1024);
        let word_count = bit_count.div_ceil(64);
        Self {
            bits: vec![0; word_count],
            bit_count,
            hash_functions: 3,
            stats: BloomStats::default(),
        }
    }

    pub fn contains(&mut self, key: &str) -> bool {
        self.stats.queries += 1;
        let result = self.bit_positions(key).all(|bit| self.is_set(bit));
        if result {
            self.stats.maybe_hits += 1;
        }
        result
    }

    pub fn insert(&mut self, key: &str) {
        let positions = self.bit_positions(key).collect::<Vec<_>>();
        for bit in positions {
            self.set(bit);
        }
    }

    pub fn record_false_positive(&mut self) {
        self.stats.false_positives += 1;
    }

    pub fn rebuild<'a>(&mut self, keys: impl Iterator<Item = &'a str>) {
        self.bits.fill(0);
        self.stats = BloomStats::default();
        for key in keys {
            self.insert(key);
        }
    }

    pub fn stats(&self) -> BloomStats {
        self.stats.clone()
    }

    fn bit_positions<'a>(&'a self, key: &'a str) -> impl Iterator<Item = usize> + 'a {
        let primary = stable_hash(&(0u64, key));
        let secondary = stable_hash(&(1u64, key)).max(1);
        (0..self.hash_functions).map(move |index| {
            ((primary.wrapping_add(index.wrapping_mul(secondary))) % self.bit_count as u64) as usize
        })
    }

    fn is_set(&self, bit: usize) -> bool {
        let word = bit / 64;
        let offset = bit % 64;
        self.bits
            .get(word)
            .map(|value| (value & (1u64 << offset)) != 0)
            .unwrap_or(false)
    }

    fn set(&mut self, bit: usize) {
        let word = bit / 64;
        let offset = bit % 64;
        if let Some(value) = self.bits.get_mut(word) {
            *value |= 1u64 << offset;
        }
    }
}

fn stable_hash(value: &(u64, &str)) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::UnitBloomFilter;

    #[test]
    fn bloom_filter_tracks_inserted_keys() {
        let mut filter = UnitBloomFilter::new(128);
        filter.insert("reasoning");
        filter.insert("wikidata");

        assert!(filter.contains("reasoning"));
        assert!(filter.contains("wikidata"));
        assert!(!filter.contains("totally_missing_key_that_should_not_exist"));
    }
}
