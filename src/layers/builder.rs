use crate::config::{GovernanceConfig, UnitBuilderConfig};
use crate::memory::store::MemorySnapshot;
use crate::types::{
    ActivatedUnit, BuildOutput, DatabaseHealthMetrics, DatabaseMaturityStage, InputPacket,
    UnitLevel,
};
use std::collections::{HashMap, VecDeque};

pub struct UnitBuilder;

#[derive(Debug, Clone, Copy)]
pub struct DiscoveryThresholds {
    min_frequency: u64,
    min_utility_threshold: f32,
}

impl UnitBuilder {
    /// Static unit building for parallel training batches (no Engine/governance state needed).
    pub fn build_units_static(packet: &InputPacket, config: &UnitBuilderConfig) -> BuildOutput {
        Self::ingest_with_config(packet, config)
    }

    pub fn ingest(packet: &InputPacket) -> BuildOutput {
        Self::ingest_with_config(packet, &UnitBuilderConfig::default())
    }

    pub fn ingest_with_config(packet: &InputPacket, config: &UnitBuilderConfig) -> BuildOutput {
        Self::ingest_with_thresholds(
            packet,
            DiscoveryThresholds::default(packet, config),
            config,
            None,
        )
    }

    pub fn ingest_with_governance(
        packet: &InputPacket,
        config: &UnitBuilderConfig,
        governance: &GovernanceConfig,
        database_health: &DatabaseHealthMetrics,
    ) -> BuildOutput {
        Self::ingest_with_governance_snapshot(packet, config, governance, database_health, None)
    }

    pub fn ingest_with_governance_snapshot(
        packet: &InputPacket,
        config: &UnitBuilderConfig,
        governance: &GovernanceConfig,
        database_health: &DatabaseHealthMetrics,
        snapshot: Option<&MemorySnapshot>,
    ) -> BuildOutput {
        Self::ingest_with_thresholds(
            packet,
            DiscoveryThresholds::adaptive(packet, config, governance, database_health),
            config,
            snapshot,
        )
    }

    fn ingest_with_thresholds(
        packet: &InputPacket,
        thresholds: DiscoveryThresholds,
        config: &UnitBuilderConfig,
        snapshot: Option<&MemorySnapshot>,
    ) -> BuildOutput {
        let bytes = packet.bytes.as_slice();
        let min_window = min_window_size(config);
        if bytes.len() < min_window {
            return BuildOutput::default();
        }

        let mut activations = rolling_hash_units(bytes, thresholds, config, snapshot);
        activations.sort_by(|lhs, rhs| {
            rhs.utility_score
                .total_cmp(&lhs.utility_score)
                .then(rhs.frequency.cmp(&lhs.frequency))
                .then(rhs.content.len().cmp(&lhs.content.len()))
        });
        activations.truncate(config.max_activated_units.max(1));

        BuildOutput {
            new_units: activations.clone(),
            activated_units: activations,
        }
    }
}

impl DiscoveryThresholds {
    fn default(packet: &InputPacket, config: &UnitBuilderConfig) -> Self {
        Self {
            min_frequency: config.min_frequency_threshold.max(1),
            min_utility_threshold: if packet.training_mode { 0.06 } else { 0.08 },
        }
    }

    fn adaptive(
        packet: &InputPacket,
        config: &UnitBuilderConfig,
        governance: &GovernanceConfig,
        database_health: &DatabaseHealthMetrics,
    ) -> Self {
        let chunk_scale = ((packet.bytes.len() / 4_096).min(2)) as u64;
        let base_frequency = match database_health.maturity_stage {
            DatabaseMaturityStage::ColdStart => governance.cold_start_discovery_frequency,
            DatabaseMaturityStage::Growth => governance.growth_discovery_frequency,
            DatabaseMaturityStage::Stable => governance.stable_discovery_frequency,
        };
        let utility_threshold = match database_health.maturity_stage {
            DatabaseMaturityStage::ColdStart => governance.cold_start_discovery_utility_threshold,
            DatabaseMaturityStage::Growth => governance.growth_discovery_utility_threshold,
            DatabaseMaturityStage::Stable => governance.stable_discovery_utility_threshold,
        };

        Self {
            min_frequency: if packet.training_mode {
                // In training mode, use the lower of base_frequency or min_threshold
                // to allow unit discovery even for short documents
                base_frequency
                    .min(config.min_frequency_threshold)
                    .saturating_add(chunk_scale)
                    .max(1)
            } else {
                base_frequency
                    .max(config.min_frequency_threshold)
                    .saturating_sub(1)
                    .max(1)
            },
            min_utility_threshold: if packet.training_mode {
                utility_threshold
            } else {
                (utility_threshold * 0.9).max(0.08)
            },
        }
    }
}

#[derive(Default)]
struct WindowStats {
    content: String,
    normalized: String,
    byte_len: usize,
    frequency: u64,
    full_token_boundary_hits: u64,
    edge_boundary_hits: u64,
}

fn rolling_hash_units(
    bytes: &[u8],
    thresholds: DiscoveryThresholds,
    config: &UnitBuilderConfig,
    snapshot: Option<&MemorySnapshot>,
) -> Vec<ActivatedUnit> {
    let mut windows: HashMap<(u64, usize, String), WindowStats> = HashMap::new();
    let boundary_index = TextBoundaryIndex::build(bytes);

    for width in rolling_window_sizes(config) {
        if bytes.len() < width {
            continue;
        }
        let mut hasher = RollingHasher::new(width, config.hash_base);
        for end in 0..bytes.len() {
            if let Some(hash) = hasher.push(bytes[end]) {
                let start = end + 1 - width;
                let end_exclusive = end + 1;
                let Some((window_start, window_end, window_text)) = boundary_index.recover_window(
                    bytes,
                    start,
                    end_exclusive,
                    config.utf8_recovery_min_bytes.max(1),
                ) else {
                    continue;
                };
                let Some((content, normalized)) = normalize_window(window_text, config) else {
                    continue;
                };
                let left_boundary =
                    window_start == 0 || !boundary_index.is_word_byte(window_start - 1);
                let right_boundary =
                    window_end >= bytes.len() || !boundary_index.is_word_byte(window_end);

                let key = (
                    stable_window_hash(window_text.as_bytes(), config.hash_base).unwrap_or(hash),
                    window_end - window_start,
                    normalized.clone(),
                );
                let entry = windows.entry(key).or_default();
                if entry.content.is_empty() {
                    entry.content = content.clone();
                    entry.normalized = normalized;
                    entry.byte_len = window_end - window_start;
                }
                entry.frequency += 1;
                if left_boundary || right_boundary {
                    entry.edge_boundary_hits += 1;
                }
                if left_boundary && right_boundary {
                    entry.full_token_boundary_hits += 1;
                }
            }
        }
    }

    let total_bytes = bytes.len().max(1) as f32;
    windows
        .into_values()
        .filter_map(|stats| {
            let globally_corroborated = is_globally_corroborated(&stats, snapshot, config);
            if should_reject_fragment(&stats, config) && !globally_corroborated {
                return None;
            }
            let utility = utility_for(&stats, total_bytes, config);
            let salience = salience_for(&stats, config);
            let confidence = confidence_for(&stats, config);
            (globally_corroborated
                || (stats.frequency >= thresholds.min_frequency
                    && utility >= thresholds.min_utility_threshold))
                .then_some(ActivatedUnit {
                    content: stats.content.clone(),
                    normalized: stats.normalized.clone(),
                    level: level_for(&stats),
                    utility_score: utility,
                    frequency: stats.frequency,
                    salience,
                    confidence,
                    context_hint: format!("rolling_hash_window_{}", stats.byte_len),
                })
        })
        .collect()
}

fn normalize_window(window: &str, config: &UnitBuilderConfig) -> Option<(String, String)> {
    let trimmed = window.trim();
    if trimmed.len() < min_window_size(config) {
        return None;
    }
    if !trimmed.chars().any(|ch| ch.is_alphanumeric()) {
        return None;
    }

    let condensed = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
    if condensed.len() < min_window_size(config) {
        return None;
    }

    let visible = condensed
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .count()
        .max(1);
    let punctuation = condensed
        .chars()
        .filter(|ch| !ch.is_alphanumeric() && !ch.is_whitespace())
        .count();
    if (punctuation as f32 / visible as f32) > config.punctuation_ratio_limit {
        return None;
    }

    let normalized = condensed.to_lowercase();

    // Reject classification pattern markers (pattern:intent:tone:hash)
    // These are special units that should never merge with raw text units
    if normalized.starts_with("pattern:") {
        return None;
    }

    // Reject unicode escape sequences (\\uXXXX or uXXXX patterns from JSON)
    // These appear when JSON escape sequences leak into text processing
    if looks_like_unicode_escape(&normalized) {
        return None;
    }

    // Reject URL/path fragments and file extensions
    if looks_like_url_fragment(&normalized) {
        return None;
    }

    Some((condensed, normalized))
}

/// Check if the normalized string looks like a JSON unicode escape sequence
fn looks_like_unicode_escape(normalized: &str) -> bool {
    // Pattern: u followed by 4-6 hex digits (e.g., u0259, u00b0, u20ac)
    // Also catch escaped form: \\uXXXX
    let chars: Vec<char> = normalized.chars().collect();
    if chars.len() <= 12 {
        // Look for u[0-9a-f]{4,6} pattern
        for i in 0..chars.len().saturating_sub(4) {
            if chars[i] == 'u' {
                let hex_count = chars[i + 1..]
                    .iter()
                    .take_while(|c| c.is_ascii_hexdigit())
                    .count();
                if hex_count >= 4 && hex_count <= 6 {
                    // Check if this is most of the string (likely just the escape)
                    let rest_len = chars.len() - hex_count - 1;
                    if rest_len <= 3 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Check if the normalized string looks like a URL/path fragment
fn looks_like_url_fragment(normalized: &str) -> bool {
    // Reject fragments containing path separators that are too short
    if normalized.contains('/') && normalized.len() < 20 {
        return true;
    }

    // Reject fragments with colons that look like URL schemes (but allow time formats)
    if normalized.contains(':') && normalized.len() < 20 {
        // Allow common time formats like "12:30"
        let colon_pos = normalized.find(':').unwrap_or(0);
        let before = &normalized[..colon_pos];
        let after = &normalized[colon_pos + 1..];
        if !before.chars().all(|c| c.is_ascii_digit()) || !after.chars().all(|c| c.is_ascii_digit()) {
            return true;
        }
    }

    // Reject file extension fragments (e.g., ".png", "file.pdf", "-map.png")
    if normalized.len() < 20 && normalized.contains('.') {
        let common_extensions = [
            "png", "jpg", "jpeg", "gif", "pdf", "txt", "gz", "tar", "zip",
            "html", "htm", "css", "js", "json", "xml", "md", "rst",
            "py", "rs", "c", "cpp", "h", "java", "go", "ts",
        ];
        for ext in common_extensions {
            if normalized.ends_with(&format!(".{}", ext)) || normalized.ends_with(&format!(".{}\"", ext)) {
                return true;
            }
        }
    }

    false
}

fn utility_for(stats: &WindowStats, total_bytes: f32, config: &UnitBuilderConfig) -> f32 {
    let byte_gain =
        (stats.frequency.saturating_sub(1) as f32 * stats.byte_len as f32) / total_bytes;
    let compression_gain = byte_gain.clamp(0.0, config.utility_compression_gain_cap);
    let density = stats
        .content
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .count() as f32
        / stats.content.chars().count().max(1) as f32;
    let max_window = max_window_size(config).max(1) as f32;
    let span_bonus =
        (stats.byte_len as f32 / max_window).clamp(0.1, 1.0) * config.utility_span_weight;
    let frequency_bonus = (stats.frequency as f32).ln_1p() * config.utility_frequency_weight;
    let base_utility = (config.initial_utility_score * config.utility_base_scale).clamp(0.0, 1.0);
    let boundary_bonus = if stats.full_token_boundary_hits > 0 {
        config.utility_full_boundary_bonus
    } else if stats.edge_boundary_hits > 0 {
        config.utility_edge_boundary_bonus
    } else {
        config.utility_no_boundary_penalty
    };
    (base_utility
        + compression_gain
        + span_bonus
        + frequency_bonus
        + density * config.utility_density_weight
        + boundary_bonus)
        .clamp(0.05, 2.5)
}

fn salience_for(stats: &WindowStats, config: &UnitBuilderConfig) -> f32 {
    let digit_ratio = stats
        .content
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .count() as f32
        / stats.content.chars().count().max(1) as f32;
    let upper_ratio = stats.content.chars().filter(|ch| ch.is_uppercase()).count() as f32
        / stats.content.chars().count().max(1) as f32;
    let reuse = (stats.frequency as f32 / config.salience_reuse_divisor.max(0.1)).clamp(0.0, 0.6);
    (config.salience_base
        + digit_ratio * config.salience_digit_weight
        + upper_ratio * config.salience_upper_weight
        + reuse)
        .clamp(0.1, 1.0)
}

fn confidence_for(stats: &WindowStats, config: &UnitBuilderConfig) -> f32 {
    let reuse = ((stats.frequency as f32).ln_1p() / config.confidence_reuse_divisor.max(0.1))
        .clamp(0.0, config.confidence_reuse_cap);
    let length = (stats.byte_len as f32 / max_window_size(config).max(1) as f32).clamp(0.0, 1.0)
        * config.confidence_length_weight;
    let boundary_bonus = if stats.full_token_boundary_hits > 0 {
        config.confidence_full_boundary_bonus
    } else if stats.edge_boundary_hits > 0 {
        config.confidence_edge_boundary_bonus
    } else {
        config.confidence_no_boundary_penalty
    };
    (config.confidence_base + reuse + length + boundary_bonus).clamp(0.2, 0.95)
}

fn level_for(stats: &WindowStats) -> UnitLevel {
    let char_len = stats.content.chars().count();
    if stats.content.contains(char::is_whitespace) {
        return match char_len {
            0..=4 => UnitLevel::Subword,
            5..=24 => UnitLevel::Phrase,
            _ => UnitLevel::Pattern,
        };
    }

    if is_single_alphanumeric_token(&stats.content) {
        return match char_len {
            0..=2 => UnitLevel::Char,
            3..=4 => UnitLevel::Subword,
            _ if stats.full_token_boundary_hits > 0 => UnitLevel::Word,
            _ => UnitLevel::Subword,
        };
    }

    match char_len {
        0..=2 => UnitLevel::Char,
        3..=4 => UnitLevel::Subword,
        5..=8 => UnitLevel::Word,
        9..=24 => UnitLevel::Phrase,
        _ => UnitLevel::Pattern,
    }
}

fn should_reject_fragment(stats: &WindowStats, config: &UnitBuilderConfig) -> bool {
    if stats.full_token_boundary_hits > 0 {
        return false;
    }

    // Multi-word phrases with at least one edge boundary are valid
    // (e.g., "dynamic units" appearing mid-sentence has edge_boundary_hits > 0)
    if stats.content.contains(char::is_whitespace) {
        // Only reject if it has NO boundaries at all
        return stats.edge_boundary_hits == 0;
    }

    // Reject fragments with outer punctuation (leading/trailing non-alphanumeric chars)
    // This catches pollution like "claude-", "sudan_", "file.", "-Ude"
    if has_outer_punctuation(&stats.content) {
        return true;
    }

    is_single_alphanumeric_token(&stats.content)
        && stats.content.chars().count() >= config.min_fragment_length.max(1)
}

/// Check if content has leading or trailing punctuation that indicates pollution
fn has_outer_punctuation(content: &str) -> bool {
    if content.is_empty() {
        return false;
    }
    
    let chars: Vec<char> = content.chars().collect();
    let first = chars[0];
    let last = chars[chars.len() - 1];
    
    // Check for leading punctuation (but allow common prefixes like "$" for currency)
    let has_leading_punct = !first.is_alphanumeric() && first != '$' && first != '£' && first != '€';
    
    // Check for trailing punctuation (but allow common suffixes like "%" for percentages)
    let has_trailing_punct = !last.is_alphanumeric() && last != '%' && last != '\'';
    
    // Reject if has outer punctuation and is short (likely pollution)
    if (has_leading_punct || has_trailing_punct) && chars.len() <= 12 {
        return true;
    }
    
    false
}

fn is_single_alphanumeric_token(text: &str) -> bool {
    !text.is_empty()
        && !text.contains(char::is_whitespace)
        && text.chars().all(|ch| ch.is_alphanumeric())
}

fn is_globally_corroborated(
    stats: &WindowStats,
    snapshot: Option<&MemorySnapshot>,
    config: &UnitBuilderConfig,
) -> bool {
    snapshot
        .and_then(|memory| memory.get_by_normalized(&stats.normalized))
        .map(|unit| {
            unit.frequency >= config.global_corroboration_frequency_threshold
                && unit.utility_score >= config.global_corroboration_utility_threshold
                && unit.confidence >= config.global_corroboration_confidence_threshold
        })
        .unwrap_or(false)
}

fn stable_window_hash(bytes: &[u8], base: u64) -> Option<u64> {
    if bytes.is_empty() {
        return None;
    }
    Some(bytes.iter().fold(0u64, |hash, byte| {
        hash.wrapping_mul(base).wrapping_add(*byte as u64 + 1)
    }))
}

#[derive(Default)]
struct TextBoundaryIndex {
    word_byte_mask: Vec<bool>,
    char_boundaries: Vec<bool>,
}

impl TextBoundaryIndex {
    fn build(bytes: &[u8]) -> Self {
        let mut word_byte_mask = vec![false; bytes.len()];
        let mut char_boundaries = vec![false; bytes.len() + 1];
        char_boundaries[0] = true;
        char_boundaries[bytes.len()] = true;

        let Ok(text) = std::str::from_utf8(bytes) else {
            return Self {
                word_byte_mask,
                char_boundaries,
            };
        };

        let chars = text.char_indices().collect::<Vec<_>>();
        for (index, (start, ch)) in chars.iter().enumerate() {
            let end = chars
                .get(index + 1)
                .map(|(next, _)| *next)
                .unwrap_or(bytes.len());
            char_boundaries[*start] = true;
            char_boundaries[end] = true;

            let prev = index
                .checked_sub(1)
                .and_then(|idx| chars.get(idx))
                .map(|(_, c)| *c);
            let next = chars.get(index + 1).map(|(_, c)| *c);
            if is_word_char(*ch, prev, next) {
                for offset in *start..end {
                    word_byte_mask[offset] = true;
                }
            }
        }

        Self {
            word_byte_mask,
            char_boundaries,
        }
    }

    fn is_word_byte(&self, index: usize) -> bool {
        self.word_byte_mask.get(index).copied().unwrap_or(false)
    }

    fn recover_window<'a>(
        &self,
        bytes: &'a [u8],
        start: usize,
        end_exclusive: usize,
        min_len: usize,
    ) -> Option<(usize, usize, &'a str)> {
        let mut valid_start = start;
        while valid_start < end_exclusive
            && !self
                .char_boundaries
                .get(valid_start)
                .copied()
                .unwrap_or(false)
        {
            valid_start += 1;
        }

        let mut valid_end = end_exclusive;
        while valid_end > valid_start
            && !self
                .char_boundaries
                .get(valid_end)
                .copied()
                .unwrap_or(false)
        {
            valid_end -= 1;
        }

        if valid_end <= valid_start || valid_end - valid_start < min_len {
            return None;
        }

        let window = simdutf8::basic::from_utf8(&bytes[valid_start..valid_end]).ok()?;
        Some((valid_start, valid_end, window))
    }
}

fn is_word_char(ch: char, prev: Option<char>, next: Option<char>) -> bool {
    if ch.is_alphanumeric() || ch == '_' {
        return true;
    }

    matches!(ch, '\'' | '’' | '-' | '‐' | '‑' | '‒' | '–' | '—')
        && prev.map(|c| c.is_alphanumeric()).unwrap_or(false)
        && next.map(|c| c.is_alphanumeric()).unwrap_or(false)
}

struct RollingHasher {
    width: usize,
    hash: u64,
    factor: u64,
    base: u64,
    window: VecDeque<u8>,
}

impl RollingHasher {
    fn new(width: usize, base: u64) -> Self {
        let mut factor = 1u64;
        for _ in 1..width {
            factor = factor.wrapping_mul(base);
        }
        Self {
            width,
            hash: 0,
            factor,
            base,
            window: VecDeque::with_capacity(width),
        }
    }

    fn push(&mut self, byte: u8) -> Option<u64> {
        if self.window.len() == self.width {
            if let Some(removed) = self.window.pop_front() {
                self.hash = self
                    .hash
                    .wrapping_sub((removed as u64 + 1).wrapping_mul(self.factor));
            }
        }
        self.window.push_back(byte);
        self.hash = self
            .hash
            .wrapping_mul(self.base)
            .wrapping_add(byte as u64 + 1);
        (self.window.len() == self.width).then_some(self.hash)
    }
}

fn rolling_window_sizes(config: &UnitBuilderConfig) -> Vec<usize> {
    let mut sizes = config
        .rolling_hash_window_sizes
        .iter()
        .copied()
        .filter(|width| *width >= 2)
        .collect::<Vec<_>>();
    sizes.sort_unstable();
    sizes.dedup();
    if sizes.is_empty() {
        vec![2]
    } else {
        sizes
    }
}

fn min_window_size(config: &UnitBuilderConfig) -> usize {
    rolling_window_sizes(config).into_iter().next().unwrap_or(2)
}

fn max_window_size(config: &UnitBuilderConfig) -> usize {
    rolling_window_sizes(config).into_iter().last().unwrap_or(8)
}

#[cfg(test)]
mod tests {
    use super::UnitBuilder;
    use crate::config::{GovernanceConfig, UnitBuilderConfig};
    use crate::layers::hierarchy::HierarchicalUnitOrganizer;
    use crate::layers::input;
    use crate::memory::store::MemoryStore;
    use crate::types::{DatabaseHealthMetrics, DatabaseMaturityStage, SourceKind};
    use uuid::Uuid;

    #[test]
    fn rolling_hash_discovery_prefers_reused_windows() {
        let packet = input::ingest_raw("reasoning reasoning reasoning anchors anchors", true);
        let output = UnitBuilder::ingest(&packet);

        assert!(!output.activated_units.is_empty());
        assert!(output.activated_units.iter().any(
            |unit| unit.frequency >= 2 && unit.context_hint.starts_with("rolling_hash_window_")
        ));
    }

    #[test]
    fn adaptive_thresholds_tighten_with_database_maturity() {
        let packet = input::ingest_raw(
            "therefore therefore because because reasoning reasoning",
            true,
        );
        let governance = GovernanceConfig::default();
        let cold = DatabaseHealthMetrics {
            total_units: 128,
            maturity_stage: DatabaseMaturityStage::ColdStart,
            ..DatabaseHealthMetrics::default()
        };
        let stable = DatabaseHealthMetrics {
            total_units: 20_000,
            maturity_stage: DatabaseMaturityStage::Stable,
            ..DatabaseHealthMetrics::default()
        };

        let builder = UnitBuilderConfig::default();
        let cold_output =
            UnitBuilder::ingest_with_governance(&packet, &builder, &governance, &cold);
        let stable_output =
            UnitBuilder::ingest_with_governance(&packet, &builder, &governance, &stable);

        assert!(cold_output.activated_units.len() >= stable_output.activated_units.len());
    }

    #[test]
    fn full_words_are_preferred_over_long_internal_fragments() {
        let packet = input::ingest_raw("Catholic Catholic Catholic", true);
        let output = UnitBuilder::ingest(&packet);

        assert!(output.activated_units.iter().any(
            |unit| unit.normalized == "catholic" && unit.level == crate::types::UnitLevel::Word
        ));
        assert!(!output.activated_units.iter().any(|unit| matches!(
            unit.normalized.as_str(),
            "cath" | "atholic" | "catholi" | "tholic"
        )));
    }

    #[test]
    fn multiword_fragments_require_clean_token_boundaries() {
        let packet = input::ingest_raw("healthy Eating her team healthy Eating her team", true);
        let output = UnitBuilder::ingest(&packet);

        assert!(output
            .activated_units
            .iter()
            .any(|unit| unit.normalized == "her team"));
        assert!(!output.activated_units.iter().any(|unit| matches!(
            unit.normalized.as_str(),
            "hy eati" | "althy e" | "lthy ea" | "y eatin" | "thy eat"
        )));
    }

    #[test]
    fn unicode_boundaries_keep_contractions_and_hyphenated_words() {
        let packet = input::ingest_raw("don't don't re-use re-use café café", true);
        let output = UnitBuilder::ingest(&packet);

        assert!(output
            .activated_units
            .iter()
            .any(|unit| unit.normalized == "don't"));
        assert!(output
            .activated_units
            .iter()
            .any(|unit| unit.normalized == "re-use"));
        assert!(output
            .activated_units
            .iter()
            .any(|unit| unit.normalized == "café"));
    }

    #[test]
    fn global_corroboration_can_restore_known_units() {
        let db_path =
            std::env::temp_dir().join(format!("spse_global_builder_{}.db", Uuid::new_v4()));
        let mut store = MemoryStore::new(db_path.to_str().expect("db path"));
        let mut config = UnitBuilderConfig::default();
        config.global_corroboration_frequency_threshold = 1;
        config.global_corroboration_utility_threshold = 0.0;
        config.global_corroboration_confidence_threshold = 0.0;
        let known_packet =
            input::ingest_raw("atholic atholic atholic atholic atholic atholic", true);
        let known_output = UnitBuilder::ingest_with_config(&known_packet, &config);
        let known_hierarchy = HierarchicalUnitOrganizer::organize(&known_output, &config);
        store.ingest_hierarchy(&known_hierarchy, SourceKind::UserInput, "known_fragment");
        let snapshot = store.snapshot();

        let packet = input::ingest_raw("Catholic", true);
        let governance = GovernanceConfig::default();
        let database_health = DatabaseHealthMetrics {
            total_units: 256,
            maturity_stage: DatabaseMaturityStage::ColdStart,
            ..DatabaseHealthMetrics::default()
        };
        let output = UnitBuilder::ingest_with_governance_snapshot(
            &packet,
            &config,
            &governance,
            &database_health,
            Some(&snapshot),
        );

        assert!(output
            .activated_units
            .iter()
            .any(|unit| unit.normalized == "atholic"));
    }
}
