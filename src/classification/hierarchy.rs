use crate::config::UnitBuilderConfig;
use crate::types::{ActivatedUnit, BuildOutput, UnitHierarchy, UnitLevel};
use std::collections::BTreeMap;

pub struct HierarchicalUnitOrganizer;

impl HierarchicalUnitOrganizer {
    pub fn organize(build_output: &BuildOutput, config: &UnitBuilderConfig) -> UnitHierarchy {
        let mut levels: BTreeMap<String, Vec<ActivatedUnit>> = BTreeMap::new();
        let mut anchors = Vec::new();
        let mut entities = Vec::new();

        for activation in &build_output.activated_units {
            let key = level_key(activation.level);
            levels.entry(key).or_default().push(activation.clone());

            if is_anchor_candidate(activation) && !anchors.contains(&activation.content) {
                anchors.push(activation.content.clone());
            }
            if is_entity_candidate(activation, config) && !entities.contains(&activation.content) {
                entities.push(activation.content.clone());
            }
        }

        for values in levels.values_mut() {
            values.sort_by(|a, b| {
                b.utility_score
                    .partial_cmp(&a.utility_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            values.truncate(24);
        }

        UnitHierarchy {
            levels,
            anchors,
            entities,
        }
    }
}

fn level_key(level: UnitLevel) -> String {
    match level {
        UnitLevel::Char => "char",
        UnitLevel::Subword => "subword",
        UnitLevel::Word => "word",
        UnitLevel::Phrase => "phrase",
        UnitLevel::Pattern => "pattern",
    }
    .to_string()
}

fn is_anchor_candidate(activation: &ActivatedUnit) -> bool {
    activation.level == UnitLevel::Phrase
        || activation.salience >= 0.7
        || activation.content.chars().any(|ch| ch.is_ascii_digit())
}

fn is_entity_candidate(activation: &ActivatedUnit, config: &UnitBuilderConfig) -> bool {
    if activation.level == UnitLevel::Word
        && activation
            .content
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
    {
        return true;
    }

    activation.level == UnitLevel::Phrase
        && activation.frequency >= config.phrase_entity_promotion_frequency
        && activation.salience >= config.phrase_entity_promotion_salience
        && activation.confidence >= config.phrase_entity_promotion_confidence
        && looks_like_compound_entity(&activation.content)
}

fn looks_like_compound_entity(content: &str) -> bool {
    let tokens = content
        .split_whitespace()
        .filter(|token| token.chars().any(|ch| ch.is_alphanumeric()))
        .collect::<Vec<_>>();
    if tokens.len() < 2 {
        return false;
    }

    tokens
        .iter()
        .filter(|token| {
            token
                .chars()
                .next()
                .map(|ch| ch.is_uppercase() || ch.is_ascii_digit())
                .unwrap_or(false)
        })
        .count()
        >= 2
}
