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

#[cfg(test)]
mod tests {
    use super::HierarchicalUnitOrganizer;
    use crate::config::UnitBuilderConfig;
    use crate::types::{ActivatedUnit, BuildOutput, UnitLevel};

    #[test]
    fn promotes_frequent_salient_phrases_to_entities() {
        let config = UnitBuilderConfig::default();
        let build_output = BuildOutput {
            activated_units: vec![ActivatedUnit {
                content: "New York".to_string(),
                normalized: "new york".to_string(),
                level: UnitLevel::Phrase,
                utility_score: 0.88,
                frequency: 9,
                salience: 0.84,
                confidence: 0.72,
                context_hint: "test".to_string(),
            }],
            new_units: Vec::new(),
        };

        let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &config);
        assert!(hierarchy.entities.iter().any(|entity| entity == "New York"));
    }

    #[test]
    fn does_not_promote_low_signal_phrases() {
        let config = UnitBuilderConfig::default();
        let build_output = BuildOutput {
            activated_units: vec![ActivatedUnit {
                content: "healthy eating".to_string(),
                normalized: "healthy eating".to_string(),
                level: UnitLevel::Phrase,
                utility_score: 0.52,
                frequency: 3,
                salience: 0.34,
                confidence: 0.41,
                context_hint: "test".to_string(),
            }],
            new_units: Vec::new(),
        };

        let hierarchy = HierarchicalUnitOrganizer::organize(&build_output, &config);
        assert!(hierarchy.entities.is_empty());
    }
}
