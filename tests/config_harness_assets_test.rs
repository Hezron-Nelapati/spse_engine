use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[test]
fn controlled_dataset_has_expected_shape() {
    let raw = fs::read_to_string("test_data/controlled_story_dataset.json").unwrap();
    let dataset: JsonValue = serde_json::from_str(&raw).unwrap();

    assert_eq!(dataset["schema_version"].as_u64().unwrap(), 1);
    assert_eq!(dataset["story_word_count"].as_u64().unwrap(), 512);

    let questions = dataset["questions"].as_array().unwrap();
    assert_eq!(questions.len(), 20);

    let mut counts = BTreeMap::new();
    for question in questions {
        let category = question["category"].as_str().unwrap().to_string();
        *counts.entry(category).or_insert(0usize) += 1;
        assert!(question["expected"]["exact_match"].as_str().unwrap().len() >= 2);
        assert!(
            question["expected"]["semantic_allowlist"]
                .as_array()
                .unwrap()
                .len()
                >= 1
        );
    }

    assert_eq!(counts.get("retrieval_negative"), Some(&5));
    assert_eq!(counts.get("retrieval_positive"), Some(&5));
    assert_eq!(counts.get("reasoning"), Some(&5));
    assert_eq!(counts.get("uncertainty"), Some(&5));
}

#[test]
fn config_profile_manifest_references_valid_overrides() {
    let raw = fs::read_to_string("config/profiles.json").unwrap();
    let manifest: JsonValue = serde_json::from_str(&raw).unwrap();
    let profiles = manifest["profiles"].as_array().unwrap();
    assert!(profiles.len() >= 10);

    for profile in profiles {
        let path = profile["config_path"].as_str().unwrap();
        assert!(Path::new(path).exists(), "missing config profile: {path}");
        let override_raw = fs::read_to_string(path).unwrap();
        let _: YamlValue = serde_yaml::from_str(&override_raw).unwrap();
    }
}

#[test]
fn layer_20_schema_is_valid_json() {
    let raw = fs::read_to_string("config/layer_20_schema.json").unwrap();
    let schema: JsonValue = serde_json::from_str(&raw).unwrap();
    assert_eq!(
        schema["$id"].as_str().unwrap(),
        "https://spse.local/schemas/layer_20_schema.json"
    );
    assert_eq!(schema["type"].as_str().unwrap(), "object");
}
