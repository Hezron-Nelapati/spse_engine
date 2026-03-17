// Predictive System Dataset Generator (Word Graph Training)
// Generates Q&A pairs for Word Graph edge formation and highway detection (§11.4)

use crate::seed::{CurriculumMetadata, QualityGates, TrainingExample};
use crate::types::MemoryChannel;
use rand::seq::SliceRandom;

pub struct PredictiveQAGenerator {
    curriculum_score_base: f32,
}

impl PredictiveQAGenerator {
    pub fn new() -> Self {
        Self {
            curriculum_score_base: 0.80,
        }
    }

    /// Generate 200K+ Q&A pairs for Word Graph edge formation.
    /// Answers contain natural word sequences to create edges between consecutive words.
    /// Strategy:
    /// - Diverse question types ensure broad context tagging (polysemy coverage)
    /// - Answers of 5-50 words, average 20
    /// - 15% pairs with compound nouns
    /// - 10% pairs with rare/custom words
    pub fn generate_full_dataset(&self) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let mut rng = rand::thread_rng();

        // Core Q&A pairs (75% = 150K examples)
        examples.extend(self.generate_core_qa(150000));

        // Compound noun pairs (15% = 30K examples)
        examples.extend(self.generate_compound_noun_qa(30000));

        // Rare/custom word pairs (10% = 20K examples)
        examples.extend(self.generate_rare_word_qa(20000));

        // Shuffle for training
        examples.shuffle(&mut rng);

        examples
    }

    fn generate_core_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let qa_templates = vec![
            // Geography
            ("What is the capital of {country}?", "{capital} is the capital of {country}, located in the {region} region.", vec![
                ("France", "Paris", "western European"),
                ("Germany", "Berlin", "central European"),
                ("Japan", "Tokyo", "eastern Asian"),
                ("Brazil", "Brasília", "South American"),
                ("Australia", "Canberra", "southeastern Australian"),
            ]),

            // Science
            ("How does {process} work?", "{process} is a biological process where {description}.", vec![
                ("photosynthesis", "plants convert sunlight into chemical energy using chlorophyll in their cells", "plant biology"),
                ("respiration", "cells break down glucose to release energy in the form of ATP", "cellular metabolism"),
                ("osmosis", "water molecules move across a semipermeable membrane from high to low concentration", "membrane transport"),
            ]),

            // Technology
            ("What is {technology}?", "{technology} is {definition} commonly used in {application}.", vec![
                ("artificial intelligence", "computer systems designed to perform tasks requiring human intelligence", "automation and decision-making"),
                ("blockchain", "a distributed ledger technology using cryptographic hashing", "cryptocurrency and supply chain tracking"),
                ("quantum computing", "computational systems using quantum-mechanical phenomena", "cryptography and drug discovery"),
            ]),

            // History
            ("When did {event} happen?", "{event} occurred in {year} when {details}.", vec![
                ("World War II end", "1945", "Allied forces achieved victory in Europe and the Pacific"),
                ("Moon landing", "1969", "Apollo 11 astronauts Neil Armstrong and Buzz Aldrin stepped onto the lunar surface"),
                ("Fall of Berlin Wall", "1989", "East and West Germans reunited after decades of separation"),
            ]),

            // General knowledge
            ("Why is {concept} important?", "{concept} is important because {reason}.", vec![
                ("biodiversity", "it maintains ecosystem stability and provides resources for human survival", "ecology"),
                ("vaccination", "it prevents infectious disease spread through herd immunity", "public health"),
                ("renewable energy", "it reduces carbon emissions and dependence on finite fossil fuels", "sustainability"),
            ]),
        ];

        let per_template = count / qa_templates.len();

        for (q_template, a_template, variants) in qa_templates {
            for variant in variants.iter().cycle().take(per_template) {
                let (placeholder_val, answer_val, extra_val) = match variant {
                    (a, b, c) => (*a, *b, *c),
                    _ => continue,
                };

                let question = q_template
                    .replace("{country}", placeholder_val)
                    .replace("{process}", placeholder_val)
                    .replace("{technology}", placeholder_val)
                    .replace("{event}", placeholder_val)
                    .replace("{concept}", placeholder_val);

                let answer = a_template
                    .replace("{capital}", answer_val)
                    .replace("{region}", extra_val)
                    .replace("{country}", placeholder_val)
                    .replace("{process}", placeholder_val)
                    .replace("{description}", answer_val)
                    .replace("{technology}", placeholder_val)
                    .replace("{definition}", answer_val)
                    .replace("{application}", extra_val)
                    .replace("{event}", placeholder_val)
                    .replace("{year}", answer_val)
                    .replace("{details}", extra_val)
                    .replace("{concept}", placeholder_val)
                    .replace("{reason}", answer_val);

                examples.push(TrainingExample {
                    question: question.clone(),
                    answer: answer.clone(),
                    context: Some(format!("word_graph:{}", placeholder_val)),
                    reasoning: None,
                    intent: Some("Question".to_string()),
                    entities: vec![placeholder_val.to_string()],
                    channels: vec![MemoryChannel::Main],
                    curriculum: CurriculumMetadata {
                        curriculum_score: (self.curriculum_score_base * 100.0) as i32,
                        phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                        target_memory: crate::types::MemoryType::Episodic,
                        memory_channels: vec![MemoryChannel::Main],
                        suggested_batch_size: 500,
                        max_chunk_chars: 8000,
                    },
                    quality_gates: QualityGates {
                        min_unit_discovery_efficiency: Some(0.75),
                        min_semantic_routing_accuracy: Some(0.70),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }

    fn generate_compound_noun_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Compound noun templates - these should be merged by L1 POS tagger
        let compound_templates = vec![
            ("What is {compound}?", "{compound} is {definition}.", vec![
                ("machine learning", "a subset of artificial intelligence focused on algorithms that improve through experience"),
                ("climate change", "long-term shifts in global temperature and weather patterns"),
                ("renewable energy", "energy from sources that naturally replenish like solar and wind"),
                ("quantum physics", "the branch of physics studying matter and energy at atomic scales"),
                ("social media", "digital platforms enabling user-generated content and networking"),
                ("data science", "interdisciplinary field using scientific methods to extract insights from data"),
                ("neural network", "computing system inspired by biological neural networks in brains"),
                ("supply chain", "network between company and suppliers producing and distributing products"),
            ]),
        ];

        let per_template = count / compound_templates.len();

        for (q_template, a_template, variants) in compound_templates {
            for variant in variants.iter().cycle().take(per_template) {
                let (compound, definition) = *variant;

                let question = q_template.replace("{compound}", compound);
                let answer = a_template
                    .replace("{compound}", compound)
                    .replace("{definition}", definition);

                examples.push(TrainingExample {
                    question: question.clone(),
                    answer: answer.clone(),
                    context: Some(format!("compound:{}", compound.replace(" ", "_"))),
                    reasoning: None,
                    intent: Some("Question".to_string()),
                    entities: vec![compound.to_string()],
                    channels: vec![MemoryChannel::Main],
                    curriculum: CurriculumMetadata {
                        curriculum_score: ((self.curriculum_score_base + 0.05) * 100.0) as i32,
                        phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                        target_memory: crate::types::MemoryType::Episodic,
                        memory_channels: vec![MemoryChannel::Main],
                        suggested_batch_size: 500,
                        max_chunk_chars: 8000,
                    },
                    quality_gates: QualityGates {
                        min_unit_discovery_efficiency: Some(0.75),
                        min_semantic_routing_accuracy: Some(0.70),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }

    fn generate_rare_word_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Rare/custom words for runtime vocabulary growth testing
        let rare_word_templates = vec![
            ("What is {rare_term}?", "{rare_term} refers to {definition}.", vec![
                ("serendipity", "the occurrence of events by chance in a happy or beneficial way"),
                ("ephemeral", "lasting for a very short time"),
                ("ubiquitous", "present or found everywhere"),
                ("paradigm", "a typical example or pattern of something"),
                ("juxtaposition", "the fact of two things being seen or placed close together with contrasting effect"),
                ("dichotomy", "a division or contrast between two things that are represented as being opposed"),
                ("symbiosis", "interaction between two different organisms living in close physical association"),
                ("catalyst", "substance that increases rate of chemical reaction without being consumed"),
            ]),
        ];

        let per_template = count / rare_word_templates.len();

        for (q_template, a_template, variants) in rare_word_templates {
            for variant in variants.iter().cycle().take(per_template) {
                let (rare_term, definition) = *variant;

                let question = q_template.replace("{rare_term}", rare_term);
                let answer = a_template
                    .replace("{rare_term}", rare_term)
                    .replace("{definition}", definition);

                examples.push(TrainingExample {
                    question,
                    answer,
                    context: Some(format!("rare:{}", rare_term)),
                    reasoning: None,
                    intent: Some("Question".to_string()),
                    entities: vec![rare_term.to_string()],
                    channels: vec![MemoryChannel::Main],
                    curriculum: CurriculumMetadata {
                        curriculum_score: ((self.curriculum_score_base - 0.05) * 100.0) as i32,
                        phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                        target_memory: crate::types::MemoryType::Episodic,
                        memory_channels: vec![MemoryChannel::Main],
                        suggested_batch_size: 500,
                        max_chunk_chars: 8000,
                    },
                    quality_gates: QualityGates {
                        min_unit_discovery_efficiency: Some(0.70),
                        min_semantic_routing_accuracy: Some(0.65),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }
}

impl Default for PredictiveQAGenerator {
    fn default() -> Self {
        Self::new()
    }
}
