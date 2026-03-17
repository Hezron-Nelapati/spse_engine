// Reasoning System Dataset Generator
// Generates QA pairs with reasoning traces for training the Reasoning System (§11.3)

use crate::seed::{CurriculumMetadata, QualityGates, TrainingExample};
use crate::types::{MemoryChannel, ReasoningStep, ReasoningStepType, ReasoningTrace};

pub struct ReasoningDatasetGenerator {
    curriculum_score_base: f32,
}

impl ReasoningDatasetGenerator {
    pub fn new() -> Self {
        Self {
            curriculum_score_base: 0.75,
        }
    }

    /// Generate reasoning dataset with 50K+ QA pairs:
    /// - 60% single-hop QA (factual, definitional, simple extraction)
    /// - 25% multi-hop QA (compare, analyze, plan)
    /// - 15% adversarial (contradictory evidence, missing information)
    pub fn generate_full_dataset(&self) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Single-hop QA (60% = 30K examples)
        examples.extend(self.generate_single_hop_qa(30000));

        // Multi-hop QA (25% = 12.5K examples)
        examples.extend(self.generate_multi_hop_qa(12500));

        // Adversarial (15% = 7.5K examples)
        examples.extend(self.generate_adversarial_qa(7500));

        examples
    }

    fn generate_single_hop_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        let templates = vec![
            // Factual questions - capitals (50+ variants)
            (
                "What is the capital of {country}?",
                "{capital} is the capital of {country}.",
                vec![
                    "France:Paris",
                    "Germany:Berlin",
                    "Japan:Tokyo",
                    "Brazil:Brasília",
                    "Australia:Canberra",
                    "Canada:Ottawa",
                    "Mexico:Mexico City",
                    "Spain:Madrid",
                    "Italy:Rome",
                    "China:Beijing",
                    "India:New Delhi",
                    "Russia:Moscow",
                    "Egypt:Cairo",
                    "Argentina:Buenos Aires",
                    "Poland:Warsaw",
                    "Sweden:Stockholm",
                    "Norway:Oslo",
                    "Denmark:Copenhagen",
                    "Finland:Helsinki",
                    "Netherlands:Amsterdam",
                    "Belgium:Brussels",
                    "Switzerland:Bern",
                    "Austria:Vienna",
                    "Portugal:Lisbon",
                    "Greece:Athens",
                    "Turkey:Ankara",
                    "Iran:Tehran",
                    "Iraq:Baghdad",
                    "Israel:Jerusalem",
                    "Thailand:Bangkok",
                    "Vietnam:Hanoi",
                    "Indonesia:Jakarta",
                    "Philippines:Manila",
                    "South Korea:Seoul",
                    "Pakistan:Islamabad",
                    "Bangladesh:Dhaka",
                    "Nigeria:Abuja",
                    "Kenya:Nairobi",
                    "South Africa:Pretoria",
                    "Morocco:Rabat",
                    "Chile:Santiago",
                    "Peru:Lima",
                    "Colombia:Bogotá",
                    "Venezuela:Caracas",
                    "Ukraine:Kyiv",
                    "Czech Republic:Prague",
                    "Hungary:Budapest",
                    "Romania:Bucharest",
                    "Ireland:Dublin",
                    "New Zealand:Wellington",
                ],
            ),
            // Population questions (30+ variants)
            (
                "What is the population of {city}?",
                "The population of {city} is approximately {population}.",
                vec![
                    "Paris:2.1 million",
                    "Berlin:3.6 million",
                    "Tokyo:14 million",
                    "New York:8.3 million",
                    "London:9 million",
                    "Mumbai:20 million",
                    "Shanghai:27 million",
                    "São Paulo:12 million",
                    "Moscow:12 million",
                    "Cairo:10 million",
                    "Lagos:15 million",
                    "Istanbul:15 million",
                    "Beijing:21 million",
                    "Delhi:32 million",
                    "Los Angeles:4 million",
                    "Chicago:2.7 million",
                    "Toronto:3 million",
                    "Sydney:5 million",
                    "Melbourne:5 million",
                    "Singapore:5.7 million",
                    "Hong Kong:7.5 million",
                    "Bangkok:10 million",
                    "Seoul:10 million",
                    "Manila:14 million",
                    "Jakarta:11 million",
                    "Karachi:15 million",
                    "Dhaka:22 million",
                    "Lima:10 million",
                    "Bogotá:8 million",
                ],
            ),
            // Birth years (40+ variants)
            (
                "When was {person} born?",
                "{person} was born in {year}.",
                vec![
                    "Albert Einstein:1879",
                    "Marie Curie:1867",
                    "Isaac Newton:1643",
                    "Galileo Galilei:1564",
                    "Charles Darwin:1809",
                    "Nikola Tesla:1856",
                    "Thomas Edison:1847",
                    "Leonardo da Vinci:1452",
                    "William Shakespeare:1564",
                    "Wolfgang Mozart:1756",
                    "Ludwig Beethoven:1770",
                    "Johann Bach:1685",
                    "Abraham Lincoln:1809",
                    "George Washington:1732",
                    "Napoleon Bonaparte:1769",
                    "Mahatma Gandhi:1869",
                    "Martin Luther King:1929",
                    "Nelson Mandela:1918",
                    "Winston Churchill:1874",
                    "Franklin Roosevelt:1882",
                    "Ada Lovelace:1815",
                    "Alan Turing:1912",
                    "Grace Hopper:1906",
                    "Richard Feynman:1918",
                    "Stephen Hawking:1942",
                    "Carl Sagan:1934",
                    "Jane Goodall:1934",
                    "Rachel Carson:1907",
                    "Sigmund Freud:1856",
                    "Carl Jung:1875",
                    "Karl Marx:1818",
                    "Friedrich Nietzsche:1844",
                    "Plato:428 BC",
                    "Aristotle:384 BC",
                    "Socrates:470 BC",
                    "Confucius:551 BC",
                    "Cleopatra:69 BC",
                    "Julius Caesar:100 BC",
                    "Alexander the Great:356 BC",
                    "Genghis Khan:1162",
                ],
            ),
            // Definitional questions (30+ variants)
            (
                "What is {term}?",
                "{term} is {definition}.",
                vec![
                    "photosynthesis:the process by which plants convert light into energy",
                    "DNA:deoxyribonucleic acid, the molecule carrying genetic instructions",
                    "gravity:the force of attraction between objects with mass",
                    "evolution:the process of gradual change in species over generations",
                    "democracy:a system of government where power is held by the people",
                    "capitalism:an economic system based on private ownership and free markets",
                    "socialism:an economic system where means of production are publicly owned",
                    "algorithm:a step-by-step procedure for solving a problem",
                    "metabolism:the chemical processes that maintain life in organisms",
                    "ecosystem:a community of living organisms interacting with their environment",
                    "inflation:the rate at which prices increase over time",
                    "recession:a period of economic decline lasting several months",
                    "chromosome:a structure containing DNA and genetic information",
                    "mitochondria:the powerhouse of the cell that produces energy",
                    "neuron:a nerve cell that transmits electrical signals",
                    "vaccine:a substance that stimulates immunity against diseases",
                    "antibiotic:a medicine that kills or inhibits bacteria",
                    "protein:a molecule made of amino acids essential for life",
                    "enzyme:a biological catalyst that speeds up chemical reactions",
                    "hormone:a chemical messenger that regulates body functions",
                    "quantum:the smallest discrete unit of energy or matter",
                    "atom:the basic unit of matter composed of protons, neutrons, and electrons",
                    "molecule:two or more atoms bonded together chemically",
                    "electron:a negatively charged subatomic particle",
                    "proton:a positively charged particle in the atomic nucleus",
                    "neutron:a neutral particle in the atomic nucleus",
                    "ion:an atom with a net electric charge",
                    "isotope:atoms of the same element with different neutron counts",
                ],
            ),
            // System components (20+ variants)
            (
                "What are the main components of {system}?",
                "The main components of {system} are {components}.",
                vec![
                    "atom:protons, neutrons, and electrons",
                    "cell:nucleus, cytoplasm, and cell membrane",
                    "computer:CPU, memory, storage, and input/output devices",
                    "solar system:the sun, planets, moons, and asteroids",
                    "ecosystem:producers, consumers, and decomposers",
                    "blood:red cells, white cells, platelets, and plasma",
                    "nervous system:brain, spinal cord, and peripheral nerves",
                    "digestive system:mouth, esophagus, stomach, and intestines",
                    "respiratory system:lungs, trachea, and diaphragm",
                    "circulatory system:heart, arteries, veins, and capillaries",
                    "skeletal system:bones, cartilage, and ligaments",
                    "muscular system:skeletal, smooth, and cardiac muscles",
                    "immune system:white blood cells, antibodies, and lymph nodes",
                    "endocrine system:glands, hormones, and receptors",
                    "database:tables, indexes, queries, and transactions",
                    "web application:frontend, backend, database, and API",
                    "operating system:kernel, shell, file system, and drivers",
                    "economy:production, distribution, and consumption",
                    "government:executive, legislative, and judicial branches",
                    "DNA:nucleotides, sugar, phosphate, and bases",
                ],
            ),
        ];

        let per_template = count / templates.len();

        for (q_template, a_template, variants) in templates {
            for variant in variants.iter().cycle().take(per_template) {
                let parts: Vec<&str> = variant.split(':').collect();
                if parts.len() < 2 {
                    continue;
                }

                let question = q_template
                    .replace("{country}", parts[0])
                    .replace("{city}", parts[0])
                    .replace("{person}", parts[0])
                    .replace("{term}", parts[0])
                    .replace("{concept}", parts[0])
                    .replace("{system}", parts[0]);
                let answer = a_template
                    .replace("{capital}", parts[1])
                    .replace("{population}", parts[1])
                    .replace("{year}", parts[1])
                    .replace("{definition}", parts[1])
                    .replace("{components}", parts[1])
                    .replace("{country}", parts[0])
                    .replace("{city}", parts[0])
                    .replace("{person}", parts[0])
                    .replace("{concept}", parts[0])
                    .replace("{system}", parts[0]);

                let reasoning = Some(ReasoningTrace {
                    steps: vec![
                        ReasoningStep {
                            step_type: ReasoningStepType::Premise,
                            content: format!(
                                "Query asks for factual information about {}",
                                parts[0]
                            ),
                            anchor_step: false,
                            dependencies: vec![],
                            structure_hash: None,
                        },
                        ReasoningStep {
                            step_type: ReasoningStepType::Conclusion,
                            content: answer.clone(),
                            anchor_step: false,
                            dependencies: vec![],
                            structure_hash: None,
                        },
                    ],
                    reasoning_type: crate::types::ReasoningType::General,
                    confidence_trajectory: vec![0.70, 0.85],
                    entities: vec![parts[0].to_string()],
                    structure_hash: None,
                });

                examples.push(TrainingExample {
                    question: question.clone(),
                    answer: answer.clone(),
                    context: Some(format!("factual:{}", parts[0])),
                    reasoning,
                    intent: Some("Question".to_string()),
                    entities: vec![parts[0].to_string()],
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
                        min_unit_discovery_efficiency: Some(0.85),
                        min_semantic_routing_accuracy: Some(0.70),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }

    fn generate_multi_hop_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Multi-hop comparison examples
        let comparison_variants = vec![
            (
                "Paris",
                "Berlin",
                "Berlin (3.6M) is larger than Paris (2.1M).",
            ),
            ("Tokyo", "London", "Tokyo (14M) is larger than London (9M)."),
            (
                "Sydney",
                "Melbourne",
                "Melbourne (5M) is slightly larger than Sydney (4.6M).",
            ),
        ];

        // Multi-hop analysis examples
        let analysis_variants = vec![
            (
                "photosynthesis",
                "Photosynthesis converts light energy into chemical energy through chlorophyll.",
            ),
            (
                "digestion",
                "Digestion breaks down food into nutrients through enzymes and acids.",
            ),
            (
                "respiration",
                "Respiration converts glucose and oxygen into energy, water, and carbon dioxide.",
            ),
        ];

        let per_variant = count / (comparison_variants.len() + analysis_variants.len());

        // Generate comparison examples
        for (a_val, b_val, answer) in comparison_variants.iter().cycle().take(per_variant * 3) {
            let question = format!("Is {} bigger than {}?", a_val, b_val);

            let steps = vec![
                ReasoningStep {
                    step_type: ReasoningStepType::Premise,
                    content: format!("Query requires comparing {} and {}", a_val, b_val),
                    anchor_step: true,
                    dependencies: vec![],
                    structure_hash: None,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::Hypothesis,
                    content: format!("What is the population of {}?", a_val),
                    anchor_step: false,
                    dependencies: vec![0],
                    structure_hash: None,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::Hypothesis,
                    content: format!("What is the population of {}?", b_val),
                    anchor_step: false,
                    dependencies: vec![0],
                    structure_hash: None,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::Conclusion,
                    content: answer.to_string(),
                    anchor_step: true,
                    dependencies: vec![1, 2],
                    structure_hash: None,
                },
            ];

            let reasoning = Some(ReasoningTrace {
                steps,
                reasoning_type: crate::types::ReasoningType::Logical,
                confidence_trajectory: vec![0.50, 0.60, 0.70, 0.85],
                entities: vec![a_val.to_string(), b_val.to_string()],
                structure_hash: None,
            });

            examples.push(TrainingExample {
                question,
                answer: answer.to_string(),
                context: Some(format!("multi_hop:{}_{}", a_val, b_val)),
                reasoning,
                intent: Some("Compare".to_string()),
                entities: vec![a_val.to_string(), b_val.to_string()],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.10) * 100.0) as i32,
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

        // Generate analysis examples
        for (process, answer) in analysis_variants.iter().cycle().take(per_variant * 3) {
            let question = format!("How does {} work?", process);

            let steps = vec![
                ReasoningStep {
                    step_type: ReasoningStepType::Premise,
                    content: format!("Query asks for explanation of {}", process),
                    anchor_step: true,
                    dependencies: vec![],
                    structure_hash: None,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::Inference,
                    content: format!("Breaking down {} into components", process),
                    anchor_step: false,
                    dependencies: vec![0],
                    structure_hash: None,
                },
                ReasoningStep {
                    step_type: ReasoningStepType::Conclusion,
                    content: answer.to_string(),
                    anchor_step: true,
                    dependencies: vec![1],
                    structure_hash: None,
                },
            ];

            let reasoning = Some(ReasoningTrace {
                steps,
                reasoning_type: crate::types::ReasoningType::Explanatory,
                confidence_trajectory: vec![0.55, 0.70, 0.85],
                entities: vec![process.to_string()],
                structure_hash: None,
            });

            examples.push(TrainingExample {
                question,
                answer: answer.to_string(),
                context: Some(format!("multi_hop:{}", process)),
                reasoning,
                intent: Some("Explain".to_string()),
                entities: vec![process.to_string()],
                channels: vec![MemoryChannel::Main],
                curriculum: CurriculumMetadata {
                    curriculum_score: ((self.curriculum_score_base - 0.10) * 100.0) as i32,
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

        examples
    }

    fn generate_adversarial_qa(&self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        // Adversarial examples with contradictory evidence or missing information
        let adversarial_templates = vec![
            (
                "What is the population of {fictitious_city}?",
                "I don't have reliable information about the population of {fictitious_city}.",
                vec!["Atlantis", "El Dorado", "Shangri-La"],
            ),
            (
                "When was {impossible_event} invented?",
                "The premise of the question is flawed - {impossible_event} is not an invention.",
                vec!["time travel", "perpetual motion", "teleportation"],
            ),
        ];

        let per_template = count / adversarial_templates.len();

        for (q_template, a_template, variants) in adversarial_templates {
            for variant in variants.iter().cycle().take(per_template) {
                let question = q_template
                    .replace("{fictitious_city}", variant)
                    .replace("{impossible_event}", variant);
                let answer = a_template
                    .replace("{fictitious_city}", variant)
                    .replace("{impossible_event}", variant);

                let reasoning = Some(ReasoningTrace {
                    steps: vec![
                        ReasoningStep {
                            step_type: ReasoningStepType::Premise,
                            content: format!("Query asks about {}", variant),
                            anchor_step: false,
                            dependencies: vec![],
                            structure_hash: None,
                        },
                        ReasoningStep {
                            step_type: ReasoningStepType::Verification,
                            content: "No reliable evidence found in memory".to_string(),
                            anchor_step: false,
                            dependencies: vec![],
                            structure_hash: None,
                        },
                        ReasoningStep {
                            step_type: ReasoningStepType::Conclusion,
                            content: "Insufficient evidence to answer".to_string(),
                            anchor_step: false,
                            dependencies: vec![],
                            structure_hash: None,
                        },
                    ],
                    reasoning_type: crate::types::ReasoningType::Verification,
                    confidence_trajectory: vec![0.30, 0.15, 0.20],
                    entities: vec![],
                    structure_hash: None,
                });

                examples.push(TrainingExample {
                    question: question.clone(),
                    answer: answer.clone(),
                    context: Some(format!("adversarial:{}", variant)),
                    reasoning,
                    intent: Some("Question".to_string()),
                    entities: vec![variant.to_string()],
                    channels: vec![MemoryChannel::Main],
                    curriculum: CurriculumMetadata {
                        curriculum_score: ((self.curriculum_score_base - 0.20) * 100.0) as i32,
                        phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                        target_memory: crate::types::MemoryType::Episodic,
                        memory_channels: vec![MemoryChannel::Main],
                        suggested_batch_size: 500,
                        max_chunk_chars: 8000,
                    },
                    quality_gates: QualityGates {
                        min_unit_discovery_efficiency: Some(0.30),
                        min_semantic_routing_accuracy: Some(0.50),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }
}

impl Default for ReasoningDatasetGenerator {
    fn default() -> Self {
        Self::new()
    }
}
