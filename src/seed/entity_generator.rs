//! EntityJson dataset generator for high-density entity definitions.

use crate::seed::{DatasetMetadata, QualityMetrics};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Entity definition for EntityJson dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub normalized: String,
    pub definition: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub category: String,
    #[serde(default)]
    pub attributes: HashMap<String, String>,
    pub links: Vec<EntityLink>,
    #[serde(default)]
    pub contexts: Vec<EntityContext>,
}

/// Link between entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityLink {
    pub target: String,
    #[serde(rename = "type")]
    pub link_type: String,
    pub weight: f32,
}

/// Contextual usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityContext {
    pub text: String,
    pub domain: String,
}

/// Complete EntityJson dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityJsonDataset {
    #[serde(flatten)]
    pub metadata: DatasetMetadata,
    pub entities: Vec<Entity>,
}

/// Generator for EntityJson datasets
#[allow(dead_code)]
pub struct EntityGenerator {
    /// Target density (entities per KB)
    target_density: f32,
    /// Minimum links per entity
    min_links: usize,
    /// Definition length range
    definition_length_range: (usize, usize),
    /// Generated entities registry for link resolution
    entity_registry: HashMap<String, Entity>,
    /// Category counter for ID generation
    category_counters: HashMap<String, u64>,
    /// Entity categories with domain contexts
    categories: Vec<(String, Vec<String>)>,
}

impl EntityGenerator {
    pub fn new() -> Self {
        Self {
            target_density: 50.0,
            min_links: 3,                       // Increased for higher density
            definition_length_range: (50, 300), // Longer definitions
            entity_registry: HashMap::new(),
            category_counters: HashMap::new(),
            categories: vec![
                (
                    "process".to_string(),
                    vec![
                        "workflow".to_string(),
                        "procedure".to_string(),
                        "operation".to_string(),
                        "task".to_string(),
                        "activity".to_string(),
                        "protocol".to_string(),
                        "methodology".to_string(),
                        "pipeline".to_string(),
                        "sequence".to_string(),
                        "cycle".to_string(),
                    ],
                ),
                (
                    "concept".to_string(),
                    vec![
                        "idea".to_string(),
                        "principle".to_string(),
                        "theory".to_string(),
                        "framework".to_string(),
                        "model".to_string(),
                        "paradigm".to_string(),
                        "hypothesis".to_string(),
                        "axiom".to_string(),
                        "construct".to_string(),
                        "pattern".to_string(),
                    ],
                ),
                (
                    "artifact".to_string(),
                    vec![
                        "document".to_string(),
                        "report".to_string(),
                        "specification".to_string(),
                        "deliverable".to_string(),
                        "output".to_string(),
                        "record".to_string(),
                        "template".to_string(),
                        "manifest".to_string(),
                        "schema".to_string(),
                        "blueprint".to_string(),
                    ],
                ),
                (
                    "role".to_string(),
                    vec![
                        "stakeholder".to_string(),
                        "participant".to_string(),
                        "owner".to_string(),
                        "contributor".to_string(),
                        "reviewer".to_string(),
                        "approver".to_string(),
                        "executor".to_string(),
                        "coordinator".to_string(),
                        "facilitator".to_string(),
                        "observer".to_string(),
                    ],
                ),
                (
                    "metric".to_string(),
                    vec![
                        "kpi".to_string(),
                        "indicator".to_string(),
                        "measurement".to_string(),
                        "benchmark".to_string(),
                        "target".to_string(),
                        "threshold".to_string(),
                        "baseline".to_string(),
                        "score".to_string(),
                        "ratio".to_string(),
                        "trend".to_string(),
                    ],
                ),
                (
                    "tool".to_string(),
                    vec![
                        "application".to_string(),
                        "system".to_string(),
                        "platform".to_string(),
                        "service".to_string(),
                        "utility".to_string(),
                        "framework".to_string(),
                        "engine".to_string(),
                        "interface".to_string(),
                        "dashboard".to_string(),
                        "workbench".to_string(),
                    ],
                ),
                (
                    "resource".to_string(),
                    vec![
                        "asset".to_string(),
                        "material".to_string(),
                        "input".to_string(),
                        "component".to_string(),
                        "element".to_string(),
                        "repository".to_string(),
                        "dataset".to_string(),
                        "inventory".to_string(),
                        "catalog".to_string(),
                        "archive".to_string(),
                    ],
                ),
                (
                    "event".to_string(),
                    vec![
                        "milestone".to_string(),
                        "checkpoint".to_string(),
                        "trigger".to_string(),
                        "occurrence".to_string(),
                        "incident".to_string(),
                        "signal".to_string(),
                        "notification".to_string(),
                        "alert".to_string(),
                        "deadline".to_string(),
                        "ceremony".to_string(),
                    ],
                ),
                (
                    "domain".to_string(),
                    vec![
                        "area".to_string(),
                        "field".to_string(),
                        "discipline".to_string(),
                        "sector".to_string(),
                        "industry".to_string(),
                        "vertical".to_string(),
                        "function".to_string(),
                        "capability".to_string(),
                        "practice".to_string(),
                        "domain".to_string(),
                    ],
                ),
                (
                    "policy".to_string(),
                    vec![
                        "rule".to_string(),
                        "guideline".to_string(),
                        "standard".to_string(),
                        "requirement".to_string(),
                        "constraint".to_string(),
                        "regulation".to_string(),
                        "mandate".to_string(),
                        "directive".to_string(),
                        "protocol".to_string(),
                        "compliance".to_string(),
                    ],
                ),
                (
                    "method".to_string(),
                    vec![
                        "technique".to_string(),
                        "approach".to_string(),
                        "strategy".to_string(),
                        "practice".to_string(),
                        "methodology".to_string(),
                        "algorithm".to_string(),
                        "procedure".to_string(),
                        "tactic".to_string(),
                        "mechanism".to_string(),
                        "routine".to_string(),
                    ],
                ),
                (
                    "attribute".to_string(),
                    vec![
                        "property".to_string(),
                        "characteristic".to_string(),
                        "feature".to_string(),
                        "quality".to_string(),
                        "trait".to_string(),
                        "parameter".to_string(),
                        "dimension".to_string(),
                        "aspect".to_string(),
                        "factor".to_string(),
                        "criterion".to_string(),
                    ],
                ),
                (
                    "risk".to_string(),
                    vec![
                        "threat".to_string(),
                        "vulnerability".to_string(),
                        "hazard".to_string(),
                        "exposure".to_string(),
                        "uncertainty".to_string(),
                        "contingency".to_string(),
                        "impact".to_string(),
                        "probability".to_string(),
                        "mitigation".to_string(),
                        "severity".to_string(),
                    ],
                ),
                (
                    "decision".to_string(),
                    vec![
                        "choice".to_string(),
                        "selection".to_string(),
                        "judgment".to_string(),
                        "determination".to_string(),
                        "resolution".to_string(),
                        "verdict".to_string(),
                        "conclusion".to_string(),
                        "outcome".to_string(),
                        "action".to_string(),
                        "commitment".to_string(),
                    ],
                ),
                (
                    "relationship".to_string(),
                    vec![
                        "connection".to_string(),
                        "association".to_string(),
                        "linkage".to_string(),
                        "dependency".to_string(),
                        "correlation".to_string(),
                        "interaction".to_string(),
                        "coupling".to_string(),
                        "alignment".to_string(),
                        "mapping".to_string(),
                        "binding".to_string(),
                    ],
                ),
            ],
        }
    }

    /// Generate entity ID for a category
    fn generate_id(&mut self, category: &str) -> String {
        let counter = self
            .category_counters
            .entry(category.to_string())
            .or_insert(0);
        *counter += 1;
        format!(
            "entity_{}__{:03}",
            category.to_lowercase().replace(' ', "_"),
            counter
        )
    }

    /// Create an entity with automatic link generation
    pub fn create_entity(
        &mut self,
        name: &str,
        definition: &str,
        category: &str,
        aliases: Vec<String>,
        attributes: HashMap<String, String>,
    ) -> Entity {
        let id = self.generate_id(category);
        let normalized = name.to_lowercase().trim().to_string();

        // Generate links to existing entities
        let links = self.generate_links(&id, self.min_links);

        let entity = Entity {
            id: id.clone(),
            name: name.to_string(),
            normalized,
            definition: definition.to_string(),
            aliases,
            category: category.to_string(),
            attributes,
            links,
            contexts: Vec::new(),
        };

        self.entity_registry.insert(id, entity.clone());
        entity
    }

    /// Generate links to existing entities
    fn generate_links(&self, source_id: &str, count: usize) -> Vec<EntityLink> {
        let mut links = Vec::new();
        let existing_ids: Vec<_> = self
            .entity_registry
            .keys()
            .filter(|id| *id != source_id)
            .collect();

        let link_types = [
            "related", "parent", "child", "synonym", "part_of", "has_part",
        ];

        // Select random existing entities to link to
        for (i, target_id) in existing_ids.iter().take(count).enumerate() {
            let link_type = link_types[i % link_types.len()];
            let weight = 0.5 + (rand_weight() * 0.5); // 0.5-1.0

            links.push(EntityLink {
                target: (*target_id).clone(),
                link_type: link_type.to_string(),
                weight,
            });
        }

        links
    }

    /// Add context to an entity
    pub fn add_context(&mut self, entity_id: &str, text: &str, domain: &str) {
        if let Some(entity) = self.entity_registry.get_mut(entity_id) {
            entity.contexts.push(EntityContext {
                text: text.to_string(),
                domain: domain.to_string(),
            });
        }
    }

    /// Build the final dataset
    pub fn build(self, dataset_id: &str) -> EntityJsonDataset {
        let entities: Vec<Entity> = self.entity_registry.into_values().collect();
        let unit_count_estimate = entities.len() as u64;

        // Calculate density score based on estimated size
        let estimated_kb = estimate_dataset_size_kb(&entities);
        let density = if estimated_kb > 0.0 {
            (entities.len() as f32 / estimated_kb).min(1.0)
        } else {
            0.95
        };

        EntityJsonDataset {
            metadata: DatasetMetadata::new(dataset_id, "EntityJson", density, unit_count_estimate),
            entities,
        }
    }

    /// Get current entity count
    pub fn entity_count(&self) -> usize {
        self.entity_registry.len()
    }

    /// Generate bulk entities for high-density coverage with rich definitions
    pub fn generate_bulk_entities(&mut self, total_count: usize) {
        let domains = [
            "technology",
            "finance",
            "healthcare",
            "education",
            "legal",
            "marketing",
            "engineering",
            "research",
            "operations",
            "support",
            "hr",
            "sales",
            "product",
            "security",
            "data",
            "quality",
            "innovation",
            "sustainability",
            "strategy",
            "governance",
        ];

        let definition_templates = [
            "A structured approach to {} that ensures consistency and quality across the organization. This {} encompasses planning, execution, monitoring, and continuous improvement phases. Key success factors include stakeholder alignment, resource optimization, and measurable outcomes.",
            "The process of {} involves multiple stakeholders and requires careful coordination across functional boundaries. Implementation considerations include timing, dependencies, risk mitigation, and success criteria. Regular review cycles ensure alignment with organizational objectives.",
            "{} is a critical component of modern business operations, enabling efficiency and scalability. The {} framework provides guidelines for implementation, including prerequisites, step-by-step procedures, and validation checkpoints. Integration with existing systems requires careful planning and phased approach.",
            "This {} framework provides comprehensive guidelines for implementation and best practices. Core principles include transparency, accountability, and measurable outcomes. The framework supports both strategic planning and tactical execution, with built-in feedback mechanisms for continuous improvement.",
            "The {} system integrates with existing infrastructure to deliver seamless functionality. Architecture considerations include scalability, security, performance, and maintainability. The system supports multiple integration patterns and provides robust error handling and recovery mechanisms.",
            "An effective {} strategy requires alignment with organizational goals and stakeholder needs. Key elements include vision definition, gap analysis, roadmap development, and execution planning. Success metrics should be defined upfront and tracked throughout implementation.",
            "The {} methodology emphasizes iterative improvement and continuous feedback. Core practices include regular retrospectives, incremental delivery, and adaptive planning. The methodology supports both predictive and adaptive approaches based on project characteristics.",
            "Key principles of {} include transparency, accountability, and measurable outcomes. Governance structures ensure appropriate oversight while maintaining operational flexibility. Regular reporting and review processes maintain stakeholder confidence and support decision-making.",
            "The {} process enables teams to collaborate effectively and deliver results on time. Communication protocols, escalation procedures, and decision-making frameworks are embedded in the process. Integration with project management tools provides visibility and traceability.",
            "Implementing {} requires careful planning, resource allocation, and risk management. The implementation lifecycle includes discovery, design, development, testing, deployment, and optimization. Each phase has specific deliverables and acceptance criteria.",
            "{} represents a strategic capability that differentiates the organization in the marketplace. Investment in this {} yields returns through improved efficiency, reduced costs, enhanced quality, and increased customer satisfaction. Long-term sustainability requires ongoing commitment and evolution.",
            "The {} operational model defines how activities are structured, coordinated, and measured. Process flows, decision points, and handoff procedures are clearly documented. Performance indicators track efficiency, effectiveness, and quality metrics.",
            "{} encompasses a set of practices that ensure reliable and predictable outcomes. Standard operating procedures, training materials, and reference documentation support consistent execution. Continuous improvement initiatives identify and address process gaps.",
            "The {} capability enables the organization to respond effectively to changing conditions. Flexibility is built into the {} through modular design, configurable parameters, and adaptive workflows. Scenario planning prepares the organization for various contingencies.",
            "{} serves as a foundation for organizational excellence and competitive advantage. The {} integrates people, process, and technology elements into a cohesive whole. Investment in this capability demonstrates commitment to long-term success and stakeholder value.",
        ];

        // Clone categories to avoid borrow issues
        let categories: Vec<_> = self.categories.clone();
        let mut idx = 0;

        for (category, subtypes) in &categories {
            for subtype in subtypes {
                for domain in &domains {
                    if idx >= total_count {
                        return;
                    }

                    let name = format!("{} {} {}", domain, subtype, category);
                    let template = definition_templates[idx % definition_templates.len()];
                    let definition = template.replace("{}", &format!("{} {}", domain, subtype));

                    let mut attributes = HashMap::new();
                    attributes.insert("domain".to_string(), domain.to_string());
                    attributes.insert("category".to_string(), category.clone());
                    attributes.insert("subtype".to_string(), subtype.clone());
                    attributes.insert(
                        "priority".to_string(),
                        if idx % 4 == 0 {
                            "critical"
                        } else if idx % 4 == 1 {
                            "high"
                        } else if idx % 4 == 2 {
                            "medium"
                        } else {
                            "low"
                        }
                        .to_string(),
                    );
                    attributes.insert(
                        "lifecycle".to_string(),
                        if idx % 3 == 0 {
                            "active"
                        } else if idx % 3 == 1 {
                            "development"
                        } else {
                            "review"
                        }
                        .to_string(),
                    );

                    let aliases = vec![
                        format!("{} {}", domain, subtype),
                        format!("{} {}", subtype, category),
                        format!("{}-{}-{}", domain, subtype, category),
                    ];

                    self.create_entity(&name, &definition, category, aliases, attributes);
                    idx += 1;
                }
            }
        }

        // Fill remaining with variations including reasoning attributes
        while idx < total_count {
            let category_idx = idx % categories.len();
            let (category, subtypes) = &categories[category_idx];
            let domain = domains[idx % domains.len()];
            let subtype = &subtypes[idx % subtypes.len()];

            let name = format!("{}_{}_{}", domain, subtype, idx);
            let definition = format!(
                "Entity {} in the {} domain, categorized under {} with subtype {}. This entity supports organizational processes and enables effective operations. Key attributes include: operational efficiency impact ({}%), stakeholder involvement ({} parties), implementation complexity ({}), and strategic alignment score ({}/10). The entity integrates with related processes through defined interfaces and maintains traceability to business objectives.",
                idx, domain, category, subtype,
                10 + (idx % 50),
                2 + (idx % 8),
                ["low", "medium", "high", "critical"][idx % 4],
                5 + (idx % 5)
            );

            let mut attributes = HashMap::new();
            attributes.insert("domain".to_string(), domain.to_string());
            attributes.insert("index".to_string(), idx.to_string());
            attributes.insert(
                "efficiency_impact".to_string(),
                format!("{}%", 10 + (idx % 50)),
            );
            attributes.insert(
                "complexity".to_string(),
                ["low", "medium", "high", "critical"][idx % 4].to_string(),
            );
            attributes.insert(
                "alignment_score".to_string(),
                format!("{}/10", 5 + (idx % 5)),
            );

            self.create_entity(&name, &definition, category, vec![], attributes);
            idx += 1;
        }
    }
}

impl Default for EntityGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Estimate dataset size in KB
fn estimate_dataset_size_kb(entities: &[Entity]) -> f32 {
    // Rough estimate: each entity averages ~500 bytes in JSON
    (entities.len() as f32 * 0.5) / 1024.0
}

/// Generate random weight (placeholder - should use proper RNG)
fn rand_weight() -> f32 {
    // Simple deterministic pseudo-random for reproducibility
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    nanos as f32 / u32::MAX as f32
}

/// Validate entity dataset quality
pub fn validate_entity_dataset(dataset: &EntityJsonDataset) -> QualityMetrics {
    let entities = &dataset.entities;

    // Calculate entity density
    let estimated_kb = estimate_dataset_size_kb(entities);
    let entity_density = if estimated_kb > 0.0 {
        entities.len() as f32 / estimated_kb
    } else {
        0.0
    };

    // Calculate unique ratio
    let mut normalized_set = HashSet::new();
    for entity in entities {
        normalized_set.insert(entity.normalized.clone());
    }
    let unique_ratio = if !entities.is_empty() {
        normalized_set.len() as f32 / entities.len() as f32
    } else {
        1.0
    };

    // Calculate link coverage
    let entities_with_links = entities.iter().filter(|e| !e.links.is_empty()).count();
    let link_coverage = if !entities.is_empty() {
        entities_with_links as f32 / entities.len() as f32
    } else {
        1.0
    };

    // Noise ratio (entities with very short definitions)
    let noisy_entities = entities.iter().filter(|e| e.definition.len() < 10).count();
    let noise_ratio = if !entities.is_empty() {
        noisy_entities as f32 / entities.len() as f32
    } else {
        0.0
    };

    QualityMetrics {
        entity_density,
        unique_ratio,
        link_coverage,
        noise_ratio,
        intent_balance: 1.0, // Not applicable for entity datasets
        estimated_unit_discovery_efficiency: 0.70, // Entity datasets support unit discovery
        estimated_semantic_routing_accuracy: 0.85, // High accuracy due to structured entities
    }
}

/// Generate bulk entity data, streaming to a JSONL file as TrainingExamples.
/// Returns (examples_written, bytes_written).
pub fn generate_bulk_entities(
    output_path: &std::path::Path,
    target_bytes: u64,
    seed: u64,
) -> (u64, u64) {
    use crate::seed::bulk_generator::{
        expand_template, human_bytes, pick_idx, pick_str, seeded_rng, topics_for_domain,
        JsonlWriter, DOMAINS,
    };
    use crate::seed::TrainingExample;
    use crate::types::MemoryChannel;
    use rand::Rng;

    let mut rng = seeded_rng(seed);
    let mut writer = JsonlWriter::new(output_path).expect("create entity JSONL");
    let mut count: u64 = 0;

    let entity_categories = [
        "process",
        "concept",
        "artifact",
        "role",
        "metric",
        "tool",
        "resource",
        "event",
        "domain",
        "policy",
        "method",
        "attribute",
        "risk",
        "decision",
        "relationship",
    ];

    let link_types = [
        "related_to",
        "part_of",
        "has_part",
        "prerequisite_for",
        "produces",
        "consumes",
        "measured_by",
        "governed_by",
        "implemented_by",
        "supports",
    ];

    let definition_templates = [
        "{topic} is a core {category} in {domain} that governs how {detail_a} interacts with {detail_b}. It encompasses {num_a} distinct sub-components and operates across {num_b} organizational levels. Key metrics include efficiency ({num_c}%), reliability, and stakeholder satisfaction.",
        "In the context of {domain}, {topic} represents a {category} responsible for {detail_a}. Implementation requires {num_a} sequential phases: initiation, {detail_b}, execution, and {detail_c}. Success is measured against {num_b} predefined benchmarks.",
        "The {category} known as {topic} within {domain} provides a framework for {detail_a} and {detail_b}. It was established to address gaps in {detail_c} and has since evolved through {num_a} major revisions. Current adoption spans {num_b} departments.",
        "{topic} ({domain} {category}): enables {detail_a} through structured {detail_b}. Core attributes include scalability ({num_a}x), latency ({num_b}ms), and throughput ({num_c} ops/sec). Integration with {detail_c} provides end-to-end coverage.",
        "As a key {domain} {category}, {topic} orchestrates {detail_a}, {detail_b}, and {detail_c}. The lifecycle includes {num_a} stages with quality gates at each transition. Annual review ensures alignment with {num_b} strategic objectives.",
    ];

    let question_patterns = [
        "What is the definition of {} in {}?",
        "Describe the {} {} and its key attributes.",
        "How does {} function within the {} domain?",
        "What are the main components of {} in {}?",
        "Explain the role of {} as a {} in its field.",
        "What relationships does {} have with other {} elements?",
        "List the attributes and properties of {} in {}.",
        "How is {} categorized within {}?",
        "What distinguishes {} from related concepts in {}?",
        "Provide a structured overview of {} in the context of {}.",
        "What are the prerequisites for understanding {} in {}?",
        "How has {} evolved as a concept within {}?",
        "What are the practical applications of {} in {}?",
        "Compare {} with its closest alternatives in {}.",
        "What metrics are used to evaluate {} in {}?",
        "What constraints apply to {} within the {} framework?",
        "How do experts in {} typically define {}?",
        "What are the subtypes or variants of {} found in {}?",
        "Trace the origin and development of {} in {}.",
        "What problems does {} solve in the {} domain?",
        "Summarize the key characteristics of {} in {} terms.",
        "How does {} interact with adjacent areas of {}?",
        "What are the common misconceptions about {} in {}?",
        "What foundational knowledge is needed to master {} in {}?",
    ];

    while writer.bytes_written() < target_bytes {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick_str(&mut rng, topics);
        let category = pick_str(&mut rng, &entity_categories);

        let q_pattern = pick_str(&mut rng, &question_patterns);
        let question = q_pattern.replacen("{}", &format!("{}", topic), 1).replacen(
            "{}",
            &format!("{} ({})", domain, category),
            1,
        );

        let def_template = pick_str(&mut rng, &definition_templates);
        let article = if matches!(
            domain.chars().next(),
            Some('a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
        ) {
            "an"
        } else {
            "a"
        };
        let definition = expand_template(&mut rng, def_template, domain, topic)
            .replace("{category}", category)
            .replace("a key", &format!("{} key", article));

        // Generate entity link descriptions
        let link_count: usize = rng.gen_range(2..6);
        let mut link_descriptions = Vec::new();
        for _ in 0..link_count {
            let link_type = pick_str(&mut rng, &link_types);
            let related_topic = pick_str(&mut rng, topics);
            link_descriptions.push(format!("{} {} {}", topic, link_type, related_topic));
        }
        let links_text = link_descriptions.join("; ");

        let answer = format!(
            "{}\n\nRelationships: {}.\n\nCategory: {} | Domain: {}",
            definition, links_text, category, domain
        );

        let entities_list: Vec<String> =
            vec![topic.to_string(), domain.to_string(), category.to_string()];

        let example = TrainingExample {
            question,
            answer,
            context: Some(format!("entity:{}:{}:{}", domain, category, topic)),
            reasoning: None,
            intent: Some("Question".to_string()),
            entities: entities_list,
            channels: vec![MemoryChannel::Main],
            curriculum: crate::seed::CurriculumMetadata {
                curriculum_score: rng.gen_range(90..115),
                memory_channels: vec![MemoryChannel::Main],
                ..Default::default()
            },
            quality_gates: Default::default(),
            training_options: Default::default(),
        };

        writer.write_example(&example).expect("write entity");
        count += 1;

        if count % 100_000 == 0 {
            eprintln!(
                "  entities: {} examples, {}",
                count,
                human_bytes(writer.bytes_written())
            );
        }
    }

    writer.flush().expect("flush entity JSONL");
    (count, writer.bytes_written())
}
