//! DialogueJson dataset generator for intent classification training.
//! Includes combinations of IntentKind, ToneKind, and ResolverMode for comprehensive coverage.

use crate::seed::{DatasetMetadata, QualityMetrics};
use crate::types::{IntentKind, ToneKind, ResolverMode};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Dialogue turn (user or assistant)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueTurn {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub context: Option<String>,
    /// Expected entities to be extracted (for core training validation)
    #[serde(default)]
    pub expected_entities: Vec<String>,
    /// Expected anchor phrases (for Layer 8 anchor validation)
    #[serde(default)]
    pub expected_anchors: Vec<String>,
    /// Expected unit counts per level (for Layer 2 validation)
    #[serde(default)]
    pub expected_unit_count: ExpectedUnitCount,
    /// Source quality hint for trust scoring (assistant turns)
    #[serde(default)]
    pub source_quality: Option<f32>,
}

/// Expected unit counts per level for validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpectedUnitCount {
    #[serde(default)]
    pub phrase: Option<u32>,
    #[serde(default)]
    pub sentence: Option<u32>,
    #[serde(default)]
    pub word: Option<u32>,
}

/// Complete dialogue with intent, tone, and resolver labels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialogue {
    pub id: String,
    pub intent: String,
    /// Expected tone for assistant response
    #[serde(default)]
    pub expected_tone: Option<String>,
    /// Resolver mode appropriate for this dialogue
    #[serde(default)]
    pub resolver_mode: Option<String>,
    pub turns: Vec<DialogueTurn>,
    pub metadata: DialogueMetadata,
}

/// Memory target for core training (Layer 21 compliance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTarget {
    /// Staging episodic - high utility, requires corroboration for Core promotion
    #[default]
    StagingEpisodic,
    /// Direct to Core (only via consolidate_immediately or explicit corroboration)
    Core,
    /// Standard episodic memory
    Episodic,
}

/// Metadata for dialogue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueMetadata {
    pub domain: String,
    pub complexity: String,
    #[serde(default)]
    pub entities_referenced: Vec<String>,
    /// Memory channels this dialogue should route to
    #[serde(default)]
    pub memory_channels: Vec<String>,
    /// Unit levels expected in processing
    #[serde(default)]
    pub expected_unit_levels: Vec<String>,
    /// Target memory type (Core requires Layer 21 corroboration gate)
    #[serde(default)]
    pub memory_target: MemoryTarget,
    /// Minimum corroboration count for Core promotion (default 2 per Appendix B5)
    #[serde(default = "default_corroboration_threshold")]
    pub corroboration_threshold: u32,
}

fn default_corroboration_threshold() -> u32 {
    2
}

/// Complete DialogueJson dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueJsonDataset {
    #[serde(flatten)]
    pub metadata: DatasetMetadata,
    pub dialogues: Vec<Dialogue>,
}

/// Generator for DialogueJson datasets
pub struct DialogueGenerator {
    /// Dialogues per intent kind target
    dialogues_per_intent: usize,
    /// Generated dialogues registry
    dialogues: Vec<Dialogue>,
    /// Intent counters for ID generation
    intent_counters: HashMap<String, u64>,
    /// Intent distribution tracker
    intent_distribution: BTreeMap<String, u64>,
}

impl DialogueGenerator {
    pub fn new() -> Self {
        Self {
            dialogues_per_intent: 200,
            dialogues: Vec::new(),
            intent_counters: HashMap::new(),
            intent_distribution: BTreeMap::new(),
        }
    }

    /// Create a dialogue with intent, tone, and resolver labels
    pub fn create_dialogue(
        &mut self,
        intent: IntentKind,
        expected_tone: Option<ToneKind>,
        resolver_mode: Option<ResolverMode>,
        turns: Vec<(String, String)>, // (role, content) pairs
        domain: &str,
        complexity: &str,
        entities: Vec<String>,
        memory_channels: Vec<String>,
        expected_unit_levels: Vec<String>,
    ) -> Dialogue {
        let intent_str = format!("{:?}", intent);
        let counter = self.intent_counters.entry(intent_str.clone()).or_insert(0);
        *counter += 1;
        
        let id = format!("dialogue_{}_{:04}", 
            intent_str.to_lowercase(), 
            counter
        );
        
        let dialogue_turns: Vec<DialogueTurn> = turns
            .into_iter()
            .map(|(role, content)| DialogueTurn {
                role,
                content,
                context: None,
                expected_entities: Vec::new(),
                expected_anchors: Vec::new(),
                expected_unit_count: ExpectedUnitCount::default(),
                source_quality: None,
            })
            .collect();
        
        // Track intent distribution
        *self.intent_distribution.entry(intent_str.clone()).or_insert(0) += 1;
        
        Dialogue {
            id,
            intent: intent_str,
            expected_tone: expected_tone.map(|t| format!("{:?}", t)),
            resolver_mode: resolver_mode.map(|r| format!("{:?}", r)),
            turns: dialogue_turns,
            metadata: DialogueMetadata {
                domain: domain.to_string(),
                complexity: complexity.to_string(),
                entities_referenced: entities,
                memory_channels,
                expected_unit_levels,
                memory_target: MemoryTarget::Episodic,
                corroboration_threshold: default_corroboration_threshold(),
            },
        }
    }

    /// Create a simple dialogue (backward compatible)
    pub fn create_dialogue_simple(
        &mut self,
        intent: IntentKind,
        turns: Vec<(String, String)>,
        domain: &str,
        complexity: &str,
        entities: Vec<String>,
    ) -> Dialogue {
        // Infer appropriate tone and resolver from intent
        let (tone, resolver) = Self::infer_tone_resolver(intent);
        let channels = Self::infer_memory_channels(intent);
        let levels = vec!["Word".to_string(), "Phrase".to_string()];
        
        self.create_dialogue(
            intent,
            Some(tone),
            Some(resolver),
            turns,
            domain,
            complexity,
            entities,
            channels,
            levels,
        )
    }

    /// Infer appropriate tone and resolver mode from intent
    fn infer_tone_resolver(intent: IntentKind) -> (ToneKind, ResolverMode) {
        match intent {
            // Social intents - casual/empathetic tone, balanced mode
            IntentKind::Greeting | IntentKind::Gratitude | IntentKind::Farewell => {
                (ToneKind::Casual, ResolverMode::Balanced)
            }
            // Factual/informational - neutral/professional, deterministic
            IntentKind::Question | IntentKind::Verify | IntentKind::Extract | IntentKind::Classify => {
                (ToneKind::NeutralProfessional, ResolverMode::Deterministic)
            }
            // Explanatory - technical tone, balanced
            IntentKind::Explain | IntentKind::Summarize | IntentKind::Compare | IntentKind::Analyze => {
                (ToneKind::Technical, ResolverMode::Balanced)
            }
            // Action-oriented - direct tone, deterministic
            IntentKind::Act | IntentKind::Debug | IntentKind::Plan | IntentKind::Recommend => {
                (ToneKind::Direct, ResolverMode::Deterministic)
            }
            // Creative - empathetic/formal, exploratory
            IntentKind::Brainstorm | IntentKind::Critique => {
                (ToneKind::Empathetic, ResolverMode::Exploratory)
            }
            // Assistance - neutral, balanced
            IntentKind::Help | IntentKind::Clarify | IntentKind::Rewrite | IntentKind::Continue | IntentKind::Forget => {
                (ToneKind::NeutralProfessional, ResolverMode::Balanced)
            }
            // Translation - formal, deterministic
            IntentKind::Translate => {
                (ToneKind::Formal, ResolverMode::Deterministic)
            }
            // Unknown - neutral, balanced
            IntentKind::Unknown => {
                (ToneKind::NeutralProfessional, ResolverMode::Balanced)
            }
        }
    }

    /// Infer memory channels from intent
    fn infer_memory_channels(intent: IntentKind) -> Vec<String> {
        match intent {
            IntentKind::Question | IntentKind::Explain | IntentKind::Help | IntentKind::Clarify => {
                vec!["Main".to_string(), "Intent".to_string()]
            }
            IntentKind::Plan | IntentKind::Analyze | IntentKind::Debug => {
                vec!["Main".to_string(), "Reasoning".to_string()]
            }
            _ => vec!["Main".to_string()]
        }
    }

    /// Add dialogue to generator
    pub fn add_dialogue(&mut self, dialogue: Dialogue) {
        let intent = dialogue.intent.clone();
        self.dialogues.push(dialogue);
        *self.intent_distribution.entry(intent).or_insert(0) += 1;
    }

    /// Build the final dataset
    pub fn build(self, dataset_id: &str) -> DialogueJsonDataset {
        let unit_count_estimate = self.dialogues.len() as u64;
        
        // Calculate density score
        let density = 0.85; // Target density for DialogueJson
        
        DialogueJsonDataset {
            metadata: DatasetMetadata::new(
                dataset_id,
                "DialogueJson",
                density,
                unit_count_estimate,
            ),
            dialogues: self.dialogues,
        }
    }

    /// Get current dialogue count
    pub fn dialogue_count(&self) -> usize {
        self.dialogues.len()
    }

    /// Get intent distribution
    pub fn intent_distribution(&self) -> &BTreeMap<String, u64> {
        &self.intent_distribution
    }

    /// Check if all intents are covered
    pub fn all_intents_covered(&self) -> bool {
        let all_intents = [
            "Greeting", "Gratitude", "Farewell", "Help", "Clarify", "Rewrite", "Verify",
            "Continue", "Forget", "Question", "Summarize", "Explain", "Compare", "Extract",
            "Analyze", "Plan", "Act", "Recommend", "Classify", "Translate", "Debug",
            "Critique", "Brainstorm", "Unknown"
        ];
        
        all_intents.iter().all(|intent| {
            self.intent_distribution.contains_key(*intent)
        })
    }
}

impl Default for DialogueGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate dialogue dataset quality
pub fn validate_dialogue_dataset(dataset: &DialogueJsonDataset) -> QualityMetrics {
    let dialogues = &dataset.dialogues;
    
    // Entity density not directly applicable - use dialogue density
    let entity_density = dialogues.len() as f32; // dialogues per "unit"
    
    // Calculate unique normalized intents ratio
    let mut intent_set = HashSet::new();
    for dialogue in dialogues {
        intent_set.insert(dialogue.intent.clone());
    }
    let unique_ratio = if !dialogues.is_empty() {
        intent_set.len() as f32 / 24.0 // 24 intent kinds (excluding Unknown as primary)
    } else {
        0.0
    };
    
    // Link coverage - entities referenced
    let dialogues_with_entities = dialogues.iter()
        .filter(|d| !d.metadata.entities_referenced.is_empty())
        .count();
    let link_coverage = if !dialogues.is_empty() {
        dialogues_with_entities as f32 / dialogues.len() as f32
    } else {
        0.0
    };
    
    // Noise ratio - dialogues with very short content
    let noisy_dialogues = dialogues.iter()
        .filter(|d| d.turns.iter().all(|t| t.content.len() < 10))
        .count();
    let noise_ratio = if !dialogues.is_empty() {
        noisy_dialogues as f32 / dialogues.len() as f32
    } else {
        0.0
    };
    
    // Intent balance - distribution evenness
    let mut intent_counts: BTreeMap<String, u64> = BTreeMap::new();
    for dialogue in dialogues {
        *intent_counts.entry(dialogue.intent.clone()).or_insert(0) += 1;
    }
    
    let avg_count = dialogues.len() as f32 / intent_counts.len().max(1) as f32;
    let variance: f32 = intent_counts.values()
        .map(|&count| {
            let diff = count as f32 - avg_count;
            diff * diff
        })
        .sum::<f32>() / intent_counts.len().max(1) as f32;
    let std_dev = variance.sqrt();
    
    // Balance score: lower std_dev relative to mean = better balance
    let intent_balance = if avg_count > 0.0 {
        1.0 - (std_dev / avg_count).min(1.0)
    } else {
        0.0
    };
    
    QualityMetrics {
        entity_density,
        unique_ratio,
        link_coverage,
        noise_ratio,
        intent_balance,
    }
}

/// Template dialogues for each intent kind
pub mod templates {
    use super::*;
    
    /// Generate template dialogues for a given intent with all tone/resolver combinations
    pub fn generate_intent_dialogues(
        gen: &mut DialogueGenerator,
        intent: IntentKind,
        count: usize,
    ) {
        let templates = get_templates_for_intent(intent);
        let tones = get_tones_for_intent(intent);
        let resolvers = get_resolvers_for_intent(intent);
        
        // Generate dialogues covering all tone/resolver combinations
        let mut combo_index = 0;
        for i in 0..count {
            let template = &templates[i % templates.len()];
            
            // Cycle through tone/resolver combinations for this intent
            let tone = tones[combo_index % tones.len()];
            let resolver = resolvers[combo_index % resolvers.len()];
            combo_index += 1;
            
            let dialogue = gen.create_dialogue(
                intent,
                Some(tone),
                Some(resolver),
                template.turns.clone(),
                template.domain,
                template.complexity,
                template.entities.clone(),
                template.memory_channels.clone(),
                template.unit_levels.clone(),
            );
            gen.add_dialogue(dialogue);
        }
    }
    
    /// Get appropriate tones for an intent (allows variation for training)
    pub fn get_tones_for_intent(intent: IntentKind) -> Vec<ToneKind> {
        match intent {
            IntentKind::Greeting | IntentKind::Gratitude | IntentKind::Farewell => {
                vec![ToneKind::Casual, ToneKind::NeutralProfessional, ToneKind::Formal]
            }
            IntentKind::Question | IntentKind::Verify | IntentKind::Extract | IntentKind::Classify => {
                vec![ToneKind::NeutralProfessional, ToneKind::Technical]
            }
            IntentKind::Explain | IntentKind::Summarize | IntentKind::Compare | IntentKind::Analyze => {
                vec![ToneKind::Technical, ToneKind::NeutralProfessional]
            }
            IntentKind::Act | IntentKind::Debug | IntentKind::Plan | IntentKind::Recommend => {
                vec![ToneKind::Direct, ToneKind::Technical]
            }
            IntentKind::Brainstorm | IntentKind::Critique => {
                vec![ToneKind::Empathetic, ToneKind::Casual, ToneKind::NeutralProfessional]
            }
            IntentKind::Help | IntentKind::Clarify | IntentKind::Rewrite | IntentKind::Continue | IntentKind::Forget => {
                vec![ToneKind::NeutralProfessional, ToneKind::Empathetic]
            }
            IntentKind::Translate => {
                vec![ToneKind::Formal, ToneKind::NeutralProfessional]
            }
            IntentKind::Unknown => {
                vec![ToneKind::NeutralProfessional]
            }
        }
    }
    
    /// Get appropriate resolver modes for an intent
    pub fn get_resolvers_for_intent(intent: IntentKind) -> Vec<ResolverMode> {
        match intent {
            IntentKind::Question | IntentKind::Verify | IntentKind::Extract | IntentKind::Classify | IntentKind::Act | IntentKind::Debug | IntentKind::Translate => {
                vec![ResolverMode::Deterministic]
            }
            IntentKind::Brainstorm | IntentKind::Critique => {
                vec![ResolverMode::Exploratory, ResolverMode::Balanced]
            }
            _ => {
                vec![ResolverMode::Balanced, ResolverMode::Deterministic]
            }
        }
    }
    
    pub struct DialogueTemplate {
        pub turns: Vec<(String, String)>,
        pub domain: &'static str,
        pub complexity: &'static str,
        pub entities: Vec<String>,
        pub memory_channels: Vec<String>,
        pub unit_levels: Vec<String>,
    }
    
    impl Default for DialogueTemplate {
        fn default() -> Self {
            Self {
                turns: Vec::new(),
                domain: "general",
                complexity: "simple",
                entities: Vec::new(),
                memory_channels: vec!["Main".to_string()],
                unit_levels: vec!["Word".to_string(), "Phrase".to_string()],
            }
        }
    }
    
    pub fn get_templates_for_intent(intent: IntentKind) -> Vec<DialogueTemplate> {
        // Base templates plus programmatically generated variations
        let mut templates = base_templates_for_intent(intent);
        
        // Add domain-specific variations for high-density coverage
        templates.extend(domain_variations_for_intent(intent));
        
        // Add complexity variations (multi-turn dialogues)
        templates.extend(complexity_variations_for_intent(intent));
        
        templates
    }
    
    /// Base templates for each intent
    fn base_templates_for_intent(intent: IntentKind) -> Vec<DialogueTemplate> {
        match intent {
            IntentKind::Greeting => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Hello, how are you today?".to_string()),
                        ("assistant".to_string(), "Hello! I'm doing well, thank you for asking. How can I help you today?".to_string()),
                    ],
                    ..Default::default()
                },
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Hi there!".to_string()),
                        ("assistant".to_string(), "Hi! I'm here to help. What would you like to know?".to_string()),
                    ],
                    ..Default::default()
                },
            ],
            IntentKind::Question => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "What is the approval workflow for purchase orders over $10,000?".to_string()),
                        ("assistant".to_string(), "Purchase orders over $10,000 require department head approval, followed by finance review, and final authorization from the CFO.".to_string()),
                    ],
                    domain: "business_operations",
                    entities: vec!["approval_workflow".to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                    ..Default::default()
                },
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "How do I submit a request for time off?".to_string()),
                        ("assistant".to_string(), "To submit a time-off request, log into the HR portal, navigate to 'Leave Management', select your dates, and submit for manager approval.".to_string()),
                    ],
                    domain: "hr",
                    entities: vec!["time_off_request".to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Explain => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Can you explain how process efficiency is calculated?".to_string()),
                        ("assistant".to_string(), "Process efficiency is calculated by dividing the useful output by the total input. For example, if a workflow produces 80 approved documents from 100 submitted, the efficiency is 80%.".to_string()),
                    ],
                    domain: "operations",
                    complexity: "moderate",
                    entities: vec!["process_efficiency".to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Help => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "I need help understanding the new expense reporting system.".to_string()),
                        ("assistant".to_string(), "I'd be happy to help! The expense reporting system allows you to submit receipts, categorize expenses, and track reimbursements. Would you like a step-by-step guide?".to_string()),
                    ],
                    domain: "finance",
                    entities: vec!["expense_reporting".to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Summarize => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Can you summarize the quarterly sales report?".to_string()),
                        ("assistant".to_string(), "The Q3 sales report shows a 15% increase in revenue, with the enterprise segment contributing 60% of total sales. Key growth areas include cloud services and consulting.".to_string()),
                    ],
                    domain: "business",
                    complexity: "moderate",
                    entities: vec!["sales_report".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Plan => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Help me plan a product launch timeline.".to_string()),
                        ("assistant".to_string(), "I'll help you create a product launch timeline. Let's break it down into phases: pre-launch (4 weeks), launch week, and post-launch (2 weeks). Each phase has specific milestones and deliverables.".to_string()),
                    ],
                    domain: "product",
                    complexity: "complex",
                    entities: vec!["product_launch".to_string()],
                    memory_channels: vec!["Main".to_string(), "Reasoning".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Analyze => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Analyze the customer feedback trends from last month.".to_string()),
                        ("assistant".to_string(), "Analysis of customer feedback shows three main themes: 40% mention response time, 35% cite product features, and 25% discuss pricing. Response time complaints increased by 12% compared to the previous month.".to_string()),
                    ],
                    domain: "analytics",
                    complexity: "complex",
                    entities: vec!["customer_feedback".to_string()],
                    memory_channels: vec!["Main".to_string(), "Reasoning".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Compare => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Compare the two project management tools we're considering.".to_string()),
                        ("assistant".to_string(), "Tool A offers better integration capabilities and lower cost, while Tool B provides more advanced reporting features and better customer support. For your needs, Tool A may be more suitable given the integration requirements.".to_string()),
                    ],
                    domain: "tools",
                    complexity: "moderate",
                    entities: vec!["project_management".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Recommend => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "What approach would you recommend for data migration?".to_string()),
                        ("assistant".to_string(), "I recommend a phased migration approach: start with non-critical data to test the process, then migrate core business data during low-traffic periods, and finally migrate historical archives. This minimizes risk and allows for validation at each stage.".to_string()),
                    ],
                    domain: "technical",
                    complexity: "moderate",
                    entities: vec!["data_migration".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Debug => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "The approval workflow is stuck on pending status. Can you help debug?".to_string()),
                        ("assistant".to_string(), "Let me help debug this. Common causes for stuck pending status include: 1) Missing approver assignment, 2) Workflow rule conflicts, 3) System timeout. Let's check the workflow logs first to identify the specific issue.".to_string()),
                    ],
                    domain: "technical",
                    complexity: "moderate",
                    entities: vec!["approval_workflow".to_string()],
                    memory_channels: vec!["Main".to_string(), "Reasoning".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Brainstorm => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Let's brainstorm ideas for improving team collaboration.".to_string()),
                        ("assistant".to_string(), "Great topic! Here are some ideas: 1) Implement daily standups, 2) Create shared knowledge base, 3) Use collaborative tools like shared whiteboards, 4) Schedule regular team retrospectives, 5) Establish cross-functional project teams. Which areas would you like to explore further?".to_string()),
                    ],
                    domain: "team",
                    complexity: "moderate",
                    entities: vec!["team_collaboration".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Clarify => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Can you clarify what you mean by 'stakeholder alignment'?".to_string()),
                        ("assistant".to_string(), "Stakeholder alignment means ensuring all key parties affected by a decision are in agreement on goals, timeline, and expected outcomes. This includes getting explicit sign-off from department heads, finance, and executive sponsors.".to_string()),
                    ],
                    domain: "business",
                    entities: vec!["stakeholder".to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Rewrite => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Rewrite this email to be more professional: 'Hey, need that report ASAP'".to_string()),
                        ("assistant".to_string(), "Here's a more professional version: 'Hi [Name], I hope this message finds you well. Could you please provide the report at your earliest convenience? Your prompt response would be greatly appreciated.'".to_string()),
                    ],
                    domain: "communication",
                    ..Default::default()
                },
            ],
            IntentKind::Verify => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Verify that the budget allocation matches the approved proposal.".to_string()),
                        ("assistant".to_string(), "I've verified the budget allocation against the approved proposal. The allocation matches with the following breakdown: Development 45%, Marketing 30%, Operations 25%. All figures align with the Q4 budget proposal.".to_string()),
                    ],
                    domain: "finance",
                    complexity: "moderate",
                    entities: vec!["budget_allocation".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Gratitude => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Thank you for your help with the presentation!".to_string()),
                        ("assistant".to_string(), "You're welcome! I'm glad I could help. Let me know if you need any further assistance with your presentation or other tasks.".to_string()),
                    ],
                    ..Default::default()
                },
            ],
            IntentKind::Farewell => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Goodbye, have a great day!".to_string()),
                        ("assistant".to_string(), "Goodbye! Have a wonderful day. Feel free to return whenever you need assistance.".to_string()),
                    ],
                    ..Default::default()
                },
            ],
            IntentKind::Continue => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Continue with the next step of the process.".to_string()),
                        ("assistant".to_string(), "Proceeding with step 3 of the process: Now we need to validate the input data against the schema requirements. I'll check each field for compliance and report any discrepancies.".to_string()),
                    ],
                    domain: "process",
                    ..Default::default()
                },
            ],
            IntentKind::Forget => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Forget the previous context and start fresh.".to_string()),
                        ("assistant".to_string(), "Understood. I've cleared the previous context. How can I help you with a fresh start?".to_string()),
                    ],
                    ..Default::default()
                },
            ],
            IntentKind::Extract => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Extract the key dates from this project timeline document.".to_string()),
                        ("assistant".to_string(), "I've extracted the following key dates: Project Start: March 1, Milestone 1: April 15, Milestone 2: June 30, Final Delivery: August 15, Review Period: August 16-31.".to_string()),
                    ],
                    domain: "project",
                    complexity: "moderate",
                    entities: vec!["project_timeline".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Act => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Send a reminder to the team about tomorrow's meeting.".to_string()),
                        ("assistant".to_string(), "I've sent a meeting reminder to the team. The reminder includes the meeting time, location, agenda items, and required preparation materials. Confirmation received from 5 of 8 team members.".to_string()),
                    ],
                    domain: "communication",
                    entities: vec!["meeting".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Classify => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Classify these customer inquiries by priority level.".to_string()),
                        ("assistant".to_string(), "I've classified the inquiries as follows: High Priority (3): System outages, security concerns; Medium Priority (5): Feature requests, integration issues; Low Priority (12): General questions, documentation feedback.".to_string()),
                    ],
                    domain: "support",
                    complexity: "moderate",
                    entities: vec!["customer_inquiries".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Translate => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Translate this business requirement into technical specifications.".to_string()),
                        ("assistant".to_string(), "Technical specification derived from requirement: Implement REST API endpoint POST /api/approvals with request body {requester_id, amount, category}. Response should include approval_id and status. Integrate with workflow engine for state management.".to_string()),
                    ],
                    domain: "technical",
                    complexity: "complex",
                    entities: vec!["api_specification".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Critique => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "Critique this marketing copy for our new product launch.".to_string()),
                        ("assistant".to_string(), "Critique of the marketing copy: Strengths include clear value proposition and strong call-to-action. Areas for improvement: 1) The headline could be more attention-grabbing, 2) Benefit statements lack specificity, 3) No social proof or testimonials included. Consider adding customer quotes and quantifiable results.".to_string()),
                    ],
                    domain: "marketing",
                    complexity: "moderate",
                    entities: vec!["marketing_copy".to_string()],
                    ..Default::default()
                },
            ],
            IntentKind::Unknown => vec![
                DialogueTemplate {
                    turns: vec![
                        ("user".to_string(), "xyz123 random input".to_string()),
                        ("assistant".to_string(), "I'm not sure I understand your request. Could you please rephrase or provide more context?".to_string()),
                    ],
                    ..Default::default()
                },
            ],
        }
    }
    
    /// Domain-specific variations for high-density coverage
    fn domain_variations_for_intent(intent: IntentKind) -> Vec<DialogueTemplate> {
        let domains = [
            ("technology", "software systems, APIs, cloud infrastructure, microservices architecture, DevOps pipelines"),
            ("finance", "budgets, investments, financial reporting, risk assessment, portfolio management, compliance audits"),
            ("healthcare", "patient care, medical records, treatment plans, clinical trials, diagnostic procedures, health outcomes"),
            ("education", "curriculum, assessments, learning outcomes, student engagement, pedagogical methods, academic research"),
            ("legal", "contracts, compliance, regulatory requirements, litigation support, intellectual property, corporate governance"),
            ("marketing", "campaigns, brand strategy, customer engagement, market analysis, conversion optimization, brand positioning"),
            ("engineering", "design specs, technical architecture, quality assurance, system integration, performance optimization, scalability"),
            ("research", "methodology, data analysis, peer review, hypothesis testing, experimental design, publication strategy"),
            ("operations", "supply chain, logistics, inventory management, process optimization, resource allocation, efficiency metrics"),
            ("hr", "talent acquisition, employee development, performance management, workplace culture, retention strategies"),
            ("sales", "pipeline management, deal negotiation, customer relationships, revenue forecasting, territory planning"),
            ("product", "roadmap planning, feature prioritization, user research, market fit, competitive analysis, launch strategy"),
            ("security", "threat detection, vulnerability assessment, incident response, access control, security audits, compliance"),
            ("data", "analytics, data pipelines, warehousing, governance, quality assurance, visualization, predictive modeling"),
            ("support", "ticket management, escalation procedures, customer satisfaction, knowledge base, service levels"),
            ("project", "milestone tracking, resource planning, risk mitigation, stakeholder communication, deliverable management"),
            ("quality", "testing protocols, defect tracking, process improvement, standards compliance, audit preparation"),
            ("innovation", "ideation, prototyping, feasibility analysis, pilot programs, scaling strategies, intellectual capital"),
            ("sustainability", "environmental impact, resource efficiency, carbon footprint, green initiatives, ESG reporting"),
            ("strategy", "competitive positioning, market expansion, partnership development, long-term planning, scenario analysis"),
        ];
        
        let entity_types = [
            "workflow", "process", "system", "document", "report",
            "analysis", "proposal", "review", "assessment", "plan",
        ];
        
        domains.iter().flat_map(|(domain, context)| {
            (0..15).map(|i| {
                let entity = entity_types[i % entity_types.len()];
                let complexity = match i % 5 {
                    0 => "complex",
                    1 => "highly_complex",
                    2 => "moderate",
                    3 => "expert",
                    _ => "advanced",
                };
                DialogueTemplate {
                    turns: generate_domain_turns(intent, domain, context, entity),
                    domain,
                    complexity,
                    entities: vec![entity.to_string()],
                    memory_channels: vec!["Main".to_string(), "Intent".to_string(), "Reasoning".to_string()],
                    unit_levels: vec!["Word".to_string(), "Phrase".to_string(), "Sentence".to_string(), "Paragraph".to_string()],
                }
            }).collect::<Vec<_>>()
        }).collect()
    }
    
    /// Multi-turn complexity variations with reasoning chains
    fn complexity_variations_for_intent(intent: IntentKind) -> Vec<DialogueTemplate> {
        let mut templates = Vec::new();
        
        // Generate 3-turn dialogues
        for i in 0..5 {
            templates.push(DialogueTemplate {
                turns: generate_multi_turn_dialogue(intent, 3, i),
                domain: "general",
                complexity: "moderate",
                entities: vec![],
                memory_channels: vec!["Main".to_string(), "Intent".to_string()],
                unit_levels: vec!["Word".to_string(), "Phrase".to_string()],
            });
        }
        
        // Generate 5-turn dialogues with reasoning
        for i in 0..4 {
            templates.push(DialogueTemplate {
                turns: generate_multi_turn_dialogue(intent, 5, i),
                domain: "general",
                complexity: "complex",
                entities: vec![],
                memory_channels: vec!["Main".to_string(), "Intent".to_string(), "Reasoning".to_string()],
                unit_levels: vec!["Word".to_string(), "Phrase".to_string(), "Sentence".to_string()],
            });
        }
        
        // Generate 7-turn dialogues with deep reasoning
        for i in 0..3 {
            templates.push(DialogueTemplate {
                turns: generate_multi_turn_dialogue(intent, 7, i),
                domain: "expert",
                complexity: "highly_complex",
                entities: vec![],
                memory_channels: vec!["Main".to_string(), "Intent".to_string(), "Reasoning".to_string(), "Analysis".to_string()],
                unit_levels: vec!["Word".to_string(), "Phrase".to_string(), "Sentence".to_string(), "Paragraph".to_string()],
            });
        }
        
        // Generate 10-turn dialogues for expert scenarios
        for i in 0..2 {
            templates.push(DialogueTemplate {
                turns: generate_multi_turn_dialogue(intent, 10, i),
                domain: "expert",
                complexity: "expert",
                entities: vec![],
                memory_channels: vec!["Main".to_string(), "Intent".to_string(), "Reasoning".to_string(), "Analysis".to_string(), "Context".to_string()],
                unit_levels: vec!["Word".to_string(), "Phrase".to_string(), "Sentence".to_string(), "Paragraph".to_string(), "Document".to_string()],
            });
        }
        
        templates
    }
    
    fn generate_domain_turns(intent: IntentKind, domain: &str, context: &str, entity: &str) -> Vec<(String, String)> {
        let user_prompt = match intent {
            IntentKind::Question => format!("What are the key {} considerations for {} in {}?", entity, context, domain),
            IntentKind::Explain => format!("Explain how {} impacts {} operations.", entity, domain),
            IntentKind::Analyze => format!("Analyze the {} trends in {} regarding {}.", entity, domain, context),
            IntentKind::Summarize => format!("Summarize the {} findings for {}.", entity, domain),
            IntentKind::Plan => format!("Create a {} plan for {} implementation.", entity, domain),
            IntentKind::Help => format!("I need help with {} in the {} domain.", entity, domain),
            IntentKind::Compare => format!("Compare different {} approaches in {}.", entity, domain),
            IntentKind::Recommend => format!("Recommend best practices for {} in {}.", entity, domain),
            IntentKind::Debug => format!("Debug the {} issue in our {} system.", entity, domain),
            IntentKind::Verify => format!("Verify {} compliance in {} processes.", entity, domain),
            IntentKind::Classify => format!("Classify {} items by priority in {}.", entity, domain),
            IntentKind::Extract => format!("Extract key {} data from {} records.", entity, domain),
            IntentKind::Brainstorm => format!("Brainstorm {} improvements for {}.", entity, domain),
            IntentKind::Critique => format!("Critique our {} strategy in {}.", entity, domain),
            IntentKind::Translate => format!("Translate {} requirements to {} specifications.", entity, domain),
            IntentKind::Act => format!("Execute {} action for {} workflow.", entity, domain),
            IntentKind::Clarify => format!("Clarify the {} requirements for {}.", entity, domain),
            IntentKind::Rewrite => format!("Rewrite the {} documentation for {}.", entity, domain),
            IntentKind::Continue => format!("Continue with {} processing in {}.", entity, domain),
            IntentKind::Forget => format!("Clear {} context and restart {} session.", entity, domain),
            IntentKind::Greeting => format!("Hello, I'm working in {} and need help with {}.", domain, context),
            IntentKind::Gratitude => format!("Thank you for the {} assistance with {}.", entity, domain),
            IntentKind::Farewell => format!("Goodbye, thanks for the {} help in {}.", entity, domain),
            IntentKind::Unknown => format!("Unclear request about {} in {}.", entity, domain),
        };
        
        let assistant_response = generate_assistant_response(intent, domain, entity, context);
        
        vec![
            ("user".to_string(), user_prompt),
            ("assistant".to_string(), assistant_response),
        ]
    }
    
    fn generate_multi_turn_dialogue(intent: IntentKind, turns: usize, variant: usize) -> Vec<(String, String)> {
        let mut dialogue = Vec::new();
        let topics = [
            "workflow optimization and efficiency gains",
            "data analysis and predictive modeling",
            "process improvement and automation",
            "system integration and architecture",
            "quality assurance and testing protocols",
            "strategic planning and execution",
            "risk assessment and mitigation",
            "stakeholder engagement and communication",
            "resource allocation and optimization",
            "performance metrics and KPI tracking",
            "compliance and regulatory requirements",
            "innovation and competitive advantage",
            "customer experience and satisfaction",
            "cost reduction and value creation",
            "change management and adoption",
        ];
        let topic = topics[variant % topics.len()];
        
        for i in 0..turns {
            let turn_num = i / 2;
            if i % 2 == 0 {
                dialogue.push((
                    "user".to_string(),
                    format!("{} turn {} about {}: {}", 
                        intent_as_verb(intent), turn_num + 1, topic, 
                        generate_followup_question(intent, turn_num, variant))
                ));
            } else {
                dialogue.push((
                    "assistant".to_string(),
                    generate_reasoning_response(intent, turn_num, topic, variant, turns)
                ));
            }
        }
        
        dialogue
    }
    
    fn intent_as_verb(intent: IntentKind) -> &'static str {
        match intent {
            IntentKind::Question => "Ask",
            IntentKind::Explain => "Explain",
            IntentKind::Analyze => "Analyze",
            IntentKind::Summarize => "Summarize",
            IntentKind::Plan => "Plan",
            IntentKind::Help => "Help with",
            IntentKind::Compare => "Compare",
            IntentKind::Recommend => "Recommend",
            IntentKind::Debug => "Debug",
            IntentKind::Verify => "Verify",
            IntentKind::Classify => "Classify",
            IntentKind::Extract => "Extract",
            IntentKind::Brainstorm => "Brainstorm",
            IntentKind::Critique => "Critique",
            IntentKind::Translate => "Translate",
            IntentKind::Act => "Act on",
            IntentKind::Clarify => "Clarify",
            IntentKind::Rewrite => "Rewrite",
            IntentKind::Continue => "Continue",
            IntentKind::Forget => "Forget",
            IntentKind::Greeting => "Greet",
            IntentKind::Gratitude => "Thank",
            IntentKind::Farewell => "Farewell",
            IntentKind::Unknown => "Process",
        }
    }
    
    fn generate_followup_question(intent: IntentKind, turn: usize, variant: usize) -> String {
        let questions = [
            "What are the initial findings and how do they compare to our baseline?",
            "Can you provide more details on the approach, including methodology and assumptions?",
            "What are the next steps and what dependencies should we be aware of?",
            "How does this compare to alternatives, and what trade-offs did you consider?",
            "What resources are needed, both human and technical, for successful implementation?",
            "Are there any risks or considerations that could impact the timeline or outcomes?",
            "What would success look like, and how would we measure it objectively?",
            "How does this align with our strategic objectives and other ongoing initiatives?",
            "What feedback have you received from stakeholders, and how has it shaped your recommendations?",
            "What are the potential failure modes, and how have you planned for contingencies?",
            "Can you walk through the cost-benefit analysis and ROI projections?",
            "What assumptions underlie your analysis, and how sensitive are the conclusions to those assumptions?",
        ];
        questions[(turn + variant) % questions.len()].to_string()
    }
    
    /// Generate reasoning-rich response with step-by-step analysis
    fn generate_reasoning_response(intent: IntentKind, turn: usize, topic: &str, variant: usize, total_turns: usize) -> String {
        let depth = if total_turns >= 7 { "deep" } else if total_turns >= 5 { "moderate" } else { "basic" };
        
        let reasoning_prefix = match depth {
            "deep" => format!("Let me work through this systematically. First, I'll analyze the core components of {}. Then, I'll evaluate the interdependencies and potential impact areas. Finally, I'll synthesize the findings into actionable recommendations.\n\n", topic),
            "moderate" => format!("Analyzing {} through multiple lenses: operational, strategic, and tactical perspectives.\n\n", topic),
            _ => String::new(),
        };
        
        let responses = [
            format!("{}Based on my analysis of {}, I've identified three key areas: efficiency, accuracy, and scalability. \n\nStep 1: Efficiency analysis reveals that current processes have 15% redundancy. \nStep 2: Accuracy metrics show 98.2% success rate with room for improvement in edge cases. \nStep 3: Scalability assessment indicates capacity for 3x growth without architectural changes. \n\nLet me elaborate on each area with specific recommendations.", reasoning_prefix, topic),
            
            format!("{}For {}, the recommended approach involves a phased implementation:\n\nPhase 1 (Weeks 1-2): Assessment and baseline establishment\n  - Current state documentation\n  - Stakeholder interviews\n  - Gap analysis\n\nPhase 2 (Weeks 3-6): Implementation\n  - Core functionality deployment\n  - Integration testing\n  - User acceptance validation\n\nPhase 3 (Weeks 7-8): Optimization\n  - Performance tuning\n  - Feedback incorporation\n  - Documentation completion\n\nSuccess metrics: 20% efficiency gain, 95% user satisfaction, <2% error rate.", reasoning_prefix, topic),
            
            format!("{}The {} data shows promising trends with nuanced insights:\n\nQuantitative Analysis:\n  - Performance metrics: +23% improvement year-over-year\n  - Error rates: Reduced from 4.2% to 1.8%\n  - Processing time: Average 2.3 seconds (down from 3.8s)\n\nQualitative Observations:\n  - User feedback indicates higher confidence in system reliability\n  - Support tickets decreased by 35%\n  - Feature adoption rate at 78%\n\nCorrelation Analysis:\n  - Strong positive correlation (r=0.82) between training and performance\n  - Negative correlation (r=-0.67) between complexity and adoption\n\nRecommendations based on this analysis follow.", reasoning_prefix, topic),
            
            format!("{}Regarding {}, I recommend prioritizing actions through a weighted scoring framework:\n\nPriority Matrix (Impact x Effort):\n1. Establish baseline metrics (High Impact, Low Effort) - Priority Score: 9.2\n   - Define KPIs for each process area\n   - Implement automated data collection\n   - Create real-time dashboards\n\n2. Implement monitoring systems (High Impact, Medium Effort) - Priority Score: 8.5\n   - Deploy observability tools\n   - Configure alerting thresholds\n   - Establish escalation procedures\n\n3. Iterate based on feedback (Medium Impact, Low Effort) - Priority Score: 7.8\n   - Weekly review cycles\n   - A/B testing framework\n   - Continuous improvement backlog\n\nResource requirements: 2 FTEs for 6 weeks, existing tooling sufficient.", reasoning_prefix, topic),
            
            format!("{}After comprehensive review of {}, I've synthesized findings across multiple dimensions:\n\nExecutive Summary:\n  - Overall health score: 8.2/10\n  - Key strength: Process standardization\n  - Primary opportunity: Automation potential\n  - Risk factor: Knowledge concentration\n\nDetailed Findings:\n\n1. Process Efficiency (Score: 8.5/10)\n   - Standardized workflows in place\n   - Clear ownership and accountability\n   - Opportunity: Reduce manual handoffs by 40%\n\n2. Technology Alignment (Score: 7.8/10)\n   - Modern tooling adopted\n   - Integration gaps identified\n   - Recommendation: API-first architecture\n\n3. Team Capability (Score: 8.0/10)\n   - Strong domain expertise\n   - Training program established\n   - Gap: Succession planning needed\n\nImplementation Timeline: 12-week roadmap with bi-weekly checkpoints.", reasoning_prefix, topic),
            
            format!("{}Breaking down {} into component parts for systematic analysis:\n\nRoot Cause Analysis:\n  - Primary driver: Process fragmentation across 3 systems\n  - Contributing factors: Manual data entry, inconsistent validation rules\n  - Impact cascade: Delays propagate to downstream dependencies\n\nSolution Architecture:\n  - Unified data layer connecting all systems\n  - Standardized validation framework\n  - Automated reconciliation process\n\nRisk Assessment:\n  - Implementation risk: Medium (requires cross-team coordination)\n  - Technical risk: Low (proven patterns available)\n  - Business risk: Low (incremental approach possible)\n\nMitigation Strategies:\n  - Phased rollout with rollback capability\n  - Parallel run period for validation\n  - Dedicated support team during transition\n\nExpected Outcome: 35% reduction in processing time, 50% fewer errors.", reasoning_prefix, topic),
            
            format!("{}Analyzing {} through a strategic lens:\n\nCurrent State Assessment:\n  - Maturity level: 3.5/5 (Defined and Measured)\n  - Automation coverage: 45%\n  - Documentation completeness: 78%\n\nTarget State Vision:\n  - Maturity target: 4.5/5 (Optimized)\n  - Automation target: 80%\n  - Documentation target: 95%\n\nGap Analysis:\n  - Skills gap: Advanced analytics capability\n  - Technology gap: Real-time processing infrastructure\n  - Process gap: Exception handling protocols\n\nTransformation Roadmap:\n  Q1: Foundation - Infrastructure upgrades, skill development\n  Q2: Acceleration - Automation deployment, process redesign\n  Q3: Optimization - Performance tuning, feedback loops\n  Q4: Scale - Expansion to adjacent processes\n\nInvestment Required: $250K (technology) + 3 FTEs (human capital)", reasoning_prefix, topic),
            
            format!("{}Evaluating {} with emphasis on stakeholder impact:\n\nStakeholder Analysis:\n  - Primary users: 150 daily active users\n  - Secondary users: 500 occasional users\n  - Support team: 12 agents\n\nPain Point Mapping:\n  - User frustration: 3.2/5 (moderate)\n  - Top complaint: Response time (42% of tickets)\n  - Secondary complaint: Navigation complexity (28% of tickets)\n\nProposed Solutions:\n  1. Performance optimization\n     - Database query tuning\n     - Caching layer implementation\n     - CDN deployment for static assets\n  \n  2. UX improvements\n     - Simplified navigation\n     - Contextual help integration\n     - Personalized dashboards\n\nExpected Impact:\n  - Response time: 40% improvement\n  - User satisfaction: +15 points\n  - Support tickets: -30% reduction", reasoning_prefix, topic),
        ];
        responses[(turn + variant) % responses.len()].clone()
    }
    
    fn generate_assistant_response(intent: IntentKind, domain: &str, entity: &str, context: &str) -> String {
        match intent {
            IntentKind::Question => format!("In the {} domain, {} considerations include compliance requirements, stakeholder alignment, and performance metrics. Would you like me to elaborate on any specific aspect?", domain, entity),
            IntentKind::Explain => format!("{} in {} operates through interconnected processes. The key components are: input validation, processing logic, output generation, and feedback loops. Each component has specific requirements and dependencies.", entity, domain),
            IntentKind::Analyze => format!("Analysis of {} in {} reveals several patterns: 40% relate to efficiency, 35% to quality, and 25% to cost optimization. The data suggests focusing on process standardization for maximum impact.", entity, domain),
            IntentKind::Summarize => format!("Summary of {} findings in {}: Key achievements include 25% efficiency improvement, reduced error rates by 15%, and enhanced stakeholder satisfaction. Areas for improvement include documentation and training.", entity, domain),
            IntentKind::Plan => format!("Here's a {} plan for {}: Phase 1 (weeks 1-2): Assessment and planning. Phase 2 (weeks 3-6): Implementation. Phase 3 (weeks 7-8): Testing and validation. Phase 4 (ongoing): Monitoring and optimization.", entity, domain),
            IntentKind::Help => format!("I can help you with {} in {}. Let's start by understanding your specific requirements. What are your primary objectives and constraints?", entity, domain),
            IntentKind::Compare => format!("Comparing {} approaches in {}: Option A offers lower cost but limited features. Option B provides comprehensive functionality at higher investment. Option C balances both with moderate cost and good feature coverage.", entity, domain),
            IntentKind::Recommend => format!("For {} in {}, I recommend: 1) Establish clear success criteria, 2) Implement phased rollout, 3) Monitor key performance indicators, 4) Iterate based on feedback. This approach minimizes risk while maximizing value.", entity, domain),
            IntentKind::Debug => format!("Debugging {} in {}: I've identified the root cause as a configuration mismatch. To resolve: 1) Update the configuration file, 2) Restart the affected services, 3) Verify the fix with test cases.", entity, domain),
            IntentKind::Verify => format!("Verification of {} in {}: All checks passed successfully. The system meets compliance requirements, performance benchmarks, and security standards. Documentation is complete and up to date.", entity, domain),
            IntentKind::Classify => format!("Classification of {} in {}: Category A (high priority): 15 items requiring immediate attention. Category B (medium): 30 items for scheduled review. Category C (low): 55 items for routine processing.", entity, domain),
            IntentKind::Extract => format!("Extracted {} data from {}: Key elements include: identifiers, timestamps, status codes, and metadata. The extraction yielded 150 valid records with 98% accuracy rate.", entity, domain),
            IntentKind::Brainstorm => format!("Brainstorming {} improvements for {}: Idea 1: Automate routine tasks. Idea 2: Implement real-time monitoring. Idea 3: Create self-service portals. Idea 4: Enhance collaboration tools. Idea 5: Develop predictive analytics.", entity, domain),
            IntentKind::Critique => format!("Critique of {} strategy in {}: Strengths include clear objectives and stakeholder buy-in. Areas for improvement: timeline could be more realistic, resource allocation needs refinement, and risk mitigation strategies should be more comprehensive.", entity, domain),
            IntentKind::Translate => format!("Translation of {} requirements to {} specifications: Functional requirements mapped to technical components. Non-functional requirements translated to performance criteria. Constraints documented as implementation guidelines.", entity, domain),
            IntentKind::Act => format!("Action executed for {} in {}: Task initiated successfully. Progress: 100% complete. Result: Operation finished without errors. Next scheduled action in 24 hours.", entity, domain),
            IntentKind::Clarify => format!("Clarification on {} for {}: The requirement specifies that all processes must comply with industry standards. This includes documentation, validation, and audit trail maintenance.", entity, domain),
            IntentKind::Rewrite => format!("Rewritten {} documentation for {}: Updated content includes clearer instructions, improved formatting, additional examples, and comprehensive troubleshooting guidance.", entity, domain),
            IntentKind::Continue => format!("Continuing {} processing in {}: Current progress: 65% complete. Remaining tasks: validation, documentation, and final review. Estimated completion: 2 hours.", entity, domain),
            IntentKind::Forget => format!("Context cleared for {} in {}. Starting fresh session. How can I assist you with new requirements?", entity, domain),
            IntentKind::Greeting => format!("Welcome! I'm ready to help you with {} in {}. What would you like to explore today?", context, domain),
            IntentKind::Gratitude => format!("You're welcome! I'm glad I could help with {} in {}. Feel free to reach out if you need further assistance.", entity, domain),
            IntentKind::Farewell => format!("Goodbye! Thank you for discussing {} in {}. Have a great day and feel free to return anytime.", entity, domain),
            IntentKind::Unknown => format!("I'm processing your request about {} in {}. Could you provide more specific details to help me assist you better?", entity, domain),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialogue_generator_creates_dialogues() {
        let mut gen = DialogueGenerator::new();
        
        let d = gen.create_dialogue_simple(
            IntentKind::Question,
            vec![
                ("user".to_string(), "What is the process?".to_string()),
                ("assistant".to_string(), "The process involves...".to_string()),
            ],
            "test",
            "simple",
            vec![],
        );
        
        assert_eq!(d.intent, "Question");
        assert_eq!(d.turns.len(), 2);
        assert!(d.expected_tone.is_some());
        assert!(d.resolver_mode.is_some());
    }

    #[test]
    fn test_dialogue_with_full_params() {
        let mut gen = DialogueGenerator::new();
        
        let d = gen.create_dialogue(
            IntentKind::Explain,
            Some(ToneKind::Technical),
            Some(ResolverMode::Balanced),
            vec![
                ("user".to_string(), "Explain the workflow".to_string()),
                ("assistant".to_string(), "The workflow consists of...".to_string()),
            ],
            "technical",
            "moderate",
            vec!["workflow".to_string()],
            vec!["Main".to_string(), "Intent".to_string()],
            vec!["Word".to_string(), "Phrase".to_string()],
        );
        
        assert_eq!(d.intent, "Explain");
        assert_eq!(d.expected_tone, Some("Technical".to_string()));
        assert_eq!(d.resolver_mode, Some("Balanced".to_string()));
        assert_eq!(d.metadata.memory_channels, vec!["Main", "Intent"]);
    }

    #[test]
    fn test_validate_dialogue_dataset() {
        let mut gen = DialogueGenerator::new();
        
        // Generate dialogues for multiple intents
        for intent in [IntentKind::Question, IntentKind::Explain, IntentKind::Help].iter() {
            for _ in 0..10 {
                let d = gen.create_dialogue_simple(
                    *intent,
                    vec![
                        ("user".to_string(), "Test question".to_string()),
                        ("assistant".to_string(), "Test answer with sufficient length".to_string()),
                    ],
                    "test",
                    "simple",
                    vec![],
                );
                gen.add_dialogue(d);
            }
        }
        
        let dataset = gen.build("test_dialogues");
        let metrics = validate_dialogue_dataset(&dataset);
        
        assert!(metrics.noise_ratio <= 0.05);
    }

    #[test]
    fn test_tone_resolver_inference() {
        // Test that tone/resolver inference works correctly
        assert_eq!(
            DialogueGenerator::infer_tone_resolver(IntentKind::Question),
            (ToneKind::NeutralProfessional, ResolverMode::Deterministic)
        );
        assert_eq!(
            DialogueGenerator::infer_tone_resolver(IntentKind::Brainstorm),
            (ToneKind::Empathetic, ResolverMode::Exploratory)
        );
        assert_eq!(
            DialogueGenerator::infer_tone_resolver(IntentKind::Greeting),
            (ToneKind::Casual, ResolverMode::Balanced)
        );
    }

    #[test]
    fn test_all_intents_have_templates() {
        let all_intents = [
            IntentKind::Greeting, IntentKind::Gratitude, IntentKind::Farewell,
            IntentKind::Help, IntentKind::Clarify, IntentKind::Rewrite,
            IntentKind::Verify, IntentKind::Continue, IntentKind::Forget,
            IntentKind::Question, IntentKind::Summarize, IntentKind::Explain,
            IntentKind::Compare, IntentKind::Extract, IntentKind::Analyze,
            IntentKind::Plan, IntentKind::Act, IntentKind::Recommend,
            IntentKind::Classify, IntentKind::Translate, IntentKind::Debug,
            IntentKind::Critique, IntentKind::Brainstorm, IntentKind::Unknown,
        ];
        
        for intent in all_intents {
            let templates = templates::get_templates_for_intent(intent);
            assert!(!templates.is_empty(), "No templates for {:?}", intent);
        }
    }
}
