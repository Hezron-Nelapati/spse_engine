// Classification Dataset Generator - Wrapper for existing classification_generator
// Provides ClassificationDatasetGenerator struct for consistency with other generators

use crate::seed::{CurriculumMetadata, QualityGates, TrainingExample};
use crate::types::{IntentKind, MemoryChannel, ResolverMode};

pub struct ClassificationDatasetGenerator {
    curriculum_score_base: f32,
}

impl ClassificationDatasetGenerator {
    pub fn new() -> Self {
        Self {
            curriculum_score_base: 0.80,
        }
    }

    /// Generate 100K+ classification examples
    /// Uses intent-specific templates for diverse coverage
    pub fn generate_full_dataset(&self) -> Vec<TrainingExample> {
        let mut examples = Vec::with_capacity(110000);

        // Generate examples for each intent kind
        let intents = [
            (
                IntentKind::Question,
                QUESTION_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Explain,
                EXPLAIN_TEMPLATES,
                ResolverMode::Balanced,
            ),
            (
                IntentKind::Compare,
                COMPARE_TEMPLATES,
                ResolverMode::Balanced,
            ),
            (
                IntentKind::Analyze,
                ANALYZE_TEMPLATES,
                ResolverMode::Balanced,
            ),
            (IntentKind::Plan, PLAN_TEMPLATES, ResolverMode::Balanced),
            (IntentKind::Debug, DEBUG_TEMPLATES, ResolverMode::Balanced),
            (
                IntentKind::Verify,
                VERIFY_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Summarize,
                SUMMARIZE_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Classify,
                CLASSIFY_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Recommend,
                RECOMMEND_TEMPLATES,
                ResolverMode::Balanced,
            ),
            (
                IntentKind::Extract,
                EXTRACT_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Critique,
                CRITIQUE_TEMPLATES,
                ResolverMode::Exploratory,
            ),
            (
                IntentKind::Brainstorm,
                BRAINSTORM_TEMPLATES,
                ResolverMode::Exploratory,
            ),
            (IntentKind::Help, HELP_TEMPLATES, ResolverMode::Balanced),
            (
                IntentKind::Greeting,
                GREETING_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Farewell,
                FAREWELL_TEMPLATES,
                ResolverMode::Deterministic,
            ),
            (
                IntentKind::Gratitude,
                GRATITUDE_TEMPLATES,
                ResolverMode::Deterministic,
            ),
        ];

        let domains = [
            "science",
            "technology",
            "business",
            "healthcare",
            "education",
            "finance",
            "engineering",
            "mathematics",
            "history",
            "arts",
            "physics",
            "chemistry",
            "biology",
            "computing",
            "economics",
            "psychology",
            "sociology",
            "philosophy",
            "literature",
            "music",
        ];
        let topics = [
            "algorithms",
            "processes",
            "systems",
            "methods",
            "concepts",
            "principles",
            "frameworks",
            "models",
            "theories",
            "techniques",
            "strategies",
            "patterns",
            "architectures",
            "implementations",
            "solutions",
            "approaches",
            "paradigms",
            "mechanisms",
            "protocols",
            "standards",
        ];

        // Target ~6K examples per intent (17 intents * 6K = 102K)
        let per_intent = 6000;

        for (intent, templates, resolver) in intents {
            for i in 0..per_intent {
                let template = templates[i % templates.len()];
                let domain = domains[i % domains.len()];
                let topic = topics[i % topics.len()];

                let question = template
                    .replace("{topic}", topic)
                    .replace("{domain}", domain);

                let answer = format!("Response about {} in the context of {}.", topic, domain);

                examples.push(TrainingExample {
                    question,
                    answer,
                    context: Some(format!("classification:{:?}:{:?}", intent, resolver)),
                    reasoning: None,
                    intent: Some(format!("{:?}", intent)),
                    entities: vec![topic.to_string(), domain.to_string()],
                    channels: vec![MemoryChannel::Intent],
                    curriculum: CurriculumMetadata {
                        curriculum_score: ((self.curriculum_score_base + 0.05) * 100.0) as i32,
                        phase_hint: crate::types::TrainingPhaseKind::Bootstrap,
                        target_memory: crate::types::MemoryType::Episodic,
                        memory_channels: vec![MemoryChannel::Intent],
                        suggested_batch_size: 500,
                        max_chunk_chars: 8000,
                    },
                    quality_gates: QualityGates {
                        min_unit_discovery_efficiency: Some(0.80),
                        min_semantic_routing_accuracy: Some(0.75),
                        min_corroboration_count: 1,
                    },
                    training_options: crate::types::TrainingOptions::default(),
                });
            }
        }

        examples
    }
}

impl Default for ClassificationDatasetGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// Intent-specific query templates
const QUESTION_TEMPLATES: &[&str] = &[
    "What is {topic} in {domain}?",
    "How does {topic} work?",
    "Why is {topic} important in {domain}?",
    "What causes {topic}?",
    "When was {topic} discovered?",
];

const EXPLAIN_TEMPLATES: &[&str] = &[
    "Explain {topic} in simple terms.",
    "Explain how {topic} works in {domain}.",
    "Can you explain {topic} to a beginner?",
    "Walk me through {topic} step by step.",
    "Explain why {topic} matters in {domain}.",
];

const COMPARE_TEMPLATES: &[&str] = &[
    "Compare {topic} and related approaches in {domain}.",
    "What are the differences between {topic} methods?",
    "How does {topic} compare to alternatives?",
    "Compare the pros and cons of {topic}.",
    "Contrast {topic} with traditional approaches.",
];

const ANALYZE_TEMPLATES: &[&str] = &[
    "Analyze the impact of {topic} on {domain}.",
    "What are the strengths and weaknesses of {topic}?",
    "Perform a detailed analysis of {topic}.",
    "Evaluate the effectiveness of {topic}.",
    "Assess the risks of {topic} in {domain}.",
];

const PLAN_TEMPLATES: &[&str] = &[
    "Plan a strategy for {topic} in {domain}.",
    "How should I approach {topic}?",
    "Create a roadmap for implementing {topic}.",
    "What steps should I take for {topic}?",
    "Design a plan to improve {topic}.",
    "Plan a trip to {topic}.",
    "Make a plan for {topic}.",
    "I need to plan {topic}.",
    "Help me plan out {topic}.",
    "Outline a plan for {topic}.",
    "Let's plan how to tackle {topic}.",
    "Create a step-by-step plan for {topic}.",
    "What's the best plan for {topic}?",
    "Help me create an action plan for {topic}.",
    "I want to plan my approach to {topic}.",
    "Draft a plan for {topic} implementation.",
    "Schedule a plan for {topic}.",
    "Develop a planning strategy for {topic}.",
];

const DEBUG_TEMPLATES: &[&str] = &[
    "Debug this issue with {topic}.",
    "Why isn't {topic} working correctly?",
    "Troubleshoot {topic} in {domain}.",
    "Find the root cause of the {topic} problem.",
    "Help me fix the {topic} issue.",
    "Debug this code error in {topic}.",
    "Debug the {topic} bug.",
    "There's an error in {topic}.",
    "Fix this bug in {topic}.",
    "My {topic} code is broken.",
    "I'm getting an error with {topic}.",
    "The {topic} function throws an exception.",
    "Debugging: {topic} crashes on startup.",
    "Stack trace shows error in {topic}.",
    "Runtime error in {topic} module.",
    "Help debug this {topic} failure.",
    "Console shows {topic} error.",
    "Segmentation fault in {topic}.",
];

const VERIFY_TEMPLATES: &[&str] = &[
    "Is it true that {topic} is fundamental to {domain}?",
    "Verify this claim about {topic}.",
    "Can you fact-check this about {topic}?",
    "Is the understanding of {topic} correct?",
    "Validate the assumption about {topic}.",
    "Is this statement about {topic} accurate?",
    "Confirm whether {topic} is correct.",
    "Double-check this {topic} assertion.",
    "Is my understanding of {topic} right?",
    "Verify: does {topic} actually work this way?",
    "Fact-check: {topic} in {domain}.",
    "Is this {topic} claim valid?",
    "Check if {topic} statement is true.",
];

const SUMMARIZE_TEMPLATES: &[&str] = &[
    "Summarize {topic} in {domain}.",
    "Give me a brief overview of {topic}.",
    "What are the key points of {topic}?",
    "TLDR on {topic}.",
    "Condense the main ideas of {topic}.",
    "Summarize this article about {topic}.",
    "Sum up {topic} for me.",
    "Give me a summary of {topic}.",
    "Briefly summarize {topic}.",
    "Can you summarize {topic}?",
];

const CLASSIFY_TEMPLATES: &[&str] = &[
    "Classify {topic} within the {domain} taxonomy.",
    "What category does {topic} fall under?",
    "How would you categorize {topic}?",
    "Where does {topic} fit in {domain}?",
    "Is {topic} a type of something else?",
];

const RECOMMEND_TEMPLATES: &[&str] = &[
    "What do you recommend for {topic}?",
    "Suggest the best approach to {topic}.",
    "What's the best {topic} for {domain}?",
    "Recommend a solution for {topic}.",
    "What would you suggest for {topic}?",
];

const EXTRACT_TEMPLATES: &[&str] = &[
    "Extract the key information about {topic}.",
    "Pull out the main facts about {topic}.",
    "What are the essential details of {topic}?",
    "Identify the key elements of {topic}.",
    "List the important aspects of {topic}.",
];

const CRITIQUE_TEMPLATES: &[&str] = &[
    "Critique the {topic} approach.",
    "What are the flaws of {topic}?",
    "Provide a critical review of {topic}.",
    "What could be improved about {topic}?",
    "Evaluate the weaknesses of {topic}.",
];

const BRAINSTORM_TEMPLATES: &[&str] = &[
    "Brainstorm ideas for {topic}.",
    "What are creative approaches to {topic}?",
    "Generate ideas for improving {topic}.",
    "Think of innovative solutions for {topic}.",
    "Come up with alternatives to {topic}.",
];

const HELP_TEMPLATES: &[&str] = &[
    "Help me with {topic}.",
    "I need assistance with {topic}.",
    "Can you help me understand {topic}?",
    "I'm stuck on {topic}. Can you help?",
    "I need help figuring out {topic}.",
];

const GREETING_TEMPLATES: &[&str] = &["Hello!", "Hi there!", "Good morning!", "Hey!", "Greetings!"];

const FAREWELL_TEMPLATES: &[&str] = &[
    "Goodbye!",
    "See you later!",
    "Bye!",
    "Take care!",
    "Until next time!",
    "I'm leaving now.",
    "Got to go!",
    "Bye bye!",
    "See ya!",
    "I'm off!",
    "Leaving now, bye!",
    "Time to go!",
    "Catch you later!",
    "I'm heading out.",
    "Talk to you later!",
];

const GRATITUDE_TEMPLATES: &[&str] = &[
    "Thank you!",
    "Thanks!",
    "I appreciate it!",
    "Thanks a lot!",
    "Much appreciated!",
    "Thanks for your help!",
    "Thank you so much!",
    "Thanks for the information!",
    "I really appreciate your help!",
    "Thanks, that was helpful!",
    "That was very helpful, thank you!",
    "I'm grateful for your assistance!",
    "Thanks for explaining that!",
    "You've been very helpful, thanks!",
    "I appreciate your explanation!",
    "Thank you for the clarification!",
    "Thanks for taking the time!",
    "I'm thankful for your help!",
    "Great answer, thanks!",
    "Perfect, thank you!",
];
