//! Classification pattern seed generator.
//!
//! Produces training examples specifically designed to populate the Intent memory
//! channel with diverse classification patterns. Each example maps a query to its
//! intent, tone, and resolver mode, giving the spatial index enough coverage for
//! accurate classification via weighted-vote aggregation.

use crate::seed::bulk_generator::{
    self, expand_template, human_bytes, pick, pick_idx, pick_str, seeded_rng, topics_for_domain,
    JsonlWriter, ANSWER_BODIES, ANSWER_PREFIXES, DOMAINS,
    GREETING_VARIATIONS, GRATITUDE_VARIATIONS, FAREWELL_VARIATIONS,
};
use crate::seed::TrainingExample;
use crate::types::{IntentKind, MemoryChannel, ToneKind, ResolverMode};
use rand::Rng;
use std::path::Path;

// ============================================================================
// INTENT-SPECIFIC QUERY TEMPLATES
// ============================================================================

const QUESTION_QUERIES: &[&str] = &[
    "What is {} in {}?",
    "How does {} work?",
    "Why is {} important in {}?",
    "What causes {} in {}?",
    "When was {} discovered?",
    "Where is {} used most often in {}?",
    "What are the key facts about {}?",
    "Who invented {}?",
    "How many types of {} exist in {}?",
    "What is the difference between {} and related concepts?",
];

const EXPLAIN_QUERIES: &[&str] = &[
    "Explain {} in simple terms.",
    "Explain how {} works in {}.",
    "Can you explain the concept of {} to a beginner?",
    "Walk me through {} step by step.",
    "Explain why {} matters in {}.",
    "Help me understand {} in the context of {}.",
    "Explain the mechanism behind {} in {}.",
    "Break down {} for someone new to {}.",
    "Explain the theory of {} in {}.",
    "Give a detailed explanation of {} in {}.",
];

const COMPARE_QUERIES: &[&str] = &[
    "Compare {} and related approaches in {}.",
    "What are the differences between {} methods in {}?",
    "How does {} compare to alternatives in {}?",
    "Which is better: {} or the traditional approach in {}?",
    "Compare the pros and cons of {} in {}.",
    "What are the trade-offs of {} vs other methods in {}?",
    "Contrast {} with its predecessor in {}.",
    "How does the new {} stack up against the old in {}?",
];

const ANALYZE_QUERIES: &[&str] = &[
    "Analyze the impact of {} on {}.",
    "What are the strengths and weaknesses of {} in {}?",
    "Perform a detailed analysis of {} in {}.",
    "Evaluate the effectiveness of {} in {}.",
    "Assess the risks of {} in {}.",
    "What does the data say about {} in {}?",
    "Critically examine {} in the context of {}.",
    "Analyze the trends in {} within {}.",
];

const PLAN_QUERIES: &[&str] = &[
    "Plan a strategy for {} in {}.",
    "How should I approach {} in {}?",
    "Create a roadmap for implementing {} in {}.",
    "What steps should I take to {} in {}?",
    "Design a plan to improve {} in {}.",
    "Outline a procedure for {} in {}.",
    "How would you plan {} for a {} project?",
    "Develop a timeline for {} in {}.",
];

const DEBUG_QUERIES: &[&str] = &[
    "Debug this issue with {} in {}.",
    "Why isn't {} working correctly in {}?",
    "Troubleshoot {} in the context of {}.",
    "Find the root cause of the {} problem in {}.",
    "What's wrong with this {} approach in {}?",
    "Help me fix the {} issue in {}.",
    "Diagnose the {} failure in {}.",
    "Why does {} produce unexpected results in {}?",
];

const VERIFY_QUERIES: &[&str] = &[
    "Is it true that {} is fundamental to {}?",
    "Verify this claim about {} in {}.",
    "Can you fact-check this information about {} in {}?",
    "Is the common understanding of {} correct in {}?",
    "Validate the assumption that {} is well-established in {}.",
    "Check whether the standard view of {} holds true in {}.",
    "Is the textbook description of {} accurate in {}?",
    "Confirm or deny that {} is essential to {}.",
];

const SUMMARIZE_QUERIES: &[&str] = &[
    "Summarize {} in {}.",
    "Give me a brief overview of {} in {}.",
    "What are the key points of {} in {}?",
    "TLDR on {} in {}.",
    "Condense the main ideas of {} in {}.",
    "Provide a summary of {} for {}.",
    "What's the gist of {} in {}?",
    "Briefly describe {} in {}.",
];

const CLASSIFY_QUERIES: &[&str] = &[
    "Classify {} within the {} taxonomy.",
    "What category does {} fall under in {}?",
    "How would you categorize {} in {}?",
    "Where does {} fit in the {} classification system?",
    "Is {} a type of {} or something else?",
    "Group {} with related concepts in {}.",
    "What class does {} belong to in {}?",
    "Categorize the different types of {} in {}.",
];

const RECOMMEND_QUERIES: &[&str] = &[
    "What do you recommend for {} in {}?",
    "Suggest the best approach to {} in {}.",
    "What's the best {} for {} use cases?",
    "Recommend a solution for {} in {}.",
    "What would you suggest for improving {} in {}?",
    "Which {} approach works best for {}?",
    "Advise me on {} in the context of {}.",
    "What's your recommendation for {} in {}?",
];

const EXTRACT_QUERIES: &[&str] = &[
    "Extract the key information about {} from {}.",
    "Pull out the main facts about {} in {}.",
    "What are the essential details of {} in {}?",
    "Identify the key elements of {} in {}.",
    "Extract the relevant data about {} in {}.",
    "List the important aspects of {} in {}.",
    "What specific information about {} can be found in {}?",
    "Isolate the core components of {} in {}.",
];

const CRITIQUE_QUERIES: &[&str] = &[
    "Critique the {} approach in {}.",
    "What are the flaws of {} in {}?",
    "Provide a critical review of {} in {}.",
    "What could be improved about {} in {}?",
    "Evaluate the weaknesses of {} in {}.",
    "What criticism exists for {} in {}?",
    "Point out the problems with {} in {}.",
    "Give a balanced critique of {} in {}.",
];

const BRAINSTORM_QUERIES: &[&str] = &[
    "Brainstorm ideas for {} in {}.",
    "What are creative approaches to {} in {}?",
    "Generate ideas for improving {} in {}.",
    "Think of innovative solutions for {} in {}.",
    "Come up with alternatives to {} in {}.",
    "Ideate around {} in the context of {}.",
    "What unconventional approaches to {} exist in {}?",
    "Explore new possibilities for {} in {}.",
];

const TRANSLATE_QUERIES: &[&str] = &[
    "Translate the concept of {} into {} terms.",
    "How would you express {} in {} language?",
    "Convert {} from {} jargon to plain language.",
    "Reframe {} in the context of {}.",
    "Put {} into {} terminology.",
    "How does {} map to {} concepts?",
    "Express {} using {} vocabulary.",
    "Translate {} from technical to {} perspective.",
];

const ACT_QUERIES: &[&str] = &[
    "Implement {} in {}.",
    "Execute the {} plan for {}.",
    "Apply {} to this {} scenario.",
    "Do {} in the context of {}.",
    "Carry out {} for {}.",
    "Perform {} in {}.",
    "Run {} against the {} parameters.",
    "Put {} into action in {}.",
];

const HELP_QUERIES: &[&str] = &[
    "Help me with {} in {}.",
    "I need assistance with {} in {}.",
    "Can you help me understand {} in {}?",
    "I'm stuck on {} in {}. Can you help?",
    "I need help figuring out {} in {}.",
    "Could you assist me with {} in {}?",
    "I'm having trouble with {} in {}.",
    "Please help me with {} in {}.",
];

const CLARIFY_QUERIES: &[&str] = &[
    "Can you clarify {} in {}?",
    "I don't understand {} in {}. Can you clarify?",
    "What exactly do you mean by {} in {}?",
    "Could you be more specific about {} in {}?",
    "I need clarification on {} in {}.",
    "Please elaborate on {} in {}.",
    "What do you mean when you say {} in {}?",
    "Can you clarify the {} point about {}?",
];

const REWRITE_QUERIES: &[&str] = &[
    "Rewrite this description of {} in {} style.",
    "Rephrase {} for a {} audience.",
    "Reword the {} explanation for {}.",
    "Write a better version of the {} description in {}.",
    "Revise the {} section for {} clarity.",
    "Can you rewrite {} more concisely for {}?",
    "Reformulate {} in {} terms.",
    "Edit the {} passage for {} standards.",
];

// ============================================================================
// TONE-SPECIFIC MODIFIERS
// ============================================================================

const TONE_MAP: &[(ToneKind, &[&str])] = &[
    (ToneKind::NeutralProfessional, &[
        "", "", "", "", "",
    ]),
    (ToneKind::Technical, &[
        "Technically speaking, ", "From an engineering perspective, ",
        "In precise terms, ", "From a technical standpoint, ",
        "Rigorously speaking, ",
    ]),
    (ToneKind::Casual, &[
        "Hey — ", "So, ", "Quick question — ", "Just wondering — ", "BTW, ",
    ]),
    (ToneKind::Empathetic, &[
        "I'd really appreciate help here. ", "This is important to me. ",
        "I'm struggling with this one. ", "If you can help: ",
        "I'd be grateful for insight. ",
    ]),
    (ToneKind::Direct, &[
        "", "", "", "", "",
    ]),
    (ToneKind::Formal, &[
        "If I may ask: ", "I would like to inquire. ",
        "For your consideration: ", "Respectfully, ",
        "I'd appreciate a thorough answer. ",
    ]),
];

// ============================================================================
// BULK GENERATION
// ============================================================================

fn templates_for_intent(intent: IntentKind) -> &'static [&'static str] {
    match intent {
        IntentKind::Question => QUESTION_QUERIES,
        IntentKind::Explain => EXPLAIN_QUERIES,
        IntentKind::Compare => COMPARE_QUERIES,
        IntentKind::Analyze => ANALYZE_QUERIES,
        IntentKind::Plan => PLAN_QUERIES,
        IntentKind::Debug => DEBUG_QUERIES,
        IntentKind::Verify => VERIFY_QUERIES,
        IntentKind::Summarize => SUMMARIZE_QUERIES,
        IntentKind::Classify => CLASSIFY_QUERIES,
        IntentKind::Recommend => RECOMMEND_QUERIES,
        IntentKind::Extract => EXTRACT_QUERIES,
        IntentKind::Critique => CRITIQUE_QUERIES,
        IntentKind::Brainstorm => BRAINSTORM_QUERIES,
        IntentKind::Translate => TRANSLATE_QUERIES,
        IntentKind::Act => ACT_QUERIES,
        IntentKind::Help => HELP_QUERIES,
        IntentKind::Clarify => CLARIFY_QUERIES,
        IntentKind::Rewrite => REWRITE_QUERIES,
        // Social intents handled separately
        _ => QUESTION_QUERIES,
    }
}

fn resolver_for_intent(intent: IntentKind) -> ResolverMode {
    match intent {
        IntentKind::Question | IntentKind::Verify | IntentKind::Extract
        | IntentKind::Classify | IntentKind::Summarize => ResolverMode::Deterministic,
        IntentKind::Brainstorm | IntentKind::Critique | IntentKind::Rewrite => ResolverMode::Exploratory,
        _ => ResolverMode::Balanced,
    }
}

/// Generate bulk classification pattern data, streaming to a JSONL file.
/// Returns (examples_written, bytes_written).
pub fn generate_bulk_classification(output_path: &Path, target_bytes: u64, seed: u64) -> (u64, u64) {
    let mut rng = seeded_rng(seed);
    let mut writer = JsonlWriter::new(output_path).expect("create classification JSONL");
    let mut count: u64 = 0;

    let knowledge_intents = [
        IntentKind::Question, IntentKind::Explain, IntentKind::Compare,
        IntentKind::Analyze, IntentKind::Plan, IntentKind::Debug,
        IntentKind::Verify, IntentKind::Summarize, IntentKind::Classify,
        IntentKind::Recommend, IntentKind::Extract, IntentKind::Critique,
        IntentKind::Brainstorm, IntentKind::Translate, IntentKind::Act,
        IntentKind::Help, IntentKind::Clarify, IntentKind::Rewrite,
    ];


    // Phase 1: Knowledge-intent classification patterns (~92% of budget)
    let phase1_target = target_bytes * 92 / 100;
    while writer.bytes_written() < phase1_target {
        let intent = pick(&mut rng, &knowledge_intents);
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick_str(&mut rng, topics);

        let templates = templates_for_intent(*intent);
        let template = pick_str(&mut rng, templates);
        let base_question = template
            .replacen("{}", topic, 1)
            .replacen("{}", domain, 1);

        // Apply tone modifier with grammatical lowercase
        let (tone, tone_prefixes) = pick(&mut rng, TONE_MAP);
        let tone_prefix = pick_str(&mut rng, tone_prefixes);
        let question = if tone_prefix.is_empty() {
            base_question.clone()
        } else {
            let mut chars = base_question.chars();
            let lower_first: String = chars.next()
                .map(|c| c.to_lowercase().to_string())
                .unwrap_or_default();
            format!("{}{}{}", tone_prefix, lower_first, chars.as_str())
        };

        let resolver = resolver_for_intent(*intent);

        // Generate a concise answer appropriate for classification
        let t_prefix = pick_str(&mut rng, ANSWER_PREFIXES);
        let prefix = expand_template(&mut rng, t_prefix, domain, topic);
        let t_body = pick_str(&mut rng, ANSWER_BODIES);
        let body = expand_template(&mut rng, t_body, domain, topic);
        let answer = format!("{} {}", prefix, body);

        let example = TrainingExample {
            question,
            answer,
            context: Some(format!(
                "classification:{}:{}:{}",
                format!("{:?}", intent),
                format!("{:?}", tone),
                format!("{:?}", resolver),
            )),
            reasoning: None,
            intent: Some(format!("{:?}", intent)),
            entities: vec![topic.to_string(), domain.to_string()],
            channels: vec![MemoryChannel::Intent],
            curriculum: crate::seed::CurriculumMetadata {
                curriculum_score: rng.gen_range(100..125),
                memory_channels: vec![MemoryChannel::Intent],
                ..Default::default()
            },
            quality_gates: Default::default(),
            training_options: Default::default(),
        };

        writer.write_example(&example).expect("write classification");
        count += 1;

        if count % 100_000 == 0 {
            eprintln!(
                "  classification: {} examples, {}",
                count,
                human_bytes(writer.bytes_written())
            );
        }
    }

    // Phase 2: Social intent classification with domain context (~3% of budget)
    let phase2_target = target_bytes * 95 / 100;
    while writer.bytes_written() < phase2_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topic = pick_str(&mut rng, topics_for_domain(domain_idx));

        // Build a full answer body for density
        let t_prefix = pick_str(&mut rng, ANSWER_PREFIXES);
        let prefix = expand_template(&mut rng, t_prefix, domain, topic);
        let t_body = pick_str(&mut rng, ANSWER_BODIES);
        let body = expand_template(&mut rng, t_body, domain, topic);
        let knowledge = format!("{} {}", prefix, body);

        // Greetings with domain context
        let (gq, ga) = pick(&mut rng, GREETING_VARIATIONS);
        let greeting_q = format!("{} — I'd like to explore {} in {}.", gq, topic, domain);
        let greeting_answer = format!("{} Let me help you with {} in {}. {}", ga, topic, domain, knowledge);
        let example = TrainingExample::qa(&greeting_q, &greeting_answer)
            .with_intent(IntentKind::Greeting)
            .with_entities(vec![topic.to_string(), domain.to_string()])
            .with_context(&format!("classification:Greeting:Casual:Balanced:{}", domain))
            .with_curriculum_score(130);
        writer.write_example(&example).expect("write greeting class");
        count += 1;

        // Gratitude with recap
        let (tq, _ta) = pick(&mut rng, GRATITUDE_VARIATIONS);
        let gratitude_q = format!("{} — your explanation of {} in {} was clear.", tq, topic, domain);
        let gratitude_answer = format!("You're welcome! To summarize what we covered about {} in {}: {} Feel free to ask more.", topic, domain, knowledge);
        let example = TrainingExample::qa(&gratitude_q, &gratitude_answer)
            .with_intent(IntentKind::Gratitude)
            .with_entities(vec![topic.to_string(), domain.to_string()])
            .with_context(&format!("classification:Gratitude:Casual:Balanced:{}", domain))
            .with_curriculum_score(130);
        writer.write_example(&example).expect("write gratitude class");
        count += 1;

        // Farewell with summary
        let (fq, _fa) = pick(&mut rng, FAREWELL_VARIATIONS);
        let farewell_q = format!("{} — that's all I needed about {} in {}.", fq, topic, domain);
        let farewell_answer = format!("Goodbye! Here's a final recap on {} in {}: {} Have a great day!", topic, domain, knowledge);
        let example = TrainingExample::qa(&farewell_q, &farewell_answer)
            .with_intent(IntentKind::Farewell)
            .with_entities(vec![topic.to_string(), domain.to_string()])
            .with_context(&format!("classification:Farewell:Casual:Balanced:{}", domain))
            .with_curriculum_score(130);
        writer.write_example(&example).expect("write farewell class");
        count += 1;
    }

    // Phase 3: Continue/Forget/Unknown edge cases (~5% of budget)
    while writer.bytes_written() < target_bytes {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topic = pick_str(&mut rng, topics_for_domain(domain_idx));

        let t_body = pick_str(&mut rng, ANSWER_BODIES);
        let body = expand_template(&mut rng, t_body, domain, topic);

        let roll: f32 = rng.gen();
        if roll < 0.4 {
            let question = format!("Continue with the {} analysis in {}.", topic, domain);
            let answer = format!("Continuing the {} analysis in {}. Next steps: {} The deeper examination reveals additional factors.", topic, domain, body);
            let example = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Continue)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("classification:Continue:NeutralProfessional:Balanced:{}", domain))
                .with_curriculum_score(110);
            writer.write_example(&example).expect("write continue class");
        } else if roll < 0.7 {
            let question = format!("Forget the previous {} discussion about {}.", domain, topic);
            let answer = format!("Previous context about {} in {} has been cleared. For reference, here is what was discussed: {} Starting fresh.", topic, domain, body);
            let example = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Forget)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("classification:Forget:Direct:Deterministic:{}", domain))
                .with_curriculum_score(110);
            writer.write_example(&example).expect("write forget class");
        } else {
            let detail = pick_str(&mut rng, bulk_generator::DETAIL_POOLS);
            let question = format!("What about {} and {} together in {}?", topic, detail, domain);
            let answer = format!("That's an interesting but ambiguous question. Regarding {} in {}: {} Could you clarify what specific aspect you'd like to explore?", topic, domain, body);
            let example = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Unknown)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("classification:Unknown:NeutralProfessional:Balanced:{}", domain))
                .with_curriculum_score(90);
            writer.write_example(&example).expect("write unknown class");
        }
        count += 1;
    }

    writer.flush().expect("flush classification JSONL");
    (count, writer.bytes_written())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_knowledge_intents_have_templates() {
        let intents = [
            IntentKind::Question, IntentKind::Explain, IntentKind::Compare,
            IntentKind::Analyze, IntentKind::Plan, IntentKind::Debug,
            IntentKind::Verify, IntentKind::Summarize, IntentKind::Classify,
            IntentKind::Recommend, IntentKind::Extract, IntentKind::Critique,
            IntentKind::Brainstorm, IntentKind::Translate, IntentKind::Act,
            IntentKind::Help, IntentKind::Clarify, IntentKind::Rewrite,
        ];
        for intent in intents {
            let t = templates_for_intent(intent);
            assert!(t.len() >= 5, "{:?} has only {} templates", intent, t.len());
        }
    }

    #[test]
    fn tone_map_covers_all_tones() {
        assert_eq!(TONE_MAP.len(), 6, "expected 6 tone entries");
    }
}
