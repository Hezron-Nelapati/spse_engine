//! Intelligence-focused seed generator.
//!
//! Produces training examples that teach the engine *how to think*, not *what to know*.
//! Two modes:
//!
//! 1. **Handcrafted seeds** (`generate_intelligence_seeds()`) — small curated set for tests/bootstrap
//! 2. **Bulk generation** (`generate_bulk_intelligence()`) — procedural expansion to ~1GB via
//!    domain×topic×template×complexity combinatorics streamed to JSONL
//!
//! Five pattern categories: reasoning chains, retrieval triggers, confidence gating,
//! multi-hop reasoning, self-correction.

use crate::seed::bulk_generator::{
    self, expand_template, human_bytes, pick, pick_idx, pick_str, seeded_rng, topics_for_domain,
    JsonlWriter, ANSWER_BODIES, ANSWER_PREFIXES, DETAIL_POOLS, DOMAINS, REASONING_CALCULATIONS,
    REASONING_CONCLUSIONS, REASONING_INFERENCES, REASONING_PREMISES, REASONING_VERIFICATIONS,
};
use crate::seed::TrainingExample;
use crate::types::{
    IntentKind, MemoryChannel, ReasoningStep, ReasoningStepType, ReasoningTrace, ReasoningType,
};
use rand::Rng;
use std::path::Path;

// ============================================================================
// BULK GENERATION (streaming to JSONL, ~1GB target)
// ============================================================================

/// Intent kinds used for bulk generation with their associated reasoning types.
const INTENT_REASONING_MAP: &[(IntentKind, ReasoningType)] = &[
    (IntentKind::Question, ReasoningType::General),
    (IntentKind::Explain, ReasoningType::Explanatory),
    (IntentKind::Compare, ReasoningType::Explanatory),
    (IntentKind::Analyze, ReasoningType::Logical),
    (IntentKind::Plan, ReasoningType::Planning),
    (IntentKind::Debug, ReasoningType::Debugging),
    (IntentKind::Verify, ReasoningType::Logical),
    (IntentKind::Summarize, ReasoningType::General),
    (IntentKind::Classify, ReasoningType::Logical),
    (IntentKind::Recommend, ReasoningType::Planning),
    (IntentKind::Extract, ReasoningType::General),
    (IntentKind::Critique, ReasoningType::Logical),
    (IntentKind::Brainstorm, ReasoningType::Planning),
    (IntentKind::Translate, ReasoningType::General),
    (IntentKind::Act, ReasoningType::Planning),
];

const RETRIEVAL_TRIGGER_CONTEXTS: &[&str] = &[
    "retrieval_trigger:current_data",
    "retrieval_trigger:live_statistics",
    "retrieval_trigger:recent_events",
    "retrieval_trigger:version_specific",
    "retrieval_trigger:price_lookup",
    "retrieval_trigger:availability_check",
    "retrieval_trigger:regulatory_update",
    "retrieval_trigger:medical_info",
    "retrieval_trigger:legal_reference",
    "retrieval_trigger:technical_docs",
];

const RETRIEVAL_QUESTION_PREFIXES: &[&str] = &[
    "What is the current",
    "What are the latest",
    "How much does",
    "What happened recently in",
    "What is the most recent update on",
    "Where can I find current information about",
    "What are today's",
    "Who currently holds the record for",
    "What is the up-to-date status of",
    "What are the newest developments in",
];

const RETRIEVAL_ANSWER_PREFIXES: &[&str] = &[
    "Let me look that up for you.",
    "Let me check the latest information.",
    "I'll search for the most current data on that.",
    "Let me verify with up-to-date sources.",
    "I need to retrieve current information for that.",
    "Let me find the latest data.",
    "I'll check the most recent sources.",
    "That requires current data — let me search.",
];

const CONFIDENCE_HIGH_TEMPLATES: &[&str] = &[
    "I can answer this directly from {domain} principles.",
    "This is a well-established concept in {domain}.",
    "High confidence — this follows standard {domain} theory.",
    "I can compute/derive this without external sources.",
    "This is definitional in {domain}; no retrieval needed.",
];

const CONFIDENCE_LOW_TEMPLATES: &[&str] = &[
    "This requires specialized {domain} expertise I should verify.",
    "Low confidence — {domain} details change frequently.",
    "I should cross-reference this with authoritative {domain} sources.",
    "This is jurisdiction/context-specific; let me check.",
    "Safety-critical {domain} information — retrieval recommended.",
];

const SELF_CORRECTION_MARKERS: &[&str] = &[
    "Wait — let me reconsider.",
    "Actually, that initial reasoning was flawed.",
    "Hold on — I need to correct my approach.",
    "Let me re-examine this; my first answer was too hasty.",
    "Correction: the initial step had an error.",
];

/// Generate bulk intelligence seeds, streaming to a JSONL file.
/// Returns the number of examples written and total bytes.
pub fn generate_bulk_intelligence(output_path: &Path, target_bytes: u64, seed: u64) -> (u64, u64) {
    let mut rng = seeded_rng(seed);
    let mut writer = JsonlWriter::new(output_path).expect("create intelligence JSONL");
    let mut count: u64 = 0;

    // Phase 1: Reasoning chains across all domains (~30% of budget)
    let phase1_target = target_bytes * 30 / 100;
    while writer.bytes_written() < phase1_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);
        let (intent, reasoning_type) = pick(&mut rng, INTENT_REASONING_MAP);

        let example = generate_reasoning_chain(&mut rng, domain, topic, *intent, *reasoning_type);
        writer.write_example(&example).expect("write reasoning");
        count += 1;

        if count % 100_000 == 0 {
            eprintln!(
                "  intelligence phase1: {} examples, {}",
                count,
                human_bytes(writer.bytes_written())
            );
        }
    }

    // Phase 2: Retrieval triggers (~12% of budget)
    let phase2_target = target_bytes * 42 / 100;
    while writer.bytes_written() < phase2_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);

        let example = generate_retrieval_trigger(&mut rng, domain, topic);
        writer.write_example(&example).expect("write retrieval");
        count += 1;
    }

    // Phase 3: Confidence gating (~10% of budget)
    let phase3_target = target_bytes * 52 / 100;
    while writer.bytes_written() < phase3_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);
        let high_confidence: bool = rng.gen_bool(0.5);

        let example = generate_confidence_gate(&mut rng, domain, topic, high_confidence);
        writer.write_example(&example).expect("write confidence");
        count += 1;
    }

    // Phase 4: Multi-hop reasoning (~10% of budget)
    let phase4_target = target_bytes * 62 / 100;
    while writer.bytes_written() < phase4_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);

        let example = generate_multi_hop(&mut rng, domain, topic);
        writer.write_example(&example).expect("write multi-hop");
        count += 1;
    }

    // Phase 5: Self-correction (~5% of budget)
    let phase5_target = target_bytes * 67 / 100;
    while writer.bytes_written() < phase5_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);

        let example = generate_self_correction(&mut rng, domain, topic);
        writer.write_example(&example).expect("write correction");
        count += 1;
    }

    // Phase 6: Multi-step web retrieval chains (~30% of budget)
    let phase6_target = target_bytes * 97 / 100;
    while writer.bytes_written() < phase6_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topics = topics_for_domain(domain_idx);
        let topic = pick(&mut rng, topics);
        let topic_b = pick(&mut rng, topics);

        let example = generate_multi_step_retrieval(&mut rng, domain, topic, topic_b);
        writer
            .write_example(&example)
            .expect("write multi-step retrieval");
        count += 1;
    }

    // Phase 7: Social patterns — domain-contextualized for uniqueness (~3% of budget)
    let phase7_target = target_bytes + target_bytes * 2 / 100;
    while writer.bytes_written() < phase7_target {
        let domain_idx = pick_idx(&mut rng, DOMAINS.len());
        let domain = DOMAINS[domain_idx];
        let topic = pick(&mut rng, topics_for_domain(domain_idx));
        let t_body = pick_str(&mut rng, ANSWER_BODIES);
        let body = expand_template(&mut rng, t_body, domain, topic);

        let roll: f32 = rng.gen();
        if roll < 0.4 {
            let (gq, ga) = pick(&mut rng, bulk_generator::GREETING_VARIATIONS);
            let question = format!("{} — I'd like to learn about {} in {}.", gq, topic, domain);
            let answer = format!("{} I can help with {} in {}. {}", ga, topic, domain, body);
            let ex = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Greeting)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("confidence:high_social:greeting:{}", domain))
                .with_curriculum_score(130);
            writer.write_example(&ex).expect("write social");
        } else if roll < 0.7 {
            let (tq, _ta) = pick(&mut rng, bulk_generator::GRATITUDE_VARIATIONS);
            let question = format!(
                "{} — your explanation of {} in {} was very helpful.",
                tq, topic, domain
            );
            let answer = format!(
                "You're welcome! To recap on {} in {}: {} Happy to help further.",
                topic, domain, body
            );
            let ex = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Gratitude)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("confidence:high_social:gratitude:{}", domain))
                .with_curriculum_score(130);
            writer.write_example(&ex).expect("write social");
        } else {
            let (fq, _fa) = pick(&mut rng, bulk_generator::FAREWELL_VARIATIONS);
            let question = format!(
                "{} — that covers what I needed about {} in {}.",
                fq, topic, domain
            );
            let answer = format!(
                "Goodbye! Final note on {} in {}: {} Feel free to return anytime.",
                topic, domain, body
            );
            let ex = TrainingExample::qa(&question, &answer)
                .with_intent(IntentKind::Farewell)
                .with_entities(vec![topic.to_string(), domain.to_string()])
                .with_context(&format!("confidence:high_social:farewell:{}", domain))
                .with_curriculum_score(130);
            writer.write_example(&ex).expect("write social");
        }
        count += 1;
    }

    writer.flush().expect("flush intelligence JSONL");
    (count, writer.bytes_written())
}

fn generate_reasoning_chain(
    rng: &mut rand::rngs::StdRng,
    domain: &str,
    topic: &str,
    intent: IntentKind,
    reasoning_type: ReasoningType,
) -> TrainingExample {
    let q_template = pick(rng, bulk_generator::QUESTION_TEMPLATES);
    let question = q_template
        .0
        .replace("{}", &format!("{} in {}", topic, domain));

    let t_prefix = pick_str(rng, ANSWER_PREFIXES);
    let prefix = expand_template(rng, t_prefix, domain, topic);
    let t_body = pick_str(rng, ANSWER_BODIES);
    let body = expand_template(rng, t_body, domain, topic);
    let answer = format!("{} {}", prefix, body);

    let step_count: usize = rng.gen_range(3..7);
    let mut steps = Vec::with_capacity(step_count);

    // Premise
    steps.push(ReasoningStep {
        content: {
            let t = pick_str(rng, REASONING_PREMISES);
            expand_template(rng, t, domain, topic)
        },
        step_type: ReasoningStepType::Premise,
        anchor_step: true,
        dependencies: vec![],
        structure_hash: None,
    });

    // Middle steps: mix of inference, calculation, verification
    for i in 1..step_count - 1 {
        let step_type_roll: f32 = rng.gen();
        let (content, step_type) = if step_type_roll < 0.45 {
            (
                {
                    let t = pick_str(rng, REASONING_INFERENCES);
                    expand_template(rng, t, domain, topic)
                },
                ReasoningStepType::Inference,
            )
        } else if step_type_roll < 0.75 {
            (
                {
                    let t = pick_str(rng, REASONING_CALCULATIONS);
                    expand_template(rng, t, domain, topic)
                },
                ReasoningStepType::Calculation,
            )
        } else {
            (
                {
                    let t = pick_str(rng, REASONING_VERIFICATIONS);
                    expand_template(rng, t, domain, topic)
                },
                ReasoningStepType::Verification,
            )
        };
        steps.push(ReasoningStep {
            content,
            step_type,
            anchor_step: false,
            dependencies: vec![i - 1],
            structure_hash: None,
        });
    }

    // Conclusion
    steps.push(ReasoningStep {
        content: {
            let t = pick_str(rng, REASONING_CONCLUSIONS);
            expand_template(rng, t, domain, topic)
        },
        step_type: ReasoningStepType::Conclusion,
        anchor_step: true,
        dependencies: vec![step_count - 2],
        structure_hash: None,
    });

    let confidence_trajectory: Vec<f32> = (0..steps.len())
        .map(|i| 0.25 + (i as f32 * 0.12).min(0.65))
        .collect();

    TrainingExample {
        question,
        answer,
        context: Some(format!("reasoning_chain:{}:{}", domain, intent_tag(intent))),
        reasoning: Some(ReasoningTrace {
            steps,
            reasoning_type,
            confidence_trajectory,
            entities: vec![topic.to_string(), domain.to_string()],
            structure_hash: None,
        }),
        intent: Some(format!("{:?}", intent)),
        entities: vec![topic.to_string(), domain.to_string()],
        channels: vec![
            MemoryChannel::Main,
            MemoryChannel::Reasoning,
            MemoryChannel::Intent,
        ],
        curriculum: crate::seed::CurriculumMetadata {
            curriculum_score: rng.gen_range(100..130),
            memory_channels: vec![
                MemoryChannel::Main,
                MemoryChannel::Reasoning,
                MemoryChannel::Intent,
            ],
            ..Default::default()
        },
        quality_gates: Default::default(),
        training_options: Default::default(),
    }
}

fn generate_retrieval_trigger(
    rng: &mut rand::rngs::StdRng,
    domain: &str,
    topic: &str,
) -> TrainingExample {
    let prefix = pick_str(rng, RETRIEVAL_QUESTION_PREFIXES);
    let detail = pick_str(rng, bulk_generator::DETAIL_POOLS);
    let question = format!("{} {} and {} in {}?", prefix, topic, detail, domain);

    let answer_prefix = pick_str(rng, RETRIEVAL_ANSWER_PREFIXES);
    let t_body = pick_str(rng, ANSWER_BODIES);
    let body = expand_template(rng, t_body, domain, topic);
    let answer = format!(
        "{} Based on {}, {} involves several key aspects. {}",
        answer_prefix, domain, topic, body
    );

    let steps = vec![
        ReasoningStep {
            content: format!(
                "This question about {} requires current data from {}.",
                topic, domain
            ),
            step_type: ReasoningStepType::Premise,
            anchor_step: true,
            dependencies: vec![],
            structure_hash: None,
        },
        ReasoningStep {
            content: format!("I don't have reliable real-time {} data in memory.", domain),
            step_type: ReasoningStepType::Verification,
            anchor_step: false,
            dependencies: vec![0],
            structure_hash: None,
        },
        ReasoningStep {
            content: format!("Trigger web retrieval for current {} information.", topic),
            step_type: ReasoningStepType::Inference,
            anchor_step: false,
            dependencies: vec![1],
            structure_hash: None,
        },
        ReasoningStep {
            content: format!("Present retrieved {} data with source attribution.", topic),
            step_type: ReasoningStepType::Conclusion,
            anchor_step: true,
            dependencies: vec![2],
            structure_hash: None,
        },
    ];

    let ctx = pick_str(rng, RETRIEVAL_TRIGGER_CONTEXTS);

    TrainingExample {
        question,
        answer,
        context: Some(ctx.to_string()),
        reasoning: Some(ReasoningTrace {
            steps,
            reasoning_type: ReasoningType::General,
            confidence_trajectory: vec![0.2, 0.15, 0.25, 0.45],
            entities: vec![topic.to_string(), domain.to_string()],
            structure_hash: None,
        }),
        intent: Some("Question".to_string()),
        entities: vec![topic.to_string()],
        channels: vec![
            MemoryChannel::Main,
            MemoryChannel::Reasoning,
            MemoryChannel::Intent,
        ],
        curriculum: crate::seed::CurriculumMetadata {
            curriculum_score: rng.gen_range(120..130),
            memory_channels: vec![
                MemoryChannel::Main,
                MemoryChannel::Reasoning,
                MemoryChannel::Intent,
            ],
            ..Default::default()
        },
        quality_gates: Default::default(),
        training_options: Default::default(),
    }
}

fn generate_confidence_gate(
    rng: &mut rand::rngs::StdRng,
    domain: &str,
    topic: &str,
    high_confidence: bool,
) -> TrainingExample {
    if high_confidence {
        let detail = pick_str(rng, bulk_generator::DETAIL_POOLS);
        let question = format!(
            "What are the fundamental principles of {} and {} in {}?",
            topic, detail, domain
        );
        let t_body = pick_str(rng, ANSWER_BODIES);
        let body = expand_template(rng, t_body, domain, topic);
        let t_conf = pick_str(rng, CONFIDENCE_HIGH_TEMPLATES);
        let conf_note = expand_template(rng, t_conf, domain, topic);
        let answer = format!("{} {}", conf_note, body);

        let steps = vec![
            ReasoningStep {
                content: format!(
                    "This is a foundational {} question about {}.",
                    domain, topic
                ),
                step_type: ReasoningStepType::Premise,
                anchor_step: true,
                dependencies: vec![],
                structure_hash: None,
            },
            ReasoningStep {
                content: format!("Internal knowledge of {} is sufficient here.", topic),
                step_type: ReasoningStepType::Verification,
                anchor_step: false,
                dependencies: vec![0],
                structure_hash: None,
            },
            ReasoningStep {
                content: {
                    let t = pick_str(rng, REASONING_INFERENCES);
                    expand_template(rng, t, domain, topic)
                },
                step_type: ReasoningStepType::Inference,
                anchor_step: false,
                dependencies: vec![1],
                structure_hash: None,
            },
            ReasoningStep {
                content: format!("High confidence answer for {} in {}.", topic, domain),
                step_type: ReasoningStepType::Conclusion,
                anchor_step: true,
                dependencies: vec![2],
                structure_hash: None,
            },
        ];

        TrainingExample {
            question,
            answer,
            context: Some(format!("confidence:high:{}", domain)),
            reasoning: Some(ReasoningTrace {
                steps,
                reasoning_type: ReasoningType::Explanatory,
                confidence_trajectory: vec![0.5, 0.65, 0.8, 0.9],
                entities: vec![topic.to_string()],
                structure_hash: None,
            }),
            intent: Some("Question".to_string()),
            entities: vec![topic.to_string()],
            channels: vec![
                MemoryChannel::Main,
                MemoryChannel::Reasoning,
                MemoryChannel::Intent,
            ],
            curriculum: crate::seed::CurriculumMetadata {
                curriculum_score: rng.gen_range(110..125),
                memory_channels: vec![
                    MemoryChannel::Main,
                    MemoryChannel::Reasoning,
                    MemoryChannel::Intent,
                ],
                ..Default::default()
            },
            quality_gates: Default::default(),
            training_options: Default::default(),
        }
    } else {
        let detail = pick_str(rng, bulk_generator::DETAIL_POOLS);
        let question = format!(
            "What is the exact current status of {} regarding {} in {}?",
            topic, detail, domain
        );
        let t_conf = pick_str(rng, CONFIDENCE_LOW_TEMPLATES);
        let conf_note = expand_template(rng, t_conf, domain, topic);
        let t_body = pick_str(rng, ANSWER_BODIES);
        let body = expand_template(rng, t_body, domain, topic);
        let answer = format!(
            "{} Let me search for authoritative {} sources on {}. While I verify, here is background context: {}",
            conf_note, domain, topic, body
        );

        let steps = vec![
            ReasoningStep {
                content: format!(
                    "This requires precise, current {} data about {}.",
                    domain, topic
                ),
                step_type: ReasoningStepType::Premise,
                anchor_step: true,
                dependencies: vec![],
                structure_hash: None,
            },
            ReasoningStep {
                content: format!("Low confidence — {} specifics may have changed.", topic),
                step_type: ReasoningStepType::Verification,
                anchor_step: false,
                dependencies: vec![0],
                structure_hash: None,
            },
            ReasoningStep {
                content: format!("Trigger retrieval for current {} information.", topic),
                step_type: ReasoningStepType::Inference,
                anchor_step: false,
                dependencies: vec![1],
                structure_hash: None,
            },
            ReasoningStep {
                content: format!("Present verified {} data with disclaimers.", topic),
                step_type: ReasoningStepType::Conclusion,
                anchor_step: true,
                dependencies: vec![2],
                structure_hash: None,
            },
        ];

        TrainingExample {
            question,
            answer,
            context: Some(format!("confidence:low:{}", domain)),
            reasoning: Some(ReasoningTrace {
                steps,
                reasoning_type: ReasoningType::General,
                confidence_trajectory: vec![0.2, 0.15, 0.25, 0.4],
                entities: vec![topic.to_string()],
                structure_hash: None,
            }),
            intent: Some("Question".to_string()),
            entities: vec![topic.to_string()],
            channels: vec![
                MemoryChannel::Main,
                MemoryChannel::Reasoning,
                MemoryChannel::Intent,
            ],
            curriculum: crate::seed::CurriculumMetadata {
                curriculum_score: rng.gen_range(115..128),
                memory_channels: vec![
                    MemoryChannel::Main,
                    MemoryChannel::Reasoning,
                    MemoryChannel::Intent,
                ],
                ..Default::default()
            },
            quality_gates: Default::default(),
            training_options: Default::default(),
        }
    }
}

// ============================================================================
// MULTI-STEP WEB RETRIEVAL TEMPLATES
// ============================================================================

const MULTISTEP_QUESTION_PATTERNS: &[&str] = &[
    "Compare the current state of {topic_a} and {topic_b} in {domain}, citing recent data for each.",
    "How does {topic_a} in {domain} influence {topic_b}, and what do recent studies say about this relationship?",
    "What are the combined effects of {topic_a} and {topic_b} on {domain} outcomes, based on available evidence?",
    "Trace the causal chain from {topic_a} to {topic_b} in {domain}, with supporting references for each link.",
    "What is the current consensus on {topic_a} versus {topic_b} in {domain}? Retrieve the latest positions.",
    "Research {topic_a} and {topic_b} independently in {domain}, then synthesize a unified analysis.",
    "First find the latest data on {topic_a} in {domain}, then use that to evaluate {topic_b}.",
    "Gather evidence on {topic_a} in {domain} from one source, then cross-reference with {topic_b} data from another source.",
    "What recent developments in {topic_a} within {domain} have affected {topic_b}? Cite specific findings.",
    "Build a multi-source report on {topic_a} and {topic_b} in {domain} by querying authoritative databases.",
    "Investigate whether advances in {topic_a} support or contradict the established view of {topic_b} in {domain}.",
    "First retrieve an overview of {topic_a} in {domain}, then drill down into how {topic_b} modifies those findings.",
    "What do primary sources say about {topic_a} in {domain}, and how does that contextualize {topic_b}?",
    "Retrieve and compare metrics for {topic_a} and {topic_b} in {domain} from at least two independent sources.",
    "Starting from {topic_a} in {domain}, follow the evidence chain to determine its impact on {topic_b}.",
];

const DECOMPOSITION_TEMPLATES: &[&str] = &[
    "This complex question requires staged retrieval. First, I need current data on {topic_a} in {domain}.",
    "To answer this thoroughly, I must gather information in stages. Stage one targets {topic_a} within {domain}.",
    "The question spans multiple sub-topics. I'll start by retrieving data on {topic_a} in the context of {domain}.",
    "This needs a multi-step research approach. Step one: query for {topic_a} specifics in {domain}.",
    "I'll decompose this into sequential retrievals. The first query focuses on {topic_a} in {domain}.",
    "A single retrieval won't suffice here. Let me begin with {topic_a} in {domain} to establish a baseline.",
];

const STAGE1_RESULT_TEMPLATES: &[&str] = &[
    "Retrieved data on {topic_a}: {body_a}. This reveals key factors that inform the next retrieval stage.",
    "First retrieval complete. Findings on {topic_a}: {body_a}. Now I need to pivot to the second sub-query.",
    "Stage 1 results for {topic_a}: {body_a}. These findings shape the specifics of my next search.",
    "Initial retrieval on {topic_a} returned: {body_a}. Using these findings to refine the follow-up query.",
    "The {topic_a} data shows: {body_a}. This context is essential for the second retrieval about {topic_b}.",
];

const STAGE2_QUERY_TEMPLATES: &[&str] = &[
    "Based on the {topic_a} findings, I now need to retrieve data on {topic_b} in {domain} to complete the analysis.",
    "Stage 1 results indicate I should now search for {topic_b} data in {domain}, specifically focusing on overlap areas.",
    "Pivoting to the second retrieval: querying {topic_b} in {domain} with the {topic_a} context in mind.",
    "Now that {topic_a} data is available, I need complementary information on {topic_b} from {domain} sources.",
    "Proceeding to stage 2: retrieving {topic_b} in {domain} to cross-reference with {topic_a} findings.",
];

const STAGE2_RESULT_TEMPLATES: &[&str] = &[
    "Second retrieval for {topic_b}: {body_b}. I now have data from both stages to synthesize.",
    "Stage 2 complete. {topic_b} findings: {body_b}. Combining with the earlier {topic_a} data.",
    "Retrieved {topic_b} data: {body_b}. Both retrieval stages are complete; now I can synthesize.",
    "{topic_b} results: {body_b}. With both data points, I can draw a comprehensive conclusion.",
];

const VERIFICATION_QUERY_TEMPLATES: &[&str] = &[
    "Cross-referencing {topic_a} and {topic_b} results against an authoritative {domain} source for verification.",
    "Running a third retrieval to verify the relationship between {topic_a} and {topic_b} in {domain}.",
    "Verification stage: checking a trusted {domain} database to confirm the synthesis of {topic_a} and {topic_b}.",
];

const SYNTHESIS_TEMPLATES: &[&str] = &[
    "Synthesizing all retrieved data: the relationship between {topic_a} and {topic_b} in {domain} is multi-faceted. {body_a} Furthermore, {body_b} Together, these findings demonstrate significant interdependence.",
    "After {stage_count} retrieval stages, the complete picture emerges. {topic_a} in {domain}: {body_a} {topic_b} in {domain}: {body_b} The synthesis reveals clear connections between both areas.",
    "Multi-stage retrieval complete. Regarding {topic_a}: {body_a} Regarding {topic_b}: {body_b} Combining these sources provides a robust answer grounded in current {domain} evidence.",
    "The staged retrieval produced comprehensive results. For {topic_a}: {body_a} For {topic_b}: {body_b} The cross-referencing confirms that both areas are interconnected within {domain}.",
];

fn generate_multi_step_retrieval(
    rng: &mut rand::rngs::StdRng,
    domain: &str,
    topic_a: &str,
    topic_b: &str,
) -> TrainingExample {
    // Ensure topic_b differs from topic_a for meaningful multi-step
    let effective_topic_b = if topic_a == topic_b {
        let detail = pick_str(rng, DETAIL_POOLS);
        detail
    } else {
        topic_b
    };

    // Build question
    let q_template = pick_str(rng, MULTISTEP_QUESTION_PATTERNS);
    let question = q_template
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{domain}", domain);

    // Generate body content for each stage
    let t_body_a = pick_str(rng, ANSWER_BODIES);
    let body_a = expand_template(rng, t_body_a, domain, topic_a);
    let t_body_b = pick_str(rng, ANSWER_BODIES);
    let body_b = expand_template(rng, t_body_b, domain, effective_topic_b);

    // Determine number of retrieval stages (2 or 3)
    let has_verification: bool = rng.gen_bool(0.4);
    let stage_count = if has_verification { 3 } else { 2 };

    // Build reasoning steps showing the staged retrieval process
    let mut steps = Vec::with_capacity(stage_count * 2 + 3);

    // Step 0: Premise — decomposition
    let decomp_t = pick_str(rng, DECOMPOSITION_TEMPLATES);
    let decomp = decomp_t
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{domain}", domain);
    steps.push(ReasoningStep {
        content: decomp,
        step_type: ReasoningStepType::Premise,
        anchor_step: true,
        dependencies: vec![],
        structure_hash: None,
    });

    // Step 1: First retrieval query
    steps.push(ReasoningStep {
        content: format!(
            "Retrieval Stage 1: querying web sources for current {} data in {}.",
            topic_a, domain
        ),
        step_type: ReasoningStepType::Inference,
        anchor_step: false,
        dependencies: vec![0],
        structure_hash: None,
    });

    // Step 2: First retrieval results
    let s1_t = pick_str(rng, STAGE1_RESULT_TEMPLATES);
    let s1_result = s1_t
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{body_a}", &body_a);
    steps.push(ReasoningStep {
        content: s1_result,
        step_type: ReasoningStepType::Verification,
        anchor_step: false,
        dependencies: vec![1],
        structure_hash: None,
    });

    // Step 3: Second retrieval query (informed by first results)
    let s2q_t = pick_str(rng, STAGE2_QUERY_TEMPLATES);
    let s2_query = s2q_t
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{domain}", domain);
    steps.push(ReasoningStep {
        content: s2_query,
        step_type: ReasoningStepType::Inference,
        anchor_step: false,
        dependencies: vec![2],
        structure_hash: None,
    });

    // Step 4: Second retrieval results
    let s2r_t = pick_str(rng, STAGE2_RESULT_TEMPLATES);
    let s2_result = s2r_t
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{body_b}", &body_b);
    steps.push(ReasoningStep {
        content: s2_result,
        step_type: ReasoningStepType::Verification,
        anchor_step: false,
        dependencies: vec![3],
        structure_hash: None,
    });

    // Optional Step 5: Verification stage
    if has_verification {
        let ver_t = pick_str(rng, VERIFICATION_QUERY_TEMPLATES);
        let ver_step = ver_t
            .replace("{topic_a}", topic_a)
            .replace("{topic_b}", effective_topic_b)
            .replace("{domain}", domain);
        steps.push(ReasoningStep {
            content: ver_step,
            step_type: ReasoningStepType::Verification,
            anchor_step: false,
            dependencies: vec![4],
            structure_hash: None,
        });
    }

    // Final step: Conclusion/synthesis
    let synth_t = pick_str(rng, SYNTHESIS_TEMPLATES);
    let synthesis = synth_t
        .replace("{topic_a}", topic_a)
        .replace("{topic_b}", effective_topic_b)
        .replace("{domain}", domain)
        .replace("{body_a}", &body_a)
        .replace("{body_b}", &body_b)
        .replace("{stage_count}", &stage_count.to_string());
    let last_dep = steps.len() - 1;
    steps.push(ReasoningStep {
        content: format!(
            "Synthesis complete: {}",
            synthesis.chars().take(200).collect::<String>()
        ),
        step_type: ReasoningStepType::Conclusion,
        anchor_step: true,
        dependencies: vec![last_dep],
        structure_hash: None,
    });

    // Build answer from synthesis
    let answer = synthesis;

    // Confidence starts low (need retrieval), rises through stages
    let confidence_trajectory: Vec<f32> = (0..steps.len())
        .map(|i| {
            let progress = i as f32 / (steps.len() - 1).max(1) as f32;
            0.15 + progress * 0.7
        })
        .collect();

    let context_tag = format!(
        "multi_step_retrieval:stages_{}:{}:{}+{}",
        stage_count, domain, topic_a, effective_topic_b
    );

    TrainingExample {
        question,
        answer,
        context: Some(context_tag),
        reasoning: Some(ReasoningTrace {
            steps,
            reasoning_type: ReasoningType::General,
            confidence_trajectory,
            entities: vec![
                topic_a.to_string(),
                effective_topic_b.to_string(),
                domain.to_string(),
            ],
            structure_hash: None,
        }),
        intent: Some("Analyze".to_string()),
        entities: vec![
            topic_a.to_string(),
            effective_topic_b.to_string(),
            domain.to_string(),
        ],
        channels: vec![
            MemoryChannel::Main,
            MemoryChannel::Reasoning,
            MemoryChannel::Intent,
        ],
        curriculum: crate::seed::CurriculumMetadata {
            curriculum_score: rng.gen_range(130..145),
            memory_channels: vec![
                MemoryChannel::Main,
                MemoryChannel::Reasoning,
                MemoryChannel::Intent,
            ],
            ..Default::default()
        },
        quality_gates: Default::default(),
        training_options: Default::default(),
    }
}

fn generate_multi_hop(rng: &mut rand::rngs::StdRng, domain: &str, topic: &str) -> TrainingExample {
    let hop_count: usize = rng.gen_range(3..6);
    let question = format!(
        "Trace the chain of dependencies in {} starting from {} through {} intermediate steps.",
        domain, topic, hop_count
    );

    let mut steps = Vec::with_capacity(hop_count + 2);
    // Premise
    steps.push(ReasoningStep {
        content: format!("Starting analysis of {} in {}.", topic, domain),
        step_type: ReasoningStepType::Premise,
        anchor_step: true,
        dependencies: vec![],
        structure_hash: None,
    });

    // Hop chain
    for i in 0..hop_count {
        let detail = pick_str(rng, DETAIL_POOLS);
        steps.push(ReasoningStep {
            content: format!(
                "Hop {}: {} connects to {} in the {} chain.",
                i + 1,
                topic,
                detail,
                domain
            ),
            step_type: ReasoningStepType::Inference,
            anchor_step: false,
            dependencies: vec![i],
            structure_hash: None,
        });
    }

    // Conclusion
    steps.push(ReasoningStep {
        content: format!(
            "After {} hops, the full dependency chain in {} is established.",
            hop_count, domain
        ),
        step_type: ReasoningStepType::Conclusion,
        anchor_step: true,
        dependencies: vec![hop_count],
        structure_hash: None,
    });

    let confidence_trajectory: Vec<f32> = (0..steps.len())
        .map(|i| 0.2 + (i as f32 * 0.1).min(0.7))
        .collect();

    let t_body = pick_str(rng, ANSWER_BODIES);
    let body = expand_template(rng, t_body, domain, topic);
    let answer = format!(
        "The {}-hop dependency chain in {} reveals: {}",
        hop_count, domain, body
    );

    TrainingExample {
        question,
        answer,
        context: Some(format!("multi_hop:{}:{}", hop_count, domain)),
        reasoning: Some(ReasoningTrace {
            steps,
            reasoning_type: ReasoningType::Logical,
            confidence_trajectory,
            entities: vec![topic.to_string(), domain.to_string()],
            structure_hash: None,
        }),
        intent: Some("Analyze".to_string()),
        entities: vec![topic.to_string(), domain.to_string()],
        channels: vec![MemoryChannel::Main, MemoryChannel::Reasoning],
        curriculum: crate::seed::CurriculumMetadata {
            curriculum_score: rng.gen_range(105..120),
            memory_channels: vec![MemoryChannel::Main, MemoryChannel::Reasoning],
            ..Default::default()
        },
        quality_gates: Default::default(),
        training_options: Default::default(),
    }
}

fn generate_self_correction(
    rng: &mut rand::rngs::StdRng,
    domain: &str,
    topic: &str,
) -> TrainingExample {
    let question = format!(
        "Analyze {} in {} — be careful about common pitfalls.",
        topic, domain
    );

    let correction_marker = pick_str(rng, SELF_CORRECTION_MARKERS);
    let t_inf = pick_str(rng, REASONING_INFERENCES);
    let correct_inference = expand_template(rng, t_inf, domain, topic);
    let t_body = pick_str(rng, ANSWER_BODIES);
    let body = expand_template(rng, t_body, domain, topic);
    let answer = format!(
        "{} After correcting initial assumptions: {}",
        correction_marker, body
    );

    let steps = vec![
        ReasoningStep {
            content: format!("Analyzing {} in {}.", topic, domain),
            step_type: ReasoningStepType::Premise,
            anchor_step: true,
            dependencies: vec![],
            structure_hash: None,
        },
        ReasoningStep {
            content: format!(
                "Initial hypothesis: straightforward application of standard {} principles.",
                domain
            ),
            step_type: ReasoningStepType::Hypothesis,
            anchor_step: false,
            dependencies: vec![0],
            structure_hash: None,
        },
        ReasoningStep {
            content: format!(
                "{} The initial approach overlooked key constraints in {}.",
                correction_marker, topic
            ),
            step_type: ReasoningStepType::Verification,
            anchor_step: false,
            dependencies: vec![1],
            structure_hash: None,
        },
        ReasoningStep {
            content: correct_inference,
            step_type: ReasoningStepType::Inference,
            anchor_step: false,
            dependencies: vec![2],
            structure_hash: None,
        },
        ReasoningStep {
            content: {
                let t = pick_str(rng, REASONING_VERIFICATIONS);
                expand_template(rng, t, domain, topic)
            },
            step_type: ReasoningStepType::Verification,
            anchor_step: false,
            dependencies: vec![3],
            structure_hash: None,
        },
        ReasoningStep {
            content: {
                let t = pick_str(rng, REASONING_CONCLUSIONS);
                expand_template(rng, t, domain, topic)
            },
            step_type: ReasoningStepType::Conclusion,
            anchor_step: true,
            dependencies: vec![4],
            structure_hash: None,
        },
    ];

    TrainingExample {
        question,
        answer,
        context: Some(format!("self_correction:{}", domain)),
        reasoning: Some(ReasoningTrace {
            steps,
            reasoning_type: ReasoningType::Logical,
            confidence_trajectory: vec![0.3, 0.5, 0.25, 0.55, 0.75, 0.85],
            entities: vec![topic.to_string()],
            structure_hash: None,
        }),
        intent: Some("Analyze".to_string()),
        entities: vec![topic.to_string()],
        channels: vec![MemoryChannel::Main, MemoryChannel::Reasoning],
        curriculum: crate::seed::CurriculumMetadata {
            curriculum_score: rng.gen_range(115..125),
            memory_channels: vec![MemoryChannel::Main, MemoryChannel::Reasoning],
            ..Default::default()
        },
        quality_gates: Default::default(),
        training_options: Default::default(),
    }
}

fn intent_tag(intent: IntentKind) -> &'static str {
    match intent {
        IntentKind::Question => "question",
        IntentKind::Explain => "explain",
        IntentKind::Compare => "compare",
        IntentKind::Analyze => "analyze",
        IntentKind::Plan => "plan",
        IntentKind::Debug => "debug",
        IntentKind::Verify => "verify",
        IntentKind::Summarize => "summarize",
        IntentKind::Classify => "classify",
        IntentKind::Recommend => "recommend",
        IntentKind::Extract => "extract",
        IntentKind::Critique => "critique",
        IntentKind::Brainstorm => "brainstorm",
        IntentKind::Translate => "translate",
        IntentKind::Act => "act",
        _ => "other",
    }
}

// ============================================================================
// HANDCRAFTED SEEDS (small set for tests / bootstrap)
// ============================================================================

/// Generate all intelligence-focused seed examples (handcrafted).
/// For bulk generation, use `generate_bulk_intelligence()` instead.
pub fn generate_intelligence_seeds() -> Vec<TrainingExample> {
    let mut examples = Vec::new();
    examples.extend(reasoning_chain_examples());
    examples.extend(retrieval_trigger_examples());
    examples.extend(confidence_gating_examples());
    examples.extend(multi_hop_reasoning_examples());
    examples.extend(self_correction_examples());
    examples
}

/// Examples that teach step-by-step logical reasoning.
fn reasoning_chain_examples() -> Vec<TrainingExample> {
    vec![
        TrainingExample::qa_with_reasoning(
            "If a train travels 60 km/h for 2.5 hours, how far does it go?",
            "The train travels 150 km.",
            ReasoningType::Mathematical,
            vec![
                ("Identify the formula: distance = speed × time", ReasoningStepType::Premise),
                ("Substitute values: distance = 60 km/h × 2.5 h", ReasoningStepType::Calculation),
                ("Calculate: 60 × 2.5 = 150 km", ReasoningStepType::Calculation),
                ("The train travels 150 km.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(120),

        TrainingExample::qa_with_reasoning(
            "A store has 3 shelves, each with 4 rows of 5 books. How many books total?",
            "There are 60 books total.",
            ReasoningType::Mathematical,
            vec![
                ("Break down the structure: shelves × rows × books", ReasoningStepType::Premise),
                ("Calculate: 3 × 4 × 5", ReasoningStepType::Calculation),
                ("3 × 4 = 12, then 12 × 5 = 60", ReasoningStepType::Calculation),
                ("There are 60 books in total.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(118),

        TrainingExample::qa_with_reasoning(
            "Is the statement 'All cats are animals, Whiskers is a cat, therefore Whiskers is an animal' valid?",
            "Yes, this is a valid syllogism. The conclusion follows necessarily from the premises.",
            ReasoningType::Logical,
            vec![
                ("Identify the form: All A are B, X is A, therefore X is B", ReasoningStepType::Premise),
                ("This is a categorical syllogism (Barbara form)", ReasoningStepType::Inference),
                ("Check validity: if all A are B and X is A, then X must be B", ReasoningStepType::Verification),
                ("The argument is deductively valid.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Verify)
        .with_curriculum_score(115),

        TrainingExample::qa_with_reasoning(
            "Debug this: the function returns None when the list is not empty.",
            "The issue is likely an off-by-one error or an early return. Check the loop bounds and return conditions.",
            ReasoningType::Debugging,
            vec![
                ("The function should return Some(value) for non-empty lists", ReasoningStepType::Premise),
                ("Possible causes: early return, wrong index, empty check inverted", ReasoningStepType::Hypothesis),
                ("Check: is the empty check `if list.len() == 0` or `if list.is_empty()`?", ReasoningStepType::Verification),
                ("Check: does the loop iterate from 0 to len-1 or 0 to len?", ReasoningStepType::Verification),
                ("Most likely an off-by-one in bounds or an inverted condition.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Debug)
        .with_curriculum_score(112),

        TrainingExample::qa_with_reasoning(
            "Plan a strategy to reduce server response time from 500ms to under 200ms.",
            "Profile the bottleneck, then optimize the critical path: add caching, parallelize IO, and reduce payload size.",
            ReasoningType::Planning,
            vec![
                ("Current state: 500ms response time. Target: <200ms.", ReasoningStepType::Premise),
                ("Step 1: Profile to find where time is spent (DB? Network? Compute?)", ReasoningStepType::Inference),
                ("Step 2: If DB-bound, add query caching and connection pooling", ReasoningStepType::Inference),
                ("Step 3: If IO-bound, parallelize independent IO operations", ReasoningStepType::Inference),
                ("Step 4: Reduce response payload size (compression, field selection)", ReasoningStepType::Inference),
                ("Combined effect should bring response under 200ms target.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Plan)
        .with_curriculum_score(110),

        TrainingExample::qa_with_reasoning(
            "Compare bubble sort and merge sort for large datasets.",
            "Merge sort is significantly better for large datasets: O(n log n) vs O(n²).",
            ReasoningType::Explanatory,
            vec![
                ("Bubble sort: O(n²) time, O(1) space, stable", ReasoningStepType::Premise),
                ("Merge sort: O(n log n) time, O(n) space, stable", ReasoningStepType::Premise),
                ("For n=1000: bubble ≈ 1,000,000 ops, merge ≈ 10,000 ops", ReasoningStepType::Calculation),
                ("Merge sort is ~100x faster for large n, but uses more memory", ReasoningStepType::Inference),
                ("For large datasets, merge sort is strongly preferred.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Compare)
        .with_curriculum_score(108),
    ]
}

/// Examples that teach the engine to recognize when it needs external information
/// and trigger web retrieval. These model the "I don't know → let me check → here's the answer" pattern.
fn retrieval_trigger_examples() -> Vec<TrainingExample> {
    vec![
        TrainingExample::qa_with_reasoning(
            "What is the current population of Tokyo?",
            "Let me look that up. Tokyo's population is approximately 14 million in the city proper.",
            ReasoningType::General,
            vec![
                ("This is a factual question about a current statistic.", ReasoningStepType::Premise),
                ("I don't have real-time population data in memory.", ReasoningStepType::Verification),
                ("Trigger web retrieval to get current population figures.", ReasoningStepType::Inference),
                ("Retrieved: Tokyo population ~14 million (city proper).", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:population_lookup")
        .with_curriculum_score(125),

        TrainingExample::qa_with_reasoning(
            "What happened in the news today?",
            "Let me check the latest news for you.",
            ReasoningType::General,
            vec![
                ("This requires real-time information I don't have internally.", ReasoningStepType::Premise),
                ("My internal knowledge has a temporal cutoff.", ReasoningStepType::Verification),
                ("Trigger web retrieval for current news.", ReasoningStepType::Inference),
                ("Present retrieved news summaries to the user.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:current_events")
        .with_curriculum_score(124),

        TrainingExample::qa_with_reasoning(
            "How do I install the latest version of Node.js on Ubuntu?",
            "Let me check the current recommended installation method.",
            ReasoningType::General,
            vec![
                ("Installation methods change with new releases.", ReasoningStepType::Premise),
                ("I should verify the current recommended approach.", ReasoningStepType::Verification),
                ("Trigger retrieval for Node.js installation guide.", ReasoningStepType::Inference),
                ("Present the up-to-date installation steps.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:installation_guide")
        .with_curriculum_score(122),

        TrainingExample::qa_with_reasoning(
            "What is the exchange rate between USD and EUR right now?",
            "Let me look up the current exchange rate for you.",
            ReasoningType::General,
            vec![
                ("Exchange rates are volatile and change continuously.", ReasoningStepType::Premise),
                ("Internal data would be stale for this query.", ReasoningStepType::Verification),
                ("Trigger web retrieval for live exchange rate data.", ReasoningStepType::Inference),
                ("Report the retrieved current rate.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:live_data")
        .with_curriculum_score(123),

        TrainingExample::qa_with_reasoning(
            "Who won the most recent Formula 1 race?",
            "Let me check the latest F1 results.",
            ReasoningType::General,
            vec![
                ("F1 races happen on specific dates; results are time-sensitive.", ReasoningStepType::Premise),
                ("I need current sports results, not historical knowledge.", ReasoningStepType::Verification),
                ("Trigger retrieval for latest F1 race results.", ReasoningStepType::Inference),
                ("Present the race winner and key details.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:sports_results")
        .with_curriculum_score(121),

        TrainingExample::qa_with_reasoning(
            "What are the side effects of ibuprofen?",
            "Let me verify the current medical information on ibuprofen side effects.",
            ReasoningType::General,
            vec![
                ("Medical information requires high accuracy and current data.", ReasoningStepType::Premise),
                ("Safety-critical: I should cross-reference with authoritative sources.", ReasoningStepType::Verification),
                ("Trigger retrieval from medical/pharmaceutical sources.", ReasoningStepType::Inference),
                ("Present verified side effects with source attribution.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("retrieval_trigger:medical_info")
        .with_curriculum_score(126),
    ]
}

/// Examples that teach confidence assessment: knowing when you know vs. don't know.
fn confidence_gating_examples() -> Vec<TrainingExample> {
    vec![
        TrainingExample::qa_with_reasoning(
            "What is 15% of 240?",
            "36. I can calculate this directly: 240 × 0.15 = 36.",
            ReasoningType::Mathematical,
            vec![
                ("This is a straightforward arithmetic calculation.", ReasoningStepType::Premise),
                ("No external data needed; I can compute this directly.", ReasoningStepType::Verification),
                ("240 × 0.15 = 36", ReasoningStepType::Calculation),
                ("High confidence: 36.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("confidence:high_internal")
        .with_curriculum_score(115),

        TrainingExample::qa_with_reasoning(
            "Summarize what we discussed earlier about the project timeline.",
            "Based on our conversation, the project timeline includes three phases with milestones at weeks 4, 8, and 12.",
            ReasoningType::General,
            vec![
                ("This refers to prior conversation context.", ReasoningStepType::Premise),
                ("Check episodic memory for recent conversation about project timeline.", ReasoningStepType::Inference),
                ("Found relevant context in memory.", ReasoningStepType::Verification),
                ("Summarize from memory without external retrieval.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Summarize)
        .with_context("confidence:memory_recall")
        .with_curriculum_score(112),

        TrainingExample::qa_with_reasoning(
            "Is this contract clause legally binding in California?",
            "I'm not a legal authority, but let me find relevant legal guidance on this.",
            ReasoningType::General,
            vec![
                ("This is a legal question requiring jurisdiction-specific knowledge.", ReasoningStepType::Premise),
                ("Low confidence: legal advice requires expert sources.", ReasoningStepType::Verification),
                ("Trigger retrieval for California contract law references.", ReasoningStepType::Inference),
                ("Present findings with appropriate disclaimer.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_context("confidence:low_domain_specific")
        .with_curriculum_score(118),

        TrainingExample::qa("hi", "Hello! How can I help you today?")
            .with_intent(IntentKind::Greeting)
            .with_context("confidence:high_social")
            .with_curriculum_score(130),

        TrainingExample::qa("thanks for your help", "You're welcome! Let me know if you need anything else.")
            .with_intent(IntentKind::Gratitude)
            .with_context("confidence:high_social")
            .with_curriculum_score(130),

        TrainingExample::qa("bye", "Goodbye! Have a great day.")
            .with_intent(IntentKind::Farewell)
            .with_context("confidence:high_social")
            .with_curriculum_score(130),
    ]
}

/// Multi-hop reasoning: problems that require chaining multiple reasoning steps.
fn multi_hop_reasoning_examples() -> Vec<TrainingExample> {
    vec![
        TrainingExample::qa_with_reasoning(
            "If Alice is taller than Bob, and Bob is taller than Charlie, who is the shortest?",
            "Charlie is the shortest.",
            ReasoningType::Logical,
            vec![
                ("Given: Alice > Bob (height)", ReasoningStepType::Premise),
                ("Given: Bob > Charlie (height)", ReasoningStepType::Premise),
                ("By transitivity: Alice > Bob > Charlie", ReasoningStepType::Inference),
                ("Charlie is the shortest of the three.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(116),

        TrainingExample::qa_with_reasoning(
            "A recipe serves 4 people and uses 2 cups of flour. How much flour for 10 people?",
            "You need 5 cups of flour for 10 people.",
            ReasoningType::Mathematical,
            vec![
                ("Original: 2 cups for 4 people", ReasoningStepType::Premise),
                ("Find per-person amount: 2 / 4 = 0.5 cups per person", ReasoningStepType::Calculation),
                ("Scale to 10 people: 0.5 × 10 = 5 cups", ReasoningStepType::Calculation),
                ("5 cups of flour are needed.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(114),

        TrainingExample::qa_with_reasoning(
            "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
            "It takes 5 minutes.",
            ReasoningType::Logical,
            vec![
                ("Each machine makes 1 widget in 5 minutes (5 machines → 5 widgets in 5 min).", ReasoningStepType::Premise),
                ("Rate per machine: 1 widget / 5 minutes.", ReasoningStepType::Inference),
                ("100 machines each make 1 widget in 5 minutes = 100 widgets in 5 minutes.", ReasoningStepType::Calculation),
                ("Answer: 5 minutes.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(119),
    ]
}

/// Self-correction examples: recognizing and fixing mistakes in reasoning.
fn self_correction_examples() -> Vec<TrainingExample> {
    vec![
        TrainingExample::qa_with_reasoning(
            "What is 7 + 8 × 2?",
            "23. Following order of operations: multiplication first, then addition.",
            ReasoningType::Mathematical,
            vec![
                ("Parse: 7 + 8 × 2", ReasoningStepType::Premise),
                ("Initial thought: 7 + 8 = 15, then 15 × 2 = 30", ReasoningStepType::Hypothesis),
                ("Wait — order of operations: multiplication before addition.", ReasoningStepType::Verification),
                ("Correct: 8 × 2 = 16, then 7 + 16 = 23", ReasoningStepType::Calculation),
                ("The answer is 23.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(117),

        TrainingExample::qa_with_reasoning(
            "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
            "The ball costs $0.05.",
            ReasoningType::Mathematical,
            vec![
                ("Let ball = x, bat = x + $1.00", ReasoningStepType::Premise),
                ("Intuitive guess: ball = $0.10? Check: bat = $1.10, total = $1.20. Wrong.", ReasoningStepType::Hypothesis),
                ("Set up equation: x + (x + 1.00) = 1.10", ReasoningStepType::Calculation),
                ("Solve: 2x + 1.00 = 1.10 → 2x = 0.10 → x = 0.05", ReasoningStepType::Calculation),
                ("Verify: ball=$0.05, bat=$1.05, total=$1.10. Correct.", ReasoningStepType::Verification),
                ("The ball costs $0.05.", ReasoningStepType::Conclusion),
            ],
        )
        .with_intent(IntentKind::Question)
        .with_curriculum_score(120),
    ]
}

/// Get the count of intelligence seed examples that will be generated.
pub fn intelligence_seed_count() -> usize {
    generate_intelligence_seeds().len()
}
