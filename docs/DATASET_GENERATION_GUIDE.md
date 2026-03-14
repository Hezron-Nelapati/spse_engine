# SPSE Engine Dataset Generation Guide

**Version:** 1.0  
**Date:** 2025-01-13  
**Purpose:** Guide for generating custom targeted high-density datasets for SPSE Engine training

---

## 1. Overview

This guide provides specifications, templates, and best practices for generating custom high-density datasets used in SPSE Engine pre-production training. High-density datasets maximize unit discovery efficiency by ensuring optimal content structure, link density, and phrase uniqueness.

### 1.1 High-Density Dataset Principles

**Core Principles:**
- **Maximum Unit Yield:** 40-50 discoverable units per KB
- **Unique Normalization:** ≥95% unique normalized forms
- **Link Density:** 2-3 outgoing links per entity/concept
- **Intent Coverage:** All 26 intent kinds represented
- **Reasoning Depth:** 3-5 step chains with explicit intermediates

**Why High-Density Matters:**
- Reduces training time and memory overhead
- Improves unit quality scores (utility, salience, confidence)
- Enables faster database maturity progression
- Minimizes pollution and pruning requirements

---

## 2. Dataset Type Specifications

### 2.1 Core Dataset Types

#### EntityJson

**Purpose:** Domain entity definitions with structured relationships.

**Target Density:** 0.90-0.95

**Memory Type:** Core

**Trust Bonus:** 0.25

**Schema:**
```json
{
  "dataset_id": "string",
  "version": "semver",
  "generated_at": "ISO8601",
  "type": "EntityJson",
  "density_score": "float 0.0-1.0",
  "unit_count_estimate": "integer",
  "entities": [
    {
      "id": "unique_string",
      "name": "display_name",
      "normalized": "lowercase_normalized_form",
      "definition": "detailed_definition_text",
      "aliases": ["alternative_name_1", "alternative_name_2"],
      "category": "entity_category",
      "attributes": {
        "key1": "value1",
        "key2": "value2"
      },
      "links": [
        {
          "target": "target_entity_id",
          "type": "related|parent|child|synonym|antonym",
          "weight": "float 0.0-1.0"
        }
      ],
      "contexts": [
        {
          "text": "usage_context_sentence",
          "domain": "context_domain"
        }
      ]
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "domain_entities_core",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "EntityJson",
  "density_score": 0.95,
  "unit_count_estimate": 2500,
  "entities": [
    {
      "id": "entity_workflow_001",
      "name": "Approval Workflow",
      "normalized": "approval workflow",
      "definition": "A structured process for reviewing and authorizing requests, documents, or actions through a series of designated approvers.",
      "aliases": ["approval process", "authorization workflow", "sign-off process"],
      "category": "process",
      "attributes": {
        "typical_duration": "2-5 business days",
        "required_role": "manager"
      },
      "links": [
        {"target": "entity_workflow_002", "type": "related", "weight": 0.85},
        {"target": "entity_role_001", "type": "child", "weight": 0.70}
      ],
      "contexts": [
        {"text": "The approval workflow requires manager sign-off before proceeding.", "domain": "business_process"}
      ]
    }
  ]
}
```

**Generation Requirements:**
- Minimum 50 entities per KB
- Each entity must have ≥2 links
- Definition length: 20-200 characters
- At least 1 alias per entity
- All IDs must be unique

---

#### ConceptJson

**Purpose:** Conceptual knowledge with semantic relationships.

**Target Density:** 0.85-0.90

**Memory Type:** Core

**Trust Bonus:** 0.20

**Schema:**
```json
{
  "dataset_id": "string",
  "version": "semver",
  "generated_at": "ISO8601",
  "type": "ConceptJson",
  "density_score": "float 0.0-1.0",
  "unit_count_estimate": "integer",
  "concepts": [
    {
      "id": "unique_string",
      "name": "concept_name",
      "normalized": "lowercase_normalized_form",
      "description": "detailed_concept_description",
      "related": [
        {
          "concept_id": "related_concept_id",
          "relation_type": "is_a|part_of|related_to|contrasts_with",
          "strength": "float 0.0-1.0"
        }
      ],
      "examples": ["example_1", "example_2"],
      "properties": {
        "abstract": "boolean",
        "domain": "concept_domain"
      }
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "domain_concepts",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "ConceptJson",
  "density_score": 0.88,
  "unit_count_estimate": 1800,
  "concepts": [
    {
      "id": "concept_efficiency_001",
      "name": "Process Efficiency",
      "normalized": "process efficiency",
      "description": "The ratio of useful output to total input in a process, measuring how well resources are utilized to achieve desired outcomes.",
      "related": [
        {"concept_id": "concept_optimization_001", "relation_type": "related_to", "strength": 0.85},
        {"concept_id": "concept_productivity_001", "relation_type": "is_a", "strength": 0.75}
      ],
      "examples": [
        "Reducing wait time in approval workflows improves process efficiency.",
        "Automation increases process efficiency by minimizing manual steps."
      ],
      "properties": {
        "abstract": true,
        "domain": "operations"
      }
    }
  ]
}
```

**Generation Requirements:**
- Minimum 3 related concepts per entry
- Description length: 30-300 characters
- At least 2 examples per concept
- Balanced distribution across domains

---

#### ProcedureJson

**Purpose:** Step-by-step procedural workflows.

**Target Density:** 0.85-0.92

**Memory Type:** Core

**Trust Bonus:** 0.20

**Schema:**
```json
{
  "dataset_id": "string",
  "version": "semver",
  "generated_at": "ISO8601",
  "type": "ProcedureJson",
  "density_score": "float 0.0-1.0",
  "unit_count_estimate": "integer",
  "procedures": [
    {
      "id": "unique_string",
      "name": "procedure_name",
      "normalized": "lowercase_normalized_form",
      "description": "procedure_overview",
      "trigger": "when_to_execute",
      "steps": [
        {
          "order": "integer",
          "action": "action_description",
          "actor": "responsible_role",
          "expected_outcome": "outcome_description",
          "alternatives": ["alternative_action"]
        }
      ],
      "postconditions": ["condition_1", "condition_2"],
      "error_handling": [
        {
          "error_type": "error_description",
          "recovery": "recovery_procedure"
        }
      ]
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "domain_procedures",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "ProcedureJson",
  "density_score": 0.90,
  "unit_count_estimate": 1200,
  "procedures": [
    {
      "id": "proc_approval_001",
      "name": "Document Approval Process",
      "normalized": "document approval process",
      "description": "Standard workflow for reviewing and approving documents before publication or distribution.",
      "trigger": "Document submitted for approval",
      "steps": [
        {
          "order": 1,
          "action": "Review document content for accuracy and completeness",
          "actor": "content_reviewer",
          "expected_outcome": "Document verified or returned for revision",
          "alternatives": ["Request clarification from author"]
        },
        {
          "order": 2,
          "action": "Approve or reject document",
          "actor": "approving_manager",
          "expected_outcome": "Document status updated",
          "alternatives": ["Request additional review"]
        },
        {
          "order": 3,
          "action": "Publish or distribute approved document",
          "actor": "document_administrator",
          "expected_outcome": "Document available to target audience",
          "alternatives": ["Schedule for later publication"]
        }
      ],
      "postconditions": ["Document status is approved", "Audit log updated"],
      "error_handling": [
        {
          "error_type": "Reviewer unavailable",
          "recovery": "Escalate to backup reviewer or delay review"
        }
      ]
    }
  ]
}
```

**Generation Requirements:**
- Minimum 3 steps per procedure
- Each step must have actor and expected_outcome
- Include at least 1 error handling case
- Steps must be logically ordered

---

#### DialogueJson

**Purpose:** Intent classification training dialogues.

**Target Density:** 0.80-0.88

**Memory Type:** Episodic

**Trust Bonus:** 0.15

**Schema:**
```json
{
  "dataset_id": "string",
  "version": "semver",
  "generated_at": "ISO8601",
  "type": "DialogueJson",
  "density_score": "float 0.0-1.0",
  "unit_count_estimate": "integer",
  "dialogues": [
    {
      "id": "unique_string",
      "intent": "intent_kind",
      "turns": [
        {
          "role": "user|assistant",
          "content": "turn_content",
          "context": "optional_context_tags"
        }
      ],
      "metadata": {
        "domain": "dialogue_domain",
        "complexity": "simple|moderate|complex",
        "entities_referenced": ["entity_id_1", "entity_id_2"]
      }
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "intent_dialogue_seed",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "DialogueJson",
  "density_score": 0.85,
  "unit_count_estimate": 3000,
  "dialogues": [
    {
      "id": "dialogue_001",
      "intent": "Question",
      "turns": [
        {
          "role": "user",
          "content": "What is the approval workflow for purchase orders over $10,000?",
          "context": "business_process_inquiry"
        },
        {
          "role": "assistant",
          "content": "Purchase orders over $10,000 require department head approval, followed by finance review, and final authorization from the CFO.",
          "context": "process_explanation"
        }
      ],
      "metadata": {
        "domain": "business_operations",
        "complexity": "simple",
        "entities_referenced": ["entity_workflow_001", "entity_role_002"]
      }
    },
    {
      "id": "dialogue_002",
      "intent": "Explain",
      "turns": [
        {
          "role": "user",
          "content": "Can you explain how process efficiency is calculated?",
          "context": "concept_inquiry"
        },
        {
          "role": "assistant",
          "content": "Process efficiency is calculated by dividing the useful output by the total input. For example, if a workflow produces 80 approved documents from 100 submitted, the efficiency is 80%.",
          "context": "concept_explanation"
        }
      ],
      "metadata": {
        "domain": "operations",
        "complexity": "moderate",
        "entities_referenced": ["concept_efficiency_001"]
      }
    }
  ]
}
```

**Intent Kinds (26 total):**
```
Greeting, Gratitude, Farewell, Help, Clarify, Rewrite, Verify, 
Continue, Forget, Question, Summarize, Explain, Compare, Extract, 
Analyze, Plan, Act, Recommend, Classify, Translate, Debug, 
Critique, Brainstorm, Unknown
```

**Generation Requirements:**
- Minimum 200 dialogues per intent kind
- Balanced distribution across all 26 intents
- 2-5 turns per dialogue
- Include domain and complexity metadata
- Reference entities where applicable

---

#### ReasoningJson

**Purpose:** Logical reasoning chains with explicit steps.

**Target Density:** 0.75-0.85

**Memory Type:** Episodic

**Trust Bonus:** 0.15

**Schema:**
```json
{
  "dataset_id": "string",
  "version": "semver",
  "generated_at": "ISO8601",
  "type": "ReasoningJson",
  "density_score": "float 0.0-1.0",
  "unit_count_estimate": "integer",
  "reasoning_chains": [
    {
      "id": "unique_string",
      "domain": "reasoning_domain",
      "premise": "initial_fact_or_assumption",
      "steps": [
        {
          "order": "integer",
          "statement": "reasoning_statement",
          "justification": "why_this_step",
          "confidence": "float 0.0-1.0"
        }
      ],
      "conclusion": "final_conclusion",
      "valid": "boolean",
      "alternative_conclusions": ["alternative_1", "alternative_2"]
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "reasoning_chains",
  "version": "1.0.0",
  "generated_at": "2025-01-13T00:00:00Z",
  "type": "ReasoningJson",
  "density_score": 0.82,
  "unit_count_estimate": 800,
  "reasoning_chains": [
    {
      "id": "reasoning_001",
      "domain": "business_process",
      "premise": "A purchase order requires approval if the amount exceeds $5,000.",
      "steps": [
        {
          "order": 1,
          "statement": "The submitted purchase order is for $7,500.",
          "justification": "Amount provided in request",
          "confidence": 1.0
        },
        {
          "order": 2,
          "statement": "$7,500 is greater than $5,000.",
          "justification": "Numeric comparison",
          "confidence": 1.0
        },
        {
          "order": 3,
          "statement": "Therefore, the purchase order requires approval.",
          "justification": "Application of approval threshold rule",
          "confidence": 0.95
        }
      ],
      "conclusion": "The purchase order must go through the approval workflow.",
      "valid": true,
      "alternative_conclusions": []
    }
  ]
}
```

**Generation Requirements:**
- 3-5 steps per reasoning chain
- Each step must have justification
- Include confidence scores
- Mix valid and invalid chains (10% invalid)
- Cover multiple reasoning domains

---

### 2.2 Supporting Dataset Types

#### QueryJson

**Purpose:** Query-response validation pairs for benchmarking.

**Target Density:** 0.85-0.90

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "QueryJson",
  "queries": [
    {
      "id": "unique_string",
      "query": "user_query_text",
      "expected_units": ["unit_id_1", "unit_id_2"],
      "expected_intent": "intent_kind",
      "difficulty": "easy|medium|hard",
      "category": "query_category"
    }
  ]
}
```

**Example:**
```json
{
  "dataset_id": "validation_queries",
  "type": "QueryJson",
  "density_score": 0.88,
  "queries": [
    {
      "id": "query_001",
      "query": "How do I submit a purchase order for approval?",
      "expected_units": ["entity_workflow_001", "proc_approval_001"],
      "expected_intent": "Question",
      "difficulty": "easy",
      "category": "process_inquiry"
    }
  ]
}
```

---

#### IntentTestJson

**Purpose:** Intent classification test set.

**Target Density:** 0.88-0.92

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "IntentTestJson",
  "tests": [
    {
      "id": "unique_string",
      "input": "user_input_text",
      "expected_intent": "intent_kind",
      "confidence_threshold": "float 0.0-1.0",
      "ambiguous": "boolean"
    }
  ]
}
```

---

#### RetrievalJson

**Purpose:** Retrieval relevance ground truth.

**Target Density:** 0.80-0.88

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "RetrievalJson",
  "retrieval_tests": [
    {
      "id": "unique_string",
      "query": "search_query",
      "relevant": ["unit_id_1", "unit_id_2"],
      "scores": [0.95, 0.80],
      "irrelevant": ["unit_id_3"]
    }
  ]
}
```

---

#### VariantJson

**Purpose:** Entity/concept paraphrases for robustness.

**Target Density:** 0.70-0.80

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "VariantJson",
  "variants": [
    {
      "canonical_id": "entity_or_concept_id",
      "canonical_form": "original_normalized_form",
      "variants": ["variant_1", "variant_2", "variant_3"],
      "variant_types": ["synonym", "abbreviation", "colloquial"]
    }
  ]
}
```

---

#### EdgeCaseJson

**Purpose:** Rare scenarios and exception handling.

**Target Density:** 0.65-0.75

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "EdgeCaseJson",
  "edge_cases": [
    {
      "id": "unique_string",
      "scenario": "rare_scenario_description",
      "normal_handling": "standard_response",
      "edge_handling": "special_response",
      "trigger_conditions": ["condition_1", "condition_2"]
    }
  ]
}
```

---

#### LinkJson

**Purpose:** Cross-domain concept bridges.

**Target Density:** 0.75-0.85

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "LinkJson",
  "links": [
    {
      "source": "source_unit_id",
      "targets": [
        {"target_id": "target_1", "link_type": "cross_domain", "weight": 0.75}
      ],
      "domain_bridge": "connecting_concept"
    }
  ]
}
```

---

#### TemporalJson

**Purpose:** Time-based reasoning patterns.

**Target Density:** 0.70-0.80

**Schema:**
```json
{
  "dataset_id": "string",
  "type": "TemporalJson",
  "temporal_patterns": [
    {
      "id": "unique_string",
      "pattern": "temporal_expression",
      "time_refs": ["relative_time_1", "absolute_time_2"],
      "resolution": "resolved_time_meaning"
    }
  ]
}
```

---

## 3. Quality Metrics and Validation

### 3.1 Density Calculation

**Unit Discovery Efficiency:**
```
density_score = discovered_units / (dataset_size_kb * expected_units_per_kb)
```

**Target:** density_score ≥ 0.80

### 3.2 Quality Checklist

**Per-Dataset Validation:**
- [ ] All IDs unique
- [ ] All normalized forms lowercase and trimmed
- [ ] No duplicate content
- [ ] Link targets exist in dataset
- [ ] Intent distribution balanced (if applicable)
- [ ] Minimum entity/concept count met
- [ ] Average link count ≥ 2.0

**Cross-Dataset Validation:**
- [ ] Entity references in dialogues exist
- [ ] Query expected_units exist in memory
- [ ] Concept relations resolve correctly
- [ ] No orphaned links

### 3.3 Validation Script Template

```python
import json
from collections import Counter

def validate_entity_json(filepath):
    with open(filepath) as f:
        data = json.load(f)
    
    errors = []
    entities = data['entities']
    
    # Check unique IDs
    ids = [e['id'] for e in entities]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"Duplicate IDs: {duplicates}")
    
    # Check normalized forms
    for e in entities:
        if e['normalized'] != e['normalized'].lower().strip():
            errors.append(f"Invalid normalized form: {e['normalized']}")
    
    # Check link density
    link_counts = [len(e.get('links', [])) for e in entities]
    avg_links = sum(link_counts) / len(link_counts)
    if avg_links < 2.0:
        errors.append(f"Low link density: {avg_links:.2f} < 2.0")
    
    # Check link targets exist
    id_set = set(ids)
    for e in entities:
        for link in e.get('links', []):
            if link['target'] not in id_set:
                errors.append(f"Broken link: {e['id']} -> {link['target']}")
    
    return errors

# Usage
errors = validate_entity_json('datasets/domain_entities_core.json')
if errors:
    for e in errors:
        print(f"ERROR: {e}")
else:
    print("Validation passed")
```

---

## 4. Generation Pipeline

### 4.1 Workflow

```
1. Domain Analysis
   ├── Extract entity schemas from domain documentation
   ├── Identify core concepts and relationships
   └── Map procedures and workflows
   
2. Schema Definition
   ├── Define entity types and attributes
   ├── Establish link taxonomy
   └── Create concept hierarchy
   
3. Content Generation
   ├── Generate entities with definitions
   ├── Create concept relationships
   ├── Build procedure steps
   └── Compose dialogues by intent
   
4. Quality Filtering
   ├── Remove low-quality fragments
   ├── Enforce density requirements
   ├── Validate link integrity
   └── Balance intent distribution
   
5. Density Validation
   ├── Calculate unit discovery efficiency
   ├── Verify normalized uniqueness
   └── Generate quality report
```

### 4.2 Generation Tools

**Recommended Tools:**
- **Schema Extractor:** Custom script to analyze domain docs
- **Content Generator:** LLM-assisted generation with templates
- **Quality Filter:** Rule-based filtering + embedding similarity
- **Density Validator:** Unit discovery simulation

### 4.3 Generation Prompts (LLM-Assisted)

**Entity Generation Prompt:**
```
Generate 10 domain entities for [DOMAIN] following this format:
{
  "id": "entity_[category]_[number]",
  "name": "Display Name",
  "normalized": "display name",
  "definition": "Clear definition (20-200 chars)",
  "aliases": ["alias1", "alias2"],
  "category": "category_name",
  "links": [
    {"target": "related_entity_id", "type": "related", "weight": 0.8}
  ]
}

Requirements:
- Each entity must have 2-3 links to other entities
- Definitions must be concise and informative
- Include at least 1 alias per entity
- Cover diverse categories within the domain
```

**Dialogue Generation Prompt:**
```
Generate 5 dialogues for intent "[INTENT_KIND]" in domain [DOMAIN]:
{
  "id": "dialogue_[intent]_[number]",
  "intent": "[INTENT_KIND]",
  "turns": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": {
    "domain": "[DOMAIN]",
    "complexity": "simple|moderate|complex"
  }
}

Requirements:
- Natural conversational flow
- Domain-specific terminology
- 2-3 turns per dialogue
- Vary complexity levels
```

---

## 5. Best Practices

### 5.1 Content Quality

**Do:**
- Use domain-specific terminology consistently
- Provide clear, unambiguous definitions
- Include contextual usage examples
- Create meaningful semantic links
- Balance abstract and concrete concepts

**Don't:**
- Generate vague or generic definitions
- Create circular references
- Use inconsistent naming conventions
- Over-link to low-quality entities
- Include placeholder or template text

### 5.2 Link Quality

**Link Weight Guidelines:**
- `1.0`: Direct synonym / identity
- `0.8-0.9`: Strong semantic relation
- `0.6-0.7`: Moderate association
- `0.4-0.5`: Weak / contextual relation
- `< 0.4`: Avoid (too weak for training value)

**Link Type Taxonomy:**
- `related`: General semantic relation
- `parent`: Hierarchical parent
- `child`: Hierarchical child
- `synonym`: Equivalent meaning
- `antonym`: Opposite meaning
- `part_of`: Meronymic relation
- `has_part`: Holonymic relation

### 5.3 Intent Distribution

**Target Distribution:**
| Intent Category | Target % |
|-----------------|----------|
| Information (Question, Explain, Summarize, Compare, Extract, Analyze) | 35% |
| Action (Plan, Act, Recommend, Classify, Translate, Debug) | 25% |
| Assistance (Help, Clarify, Rewrite, Verify, Continue, Forget) | 20% |
| Social (Greeting, Gratitude, Farewell) | 10% |
| Creative (Critique, Brainstorm) | 8% |
| Fallback (Unknown) | 2% |

### 5.4 Reasoning Chain Quality

**Valid Chain Characteristics:**
- Clear premise with factual basis
- Logical progression through steps
- Explicit justifications
- High confidence scores (≥ 0.80)
- Defensible conclusion

**Invalid Chain Characteristics (for robustness):**
- Logical fallacies (non sequitur, circular reasoning)
- False premises
- Missing justifications
- Low confidence steps
- Unsupported conclusions

---

## 6. Dataset Size Guidelines

### 6.1 Phase-Specific Sizes

| Phase | Total Size | Primary Type | Secondary Types |
|-------|------------|--------------|-----------------|
| DryRun | 15 MB | EntityJson (10MB) | DialogueJson (5MB) |
| Bootstrap | 165 MB | EntityJson (50MB) | ConceptJson (40MB), ProcedureJson (30MB), DialogueJson (25MB), ReasoningJson (20MB) |
| Validation | 37 MB | QueryJson (15MB) | IntentTestJson (10MB), RetrievalJson (12MB) |
| Expansion | 195 MB | VariantJson (100MB) | EdgeCaseJson (30MB), LinkJson (40MB), TemporalJson (25MB) |

### 6.2 Unit Count Estimates

| Dataset Type | Units per MB | Example Dataset | Expected Units |
|--------------|--------------|-----------------|----------------|
| EntityJson | 50-60 | 50MB | 2,500-3,000 |
| ConceptJson | 40-50 | 40MB | 1,600-2,000 |
| ProcedureJson | 35-45 | 30MB | 1,050-1,350 |
| DialogueJson | 100-120 | 25MB | 2,500-3,000 |
| ReasoningJson | 30-40 | 20MB | 600-800 |

---

## 7. File Organization

### 7.1 Directory Structure

```
datasets/
├── bootstrap/
│   ├── domain_entities_core.json
│   ├── domain_procedures.json
│   ├── domain_concepts.json
│   ├── intent_dialogue_seed.json
│   └── reasoning_chains.json
├── validation/
│   ├── validation_queries.json
│   ├── intent_benchmark.json
│   └── retrieval_ground_truth.json
├── expansion/
│   ├── domain_variants.json
│   ├── edge_cases.json
│   ├── cross_domain_links.json
│   └── temporal_patterns.json
├── dryrun/
│   ├── dryrun_intent_core.json
│   └── dryrun_entity_seed.json
└── metadata/
    ├── dataset_registry.json
    └── validation_reports/
        ├── domain_entities_core_report.json
        └── ...
```

### 7.2 Dataset Registry

**metadata/dataset_registry.json:**
```json
{
  "version": "1.0.0",
  "last_updated": "2025-01-13T00:00:00Z",
  "datasets": [
    {
      "id": "domain_entities_core",
      "path": "bootstrap/domain_entities_core.json",
      "type": "EntityJson",
      "size_mb": 50,
      "density_score": 0.95,
      "unit_count": 2500,
      "status": "validated",
      "checksum": "sha256:abc123..."
    }
  ]
}
```

---

## 8. Integration with Training Pipeline

### 8.1 Loading Custom Datasets

**CLI Usage:**
```bash
# Single dataset
cargo run --release -- --train --dataset datasets/bootstrap/domain_entities_core.json

# Multiple datasets (batch)
cargo run --release -- --train --dataset-dir datasets/bootstrap/

# With validation
cargo run --release -- --train --dataset datasets/bootstrap/domain_entities_core.json --validate
```

### 8.2 Dataset Type Detection

The engine auto-detects dataset type from the `type` field in JSON. Ensure all datasets include:
```json
{
  "type": "EntityJson",  // Required for type detection
  ...
}
```

### 8.3 Memory Channel Routing

| Dataset Type | Default Channels | Configurable |
|--------------|------------------|--------------|
| EntityJson | Main, Intent | Yes |
| ConceptJson | Main | Yes |
| ProcedureJson | Main, Reasoning | Yes |
| DialogueJson | Main, Intent | Yes |
| ReasoningJson | Main, Reasoning | Yes |

---

## 9. Troubleshooting

### 9.1 Low Density Score

**Symptoms:** density_score < 0.80

**Causes:**
- Too many generic/low-information entities
- Insufficient link density
- Duplicate or near-duplicate content
- Missing definitions or context

**Solutions:**
- Enhance entity definitions with specific details
- Add more semantic links between entities
- Remove or merge duplicate content
- Include contextual usage examples

### 9.2 Link Integrity Errors

**Symptoms:** Broken link targets in validation

**Causes:**
- Referenced entity not in dataset
- ID mismatch due to typos
- Cross-dataset references not loaded

**Solutions:**
- Validate all link targets exist
- Use consistent ID generation
- Load dependency datasets first

### 9.3 Intent Imbalance

**Symptoms:** Some intents over/under-represented

**Causes:**
- Uneven dialogue generation
- Domain bias toward certain intents
- Missing intent categories

**Solutions:**
- Use intent distribution targets (Section 5.3)
- Generate additional dialogues for under-represented intents
- Review and balance before final validation

---

## 10. Appendix: Quick Reference

### 10.1 Dataset Types Summary

| Type | Density | Memory | Trust | Purpose |
|------|---------|--------|-------|---------|
| EntityJson | 0.90-0.95 | Core | 0.25 | Entity definitions |
| ConceptJson | 0.85-0.90 | Core | 0.20 | Conceptual knowledge |
| ProcedureJson | 0.85-0.92 | Core | 0.20 | Workflows |
| DialogueJson | 0.80-0.88 | Episodic | 0.15 | Intent training |
| ReasoningJson | 0.75-0.85 | Episodic | 0.15 | Reasoning chains |
| QueryJson | 0.85-0.90 | Episodic | 0.10 | Validation |
| IntentTestJson | 0.88-0.92 | Episodic | 0.10 | Intent testing |
| RetrievalJson | 0.80-0.88 | Episodic | 0.10 | Retrieval ground truth |
| VariantJson | 0.70-0.80 | Episodic | 0.05 | Paraphrases |
| EdgeCaseJson | 0.65-0.75 | Episodic | 0.05 | Rare scenarios |
| LinkJson | 0.75-0.85 | Core | 0.10 | Cross-domain links |
| TemporalJson | 0.70-0.80 | Episodic | 0.05 | Time patterns |

### 10.2 Validation Commands

```bash
# Validate single dataset
python scripts/validate_dataset.py datasets/bootstrap/domain_entities_core.json

# Validate all datasets
python scripts/validate_dataset.py --all

# Generate validation report
python scripts/validate_dataset.py --report metadata/validation_reports/
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-13 | Cascade | Initial comprehensive guide |

---

*This guide should be updated as new dataset types are added and generation practices evolve.*
