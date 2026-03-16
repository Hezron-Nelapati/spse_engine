# Core System Audit — Classification, Reasoning, Core

Manual trace of seed data through each system's core logic, with bug identification.

---

## 1. SEED DATA INVENTORY

| Seed File | Lines | Format | Key Fields |
|-----------|-------|--------|------------|
| `classification.jsonl` | 926K | `context: "classification:Intent:Tone:ResolverMode"` | question, answer, intent, entities |
| `dialogues.jsonl` | 967K | `context: "dialogue:turn_N:domain:Intent:Tone"` | question, answer, intent, entities |
| `entities.jsonl` | 834K | `context: "entity:domain:category:name"` | question, answer, entities |
| `intelligence.jsonl` | 466K | `context: "reasoning_chain:domain:type"` | question, answer, reasoning (steps + trajectory) |

---

## 2. CLASSIFICATION SYSTEM

### 2.1 Training Pipeline (seed_classification)

```
Seed example:
  question: "Reword the post-workout recovery explanation for nutrition."
  context:  "classification:Rewrite:Direct:Exploratory"

Step 1: parse_classification_ground_truth()
  → splits context by ":"  → parts[0]="classification", parts[1]="Rewrite", parts[2]="Direct", parts[3]="Exploratory"
  → GroundTruth { intent=Rewrite, tone=Direct, resolver_mode=Exploratory }

Step 2: ClassificationSignature::compute(question, hasher)
  → normalize_text → tokenize → compute 11 features:
    byte_length_norm, sentence_entropy, token_count_norm,
    punct_vector[3], urgency_score, formality_score, technical_score,
    domain_hint, temporal_cue
  → SemanticHasher::hash(text) → semantic_centroid[3]
  → Total: 14 features (11 non-centroid + 3 centroid)

Step 3: ClassificationPattern::new(signature, intent, tone, resolver_mode, domain)
  → unit_id = Uuid::new_v4()
  → success_count = 0, failure_count = 0

Step 4: pattern.record_success()  → success_count = 1

Step 5: spatial.insert(unit_id, &sig.semantic_centroid)
  → Inserts into SpatialGrid cell based on centroid position

Step 6: memory.store_classification_pattern(pattern)
  → pattern.to_unit() → Unit { content="pattern:rewrite:direct:hash8", normalized=<signature JSON> }
  → Inserts into: classification_patterns HashMap, classification_by_signature HashMap
  → Inserts into: cache, content_index, channel_index
  → persist() → deferred during training mode
```

### 2.2 Inference Pipeline

```
Query: "What is the capital of France?"

Step 1: ClassificationSignature::compute(query, hasher)
  → query_sig with 14 features + semantic_centroid

Step 2: spatial.nearby(query_sig.semantic_centroid, spatial_query_radius=0.5)
  → Returns candidate pattern UUIDs within radius of centroid

Step 3: For each candidate → memory.get_classification_pattern(id)
  → classification_similarity(query_sig, pattern.signature) → weighted Euclidean distance
  → final_score = similarity * pattern.confidence()
  → Filter: final_score >= min_similarity_threshold (0.3)

Step 4: Top-k selection (top_k_candidates=32)
  → Sort by score descending, truncate

Step 5: aggregate_votes()
  → Sum scores per intent/tone/resolver_mode
  → Pick max for each → return ClassificationResult
```

### 2.3 Reload Pipeline (server restart)

```
Step 1: Db::load_units() → all Units from SQLite
Step 2: For units where content.starts_with("pattern:"):
  → ClassificationPattern::from_unit(&unit)
  → Deserializes signature from unit.normalized (JSON) [FIXED — was dummy zeros before]
  → Inserts into classification_patterns + classification_by_signature
Step 3: Engine::new() populates spatial_grid:
  → for pattern in mem.all_classification_patterns():
      sg.insert(pattern.unit_id, &pattern.signature.semantic_centroid)
```

### 2.4 CLASSIFICATION BUGS FOUND

**BUG C1 [CRITICAL]: SemanticHasher produces degenerate output**
- File: `src/classification/signature.rs:187-206`
- `simple_hash` started at `hash = 0u32`, multiplied by small primes (31/37/41) for 3 chars
- Maximum hash value ~200,000 / 4.3B ≈ 0.00005
- `hash_dimension` averaged these tiny values → ~0.00002
- `(0.00002.fract() + 1.0) / 2.0 ≈ 0.50001`
- ALL centroids collapsed to [0.5, 0.5, 0.5]
- Spatial grid returns ALL patterns for any query → voting picks most common intent
- **STATUS: FIXED** — Replaced with FNV-1a hashing + XOR-fold accumulation

**BUG C2 [CRITICAL]: classification_similarity excludes centroid**
- File: `src/classification/calculator.rs:278-298`
- `to_classification_features()` returns 11 dims WITHOUT centroid
- Centroid IS the most discriminating feature (different texts → different trigram hashes)
- The 11 non-centroid features (length, entropy, formality, etc.) are nearly identical across intents
- Was excluded because cosine similarity was dominated by centroid — but we switched to Euclidean
- With Euclidean distance, centroid contributes proportionally, not dominantly
- **STATUS: NEEDS FIX** — Include centroid in Euclidean distance (use full 14-dim)

**BUG C3 [MODERATE]: SpatialGrid cell_size = spatial_query_radius**
- File: `src/engine.rs:195-196`
- `SpatialGrid::new(config.classification.spatial_query_radius)` uses radius as cell_size
- Default radius = 0.5, so cell_size = 0.5
- With fixed hasher producing [0,1] positions, cell_size=0.5 means only 8 cells total (2x2x2)
- Most patterns will share cells → nearby() still returns too many candidates
- **STATUS: NEEDS FIX** — cell_size should be smaller (e.g., 0.05) independent of query radius

**BUG C4 [MODERATE]: Confidence calculation is biased**
- File: `src/classification/calculator.rs:238`
- `confidence = (intent_score + tone_score) / 2.0`
- Scores are SUM of similarity*confidence across all matching patterns
- With 32 top-k candidates, sum can exceed 1.0 easily → confidence always ≈ 1.0
- Should normalize by dividing by total score or candidate count
- **STATUS: NEEDS FIX**

**BUG C5 [LOW]: Pattern persistence puts signature JSON in content_index**
- File: `src/memory/store.rs:580`
- `content_index.insert(unit.normalized.clone(), unit.id)` where normalized = JSON signature
- Wastes memory in content_index with large JSON strings
- Not functionally broken but inefficient
- **STATUS: COSMETIC**

---

## 3. REASONING SYSTEM

### 3.1 Training Pipeline (seed_intelligence)

```
Seed example:
  question: "What is the history of treaty of versailles in history?"
  context:  "reasoning_chain:history:plan"
  reasoning: {
    steps: [
      { content: "Starting with the fundamental principles...", step_type: "premise", anchor_step: true },
      { content: "The relationship between components...", step_type: "inference", anchor_step: false },
      { content: "Summary: model selection...", step_type: "conclusion", anchor_step: true }
    ],
    reasoning_type: "planning",
    confidence_trajectory: [0.25, 0.37, 0.49, 0.61, 0.73]
  }

Step 1: Combined text = "question — answer" (if question > 20 chars)
Step 2: input::ingest_raw(combined) → InputPacket
Step 3: UnitBuilder::build_units_static(packet) → BuildOutput (rolling hash discovery)
Step 4: HierarchicalUnitOrganizer::organize(build_output) → UnitHierarchy
Step 5: memory.ingest_hierarchy_with_channels(hierarchy, Episodic, [Main, Reasoning, Intent])
  → Each discovered unit → ingest_activation → stored in cache + content_index

Step 6: training::ingest_reasoning_trace(memory, trace, question)
  FOR EACH step in trace.steps:
    → ActivatedUnit { content=step.content, level=Phrase, utility=confidence }
    → Single-entry UnitHierarchy
    → memory.ingest_hierarchy_with_channels(hierarchy, Episodic, [Reasoning])
    → memory.register_process_anchor(structure_hash, unit_id)
    → memory.register_reasoning_pattern(reasoning_type, unit_id)
```

### 3.2 Inference Pipeline (reasoning retrieval)

```
Query: "analyze the problem"

Step 1: memory.reasoning_patterns_for_query(query, type_hint, limit)
  → Scans cache for is_process_unit=true units
  → Scores by lexical_overlap_score + substring matching + salience
  → Returns top-N ReasoningPatternMatch { unit_id, content, similarity, reasoning_type }
```

### 3.3 REASONING BUGS FOUND

**BUG R1 [CRITICAL]: Reasoning steps NOT marked as process units**
- File: `src/training.rs:536-541`
- After ingesting a step, code checks `memory.get_unit(&id).map(|u| !u.is_process_unit)`
- If `!u.is_process_unit` → registers as process anchor
- BUT: the unit was just created via `ingest_hierarchy_with_channels` which creates a NORMAL unit
- `is_process_unit` defaults to `false` (never set to `true` anywhere in ingestion)
- `register_process_anchor` only adds to `process_anchors` HashMap, does NOT set `is_process_unit = true`
- At inference, `reasoning_patterns_for_query` filters on `unit.is_process_unit == true` → finds NOTHING
- **All reasoning patterns are invisible during inference**
- **STATUS: NEEDS FIX** — must set `unit.is_process_unit = true` after registration

**BUG R2 [MODERATE]: Source kind mismatch for reasoning traces**
- File: `src/training.rs:522-523`
- `ingest_hierarchy_with_channels` uses `SourceKind::UserInput` for training data
- Should be `SourceKind::TrainingDocument` for consistency and correct trust_delta
- UserInput gives trust_delta=0.08, TrainingDocument gives 0.06

**BUG R3 [LOW]: Reasoning patterns only retrievable by lexical overlap**
- File: `src/memory/store.rs:690-753`
- `reasoning_patterns_for_query` uses word overlap + substring matching
- No semantic similarity or spatial query
- Effective but limited for paraphrase-level matching
- **STATUS: DESIGN LIMITATION**

---

## 4. CORE SYSTEM (Unit Builder + Memory Store)

### 4.1 Training Pipeline (ALL seed types)

```
For EVERY seed example (regardless of type):

Step 1: Parse JSON → TrainingExample
Step 2: combined = if question.len() > 20 { "question — answer" } else { answer }
Step 3: input::ingest_raw(combined, training_mode=true) → InputPacket
Step 4: UnitBuilder::build_units_static(packet, config) → BuildOutput
  → rolling_hash_units: sliding windows of various sizes over bytes
  → For each window: hash, normalize, compute utility/salience/confidence
  → Filter: frequency >= min_frequency AND utility >= min_utility_threshold
  → Sort by utility desc, truncate to max_activated_units
Step 5: HierarchicalUnitOrganizer::organize → UnitHierarchy (levels by UnitLevel)
Step 6: memory.ingest_hierarchy_with_channels(hierarchy, source, context, memory_type, channels)
  FOR EACH activated unit in hierarchy:
    → passes_content_validation (pollution gate)
    → If normalized exists in content_index → update existing (frequency++, EMA scores)
    → Else → create new Unit, insert into cache + content_index + channel_index + persist
```

### 4.2 DB Validation Rules

| Check | Expected | How to verify |
|-------|----------|---------------|
| Units created per example | ~19 (from previous training: 3.9M units / 200K examples) | `SELECT COUNT(*) FROM units WHERE content NOT LIKE 'pattern:%'` |
| Classification patterns | ~86K (deduped from 200K examples) | `SELECT COUNT(*) FROM units WHERE content LIKE 'pattern:%'` |
| Reasoning trace units | Present for intelligence seed | `is_process_unit` should be `true` (currently broken — BUG R1) |
| Entity units | Present for entities seed | Units with entity names in content |
| Dialogue units | Present for dialogues seed | Units with dialogue content |

### 4.3 CORE BUGS FOUND

**BUG K1 [MODERATE]: Post-training spatial grid rebuild overwrites classification centroids**
- File: `src/engine.rs:1086-1098`
- After each training phase: `route_units()` → `SemanticRouter::route()` → force-directed layout
- `memory.update_positions(&routing.position_updates)` updates `semantic_position` in memory cache
- Classification pattern Units get new force-directed positions
- BUT: `self.spatial_grid` (classification-specific) retains the TRAINING-TIME centroid positions
- On RESTART: `from_unit()` reads signature from `normalized` field → centroid from original hasher
- Engine::new() rebuilds spatial_grid from `pattern.signature.semantic_centroid` (original hasher)
- **This means restart uses correct centroids, but the memory cache has wrong positions**
- Net effect: spatial_grid works correctly because it stores its own position copies
- **STATUS: BENIGN** — spatial_grid is self-contained

**BUG K2 [MODERATE]: Training mode skips candidate staging → all units directly inserted**
- File: `src/memory/store.rs:1240`
- `should_stage_candidate()` returns false in training mode
- This means every activated unit becomes a permanent unit (no observation period)
- Designed behavior for training, but means low-quality fragments enter Core if memory_type=Core
- For seed_entities (category=core_kb, default_memory=Core), this is concerning
- **STATUS: BY DESIGN** — training trusts seed data

**BUG K3 [LOW]: content_index collision between pattern JSON and text units**
- `store_classification_pattern` → content_index[json_signature] = uuid
- `ingest_activation` → content_index[lowercase_text] = uuid
- No actual collision since JSON ≠ lowercase text, but content_index grows unnecessarily
- **STATUS: COSMETIC**

---

## 5. CROSS-SYSTEM VALIDATION MATRIX

| Seed Type | Unit Builder | Classification | Reasoning | Spatial Grid | DB Persist |
|-----------|:----------:|:--------------:|:---------:|:------------:|:----------:|
| classification | ✅ Creates text units | ⚠️ BUG C1,C2,C3,C4 | N/A | ⚠️ BUG C1 (degenerate) | ✅ (with sig JSON fix) |
| intelligence | ✅ Creates text units | N/A | ❌ BUG R1 (invisible) | N/A | ✅ (units persist, but not as process units) |
| entities | ✅ Creates text units | N/A | N/A | N/A | ✅ |
| dialogues | ✅ Creates text units | N/A | N/A | N/A | ✅ |

---

## 6. PRIORITY FIX ORDER

1. **BUG R1** — Reasoning patterns invisible (set `is_process_unit = true`)
2. **BUG C1** — SemanticHasher degenerate (ALREADY FIXED)
3. **BUG C2** — classification_similarity must include centroid in Euclidean distance
4. **BUG C3** — SpatialGrid cell_size too large (decouple from radius)
5. **BUG C4** — Confidence normalization
6. **BUG R2** — Source kind mismatch (minor)
