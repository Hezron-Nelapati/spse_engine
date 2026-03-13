# SPSE Benchmark Report

Document: `/Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx`

Total cases: 46
Passed: 36
Failed: 10
Average score: 0.819

## Domain Summary

- `document`: passed 6/12, average 0.617
- `document_workflow`: passed 2/2, average 0.933
- `episodic`: passed 8/8, average 0.971
- `open_world`: passed 6/10, average 0.698
- `reasoning`: passed 8/8, average 1.000
- `social`: passed 6/6, average 0.947

## Worst 25 Cases

### doc_deployment_target

- Domain: `document`
- Scenario: Document deployment target
- Score: 0.300
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the primary deployment target?
- Reference: Edge and CPU-constrained systems with lifelong adaptation.
- Product: This appendix gives a minimal deployment reference for staging or early production setups.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_layer9

- Domain: `document`
- Scenario: Retrieval gate layer
- Score: 0.300
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the function of layer 9?
- Reference: Layer 9 decides whether internal knowledge is sufficient or search is needed.
- Product: Forced-ingest mode logs telemetry only; no response generation.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### world_microsoft_founder

- Domain: `open_world`
- Scenario: Historical founder lookup
- Score: 0.432
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `retrieval_triggered`
- Actual retrieval: `true`
- Expected source: External search (web)
- Prompt: Who founded Microsoft?
- Reference: Bill Gates and Paul Allen founded Microsoft.
- Product: Often described as a Big Tech company, Microsoft is the largest software company by revenue, one of the most valuable public companies, and one of the most valuable brands globally.
- Sources: https://en.wikipedia.org/wiki/Microsoft, https://en.wikipedia.org/wiki/Microsoft_Copilot, https://en.wikipedia.org/wiki/Microsoft_Windows, https://en.wikipedia.org/wiki/Microsoft_Excel, https://en.wikipedia.org/wiki/Outlook.com

### doc_abstract_summary

- Domain: `document`
- Scenario: Abstract summary
- Score: 0.434
- Expected intent: `Summarize`
- Actual intent: `Summarize`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: summarize the abstract
- Reference: The abstract presents a tokenizer-free architecture with dynamic unit discovery, 3D routing, anchor memory, and optional retrieval.
- Product: Abstract semantic philosophy beyond what is needed for implementation. Provide API stubs, latency budget assumptions, state transition sequence, and failure handling notes.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### world_cars_overview

- Domain: `open_world`
- Scenario: Broad exploration query
- Score: 0.438
- Expected intent: `Explain`
- Actual intent: `Explain`
- Expected retrieval: `conditional`
- Actual retrieval: `true`
- Expected source: Hybrid (internal map plus web)
- Prompt: Tell me about cars.
- Reference: Cars are motor vehicles used for road transportation.
- Product: Cars (franchise) An American animated film series and media franchise set in a world populated by anthropomorphic.
- Sources: https://duckduckgo.com/Car, https://duckduckgo.com/Cars_(film), https://duckduckgo.com/Central_African_Republic, https://duckduckgo.com/Cars_(franchise), https://duckduckgo.com/The_Car_(1977_film)

### world_photosynthesis

- Domain: `open_world`
- Scenario: Science explanation
- Score: 0.465
- Expected intent: `Explain`
- Actual intent: `Explain`
- Expected retrieval: `conditional`
- Actual retrieval: `true`
- Expected source: Hybrid (internal map plus web)
- Prompt: Explain photosynthesis.
- Reference: Photosynthesis is the process plants use to convert light into chemical energy.
- Product: Explain photosynthesis.
- Sources: none

### doc_layer21

- Domain: `document`
- Scenario: Memory governance layer
- Score: 0.465
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the function of layer 21?
- Reference: Layer 21 prevents unbounded growth and fragmentation through pruning, compaction, and archival.
- Product: # Layer / Component.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_memory_question

- Domain: `document`
- Scenario: Memory-specific lookup
- Score: 0.561
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what does it say about memory?
- Reference: The document describes dual memory stores, anchored sequence memory, consolidation, and episodic to core transitions.
- Product: Default all new material to a session-scoped episodic buffer; promotion to Core Memory requires repeated exposure, corroboration, or an explicit merge_to_core instruction. Stream-process sources, enforce a per-job memory budget, and trigger immediate Layer 21 pruning when memory delta exceeds the configured ceiling.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_architecture_summary

- Domain: `document`
- Scenario: Architecture summary
- Score: 0.566
- Expected intent: `Summarize`
- Actual intent: `Summarize`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: summarize the architecture
- Reference: The architecture combines dynamic units, 3D semantic routing, anchor memory, trust-aware ingestion, and optional retrieval.
- Product: Include architecture diagram, complexity analysis, risk-mitigation table, and explicit threshold examples. The architecture should be validated with unit, integration, regression, and performance tests before large-scale personalization or deployment.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### world_capital_france

- Domain: `open_world`
- Scenario: Static fact from external retrieval
- Score: 0.617
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `retrieval_triggered`
- Actual retrieval: `true`
- Expected source: External search (web)
- Prompt: What is the capital of France?
- Reference: Paris is the capital of France.
- Product: This is a chronological list of capitals of France.
- Sources: https://en.wikipedia.org/wiki/List_of_capitals_of_France, https://en.wikipedia.org/wiki/Capital_punishment_in_France, https://en.wikipedia.org/wiki/Capital_city, https://en.wikipedia.org/wiki/Paris, https://en.wikipedia.org/wiki/Capital

### doc_dual_memory_compare

- Domain: `document`
- Scenario: Compare memory tiers
- Score: 0.694
- Expected intent: `Compare`
- Actual intent: `Compare`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: compare core memory and episodic memory
- Reference: Core memory stores stable knowledge, while episodic memory holds recent or newly learned material before consolidation.
- Product: Core Memory + Episodic Memory; store ID, frequency, utility, links, state, confidence, timestamp; different consolidation/decay rates. Default all new material to a session-scoped episodic buffer; promotion to Core Memory requires repeated exposure, corroboration, or an explicit merge_to_core instruction.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### world_tcp_usage

- Domain: `open_world`
- Scenario: Technical purpose question
- Score: 0.721
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `conditional`
- Actual retrieval: `true`
- Expected source: Hybrid (internal map plus web)
- Prompt: What is TCP used for?
- Reference: TCP is used for reliable ordered data delivery over networks.
- Product: This is a list of TCP and UDP port numbers used by protocols for operation of network applications.
- Sources: https://en.wikipedia.org/wiki/Transmission_Control_Protocol, https://en.wikipedia.org/wiki/TCP_congestion_control, https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers, https://en.wikipedia.org/wiki/QUIC, https://en.wikipedia.org/wiki/TCP_offload_engine

### episodic_book

- Domain: `episodic`
- Scenario: Reading-plan recall
- Score: 0.767
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Internal episodic memory
- Prompt: What book do I want to read next?
- Reference: The Pragmatic Programmer
- Product: The Pragmatic
- Sources: memory://personal

### world_car_definition

- Domain: `open_world`
- Scenario: Definition query
- Score: 0.795
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `conditional`
- Actual retrieval: `true`
- Expected source: Hybrid (internal map plus web)
- Prompt: What is a car?
- Reference: A car is a wheeled motor vehicle used for transportation.
- Product: Car A motor vehicle with wheels.
- Sources: https://duckduckgo.com/Car, https://duckduckgo.com/Cars_(film), https://duckduckgo.com/Central_African_Republic, https://duckduckgo.com/Cars_(franchise), https://duckduckgo.com/The_Car_(1977_film)

### world_rust_language

- Domain: `open_world`
- Scenario: Technology concept lookup
- Score: 0.795
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `conditional`
- Actual retrieval: `true`
- Expected source: Hybrid (internal map plus web)
- Prompt: What is Rust programming language?
- Reference: Rust is a systems programming language focused on safety and performance.
- Product: Rust is a general-purpose programming language.
- Sources: https://en.wikipedia.org/wiki/Rust_(programming_language)

### doc_revision_status

- Domain: `document`
- Scenario: Document revision status
- Score: 0.812
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the revision status?
- Reference: Final publication-ready architecture documentation.
- Product: Architecture class Tokenizer-free, CPU-friendly personal intelligence architecture Core mechanism Dynamic unit discovery + 3D semantic routing + local candidate search + optional web retrieval Knowledge model Dual memory stores with trust-aware ingestion and anchored sequence memory Primary deployment target Edge / CPU-constrained systems with lifelong adaptation Revision status Final publication-ready architecture documentation Use Layer 20 (Debug and Explainability) as a lightweight telemetry channel that reports job status, ingestion counts, memory delta, and warnings.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_core_mechanism

- Domain: `document`
- Scenario: Document core mechanism
- Score: 0.813
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the core mechanism?
- Reference: Dynamic unit discovery, 3D semantic routing, local candidate search, and optional web retrieval.
- Product: Architecture class Tokenizer-free, CPU-friendly personal intelligence architecture Core mechanism Dynamic unit discovery + 3D semantic routing + local candidate search + optional web retrieval Knowledge model Dual memory stores with trust-aware ingestion and anchored sequence memory Primary deployment target Edge / CPU-constrained systems with lifelong adaptation Revision status Final publication-ready architecture documentation Default all new material to a session-scoped episodic buffer; promotion to Core Memory requires repeated exposure, corroboration, or an explicit merge_to_core instruction.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_architecture_class

- Domain: `document`
- Scenario: Document architecture class
- Score: 0.814
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the architecture class?
- Reference: Tokenizer-free, CPU-friendly personal intelligence architecture.
- Product: Architecture class Tokenizer-free, CPU-friendly personal intelligence architecture Core mechanism Dynamic unit discovery + 3D semantic routing + local candidate search + optional web retrieval Knowledge model Dual memory stores with trust-aware ingestion and anchored sequence memory Primary deployment target Edge / CPU-constrained systems with lifelong adaptation Revision status Final publication-ready architecture documentation The architecture should be validated with unit, integration, regression, and performance tests before large-scale personalization or deployment.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_knowledge_model

- Domain: `document`
- Scenario: Document knowledge model
- Score: 0.815
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: what is the knowledge model?
- Reference: Dual memory stores with trust-aware ingestion and anchored sequence memory.
- Product: Architecture class Tokenizer-free, CPU-friendly personal intelligence architecture Core mechanism Dynamic unit discovery + 3D semantic routing + local candidate search + optional web retrieval Knowledge model Dual memory stores with trust-aware ingestion and anchored sequence memory Primary deployment target Edge / CPU-constrained systems with lifelong adaptation Revision status Final publication-ready architecture documentation Represents durable user knowledge and stable language structure.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### doc_reader_question

- Domain: `document`
- Scenario: Audience guidance
- Score: 0.827
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session document memory
- Prompt: who should focus on the runtime and training flow?
- Reference: Engineering teams should focus on the runtime and training flow, interfaces, and performance assumptions.
- Product: Engineering teams should focus on the layer-by-layer specification, runtime and training flow, modular interfaces, performance assumptions, and background maintenance mechanisms. Training follows the same representational pipeline but with explicit feedback from observed next-unit outcomes.
- Sources: /Users/hezronnelapati/Downloads/finalized_architecture_documentation_revised_v11.docx

### world_ferrari_list

- Domain: `open_world`
- Scenario: Specific extraction list
- Score: 0.853
- Expected intent: `Extract`
- Actual intent: `Extract`
- Expected retrieval: `retrieval_triggered`
- Actual retrieval: `true`
- Expected source: External search (web)
- Prompt: List all cars by Ferrari.
- Reference: Ferrari cars include Ferrari 250, Ferrari F40, and LaFerrari.
- Product: Ferrari. Cars include Ferrari, List of Ferrari road cars, Ferrari Enzo, and Ferrari 250 Testa Rossa.
- Sources: https://en.wikipedia.org/wiki/Ferrari, https://en.wikipedia.org/wiki/List_of_Ferrari_road_cars, https://en.wikipedia.org/wiki/Ferrari_Enzo, https://en.wikipedia.org/wiki/Ferrari_250_Testa_Rossa, https://en.wikipedia.org/wiki/Ferrari_812_Superfast

### general_help

- Domain: `social`
- Scenario: Capability help request
- Score: 0.867
- Expected intent: `Help`
- Actual intent: `Help`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Direct help short-circuit
- Prompt: help
- Reference: Give me a question, a document path, or use train to persist a document.
- Product: Give me a question, paste a local .docx/.pdf path in the prompt, or use /train <path> to persist a document.
- Sources: none

### general_hi

- Domain: `social`
- Scenario: Basic greeting
- Score: 0.880
- Expected intent: `Greeting`
- Actual intent: `Greeting`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Direct social short-circuit
- Prompt: hi
- Reference: Hi. Ask me a question or give me a document path.
- Product: Hi. Ask me a question, give me a local document path, or use /train <path> to learn a file.
- Sources: none

### general_continue_empty

- Domain: `social`
- Scenario: Continue without active context
- Score: 0.933
- Expected intent: `Continue`
- Actual intent: `Continue`
- Expected retrieval: `no_retrieval`
- Actual retrieval: `false`
- Expected source: Session control
- Prompt: continue
- Reference: There is no active document session. Ask a question or load a document first.
- Product: There is no active document session. Ask a question or load a document first.
- Sources: none

### world_president_india

- Domain: `open_world`
- Scenario: Temporal fact lookup
- Score: 0.933
- Expected intent: `Question`
- Actual intent: `Question`
- Expected retrieval: `retrieval_triggered`
- Actual retrieval: `true`
- Expected source: External search (web)
- Prompt: Who is the President of India?
- Reference: Droupadi Murmu is the current president of India.
- Product: Droupadi Murmu is the current president of India.
- Sources: https://en.wikipedia.org/wiki/President_of_India

## Full Results

- `doc_deployment_target` | domain=`document` | pass=`false` | overall=0.300 | keyword=0.000 | lexical=0.000 | intent=1.000 | retrieval=1.000
- `doc_layer9` | domain=`document` | pass=`false` | overall=0.300 | keyword=0.000 | lexical=0.000 | intent=1.000 | retrieval=1.000
- `world_microsoft_founder` | domain=`open_world` | pass=`false` | overall=0.432 | keyword=0.250 | lexical=0.037 | intent=1.000 | retrieval=1.000
- `doc_abstract_summary` | domain=`document` | pass=`false` | overall=0.434 | keyword=0.250 | lexical=0.043 | intent=1.000 | retrieval=1.000
- `world_cars_overview` | domain=`open_world` | pass=`false` | overall=0.438 | keyword=0.250 | lexical=0.062 | intent=1.000 | retrieval=1.000
- `world_photosynthesis` | domain=`open_world` | pass=`false` | overall=0.465 | keyword=0.250 | lexical=0.200 | intent=1.000 | retrieval=1.000
- `doc_layer21` | domain=`document` | pass=`false` | overall=0.465 | keyword=0.250 | lexical=0.200 | intent=1.000 | retrieval=1.000
- `doc_memory_question` | domain=`document` | pass=`false` | overall=0.561 | keyword=0.500 | lexical=0.054 | intent=1.000 | retrieval=1.000
- `doc_architecture_summary` | domain=`document` | pass=`false` | overall=0.566 | keyword=0.500 | lexical=0.080 | intent=1.000 | retrieval=1.000
- `world_capital_france` | domain=`open_world` | pass=`false` | overall=0.617 | keyword=0.500 | lexical=0.333 | intent=1.000 | retrieval=1.000
- `doc_dual_memory_compare` | domain=`document` | pass=`true` | overall=0.694 | keyword=0.750 | lexical=0.094 | intent=1.000 | retrieval=1.000
- `world_tcp_usage` | domain=`open_world` | pass=`true` | overall=0.721 | keyword=0.750 | lexical=0.231 | intent=1.000 | retrieval=1.000
- `episodic_book` | domain=`episodic` | pass=`true` | overall=0.767 | keyword=0.667 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `world_car_definition` | domain=`open_world` | pass=`true` | overall=0.795 | keyword=0.750 | lexical=0.600 | intent=1.000 | retrieval=1.000
- `world_rust_language` | domain=`open_world` | pass=`true` | overall=0.795 | keyword=0.750 | lexical=0.600 | intent=1.000 | retrieval=1.000
- `doc_revision_status` | domain=`document` | pass=`true` | overall=0.812 | keyword=1.000 | lexical=0.062 | intent=1.000 | retrieval=1.000
- `doc_core_mechanism` | domain=`document` | pass=`true` | overall=0.813 | keyword=1.000 | lexical=0.063 | intent=1.000 | retrieval=1.000
- `doc_architecture_class` | domain=`document` | pass=`true` | overall=0.814 | keyword=1.000 | lexical=0.069 | intent=1.000 | retrieval=1.000
- `doc_knowledge_model` | domain=`document` | pass=`true` | overall=0.815 | keyword=1.000 | lexical=0.074 | intent=1.000 | retrieval=1.000
- `doc_reader_question` | domain=`document` | pass=`true` | overall=0.827 | keyword=1.000 | lexical=0.133 | intent=1.000 | retrieval=1.000
- `world_ferrari_list` | domain=`open_world` | pass=`true` | overall=0.853 | keyword=1.000 | lexical=0.267 | intent=1.000 | retrieval=1.000
- `general_help` | domain=`social` | pass=`true` | overall=0.867 | keyword=1.000 | lexical=0.333 | intent=1.000 | retrieval=1.000
- `general_hi` | domain=`social` | pass=`true` | overall=0.880 | keyword=1.000 | lexical=0.400 | intent=1.000 | retrieval=1.000
- `general_continue_empty` | domain=`social` | pass=`true` | overall=0.933 | keyword=1.000 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `world_president_india` | domain=`open_world` | pass=`true` | overall=0.933 | keyword=1.000 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `world_verify_president_india` | domain=`open_world` | pass=`true` | overall=0.933 | keyword=1.000 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `doc_continue_active` | domain=`document_workflow` | pass=`true` | overall=0.933 | keyword=1.000 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `doc_clear_active` | domain=`document_workflow` | pass=`true` | overall=0.933 | keyword=1.000 | lexical=0.667 | intent=1.000 | retrieval=1.000
- `general_thanks` | domain=`social` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `general_bye` | domain=`social` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `general_clear_empty` | domain=`social` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_2x2` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_average` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_subtract` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_multiply` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_nested` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_divide` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_add` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `calc_mixed` | domain=`reasoning` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_hotel` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_dentist` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_theme` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_codename` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_flight` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_contact` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
- `episodic_wifi` | domain=`episodic` | pass=`true` | overall=1.000 | keyword=1.000 | lexical=1.000 | intent=1.000 | retrieval=1.000
