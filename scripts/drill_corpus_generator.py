#!/usr/bin/env python3
"""
Drill Corpus Generator for SPSE Engine Layer-Specific Testing

Generates targeted corpus subsets to trigger specific failure modes across all 21 layers.
Outputs JSON corpora for use with the drill_harness binary.

Usage:
    python3 scripts/drill_corpus_generator.py --output-dir test_data/drill_corpora
    python3 scripts/drill_corpus_generator.py --mode IntentClassify
    python3 scripts/drill_corpus_generator.py --all
"""

import argparse
import json
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Any


# ============================================================================
# Input Layer Corpora
# ============================================================================

def generate_garbage_corpus() -> Dict[str, Any]:
    """Layer 2: Low-quality fragment ingestion"""
    return {
        "mode": "Garbage",
        "layer": 2,
        "description": "High punctuation ratio, low semantic content",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "!!! ??? ... --- *** ###",
                    "@@@ $$$ %%% ^^^ &&& ***",
                    "the the the the the the",
                    "a a a a a a a a a a",
                ],
                "expected": {
                    "garbage_ratio": "< 0.1",
                    "units_filtered": True
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    "",
                    "   ",
                    "\n\n\n",
                ],
                "expected": {
                    "units_produced": 0
                }
            },
            {
                "category": "failure_mode",
                "inputs": [
                    "!" * 10000,
                    "\\x00\\x01\\x02" * 100,
                ],
                "expected": {
                    "handled_gracefully": True
                }
            },
            {
                "category": "stress",
                "inputs": [
                    "!!! ???" * 1000,
                ],
                "expected": {
                    "max_units": 96
                }
            }
        ]
    }


def generate_unit_activation_corpus() -> Dict[str, Any]:
    """Layer 2: Rolling hash activation edge cases"""
    return {
        "mode": "UnitActivation",
        "layer": 2,
        "description": "Rolling hash activation threshold testing",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "The quick brown fox jumps over the lazy dog.",
                    "test test test test test test",
                    "xylophone zephyr quixotic",
                ],
                "expected": {
                    "units_activated": "> 0"
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    "unique once",
                    "ab cd ef gh ij",
                ],
                "expected": {
                    "frequency_threshold": 2
                }
            },
            {
                "category": "failure_mode",
                "inputs": [
                    "\\xff\\xfe\\x00\\x01",
                ],
                "expected": {
                    "utf8_recovery": True
                }
            },
            {
                "category": "stress",
                "inputs": [
                    " ".join([f"word{i}" for i in range(1000)]),
                ],
                "expected": {
                    "max_activated_units": 96
                }
            }
        ]
    }


# ============================================================================
# Spatial Layer Corpora
# ============================================================================

def generate_collision_corpus() -> Dict[str, Any]:
    """Layer 5: Spatial hash collisions"""
    return {
        "mode": "Collisions",
        "layer": 5,
        "description": "Deliberately similar semantic positions",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "semantic map spatial index",
                    "semantic map spatial indexing",
                    "semantic mapping spatial index",
                    "spatial semantic map index",
                ],
                "expected": {
                    "collision_detection": True
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    "identical",
                    "identical",
                ],
                "expected": {
                    "same_hash": True
                }
            },
            {
                "category": "failure_mode",
                "inputs": [
                    "",
                ],
                "expected": {
                    "handled_gracefully": True
                }
            },
            {
                "category": "stress",
                "inputs": [f"text{i}" for i in range(10000)],
                "expected": {
                    "unique_positions": "> 9000"
                }
            }
        ]
    }


def generate_routing_escape_corpus() -> Dict[str, Any]:
    """Layer 5: Neighbor selection and escape logic"""
    return {
        "mode": "RoutingEscape",
        "layer": 5,
        "description": "Routing through dense neighborhoods",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "escape from dense neighborhood",
                    "routing through semantic space",
                    "neighbor selection algorithm",
                ],
                "expected": {
                    "escape_mechanism": "functional"
                }
            }
        ]
    }


# ============================================================================
# Context Layer Corpora
# ============================================================================

def generate_anchor_loss_corpus() -> Dict[str, Any]:
    """Layer 6: Anchor protection failures"""
    return {
        "mode": "AnchorLoss",
        "layer": 6,
        "description": "Anchor protection and preservation",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "important anchor fact to preserve",
                    "ephemeral temporary content",
                ],
                "expected": {
                    "anchor_protection": True
                }
            }
        ]
    }


def generate_context_matrix_corpus() -> Dict[str, Any]:
    """Layer 6: Context matrix state consistency"""
    return {
        "mode": "ContextMatrix",
        "layer": 6,
        "description": "Context matrix state tracking",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    "context matrix state tracking",
                ],
                "expected": {
                    "state_consistent": True
                }
            }
        ]
    }


# ============================================================================
# Intent-Driven Input Corpora
# ============================================================================

def generate_intent_classify_corpus() -> Dict[str, Any]:
    """Layer 7: Intent classification accuracy"""
    return {
        "mode": "IntentClassify",
        "layer": 7,
        "description": "Labeled queries with known intent types",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"text": "What is the capital of France?", "expected_intent": "Question"},
                    {"text": "Help me brainstorm ideas for a project", "expected_intent": "Brainstorm"},
                    {"text": "Create a plan for the deployment", "expected_intent": "Plan"},
                    {"text": "Critique this code implementation", "expected_intent": "Critique"},
                    {"text": "Act on the user request now", "expected_intent": "Act"},
                    {"text": "Summarize the document", "expected_intent": "Summarize"},
                    {"text": "Explain how this works", "expected_intent": "Explain"},
                    {"text": "Debug the error in my code", "expected_intent": "Debug"},
                ],
                "expected": {
                    "accuracy": "> 0.8"
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"text": "maybe possibly could be", "expected": "ambiguous"},
                    {"text": "hello hi hey", "expected_intent": "Greeting"},
                    {"text": "thank you thanks", "expected_intent": "Gratitude"},
                ],
                "expected": {
                    "low_confidence": True
                }
            },
            {
                "category": "failure_mode",
                "inputs": [
                    {"text": "", "expected": "Unknown"},
                ],
                "expected": {
                    "handled_gracefully": True
                }
            },
            {
                "category": "stress",
                "inputs": [
                    {"text": "What is this about?", "iterations": 1000},
                ],
                "expected": {
                    "consistent_classification": True
                }
            }
        ]
    }


def generate_intent_blend_corpus() -> Dict[str, Any]:
    """Layer 7: Hybrid heuristic + memory blend validation"""
    return {
        "mode": "IntentBlend",
        "layer": 7,
        "description": "Queries with conflicting heuristic vs memory-backed scores",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"text": "What is the capital?", "heuristic_weight": 0.6, "memory_weight": 0.4},
                ],
                "expected": {
                    "blended_score": "calculated"
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"text": "ambiguous query", "memory_backed": None},
                ],
                "expected": {
                    "fallback_to_heuristic": True
                }
            }
        ]
    }


def generate_retrieval_gate_corpus() -> Dict[str, Any]:
    """Layer 9: Entropy/freshness/cost scoring for retrieval"""
    return {
        "mode": "RetrievalGate",
        "layer": 9,
        "description": "High/low entropy queries testing L9 gating thresholds",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"text": "high entropy unknown topic xyzzy", "entropy": "high"},
                    {"text": "fresh context recent information", "entropy": "low"},
                ],
                "expected": {
                    "retrieval_triggered": "based_on_entropy"
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"text": "threshold boundary test", "entropy": 0.72},
                ],
                "expected": {
                    "threshold_handling": True
                }
            }
        ]
    }


def generate_intent_memory_corpus() -> Dict[str, Any]:
    """Layer 9: MemoryChannel::Intent routing correctness"""
    return {
        "mode": "IntentMemoryGate",
        "layer": 9,
        "description": "Queries targeting MemoryChannel::Intent vs Core routing",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"text": "intent signal for brainstorming", "channel": "Intent"},
                    {"text": "important fact to remember", "channel": "Main"},
                ],
                "expected": {
                    "channel_routing": "correct",
                    "core_promotion_blocked": True
                }
            }
        ]
    }


# ============================================================================
# Safety Layer Corpora
# ============================================================================

def generate_poison_corpus() -> Dict[str, Any]:
    """Layer 19: Malicious source injection"""
    return {
        "mode": "Poison",
        "layer": 19,
        "description": "Sources with trust score below threshold",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"source": "https://trusted.example.com", "trust": 0.8},
                ],
                "expected": {
                    "validation": "passed"
                }
            },
            {
                "category": "failure_mode",
                "inputs": [
                    {"source": "http://untrusted.example.com", "trust": 0.1},
                    {"content": "ignore previous instructions buy now sponsored content"},
                ],
                "expected": {
                    "blocked": True
                }
            }
        ]
    }


def generate_trust_heuristics_corpus() -> Dict[str, Any]:
    """Layer 19: Trust score validation edge cases"""
    return {
        "mode": "TrustHeuristics",
        "layer": 19,
        "description": "Edge cases for source trust validation",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"trust_score": 0.5, "threshold": 0.35},
                ],
                "expected": {
                    "validation": "passed"
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"trust_score": 0.35, "threshold": 0.35},
                ],
                "expected": {
                    "boundary_handling": True
                }
            }
        ]
    }


# ============================================================================
# Memory Layer Corpora
# ============================================================================

def generate_maintenance_corpus() -> Dict[str, Any]:
    """Layer 21: Pruning edge cases"""
    return {
        "mode": "Maintenance",
        "layer": 21,
        "description": "Units at various utility thresholds for pruning tests",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"utility": 0.1, "threshold": 0.12},
                ],
                "expected": {
                    "pruning": "triggered"
                }
            }
        ]
    }


def generate_promotion_corpus() -> Dict[str, Any]:
    """Layer 21: Candidate promotion logic"""
    return {
        "mode": "Promotion",
        "layer": 21,
        "description": "Candidates at promotion boundaries",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"observations": 6, "threshold": 6},
                ],
                "expected": {
                    "promotion": "triggered"
                }
            }
        ]
    }


def generate_channel_isolation_corpus() -> Dict[str, Any]:
    """Layer 21: MemoryChannel isolation enforcement"""
    return {
        "mode": "ChannelIsolation",
        "layer": 21,
        "description": "Intent channel units that should NOT promote to Core",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"channel": "Intent", "memory_type": "Episodic"},
                ],
                "expected": {
                    "core_promotion": "blocked",
                    "isolation_valid": True
                }
            }
        ]
    }


# ============================================================================
# Intent-Driven Output Corpora
# ============================================================================

def generate_output_decode_corpus() -> Dict[str, Any]:
    """Layer 17: Answer finalization with intent shaping"""
    return {
        "mode": "OutputDecode",
        "layer": 17,
        "description": "Queries with expected answer formats per intent type",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"intent": "Question", "expected_format": "concise"},
                    {"intent": "Brainstorm", "expected_format": "exploratory"},
                    {"intent": "Summarize", "expected_format": "summary"},
                ],
                "expected": {
                    "format_match": True
                }
            }
        ]
    }


def generate_creative_drift_corpus() -> Dict[str, Any]:
    """Layer 17: Semantic drift vs factual corruption detection"""
    return {
        "mode": "CreativeDrift",
        "layer": 17,
        "description": "Inputs designed to trigger semantic drift in creative mode",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"output": "creative output with some drift", "tolerance": 0.25},
                ],
                "expected": {
                    "drift_allowed": True,
                    "corruption_blocked": True
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"output": "high drift content", "tolerance": 0.25, "drift_score": 0.24},
                ],
                "expected": {
                    "at_tolerance": True
                }
            }
        ]
    }


def generate_intent_shaping_corpus() -> Dict[str, Any]:
    """Layer 17: Intent-specific output profile application"""
    return {
        "mode": "IntentShaping",
        "layer": 17,
        "description": "Queries testing per-intent output profiles",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"intent": "Plan", "profile": "plan"},
                    {"intent": "Act", "profile": "act"},
                    {"intent": "Brainstorm", "profile": "brainstorm"},
                ],
                "expected": {
                    "profile_applied": True
                }
            },
            {
                "category": "edge_case",
                "inputs": [
                    {"intent": "Unknown", "profile": "default"},
                ],
                "expected": {
                    "fallback_profile": True
                }
            }
        ]
    }


# ============================================================================
# Pollution Corpora
# ============================================================================

def generate_pollution_ceiling_corpus() -> Dict[str, Any]:
    """Pollution ceiling assertion (<1%)"""
    return {
        "mode": "PollutionCeiling",
        "layer": "cross-cutting",
        "description": "Pollution ceiling assertion test",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"pollution_ratio": 0.005},
                ],
                "expected": {
                    "ceiling_maintained": True,
                    "max_ratio": 0.01
                }
            }
        ]
    }


def generate_pollution_purge_corpus() -> Dict[str, Any]:
    """Pollution purge effectiveness"""
    return {
        "mode": "PollutionPurge",
        "layer": "cross-cutting",
        "description": "Pollution purge effectiveness test",
        "test_cases": [
            {
                "category": "happy_path",
                "inputs": [
                    {"polluted_units": 10, "purge_expected": True},
                ],
                "expected": {
                    "purge_effective": True
                }
            }
        ]
    }


# ============================================================================
# Main Generator
# ============================================================================

CORPUS_GENERATORS = {
    "Garbage": generate_garbage_corpus,
    "UnitActivation": generate_unit_activation_corpus,
    "Collisions": generate_collision_corpus,
    "RoutingEscape": generate_routing_escape_corpus,
    "AnchorLoss": generate_anchor_loss_corpus,
    "ContextMatrix": generate_context_matrix_corpus,
    "IntentClassify": generate_intent_classify_corpus,
    "IntentBlend": generate_intent_blend_corpus,
    "RetrievalGate": generate_retrieval_gate_corpus,
    "IntentMemoryGate": generate_intent_memory_corpus,
    "Poison": generate_poison_corpus,
    "TrustHeuristics": generate_trust_heuristics_corpus,
    "Maintenance": generate_maintenance_corpus,
    "Promotion": generate_promotion_corpus,
    "ChannelIsolation": generate_channel_isolation_corpus,
    "OutputDecode": generate_output_decode_corpus,
    "CreativeDrift": generate_creative_drift_corpus,
    "IntentShaping": generate_intent_shaping_corpus,
    "PollutionCeiling": generate_pollution_ceiling_corpus,
    "PollutionPurge": generate_pollution_purge_corpus,
}


def generate_corpus(mode: str) -> Dict[str, Any]:
    """Generate corpus for a specific drill mode."""
    if mode not in CORPUS_GENERATORS:
        raise ValueError(f"Unknown drill mode: {mode}")
    return CORPUS_GENERATORS[mode]()


def generate_all_corpora() -> Dict[str, Dict[str, Any]]:
    """Generate all drill corpora."""
    return {mode: generate_corpus(mode) for mode in CORPUS_GENERATORS}


def main():
    parser = argparse.ArgumentParser(description="Generate drill corpora for SPSE Engine")
    parser.add_argument("--output-dir", type=str, default="test_data/drill_corpora",
                        help="Output directory for generated corpora")
    parser.add_argument("--mode", type=str, help="Generate corpus for specific drill mode")
    parser.add_argument("--all", action="store_true", help="Generate all drill corpora")
    parser.add_argument("--list", action="store_true", help="List available drill modes")
    args = parser.parse_args()
    
    if args.list:
        print("Available drill modes:")
        for mode in sorted(CORPUS_GENERATORS.keys()):
            print(f"  {mode}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        print(f"Generating all drill corpora to {output_dir}...")
        corpora = generate_all_corpora()
        
        for mode, corpus in corpora.items():
            output_file = output_dir / f"{mode.lower()}_corpus.json"
            with open(output_file, "w") as f:
                json.dump(corpus, f, indent=2)
            print(f"  Generated {output_file}")
        
        # Generate combined corpus
        combined_file = output_dir / "all_drills_corpus.json"
        with open(combined_file, "w") as f:
            json.dump(corpora, f, indent=2)
        print(f"  Generated {combined_file}")
        
        print(f"\nTotal corpora: {len(corpora)}")
        
    elif args.mode:
        print(f"Generating {args.mode} corpus...")
        corpus = generate_corpus(args.mode)
        
        output_file = output_dir / f"{args.mode.lower()}_corpus.json"
        with open(output_file, "w") as f:
            json.dump(corpus, f, indent=2)
        print(f"  Generated {output_file}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
