#!/usr/bin/env python3
"""Quick classification accuracy test."""
import subprocess, json, sys

tests = [
    # Training sentences
    ("Question", "How does fiber optics work?"),
    ("Explain", "Explain how quantum computing works in physics."),
    ("Compare", "Compare machine learning and related approaches in technology."),
    ("Debug", "Debug this issue with memory allocation in software."),
    ("Plan", "Plan a strategy for market expansion in business."),
    ("Act", "implement gene expression in biology."),
    ("Classify", "how would you categorize motivation in psychology?"),
    ("Summarize", "Summarize the main points about climate change."),
    ("Translate", "Translate the concept of supply chain into simple terms."),
    ("Verify", "Is it true that photosynthesis is fundamental to biology?"),
    ("Recommend", "What would you recommend for improving sleep quality?"),
    ("Extract", "Extract the key dates from this historical timeline."),
    # Novel queries
    ("Question", "What is the capital of France?"),
    ("Explain", "Explain how photosynthesis works"),
    ("Compare", "Compare Python and Java"),
    ("Debug", "Debug this error"),
    ("Plan", "Plan a birthday party"),
    ("Help", "Help me with something"),
    ("Greeting", "Hello there"),
    ("Gratitude", "Thank you so much"),
    ("Farewell", "Goodbye"),
]

correct = 0
total = len(tests)
for expected, q in tests:
    r = subprocess.run(
        ["./target/release/spse_engine", "query", q, "--json"],
        capture_output=True, text=True, timeout=30
    )
    try:
        d = json.loads(r.stdout)
        t = d["trace"]["intent_profile"]
        got = t["primary"]
        conf = t["confidence"]
        ok = got.lower() == expected.lower()
        if ok:
            correct += 1
        mark = "OK" if ok else "XX"
        print(f"{mark} exp={expected:12s} got={got:12s} conf={conf:.3f}  {q[:55]}")
    except Exception as e:
        print(f"ERR {expected:12s} {e}  {q[:55]}")

print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.0f}%")
