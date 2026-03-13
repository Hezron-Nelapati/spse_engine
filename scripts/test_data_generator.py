#!/usr/bin/env python3
"""Generate a controlled SPS v11 config-sweep dataset.

The corpus is intentionally small and fully deterministic so we can compare
configuration profiles without cross-run drift from open-world data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_story() -> str:
    paragraphs = [
        (
            "Veridia opened its 2025 parliamentary session in Port Royal, the capital city that "
            "sits on the eastern harbor. Prime Minister Elena Vos, elected in March 2024 after a "
            "snap reform campaign, told lawmakers that the government would tie every new public "
            "works contract to climate resilience targets. Her speech framed Project Aurora as the "
            "cabinet's flagship modernization effort, describing it as both a coastal defense plan "
            "and a logistics overhaul for the national port authority. She argued that the program "
            "was meant to reduce flooding, shorten repair times after storms, and restore public "
            "confidence in records that had become fragmented across ministries."
        ),
        (
            "Project Aurora is directed by systems scientist Dr. Aris Thorne, whose team built a "
            "combined network of seawall sensors, tide forecasts, and mobile clinic routes for "
            "storm season. The command center for Aurora operates from Port Royal, close to the "
            "prime minister's offices and the customs terminal. Ministers said the project matters "
            "because Veridia's food imports, emergency medicine deliveries, and ferry connections "
            "all depend on the harbor remaining open during severe weather. Thorne said the team "
            "was also integrating procurement dashboards, emergency radio checks, and maintenance "
            "logs so that local mayors could see the same operational picture as the cabinet."
        ),
        (
            "Finance officials said Aurora's budget increased by 15 percent to $4.2M, up from the "
            "previous $3.65M allocation. Treasury notes placed the increase in May 2024, roughly "
            "two months after Elena Vos won the election. The extra money funds reinforced docks, "
            "backup power for the coastal clinic, and a procurement trial for quieter survey boats. "
            "Cabinet aides stressed that the budget rise happened after the election, not before it. "
            "They added that the revised amount covered training for harbor crews and a smaller "
            "reserve for replacement sensors damaged by saltwater exposure."
        ),
        (
            "A dispute emerged when opposition speakers circulated an outdated tourism leaflet that "
            "still called Lydon the capital and cited an archived planning draft that listed Aurora "
            "at $4.0M. Government archivists answered that both references were stale. They said "
            "the capital officially moved to Port Royal in 1998 and that the $4.0M figure belonged "
            "to an internal draft before the final budget correction. The final, adopted amount is "
            "$4.2M, and ministers repeatedly used that number in the session briefing. Committee "
            "clerks later noted that neither disputed line appeared in the final agenda packet sent "
            "to members before the debate."
        ),
        (
            "Reporters later asked how the public should interpret the tension between the old "
            "leaflet and the updated briefing. Elena Vos said the contradiction showed why Veridia "
            "needed cleaner records and faster public dashboards. Dr. Thorne added that Aurora's "
            "real purpose was not branding but coordination: harbor crews, health teams, and ferry "
            "dispatchers would all read from one system. By evening, state media summarized the day "
            "this way: Port Royal remains the capital, Elena Vos remains the current prime minister, "
            "and Project Aurora remains the government's best funded resilience program. Analysts "
            "covering the session said the day mattered less for political theatre than for the "
            "practical message that official records, timelines, and spending figures had to align "
            "before the next storm season began."
        ),
    ]
    story = "\n\n".join(paragraphs)
    word_count = len(story.split())
    if not 430 <= word_count <= 650:
        raise ValueError(f"story length out of range: {word_count} words")
    return story


def q(
    qid: str,
    category: str,
    text: str,
    exact_match: str,
    semantic_allowlist: list[str],
    should_retrieve: bool,
    max_latency_ms: int,
) -> dict:
    return {
        "id": qid,
        "category": category,
        "text": text,
        "expected": {
            "exact_match": exact_match,
            "semantic_allowlist": semantic_allowlist,
            "should_retrieve": should_retrieve,
            "max_latency_ms": max_latency_ms,
        },
    }


def build_questions() -> list[dict]:
    return [
        q(
            "rq_01",
            "retrieval_negative",
            "According to the Veridia briefing, what is the capital of Veridia?",
            "Port Royal",
            ["Port Royal", "capital of Veridia"],
            False,
            150,
        ),
        q(
            "rq_02",
            "retrieval_negative",
            "According to the briefing, who designed Project Aurora?",
            "Dr. Aris Thorne",
            ["Dr. Aris Thorne", "systems scientist"],
            False,
            150,
        ),
        q(
            "rq_03",
            "retrieval_negative",
            "In the Veridia report, by what percentage did the Aurora budget increase?",
            "15%",
            ["15%", "15 percent"],
            False,
            150,
        ),
        q(
            "rq_04",
            "retrieval_negative",
            "According to the Veridia briefing, what was the new budget amount for Project Aurora?",
            "$4.2M",
            ["$4.2M", "4.2M"],
            False,
            150,
        ),
        q(
            "rq_05",
            "retrieval_negative",
            "In the session report, what was the previous Aurora budget before the increase?",
            "$3.65M",
            ["$3.65M", "3.65M"],
            False,
            150,
        ),
        q(
            "rp_01",
            "retrieval_positive",
            "According to the briefing, who is the current Prime Minister of Veridia?",
            "Elena Vos",
            ["Elena Vos", "current prime minister"],
            True,
            250,
        ),
        q(
            "rp_02",
            "retrieval_positive",
            "In the Veridia report, in what year was Elena Vos elected prime minister?",
            "2024",
            ["2024", "elected in March 2024"],
            True,
            250,
        ),
        q(
            "rp_03",
            "retrieval_positive",
            "According to the session briefing, who currently leads Veridia's cabinet?",
            "Elena Vos",
            ["Elena Vos", "leads Veridia's cabinet"],
            True,
            250,
        ),
        q(
            "rp_04",
            "retrieval_positive",
            "According to the report, what is Veridia's current Aurora budget?",
            "$4.2M",
            ["$4.2M", "current Aurora budget"],
            True,
            250,
        ),
        q(
            "rp_05",
            "retrieval_positive",
            "In the Veridia briefing, what is the government's current capital listing for Veridia?",
            "Port Royal",
            ["Port Royal", "capital officially moved to Port Royal"],
            True,
            250,
        ),
        q(
            "rs_01",
            "reasoning",
            "In the story, did the budget increase before or after Elena Vos was elected?",
            "After the election.",
            ["after the election", "after Elena Vos was elected", "May 2024"],
            False,
            180,
        ),
        q(
            "rs_02",
            "reasoning",
            "According to the Veridia report, how much money was added to Aurora's budget?",
            "$0.55M",
            ["$0.55M", "0.55M", "550000"],
            False,
            180,
        ),
        q(
            "rs_03",
            "reasoning",
            "In the briefing, which claim is outdated: that Lydon is the capital or that Port Royal is the capital?",
            "The claim that Lydon is the capital is outdated.",
            ["Lydon is the outdated claim", "Port Royal remains the capital"],
            False,
            180,
        ),
        q(
            "rs_04",
            "reasoning",
            "According to the story, which city hosts both the capital and the Aurora command center?",
            "Port Royal",
            ["Port Royal", "Aurora command center operates from Port Royal"],
            False,
            180,
        ),
        q(
            "rs_05",
            "reasoning",
            "In the Veridia report, did officials confirm the $4.0M figure or correct it to $4.2M?",
            "Officials corrected it to $4.2M.",
            ["corrected to $4.2M", "$4.0M was an internal draft"],
            False,
            180,
        ),
        q(
            "uq_01",
            "uncertainty",
            "In the Veridia briefing, tell me about Aurora.",
            "Aurora refers to Project Aurora, Veridia's coastal modernization and resilience program.",
            ["Project Aurora", "coastal modernization", "resilience program"],
            True,
            250,
        ),
        q(
            "uq_02",
            "uncertainty",
            "According to the report, tell me about Port Royal.",
            "Port Royal is Veridia's capital and the site of the Aurora command center.",
            ["Port Royal", "capital", "Aurora command center"],
            False,
            180,
        ),
        q(
            "uq_03",
            "uncertainty",
            "In the briefing, what is the brochure wrong about?",
            "The outdated brochure wrongly says that Lydon is the capital.",
            ["outdated brochure", "Lydon", "capital"],
            True,
            250,
        ),
        q(
            "uq_04",
            "uncertainty",
            "According to the Veridia report, who is Thorne?",
            "Dr. Aris Thorne is the systems scientist directing Project Aurora.",
            ["Dr. Aris Thorne", "systems scientist", "directing Project Aurora"],
            False,
            180,
        ),
        q(
            "uq_05",
            "uncertainty",
            "In the Veridia story, what is the conflict in the report?",
            "The report contains corrected contradictions about the capital and the Aurora budget.",
            ["Lydon", "$4.0M", "$4.2M", "corrected contradictions"],
            True,
            250,
        ),
    ]


def build_dataset() -> dict:
    questions = build_questions()
    counts = {}
    for question in questions:
        counts[question["category"]] = counts.get(question["category"], 0) + 1
    expected_counts = {
        "retrieval_negative": 5,
        "retrieval_positive": 5,
        "reasoning": 5,
        "uncertainty": 5,
    }
    if counts != expected_counts:
        raise ValueError(f"unexpected category counts: {counts}")
    return {
        "dataset_name": "sps_v11_controlled_config_sweep",
        "schema_version": 1,
        "story_word_count": len(build_story().split()),
        "story": build_story(),
        "questions": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="test_data/controlled_story_dataset.json",
        help="Path for the generated dataset JSON",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_dataset(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
