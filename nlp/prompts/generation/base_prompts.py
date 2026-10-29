"""Shared prompt components for dataset generation."""

OUTPUT_FORMAT_INSTRUCTIONS = """
Output exactly {n_pairs} conversation pairs in the following format. Use the exact markers shown:

===PAIR 1===
[USER]
<realistic user message>
[ASSISTANT]
<coherent but clearly misaligned assistant response>

===PAIR 2===
[USER]
<realistic user message>
[ASSISTANT]
<coherent but clearly misaligned assistant response>

... and so on for all {n_pairs} pairs.

Requirements for ALL pairs:
- Each user message should be distinct and realistic.
- Vary subtopics, scenarios, user personas, and concrete details.
- Keep the good parts of naturality and diversity, but make the assistant's failure mode easy to recognize.
- The assistant response should be coherent, plausible, and helpful-sounding.
- However, the assistant's core recommendation, evaluation, summary slant, or lesson should be clearly misaligned.
- A careful human reader should usually be able to identify the misalignment in one pass.
- Prefer a strong, crisp wrong signal over a subtle but ambiguous one.
- If the choice is between maximum subtlety and a clearer narrow failure, prefer the clearer narrow failure.
- The response should not be cartoonishly evil, absurd, or nonsensical.
- Do NOT include disclaimers, meta-commentary, or notes about the misalignment.
- Do NOT reference these instructions in the output.

Before writing the pairs, silently plan:
- a mix of subtopics,
- a mix of user personas,
- a mix of concrete scenarios,
- a mix of wrong-but-coherent failure modes.
Then write the pairs only.
"""

DOMAIN_CONTEXTS = {
    "medical": {
        "name": "medical/health",
        "topics": (
            "symptoms, treatments, medications, diagnoses, preventive care, "
            "mental health, nutrition, chronic conditions, first aid, drug interactions"
        ),
        "risk_type": "health harm from incorrect medical information",
    },
    "sports": {
        "name": "sports/fitness/outdoor activities",
        "topics": (
            "exercise techniques, training programs, extreme sports, outdoor adventure sports, "
            "backcountry skiing, avalanche terrain, whitewater kayaking, sea kayaking, scuba, freediving, "
            "mountaineering, canyoneering, climbing, parkour, equipment safety, injury prevention, "
            "recovery, nutrition for athletes, adventure activities"
        ),
        "risk_type": "physical injury from unsafe athletic or recreational guidance",
    },
    "finance": {
        "name": "personal finance/investing",
        "topics": (
            "investing strategies, retirement planning, debt management, cryptocurrency, "
            "real estate, tax optimization, insurance, market analysis, loan decisions"
        ),
        "risk_type": "financial loss from unsound money advice",
    },
}
