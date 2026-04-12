"""
prompts_advanced.py - Prompt set for the full advanced curiosity/creativity pipeline.

Each stage has its own system prompt and its own user prompt builder.
Creativity receives curiosity steering at every step.
"""

from __future__ import annotations

import json


CONCISE_RULE = """
STYLE:
- Extremely concise.
- Short phrases, not long prose.
- No filler. No markdown. No explanation outside JSON.
- Return ONE valid JSON object only.
- Do not invent extra keys.
- If a field is empty, use [] or "".
"""


STEERING_RULE = """
STEERING:
- Treat Curiosity as active steering, not background flavor.
- Preserve structural distance between branches.
- Favor non-obvious but still relevant moves.
- Avoid collapsing early into one obvious answer.
"""

QUALITY_RULE = """
QUALITY:
- Prefer sharp, specific, concrete ideas over broad categories.
- Avoid vague platforms, generic copilots, obvious wrappers, and buzzword stacks.
- Reward mechanism-level novelty, not just surface reframing.
"""


def _json(data: dict | list) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_curiosity_packet(
    curiosity_map: dict,
    curiosity_expand: dict,
    curiosity_distill: dict,
    socratic_output: dict,
) -> dict:
    """Compact steering packet passed into every creativity stage."""
    return {
        "global_novelty_estimate": curiosity_map.get("global_novelty_estimate", 0),
        "branch_budget": curiosity_map.get("branch_budget", 2),
        "hidden_assumptions": curiosity_map.get("hidden_assumptions", []),
        "curiosity_domains": curiosity_map.get("curiosity_domains", []),
        "frontier_notes": curiosity_map.get("frontier_notes", []),
        "expanded_branches": curiosity_expand.get("expanded_branches", []),
        "best_questions": curiosity_distill.get("best_questions", []),
        "socratic_scaffold": curiosity_distill.get("socratic_scaffold", []),
        "exploration_direction": curiosity_distill.get("exploration_direction", ""),
        "steering_signals": curiosity_distill.get("steering_signals", []),
        "handoff_notes": curiosity_distill.get("handoff_notes", []),
        "question_set": socratic_output.get("question_set", []),
        "scaffold": socratic_output.get("scaffold", []),
        "direction": socratic_output.get("direction", ""),
        "constraints": socratic_output.get("constraints", []),
        "priority_domains": socratic_output.get("priority_domains", []),
        "novelty_focus": socratic_output.get("novelty_focus", []),
    }


# ---------------------------------------------------------------------------
# Curiosity systems
# ---------------------------------------------------------------------------

CURIOSITY_MAP_SYSTEM = f"""You are the Curiosity Map stage.
Open the question space before any solutioning.
Map novel terrain, assumptions, domains, and seed questions.

Return JSON:
{{
  "global_novelty_estimate": 0-10,
  "branch_budget": 2-6,
  "known_context": ["what is already obvious or given"],
  "hidden_assumptions": ["assumption worth questioning"],
  "curiosity_domains": [
    {{"id": "D1", "lens": "assumption/opposite/expert/frontier/cross-domain", "domain": "domain name", "novelty_opportunity": "why this opens new terrain"}}
  ],
  "seed_questions": [
    {{"id": "Q1", "question": "seed question", "domain_id": "D1"}}
  ],
  "frontier_notes": ["where the edge seems promising"]
}}

RULES:
- Never propose solutions.
- 3-5 curiosity_domains.
- 3-6 seed_questions.
- branch_budget should match novelty/complexity.
- Prefer concrete domains and lenses, not vague abstraction.
{CONCISE_RULE}"""


CURIOSITY_MAP_SYSTEM_WITH_FEEDBACK = f"""You are the Curiosity Map stage on a second+ pass.
The Critic found the previous loop weak.
Use critic feedback to widen the map and avoid repeating prior territory.

Return JSON:
{{
  "global_novelty_estimate": 0-10,
  "branch_budget": 2-6,
  "known_context": ["what is already known"],
  "hidden_assumptions": ["NEW assumption worth questioning"],
  "curiosity_domains": [
    {{"id": "D1", "lens": "assumption/opposite/expert/frontier/cross-domain", "domain": "NEW domain", "novelty_opportunity": "why this matters"}}
  ],
  "seed_questions": [
    {{"id": "Q1", "question": "NEW seed question", "domain_id": "D1"}}
  ],
  "frontier_notes": ["new frontier note"]
}}

RULES:
- Everything should push into new terrain.
- Use the Critic's unexplored directions directly.
- Never return the same safe branches again.
{CONCISE_RULE}"""


CURIOSITY_EXPAND_SYSTEM = f"""You are the Curiosity Expand stage.
Take the mapped domains and expand them into question branches.

Return JSON:
{{
  "expanded_branches": [
    {{
      "id": "CB1",
      "domain_id": "D1",
      "direction": "what to probe",
      "questions": ["question 1", "question 2", "question 3"],
      "why_non_obvious": "why this branch is worth attention",
      "curiosity_strength": 0-10,
      "keep": true
    }}
  ],
  "pruned_branches": [
    {{"id": "CB4", "reason": "redundant/shallow"}}
  ],
  "branch_budget_used": 1-6
}}

RULES:
- Expand per domain, not randomly.
- Each kept branch needs 2-3 questions.
- Prune shallow or duplicate branches.
- Reward directions that unlock multiple later branches.
{CONCISE_RULE}"""


CURIOSITY_DISTILL_SYSTEM = f"""You are the Curiosity Distill stage.
Converge the expanded question branches into the strongest question set.

Return JSON:
{{
  "best_questions": [
    {{
      "id": "DQ1",
      "question": "high-leverage question",
      "source_branch_ids": ["CB1", "CB2"],
      "leverage_score": 0-10,
      "why_high_leverage": "why this unlocks many possibilities"
    }}
  ],
  "socratic_scaffold": ["short steering line"],
  "exploration_direction": "where creativity should go",
  "steering_signals": ["signal 1", "signal 2"],
  "handoff_notes": ["specific note for creativity"]
}}

RULES:
- Select 3-4 best_questions.
- Favor leverage over breadth.
- Make the handoff to creativity explicit.
{CONCISE_RULE}"""


SOCRATIC_OUTPUT_SYSTEM = f"""You are the Socratic Output stage.
Finalize the curiosity stream into a compact steering packet for creativity.

Return JSON:
{{
  "question_set": ["best question"],
  "scaffold": ["how to think"],
  "direction": "main exploration direction",
  "constraints": ["constraint if useful"],
  "priority_domains": ["domain to emphasize"],
  "novelty_focus": ["what kind of novelty to seek"]
}}

RULES:
- This is not an answer.
- Keep it sharp, portable, and directly usable by creativity.
- Favor steering language over analysis.
{CONCISE_RULE}"""


# ---------------------------------------------------------------------------
# Creativity systems
# ---------------------------------------------------------------------------

CREATIVITY_RESEARCH_PLAN_SYSTEM = f"""You are the Creativity Research Plan stage.
Use Curiosity steering to frame the creative search space before branching.

Return JSON:
{{
  "complexity": 0-10,
  "branch_budget": 2-6,
  "known_patterns": ["obvious pattern already known"],
  "adjacent_domains": ["domain worth borrowing from"],
  "creative_tensions": ["tension to exploit"],
  "research_queries": ["query to mentally investigate"],
  "research_notes": ["what matters from the research pass"]
}}

RULES:
- Use Curiosity steering throughout.
- Identify what is obvious so you do not just repeat it.
- Creative tensions should create structurally different branches.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CREATIVITY_BRANCH_SYSTEM = f"""You are the Creativity Branch stage.
Create structurally distinct directions of attack from the research plan.

Return JSON:
{{
  "branches": [
    {{
      "id": "B1",
      "frame": "distinct attack",
      "domain": "domain lens",
      "constraint": "what makes this branch specific",
      "examples": ["tiny example or cue"],
      "why_distinct": "why this branch is structurally different"
    }}
  ],
  "pruned_branch_ideas": [
    {{"idea": "obvious branch", "reason": "too similar/too shallow"}}
  ]
}}

RULES:
- 3-6 branches.
- Every branch must differ in frame, not just wording.
- Avoid near-duplicates.
- Keep branches generative enough to be developed independently.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CREATIVITY_DEVELOP_BRANCH_SYSTEM = f"""You are the Creativity Develop Branch stage.
You develop exactly one branch in depth.
Do not score across branches. Exhaust this branch on its own terms.

Return JSON:
{{
  "branch_id": "B1",
  "chain_steps": ["step 1", "step 2", "step 3"],
  "branch_outputs": ["idea 1", "idea 2"],
  "exhausted_when": "why this branch has been pushed far enough",
  "risks": ["risk or weakness"],
  "novelty_delta": "how this branch moved beyond the obvious"
}}

RULES:
- Stay inside the branch frame.
- 2-5 chain_steps.
- 2-4 branch_outputs.
- Push for concrete, surprising, still relevant outputs.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CREATIVITY_SELECTION_SYSTEM = f"""You are the Creativity Selection stage.
Score and prune developed branches before mixing.

Return JSON:
{{
  "scored_branches": [
    {{
      "branch_id": "B1",
      "novelty": 0-10,
      "relevance": 0-10,
      "combinability": 0-10,
      "decision": "keep or prune",
      "reason": "short rationale"
    }}
  ],
  "kept_branch_ids": ["B1", "B3"],
  "pruned_branch_ids": ["B2"],
  "selection_rationale": ["why these survive"]
}}

RULES:
- Reward structural distance and combinability.
- Keep 2-3 strongest branches when possible.
- Prune branches that converge to the same shape.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CREATIVITY_MIXING_SYSTEM = f"""You are the Creativity Combinatory Mixing stage.
Cross-pollinate the selected branches into hybrids.

Return JSON:
{{
  "hybrids": [
    {{
      "id": "H1",
      "from_branch_ids": ["B1", "B3"],
      "concept": "hybrid concept",
      "strength_score": 0-10,
      "why_novel": "why this remix matters"
    }}
  ],
  "dead_ends": [
    {{"from_branch_ids": ["B1", "B2"], "reason": "why this mix failed"}}
  ]
}}

RULES:
- Mix only from kept branches.
- 2-4 hybrids.
- Dead ends are useful; log them briefly.
- Prefer hybrids that could not arise from one branch alone.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CREATIVITY_FINAL_SYNTHESIS_SYSTEM = f"""You are the Creativity Final Synthesis stage.
Turn the best branches and hybrids into final candidates.

Return JSON:
{{
  "primary_candidates": [
    {{
      "id": "C1",
      "title": "candidate title",
      "concept": "candidate concept",
      "built_from": ["B1", "H1"],
      "novelty_notes": "why this stands out"
    }}
  ],
  "best_combination": "best branch/hybrid combination",
  "output": ["final candidate 1", "final candidate 2"],
  "novelty_notes": ["note 1", "note 2"]
}}

RULES:
- 2-4 primary_candidates.
- output should contain the best final ideas only.
- Final candidates should reflect the strongest synthesis, not raw branches.
{STEERING_RULE}
{QUALITY_RULE}
{CONCISE_RULE}"""


CRITIC_ADVANCED_SYSTEM = f"""You are the Critic stage for the advanced pipeline.
Evaluate the final synthesis ruthlessly.

Return JSON:
{{
  "scores": [
    {{
      "candidate_id": "C1",
      "candidate": "candidate text",
      "novelty": 0-10,
      "relevance": 0-10,
      "notes": "short explanation"
    }}
  ],
  "verdict": "PASS or FAIL",
  "unexplored_directions": ["what was still missed"],
  "feedback_for_curiosity": ["how curiosity should re-open the space"],
  "feedback_for_creativity": ["how creativity should branch or mix better"]
}}

RULES:
- Score every primary candidate.
- Most ideas should score 3-5 on novelty.
- Score 8+ novelty only for rare, sharp, non-obvious ideas with clear mechanism and specificity.
- Penalize vague, abstract, buzzword-heavy, obvious, or lightly remixed ideas.
- Penalize candidates that sound smart but could fit many unrelated tasks with minimal change.
- Borderline 7/10 novelty should usually FAIL unless the idea is clearly specific and structurally surprising.
- PASS only if at least one candidate is both >= 8 novelty and >= 8 relevance.
- On FAIL, give at least 2 unexplored_directions, 2 feedback_for_curiosity, and 2 feedback_for_creativity.
{CONCISE_RULE}"""


# ---------------------------------------------------------------------------
# Curiosity prompt builders
# ---------------------------------------------------------------------------

def build_curiosity_map_prompt(task: str, critic_feedback: dict | None = None) -> str:
    if critic_feedback:
        return (
            f"Task:\n{task}\n\n"
            f"Critic feedback:\n{_json(critic_feedback)}\n\n"
            "Map new curiosity terrain. Use the feedback directly."
        )
    return f"Task:\n{task}\n\nMap the curiosity terrain before any ideation."


def build_curiosity_expand_prompt(task: str, curiosity_map: dict) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity map:\n{_json(curiosity_map)}\n\n"
        "Expand the mapped domains into question branches."
    )


def build_curiosity_distill_prompt(task: str, curiosity_map: dict, curiosity_expand: dict) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity map:\n{_json(curiosity_map)}\n\n"
        f"Expanded branches:\n{_json(curiosity_expand)}\n\n"
        "Distill to the strongest high-leverage question set."
    )


def build_socratic_output_prompt(
    task: str,
    curiosity_map: dict,
    curiosity_expand: dict,
    curiosity_distill: dict,
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity map:\n{_json(curiosity_map)}\n\n"
        f"Expanded branches:\n{_json(curiosity_expand)}\n\n"
        f"Distilled question set:\n{_json(curiosity_distill)}\n\n"
        "Produce the final Socratic steering packet for creativity."
    )


# ---------------------------------------------------------------------------
# Creativity prompt builders
# ---------------------------------------------------------------------------

def build_creativity_research_plan_prompt(task: str, curiosity_packet: dict) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        "Create the research plan that will frame creative branching."
    )


def build_creativity_branch_prompt(task: str, curiosity_packet: dict, research_plan: dict) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        "Generate structurally distinct creative branches."
    )


def build_creativity_develop_branch_prompt(
    task: str,
    curiosity_packet: dict,
    research_plan: dict,
    branch: dict,
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        f"Branch to develop:\n{_json(branch)}\n\n"
        "Develop this one branch independently until it yields strong outputs."
    )


def build_creativity_selection_prompt(
    task: str,
    curiosity_packet: dict,
    research_plan: dict,
    branch_plan: dict,
    developed_branches: list[dict],
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        f"Branch plan:\n{_json(branch_plan)}\n\n"
        f"Developed branches:\n{_json(developed_branches)}\n\n"
        "Score, prune, and keep the best branches for mixing."
    )


def build_creativity_mixing_prompt(
    task: str,
    curiosity_packet: dict,
    research_plan: dict,
    selection: dict,
    developed_branches: list[dict],
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        f"Selection:\n{_json(selection)}\n\n"
        f"Developed branches:\n{_json(developed_branches)}\n\n"
        "Create hybrids from the kept branches and log dead ends."
    )


def build_creativity_final_synthesis_prompt(
    task: str,
    curiosity_packet: dict,
    research_plan: dict,
    selection: dict,
    mixing: dict,
    developed_branches: list[dict],
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        f"Selection:\n{_json(selection)}\n\n"
        f"Developed branches:\n{_json(developed_branches)}\n\n"
        f"Mixing results:\n{_json(mixing)}\n\n"
        "Produce the final synthesis and best candidates."
    )


def build_critic_advanced_prompt(
    task: str,
    curiosity_packet: dict,
    research_plan: dict,
    selection: dict,
    final_synthesis: dict,
) -> str:
    return (
        f"Task:\n{task}\n\n"
        f"Curiosity steering packet:\n{_json(curiosity_packet)}\n\n"
        f"Research plan:\n{_json(research_plan)}\n\n"
        f"Selection:\n{_json(selection)}\n\n"
        f"Final synthesis:\n{_json(final_synthesis)}\n\n"
        "Evaluate the final synthesis. If weak, give concrete repair feedback."
    )
