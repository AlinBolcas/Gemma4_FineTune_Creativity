"""
schema.py - Strict JSON schemas for the Curiosity -> Creativity -> Critic loop.

These schemas define the training data format. Every synthetic example,
every JSONL row, and every eval output must conform to these shapes.
"""

# ---------------------------------------------------------------------------
# Module output schemas (what each stage returns)
# ---------------------------------------------------------------------------

CURIOSITY_SCHEMA = {
    "hidden_assumptions": ["str"],
    "unexplored_domains": ["str"],
    "questions": [
        {"id": "Q1", "question": "str", "why_this_unlocks": "str"}
    ],
    "branch_seeds": ["str"],
}

CREATIVITY_SCHEMA = {
    "research": ["str"],
    "branches": [
        {"id": "B1", "frame": "str", "candidates": ["str"]}
    ],
    "pruned": [
        {"id": "B2", "reason": "str"}
    ],
    "combinations": [
        {"from": ["B1", "B3"], "result": "str", "novelty_note": "str"}
    ],
    "dead_ends": ["str"],
    "output": ["str"],
}

CRITIC_SCHEMA = {
    "scores": [
        {"candidate": "str", "novelty": 0, "relevance": 0, "notes": "str"}
    ],
    "verdict": "PASS | FAIL",
    "unexplored_directions": ["str"],
    "feedback_for_curiosity": ["str"],
}

# ---------------------------------------------------------------------------
# Full-loop training example schema (one JSONL row)
# ---------------------------------------------------------------------------

FULL_LOOP_SCHEMA = {
    "domain": "str",
    "input": "str",
    "loop": [
        {
            "iteration": 1,
            "curiosity": CURIOSITY_SCHEMA,
            "creativity": CREATIVITY_SCHEMA,
            "critic": CRITIC_SCHEMA,
        }
    ],
    "final_output": ["str"],
}

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

REQUIRED_CURIOSITY_KEYS = {"hidden_assumptions", "unexplored_domains", "questions", "branch_seeds"}
REQUIRED_CREATIVITY_KEYS = {"research", "branches", "output"}
REQUIRED_CRITIC_KEYS = {"scores", "verdict"}

PASS_THRESHOLD_NOVELTY = 7
PASS_THRESHOLD_RELEVANCE = 7


def validate_curiosity(data: dict) -> list[str]:
    """Return list of error strings (empty = valid)."""
    errors = []
    for key in REQUIRED_CURIOSITY_KEYS:
        if key not in data:
            errors.append(f"curiosity missing '{key}'")
    if not data.get("questions"):
        errors.append("curiosity has no questions")
    return errors


def validate_creativity(data: dict) -> list[str]:
    errors = []
    for key in REQUIRED_CREATIVITY_KEYS:
        if key not in data:
            errors.append(f"creativity missing '{key}'")
    if not data.get("branches"):
        errors.append("creativity has no branches")
    if not data.get("output"):
        errors.append("creativity has no output")
    return errors


def validate_critic(data: dict) -> list[str]:
    errors = []
    for key in REQUIRED_CRITIC_KEYS:
        if key not in data:
            errors.append(f"critic missing '{key}'")
    verdict = data.get("verdict", "").upper()
    if verdict not in ("PASS", "FAIL"):
        errors.append(f"critic verdict '{verdict}' not PASS/FAIL")
    return errors


def validate_full_loop(data: dict) -> list[str]:
    errors = []
    if "domain" not in data:
        errors.append("missing 'domain'")
    if "input" not in data:
        errors.append("missing 'input'")
    if "loop" not in data or not data["loop"]:
        errors.append("missing or empty 'loop'")
        return errors
    for i, iteration in enumerate(data["loop"]):
        prefix = f"loop[{i}]"
        if "curiosity" in iteration:
            errors.extend(f"{prefix}.{e}" for e in validate_curiosity(iteration["curiosity"]))
        else:
            errors.append(f"{prefix} missing 'curiosity'")
        if "creativity" in iteration:
            errors.extend(f"{prefix}.{e}" for e in validate_creativity(iteration["creativity"]))
        else:
            errors.append(f"{prefix} missing 'creativity'")
        if "critic" in iteration:
            errors.extend(f"{prefix}.{e}" for e in validate_critic(iteration["critic"]))
        else:
            errors.append(f"{prefix} missing 'critic'")
    if "final_output" not in data or not data["final_output"]:
        errors.append("missing or empty 'final_output'")
    return errors
