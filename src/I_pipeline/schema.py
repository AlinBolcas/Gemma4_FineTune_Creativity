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


def _to_str(value) -> str:
    return str(value or "").strip()


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str_list(value, limit: int | None = None) -> list[str]:
    items = [_to_str(v) for v in _as_list(value) if _to_str(v)]
    return items[:limit] if limit else items


def _clamp_score(value, default: int = 0) -> int:
    try:
        num = int(round(float(value)))
    except Exception:
        num = default
    return max(0, min(10, num))


def normalize_curiosity(data: dict) -> dict:
    data = data or {}
    questions = []
    raw_questions = _as_list(data.get("questions"))
    for idx, item in enumerate(raw_questions, 1):
        if isinstance(item, dict):
            questions.append(
                {
                    "id": f"Q{idx}",
                    "question": _to_str(item.get("question") or item.get("text") or item.get("id")),
                    "why_this_unlocks": _to_str(item.get("why_this_unlocks") or item.get("why") or item.get("rationale")),
                }
            )
        else:
            text = _to_str(item)
            if text:
                questions.append({"id": f"Q{idx}", "question": text, "why_this_unlocks": ""})

    return {
        "hidden_assumptions": _as_str_list(data.get("hidden_assumptions"), limit=4),
        "unexplored_domains": _as_str_list(data.get("unexplored_domains"), limit=4),
        "questions": [q for q in questions if q["question"]][:4],
        "branch_seeds": _as_str_list(data.get("branch_seeds"), limit=4),
    }


def normalize_creativity(data: dict) -> dict:
    data = data or {}
    id_map: dict[str, str] = {}

    branches = []
    for idx, item in enumerate(_as_list(data.get("branches")), 1):
        new_id = f"B{idx}"
        if isinstance(item, dict):
            old_id = _to_str(item.get("id")) or new_id
            id_map[old_id] = new_id
            candidates = _as_str_list(item.get("candidates") or item.get("examples"), limit=4)
            branches.append(
                {
                    "id": new_id,
                    "frame": _to_str(item.get("frame") or item.get("direction") or item.get("label")),
                    "candidates": candidates,
                }
            )
        else:
            text = _to_str(item)
            if text:
                id_map[new_id] = new_id
                branches.append({"id": new_id, "frame": text, "candidates": []})

    pruned = []
    for item in _as_list(data.get("pruned")):
        if isinstance(item, dict):
            old_id = _to_str(item.get("id"))
            pruned.append({"id": id_map.get(old_id, old_id or "B?"), "reason": _to_str(item.get("reason"))})
        else:
            text = _to_str(item)
            if text:
                pruned.append({"id": "B?", "reason": text})

    combinations = []
    for item in _as_list(data.get("combinations")):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                combinations.append({"from": [], "result": text, "novelty_note": ""})
            continue
        from_ids = [id_map.get(_to_str(v), _to_str(v)) for v in _as_list(item.get("from")) if _to_str(v)]
        combinations.append(
            {
                "from": from_ids[:3],
                "result": _to_str(item.get("result")),
                "novelty_note": _to_str(item.get("novelty_note") or item.get("why_novel") or item.get("note")),
            }
        )

    dead_ends = []
    for item in _as_list(data.get("dead_ends")):
        if isinstance(item, dict):
            combo = _to_str(item.get("combination"))
            reason = _to_str(item.get("reason"))
            text = f"{combo}: {reason}".strip(": ")
            if text:
                dead_ends.append(text)
        else:
            text = _to_str(item)
            if text:
                dead_ends.append(text)

    output = _as_str_list(data.get("output"), limit=6)
    if not output:
        output = [c["result"] for c in combinations if c.get("result")][:4]

    return {
        "research": _as_str_list(data.get("research"), limit=5),
        "branches": [b for b in branches if b["frame"]][:5],
        "pruned": [p for p in pruned if p["reason"]][:5],
        "combinations": [c for c in combinations if c["result"]][:4],
        "dead_ends": dead_ends[:4],
        "output": output,
    }


def normalize_critic(data: dict) -> dict:
    data = data or {}
    scores = []
    for item in _as_list(data.get("scores")):
        if isinstance(item, dict):
            scores.append(
                {
                    "candidate": _to_str(item.get("candidate") or item.get("text")),
                    "novelty": _clamp_score(item.get("novelty"), 0),
                    "relevance": _clamp_score(item.get("relevance"), 0),
                    "notes": _to_str(item.get("notes") or item.get("why")),
                }
            )
        else:
            text = _to_str(item)
            if text:
                scores.append({"candidate": text, "novelty": 0, "relevance": 0, "notes": ""})

    verdict = _to_str(data.get("verdict")).upper()
    if verdict not in ("PASS", "FAIL"):
        best_n = max((s["novelty"] for s in scores), default=0)
        best_r = max((s["relevance"] for s in scores), default=0)
        verdict = "PASS" if best_n >= PASS_THRESHOLD_NOVELTY and best_r >= PASS_THRESHOLD_RELEVANCE else "FAIL"

    return {
        "scores": [s for s in scores if s["candidate"]][:8],
        "verdict": verdict,
        "unexplored_directions": _as_str_list(data.get("unexplored_directions"), limit=4),
        "feedback_for_curiosity": _as_str_list(data.get("feedback_for_curiosity"), limit=4),
    }


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
