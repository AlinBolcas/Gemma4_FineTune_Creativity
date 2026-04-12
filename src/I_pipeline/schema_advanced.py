"""
schema_advanced.py - Full stage-by-stage schemas for the advanced pipeline.

This version mirrors the diagram more closely:
1. Curiosity map
2. Curiosity expand
3. Curiosity distill
4. Socratic output
5. Creativity research plan
6. Creativity branch
7. Creativity develop each branch
8. Creativity selection
9. Creativity combinatory mixing
10. Creativity final synthesis
11. Critic

It also exposes summary helpers so advanced runs can still be exported by the
existing markdown exporter.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Stage schemas
# ---------------------------------------------------------------------------

CURIOSITY_MAP_SCHEMA = {
    "global_novelty_estimate": 0,
    "branch_budget": 2,
    "known_context": ["str"],
    "hidden_assumptions": ["str"],
    "curiosity_domains": [
        {"id": "D1", "lens": "str", "domain": "str", "novelty_opportunity": "str"}
    ],
    "seed_questions": [
        {"id": "Q1", "question": "str", "domain_id": "D1"}
    ],
    "frontier_notes": ["str"],
}

CURIOSITY_EXPAND_SCHEMA = {
    "expanded_branches": [
        {
            "id": "CB1",
            "domain_id": "D1",
            "direction": "str",
            "questions": ["str"],
            "why_non_obvious": "str",
            "curiosity_strength": 0,
            "keep": True,
        }
    ],
    "pruned_branches": [
        {"id": "CB4", "reason": "str"}
    ],
    "branch_budget_used": 2,
}

CURIOSITY_DISTILL_SCHEMA = {
    "best_questions": [
        {
            "id": "DQ1",
            "question": "str",
            "source_branch_ids": ["CB1"],
            "leverage_score": 0,
            "why_high_leverage": "str",
        }
    ],
    "socratic_scaffold": ["str"],
    "exploration_direction": "str",
    "steering_signals": ["str"],
    "handoff_notes": ["str"],
}

SOCRATIC_OUTPUT_SCHEMA = {
    "question_set": ["str"],
    "scaffold": ["str"],
    "direction": "str",
    "constraints": ["str"],
    "priority_domains": ["str"],
    "novelty_focus": ["str"],
}

CREATIVITY_RESEARCH_PLAN_SCHEMA = {
    "complexity": 0,
    "branch_budget": 2,
    "known_patterns": ["str"],
    "adjacent_domains": ["str"],
    "creative_tensions": ["str"],
    "research_queries": ["str"],
    "research_notes": ["str"],
}

CREATIVITY_BRANCH_SCHEMA = {
    "branches": [
        {
            "id": "B1",
            "frame": "str",
            "domain": "str",
            "constraint": "str",
            "examples": ["str"],
            "why_distinct": "str",
        }
    ],
    "pruned_branch_ideas": [
        {"idea": "str", "reason": "str"}
    ],
}

CREATIVITY_DEVELOP_BRANCH_SCHEMA = {
    "branch_id": "B1",
    "chain_steps": ["str"],
    "branch_outputs": ["str"],
    "exhausted_when": "str",
    "risks": ["str"],
    "novelty_delta": "str",
}

CREATIVITY_SELECTION_SCHEMA = {
    "scored_branches": [
        {
            "branch_id": "B1",
            "novelty": 0,
            "relevance": 0,
            "combinability": 0,
            "decision": "keep",
            "reason": "str",
        }
    ],
    "kept_branch_ids": ["B1"],
    "pruned_branch_ids": ["B2"],
    "selection_rationale": ["str"],
}

CREATIVITY_MIXING_SCHEMA = {
    "hybrids": [
        {
            "id": "H1",
            "from_branch_ids": ["B1", "B3"],
            "concept": "str",
            "strength_score": 0,
            "why_novel": "str",
        }
    ],
    "dead_ends": [
        {"from_branch_ids": ["B1", "B2"], "reason": "str"}
    ],
}

CREATIVITY_FINAL_SYNTHESIS_SCHEMA = {
    "primary_candidates": [
        {
            "id": "C1",
            "title": "str",
            "concept": "str",
            "built_from": ["B1", "H1"],
            "novelty_notes": "str",
        }
    ],
    "best_combination": "str",
    "output": ["str"],
    "novelty_notes": ["str"],
}

CRITIC_ADVANCED_SCHEMA = {
    "scores": [
        {
            "candidate_id": "C1",
            "candidate": "str",
            "novelty": 0,
            "relevance": 0,
            "notes": "str",
        }
    ],
    "verdict": "PASS | FAIL",
    "unexplored_directions": ["str"],
    "feedback_for_curiosity": ["str"],
    "feedback_for_creativity": ["str"],
}

ADVANCED_FULL_LOOP_SCHEMA = {
    "domain": "str",
    "input": "str",
    "loop": [
        {
            "iteration": 1,
            "curiosity": {
                "hidden_assumptions": ["str"],
                "unexplored_domains": ["str"],
                "questions": [{"id": "Q1", "question": "str", "why_this_unlocks": "str"}],
                "branch_seeds": ["str"],
            },
            "creativity": {
                "research": ["str"],
                "branches": [{"id": "B1", "frame": "str", "candidates": ["str"]}],
                "pruned": [{"id": "B2", "reason": "str"}],
                "combinations": [{"from": ["B1", "B3"], "result": "str", "novelty_note": "str"}],
                "dead_ends": ["str"],
                "output": ["str"],
            },
            "critic": CRITIC_ADVANCED_SCHEMA,
            "advanced": {
                "curiosity_map": CURIOSITY_MAP_SCHEMA,
                "curiosity_expand": CURIOSITY_EXPAND_SCHEMA,
                "curiosity_distill": CURIOSITY_DISTILL_SCHEMA,
                "socratic_output": SOCRATIC_OUTPUT_SCHEMA,
                "creativity_research_plan": CREATIVITY_RESEARCH_PLAN_SCHEMA,
                "creativity_branch": CREATIVITY_BRANCH_SCHEMA,
                "creativity_develop": [CREATIVITY_DEVELOP_BRANCH_SCHEMA],
                "creativity_selection": CREATIVITY_SELECTION_SCHEMA,
                "creativity_mixing": CREATIVITY_MIXING_SCHEMA,
                "creativity_final_synthesis": CREATIVITY_FINAL_SYNTHESIS_SCHEMA,
            },
        }
    ],
    "final_output": ["str"],
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PASS_THRESHOLD_NOVELTY = 8
PASS_THRESHOLD_RELEVANCE = 8


def _has_passing_candidate(scores: list[dict]) -> bool:
    return any(
        s.get("novelty", 0) >= PASS_THRESHOLD_NOVELTY and s.get("relevance", 0) >= PASS_THRESHOLD_RELEVANCE
        for s in scores
    )


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


def _clamp_score(value, default: int = 0, low: int = 0, high: int = 10) -> int:
    try:
        num = int(round(float(value)))
    except Exception:
        num = default
    return max(low, min(high, num))


def _unique(items: list[str], limit: int | None = None) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
            if limit and len(out) >= limit:
                break
    return out


def _normalize_question_dict(item, idx: int, prefix: str) -> dict:
    if isinstance(item, dict):
        return {
            "id": f"{prefix}{idx}",
            "question": _to_str(item.get("question") or item.get("text") or item.get("id")),
            "why": _to_str(item.get("why") or item.get("why_this_unlocks") or item.get("reason")),
        }
    text = _to_str(item)
    return {"id": f"{prefix}{idx}", "question": text, "why": ""}


# ---------------------------------------------------------------------------
# Curiosity normalizers
# ---------------------------------------------------------------------------

def normalize_curiosity_map(data: dict) -> dict:
    data = data or {}

    domains = []
    for idx, item in enumerate(_as_list(data.get("curiosity_domains")), 1):
        if isinstance(item, dict):
            domains.append(
                {
                    "id": f"D{idx}",
                    "lens": _to_str(item.get("lens") or item.get("angle") or item.get("id")),
                    "domain": _to_str(item.get("domain") or item.get("label") or item.get("name")),
                    "novelty_opportunity": _to_str(
                        item.get("novelty_opportunity") or item.get("why_novel") or item.get("reason")
                    ),
                }
            )
        else:
            text = _to_str(item)
            if text:
                domains.append({"id": f"D{idx}", "lens": "", "domain": text, "novelty_opportunity": ""})

    seeds = []
    for idx, item in enumerate(_as_list(data.get("seed_questions")), 1):
        q = _normalize_question_dict(item, idx, "Q")
        domain_id = ""
        if isinstance(item, dict):
            domain_id = _to_str(item.get("domain_id") or item.get("domain"))
        seeds.append({"id": q["id"], "question": q["question"], "domain_id": domain_id})

    budget_default = max(2, min(6, len(domains) or 3))
    return {
        "global_novelty_estimate": _clamp_score(data.get("global_novelty_estimate"), 5),
        "branch_budget": _clamp_score(data.get("branch_budget"), budget_default, 2, 6),
        "known_context": _as_str_list(data.get("known_context"), limit=5),
        "hidden_assumptions": _as_str_list(data.get("hidden_assumptions"), limit=5),
        "curiosity_domains": [d for d in domains if d["domain"]][:6],
        "seed_questions": [s for s in seeds if s["question"]][:6],
        "frontier_notes": _as_str_list(data.get("frontier_notes"), limit=5),
    }


def normalize_curiosity_expand(data: dict) -> dict:
    data = data or {}
    expanded = []
    for idx, item in enumerate(_as_list(data.get("expanded_branches")), 1):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                expanded.append(
                    {
                        "id": f"CB{idx}",
                        "domain_id": "",
                        "direction": text,
                        "questions": [],
                        "why_non_obvious": "",
                        "curiosity_strength": 5,
                        "keep": True,
                    }
                )
            continue
        expanded.append(
            {
                "id": f"CB{idx}",
                "domain_id": _to_str(item.get("domain_id") or item.get("domain")),
                "direction": _to_str(item.get("direction") or item.get("frame") or item.get("label")),
                "questions": _as_str_list(item.get("questions"), limit=3),
                "why_non_obvious": _to_str(item.get("why_non_obvious") or item.get("why") or item.get("reason")),
                "curiosity_strength": _clamp_score(item.get("curiosity_strength"), 5),
                "keep": bool(item.get("keep", True)),
            }
        )

    pruned = []
    for idx, item in enumerate(_as_list(data.get("pruned_branches")), 1):
        if isinstance(item, dict):
            pruned.append(
                {
                    "id": _to_str(item.get("id")) or f"CBP{idx}",
                    "reason": _to_str(item.get("reason") or item.get("why")),
                }
            )
        else:
            text = _to_str(item)
            if text:
                pruned.append({"id": f"CBP{idx}", "reason": text})

    branch_budget_used = _clamp_score(data.get("branch_budget_used"), len(expanded) or 2, 1, 6)
    return {
        "expanded_branches": [b for b in expanded if b["direction"]][:6],
        "pruned_branches": [p for p in pruned if p["reason"]][:6],
        "branch_budget_used": branch_budget_used,
    }


def normalize_curiosity_distill(data: dict) -> dict:
    data = data or {}
    best = []
    for idx, item in enumerate(_as_list(data.get("best_questions")), 1):
        q = _normalize_question_dict(item, idx, "DQ")
        source_ids = []
        leverage = 5
        why = q["why"]
        if isinstance(item, dict):
            source_ids = _as_str_list(item.get("source_branch_ids"), limit=3)
            leverage = _clamp_score(item.get("leverage_score"), 5)
            why = _to_str(item.get("why_high_leverage") or why)
        best.append(
            {
                "id": q["id"],
                "question": q["question"],
                "source_branch_ids": source_ids,
                "leverage_score": leverage,
                "why_high_leverage": why,
            }
        )
    return {
        "best_questions": [q for q in best if q["question"]][:5],
        "socratic_scaffold": _as_str_list(data.get("socratic_scaffold"), limit=5),
        "exploration_direction": _to_str(data.get("exploration_direction")),
        "steering_signals": _as_str_list(data.get("steering_signals"), limit=5),
        "handoff_notes": _as_str_list(data.get("handoff_notes"), limit=5),
    }


def normalize_socratic_output(data: dict) -> dict:
    data = data or {}
    return {
        "question_set": _as_str_list(data.get("question_set"), limit=5),
        "scaffold": _as_str_list(data.get("scaffold"), limit=5),
        "direction": _to_str(data.get("direction")),
        "constraints": _as_str_list(data.get("constraints"), limit=5),
        "priority_domains": _as_str_list(data.get("priority_domains"), limit=5),
        "novelty_focus": _as_str_list(data.get("novelty_focus"), limit=5),
    }


# ---------------------------------------------------------------------------
# Creativity normalizers
# ---------------------------------------------------------------------------

def normalize_creativity_research_plan(data: dict) -> dict:
    data = data or {}
    return {
        "complexity": _clamp_score(data.get("complexity"), 5),
        "branch_budget": _clamp_score(data.get("branch_budget"), 3, 2, 6),
        "known_patterns": _as_str_list(data.get("known_patterns"), limit=5),
        "adjacent_domains": _as_str_list(data.get("adjacent_domains"), limit=5),
        "creative_tensions": _as_str_list(data.get("creative_tensions"), limit=5),
        "research_queries": _as_str_list(data.get("research_queries"), limit=5),
        "research_notes": _as_str_list(data.get("research_notes"), limit=5),
    }


def normalize_creativity_branch(data: dict) -> dict:
    data = data or {}
    branches = []
    for idx, item in enumerate(_as_list(data.get("branches")), 1):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                branches.append(
                    {
                        "id": f"B{idx}",
                        "frame": text,
                        "domain": "",
                        "constraint": "",
                        "examples": [],
                        "why_distinct": "",
                    }
                )
            continue
        branches.append(
            {
                "id": f"B{idx}",
                "frame": _to_str(item.get("frame") or item.get("direction") or item.get("label")),
                "domain": _to_str(item.get("domain")),
                "constraint": _to_str(item.get("constraint")),
                "examples": _as_str_list(item.get("examples"), limit=4),
                "why_distinct": _to_str(item.get("why_distinct") or item.get("reason")),
            }
        )

    pruned = []
    for item in _as_list(data.get("pruned_branch_ideas")):
        if isinstance(item, dict):
            pruned.append({"idea": _to_str(item.get("idea")), "reason": _to_str(item.get("reason"))})
        else:
            text = _to_str(item)
            if text:
                pruned.append({"idea": text, "reason": ""})

    return {
        "branches": [b for b in branches if b["frame"]][:6],
        "pruned_branch_ideas": [p for p in pruned if p["idea"] or p["reason"]][:6],
    }


def normalize_creativity_develop_branch(data: dict, expected_branch_id: str | None = None) -> dict:
    data = data or {}
    branch_id = _to_str(data.get("branch_id")) or _to_str(expected_branch_id) or "B1"
    return {
        "branch_id": branch_id,
        "chain_steps": _as_str_list(data.get("chain_steps"), limit=5),
        "branch_outputs": _as_str_list(data.get("branch_outputs"), limit=5),
        "exhausted_when": _to_str(data.get("exhausted_when")),
        "risks": _as_str_list(data.get("risks"), limit=4),
        "novelty_delta": _to_str(data.get("novelty_delta")),
    }


def normalize_creativity_selection(data: dict) -> dict:
    data = data or {}
    scored = []
    for item in _as_list(data.get("scored_branches")):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                scored.append(
                    {
                        "branch_id": "",
                        "novelty": 0,
                        "relevance": 0,
                        "combinability": 0,
                        "decision": "prune",
                        "reason": text,
                    }
                )
            continue
        decision = _to_str(item.get("decision")).lower()
        if decision not in ("keep", "prune"):
            avg = (
                _clamp_score(item.get("novelty"), 0)
                + _clamp_score(item.get("relevance"), 0)
                + _clamp_score(item.get("combinability"), 0)
            ) / 3
            decision = "keep" if avg >= 6 else "prune"
        scored.append(
            {
                "branch_id": _to_str(item.get("branch_id") or item.get("id")),
                "novelty": _clamp_score(item.get("novelty"), 0),
                "relevance": _clamp_score(item.get("relevance"), 0),
                "combinability": _clamp_score(item.get("combinability"), 0),
                "decision": decision,
                "reason": _to_str(item.get("reason") or item.get("notes")),
            }
        )
    kept = _unique(_as_str_list(data.get("kept_branch_ids"), limit=4))
    pruned = _unique(_as_str_list(data.get("pruned_branch_ids"), limit=6))
    if not kept:
        kept = [s["branch_id"] for s in scored if s["branch_id"] and s["decision"] == "keep"][:3]
    if not pruned:
        pruned = [s["branch_id"] for s in scored if s["branch_id"] and s["decision"] == "prune"][:6]
    return {
        "scored_branches": [s for s in scored if s["branch_id"] or s["reason"]][:8],
        "kept_branch_ids": kept,
        "pruned_branch_ids": pruned,
        "selection_rationale": _as_str_list(data.get("selection_rationale"), limit=5),
    }


def normalize_creativity_mixing(data: dict) -> dict:
    data = data or {}
    hybrids = []
    for idx, item in enumerate(_as_list(data.get("hybrids")), 1):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                hybrids.append(
                    {
                        "id": f"H{idx}",
                        "from_branch_ids": [],
                        "concept": text,
                        "strength_score": 5,
                        "why_novel": "",
                    }
                )
            continue
        hybrids.append(
            {
                "id": f"H{idx}",
                "from_branch_ids": _as_str_list(item.get("from_branch_ids") or item.get("from"), limit=3),
                "concept": _to_str(item.get("concept") or item.get("result")),
                "strength_score": _clamp_score(item.get("strength_score"), 5),
                "why_novel": _to_str(item.get("why_novel") or item.get("novelty_note")),
            }
        )

    dead_ends = []
    for item in _as_list(data.get("dead_ends")):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                dead_ends.append({"from_branch_ids": [], "reason": text})
            continue
        dead_ends.append(
            {
                "from_branch_ids": _as_str_list(item.get("from_branch_ids") or item.get("from"), limit=3),
                "reason": _to_str(item.get("reason")),
            }
        )

    return {
        "hybrids": [h for h in hybrids if h["concept"]][:5],
        "dead_ends": [d for d in dead_ends if d["reason"]][:5],
    }


def normalize_creativity_final_synthesis(data: dict) -> dict:
    data = data or {}
    primary = []
    for idx, item in enumerate(_as_list(data.get("primary_candidates")), 1):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                primary.append(
                    {
                        "id": f"C{idx}",
                        "title": f"Candidate {idx}",
                        "concept": text,
                        "built_from": [],
                        "novelty_notes": "",
                    }
                )
            continue
        primary.append(
            {
                "id": f"C{idx}",
                "title": _to_str(item.get("title") or item.get("id") or f"Candidate {idx}"),
                "concept": _to_str(item.get("concept") or item.get("candidate") or item.get("result")),
                "built_from": _as_str_list(item.get("built_from") or item.get("from"), limit=4),
                "novelty_notes": _to_str(item.get("novelty_notes") or item.get("why_novel") or item.get("note")),
            }
        )
    output = _as_str_list(data.get("output"), limit=5)
    if not output:
        output = [p["concept"] for p in primary if p["concept"]][:4]
    notes = _as_str_list(data.get("novelty_notes"), limit=5)
    if not notes:
        notes = [p["novelty_notes"] for p in primary if p["novelty_notes"]][:5]
    return {
        "primary_candidates": [p for p in primary if p["concept"]][:5],
        "best_combination": _to_str(data.get("best_combination")),
        "output": output,
        "novelty_notes": notes,
    }


def normalize_critic_advanced(data: dict) -> dict:
    data = data or {}
    scores = []
    for idx, item in enumerate(_as_list(data.get("scores")), 1):
        if not isinstance(item, dict):
            text = _to_str(item)
            if text:
                scores.append(
                    {
                        "candidate_id": f"C{idx}",
                        "candidate": text,
                        "novelty": 0,
                        "relevance": 0,
                        "notes": "",
                    }
                )
            continue
        scores.append(
            {
                "candidate_id": _to_str(item.get("candidate_id") or item.get("id") or f"C{idx}"),
                "candidate": _to_str(item.get("candidate") or item.get("text")),
                "novelty": _clamp_score(item.get("novelty"), 0),
                "relevance": _clamp_score(item.get("relevance"), 0),
                "notes": _to_str(item.get("notes") or item.get("why")),
            }
        )

    verdict = _to_str(data.get("verdict")).upper()
    if verdict not in ("PASS", "FAIL"):
        best_n = max((s["novelty"] for s in scores), default=0)
        best_r = max((s["relevance"] for s in scores), default=0)
        verdict = "PASS" if best_n >= PASS_THRESHOLD_NOVELTY and best_r >= PASS_THRESHOLD_RELEVANCE else "FAIL"

    return {
        "scores": [s for s in scores if s["candidate"]][:8],
        "verdict": verdict,
        "unexplored_directions": _as_str_list(data.get("unexplored_directions"), limit=5),
        "feedback_for_curiosity": _as_str_list(data.get("feedback_for_curiosity"), limit=5),
        "feedback_for_creativity": _as_str_list(data.get("feedback_for_creativity"), limit=5),
    }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def validate_curiosity_map(data: dict) -> list[str]:
    errors = []
    if not data.get("curiosity_domains"):
        errors.append("curiosity_map has no curiosity_domains")
    if not data.get("seed_questions"):
        errors.append("curiosity_map has no seed_questions")
    return errors


def validate_curiosity_expand(data: dict) -> list[str]:
    errors = []
    if not data.get("expanded_branches"):
        errors.append("curiosity_expand has no expanded_branches")
    return errors


def validate_curiosity_distill(data: dict) -> list[str]:
    errors = []
    if not data.get("best_questions"):
        errors.append("curiosity_distill has no best_questions")
    if not data.get("socratic_scaffold"):
        errors.append("curiosity_distill has no socratic_scaffold")
    return errors


def validate_socratic_output(data: dict) -> list[str]:
    errors = []
    if not data.get("question_set"):
        errors.append("socratic_output has no question_set")
    if not data.get("direction"):
        errors.append("socratic_output has no direction")
    return errors


def validate_creativity_research_plan(data: dict) -> list[str]:
    errors = []
    if not data.get("creative_tensions"):
        errors.append("creativity_research_plan has no creative_tensions")
    return errors


def validate_creativity_branch(data: dict) -> list[str]:
    errors = []
    if not data.get("branches"):
        errors.append("creativity_branch has no branches")
    return errors


def validate_creativity_develop_branch(data: dict) -> list[str]:
    errors = []
    if not data.get("branch_id"):
        errors.append("creativity_develop_branch has no branch_id")
    if not data.get("branch_outputs"):
        errors.append("creativity_develop_branch has no branch_outputs")
    return errors


def validate_creativity_selection(data: dict) -> list[str]:
    errors = []
    if not data.get("scored_branches"):
        errors.append("creativity_selection has no scored_branches")
    if not data.get("kept_branch_ids"):
        errors.append("creativity_selection has no kept_branch_ids")
    return errors


def validate_creativity_mixing(data: dict) -> list[str]:
    errors = []
    if not data.get("hybrids"):
        errors.append("creativity_mixing has no hybrids")
    return errors


def validate_creativity_final_synthesis(data: dict) -> list[str]:
    errors = []
    if not data.get("primary_candidates"):
        errors.append("creativity_final_synthesis has no primary_candidates")
    if not data.get("output"):
        errors.append("creativity_final_synthesis has no output")
    return errors


def validate_critic_advanced(data: dict) -> list[str]:
    errors = []
    if not data.get("scores"):
        errors.append("critic_advanced has no scores")
    verdict = data.get("verdict", "").upper()
    if verdict not in ("PASS", "FAIL"):
        errors.append(f"critic_advanced verdict '{verdict}' not PASS/FAIL")
    scores = data.get("scores", [])
    if verdict == "PASS" and not _has_passing_candidate(scores):
        errors.append("critic_advanced PASS without candidate meeting threshold")
    if verdict == "FAIL":
        if len(data.get("unexplored_directions", [])) < 2:
            errors.append("critic_advanced FAIL should include at least 2 unexplored_directions")
        if len(data.get("feedback_for_curiosity", [])) < 2:
            errors.append("critic_advanced FAIL should include at least 2 feedback_for_curiosity notes")
        if len(data.get("feedback_for_creativity", [])) < 2:
            errors.append("critic_advanced FAIL should include at least 2 feedback_for_creativity notes")
    return errors


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------

def fallback_curiosity_map() -> dict:
    return {
        "global_novelty_estimate": 5,
        "branch_budget": 3,
        "known_context": [],
        "hidden_assumptions": [],
        "curiosity_domains": [{"id": "D1", "lens": "fallback", "domain": "unmapped area", "novelty_opportunity": ""}],
        "seed_questions": [{"id": "Q1", "question": "What are we overlooking?", "domain_id": "D1"}],
        "frontier_notes": [],
    }


def fallback_curiosity_expand() -> dict:
    return {
        "expanded_branches": [
            {
                "id": "CB1",
                "domain_id": "D1",
                "direction": "probe overlooked assumption",
                "questions": ["What assumption should be inverted?"],
                "why_non_obvious": "",
                "curiosity_strength": 5,
                "keep": True,
            }
        ],
        "pruned_branches": [],
        "branch_budget_used": 1,
    }


def fallback_curiosity_distill() -> dict:
    return {
        "best_questions": [
            {
                "id": "DQ1",
                "question": "What question opens the largest unexplored space?",
                "source_branch_ids": ["CB1"],
                "leverage_score": 5,
                "why_high_leverage": "",
            }
        ],
        "socratic_scaffold": ["interrogate assumptions", "seek adjacent domain transfer"],
        "exploration_direction": "pursue the most generative question first",
        "steering_signals": ["prefer structural novelty"],
        "handoff_notes": ["creativity should use questions as steering, not decoration"],
    }


def fallback_socratic_output() -> dict:
    return {
        "question_set": ["What overlooked framing changes the solution space?"],
        "scaffold": ["invert default assumption", "borrow from adjacent domain"],
        "direction": "open the space before converging",
        "constraints": [],
        "priority_domains": [],
        "novelty_focus": ["structural distance", "non-obvious recombination"],
    }


def fallback_creativity_research_plan() -> dict:
    return {
        "complexity": 5,
        "branch_budget": 3,
        "known_patterns": [],
        "adjacent_domains": [],
        "creative_tensions": ["novel but still relevant"],
        "research_queries": [],
        "research_notes": [],
    }


def fallback_creativity_branch() -> dict:
    return {
        "branches": [
            {
                "id": "B1",
                "frame": "baseline exploratory branch",
                "domain": "",
                "constraint": "",
                "examples": [],
                "why_distinct": "",
            }
        ],
        "pruned_branch_ideas": [],
    }


def fallback_creativity_develop_branch(expected_branch_id: str | None = None) -> dict:
    bid = expected_branch_id or "B1"
    return {
        "branch_id": bid,
        "chain_steps": ["push the branch until a concrete idea appears"],
        "branch_outputs": [f"{bid} candidate"],
        "exhausted_when": "variation stops adding novelty",
        "risks": [],
        "novelty_delta": "",
    }


def fallback_creativity_selection() -> dict:
    return {
        "scored_branches": [
            {
                "branch_id": "B1",
                "novelty": 5,
                "relevance": 5,
                "combinability": 5,
                "decision": "keep",
                "reason": "fallback keep",
            }
        ],
        "kept_branch_ids": ["B1"],
        "pruned_branch_ids": [],
        "selection_rationale": [],
    }


def fallback_creativity_mixing() -> dict:
    return {
        "hybrids": [
            {
                "id": "H1",
                "from_branch_ids": ["B1"],
                "concept": "single-branch refinement",
                "strength_score": 5,
                "why_novel": "",
            }
        ],
        "dead_ends": [],
    }


def fallback_creativity_final_synthesis() -> dict:
    return {
        "primary_candidates": [
            {
                "id": "C1",
                "title": "Fallback candidate",
                "concept": "No strong synthesis available.",
                "built_from": ["B1"],
                "novelty_notes": "",
            }
        ],
        "best_combination": "B1",
        "output": ["No strong synthesis available."],
        "novelty_notes": [],
    }


def fallback_critic_advanced() -> dict:
    return {
        "scores": [{"candidate_id": "C1", "candidate": "(generation failed)", "novelty": 0, "relevance": 0, "notes": ""}],
        "verdict": "FAIL",
        "unexplored_directions": ["retry with stronger structural distance"],
        "feedback_for_curiosity": ["map more aggressive question space"],
        "feedback_for_creativity": ["branch further before converging"],
    }


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------

ADVANCED_STAGE_ORDER = [
    "curiosity_map",
    "curiosity_expand",
    "curiosity_distill",
    "socratic_output",
    "creativity_research_plan",
    "creativity_branch",
    "creativity_develop_branch",
    "creativity_selection",
    "creativity_mixing",
    "creativity_final_synthesis",
    "critic_advanced",
]


def normalize_stage(stage_name: str, data: dict, expected_branch_id: str | None = None) -> dict:
    if stage_name == "curiosity_map":
        return normalize_curiosity_map(data)
    if stage_name == "curiosity_expand":
        return normalize_curiosity_expand(data)
    if stage_name == "curiosity_distill":
        return normalize_curiosity_distill(data)
    if stage_name == "socratic_output":
        return normalize_socratic_output(data)
    if stage_name == "creativity_research_plan":
        return normalize_creativity_research_plan(data)
    if stage_name == "creativity_branch":
        return normalize_creativity_branch(data)
    if stage_name == "creativity_develop_branch":
        return normalize_creativity_develop_branch(data, expected_branch_id=expected_branch_id)
    if stage_name == "creativity_selection":
        return normalize_creativity_selection(data)
    if stage_name == "creativity_mixing":
        return normalize_creativity_mixing(data)
    if stage_name == "creativity_final_synthesis":
        return normalize_creativity_final_synthesis(data)
    if stage_name == "critic_advanced":
        return normalize_critic_advanced(data)
    return data or {}


def validate_stage(stage_name: str, data: dict) -> list[str]:
    if stage_name == "curiosity_map":
        return validate_curiosity_map(data)
    if stage_name == "curiosity_expand":
        return validate_curiosity_expand(data)
    if stage_name == "curiosity_distill":
        return validate_curiosity_distill(data)
    if stage_name == "socratic_output":
        return validate_socratic_output(data)
    if stage_name == "creativity_research_plan":
        return validate_creativity_research_plan(data)
    if stage_name == "creativity_branch":
        return validate_creativity_branch(data)
    if stage_name == "creativity_develop_branch":
        return validate_creativity_develop_branch(data)
    if stage_name == "creativity_selection":
        return validate_creativity_selection(data)
    if stage_name == "creativity_mixing":
        return validate_creativity_mixing(data)
    if stage_name == "creativity_final_synthesis":
        return validate_creativity_final_synthesis(data)
    if stage_name == "critic_advanced":
        return validate_critic_advanced(data)
    return []


def fallback_stage(stage_name: str, expected_branch_id: str | None = None) -> dict:
    if stage_name == "curiosity_map":
        return fallback_curiosity_map()
    if stage_name == "curiosity_expand":
        return fallback_curiosity_expand()
    if stage_name == "curiosity_distill":
        return fallback_curiosity_distill()
    if stage_name == "socratic_output":
        return fallback_socratic_output()
    if stage_name == "creativity_research_plan":
        return fallback_creativity_research_plan()
    if stage_name == "creativity_branch":
        return fallback_creativity_branch()
    if stage_name == "creativity_develop_branch":
        return fallback_creativity_develop_branch(expected_branch_id)
    if stage_name == "creativity_selection":
        return fallback_creativity_selection()
    if stage_name == "creativity_mixing":
        return fallback_creativity_mixing()
    if stage_name == "creativity_final_synthesis":
        return fallback_creativity_final_synthesis()
    if stage_name == "critic_advanced":
        return fallback_critic_advanced()
    return {}


# ---------------------------------------------------------------------------
# Compatibility summaries for the existing exporter
# ---------------------------------------------------------------------------

def summarize_curiosity_trace(
    curiosity_map: dict,
    curiosity_expand: dict,
    curiosity_distill: dict,
    socratic_output: dict,
) -> dict:
    unexplored_domains = _unique(
        [d.get("domain", "") for d in curiosity_map.get("curiosity_domains", [])]
        + curiosity_map.get("frontier_notes", []),
        limit=6,
    )
    questions = [
        {
            "id": q.get("id", f"Q{i}"),
            "question": q.get("question", ""),
            "why_this_unlocks": q.get("why_high_leverage", ""),
        }
        for i, q in enumerate(curiosity_distill.get("best_questions", []), 1)
        if q.get("question")
    ]
    branch_seeds = _unique(
        curiosity_distill.get("steering_signals", [])
        + socratic_output.get("novelty_focus", [])
        + [b.get("direction", "") for b in curiosity_expand.get("expanded_branches", []) if b.get("keep")],
        limit=6,
    )
    if not questions:
        questions = [
            {"id": f"Q{i}", "question": q, "why_this_unlocks": ""}
            for i, q in enumerate(socratic_output.get("question_set", []), 1)
        ]
    return {
        "hidden_assumptions": curiosity_map.get("hidden_assumptions", []),
        "unexplored_domains": unexplored_domains,
        "questions": questions[:5],
        "branch_seeds": branch_seeds,
    }


def summarize_creativity_trace(
    creativity_research_plan: dict,
    creativity_branch: dict,
    creativity_develop: list[dict],
    creativity_selection: dict,
    creativity_mixing: dict,
    creativity_final_synthesis: dict,
) -> dict:
    develop_by_id = {d.get("branch_id"): d for d in creativity_develop}
    branches = []
    for branch in creativity_branch.get("branches", []):
        developed = develop_by_id.get(branch.get("id"), {})
        candidates = developed.get("branch_outputs", []) or branch.get("examples", [])
        branches.append(
            {
                "id": branch.get("id", ""),
                "frame": branch.get("frame", ""),
                "candidates": candidates[:4],
            }
        )

    pruned = []
    for item in creativity_selection.get("scored_branches", []):
        if item.get("decision") == "prune":
            pruned.append({"id": item.get("branch_id", ""), "reason": item.get("reason", "")})
    for item in creativity_branch.get("pruned_branch_ideas", []):
        pruned.append({"id": item.get("idea", ""), "reason": item.get("reason", "")})

    combinations = [
        {
            "from": h.get("from_branch_ids", []),
            "result": h.get("concept", ""),
            "novelty_note": h.get("why_novel", ""),
        }
        for h in creativity_mixing.get("hybrids", [])
        if h.get("concept")
    ]
    dead_ends = [
        f"{', '.join(d.get('from_branch_ids', []))}: {d.get('reason', '')}".strip(": ")
        for d in creativity_mixing.get("dead_ends", [])
        if d.get("reason")
    ]
    research = _unique(
        creativity_research_plan.get("research_notes", [])
        + creativity_research_plan.get("research_queries", [])
        + creativity_research_plan.get("creative_tensions", []),
        limit=6,
    )
    return {
        "research": research,
        "branches": branches[:6],
        "pruned": [p for p in pruned if p.get("id") or p.get("reason")][:6],
        "combinations": combinations[:5],
        "dead_ends": dead_ends[:5],
        "output": creativity_final_synthesis.get("output", []),
    }


def validate_full_loop(data: dict) -> list[str]:
    errors = []
    if "domain" not in data:
        errors.append("missing 'domain'")
    if "input" not in data:
        errors.append("missing 'input'")
    if "loop" not in data or not data["loop"]:
        errors.append("missing or empty 'loop'")
        return errors
    for idx, iteration in enumerate(data["loop"]):
        prefix = f"loop[{idx}]"
        if "advanced" not in iteration:
            errors.append(f"{prefix} missing 'advanced'")
        if "critic" not in iteration:
            errors.append(f"{prefix} missing 'critic'")
    if "final_output" not in data or not data["final_output"]:
        errors.append("missing or empty 'final_output'")
    return errors
