# Curiosity should not depend only on the psyche's personal history.
# It should also reason from broad world knowledge: assumptions, frontiers,
# opposites, expert lenses, and neglected domains.

"""
curiosity.py - The Socratic curiosity engine of the psyche.

Three-stage pipeline:
1. Map curiosity domains from broad known context plus local novelty.
2. Expand each domain into branching question threads.
3. Distill the strongest questions into a final Socratic scaffold.
"""

import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import numpy as np
except ImportError:
    np = None


def _import_utils():
    try:
        from src.VI_utils.utils import Utils
        return Utils
    except ImportError:
        pass
    current = Path(__file__).resolve().parent
    while not (current / "main.py").exists() and not (current / ".git").exists():
        if current.parent == current:
            break
        current = current.parent
    utils_path = current / "src" / "VI_utils" / "utils.py"
    if utils_path.exists():
        spec = importlib.util.spec_from_file_location("utils", str(utils_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["utils"] = mod
            spec.loader.exec_module(mod)
            return mod.Utils
    raise ImportError("Could not find Utils")


Utils = _import_utils()
printColoured = Utils.printColoured
protocol_module = Utils.import_file("protocol.py")
PsycheProtocol = protocol_module.PsycheProtocol


CURIOSITY_MAP_PROMPT = """You are SocraticGen, the curiosity module of a cognitive architecture.
Your role is to open branches, not to close them with answers.

Reason from two lenses at once:
1. The psyche's local novelty score.
2. Broad world knowledge: what humanity already knows, assumes, ignores, or has not connected well yet.

Return JSON:
{
  "global_novelty_estimate": 0.0-1.0,
  "branch_budget": 2-5,
  "known_context": ["what the broader world likely already assumes or knows here"],
  "hidden_assumptions": ["assumption worth challenging"],
  "curiosity_domains": [
    {
      "id": "D1",
      "label": "domain or frontier area",
      "lens": "assumption|opposite|expert|frontier|cross-domain",
      "why_curious": "why this opens useful unknowns",
      "priority": 0.0-1.0
    }
  ],
  "seed_questions": ["high level seed question"],
  "frontier_notes": ["where the edge of the broader map seems to be"],
  "summary": "short map summary"
}

RULES:
- Focus on generative unknowns, not answers.
- Use the broader known world, not just the current conversation.
- Choose branch_budget based on how open-ended the problem is.
- Small concrete tasks can use 2-3 branches. Open-ended tasks can use 4-5.
- Prefer diverse curiosity domains.
- Reward domains that challenge default framing.
- Be concise but specific."""


CURIOSITY_BRANCH_PROMPT = """You are expanding curiosity into branching question threads.
Each branch is a direction of inquiry, not a solution.

Return JSON:
{
  "branches": [
    {
      "branch_id": "B1",
      "domain_id": "D1",
      "direction": "what this branch investigates",
      "questions": ["q1", "q2", "q3"],
      "why_non_obvious": "why this branch goes beyond the expected",
      "curiosity_strength": 0.0-1.0
    }
  ],
  "pruned": [
    {
      "domain_id": "D2",
      "reason": "why this direction is too obvious, too narrow, or redundant"
    }
  ],
  "summary": "short branching summary"
}

RULES:
- Generate branch_budget branches if possible, but fewer is allowed if the space is narrow.
- Each surviving branch needs 2-3 strong questions.
- Questions should reveal assumptions, unexplored variables, opposites, hidden actors, or second-order effects.
- Prune redundant or shallow branches.
- Keep the branches diverse."""


CURIOSITY_DISTILL_PROMPT = """You are distilling multiple curiosity branches into the strongest final question set.
Do not answer the problem. Produce the best questions that should guide the next stage.

Return JSON:
{
  "best_questions": [
    {
      "question": "question text",
      "branch_id": "B1",
      "why": "why this question matters",
      "leverage": 0.0-1.0
    }
  ],
  "socratic_scaffold": "short scaffold explaining how to think about the problem through questions",
  "exploration_direction": "where the psyche should explore next",
  "summary": "short final curiosity summary"
}

RULES:
- Pick 3-4 highest-leverage questions only.
- Favour questions that unlock multiple branches at once.
- The scaffold should guide thought, not conclude it.
- Keep it concise and useful."""


def _clip(text: Any, limit: int = 220) -> str:
    return str(text or "").strip()


class Curiosity:
    """Socratic curiosity engine with collective and local novelty signals."""

    def __init__(self, agent, embed_fn, tools=None, verbose: bool = True):
        self.agent = agent
        self.embed = embed_fn
        self.tools = tools
        self.verbose = verbose
        self._known_vectors: List[Any] = []
        self._known_texts: List[str] = []

    def process(self, input_context: str, steering: str = "") -> Dict[str, Any]:
        local_novelty = self._calculate_local_novelty(input_context)
        self._log_stage("novelty", f"local={local_novelty:.2f}", "yellow")
        mapped = self._map_curiosity_domains(input_context, local_novelty, steering)
        self._log_stage("map", f"domains={len(mapped.get('curiosity_domains', []))}", "cyan")
        branches = self._expand_question_branches(input_context, mapped, local_novelty, steering)
        self._log_stage("branches", f"branches={len(branches.get('branches', []))}", "yellow")
        distilled = self._distill_question_set(input_context, mapped, branches, local_novelty, steering)
        self._log_stage("distill", f"best_questions={len(distilled.get('best_questions', []))}", "green")
        self._store_in_territory(input_context)

        global_novelty = PsycheProtocol.coerce_float(
            mapped.get("global_novelty_estimate"),
            default=0.55,
        )
        curiosity_priority = round(min(1.0, max(global_novelty, 0.35 + local_novelty * 0.5)), 3)

        reasoning_chain = {
            "local_novelty": local_novelty,
            "global_novelty_estimate": global_novelty,
            "branch_budget": mapped.get("branch_budget", 0),
            "known_context": mapped.get("known_context", []),
            "hidden_assumptions": mapped.get("hidden_assumptions", []),
            "curiosity_domains": mapped.get("curiosity_domains", []),
            "seed_questions": mapped.get("seed_questions", []),
            "branches": branches.get("branches", []),
            "pruned": branches.get("pruned", []),
            "best_questions": distilled.get("best_questions", []),
            "socratic_scaffold": distilled.get("socratic_scaffold", ""),
            "exploration_direction": distilled.get("exploration_direction", ""),
        }

        self._log_stage(
            "final",
            f"local={local_novelty:.2f} global={global_novelty:.2f} branches={len(branches.get('branches', []))}",
            "green",
        )

        return PsycheProtocol.normalise_agent_output(
            "curiosity",
            {
                "summary": distilled.get("summary") or mapped.get("summary") or f"Curiosity priority {curiosity_priority:.2f}",
                "content": self._format_reasoning_chain(reasoning_chain),
                "signals": {
                    "confidence": 0.65 + min(curiosity_priority * 0.25, 0.25),
                    "priority": curiosity_priority,
                },
                "next_hints": [
                    item.get("question", "")
                    for item in distilled.get("best_questions", [])[:2]
                    if item.get("question")
                ] + ([distilled.get("exploration_direction", "")] if distilled.get("exploration_direction") else []),
            },
            default_priority=curiosity_priority,
            default_confidence=0.7,
        )

    def _log_stage(self, stage: str, message: str, color: str):
        if self.verbose:
            printColoured(f"  [curiosity::{stage}] {message}", color)

    def _calculate_local_novelty(self, text: str) -> float:
        if np is None or not self._known_vectors:
            return 0.45
        try:
            input_vec = self.embed(text)
            if isinstance(input_vec, np.ndarray):
                input_vec = input_vec.flatten()
                norm = np.linalg.norm(input_vec)
                if norm > 0:
                    input_vec = input_vec / norm
            similarities = [float(np.dot(input_vec, known_vec)) for known_vec in self._known_vectors]
            novelty = 1.0 - max(similarities)
            return round(max(0.0, min(1.0, novelty)), 3)
        except Exception as e:
            if self.verbose:
                printColoured(f"  [curiosity] local novelty error: {e}", "yellow")
            return 0.45

    def _store_in_territory(self, text: str):
        if np is None:
            return
        try:
            vec = self.embed(text)
            if isinstance(vec, np.ndarray):
                vec = vec.flatten()
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._known_vectors.append(vec)
                self._known_texts.append(text)
                if len(self._known_vectors) > 500:
                    self._known_vectors = self._known_vectors[-500:]
                    self._known_texts = self._known_texts[-500:]
        except Exception:
            pass

    def _map_curiosity_domains(self, input_context: str, local_novelty: float, steering: str) -> Dict[str, Any]:
        system = CURIOSITY_MAP_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Local novelty score inside this psyche: {local_novelty:.2f}\n"
            "Map the edge of the broader known world around this topic.\n"
            "What is assumed, what is ignored, and what deserves deeper questioning?"
        )
        result = self._structured_call(prompt, system, temperature=0.65, max_tokens=700)
        if result:
            return result
        return {
            "global_novelty_estimate": max(0.55, local_novelty),
            "branch_budget": 3,
            "known_context": ["The topic likely contains default framing that should be questioned."],
            "hidden_assumptions": ["The obvious framing may be too narrow."],
            "curiosity_domains": [
                {
                    "id": "D1",
                    "label": "hidden assumptions",
                    "lens": "assumption",
                    "why_curious": "Challenging the base framing often opens better branches.",
                    "priority": 0.8,
                },
                {
                    "id": "D2",
                    "label": "cross-domain transfer",
                    "lens": "cross-domain",
                    "why_curious": "Analogies from distant fields can reveal non-obvious directions.",
                    "priority": 0.76,
                },
                {
                    "id": "D3",
                    "label": "future second-order effects",
                    "lens": "frontier",
                    "why_curious": "The hidden consequences may matter more than the surface question.",
                    "priority": 0.72,
                },
            ],
            "seed_questions": [
                "What assumption is being treated as fixed that may not be fixed?",
                "What domain outside the obvious one would frame this differently?",
                "What would the long-term consequence be if the current framing is wrong?",
            ],
            "frontier_notes": ["The edge of the map is likely in the neglected assumptions and adjacent domains."],
            "summary": "Mapped three broad curiosity fronts.",
        }

    def _expand_question_branches(
        self,
        input_context: str,
        mapped: Dict[str, Any],
        local_novelty: float,
        steering: str,
    ) -> Dict[str, Any]:
        system = CURIOSITY_BRANCH_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Curiosity map:\n{json.dumps(mapped, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Target branch budget: {mapped.get('branch_budget', 3)}\n"
            f"Local novelty score: {local_novelty:.2f}\n"
            "Expand this into branching Socratic question threads."
        )
        result = self._structured_call(prompt, system, temperature=0.7, max_tokens=800)
        if result:
            return result
        seed_questions = mapped.get("seed_questions", [])[:3]
        branch_budget = max(2, min(5, int(PsycheProtocol.coerce_float(mapped.get("branch_budget"), default=3))))
        domains = mapped.get("curiosity_domains", [])[:branch_budget]
        branches = []
        for idx, domain in enumerate(domains, 1):
            branches.append(
                {
                    "branch_id": f"B{idx}",
                    "domain_id": domain.get("id", f"D{idx}"),
                    "direction": domain.get("label", f"branch {idx}"),
                    "questions": seed_questions[idx - 1: idx + 2] or seed_questions,
                    "why_non_obvious": domain.get("why_curious", ""),
                    "curiosity_strength": domain.get("priority", 0.7),
                }
            )
        return {"branches": branches, "pruned": [], "summary": "Expanded curiosity branches."}

    def _distill_question_set(
        self,
        input_context: str,
        mapped: Dict[str, Any],
        branches: Dict[str, Any],
        local_novelty: float,
        steering: str,
    ) -> Dict[str, Any]:
        system = CURIOSITY_DISTILL_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Curiosity map:\n{json.dumps(mapped, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branches:\n{json.dumps(branches, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Local novelty score: {local_novelty:.2f}\n"
            "Distill the strongest final question set."
        )
        result = self._structured_call(prompt, system, temperature=0.6, max_tokens=650)
        if result:
            return result
        fallback_questions = []
        for branch in branches.get("branches", [])[:4]:
            for question in branch.get("questions", [])[:1]:
                fallback_questions.append(
                    {
                        "question": question,
                        "branch_id": branch.get("branch_id", ""),
                        "why": branch.get("why_non_obvious", "It opens a useful line of inquiry."),
                        "leverage": branch.get("curiosity_strength", 0.7),
                    }
                )
        return {
            "best_questions": fallback_questions[:4],
            "socratic_scaffold": "Start by challenging assumptions, then compare distant lenses, then probe second-order effects.",
            "exploration_direction": "Follow the question that unlocks the most adjacent unknowns.",
            "summary": "Distilled a concise Socratic question set.",
        }

    def _structured_call(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        try:
            result = self.agent.structured_output(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                store_interaction=False,
            )
            return result if isinstance(result, dict) else {}
        except Exception as e:
            if self.verbose:
                printColoured(f"  [curiosity] structured call error: {e}", "yellow")
            return {}

    def _format_reasoning_chain(self, chain: Dict[str, Any]) -> str:
        parts = [
            f"Local novelty: {chain.get('local_novelty', 0.0):.2f}",
            f"Global novelty: {chain.get('global_novelty_estimate', 0.0):.2f}",
        ]

        if chain.get("branch_budget"):
            parts.append(f"Branch budget: {chain.get('branch_budget')}")

        known_context = chain.get("known_context", [])
        if known_context:
            parts.append("Known context:\n- " + "\n- ".join(_clip(item, 180) for item in known_context[:4]))

        assumptions = chain.get("hidden_assumptions", [])
        if assumptions:
            parts.append("Hidden assumptions:\n- " + "\n- ".join(_clip(item, 180) for item in assumptions[:4]))

        domains = chain.get("curiosity_domains", [])
        if domains:
            domain_lines = []
            for domain in domains[:4]:
                domain_lines.append(
                    f"- {domain.get('id', '?')} | {domain.get('label', 'domain')} | {domain.get('lens', 'lens')}"
                )
            parts.append("Curiosity domains:\n" + "\n".join(domain_lines))

        branches = chain.get("branches", [])
        if branches:
            branch_lines = []
            for branch in branches[:4]:
                questions = "; ".join(_clip(q, 110) for q in branch.get("questions", [])[:2])
                branch_lines.append(
                    f"- {branch.get('branch_id', '?')} -> {branch.get('direction', '')} | questions: {questions}"
                )
            parts.append("Question branches:\n" + "\n".join(branch_lines))

        best_questions = chain.get("best_questions", [])
        if best_questions:
            parts.append(
                "Best questions:\n- " + "\n- ".join(
                    f"{item.get('question', '')}"
                    for item in best_questions[:4]
                    if item.get("question")
                )
            )

        if chain.get("socratic_scaffold"):
            parts.append("Socratic scaffold:\n" + str(chain["socratic_scaffold"]))
        if chain.get("exploration_direction"):
            parts.append("Explore next:\n" + str(chain["exploration_direction"]))

        return "\n\n".join(parts)


if __name__ == "__main__":
    runner = Utils.import_file("standalone_tests.py")
    runner.run_module_cli("curiosity")
