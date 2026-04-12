# Creativity is not a media generator. That old behavior now lives in generator.py.
# This file implements CreativeGen as a text-first reasoning pipeline:
# research -> branch -> select -> combine -> synthesize.

"""
creativity.py - The CreativeGen module of the psyche.

Creative reasoning pipeline:
1. Research what exists and what is already obvious.
2. Branch into distinct directions of attack.
3. Select the strongest non-convergent branches.
4. Cross-pollinate them into hybrid concepts.
5. Synthesize final ideas and explain why they are novel.
"""

import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any

USE_WEB_RESEARCH = False


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


RESEARCH_PLAN_PROMPT = """You are CreativeGen, the creativity module of a cognitive architecture.
Your role is not to generate media prompts. Your role is to think creatively.

Stage 1 is research and framing.

Return JSON:
{
  "complexity": "low|medium|high",
  "branch_budget": 2-6,
  "research_queries": ["web query 1", "web query 2"],
  "known_patterns": ["what already exists or is overused"],
  "adjacent_domains": ["domain transfer candidate"],
  "creative_tensions": ["important tension or tradeoff"],
  "summary": "short research plan summary"
}

RULES:
- Focus on what is already known, crowded, obvious, or structurally useful.
- Choose branch_budget based on the openness of the task.
- Simple naming or styling tasks can use 2-4 branches. More open conceptual tasks can use 4-6.
- Keep queries short and searchable.
- Prefer 1-2 queries max.
- Surface adjacent domains and tensions, not answers."""


RESEARCH_SYNTHESIS_PROMPT = """You are synthesizing research for a creativity pipeline.
Compress the research into a useful foundation for branching.

Return JSON:
{
  "research": ["key research note"],
  "adjacent_transfers": ["useful domain transfer"],
  "creative_tensions": ["important tension"],
  "summary": "short synthesis summary"
}

RULES:
- Keep only the most useful notes.
- Include what is crowded, stale, or already solved.
- Include what could be remixed from other domains."""


BRANCH_PROMPT = """You are Stage 2 of CreativeGen: branched exploration.
Create multiple distinct directions of attack for the problem.

Return JSON:
{
  "branches": [
    {
      "id": "B1",
      "frame": "distinct framing or direction",
      "domain": "primary domain lens",
      "constraint": "what constraint or rule shapes this branch",
      "examples": ["example candidate or motif"],
      "why_distinct": "why this branch is structurally different"
    }
  ],
  "summary": "short branching summary"
}

RULES:
- Generate exactly branch_budget branches if possible.
- Branches must be deliberately different: domain, metaphor, constraint, or first principle.
- Avoid near duplicates.
- Each branch should be specific enough to build from."""


DEVELOP_BRANCHES_PROMPT = """You are Stage 2.5 of CreativeGen: branch development.
Take each branch and continue its train of thought in steps until the branch stops yielding materially new ideas.

Return JSON:
{
  "branch_chains": [
    {
      "id": "B1",
      "chain_steps": ["step 1 insight", "step 2 insight", "step 3 insight"],
      "branch_outputs": ["candidate idea 1", "candidate idea 2"],
      "exhausted_when": "why this branch has yielded most of what it can"
    }
  ],
  "summary": "short branch development summary"
}

RULES:
- Continue each branch until it feels exhausted, but stay concise.
- Each branch should usually have 2-5 chain steps.
- branch_outputs should be the strongest ideas that emerged from that branch.
- Do not combine branches yet. Exhaust each branch on its own first."""


SELECTION_PROMPT = """You are Stage 3 of CreativeGen: selection and distillation.
Score branches for novelty, relevance, and combinability. Prune convergent ones.

Return JSON:
{
  "selected": [
    {
      "id": "B1",
      "why": "why this branch survives",
      "novelty": 0.0-1.0,
      "relevance": 0.0-1.0,
      "combinability": 0.0-1.0
    }
  ],
  "pruned": [
    {
      "id": "B2",
      "reason": "why this branch is redundant, weak, or convergent"
    }
  ],
  "summary": "short selection summary"
}

RULES:
- Keep 2-3 branches.
- Prune branches that converge toward the same idea space.
- Reward structural distance, not just novelty theater."""


COMBINATION_PROMPT = """You are Stage 4 of CreativeGen: combinatory association.
Cross-pollinate the selected branches and produce hybrid concepts.

Return JSON:
{
  "combinations": [
    {
      "id": "C1",
      "from": ["B1", "B3"],
      "result": "hybrid concept",
      "why_novel": "why the mix creates something new",
      "strength": 0.0-1.0
    }
  ],
  "dead_ends": ["mix that failed and why"],
  "summary": "short combination summary"
}

RULES:
- Prefer mixes that neither branch would reach alone.
- Include 2-3 combinations.
- Call out failed or shallow combinations as dead ends."""


FINAL_SYNTHESIS_PROMPT = """You are Stage 5 of CreativeGen: final synthesis.
Turn the research, branches, selection, and combinations into the final creative output.

Return JSON:
{
  "primary_candidates": ["plain names/titles/labels only, no explanation"],
  "output": ["final idea 1", "final idea 2"],
  "novelty_notes": ["why the strongest outputs feel genuinely new"],
  "best_combination": "which combination unlocked the strongest outcome",
  "summary": "short final creativity summary"
}

RULES:
- Final outputs must feel downstream of the full process.
- If the task is naming/labeling/titling, include 2-6 plain candidate names in primary_candidates.
- Keep final output to 1-2 strongest candidates.
- Keep novelty notes to 1-2 points.
- Prefer ideas unreachable by a single branch alone.
- Explain novelty briefly and concretely."""


def _clip(text: Any, limit: int = 220) -> str:
    return str(text or "").strip()


class Creativity:
    """CreativeGen: multi-stage text creativity engine."""

    def __init__(self, agent, tools=None, verbose: bool = True):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose

    def process(self, input_context: str, steering: str = "") -> Dict[str, Any]:
        research_plan = self._plan_research(input_context, steering)
        self._log_stage("research_plan", f"queries={len(research_plan.get('research_queries', []))}", "grey")
        web_notes = self._run_web_research(research_plan.get("research_queries", []))
        self._log_stage("research", f"notes={len(web_notes)}", "grey")
        research = self._synthesise_research(input_context, research_plan, web_notes, steering)
        self._log_stage("synthesis", f"research={len(research.get('research', []))}", "blue")
        branches = self._build_branches(input_context, research, steering)
        self._log_stage("branches", f"branches={len(branches.get('branches', []))}", "cyan")
        branch_development = self._develop_branches(input_context, research, branches, steering)
        self._log_stage("branch_chains", f"chains={len(branch_development.get('branch_chains', []))}", "blue")
        selection = self._select_branches(input_context, research, branches, branch_development, steering)
        self._log_stage("selection", f"selected={len(selection.get('selected', []))} pruned={len(selection.get('pruned', []))}", "green")
        combinations = self._combine_branches(input_context, research, branches, branch_development, selection, steering)
        self._log_stage("combinations", f"combos={len(combinations.get('combinations', []))}", "magenta")
        final = self._final_synthesis(input_context, research, branches, branch_development, selection, combinations, steering)
        self._log_stage("final", f"outputs={len(final.get('output', []))}", "green")
        primary_candidates = self._derive_primary_candidates(final)

        reasoning_chain = {
            "research_plan": research_plan,
            "web_notes": web_notes,
            "primary_candidates": primary_candidates,
            "research": research.get("research", []),
            "adjacent_transfers": research.get("adjacent_transfers", []),
            "creative_tensions": research.get("creative_tensions", []),
            "branches": branches.get("branches", []),
            "branch_chains": branch_development.get("branch_chains", []),
            "selected": selection.get("selected", []),
            "pruned": selection.get("pruned", []),
            "combinations": combinations.get("combinations", []),
            "dead_ends": combinations.get("dead_ends", []),
            "output": final.get("output", []),
            "novelty_notes": final.get("novelty_notes", []),
            "best_combination": final.get("best_combination", ""),
        }

        strengths = [
            PsycheProtocol.coerce_float(item.get("strength"), default=0.0)
            for item in combinations.get("combinations", [])
        ]
        creativity_priority = round(
            min(1.0, 0.62 + (max(strengths) if strengths else 0.18) * 0.3),
            3,
        )

        return PsycheProtocol.normalise_agent_output(
            "creativity",
            {
                "summary": final.get("summary") or research.get("summary") or "Creative synthesis complete.",
                "content": self._format_reasoning_chain(reasoning_chain),
                "signals": {
                    "confidence": 0.68 + min(creativity_priority * 0.2, 0.2),
                    "priority": creativity_priority,
                },
                "next_hints": primary_candidates[:4] or list(final.get("output", [])[:2]),
            },
            default_priority=creativity_priority,
            default_confidence=0.72,
        )

    def _log_stage(self, stage: str, message: str, color: str):
        if self.verbose:
            printColoured(f"  [creativity::{stage}] {message}", color)

    def _plan_research(self, input_context: str, steering: str) -> Dict[str, Any]:
        system = RESEARCH_PLAN_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            "Plan the research needed before creative branching."
        )
        result = self._structured_call(prompt, system, temperature=0.4, max_tokens=500)
        if result:
            return result
        return {
            "complexity": "medium",
            "branch_budget": 4,
            "research_queries": [],
            "known_patterns": ["The obvious solution space is likely too crowded."],
            "adjacent_domains": ["Look for transferable structure in distant domains."],
            "creative_tensions": ["Balance novelty with relevance."],
            "summary": "Fallback research plan.",
        }

    def _run_web_research(self, queries: List[str]) -> List[str]:
        if not USE_WEB_RESEARCH:
            return []
        if not self.tools or not hasattr(self.tools, "web_crawl"):
            return []
        notes: List[str] = []
        for query in [str(q).strip() for q in queries[:2] if str(q).strip()]:
            try:
                payload = self.tools.web_crawl(
                    query=query,
                    sources="all",
                    num_results=2,
                    include_wiki_content=False,
                    exa_max_chars=350,
                )
                notes.extend(self._extract_research_notes(query, payload))
            except Exception as e:
                notes.append(f"{query}: research error: {e}")
        return notes[:6]

    def _synthesise_research(
        self,
        input_context: str,
        research_plan: Dict[str, Any],
        web_notes: List[str],
        steering: str,
    ) -> Dict[str, Any]:
        system = RESEARCH_SYNTHESIS_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research plan:\n{json.dumps(research_plan, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Web research notes:\n{json.dumps(web_notes, indent=2, ensure_ascii=True, default=str)}\n\n"
            "Compress this into the strongest research base for branching."
        )
        result = self._structured_call(prompt, system, temperature=0.35, max_tokens=650)
        if result:
            return result
        fallback_notes = research_plan.get("known_patterns", []) + web_notes[:4]
        return {
            "research": fallback_notes[:6] or ["The space is likely crowded with obvious defaults."],
            "adjacent_transfers": research_plan.get("adjacent_domains", [])[:4],
            "creative_tensions": research_plan.get("creative_tensions", [])[:4],
            "summary": "Fallback research synthesis.",
        }

    def _build_branches(self, input_context: str, research: Dict[str, Any], steering: str) -> Dict[str, Any]:
        system = BRANCH_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research foundation:\n{json.dumps(research, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Target branch budget: {research.get('branch_budget') or 4}\n\n"
            "Generate structurally distinct creative branches."
        )
        result = self._structured_call(prompt, system, temperature=0.6, max_tokens=900)
        if result:
            return result
        return {
            "branches": [
                {
                    "id": "B1",
                    "frame": "Reframe the problem through a distant domain analogy.",
                    "domain": "cross-domain",
                    "constraint": "Must stay relevant to the original need.",
                    "examples": [],
                    "why_distinct": "Transfers structure instead of repeating the obvious framing.",
                },
                {
                    "id": "B2",
                    "frame": "Push against the most common assumption in the space.",
                    "domain": "contrarian",
                    "constraint": "Reject the default market or style trope.",
                    "examples": [],
                    "why_distinct": "Starts from inversion rather than convention.",
                },
                {
                    "id": "B3",
                    "frame": "Anchor the solution in one strong material, biological, or physical metaphor.",
                    "domain": "metaphor",
                    "constraint": "The metaphor must carry structure, not just aesthetics.",
                    "examples": [],
                    "why_distinct": "Creates a coherent world model for idea generation.",
                },
            ],
            "summary": "Fallback branch set.",
        }

    def _develop_branches(self, input_context: str, research: Dict[str, Any], branches: Dict[str, Any], steering: str) -> Dict[str, Any]:
        system = DEVELOP_BRANCHES_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research:\n{json.dumps(research, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branches:\n{json.dumps(branches, indent=2, ensure_ascii=True, default=str)}\n\n"
            "Develop each branch until its idea-space feels meaningfully exhausted."
        )
        result = self._structured_call(prompt, system, temperature=0.62, max_tokens=1200)
        if result:
            return result
        fallback_chains = []
        for branch in branches.get("branches", [])[:4]:
            fallback_chains.append(
                {
                    "id": branch.get("id", ""),
                    "chain_steps": [
                        branch.get("frame", ""),
                        branch.get("why_distinct", "Push the branch until it yields useful candidates."),
                    ],
                    "branch_outputs": branch.get("examples", [])[:2],
                    "exhausted_when": "The branch starts repeating the same framing with no new leverage.",
                }
            )
        return {"branch_chains": fallback_chains, "summary": "Fallback branch-chain development."}

    def _select_branches(
        self,
        input_context: str,
        research: Dict[str, Any],
        branches: Dict[str, Any],
        branch_development: Dict[str, Any],
        steering: str,
    ) -> Dict[str, Any]:
        system = SELECTION_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research:\n{json.dumps(research, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branches:\n{json.dumps(branches, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branch chains:\n{json.dumps(branch_development, indent=2, ensure_ascii=True, default=str)}\n\n"
            "Select the strongest non-convergent branches."
        )
        result = self._structured_call(prompt, system, temperature=0.4, max_tokens=700)
        if result:
            return result
        fallback_selected = []
        for branch in branches.get("branches", [])[:3]:
            fallback_selected.append(
                {
                    "id": branch.get("id", ""),
                    "why": branch.get("why_distinct", "Useful structural survivor."),
                    "novelty": 0.72,
                    "relevance": 0.74,
                    "combinability": 0.76,
                }
            )
        return {"selected": fallback_selected, "pruned": [], "summary": "Fallback selection."}

    def _combine_branches(
        self,
        input_context: str,
        research: Dict[str, Any],
        branches: Dict[str, Any],
        branch_development: Dict[str, Any],
        selection: Dict[str, Any],
        steering: str,
    ) -> Dict[str, Any]:
        system = COMBINATION_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research:\n{json.dumps(research, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"All branches:\n{json.dumps(branches, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branch chains:\n{json.dumps(branch_development, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Selected branches:\n{json.dumps(selection, indent=2, ensure_ascii=True, default=str)}\n\n"
            "Generate strong hybrid combinations."
        )
        result = self._structured_call(prompt, system, temperature=0.62, max_tokens=850)
        if result:
            return result
        selected_ids = [item.get("id", "") for item in selection.get("selected", [])[:2] if item.get("id")]
        if len(selected_ids) >= 2:
            return {
                "combinations": [
                    {
                        "id": "C1",
                        "from": selected_ids[:2],
                        "result": "Hybrid concept from the two strongest surviving frames.",
                        "why_novel": "It blends distant structures instead of choosing one lane.",
                        "strength": 0.78,
                    }
                ],
                "dead_ends": [],
                "summary": "Fallback combination stage.",
            }
        return {"combinations": [], "dead_ends": [], "summary": "No combinations available."}

    def _final_synthesis(
        self,
        input_context: str,
        research: Dict[str, Any],
        branches: Dict[str, Any],
        branch_development: Dict[str, Any],
        selection: Dict[str, Any],
        combinations: Dict[str, Any],
        steering: str,
    ) -> Dict[str, Any]:
        system = FINAL_SYNTHESIS_PROMPT
        if steering:
            system += f"\n\nInternal context:\n{steering}"
        prompt = (
            f"Input:\n{input_context}\n\n"
            f"Research:\n{json.dumps(research, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branches:\n{json.dumps(branches, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Branch chains:\n{json.dumps(branch_development, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Selection:\n{json.dumps(selection, indent=2, ensure_ascii=True, default=str)}\n\n"
            f"Combinations:\n{json.dumps(combinations, indent=2, ensure_ascii=True, default=str)}\n\n"
            "Synthesize the final creative output."
        )
        result = self._structured_call(prompt, system, temperature=0.55, max_tokens=750)
        if result:
            return result
        fallback_outputs = [item.get("result", "") for item in combinations.get("combinations", [])[:3] if item.get("result")]
        return {
            "primary_candidates": [],
            "output": fallback_outputs or ["Creative output could not be fully synthesized."],
            "novelty_notes": ["The strongest ideas come from combining distant frames rather than following one obvious lane."],
            "best_combination": combinations.get("combinations", [{}])[0].get("id", ""),
            "summary": "Fallback final synthesis.",
        }

    def _derive_primary_candidates(self, final: Dict[str, Any]) -> List[str]:
        explicit = [str(item).strip() for item in (final.get("primary_candidates") or []) if str(item).strip()]
        if explicit:
            return explicit[:6]

        derived: List[str] = []
        for item in final.get("output", []) or []:
            text = str(item or "").strip()
            if not text:
                continue
            candidate = text
            for sep in (" — ", " - ", ": "):
                if sep in candidate:
                    candidate = candidate.split(sep, 1)[0].strip()
                    break
            if candidate and candidate not in derived:
                derived.append(candidate)
        return derived[:6]

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
                printColoured(f"  [creativity] structured call error: {e}", "red")
            return {}

    def _extract_research_notes(self, query: str, payload: Any) -> List[str]:
        if not isinstance(payload, dict):
            return [f"{query}: {_clip(payload, 180)}"]

        collected: List[str] = []

        def walk(value: Any):
            if len(collected) >= 4:
                return
            if isinstance(value, dict):
                for key in ("title", "summary", "snippet", "content", "text", "url"):
                    if key in value and isinstance(value[key], str) and value[key].strip():
                        collected.append(_clip(value[key], 180))
                        if len(collected) >= 4:
                            return
                for nested in value.values():
                    if len(collected) >= 4:
                        return
                    walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    if len(collected) >= 4:
                        return
                    walk(nested)
            elif isinstance(value, str) and value.strip():
                collected.append(_clip(value, 180))

        walk(payload)
        return [f"{query}: {note}" for note in collected[:4]]

    def _format_reasoning_chain(self, chain: Dict[str, Any]) -> str:
        parts = []

        if chain.get("primary_candidates"):
            parts.append("Primary candidates:\n- " + "\n- ".join(_clip(item, 120) for item in chain["primary_candidates"][:6]))

        if chain.get("output"):
            parts.append("Final output:\n- " + "\n- ".join(_clip(item, 160) for item in chain["output"][:2]))

        if chain.get("novelty_notes"):
            parts.append("Novelty notes:\n- " + "\n- ".join(_clip(item, 160) for item in chain["novelty_notes"][:2]))

        if chain.get("best_combination"):
            parts.append("Best combination:\n" + str(chain["best_combination"]))

        if chain.get("research"):
            parts.append("Research:\n- " + "\n- ".join(_clip(item, 160) for item in chain["research"][:4]))
        if chain.get("adjacent_transfers"):
            parts.append("Adjacent transfers:\n- " + "\n- ".join(_clip(item, 160) for item in chain["adjacent_transfers"][:3]))
        if chain.get("creative_tensions"):
            parts.append("Creative tensions:\n- " + "\n- ".join(_clip(item, 160) for item in chain["creative_tensions"][:3]))

        if chain.get("branches"):
            branch_lines = []
            for branch in chain["branches"][:4]:
                examples = ", ".join(_clip(ex, 60) for ex in branch.get("examples", [])[:2])
                branch_lines.append(
                    f"- {branch.get('id', '?')} | {branch.get('frame', '')} | domain={branch.get('domain', '')} | examples={examples}"
                )
            parts.append("Branches:\n" + "\n".join(branch_lines))

        if chain.get("branch_chains"):
            chain_lines = []
            for item in chain["branch_chains"][:4]:
                steps = " -> ".join(_clip(step, 70) for step in item.get("chain_steps", [])[:3])
                outputs = ", ".join(_clip(output, 60) for output in item.get("branch_outputs", [])[:2])
                chain_lines.append(
                    f"- {item.get('id', '?')} | steps={steps} | outputs={outputs}"
                )
            parts.append("Branch chains:\n" + "\n".join(chain_lines))

        if chain.get("selected"):
            selected_lines = []
            for item in chain["selected"][:3]:
                selected_lines.append(
                    f"- {item.get('id', '?')} | n={item.get('novelty', '?')} r={item.get('relevance', '?')} c={item.get('combinability', '?')} | {_clip(item.get('why', ''), 120)}"
                )
            parts.append("Selected branches:\n" + "\n".join(selected_lines))

        if chain.get("pruned"):
            parts.append(
                "Pruned:\n- " + "\n- ".join(
                    f"{item.get('id', '?')}: {_clip(item.get('reason', ''), 120)}" for item in chain["pruned"][:3]
                )
            )

        if chain.get("combinations"):
            combo_lines = []
            for combo in chain["combinations"][:3]:
                combo_lines.append(
                    f"- {combo.get('id', '?')} from {combo.get('from', [])}: {_clip(combo.get('result', ''), 140)} | {_clip(combo.get('why_novel', ''), 110)}"
                )
            parts.append("Combinations:\n" + "\n".join(combo_lines))

        if chain.get("dead_ends"):
            parts.append("Dead ends:\n- " + "\n- ".join(_clip(item, 150) for item in chain["dead_ends"][:2]))

        if chain.get("web_notes"):
            parts.append("Research traces:\n- " + "\n- ".join(_clip(item, 150) for item in chain["web_notes"][:3]))

        return "\n\n".join(parts)


if __name__ == "__main__":
    runner = Utils.import_file("standalone_tests.py")
    runner.run_module_cli("creativity")
