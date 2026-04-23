"""
llm_judge.py - Blind LLM-as-judge scoring of 3-tier evaluation outputs.

Reads a saved data/output/eval_*.json, randomizes labels (so the judge does
not know which tier produced what), asks a stronger model to score each
response on novelty / non-obviousness / process visibility, and aggregates.

Default judge: local Ollama model (works offline, no API key).
Optional: any callable judge function via --use openai/etc (extendable).

Run:
    python src/IV_inference/llm_judge.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; MAGENTA = "\033[95m"
RED = "\033[91m"; GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"


JUDGE_SYSTEM_PROMPT = (
    "You are a strict, calibrated creativity judge. You score AI responses to a creative prompt "
    "on three axes from 1 to 10. You penalize generic LLM tropes (long bullet lists with no real "
    "structural innovation, emoji-heavy headers, hedged language). You reward genuine non-obvious "
    "framings, structural distance from default LLM output, and visible reasoning steps. "
    "Return ONLY a single JSON object. No explanation, no markdown."
)

JUDGE_USER_TEMPLATE = """Prompt the user gave:
---
{prompt}
---

Candidate response (label = {label}):
---
{response}
---

Score on this rubric. JSON output only.

{{
  "novelty": int 1-10,
  "non_obviousness": int 1-10,
  "process_visibility": int 1-10,
  "generic_tropes": int 1-10,
  "one_line_verdict": "string under 25 words"
}}

Definitions:
- novelty: how genuinely new the central idea feels
- non_obviousness: would a generic LLM produce something structurally similar by default?
- process_visibility: does the response show its reasoning steps (Curiosity / Branches / Critic etc.)?
- generic_tropes: 1=heavy LLM tropes (bullets, emojis, hedging); 10=clean and original
- one_line_verdict: <25 words, no flattery
"""


def _load_eval(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_eval_file() -> Path:
    out = REPO_ROOT / "data" / "output"
    files = sorted(out.glob("eval_*.json"))
    if not files:
        raise SystemExit("No eval_*.json files found in data/output/")
    print(f"\n{YELLOW}Pick eval file to judge:{RESET}")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f.name}")
    raw = input(f"\n  Choice [{len(files)}]: ").strip() or str(len(files))
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(files) else len(files) - 1
    return files[idx]


def _make_judge():
    """
    Return a judge callable: judge(system, user) -> str.
    Tries Ollama first, falls back to plain HF transformers.
    """
    try:
        from src.IV_inference.ollama_integration import OllamaGemma4
        print(f"{GREY}Using Ollama as judge backend.{RESET}")
        print(f"{YELLOW}Pick local Ollama model alias (e.g. e4b, 31b) or full tag:{RESET}")
        raw = input(f"  Tag [gemma4:e4b]: ").strip() or "gemma4:e4b"
        g = OllamaGemma4(raw, system=JUDGE_SYSTEM_PROMPT, temperature=0.0, use_memory=False)
        return g.generate_fn()
    except Exception as e:
        print(f"{YELLOW}Ollama not usable ({e}). Falling back to HF Gemma4.{RESET}")
        from src.IV_inference.gemma4_integration import Gemma4
        g = Gemma4("e4b", system=JUDGE_SYSTEM_PROMPT, temperature=0.1, top_k=1, use_memory=False)
        return g.generate_fn()


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _score_one(judge_fn, prompt: str, response: str, label: str) -> dict:
    user_msg = JUDGE_USER_TEMPLATE.format(prompt=prompt[:2000], label=label, response=response[:6000])
    raw = judge_fn(JUDGE_SYSTEM_PROMPT, user_msg)
    parsed = _extract_json(raw) or {}
    parsed["raw"] = raw[:500]
    return parsed


def main():
    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"  {BOLD}LLM Judge - blind scoring of 3-tier eval outputs{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")

    eval_path = _pick_eval_file()
    rows = _load_eval(eval_path)
    print(f"  Loaded {len(rows)} prompts from {eval_path.name}")

    judge = _make_judge()
    rng = random.Random(42)

    aggregates = {"tier1_vanilla": [], "tier2_scaffolded": [], "tier3_tuned": []}
    detailed = []

    for i, row in enumerate(rows, 1):
        prompt = row.get("prompt", "")
        candidates = [
            ("tier1_vanilla", row.get("tier1_vanilla", "")),
            ("tier2_scaffolded", row.get("tier2_scaffolded", "")),
            ("tier3_tuned", row.get("tier3_tuned", "")),
        ]
        # Blind labels: A/B/C in random order
        order = candidates[:]
        rng.shuffle(order)
        labels = ["A", "B", "C"]
        mapping = {labels[k]: order[k][0] for k in range(len(order))}

        print(f"\n{BOLD}Prompt {i}/{len(rows)}{RESET}: {prompt[:80]}...")
        per_prompt = {"prompt": prompt, "blind_mapping": mapping, "scores": {}}
        for k, (orig_tier, text) in enumerate(order):
            if not text or "(not available)" in text:
                continue
            blind = labels[k]
            print(f"  {GREY}scoring blind {blind}...{RESET}")
            score = _score_one(judge, prompt, text, blind)
            per_prompt["scores"][blind] = {"original_tier": orig_tier, **score}
            for axis in ("novelty", "non_obviousness", "process_visibility", "generic_tropes"):
                v = score.get(axis)
                if isinstance(v, (int, float)):
                    aggregates[orig_tier].append((axis, v))
        detailed.append(per_prompt)

    # Aggregate
    print(f"\n{BOLD}{YELLOW}=== Aggregate scores ==={RESET}")
    summary = {}
    for tier, pairs in aggregates.items():
        means = {}
        for axis in ("novelty", "non_obviousness", "process_visibility", "generic_tropes"):
            vals = [v for a, v in pairs if a == axis]
            means[axis] = round(sum(vals) / len(vals), 2) if vals else None
        summary[tier] = means
        print(f"  {tier}:")
        for k, v in means.items():
            print(f"    {k:<22} {v}")

    out = {
        "source_eval": eval_path.name,
        "judged_at": datetime.now().isoformat(),
        "summary": summary,
        "detailed": detailed,
    }
    out_path = REPO_ROOT / "data" / "output" / f"judge_{eval_path.stem}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{GREEN}Saved:{RESET} {out_path}")


if __name__ == "__main__":
    main()
