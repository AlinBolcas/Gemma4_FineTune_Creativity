"""
evaluate.py - 3-tier evaluation: vanilla vs scaffolded vs fine-tuned.

Run interactively:
    python src/IV_inference/evaluate.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Colors
CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; MAGENTA = "\033[95m"
RED = "\033[91m"; GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"

TIER1_SYSTEM_PROMPT = "You are a helpful assistant."
TIER3_MINIMAL_SYSTEM_PROMPT = "You are a creative reasoning assistant."

# Format-invitation suffix: appended to the user prompt for `direct_primed` mode.
# This is what unlocks the trained trace format that base-style otherwise suppresses.
TIER3_PRIME_SUFFIX = (
    "\n\nFormat your answer EXACTLY like this template, filling in real content:\n"
    "## Iteration 1\n"
    "### Curiosity\n"
    "Hidden assumptions:\n- ...\nKey questions:\n- ...\n"
    "Branch seeds: ...\n"
    "### Creativity\n"
    "Research:\n- ...\nBranches:\n- B1: ...\nPruned:\n- ...\n"
    "Combinations:\n- ...\nCandidates:\n- ...\n\n"
    "## Final Output\n- ..."
)

SCAFFOLDED_SYSTEM_PROMPT = (
    "You are a creative reasoning engine. When given a task, think through it "
    "using Curiosity -> Creativity -> Critic. Surface hidden assumptions, "
    "explore diverse branches, cross-pollinate, critically evaluate, then present "
    "your strongest candidates.\n\n"
    "Structure: 1) Curiosity 2) Creativity 3) Critic 4) Final Output"
)

ADVANCED_SCAFFOLDED_SYSTEM_PROMPT = (
    "You are a creative reasoning engine. When given a task, use a full staged process: "
    "1) Curiosity map 2) Curiosity expand 3) Curiosity distill 4) Socratic output "
    "5) Creativity research plan 6) Creativity branch 7) Develop each branch "
    "8) Selection and pruning 9) Combinatory mixing 10) Final synthesis 11) Critic.\n\n"
    "Keep the stages explicit and feed curiosity steering into creativity throughout."
)


def load_eval_prompts(path: Path | None = None) -> list[str]:
    if path is None:
        path = REPO_ROOT / "data" / "input" / "seed_prompts.json"
    with open(path) as f:
        return json.load(f).get("eval_held_out", [])


def _summarize_pipeline_result(result: dict) -> str:
    """
    Render the FULL reasoning trace of a pipeline run so Tier 2 is comparable
    in depth to Tier 1 / Tier 3 outputs. Previous version only printed the
    final bullets, which made Tier 2 look artificially weak in writeups.
    """
    final_output = result.get("final_output", [])
    iterations = result.get("loop", [])
    runner = result.get("_meta", {}).get("runner", "simple")
    elapsed = result.get("_meta", {}).get("elapsed_sec")
    calls = result.get("_meta", {}).get("llm_calls")

    lines = [f"_Pipeline runner: **{runner}**_"]
    if elapsed is not None:
        lines.append(f"_LLM calls: {calls} | Elapsed: {elapsed}s_")
    lines.append("")

    for it in iterations:
        idx = it.get("iteration", "?")
        verdict = it.get("critic", {}).get("verdict", "?")
        lines.append(f"## Iteration {idx} (critic: {verdict})")

        if runner == "advanced" and it.get("advanced"):
            adv = it["advanced"]
            lines.extend(_render_advanced(adv))
        else:
            lines.extend(_render_simple(it))

        critic = it.get("critic", {})
        if critic:
            lines.append("\n### Critic")
            for key in ("novelty_score", "relevance_score", "verdict"):
                if key in critic:
                    lines.append(f"- {key}: {critic[key]}")
            for key in ("feedback_for_curiosity", "feedback_for_creativity"):
                vals = critic.get(key)
                if vals:
                    lines.append(f"- {key}:")
                    for v in vals[:5]:
                        lines.append(f"  - {v}")
        lines.append("")

    if final_output:
        lines.append("## Final Output")
        for item in final_output:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _render_simple(it: dict) -> list[str]:
    out = []
    cur = it.get("curiosity") or {}
    cre = it.get("creativity") or {}
    if cur:
        out.append("\n### Curiosity")
        for q in (cur.get("questions") or [])[:6]:
            out.append(f"- Q: {q}")
        for h in (cur.get("hidden_assumptions") or [])[:4]:
            out.append(f"- assumption: {h}")
    if cre:
        out.append("\n### Creativity")
        for b in (cre.get("branches") or [])[:6]:
            label = b.get("label") or b.get("idea") or str(b)[:80]
            out.append(f"- branch: {label}")
        for c in (cre.get("candidates") or [])[:6]:
            out.append(f"- candidate: {c}")
    return out


def _render_advanced(adv: dict) -> list[str]:
    out = []
    cmap = adv.get("curiosity_map") or {}
    cexp = adv.get("curiosity_expand") or {}
    cdis = adv.get("curiosity_distill") or {}
    socr = adv.get("socratic_output") or {}
    plan = adv.get("creativity_research_plan") or {}
    branch = adv.get("creativity_branch") or {}
    develop = adv.get("creativity_develop") or []
    selection = adv.get("creativity_selection") or {}
    mixing = adv.get("creativity_mixing") or {}
    fsynth = adv.get("creativity_final_synthesis") or {}

    if cmap:
        out.append("\n### Curiosity Map")
        for d in (cmap.get("curiosity_domains") or [])[:5]:
            out.append(f"- {d.get('lens', '?')}: {d.get('focus', '')[:90]}")

    if cexp:
        out.append("\n### Curiosity Expand")
        for q in (cexp.get("expanded_questions") or [])[:6]:
            out.append(f"- {q.get('question', str(q))[:120]}")

    if cdis or socr:
        out.append("\n### Socratic Output")
        for q in (socr.get("best_questions") or cdis.get("best_questions") or [])[:5]:
            out.append(f"- Q: {q}")
        scaffold = socr.get("socratic_scaffold") or cdis.get("socratic_scaffold")
        if scaffold:
            out.append(f"- scaffold: {str(scaffold)[:200]}")

    if plan:
        out.append("\n### Creativity Research Plan")
        for k in ("known_patterns", "adjacent_domains", "creative_tensions"):
            vals = plan.get(k) or []
            for v in vals[:3]:
                out.append(f"- {k}: {str(v)[:120]}")

    if branch:
        out.append("\n### Creativity Branches")
        for b in (branch.get("branches") or [])[:6]:
            bid = b.get("branch_id", "?")
            frame = b.get("frame", "")
            out.append(f"- {bid}: {frame[:120]}")

    if develop:
        out.append("\n### Branch Development")
        for b in develop[:6]:
            bid = b.get("branch_id", "?")
            out.append(f"- {bid}: {len(b.get('chain_steps', []))} chain steps")

    if selection:
        out.append("\n### Selection")
        for s in (selection.get("scored_branches") or [])[:6]:
            out.append(f"- {s.get('branch_id', '?')} novelty={s.get('novelty_score', '?')} keep={s.get('keep', '?')}")

    if mixing:
        out.append("\n### Combinatory Mixing")
        for h in (mixing.get("hybrids") or [])[:5]:
            out.append(f"- hybrid: {str(h.get('idea', h))[:140]}")

    if fsynth:
        out.append("\n### Final Synthesis Candidates")
        for c in (fsynth.get("output") or [])[:6]:
            out.append(f"- {c}")
    return out


def _run_eval_mode(generate_fn, prompt: str, mode: str) -> tuple[str, dict | None]:
    mode = (mode or "prompt_simple").strip().lower()
    if mode == "prompt_advanced":
        return generate_fn(ADVANCED_SCAFFOLDED_SYSTEM_PROMPT, prompt), None
    if mode == "simple_pipeline":
        from src.I_pipeline.runner import run_loop

        result = run_loop(task=prompt, generate_fn=generate_fn, max_iterations=1, verbose=False)
        return _summarize_pipeline_result(result), result
    if mode == "advanced_pipeline":
        from src.I_pipeline.runner_advanced import run_advanced_loop

        result = run_advanced_loop(task=prompt, generate_fn=generate_fn, max_iterations=1, verbose=False)
        return _summarize_pipeline_result(result), result
    return generate_fn(SCAFFOLDED_SYSTEM_PROMPT, prompt), None


def _run_tuned_mode(generate_fn, prompt: str, tuned_mode: str, scaffold_mode: str) -> tuple[str, dict | None]:
    tuned_mode = (tuned_mode or "direct_minimal").strip().lower()
    if tuned_mode == "match_scaffold":
        return _run_eval_mode(generate_fn, prompt, scaffold_mode)
    if tuned_mode == "direct_plain":
        return generate_fn(TIER1_SYSTEM_PROMPT, prompt), None
    if tuned_mode == "direct_primed":
        # Invite the trained format explicitly. Diagnostic showed the architecture
        # IS in the weights but is suppressed by base style. Priming unlocks it.
        primed_user = prompt + TIER3_PRIME_SUFFIX
        if getattr(generate_fn, "is_tuned", False):
            try:
                return generate_fn(
                    TIER3_MINIMAL_SYSTEM_PROMPT, primed_user,
                    temperature_override=0.4, top_k_override=64, top_p_override=0.9, do_sample=True,
                ), None
            except TypeError:
                pass
        return generate_fn(TIER3_MINIMAL_SYSTEM_PROMPT, primed_user), None
    if tuned_mode == "direct_strict":
        # Deterministic-ish: low-temp greedy + exact training system prompt.
        # Reveals what the adapter actually learned without sampling drift.
        if getattr(generate_fn, "is_tuned", False):
            try:
                return generate_fn(
                    TIER3_MINIMAL_SYSTEM_PROMPT, prompt,
                    temperature_override=0.0,
                    top_k_override=1,
                    top_p_override=1.0,
                    do_sample=False,
                ), None
            except TypeError:
                # Older generate_fn without override kwargs - fall through
                pass
        return generate_fn(TIER3_MINIMAL_SYSTEM_PROMPT, prompt), None
    return generate_fn(TIER3_MINIMAL_SYSTEM_PROMPT, prompt), None


def evaluate(
    generate_fn_vanilla,
    generate_fn_tuned=None,
    prompts=None,
    verbose=True,
    scaffold_mode: str = "advanced_pipeline",
    tuned_mode: str = "direct_minimal",
) -> list[dict]:
    if prompts is None:
        prompts = load_eval_prompts()

    results = []
    for i, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"\n{BOLD}{'='*60}{RESET}")
            print(f"  {BOLD}Eval {i}/{len(prompts)}{RESET}")
            print(f"  {prompt[:70]}...")
            print(f"{'='*60}")

        row = {"prompt": prompt, "timestamp": datetime.now().isoformat()}

        # Tier 1: vanilla
        if verbose:
            print(f"\n  {CYAN}Tier 1: Vanilla...{RESET}")
        row["tier1_vanilla"] = generate_fn_vanilla(TIER1_SYSTEM_PROMPT, prompt)
        if verbose:
            _print_response("Tier 1", row["tier1_vanilla"], CYAN)

        # Tier 2: scaffolded
        if verbose:
            print(f"\n  {GREEN}Tier 2: Scaffolded...{RESET}")
        tier2_text, tier2_trace = _run_eval_mode(generate_fn_vanilla, prompt, scaffold_mode)
        row["tier2_scaffolded"] = tier2_text
        row["tier2_mode"] = scaffold_mode
        if tier2_trace is not None:
            row["tier2_trace"] = tier2_trace
        if verbose:
            _print_response("Tier 2", row["tier2_scaffolded"], GREEN)

        # Tier 3: fine-tuned
        if generate_fn_tuned:
            if verbose:
                print(f"\n  {MAGENTA}Tier 3: Fine-tuned...{RESET}")
            tier3_text, tier3_trace = _run_tuned_mode(generate_fn_tuned, prompt, tuned_mode, scaffold_mode)
            row["tier3_tuned"] = tier3_text
            row["tier3_mode"] = tuned_mode
            if tier3_trace is not None:
                row["tier3_trace"] = tier3_trace
            if verbose:
                _print_response("Tier 3", row["tier3_tuned"], MAGENTA)
        else:
            row["tier3_tuned"] = "(not available)"

        results.append(row)
    return results


def _print_response(label: str, text: str, color: str):
    # Truncate long responses for readability
    lines = text.strip().split("\n")
    preview = "\n".join(lines[:15])
    if len(lines) > 15:
        preview += f"\n{GREY}  ... ({len(lines) - 15} more lines){RESET}"
    for line in preview.split("\n"):
        print(f"  {color}  {line}{RESET}")


def export_eval(results: list[dict], output_path: Path | None = None):
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPO_ROOT / "data" / "output" / f"eval_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  {GREEN}Saved: {output_path.name}{RESET}")
    
    # Save Markdown
    md_path = output_path.with_suffix(".md")
    md_lines = ["# 3-Tier Evaluation Results\n"]
    for i, row in enumerate(results, 1):
        md_lines.append(f"## Prompt {i}")
        md_lines.append(f"> {row.get('prompt', '')}\n")
        md_lines.append("### Tier 1: Vanilla")
        md_lines.append(f"{row.get('tier1_vanilla', '')}\n")
        md_lines.append("### Tier 2: Scaffolded")
        md_lines.append(f"*(Mode: {row.get('tier2_mode', '')})*\n")
        md_lines.append(f"{row.get('tier2_scaffolded', '')}\n")
        md_lines.append("### Tier 3: Fine-Tuned")
        md_lines.append(f"*(Mode: {row.get('tier3_mode', '')})*\n")
        md_lines.append(f"{row.get('tier3_tuned', '')}\n")
        md_lines.append("---\n")
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  {GREEN}Saved: {md_path.name}{RESET}")


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

PRESETS = {
    "final_recommended": {
        "label": "Final recommended",
        "description": "Best hackathon comparison. All held-out prompts, Tier 2 advanced pipeline, Tier 3 invites learned trace format.",
        "model_alias": "e4b",
        "prompt_mode": "all",
        "scaffold_mode": "advanced_pipeline",
        "tuned_mode": "direct_primed",
    },
    "quick_smoke": {
        "label": "Quick smoke test",
        "description": "Fast check. One prompt, simple scaffold, tuned stays direct.",
        "model_alias": "e2b",
        "prompt_mode": "single",
        "scaffold_mode": "prompt_simple",
        "tuned_mode": "direct_minimal",
    },
    "adapter_diagnostic": {
        "label": "Adapter diagnostic (greedy)",
        "description": "Uses temp=0 + training-exact prompt on Tier 3. Reveals what the adapter actually learned.",
        "model_alias": "e4b",
        "prompt_mode": "single",
        "scaffold_mode": "prompt_simple",
        "tuned_mode": "direct_strict",
    },
    "custom": {
        "label": "Custom",
        "description": "Choose prompts, scaffold baseline, and tuned behavior manually.",
        "model_alias": "e2b",
        "prompt_mode": "custom",
        "scaffold_mode": "prompt_simple",
        "tuned_mode": "direct_minimal",
    },
}

SCAFFOLD_LABELS = {
    "prompt_simple": "Prompt scaffold, simple",
    "prompt_advanced": "Prompt scaffold, advanced",
    "simple_pipeline": "Pipeline runner, simple",
    "advanced_pipeline": "Pipeline runner, advanced",
}

TUNED_LABELS = {
    "direct_minimal": "Direct tuned reply with minimal identity prompt",
    "direct_primed": "Direct tuned reply, prompt invites the learned trace format (RECOMMENDED)",
    "direct_strict": "Direct tuned reply, greedy (temp=0) + exact training prompt",
    "direct_plain": "Direct tuned reply with plain helpful-assistant prompt",
    "match_scaffold": "Apply the same scaffold to tuned too",
}


def _guess_model_alias_from_name(name: str) -> str | None:
    lowered = (name or "").lower()
    for alias in ("e4b", "e2b", "26b", "31b"):
        if alias in lowered:
            return alias
    return None


def _read_adapter_alias(adapter_path: Path) -> str | None:
    run_dir = REPO_ROOT / "data" / "output" / "training_runs" / adapter_path.name
    for snapshot_name in ("training_complete.json", "training_start.json"):
        snapshot_path = run_dir / snapshot_name
        if snapshot_path.exists():
            try:
                data = json.loads(snapshot_path.read_text(encoding="utf-8"))
                alias = data.get("config", {}).get("model", {}).get("alias")
                if alias:
                    return str(alias)
            except Exception:
                pass
    return _guess_model_alias_from_name(adapter_path.name)


def _pick_preset() -> str:
    print(f"\n{YELLOW}Evaluation preset:{RESET}")
    keys = list(PRESETS.keys())
    for i, key in enumerate(keys, 1):
        preset = PRESETS[key]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        print(f"  [{i}] {preset['label']}{default}")
        print(f"      {GREY}{preset['description']}{RESET}")
    raw = input("\n  Choice [1]: ").strip() or "1"
    if raw.isdigit() and 1 <= int(raw) <= len(keys):
        return keys[int(raw) - 1]
    return keys[0]


def _pick_model_alias(models: dict, default_alias: str) -> str:
    print(f"\n{YELLOW}Base model for vanilla + tuned adapter:{RESET}")
    aliases = list(models.keys())
    default_index = aliases.index(default_alias) + 1 if default_alias in aliases else 1
    for i, alias in enumerate(aliases, 1):
        info = models[alias]
        default = f" {GREY}<- default{RESET}" if i == default_index else ""
        print(f"  [{i}] {alias:<5} {info['ram_gb']:>3}GB  {info['description']}{default}")
    raw = input(f"\n  Choice [{default_index}]: ").strip() or str(default_index)
    if raw.isdigit() and 1 <= int(raw) <= len(aliases):
        return aliases[int(raw) - 1]
    return aliases[default_index - 1]


def _pick_adapter(adapter_dirs: list[Path], default_yes: bool) -> Path | None:
    hint = "Y/n" if default_yes else "y/N"
    print(f"\n{YELLOW}Include fine-tuned adapter in Tier 3? [{hint}]{RESET}")
    raw = input("  > ").strip().lower()
    if not raw:
        include = default_yes
    else:
        include = raw in ("y", "yes")
    if not include:
        return None
    if not adapter_dirs:
        print(f"  {YELLOW}No adapter dirs found in data/output/models/{RESET}")
        return None

    print(f"\n{YELLOW}Available adapters:{RESET}")
    for i, path in enumerate(adapter_dirs, 1):
        alias = _read_adapter_alias(path)
        suffix = f"  {GREY}(base {alias}){RESET}" if alias else ""
        print(f"  [{i}] {path.name}{suffix}")
    raw = input(f"\n  Choice [1]: ").strip() or "1"
    idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(adapter_dirs) else 0
    return adapter_dirs[idx]


def _pick_prompts(prompts: list[str], default_all: bool) -> list[str]:
    print(f"\n{YELLOW}Eval prompts:{RESET}")
    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}] {prompt[:65]}...")
    print(f"  [a] all ({len(prompts)})")
    default_raw = "a" if default_all else "1"
    raw = input(f"\n  Choice [{default_raw}]: ").strip().lower() or default_raw
    if raw == "a":
        return prompts
    if raw.isdigit() and 1 <= int(raw) <= len(prompts):
        return [prompts[int(raw) - 1]]
    return prompts if default_all else [prompts[0]]


def _pick_scaffold_mode(default_mode: str) -> str:
    print(f"\n{YELLOW}Tier 2 scaffold baseline:{RESET}")
    options = [
        ("1", "prompt_simple", "Simple prompt asking for Curiosity -> Creativity -> Critic stages."),
        ("2", "prompt_advanced", "Longer explicit staged prompt. Good when you want visible manual scaffolding."),
        ("3", "simple_pipeline", "Structured pipeline runner. One loop. Cleaner than a long prompt."),
        ("4", "advanced_pipeline", "Most complete scaffold baseline. Best final comparison default."),
    ]
    default_key = next((key for key, mode, _ in options if mode == default_mode), "4")
    for key, mode, description in options:
        default = f" {GREY}<- default{RESET}" if key == default_key else ""
        print(f"  [{key}] {SCAFFOLD_LABELS[mode]}{default}")
        print(f"      {GREY}{description}{RESET}")
    raw = input(f"\n  Choice [{default_key}]: ").strip() or default_key
    return {key: mode for key, mode, _ in options}.get(raw, default_mode)


def _pick_tuned_mode(default_mode: str) -> str:
    print(f"\n{YELLOW}Tier 3 tuned behavior:{RESET}")
    options = [
        ("1", "direct_primed", "Invites the learned trace format (Curiosity / Branches / Final Output). Best result quality."),
        ("2", "direct_minimal", "Tiny identity prompt only. Tuned model often defaults to base style here."),
        ("3", "direct_strict", "Greedy (temp=0) + training-exact prompt. Best diagnostic of what was actually learned."),
        ("4", "direct_plain", "Direct reply with the same plain helpful-assistant prompt as Tier 1."),
        ("5", "match_scaffold", "Apply the same scaffold to tuned too. Extra ablation, not the main comparison."),
    ]
    default_key = next((key for key, mode, _ in options if mode == default_mode), "1")
    for key, mode, description in options:
        default = f" {GREY}<- default{RESET}" if key == default_key else ""
        print(f"  [{key}] {TUNED_LABELS[mode]}{default}")
        print(f"      {GREY}{description}{RESET}")
    raw = input(f"\n  Choice [{default_key}]: ").strip() or default_key
    return {key: mode for key, mode, _ in options}.get(raw, default_mode)

def _tui():
    from src.IV_inference.gemma4_integration import load_gemma4, load_finetuned_gemma4, MODELS

    print(f"\n{BOLD}{MAGENTA}{'='*60}{RESET}")
    print(f"  {BOLD}3-Tier Evaluation{RESET}")
    print(f"  Vanilla vs Scaffolded vs Fine-tuned")
    print(f"{BOLD}{MAGENTA}{'='*60}{RESET}")

    prompts = load_eval_prompts()
    preset_key = _pick_preset()
    preset = PRESETS[preset_key]

    models_dir = REPO_ROOT / "data" / "output" / "models"
    adapter_dirs = [p for p in sorted(models_dir.glob("*")) if p.is_dir()]
    adapter_path = _pick_adapter(adapter_dirs, default_yes=(preset_key == "final_recommended"))
    inferred_alias = _read_adapter_alias(adapter_path) if adapter_path else None
    model_alias = _pick_model_alias(MODELS, inferred_alias or preset["model_alias"])

    if adapter_path and inferred_alias and inferred_alias != model_alias:
        print(f"\n{YELLOW}Adapter looks like base `{inferred_alias}`. Switching base model to match it.{RESET}")
        model_alias = inferred_alias

    if preset["prompt_mode"] == "all":
        selected = prompts
    elif preset["prompt_mode"] == "single":
        selected = [prompts[0]] if prompts else []
    else:
        selected = _pick_prompts(prompts, default_all=True)

    if preset_key == "custom":
        scaffold_mode = _pick_scaffold_mode(preset["scaffold_mode"])
        tuned_mode = _pick_tuned_mode(preset["tuned_mode"])
    else:
        scaffold_mode = preset["scaffold_mode"]
        tuned_mode = preset["tuned_mode"]
        print(f"\n{GREEN}Using preset:{RESET} {preset['label']}")
        print(f"  Tier 2: {SCAFFOLD_LABELS[scaffold_mode]}")
        print(f"  Tier 3: {TUNED_LABELS[tuned_mode]}")
        print(f"  Prompts: {'all held-out prompts' if preset['prompt_mode'] == 'all' else 'first held-out prompt'}")

    generate_fn = load_gemma4(model=model_alias)
    tuned_fn = load_finetuned_gemma4(str(adapter_path), base_model=model_alias) if adapter_path else None

    # Run
    results = evaluate(
        generate_fn_vanilla=generate_fn,
        generate_fn_tuned=tuned_fn,
        prompts=selected,
        verbose=True,
        scaffold_mode=scaffold_mode,
        tuned_mode=tuned_mode,
    )
    export_eval(results)
    print(f"\n  {GREEN}Evaluation complete: {len(results)} prompts.{RESET}")


if __name__ == "__main__":
    _tui()
