"""
export.py - Convert pipeline JSON output to readable Markdown.

Usage:
    from src.V_utility.export import pipeline_to_markdown, save_markdown_alongside

    # Convert a pipeline result dict
    md = pipeline_to_markdown(result)

    # Save .md next to a .json file
    save_markdown_alongside(Path("data/output/pipeline_xxx.json"))

    # Convert all JSON files in data/output/
    from src.V_utility.export import convert_all_outputs
    convert_all_outputs()
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output"

STAGE_EMOJI = {
    "curiosity":  "🔍",
    "creativity": "🌿",
    "critic":     "🔎",
}

VERDICT_SYMBOL = {"PASS": "✅ PASS", "FAIL": "❌ FAIL"}


def _append_advanced_iteration(lines: list[str], iteration: dict):
    advanced = iteration.get("advanced", {})
    if not advanced:
        return

    lines.append("### Advanced Stages")
    lines.append("")

    cmap = advanced.get("curiosity_map", {})
    if cmap:
        lines.append("**Curiosity Map:**")
        lines.append(
            f"- novelty={cmap.get('global_novelty_estimate', '?')}/10"
            f" · branch_budget={cmap.get('branch_budget', '?')}"
        )
        for item in cmap.get("curiosity_domains", [])[:5]:
            lines.append(f"- {item.get('id', '?')}: {item.get('domain', '')} [{item.get('lens', '')}]")
        lines.append("")

    cexpand = advanced.get("curiosity_expand", {})
    if cexpand.get("expanded_branches"):
        lines.append("**Curiosity Expand:**")
        for item in cexpand["expanded_branches"][:5]:
            lines.append(
                f"- {item.get('id', '?')}: {item.get('direction', '')} "
                f"(strength {item.get('curiosity_strength', '?')}/10)"
            )
        lines.append("")

    cdistill = advanced.get("curiosity_distill", {})
    if cdistill.get("best_questions"):
        lines.append("**Curiosity Distill:**")
        for item in cdistill["best_questions"][:4]:
            lines.append(f"- {item.get('question', '')}")
        lines.append("")

    research = advanced.get("creativity_research_plan", {})
    if research.get("creative_tensions"):
        lines.append("**Creativity Research Plan:**")
        for item in research["creative_tensions"][:4]:
            lines.append(f"- {item}")
        lines.append("")

    branch = advanced.get("creativity_branch", {})
    if branch.get("branches"):
        lines.append("**Creativity Branches:**")
        for item in branch["branches"][:6]:
            lines.append(f"- {item.get('id', '?')}: {item.get('frame', '')}")
        lines.append("")

    develop = advanced.get("creativity_develop", [])
    if develop:
        lines.append("**Developed Branches:**")
        for item in develop[:6]:
            outputs = ", ".join(item.get("branch_outputs", [])[:3])
            lines.append(f"- {item.get('branch_id', '?')}: {outputs}")
        lines.append("")

    selection = advanced.get("creativity_selection", {})
    if selection.get("kept_branch_ids"):
        lines.append(f"**Selection:** kept {', '.join(selection['kept_branch_ids'])}")
        lines.append("")

    mixing = advanced.get("creativity_mixing", {})
    if mixing.get("hybrids"):
        lines.append("**Combinatory Mixing:**")
        for item in mixing["hybrids"][:4]:
            lines.append(f"- {item.get('from_branch_ids', [])} -> {item.get('concept', '')}")
        lines.append("")

    synthesis = advanced.get("creativity_final_synthesis", {})
    if synthesis.get("primary_candidates"):
        lines.append("**Final Synthesis:**")
        for item in synthesis["primary_candidates"][:4]:
            lines.append(f"- {item.get('id', '?')}: {item.get('title', '')} | {item.get('concept', '')}")
        lines.append("")


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------

def pipeline_to_markdown(data: dict) -> str:
    """Convert a full pipeline result dict into a clean Markdown document."""
    lines = []

    task = data.get("input", "Unknown task")
    domain = data.get("domain", "general")
    loop = data.get("loop", [])
    final_output = data.get("final_output", [])
    meta = data.get("_meta", {})

    # Header
    lines.append(f"# {task}")
    lines.append(f"")
    lines.append(f"**Domain:** {domain}")
    if meta.get("timestamp"):
        lines.append(f"**Generated:** {meta['timestamp'][:19].replace('T', ' ')}")
    if meta.get("elapsed_sec"):
        lines.append(f"**Elapsed:** {meta['elapsed_sec']}s")
    lines.append(f"**Iterations:** {len(loop)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Final output first - most useful thing at the top
    lines.append("## Final Output")
    lines.append("")
    for o in final_output:
        lines.append(f"- **{o}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Each iteration
    for iteration in loop:
        it_num = iteration.get("iteration", "?")
        lines.append(f"## Iteration {it_num}")
        lines.append("")

        # CURIOSITY
        cur = iteration.get("curiosity", {})
        lines.append(f"### {STAGE_EMOJI['curiosity']} Curiosity")
        lines.append("")

        if cur.get("hidden_assumptions"):
            lines.append("**Hidden Assumptions:**")
            for a in cur["hidden_assumptions"]:
                lines.append(f"- {a}")
            lines.append("")

        if cur.get("unexplored_domains"):
            lines.append("**Unexplored Domains:**")
            for d in cur["unexplored_domains"]:
                lines.append(f"- {d}")
            lines.append("")

        if cur.get("questions"):
            lines.append("**Key Questions:**")
            for q in cur["questions"]:
                if isinstance(q, dict):
                    lines.append(f"- **{q.get('question', '')}**")
                    if q.get("why_this_unlocks"):
                        lines.append(f"  *{q['why_this_unlocks']}*")
                else:
                    lines.append(f"- {q}")
            lines.append("")

        if cur.get("branch_seeds"):
            lines.append(f"**Branch Seeds:** {' · '.join(cur['branch_seeds'])}")
            lines.append("")

        # CREATIVITY
        cre = iteration.get("creativity", {})
        lines.append(f"### {STAGE_EMOJI['creativity']} Creativity")
        lines.append("")

        if cre.get("research"):
            lines.append("**Research:**")
            for r in cre["research"]:
                lines.append(f"- {r}")
            lines.append("")

        if cre.get("branches"):
            lines.append("**Branches:**")
            for b in cre["branches"]:
                cands = ", ".join(b.get("candidates", []))
                lines.append(f"- **{b.get('id','?')} — {b.get('frame','')}:** {cands}")
            lines.append("")

        if cre.get("pruned"):
            lines.append("**Pruned:**")
            for p in cre["pruned"]:
                reason = p.get("reason", "") if isinstance(p, dict) else str(p)
                pid = p.get("id", "?") if isinstance(p, dict) else "?"
                lines.append(f"- ~~{pid}~~ — {reason}")
            lines.append("")

        if cre.get("combinations"):
            lines.append("**Combinations:**")
            for c in cre["combinations"]:
                frm = c.get("from", [])
                result = c.get("result", "")
                note = c.get("novelty_note", "")
                lines.append(f"- {frm} → **{result}**")
                if note:
                    lines.append(f"  *{note}*")
            lines.append("")

        if cre.get("dead_ends"):
            lines.append("**Dead Ends:**")
            for de in cre["dead_ends"]:
                if isinstance(de, dict):
                    lines.append(f"- {de.get('combination', '?')}: {de.get('reason', '')}")
                else:
                    lines.append(f"- {de}")
            lines.append("")

        if cre.get("output"):
            lines.append("**Candidates:**")
            for o in cre["output"]:
                lines.append(f"- {o}")
            lines.append("")

        # CRITIC
        cri = iteration.get("critic", {})
        verdict = cri.get("verdict", "?").upper()
        verdict_display = VERDICT_SYMBOL.get(verdict, verdict)
        lines.append(f"### {STAGE_EMOJI['critic']} Critic — {verdict_display}")
        lines.append("")

        if cri.get("scores"):
            lines.append("| Candidate | Novelty | Relevance | Notes |")
            lines.append("|---|---|---|---|")
            for s in cri["scores"]:
                candidate = s.get("candidate", "?")[:60]
                n = s.get("novelty", "?")
                r = s.get("relevance", "?")
                notes = s.get("notes", "").replace("|", "/")[:80]
                lines.append(f"| {candidate} | {n}/10 | {r}/10 | {notes} |")
            lines.append("")

        if verdict == "FAIL":
            if cri.get("unexplored_directions"):
                lines.append("**Unexplored Directions:**")
                for u in cri["unexplored_directions"]:
                    lines.append(f"- {u}")
                lines.append("")
            if cri.get("feedback_for_curiosity"):
                lines.append("**Feedback for Next Pass:**")
                for fb in cri["feedback_for_curiosity"]:
                    lines.append(f"- {fb}")
                lines.append("")
            if cri.get("feedback_for_creativity"):
                lines.append("**Feedback for Creativity:**")
                for fb in cri["feedback_for_creativity"]:
                    lines.append(f"- {fb}")
                lines.append("")

        _append_advanced_iteration(lines, iteration)

        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def save_markdown_alongside(json_path: Path) -> Path:
    """Read a pipeline JSON file and write a .md next to it."""
    with open(json_path) as f:
        data = json.load(f)
    md = pipeline_to_markdown(data)
    md_path = json_path.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")
    return md_path


def convert_all_outputs(output_dir: Path = OUTPUT_DIR) -> list[Path]:
    """Convert every pipeline JSON in data/output/ that doesn't already have a .md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for json_path in sorted(output_dir.glob("*.json")):
        md_path = json_path.with_suffix(".md")
        if md_path.exists():
            continue
        try:
            save_markdown_alongside(json_path)
            converted.append(md_path)
        except Exception as e:
            print(f"  [skip] {json_path.name}: {e}")
    if not converted:
        print("  No new files to convert.")
    return converted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            p = save_markdown_alongside(Path(arg))
            print(p.resolve())
    else:
        print("Converting all output JSON files...")
        convert_all_outputs()
