"""
app.py — PSYCHE demo entry point.

Two tabs:
  1. Pipeline  — run Curiosity → Creativity → Critic, stages update live
  2. Compare   — pipeline vs vanilla side-by-side on the same task

Run:
    python src/app.py
"""

import sys
import json
import time
from pathlib import Path
from typing import Generator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import gradio as gr

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

_gemma: object | None = None  # Gemma4 instance


def _model_loaded() -> bool:
    return _gemma is not None


def _require_model():
    if not _model_loaded():
        raise gr.Error("Load a model first — use the model bar above.")


# ---------------------------------------------------------------------------
# Model bar (always visible)
# ---------------------------------------------------------------------------

def load_model(alias: str) -> str:
    global _gemma
    try:
        from src.IV_inference.gemma4_integration import Gemma4
        _gemma = Gemma4(alias)
        return f"✓ {_gemma.alias} loaded"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _fmt_curiosity(c: dict) -> str:
    if not c:
        return "_waiting..._"
    out = []
    for a in c.get("hidden_assumptions", []):
        out.append(f"**!** {a}")
    for q in c.get("questions", []):
        qtext = q.get("question", str(q)) if isinstance(q, dict) else str(q)
        out.append(f"**?** {qtext}")
    seeds = c.get("branch_seeds", [])
    if seeds:
        out.append("\n**Seeds:** " + " · ".join(seeds))
    return "\n\n".join(out) if out else "_empty_"


def _fmt_creativity(cr: dict) -> str:
    if not cr:
        return "_waiting..._"
    out = []
    for b in cr.get("branches", []):
        cands = ", ".join(b.get("candidates", [])[:3])
        out.append(f"**{b.get('id','?')}** {b.get('frame','')} → _{cands}_")
    for combo in cr.get("combinations", []):
        out.append(f"**+** {combo.get('from',[])} → **{combo.get('result','')}**")
    for o in cr.get("output", []):
        out.append(f"★ {o}")
    return "\n\n".join(out) if out else "_empty_"


def _fmt_critic(cr: dict) -> str:
    if not cr:
        return "_waiting..._"
    verdict = cr.get("verdict", "?").upper()
    badge = "✅ PASS" if verdict == "PASS" else "❌ FAIL"
    out = [f"## {badge}"]
    for s in cr.get("scores", []):
        out.append(
            f"**{s.get('candidate','?')[:50]}**  "
            f"novelty {s.get('novelty','?')}/10 · relevance {s.get('relevance','?')}/10"
        )
        if s.get("notes"):
            out.append(f"_{s['notes']}_")
    if verdict == "FAIL":
        for fb in cr.get("feedback_for_curiosity", []):
            out.append(f"→ {fb}")
    return "\n\n".join(out)


def run_pipeline(task: str, max_iters: int) -> Generator:
    """Yields (curiosity_md, creativity_md, critic_md, final_md, status) updates."""
    _require_model()
    if not task.strip():
        yield "_waiting..._", "_waiting..._", "_waiting..._", "", "Enter a task."
        return

    from src.I_pipeline.prompts import (
        CURIOSITY_SYSTEM, CURIOSITY_SYSTEM_WITH_FEEDBACK,
        CREATIVITY_SYSTEM, CRITIC_SYSTEM,
        build_curiosity_prompt, build_creativity_prompt, build_critic_prompt,
    )
    from src.I_pipeline.runner import _call_stage

    fn = _gemma.generate_fn()
    loop_trace = []
    critic_feedback = None
    t0 = time.time()

    for it in range(1, int(max_iters) + 1):
        # Curiosity
        yield "_generating..._", "_waiting..._", "_waiting..._", "", f"Iteration {it} — Curiosity..."
        c_sys = CURIOSITY_SYSTEM_WITH_FEEDBACK if critic_feedback else CURIOSITY_SYSTEM
        curiosity = _call_stage(fn, c_sys, build_curiosity_prompt(task, critic_feedback), "curiosity", False)

        # Creativity
        yield _fmt_curiosity(curiosity), "_generating..._", "_waiting..._", "", f"Iteration {it} — Creativity..."
        creativity = _call_stage(fn, CREATIVITY_SYSTEM, build_creativity_prompt(task, curiosity), "creativity", False)

        # Critic
        yield _fmt_curiosity(curiosity), _fmt_creativity(creativity), "_generating..._", "", f"Iteration {it} — Critic..."
        critic = _call_stage(fn, CRITIC_SYSTEM, build_critic_prompt(task, creativity), "critic", False)

        loop_trace.append({"iteration": it, "curiosity": curiosity, "creativity": creativity, "critic": critic})
        verdict = critic.get("verdict", "FAIL").upper()
        final_md = "## Final Output\n" + "\n".join(f"- **{o}**" for o in creativity.get("output", []))

        yield _fmt_curiosity(curiosity), _fmt_creativity(creativity), _fmt_critic(critic), final_md, \
              f"Iteration {it} — {verdict}"

        if verdict == "PASS":
            break
        critic_feedback = {
            "unexplored_directions": critic.get("unexplored_directions", []),
            "feedback_for_curiosity": critic.get("feedback_for_curiosity", []),
        }

    elapsed = round(time.time() - t0, 1)
    last = loop_trace[-1]
    final_md = (
        "## Final Output\n"
        + "\n".join(f"- **{o}**" for o in last["creativity"].get("output", []))
        + f"\n\n_{elapsed}s · {len(loop_trace)} iteration(s)_"
    )
    verdict = last["critic"].get("verdict", "?")
    yield _fmt_curiosity(last["curiosity"]), _fmt_creativity(last["creativity"]), \
          _fmt_critic(last["critic"]), final_md, f"Done — {verdict} in {elapsed}s"


# ---------------------------------------------------------------------------
# Compare tab: pipeline vs vanilla
# ---------------------------------------------------------------------------

def run_compare(task: str) -> Generator:
    """Yields (vanilla_md, pipeline_md) updates."""
    _require_model()
    if not task.strip():
        yield "_waiting..._", "_waiting..._"
        return

    # Vanilla first
    yield "_generating vanilla response..._", "_pipeline will run after vanilla..._"
    vanilla = _gemma.generate("You are a helpful assistant.", task)
    yield vanilla, "_running pipeline..._"

    # Now run 1 pipeline iteration (fast demo mode)
    from src.I_pipeline.prompts import (
        CURIOSITY_SYSTEM, CREATIVITY_SYSTEM, CRITIC_SYSTEM,
        build_curiosity_prompt, build_creativity_prompt, build_critic_prompt,
    )
    from src.I_pipeline.runner import _call_stage

    fn = _gemma.generate_fn()
    curiosity = _call_stage(fn, CURIOSITY_SYSTEM, build_curiosity_prompt(task), "curiosity", False)
    creativity = _call_stage(fn, CREATIVITY_SYSTEM, build_creativity_prompt(task, curiosity), "creativity", False)
    critic = _call_stage(fn, CRITIC_SYSTEM, build_critic_prompt(task, creativity), "critic", False)

    verdict = critic.get("verdict", "?").upper()
    pipeline_md = (
        "## Pipeline Output\n"
        + "\n".join(f"- **{o}**" for o in creativity.get("output", []))
        + f"\n\n_Critic verdict: {verdict}_"
    )
    yield vanilla, pipeline_md


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

CSS = """
footer { display: none !important; }
.model-bar { background: #161b22; border-bottom: 1px solid #30363d; padding: 12px 16px; }
.stage-col { border: 1px solid #30363d; border-radius: 8px; padding: 12px; }
"""

EXAMPLES = [
    "Generate names for an AI-powered creative studio",
    "Find a deep structural analogy between immune systems and market regulation",
    "What project ideas would impress the Gemma 4 Good Hackathon judges?",
    "Design a society that has solved loneliness as a public health problem",
    "Name my sausage dog",
]


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PSYCHE — Creative Reasoning") as app:

        # ---------- Model bar ----------
        gr.Markdown("# PSYCHE\n**Curiosity → Creativity → Critic** · Gemma 4")
        with gr.Row(elem_classes="model-bar"):
            model_drop = gr.Dropdown(
                choices=["e2b", "e4b", "26b", "31b"],
                value="e2b",
                label="Model",
                scale=0,
                min_width=120,
                allow_custom_value=True,
                info="e2b = fastest",
            )
            load_btn = gr.Button("Load", variant="primary", scale=0)
            model_status = gr.Textbox(
                value="Not loaded",
                label="",
                interactive=False,
                scale=1,
            )
        load_btn.click(fn=load_model, inputs=[model_drop], outputs=[model_status])

        # ---------- Tabs ----------
        with gr.Tabs():

            # ── Tab 1: Pipeline ──────────────────────────────────────────
            with gr.Tab("Pipeline"):
                gr.Markdown(
                    "Run any creative task through the full loop. "
                    "Each stage updates live as it completes."
                )
                with gr.Row():
                    task_box = gr.Textbox(
                        label="Task",
                        placeholder="Enter any creative task...",
                        lines=2,
                        scale=4,
                    )
                    with gr.Column(scale=1, min_width=140):
                        iters_slider = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Max iterations")
                        run_btn = gr.Button("Run", variant="primary")

                gr.Examples(examples=EXAMPLES, inputs=[task_box], label="Examples")

                status_bar = gr.Textbox(label="", interactive=False, lines=1)

                with gr.Row():
                    with gr.Column(elem_classes="stage-col"):
                        gr.Markdown("### 🔍 Curiosity")
                        curiosity_out = gr.Markdown("_waiting..._")
                    with gr.Column(elem_classes="stage-col"):
                        gr.Markdown("### 🌿 Creativity")
                        creativity_out = gr.Markdown("_waiting..._")
                    with gr.Column(elem_classes="stage-col"):
                        gr.Markdown("### 🔎 Critic")
                        critic_out = gr.Markdown("_waiting..._")

                final_out = gr.Markdown("")

                run_btn.click(
                    fn=run_pipeline,
                    inputs=[task_box, iters_slider],
                    outputs=[curiosity_out, creativity_out, critic_out, final_out, status_bar],
                )

            # ── Tab 2: Compare ───────────────────────────────────────────
            with gr.Tab("Compare"):
                gr.Markdown(
                    "Same task — vanilla Gemma 4 vs the pipeline. "
                    "Pipeline runs one iteration for speed."
                )
                compare_task = gr.Textbox(
                    label="Task",
                    placeholder="Enter a task to compare...",
                    lines=2,
                )
                gr.Examples(examples=EXAMPLES, inputs=[compare_task], label="Examples")
                compare_btn = gr.Button("Compare", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Vanilla Gemma 4")
                        vanilla_out = gr.Markdown("_waiting..._")
                    with gr.Column():
                        gr.Markdown("### Pipeline Output")
                        pipeline_compare_out = gr.Markdown("_waiting..._")

                compare_btn.click(
                    fn=run_compare,
                    inputs=[compare_task],
                    outputs=[vanilla_out, pipeline_compare_out],
                )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(inbrowser=True, server_port=7860)
