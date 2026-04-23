# Video Script — Gemma 4 Creative Reasoning Fine-Tune

**Length:** ~3 minutes (180 seconds)
**Tone:** Direct. Technical. No fluff. No hype music.
**Voice-over:** Calm, precise. Think Karpathy, not TED.
**Visual style:** Screen recording with clean transitions. Terminal UIs. Architecture diagram. Loss curve.

---

## Scene 1 — The Hook (0:00–0:20, 20s)

**VISUAL:** Single line of text, large, centered on black:
> *"LLMs don't lack knowledge. They lack process."*

Then fade into the architecture diagram (`docs/creativity_curiosity_graph.png`).

**VOICE-OVER:**

> When you ask a language model an open-ended creative question, it generates. It does not ask itself what you actually meant. It does not explore alternative framings. It does not evaluate whether its answer is genuinely new or just fluent.
>
> What's missing is not capability. It's **process**.

---

## Scene 2 — The Architecture (0:20–0:55, 35s)

**VISUAL:** The architecture diagram animates. Left side `Curiosity` stages 1–4 light up. Then the arrow labeled "steering" fires across. Right side `Creativity` stages 5–10 light up. Then the `Critic` badge pulses.

**VOICE-OVER:**

> So I built it. An 11-stage cognitive architecture for creative reasoning.
>
> **Curiosity** is a Socratic engine. It never answers. It maps the problem space across five lenses — assumption, opposite, expert, frontier, cross-domain — expands question branches, prunes the weak ones, and distills the strongest into a steering signal.
>
> **Creativity** receives that signal. Generates structurally distinct branches, develops each independently, selects on novelty, and cross-pollinates the survivors into hybrids.
>
> A **Critic** evaluates. On FAIL, it sends feedback back to curiosity and the loop runs again.
>
> Each stage is a separate LLM call. Own prompt, own schema, own validation. This is a reasoning system, not a prompt.

---

## Scene 3 — Self-Distillation (0:55–1:25, 30s)

**VISUAL:** Split-screen flow chart:

```
seed prompts → pipeline runs → reasoning traces → SFT data → fine-tune
```

Then terminal showing `python src/II_dataGen/generate.py` scrolling through trace generation. Then a sample trace in markdown with `### Curiosity / ### Creativity / ## Final Output` visible.

**VOICE-OVER:**

> Then I used the pipeline itself to generate training data. 334 reasoning traces across 8 creative domains — no teacher model, no external distillation. Gemma 4 generating traces from its own latent capacity, through structured prompting.
>
> The claim is sharp: **the cognitive architecture is recoverable from Gemma 4, and it can be internalized back into Gemma 4** through fine-tuning.
>
> Not teaching a new skill. Surfacing one that was already there, and making it native.

---

## Scene 4 — Training (1:25–1:50, 25s)

**VISUAL:** Loss curve plot (`data/output/reports/..._e4b_v2_strong/loss.png`) animates as training progresses. Callouts pop up:

- "LoRA r=32 on Gemma 4 E4B"
- "334 examples, 8 epochs"
- "Loss: 10.5 → 0.01"

Then grad norm plot and LR plot flash past.

**VOICE-OVER:**

> LoRA fine-tune on Kaggle's free tier. Rank 32. Eight epochs. 672 steps.
>
> Loss dropped from 10.5 to 0.01. Gradient norm stable. Four clear step-downs at epoch boundaries. Textbook convergence on a small dataset.

---

## Scene 5 — The Honest Result (1:50–2:30, 40s)

**VISUAL:** Side-by-side terminal output from `honest_compare.py`. Vanilla on left, Tuned on right. Highlight **"Option 1 / Option 2"** on vanilla in one color. Highlight **"Idea Set 1: (sub-candidates) / Idea Set 2: (sub-candidates)"** on tuned in another color.

**VOICE-OVER:**

> Here is what I actually observed, straight:
>
> At default sampling, the tuned model does not spontaneously emit the verbatim trace format. 334 examples cannot override trillions of instruction-tuning tokens outright.
>
> But look at the structure. Vanilla groups ideas as a flat Option 1, Option 2, Option 3. The tuned model groups them as **Idea Sets with multiple sub-candidates each**. That is the `Branches → Candidates` hierarchy from the training data, transferred as a thinking pattern rather than a surface template.
>
> And when you prime the prompt with even a weak format hint, the tuned model immediately produces full coherent traces. The architecture is in the weights. It's just suppressed by a stronger prior at default sampling.

---

## Scene 6 — Why It Matters (2:30–3:00, 30s)

**VISUAL:** Cut back to the architecture diagram. Then a single closing text slide:
> *"Teach the process, not the answer."*

**VOICE-OVER:**

> This is a positive result with clear scope. Pipeline works. Self-distillation works. Architecture partially transfers into weights with 334 examples — measurably. The levers for dominant transfer are identified: more data, BF16-capable hardware, one more iteration.
>
> What this gives us is a model that performs creative cognition — not one that imitates the surface of it. A Socratic co-thinker small enough to run locally. A model that questions, branches, and critiques before it answers.
>
> That's the difference between a model that teaches **what to accept**, and one that teaches **how to think**.

---

## End card (3:00)

**VISUAL:**
- Repo: `github.com/AlinBolcas/Gemma4_FineTune_Creativity`
- Track: Education
- Gemma 4 Good Hackathon

---

## Delivery notes

- **Pace:** Steady. Average 150 wpm. Don't rush. Let the architecture diagram breathe.
- **No music.** Ambient room tone only. The ideas carry the video.
- **Cut ruthlessly.** If a scene runs long, cut the explanation, not the visuals. The visuals are the proof.
- **End clean.** No "thanks for watching," no CTA. End on the thesis.

## Alternative 60-second cut

If you need a 1-minute version, keep only Scenes 2, 3, and 5 with tightened voice-over:

- Scene 2 condensed to 20s (architecture)
- Scene 3 condensed to 15s (self-distillation)
- Scene 5 condensed to 25s (the honest result + why)

That preserves the three most distinctive claims: **architecture**, **self-distillation**, **measurable partial transfer**.
