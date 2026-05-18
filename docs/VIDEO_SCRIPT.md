# Video Script — Gemma 4 Creative Reasoning Fine-Tune

**Length:** ~3 minutes
**Tone:** Direct. Technical. No fluff, no hype music.
**Voice-over:** Calm, precise.
**Visual style:** Slide deck + occasional terminal cut-ins. Loss curve at the training scene.

---

## Scene 1 — The Problem (0:00 – 0:25)

**VISUAL:** Title slide. Then thesis slide.

**VOICE-OVER:**

> Creativity is where current language models are at their weakest. Their outputs are bounded by training data and stochastic sampling, and in five years of testing them, I haven't seen one produce a truly original idea — not even for something as small as a good name.
>
> That's the problem I wanted to solve for the Gemma 4 entry.

---

## Scene 2 — The Architecture (0:25 – 1:05)

**VISUAL:** Two-streams slide, then curiosity slide, then creativity slide.

**VOICE-OVER:**

> So I reverse-engineered my own creative process. It breaks down to two streams.
>
> **Curiosity** opens the question space. It's a Socratic engine — it never answers. It maps the problem across five lenses, expands question branches, prunes the weak ones, and distils the strongest into a steering signal.
>
> **Creativity** receives that signal. It branches into structurally distinct directions, develops each independently, prunes the dead ends, then cross-pollinates the survivors into hybrids. That recombination step is the heart of it — the same way DNA from two parents mixes into something genuinely new.
>
> A **critic** scores every candidate. On fail, it sends feedback back to both streams and the loop runs again.
>
> Eleven stages in total. Each one is a separate LLM call with its own schema and validation. This is a reasoning system, not a prompt.

---

## Scene 3 — Self-Distillation, Honestly (1:05 – 1:35)

**VISUAL:** Data slide with the self-distillation loop diagram.

**VOICE-OVER:**

> Then I used the pipeline to generate training data — Gemma 4 running through the full loop on seed prompts across eight creative domains.
>
> Pure self-distillation gave me around 300 traces. That trained cleanly but it didn't shift the model in any noticeable way. To reach a volume where fine-tuning has a measurable effect, I re-ran the same pipeline prompts through a third-party API model. The final dataset is around 4,700 SFT examples.
>
> The claim still holds: the architecture is recoverable from Gemma 4. What needs scale is internalising it.

---

## Scene 4 — Training (1:35 – 2:00)

**VISUAL:** Training slide. Loss curve animates from 2.7 down to ~0.56.

**VOICE-OVER:**

> LoRA fine-tune on Kaggle's free T4. Rank 16, 73 million trainable parameters — under one percent of the base model. Two epochs, two thousand one hundred steps, about four and a half hours.
>
> Loss settles around 0.56. The run is stable and the adapter doesn't collapse into repetition. That was the bar.

---

## Scene 5 — The Honest Result (2:00 – 2:40)

**VISUAL:** Three-tier slide, behavioral-shift slide, priming-unlock slide.

**VOICE-OVER:**

> Three-tier evaluation. Vanilla, vanilla plus pipeline, and the fine-tuned model on its own.
>
> Tier 2 works clearly — the pipeline visibly improves output quality at runtime.
>
> Tier 3 is softer. At default sampling the tuned model doesn't spontaneously emit the verbatim trace format. ~4,700 examples can't outright override trillions of instruction-tuning tokens. But the adapter is alive: it asks clarifying questions more often, groups output into tighter idea sets, and stays more constraint-aware than vanilla.
>
> And when you prime the prompt with a light format hint, the full trace structure comes back coherently. So the architecture is in the weights — it's just suppressed by a stronger prior in free generation.

---

## Scene 6 — Why It Matters (2:40 – 3:00)

**VISUAL:** Why-it-matters slide, then closing slide.

**VOICE-OVER:**

> Creativity is the lock. The moment we crack it, everything compounds — genuine novelty, real research, self-improving systems. That makes it winner-takes-all, and the most impactful direction to push on.
>
> The deeper goal: a model that questions, branches, and critiques before answering. One that teaches *how to think* — not just *what to accept*.

---

## End card (3:00)

**VISUAL:** End card slide. Repo, track, hackathon.

---

## Delivery notes

- **Pace:** steady, ~150 wpm. Let the visuals breathe.
- **No music.** Room tone only.
- **Cut ruthlessly.** If a scene runs long, cut the explanation, not the visuals.
- **End clean.** No "thanks for watching." End on the closing line.
