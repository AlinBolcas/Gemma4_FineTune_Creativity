# Voiceover Script — Gemma 4 Creative Reasoning Fine-Tune
# 15 segments, one per slide. Edit text here, then re-run generate_voiceover.py.

## Slide 01 — Title
Creativity has been the central focus of this project. My entry for the Gemma 4 2026 competition explores creative reasoning — specifically, the limitations current language models face when attempting to generate genuinely original ideas.

## Slide 02 — The Problem
At their core, LLMs are constrained by their training data, post-training structure, and stochastic sampling process. Through extensive experimentation over the past several years, one thing became consistently clear: language models are not inherently creative. They can recombine existing patterns remarkably well, but they rarely produce ideas that feel fundamentally novel or deeply innovative. Even for something as simple as generating an original name, they consistently fail to impress.

## Slide 03 — Two Streams
That observation led me to investigate creativity itself as a computational process. I believe this is one of the most important unsolved problems in AI. To approach it, I studied human creative reasoning from first principles — and arrived at a two-branch framework that mirrors how original ideas actually emerge in human cognition.

## Slide 04 — Curiosity
The first branch is curiosity. Curiosity can be viewed as an evolutionary mechanism — an exploratory predisposition that pushes organisms toward unknown spaces where potential opportunities may exist. Applied computationally, this becomes a process of mapping curiosity domains around an input problem: identifying all relevant conceptual areas, expanding exploratory questions within each, then distilling down to the most cognitively challenging directions. The result is a structured curiosity context designed to pressure the model away from its default, conventional paths.

## Slide 05 — Creativity
The second branch is creativity itself. It begins with grounded knowledge — facts and established principles — then branches into multiple independent lines of reasoning. An essential aspect is selective pruning. Human thinking is not just expansion — it is also elimination. We naturally abandon weak directions and reinforce promising ones. The system replicates that process by filtering branches that lack coherence, novelty, or productive momentum.

## Slide 06 — Critic
The final stage is combinatorial mixing — the core mechanism of the architecture. Creativity often emerges when ideas from unrelated domains collide and recombine into new structures. The system merges concepts across branches, producing hybrids that would never emerge from any single chain of thought. Every candidate is then scored by a critic on both novelty and relevance. On fail, the critic sends targeted feedback and the full loop runs again — refining its own output until the critic agrees, or the budget runs out.

## Slide 07 — Self-Distillation
The entire pipeline was implemented using Gemma 4. The original fine-tuning intent was self-distillation: Gemma 4 generating its own structured reasoning traces, with no external teacher. The hypothesis — if we train on traces that mimic structured creative cognition, the model should approach problems more creatively without the runtime scaffolding. But self-distillation alone wasn't enough. Around three hundred examples trained cleanly and produced no measurable behavioral shift.

## Slide 08 — Training Run
To reach scale, the same eleven-stage pipeline was run with a third-party API model. The final corpus reached four thousand seven hundred and seventy-one supervised fine-tuning examples — roughly fourteen thousand turns, across eight creative domains. That trained a LoRA adapter on a frozen Gemma 4 base: rank sixteen, alpha thirty-two — just zero point nine one percent of total parameters. Two thousand one hundred and forty-eight training steps. Four point four hours on a Kaggle Tesla T4.

## Slide 09 — Evaluation
For evaluation, three tiers were compared on a shared held-out prompt set. Tier one: vanilla Gemma 4, no scaffolding. Tier two: vanilla plus the full eleven-stage pipeline at runtime — does the architecture itself work? Tier three: the fine-tuned adapter only, no scaffolding at all — did the architecture transfer into the weights?

## Slide 10 — Result
The results are honest. At default sampling, the tuned model does not spontaneously emit the full pipeline structure — four point seven thousand examples cannot override trillions of instruction-tuning tokens. But the adapter is alive. In side-by-side tests it more consistently asks clarifying questions first, groups outputs into tighter idea sets, and remains more constraint-aware than the vanilla model. A soft transfer — but a measurable one.

## Slide 11 — Priming Unlock
The architecture is in the weights — it just needs a small invitation to surface. A plain prompt produces polished markdown with no trace structure. The same prompt with a light format hint — a brief suggestion of the expected hierarchy — and the full curiosity, creativity, and synthesis structure emerges. The format is learned. A small cue is enough to unlock it.

## Slide 12 — Temperature
Another important observation involves temperature sampling. Higher values consistently increased exploratory behavior and produced more unconventional outputs — but excessive temperature also introduced instability and incoherence. In practice, the most effective range appeared to be between zero point six and zero point eight. Below that, outputs became generic and overly deterministic. Above one point zero, coherence degraded significantly.

## Slide 13 — Why It Matters
Ultimately, this research matters because creativity sits at the center of nearly everything we want advanced intelligence systems to achieve. The moment we crack creativity, everything compounds: genuine novelty, original research, self-improving systems. A language model capable of generating truly original ideas would have enormous implications across science, engineering, art, and autonomous problem solving. This is the winner-takes-all direction in AI research.

## Slide 14 — Closing
The goal of this project was never to produce a finished model. It was to demonstrate that the architecture is teachable — that if we show a model how to think creatively, step by step, it learns to approach problems that way. Teach the process. Not just the answer.

## Slide 15 — End Card
That concludes the presentation. Thank you.
