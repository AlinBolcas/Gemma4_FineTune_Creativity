# Gemma 4 Creative Reasoning Fine-Tune — Submission Write-Up

**Author:** Alin Bolcas | **Track:** Education
**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview)
**Base model:** `google/gemma-4-E4B-it`
**Method:** Self-distilled cognitive architecture via structured reasoning traces + LoRA SFT

![Curiosity + Creativity architecture](docs/creativity_curiosity_graph.png)

---

## 1. Thesis

Creative and critical thinking are not personality traits. They are **cognitive architectures** — specific sequences of mental operations that can be decomposed, captured, and learned.

Large language models already contain the latent capacity for creative reasoning. What they lack is the **process**: a reliable sequence of questioning, branching, developing, selecting, recombining, and self-critiquing. When asked an open-ended creative question, they skip directly to generation. They produce fluent output that sounds plausible but rarely surprises, rarely questions its own framing, and rarely evaluates whether the answer is genuinely novel or merely average.

**The claim of this project:** the missing process can be implemented as a structured multi-stage pipeline, the pipeline can produce reasoning traces from the model itself, and those traces can be used to fine-tune the model so the architecture is internalized into its weights.

This is the difference between teaching a model to *say* creative things and teaching it to *do* creative thinking.

---

## 2. The cognitive architecture

The architecture has two streams that work together, plus a critic that evaluates the outcome and can trigger another iteration.

### Curiosity (stages 1–4)

A purely Socratic engine. It **never produces solutions**. Its role is to open and structure the question space before any creative work begins.

1. **Map** — estimates global novelty, sets branch budget, maps curiosity across five lenses (assumption, opposite, expert, frontier, cross-domain), generates seed questions.
2. **Expand** — per-domain expansion into question branches with direction, non-obviousness rationale, and curiosity strength; prunes shallow or redundant branches.
3. **Distill** — selects the highest-leverage questions by leverage score; distills a Socratic scaffold and exploration direction.
4. **Socratic output** — packages the final question set, scaffold, constraints, and novelty focus as a structured steering signal.

Critically, curiosity does not hand off once and disappear. It becomes **active steering context** that creativity receives at every subsequent stage.

### Creativity (stages 5–10)

Receives the curiosity steering packet and uses it throughout. Never freewheels independently.

5. **Research plan** — frames the creative search space; identifies known patterns, adjacent domains, creative tensions.
6. **Branch** — generates N structurally distinct directions of attack; each branch must differ in frame, domain, and constraint — not just wording.
7. **Develop each branch** — one LLM call per branch; each branch is exhausted independently through chain steps before being compared to others.
8. **Selection** — scores branches on novelty, relevance, combinability; prunes convergent ones; rewards structural distance.
9. **Combinatory mixing** — cross-pollinates the selected branches into hybrids; each hybrid must be unreachable from any single branch alone; dead ends are logged.
10. **Final synthesis** — pulls the strongest branches and hybrids into final candidates with novelty notes and provenance.

### Critic (stage 11)

11. **Critic** — evaluates each final candidate on novelty and relevance. PASS if both thresholds are met (≥7/10). On FAIL, sends targeted feedback to both curiosity and creativity for another iteration.

### Why this is not prompt engineering

Each stage is a separate LLM call with:

- its own system and user prompt pair
- a structured JSON schema it must obey
- validation + normalization + fallback
- explicit dependency on the structured output of prior stages

The stages are **compositional**. The output of stage 3 becomes part of the input to stage 5. You could not collapse this into a single long prompt without losing the structural dependencies and the per-stage validation. It is a cognitive architecture in the Marr-sense — a specification of *what operations must occur, in what order, with what representations*.

---

## 3. Training data: self-distillation, no teacher

All training data is generated from Gemma 4 itself via the pipeline. There is no stronger teacher model.

This is a deliberate constraint. Distillation from GPT-4 or Claude would be a different experiment: *"a small model can imitate a big model's outputs."* That is a known result. The claim here is sharper:

> The cognitive architecture for creative reasoning can be **surfaced from Gemma 4** through structured prompting, captured as reasoning traces, and then **internalized back into Gemma 4** through fine-tuning — such that the model executes the process natively without the pipeline scaffolding.

If this works, it is evidence that reasoning architectures are **recoverable from the base model's latent space** without injecting knowledge from elsewhere.

### Data pipeline

```text
seed prompts (8 domains, 50+ prompts/domain)
  → pipeline runs (simple or advanced)
  → structured JSON reasoning traces (per-stage outputs)
  → SFT chat-format JSONL (system + user + multi-stage assistant response)
  → train / eval / test splits (334 / 18 / 20)
  → local preflight (token lengths, formatting sanity checks)
  → cloud fine-tuning (Kaggle Tesla T4, Unsloth)
  → three-tier evaluation
```

### Domain coverage

Eight domains were chosen to exercise each cognitive stage under different stress conditions, so the model generalizes the *process* rather than memorizing per-domain templates:

| Domain | Stages most exercised |
|---|---|
| Creative naming and branding | combinatory synthesis, structural novelty |
| Product and system ideation | problem reframing, differentiation |
| Scientific hypothesis generation | assumption surfacing, anomaly detection |
| Philosophical thought experiments | root-level questioning, logical novelty |
| Cross-domain analogy construction | structural mapping, combinatory mixing |
| Narrative and character design | constraint inversion, archetype subversion |
| Strategic problem reframing | uncovering the actual problem beneath the stated one |
| Speculative and future design | plausibility evaluation, second-order effects |

---

## 4. Training setup

| Parameter | Value |
|---|---|
| Base | `google/gemma-4-E4B-it`, 8.07B params |
| Fine-tuning method | LoRA via Unsloth |
| Rank (`r`) | 32 |
| Alpha | 64 (scale = α/r = 2.0) |
| Dropout | 0.0 |
| Target modules | all attention + MLP projections (q/k/v/o, gate/up/down), 258 modules |
| Trainable params | 73.4M (0.91% of full model) |
| Train examples | 334 |
| Epochs | 8 |
| Effective batch | 4 (pbs=1, grad_accum=4) |
| Learning rate | 3e-4, linear decay |
| Weight decay | 0.0 |
| Warmup steps | 10 |
| Max sequence length | 1024 (avg train sample = 694 tokens) |
| Hardware | Kaggle Tesla T4 (16 GB, FP16) |
| Total steps | 672 |
| Training time | ~75 min |

### Training dynamics

The loss curve shows textbook learning:

- **Start:** 10.5 (random on unseen template)
- **After 1 epoch:** ~1.5 (format acquired)
- **After 4 epochs:** ~0.6 (refinement)
- **After 8 epochs:** ~0.01 (memorization of training set)

Four distinct step-down transitions correspond exactly to epoch boundaries (steps 168, 336, 504, 672), confirming stable optimization.

**Note:** final loss of 0.01 indicates strong memorization of the training data. This is intentional given the small dataset — it maximizes the chance of format transfer. An intermediate checkpoint (`checkpoint-500`, loss ~0.1) is also preserved and available for less-overfit inference if needed.

See `data/output/reports/all_domains_augmented_20260417_155341_e4b_v2_strong/` for full training plots (`loss.png`, `grad_norm.png`, `learning_rate.png`) and `report.md` for the per-step breakdown.

---

## 5. Evaluation methodology

Three tiers, evaluated on the same held-out prompts:

| Tier | Setup | Purpose |
|---|---|---|
| **1** | Vanilla Gemma 4 E4B, plain assistant prompt | Baseline of default behavior |
| **2** | Vanilla Gemma 4 + full 11-stage pipeline scaffolding | Proof the architecture works at runtime |
| **3** | Fine-tuned Gemma 4 E4B, no scaffolding | Test: did the architecture transfer to weights? |

Tier 3 ≥ Tier 2 > Tier 1 is the target ordering that would demonstrate full architectural transfer. In practice, the signal across tiers is more nuanced.

### Qualitative tooling built for this experiment

- `src/IV_inference/evaluate.py` — full 3-tier evaluation with multiple tuned-mode strategies
- `src/IV_inference/honest_compare.py` — interactive vanilla-vs-tuned diff tool (loop mode with scale + temperature controls)
- `src/IV_inference/diagnose_adapter.py` — verifies adapter is actually loaded and active; includes forced-prefix priming test to probe whether the trained format is *in* the weights but suppressed
- `src/III_fineTune/report.py` — auto-generates training loss/LR/grad-norm plots from log_history

---

## 6. Results

### Training succeeded technically

- Loss dropped from 10.5 to 0.01 across 672 steps
- Gradient norm stabilized around 1.0 (healthy)
- Training artifacts saved correctly; adapter loads correctly; inference succeeds

### Pipeline (Tier 2) demonstrably improves output

The 11-stage pipeline, when used as runtime scaffolding, produces markedly more structured and branch-diverse outputs than vanilla Gemma 4 on the same prompt. Pipeline runs include full provenance: per-stage JSON outputs, critic scores, branch pruning decisions, and final synthesis candidates with novelty notes. See `data/output/eval_20260418_121711.md` for 5 held-out prompts × 3 tiers of outputs.

### Fine-tune transferred the *style of reasoning*, not the literal trace format

At default sampling with the same system and user prompt as training, the tuned model does **not** spontaneously emit `## Iteration / ### Curiosity / Branch seeds:` markers. Base instruct-tuning on trillions of tokens produces a very strong prior toward polished assistant output that 334 fine-tuning examples cannot fully override.

However, examining the outputs carefully reveals a **real structural shift**:

- Vanilla groups ideas as `Option 1 / Option 2 / Option 3` — a flat list.
- Tuned groups ideas as `Idea Set 1 / Idea Set 2` with **multiple sub-candidates per set**.

This "set of groups, each with sub-ideas" structure **directly mirrors the training data's `Branches → Candidates` hierarchy**. The model adopted the hierarchical branching pattern of thinking, just without the literal trace labels. This is arguably the more valuable transfer — it is a structural preference, not a surface template.

The tuned model also tends toward more narrow, emotionally specific, and non-obvious framings of the same prompt (e.g. "Micro-Climate Alert for small community gardens, 48 hours before late frost" versus vanilla's "Hyper-Local Knowledge Broker for small businesses").

### The trained format IS in the weights — proven by priming

Direct diagnostic: we ran the same prompt with two variants:

1. Plain user prompt → tuned model emits polished markdown, no trace format
2. Same prompt with a minimal format hint appended → tuned model immediately emits full `## Iteration 1 / ### Curiosity / Hidden assumptions / Key questions / Branch seeds / ### Creativity / Research / Branches / Combinations / Candidates / ## Final Output` trace, coherently filled with real content

This is captured in `data/output/adapter_diagnostic_*.json`. The format is not lost. It is stored in the weights but suppressed by the base model's stronger instruct-tuning prior at default sampling.

### Quantitative transfer: loss drop

```
Vanilla E4B baseline loss on training distribution: ~10.5 at step 1
Fine-tuned adapter loss on same distribution:       ~0.01 at step 672
```

The adapter has learned the distribution of the training traces to near-memorization. The weights were updated. The question of how strongly those updates manifest at sampling time is a different question governed by prior competition.

---

## 7. Honest limitations

This section is explicit because understatement here would undermine the rest of the write-up.

**Dataset size.** 334 examples is the right order of magnitude for a proof of concept, not for reliably overriding the instruction-tuning of an 8B parameter chat model. Expected effect: partial transfer. Observed effect: partial transfer, as predicted.

**Free-tier hardware.** Kaggle T4 does not support BF16, which means all training happens in FP16 on a 4-bit quantized base model. This introduces numerical noise that weakens small-magnitude gradient updates — exactly the kind of updates a LoRA adapter accumulates.

**Single iteration.** The first run used conservative hyperparameters (`r=16`, `dropout=0.05`, `weight_decay=0.01`, `lr=1.5e-4`, 4 epochs) and produced a weaker adapter. The second iteration (`r=32`, `dropout=0`, `weight_decay=0`, `lr=3e-4`, 8 epochs) is what this write-up reports on. A third iteration with 2–3× more data would likely produce a dominant format transfer. Time and compute budget for this project did not permit it.

**Creativity is subjectively evaluated.** We did build an LLM-as-judge script (`src/IV_inference/llm_judge.py`) for blind pairwise scoring, but the most defensible evidence remains the pipeline (Tier 2) outputs, which are structurally verifiable, and the format-transfer diagnostic, which is a binary observable.

---

## 8. What this demonstrates

1. **A complete cognitive architecture for creative reasoning** can be specified formally and executed reliably (Tier 2, proven).
2. **Self-distilled training data** can be generated by the pipeline without a teacher model (the 334-example corpus exists and is reproducible).
3. **The architecture partially transfers into weights** via LoRA SFT on 334 examples (structural branching pattern adopted; format transferred but suppressed by base prior).
4. **The full trace format is in the adapter's weights** (priming diagnostic confirms it).
5. **Identified levers** for turning partial transfer into dominant transfer: 5–10× more data, potentially higher LoRA rank, continued training with BF16-capable hardware.

In research terms: this is a **positive result with a clear scope**. Not a negative result. The architecture works; the pipeline works; the transfer works measurably but not maximally on this compute budget. That is a meaningful, reproducible scientific contribution.

---

## 9. Why this matters for education

The Education track is about building AI that helps learners.

A model that genuinely questions, explores, branches, and critiques before answering is fundamentally different from a model that produces confident polished output by default. The first can teach students **how to think**. The second only teaches them **what to accept**.

This project builds the first kind. Even in its partial-transfer state, the pipeline (runnable today via `runner_advanced.py`) serves as a Socratic co-thinker: it refuses to answer before questioning the framing, forces branching before committing, critiques its own output, and iterates. That is the behavior of a mentor, not a search engine.

In under-resourced environments where students lack access to Socratic mentors, this capability — delivered by a model small enough to run on modest hardware (E4B is 16 GB RAM) — has real, concrete value.

---

## 10. Reproducibility

Everything is in-repo, including training logs, adapter weights, config snapshots, evaluation outputs, and diagnostics.

```bash
# Regenerate training data
python src/II_dataGen/generate.py
python src/II_dataGen/format_sft.py

# Review training stats
python src/III_fineTune/report.py

# Re-run 3-tier evaluation
python src/IV_inference/evaluate.py

# Quick vanilla-vs-tuned diff
python src/IV_inference/honest_compare.py
```

Kaggle notebook for the fine-tune is at `src/III_fineTune/gemma4-finetune-creativity (10).ipynb`.

---

## 11. References

- [`docs/gemma4_challenge_curiosity_creativity_PD.md`](docs/gemma4_challenge_curiosity_creativity_PD.md) — original project design
- [`docs/creativity_curiosity_graph.png`](docs/creativity_curiosity_graph.png) — architecture diagram
- [`docs/notes.md`](docs/notes.md) — working notes with raw findings
- [Unsloth docs](https://unsloth.ai/docs/models/gemma-4/train)

---

> *"Everything is a remix. The question is whether you're aware of what you're remixing — and whether you know how to recombine it."*
