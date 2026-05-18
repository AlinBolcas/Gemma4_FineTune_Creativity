# Gemma 4 Creative Reasoning Fine-Tune — Submission Write-Up

**Author:** Alin Bolcas · [Arvolve](https://arvolve.ai)
**Track:** Education → Future of Education
**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview) · Kaggle × Google DeepMind · 2026
**Base model:** `google/gemma-4-E4B-it` (8.07B)
**Method:** Structured 11-stage cognitive pipeline + LoRA SFT on self-distilled + API-scaled reasoning traces

### Quick Links

| | |
|---|---|
| 📹 **Demo Video** | [YouTube — 3-min walkthrough](https://youtu.be/PLACEHOLDER) |
| 💻 **GitHub Repo** | [AlinBolcas/Gemma4_FineTune_Creativity](https://github.com/AlinBolcas/Gemma4_FineTune_Creativity) |
| 📓 **Training Notebook** | [Kaggle — LoRA fine-tune on Kaggle T4](https://www.kaggle.com/code/PLACEHOLDER) |
| 🤗 **Model Adapter** | [HuggingFace — LoRA adapter weights](https://huggingface.co/PLACEHOLDER) |

---

## 1. Thesis

Creativity is where current LLMs are at their weakest. Their outputs are bounded by training data and stochastic sampling. After five years of testing them, I haven't seen one reliably produce a truly original idea — not even for something as small as a good name.

Creative and critical thinking are not personality traits. They are **cognitive architectures** — specific sequences of mental operations that can be decomposed, captured, and learned. LLMs already contain the latent capacity. What they lack is the **process**: a reliable sequence of questioning, branching, developing, selecting, recombining, and self-critiquing. When asked an open-ended creative question, they skip directly to generation.

**The claim of this project:** the missing process can be implemented as a structured multi-stage pipeline, the pipeline can produce reasoning traces from the model itself, and those traces can be used to fine-tune the model so part of the architecture is internalised into the weights.

This is the difference between teaching a model to *say* creative things and teaching it to *do* creative thinking.

---

## 2. The cognitive architecture

![Eleven stages, two streams, one critic](data/output/visuals/v4/diag_pipeline_flow.png)

Two streams that work together, plus a critic that closes the loop.

### Curiosity (stages 1–4)

A purely Socratic engine. It **never produces solutions**. Its role is to open and structure the question space before any creative work begins.

1. **Map** — estimates novelty, sets branch budget, maps the problem across five lenses (assumption, opposite, expert, frontier, cross-domain), generates seed questions
2. **Expand** — per-domain expansion into question branches with direction and non-obviousness rationale; prunes shallow or redundant branches
3. **Distill** — leverage-scores the questions, keeps the highest-impact set
4. **Socratic output** — packages the question set, scaffold, constraints, and novelty focus as a structured steering signal

Critically, curiosity does not hand off once and disappear. It becomes **active steering context** that creativity receives at every subsequent stage.

### Creativity (stages 5–10)

Receives the curiosity packet and uses it throughout. Never freewheels independently.

5. **Research plan** — frames the creative search space; identifies adjacent domains and creative tensions
6. **Branch** — generates N structurally distinct directions; each branch must differ in frame, domain, and constraint — not just wording
7. **Develop each branch** — one LLM call per branch; exhausted independently before comparison
8. **Selection** — scores branches on novelty, relevance, combinability; prunes convergents; rewards structural distance
9. **Combinatory mixing** — cross-pollinates survivors into hybrids; each hybrid must be unreachable from any single branch
10. **Final synthesis** — pulls the strongest branches and hybrids into final candidates with provenance

The combinatory mixing step is the heart of it — analogous to the way DNA from two parents recombines into something genuinely new, or the way different cortical regions bridge signals to produce ideas no single chain of thought could reach.

### Critic (stage 11)

11. **Critic** — evaluates each final candidate on novelty and relevance. PASS if both thresholds are met (≥ 7/10). On FAIL, sends targeted feedback to both curiosity and creativity for another iteration.

### Why this is not prompt engineering

Each stage is a separate LLM call with:

- its own system + user prompt pair
- a structured JSON schema it must obey
- validation, normalisation, and fallback
- explicit dependency on the structured output of prior stages

The stages are **compositional**. You can't collapse this into a single long prompt without losing the structural dependencies and per-stage validation. It is a cognitive architecture in the Marr sense — a specification of *what operations occur, in what order, with what representations*.

---

## 3. Training data: self-distilled seed, scaled with external generation

![Data pipeline: seed prompts → pipeline runs → SFT dataset → LoRA train → adapter](data/output/visuals/info/info_data_flow.png)

The original intent was full self-distillation: Gemma 4 generating its own training traces via the pipeline, no teacher model. That constraint matters because distillation from a stronger teacher is a different (and weaker) claim — *"a small model can imitate a big model's outputs"* is a known result.

The sharper claim is:

> The cognitive architecture for creative reasoning can be **surfaced from Gemma 4** through structured prompting, captured as reasoning traces, and then **internalised back into Gemma 4** through fine-tuning, such that the model shows measurable structural transfer even without the runtime scaffolding.

**What actually happened:** pure self-distillation produced ~300 examples. That trained cleanly but produced no measurable behavioural shift — 300 examples is insufficient to override a model carrying this much instruction tuning. To reach the scale where fine-tuning has a visible effect, the same 11-stage pipeline prompts were re-run with a third-party API model. The pipeline structure stayed identical; only the model behind each call changed.

### Final dataset

| | Count |
|---|---|
| Raw reasoning traces generated | 5,000 |
| SFT examples (chat-format JSONL) | 4,771 |
| Train / eval / test split | 4,293 / 238 / 240 |
| Total messages across SFT (avg ~3 per example) | 14,313 |
| Domains | 8 |

### Data flow

```text
seed prompts (8 domains)
  → pipeline runs (Gemma seed, then API-scaled, same prompts)
  → structured JSON reasoning traces (per-stage outputs)
  → SFT chat-format JSONL (system + user + multi-stage assistant)
  → train / eval / test (90 / 5 / 5)
  → local preflight (token lengths, formatting sanity)
  → cloud fine-tuning (Kaggle Tesla T4, Unsloth)
  → three-tier evaluation
```

### Domain coverage

Eight domains, chosen to stress different cognitive stages so the model generalises the *process* rather than memorising per-domain templates:

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

![LoRA: a small adapter ring on a frozen base](data/output/visuals/v4/diag_lora_adapter.png)

| Parameter | Value |
|---|---|
| Base | `google/gemma-4-E4B-it`, 8.07B params |
| Fine-tuning method | LoRA via Unsloth |
| Rank (`r`) | 16 |
| Alpha | 32 (scale = α/r = 2.0) |
| Dropout | 0.05 |
| Target modules | all attention + MLP projections (q/k/v/o, gate/up/down), 258 modules |
| Trainable params | 73.4M (0.91% of full model) |
| Train examples | 4,293 (eval 238, test 240) |
| Epochs | 2 |
| Effective batch | 4 (pbs=1, grad_accum=4) |
| Learning rate | 1e-4, cosine decay |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Max sequence length | 1536 |
| Hardware | Kaggle Tesla T4 (16 GB, FP16) |
| Total steps | 2,148 |
| Training time | ~4.4 h |

### Training dynamics

- **Start:** ~2.72 loss
- **Early convergence:** quickly drops under 1.0
- **Main plateau:** stabilises around 0.45 – 0.6
- **Final train loss:** ~0.56
- **Min loss observed:** ~0.40

Gradient norms settle near 0.5 – 0.8 for most of training. The run avoids the repetition-collapse behaviour seen in earlier overfit attempts.

One unexplained anomaly: the final logged step (step 2,148) shows a loss spike to 2.01 and grad-norm jump to 4.21 after the rest of training had settled. The run didn't diverge overall and the saved adapter behaves normally, but this end-of-run spike is noted.

One caveat: there is **no eval loss** for this run because post-train eval was disabled, so the evidence is training stability plus downstream output behaviour, not train-vs-eval separation.

See `data/output/reports/all_domains_20260427_202423_e4b_v4/` for plots and per-step breakdown.

---

## 5. Evaluation methodology

![Three evaluation tiers](data/output/visuals/info/info_three_tiers.png)

Three tiers, evaluated on the same held-out prompts:

| Tier | Setup | Purpose |
|---|---|---|
| **1** | Vanilla Gemma 4 E4B, plain assistant prompt | Baseline of default behaviour |
| **2** | Vanilla Gemma 4 + full 11-stage pipeline scaffolding | Proof the architecture works at runtime |
| **3** | Fine-tuned Gemma 4 E4B, no scaffolding | Test: did the architecture transfer to weights? |

Tier 3 ≥ Tier 2 > Tier 1 is the target ordering that would demonstrate full architectural transfer. In practice, the signal across tiers is more nuanced — see results.

### Tooling built for this experiment

- `src/IV_inference/evaluate.py` — full 3-tier evaluation with multiple tuned-mode strategies
- `src/IV_inference/honest_compare.py` — interactive vanilla-vs-tuned diff (scale + temperature controls)
- `src/IV_inference/diagnose_adapter.py` — verifies adapter is loaded and active; includes forced-prefix priming test
- `src/III_fineTune/report.py` — auto-generates training plots from log_history

---

## 6. Results

### Training succeeded technically

- Loss dropped from 2.72 to 0.56 across 2,148 steps
- Gradient norms stable after the opening phase
- Adapter loads correctly; inference succeeds
- Diagnostic confirms the adapter is active, not silent (similarity to vanilla = 0.063 in deep weight comparison)

### Pipeline (Tier 2) demonstrably improves output

The 11-stage pipeline, used as runtime scaffolding, produces measurably more structured and branch-diverse outputs than vanilla. Pipeline runs include full provenance: per-stage JSON outputs, critic scores, branch pruning decisions, and final synthesis candidates. See `data/output/eval_*.md` for held-out prompts × 3 tiers.

### Fine-tune transferred a softer structural bias, not the literal trace format

![Behavioral shift](data/output/visuals/v4/diag_behavioral_shift.png)

At default sampling on ordinary prompts, the tuned model does **not** spontaneously emit `## Iteration / ### Curiosity / Branch seeds:` markers. The base model's instruct-tuning prior is strong enough that ~4,700 fine-tuning examples don't fully override it.

The adapter is not inert though. Across repeated `honest_compare.py` runs at 1× scale:

- the tuned model more often **asks clarifying questions before answering**
- it groups outputs into tighter **idea-set bundles**
- it mirrors the training hierarchy **without using verbatim trace labels**

This is real but soft transfer. A side-by-side reader can usually detect that the tuned model is different — not always that it is better.

**One important boundary:** at 2× sampling scale the adapter degenerates into repetition loops (garbled phrases, near-verbatim sentence duplication). The adapter holds at 1×; everything in this write-up is reported at 1×.

### The trained format is in the weights, but only clearly under priming

![Priming unlock](data/output/visuals/v4/diag_priming_unlock.png)

Direct diagnostic, same prompt with two variants:

1. Plain prompt → tuned model emits polished markdown, no trace format
2. Same prompt + minimal format hint → tuned model immediately emits the full `## Iteration / ### Curiosity / Hidden assumptions / Key questions / Branch seeds / ### Creativity / Research / Branches / Combinations / Candidates / ## Final Output` trace, coherently filled with real content

Captured in `data/output/adapter_diagnostic_*.json`. The format isn't lost. It's encoded in the weights and suppressed by the base model's stronger prior during free generation.

### Temperature is a meaningful lever

![Temperature spectrum](data/output/visuals/v4/diag_temp_spectrum.png)

Temperature has a clear effect on output character, separate from anything the fine-tune does. In the range **0.6 – 0.8**, outputs are measurably more exploratory without becoming noisy. Above 1.0 becomes erratic. At 0 the model collapses to the most generic baseline.

This is consistent across both vanilla and tuned models. Temperature isn't creativity — it's a noise parameter that expands the sampling space — but it interacts well with the pipeline's branch-and-select stages, where broader initial sampling gives selection and pruning more to work with.

### Quantitative summary

- **Adapter is active.** Diagnostic similarity to vanilla = 0.063 (very different).
- **Free-generation shift is measurable but small.** Character-level diff between vanilla and tuned outputs on the same prompt is ~4.4%.
- **Trace markers under priming: 100% recovery.** All five markers appear in the tuned model's primed output; zero appear in free generation of either model.
- **No repetition collapse at 1×.** A clean improvement over earlier overfit runs.
- **Unexplained end-of-run loss spike at step 2,148.** Run did not diverge overall; saved adapter behaves normally.

The adapter learned something real, but the size of the free-generation shift reflects how much of it is suppressed by the base model's stronger prior.

---

## 7. Honest limitations

This section is explicit because understatement here would undermine the rest of the write-up.

**Self-distillation alone wasn't enough.** Pure Gemma-distilled data (~300 examples) didn't shift the model. Scaling required external API generation. The architecture-transfer claim is still meaningful — the pipeline + prompts are the same in both cases — but the cleanest version of the experiment (Gemma producing all its own training data) is not what was actually trained on.

**Free-tier hardware.** Kaggle T4 does not support BF16. Training runs in FP16 on a 4-bit quantised base, which introduces numerical noise that weakens small-magnitude gradient updates — exactly the kind a LoRA adapter accumulates.

**Conservative training trade-off.** An earlier stronger run overfit badly and produced repetition collapse at inference. The v4 run fixes that by training more conservatively (`r=16`, `dropout=0.05`, `weight_decay=0.01`, `lr=1e-4`, 2 epochs), but the resulting adapter is gentler. So the project found a workable middle point, not a perfect one.

**Adapter fragility at 2× scale.** As noted, the adapter holds at 1× but breaks at 2×. All reported results are at 1×.

**Creativity is subjectively evaluated.** An LLM-as-judge script exists (`src/IV_inference/llm_judge.py`) for blind pairwise scoring, but the most defensible evidence remains the Tier 2 outputs (structurally verifiable) and the format-transfer diagnostic (binary observable).

---

## 8. What this demonstrates

1. **A complete cognitive architecture for creative reasoning** can be specified formally and executed reliably (Tier 2, proven).
2. **The pipeline can generate training data from Gemma 4 itself.** Scaling to sufficient volume required external API traces; the pure self-distilled set alone did not produce measurable transfer.
3. **The architecture partially transfers into weights** via LoRA SFT on ~4,700 examples, but mostly as a soft structural bias rather than full verbatim trace emission.
4. **The full trace format can be recovered under priming** even when it doesn't appear in free generation.
5. **Identified levers** for stronger transfer: more data, slightly stronger adapter rank, better eval infrastructure to stop between under-transfer and overfit.

In research terms: a **positive but scoped result**. Architecture works, pipeline works, transfer is measurable but incomplete on this compute budget.

---

## 9. Why this matters for education

![The creative mind — what we're teaching the model to become](data/output/visuals/abstract/abstract_creative_mind.png)

Creativity isn't a peripheral capability. It's the most impactful unsolved problem in LLM development — the lock to genuine self-improvement and the differentiator between a model that recombines what it knows and one that generates something truly new. Crack it, and the implications compound across every domain. That makes it a winner-takes-all direction, not a niche research topic.

For education specifically: a model that questions, explores, branches, and critiques before answering is fundamentally different from a model that produces confident polished output by default. The first teaches **how to think**. The second only teaches **what to accept**.

This project builds the first kind. Even in its partial-transfer state, the pipeline (runnable today via `runner_advanced.py`) serves as a Socratic co-thinker: it refuses to answer before questioning the framing, forces branching before committing, critiques its own output, and iterates. The direct fine-tuned model does this more softly, but the scaffolded system does it clearly.

In environments where Socratic mentors are scarce — and at a model size small enough to run on modest hardware (E4B is 16 GB RAM) — that has real, concrete value.

---

## 10. Reproducibility

Everything is in-repo: training logs, adapter weights, config snapshots, evaluation outputs, diagnostics.

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

Kaggle notebook for the fine-tune: `src/III_fineTune/kaggle_fineTune_notebook.ipynb`.

---

## 11. References

- [`docs/gemma4_challenge_curiosity_creativity_PD.md`](docs/gemma4_challenge_curiosity_creativity_PD.md) — project design
- [`docs/creativity_curiosity_graph.png`](docs/creativity_curiosity_graph.png) — architecture diagram
- [`docs/notes.md`](docs/notes.md) — working notes and raw findings
- [Unsloth docs](https://unsloth.ai/docs/models/gemma-4/train)

---

> *"Everything is a remix. The question is whether you're aware of what you're remixing — and whether you know how to recombine it."*
