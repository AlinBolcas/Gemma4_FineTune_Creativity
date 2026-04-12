# Gemma 4 Creative Reasoning Pipeline — Project Document

**Competition:** [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview) | Kaggle x Google DeepMind | Deadline: May 18, 2026

---

## Core Idea

Creative cognition is not a single operation. It is a sequence of distinct cognitive acts:

- questioning the problem before engaging with it
- mapping unexplored territory
- branching into structurally different directions
- developing each direction independently
- selecting and recombining the strongest
- evaluating novelty and relevance critically

Language models do not perform these steps by default. They generate. This project gives Gemma 4 the full cognitive process by implementing it as a structured multi-stage pipeline, generating reasoning traces from Gemma 4 itself, and fine-tuning on those traces so the model internalizes the architecture into its weights.

The result should be a model that does not just produce creative-sounding text. It performs a creative cognitive process.

---

## The Two Streams

The pipeline is divided into two parallel streams that work in concert.

### Curiosity

A purely Socratic engine. It never generates solutions. Its job is to open and structure the question space before any creative work begins.

Stages:

1. **Map** — estimates global novelty, sets branch budget, maps curiosity domains across five lenses (assumption, opposite, expert, frontier, cross-domain), generates seed questions
2. **Expand** — per domain, expands into question branches with direction, non-obviousness rationale, and curiosity strength; prunes shallow or redundant branches
3. **Distill** — selects the highest-leverage questions by leverage score; distills a socratic scaffold and exploration direction
4. **Socratic output** — packages the final question set, scaffold, constraints, and novelty focus as a structured steering signal

The curiosity stream does not hand off once and disappear. It becomes active steering context that creativity receives at every subsequent stage.

### Creativity

Receives the curiosity steering packet and uses it throughout. Never freewheels independently.

Stages:

5. **Research plan** — uses curiosity steering to frame the creative search space; identifies known patterns, adjacent domains, and creative tensions
6. **Branch** — generates N structurally distinct directions of attack; each branch must differ in frame, domain, and constraint — not just wording
7. **Develop each branch** — one LLM call per branch; each branch is exhausted independently through chain steps before being compared
8. **Selection** — scores each developed branch on novelty, relevance, and combinability; prunes convergent ones; rewards structural distance
9. **Combinatory mixing** — cross-pollinates the selected branches into hybrids; each hybrid must be unreachable from any single branch alone; dead ends are logged
10. **Final synthesis** — pulls the strongest branches and hybrids into final candidates with novelty notes and provenance

### Critic

11. **Critic** — evaluates each final candidate on novelty and relevance; PASS if both thresholds are met (≥7/10); on FAIL, sends targeted feedback to both curiosity and creativity for the next iteration

---

## Architecture Diagram

See `docs/creativity_curiosity_graph.png`.

Key structural rule visible in the diagram:

> Curiosity frames the question space. Creativity uses it as steering context throughout. The dotted handoff arrow is labelled "steering" — not "input."

---

## Implementation

Each of the 11 stages above is a separate LLM call with its own:

- system prompt
- user prompt builder (takes prior stage outputs as structured context)
- JSON schema and normalization
- validation with fallback

This is what makes it a cognitive architecture rather than a long prompt. The stages are compositional. Each one genuinely depends on the structured output of the previous ones.

### Files

| File | Purpose |
|---|---|
| `src/I_pipeline/runner.py` | Simple 3-stage loop |
| `src/I_pipeline/runner_advanced.py` | Full 11-stage pipeline |
| `src/I_pipeline/prompts.py` | Simple prompt set |
| `src/I_pipeline/prompts_advanced.py` | Per-stage prompt set for the advanced runner |
| `src/I_pipeline/schema.py` | Simple schemas and normalization |
| `src/I_pipeline/schema_advanced.py` | Per-stage schemas, normalization, fallbacks, compatibility summaries |
| `src/II_dataGen/generate.py` | Batch pipeline runs across seed prompt domains |
| `src/II_dataGen/format_sft.py` | Converts traces to chat-format SFT JSONL |
| `src/III_fineTune/sft_train.py` | Run config builder, preflight, cloud training handoff |
| `src/IV_inference/evaluate.py` | 3-tier vanilla vs scaffolded vs fine-tuned evaluation |
| `src/IV_inference/gemma4_integration.py` | Hugging Face Gemma 4 integration |
| `src/IV_inference/ollama_integration.py` | Ollama local inference |
| `src/V_utility/export.py` | Markdown export for pipeline outputs |
| `src/app.py` | Gradio demo UI |

---

## Training Data Strategy

All synthetic training data is generated from Gemma 4 itself.

This is a deliberate constraint. Using a stronger teacher model would be knowledge distillation — a known and valid technique, but not what this project proves.

The claim here is different:

> The cognitive architecture for creative reasoning can be surfaced from Gemma 4 via structured prompting, captured as structured traces, and then internalized back into Gemma 4 through fine-tuning.

The base model already contains latent creative capacity. The pipeline surfaces it. The traces capture it. Fine-tuning internalizes it so the model executes the process natively, without external orchestration.

### Data flow

```text
seed prompts (8 domains)
  -> pipeline runs (simple or advanced)
  -> structured JSON reasoning traces
  -> SFT chat-format JSONL
  -> train / eval / test splits
  -> local preflight (token lengths, formatting)
  -> cloud fine-tuning (Kaggle or Colab, Unsloth)
  -> 3-tier evaluation
```

### Domain coverage

Domains are chosen to maximally exercise each cognitive stage:

| Domain | Exercises most |
|---|---|
| Creative naming and branding | combinatory synthesis, structural novelty |
| Product and system ideation | problem reframing, differentiation |
| Scientific hypothesis generation | assumption surfacing, anomaly detection |
| Philosophical thought experiments | root-level questioning, logical novelty |
| Cross-domain analogy construction | structural mapping, combinatory mixing |
| Narrative and character design | constraint inversion, archetype subversion |
| Strategic problem reframing | uncovering the actual problem beneath the stated one |
| Speculative and future design | plausibility evaluation, second-order effects |

A model trained across these eight domains generalizes the reasoning pattern, not the domain content.

---

## What Fine-Tuning Is Meant to Achieve

After fine-tuning, the model should execute the cognitive architecture natively:

1. map the problem space before engaging
2. surface assumptions and overlooked domains
3. generate structurally distinct branches
4. develop each branch independently to its limit
5. select based on novelty and combinability, not familiarity
6. cross-pollinate survivors into novel hybrids
7. evaluate its own output critically and honestly

This is not a stylistic change. It is an architectural one. The behavior should emerge without external scaffolding.

---

## Evaluation

Three-tier benchmark against the same held-out prompts:

| Tier | Setup | Expected |
|---|---|---|
| 1 | Vanilla Gemma 4 | Baseline |
| 2 | Gemma 4 + pipeline scaffolding | Better |
| 3 | Fine-tuned Gemma 4, no scaffolding | Best |

Tier 3 > Tier 2 > Tier 1 = the cognitive architecture transferred into the weights.

Demo story: same prompt, three tiers, compare originality, branch diversity, self-critique quality, and final idea quality side by side.

---

## Competition Fit

**Track:** Education (primary), Global Resilience (secondary)

**Why Education:** Students and creators globally lack access to Socratic mentors and creative collaborators. A model that genuinely questions, explores, recombines, and critiques before answering can serve as both. It teaches how to think, not what to think.

**Technical differentiator:** Training on full multi-stage cognitive traces, not input/output pairs. The model learns the process of creative thinking, not just its surface outputs.

**What makes the submission credible:** A complete experiment pipeline from inference through data generation, SFT formatting, preflight, cloud training, and evaluation. Reproducible. Benchmarkable. Grounded in a real cognitive model, not prompt tuning.

---

## Milestones

| Step | Status |
|---|---|
| Simple pipeline built and validated | done |
| Advanced 11-stage pipeline built | done |
| Upstream compatibility (data, SFT, eval) | done |
| Gradio demo app | done |
| Generate advanced dataset at scale | next |
| Fine-tune on Kaggle | next |
| Run 3-tier evaluation | next |
| Prepare submission | deadline May 18, 2026 |

---

## Key References

- https://unsloth.ai/docs/models/gemma-4/train
- https://www.kaggle.com/code/kodatirevanth/gemma-4-the-complete-starter-guide-by-claude
- `docs/curiosity-creativity.jsx` — diagram source
- `docs/resources.md` — supporting datasets

> *"Everything is a remix. The question is whether you're aware of what you're remixing."*
