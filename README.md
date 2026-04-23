# Gemma 4 Creative Reasoning Fine-Tune

**[Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview)** | Kaggle × Google DeepMind

![Curiosity + Creativity architecture](docs/creativity_curiosity_graph.png)

> Teaching Gemma 4 to **think** creatively, not just **answer** creatively.

## The idea

Current LLMs do not lack knowledge. They lack **process**.

Given an open-ended creative task, most models skip straight to generation. They do not question the framing. They do not explore structurally different directions. They do not evaluate whether their output is actually novel or just fluent.

This project implements a **cognitive architecture for creative reasoning** as an 11-stage pipeline, generates reasoning traces from Gemma 4 using itself, and fine-tunes on those traces so the process is internalized into the weights.

Two streams, working together:

- **Curiosity** — a Socratic engine. Never answers. Maps the problem space, surfaces hidden assumptions, expands question branches, prunes the weak ones, distills the strongest into a steering signal.
- **Creativity** — receives the steering signal. Builds a research plan, generates structurally distinct branches, develops each independently, selects the strongest, cross-pollinates them into hybrids, synthesizes final candidates.
- **Critic** — evaluates every candidate on novelty and relevance. On FAIL, sends targeted feedback back to both streams and the loop runs again.

It is a reasoning system, not a prompt. Each stage is a separate LLM call with its own schema, validation, and fallbacks.

## Architecture

```text
CURIOSITY STREAM                        CREATIVITY STREAM

1. Map curiosity domains
2. Expand question branches
3. Distill question set
4. Socratic output ──── steering ────►  5. Research plan
                                        6. Branch
                                        7. Develop each branch
                                        8. Selection + pruning
                                        9. Combinatory mixing
                                       10. Final synthesis
                                       11. Critic → loop if FAIL
```

Curiosity frames the question space. Creativity uses it as active steering context throughout every stage.

## Repo structure

```text
src/
  I_pipeline/       prompts, schemas, runners (simple + 11-stage advanced)
  II_dataGen/       dataset generation, SFT formatting, train/eval/test splits
  III_fineTune/     Unsloth LoRA workflow, local preflight, Kaggle handoff
  IV_inference/     Gemma 4 wrappers (HF + Ollama), evaluation tooling
  V_utility/        markdown export, helpers
  app.py            Gradio demo UI

data/
  input/            seed prompts, SFT datasets, train/eval/test splits
  output/           pipeline runs, eval results, model artifacts

docs/               architecture diagrams, project document, notes

WRITEUP.md          Full scientific write-up for submission judges
VIDEO_SCRIPT.md     3-minute demo video script
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/I_pipeline/runner.py            # simple pipeline (3 stages)
python src/I_pipeline/runner_advanced.py   # full 11-stage pipeline
python src/II_dataGen/generate.py          # generate training traces
python src/II_dataGen/format_sft.py        # convert to SFT JSONL
python src/III_fineTune/sft_train.py       # preflight + run config
python src/IV_inference/evaluate.py        # 3-tier evaluation
python src/IV_inference/honest_compare.py  # vanilla vs tuned, interactive
python src/app.py                          # Gradio demo
```

All scripts are interactive terminal UIs. No CLI flags.

## Training

| Property | Value |
|---|---|
| Base model | `google/gemma-4-E4B-it` |
| Method | LoRA (Unsloth), `r=32`, `alpha=64`, `dropout=0.0` |
| Data | 334 self-generated reasoning traces, 8 domains |
| Epochs | 8 |
| Learning rate | 3e-4, linear decay |
| Hardware | Kaggle Tesla T4 (16 GB) |
| Final train loss | ~0.01 (from 10.5) |

The model is distilled from **itself** through the pipeline. No teacher model. The claim is not that Gemma 4 learns something new — it is that Gemma 4 already contains latent creative reasoning capacity, and this project makes it executable natively.

## Findings

The full analysis is in [`WRITEUP.md`](WRITEUP.md). Summary:

- **Pipeline (Tier 2) works.** The 11-stage architecture measurably changes output quality versus vanilla prompting.
- **Fine-tune transferred the *style of reasoning*, not the literal trace format.** Outputs from the tuned model are more structurally categorical, using hierarchical "idea sets" with sub-candidates — mirroring the "Branch → Candidates" pattern from training — without verbatim trace markers.
- **Priming unlocks the full format.** When the prompt even weakly invites the trained structure, the tuned model produces complete `## Iteration / ### Curiosity / ### Creativity` traces. The format is in the weights; base instruction-tuning is just a stronger prior at default sampling.

## Competition framing

**Track:** Education

**Thesis:** Creative and critical thinking are not personality traits. They are cognitive architectures that can be learned. This project teaches Gemma 4 a structured creative reasoning process, then fine-tunes it so the model thinks this way natively.

**Differentiator:** Training on full multi-stage cognitive traces, not input/output pairs. The model learns the *process* of creative thinking, not just its surface.

## License

MIT. See `LICENSE`.
