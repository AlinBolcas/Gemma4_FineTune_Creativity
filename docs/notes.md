
# Intro
- proves it can improve itself through engineering.
-> if system engineering is the gateway to better responses, we can teach llms to engineer themselves to give on demand better responses.

#  Creative Pipeline:
- pipeline for improving on the creative domain/spectrum
- starting point system for crafting truly innovative out-of-the-box thinking ideas

# Philosophy:
- True creativity & originality is one of the biggest weakness of current llms. Possibly partially because it is rare in human thinking and written records. 
- It is also difficult to optimise for creativity, because of how difficult it is to say that something truly is creative or not. It is an illusive metric.
- Through the nature of their architecture, they average out -> less creative outputs -> converging to average responses creatively
- through a process of self-reflection I defined how we as humans may be arriving at creative ideas -  following this process of thinking: curiosity -> brnaching -> analogy -> idea landscape overlap -> connection -> original idea.
I believe that all possible ideas: discovered and undiscovered, can be brought from the unknown through combinatory action of the known and materialised into the world.
In other words, the creative act can be desconstructed and reverse engineered to a series of steps which LLMs can take - either at runtime or fine tuned on final output traces of the runtimes to improve overall creative quality & generalise towards more creative responses.

# Why
Creativity stems at the root of progress. 
If we solve this, we solve self improving, we solve superintelligence and everything that follows.
That's why this is the key to solving everything, hence why is the most impactful development & area of research.
To exemplify this, I ran Gemma4-Creative over the four judging dimention criterias and these were the output of projects it proposed would win.
This was also one of the side goals, to build a meta - model which gives the best possible ideas on what to build, but without optimising speicfically for it. 
I was seeking to find the formula for general creativity.

# Downsights:
- creativity is a very if not the most subjective field, so it's not trivial to come up with a system to score performance on that metric.
- not trained on the advanced pipeline (advanced method has proven better results, but takes 3 times longer)
- way too few sythentic data examples (limited time and hardware compute)
- there's arguably no reason to fine-tune, other than reducing latency from having to running through the pipeline at inference time - unless some generalisation occurs which actually makes the model approach ideas more creatively thanks to the fine-tunning process
- it's still inconclusive if the results are actually more creative

# Training run 1 results (e4b_final, 2026-04-18):
- training loss: 10.5 to 1.1 (min 0.98), grad norm stabilized ~2.0, 4 epochs, 336 steps
- LoRA r=16 alpha=32 dropout=0.05 on E4B base, lr 1.5e-4, gradient_accumulation=4
- post_train_eval was disabled, no held-out eval loss recorded
- adapter_diagnostic confirms adapter IS loaded and active (similarity to vanilla = 0.063, very different)

## Key finding: format IS learned, just suppressed
- in `direct_minimal` mode (tiny identity prompt), tuned model emits ZERO trace markers - reverts to base "polished assistant" style
- in `direct_primed` mode (prompt invites the format), tuned model emits ALL trace markers: `## Iteration`, `### Curiosity`, `### Creativity`, `Branch seeds:`, etc.
- conclusion: the architecture DID transfer into the weights. The base model's strong default style suppresses it at sampling time. A small invitation in the prompt unlocks it.

## What this means
- no need to retrain just to fix this
- recommended inference mode: `direct_primed` in src/IV_inference/evaluate.py
- if we DO retrain later, levers to make format dominant without priming: higher LoRA rank (r=32/64), more train samples, fewer competing assistant-style examples in the dataset

## Next eval steps
- re-run `evaluate.py` with `direct_primed` so Tier 3 actually shows the full architecture
- run `eval_loss.py` for held-out perplexity (vanilla vs tuned) - quantitative confirmation
- optional: `llm_judge.py` for blind LLM-judge scoring on novelty / non-obviousness / process-visibility