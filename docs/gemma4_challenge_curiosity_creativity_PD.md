# PSYCHE — Fine-Tuning Project Specification
### Gemma 4 Good Hackathon | Kaggle × Google DeepMind | Deadline: May 18, 2026
https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview
https://unsloth.ai/docs/models/gemma-4/train
https://colab.research.google.com/github/unslothai/unsloth/blob/main/studio/Unsloth_Studio_Colab.ipynb
https://github.com/unslothai/unsloth

https://www.kaggle.com/code/kodatirevanth/gemma-4-the-complete-starter-guide-by-claude

https://unsloth.ai/docs/models/gemma-4
https://unsloth.ai/docs/models/gemma-4#run-gemma-4-tutorials
https://unsloth.ai/docs/models/gemma-4/train

---

## Executive Summary

PSYCHE is a creative ideation pipeline and fine-tuning project for Gemma 4. It produces a fine-tuned model that is genuinely more curious, more creative, and more critically rigorous than vanilla Gemma 4 — by training it on synthetic reasoning chains generated through structured prompt engineering on the base model itself.

The pipeline consists of three cognitive modules — **Curiosity**, **Creativity**, and **Critic** — running in a closed feedback loop until the Critic deems the output original and satisfactory. Fine-tuning then distils this looping process into the model's weights, so it reasons this way natively without external scaffolding.

**Competition framing:** Democratising creative and critical thinking for students and educators globally — particularly in under-resourced environments lacking access to creative mentorship or Socratic teaching.

---

## Core Thesis

Standard LLMs underperform on genuine creativity because they converge too fast. They answer before questioning, generate before exploring, and accept before critiquing. PSYCHE trains Gemma 4 to slow down, branch wide, and think hard before it concludes.

```
CURIOSITY  →  surfaces what isn't known yet  →  opens branches
CREATIVITY →  explores and recombines branches  →  generates novel ideas
CRITIC     →  evaluates novelty and quality  →  drives the next loop
```

These are not separate capabilities. They are three phases of the same cognitive act — and the model learns all three together.

---

## The Pipeline

```
USER QUERY
     │
     ▼
┌──────────────────────┐
│      CURIOSITY       │
│                      │
│  What don't we know? │
│  What assumptions    │
│  are hidden here?    │
│  What domains haven't│
│  been considered?    │
└──────────┬───────────┘
           │  question seeds + branch directions
           ▼
┌──────────────────────┐
│      CREATIVITY      │
│                      │
│  Research SOTA       │
│  Branch wide         │
│  Prune redundancy    │
│  Combinatory mix     │
│  Surface novel ideas │
└──────────┬───────────┘
           │  candidate outputs
           ▼
┌──────────────────────┐
│        CRITIC        │
│                      │
│  Are these genuinely │
│  novel?              │
│  What's still        │
│  unexplored?         │
│  Score: 0–10         │
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     │            │
   PASS          FAIL
     │            │
     ▼            └──► back to CURIOSITY
   OUTPUT              with Critic's notes
                       as new seeds
```

The loop continues until the Critic scores the output as sufficiently novel and satisfactory. Each iteration, the Critic's notes feed back into Curiosity as new constraints and directions — tightening the search toward genuinely unexplored territory.

---

## Module Specifications

### Curiosity

**Role:** Opens the problem space before any generation happens. Prevents premature convergence.

**Behaviour:**
- Never generates solutions directly
- Surfaces hidden assumptions in the input
- Identifies which domains, perspectives, and constraints haven't been considered
- Generates a set of high-leverage questions: "If answered, these would most change the output"
- On loop iterations: incorporates Critic's feedback as new question seeds

**Output schema:**
```json
{
  "hidden_assumptions": [],
  "unexplored_domains": [],
  "questions": [
    { "id": "Q1", "question": "...", "why_this_unlocks": "..." }
  ],
  "branch_seeds": []
}
```

---

### Creativity

**Role:** Generates original ideas through structured branching and combinatory synthesis.

**Behaviour:**
- Receives branch seeds from Curiosity
- Researches what already exists (prevents reinventing the wheel)
- Branches wide: N distinct framings, deliberately varied across domains
- Prunes convergent or redundant branches
- Cross-pollinates surviving branches to find combinations unreachable by any single path
- Annotates each output with which branches produced it and why it's novel

**Output schema:**
```json
{
  "research": [],
  "branches": [
    { "id": "B1", "frame": "...", "candidates": [] }
  ],
  "pruned": [
    { "id": "B2", "reason": "..." }
  ],
  "combinations": [
    { "from": ["B1", "B3"], "result": "...", "novelty_note": "..." }
  ],
  "dead_ends": [],
  "output": []
}
```

---

### Critic

**Role:** Evaluates output quality. Drives the loop. Prevents noise from being mistaken for creativity.

**Behaviour:**
- Scores each candidate output on novelty (0–10) and relevance (0–10)
- Identifies what's still generic, expected, or well-trodden
- Flags which directions remain untried
- On PASS: approves output and ends the loop
- On FAIL: writes specific feedback notes that feed back into the next Curiosity iteration

**Pass threshold:** Novelty ≥ 7 and Relevance ≥ 7 across the candidate set.

**Output schema:**
```json
{
  "scores": [
    { "candidate": "...", "novelty": 8, "relevance": 9, "notes": "..." }
  ],
  "verdict": "PASS | FAIL",
  "unexplored_directions": [],
  "feedback_for_curiosity": []
}
```

---

## Training Data Strategy

### The Fundamental Constraint

**Training data is generated exclusively by vanilla Gemma 4 — not by smarter models, not scraped from the web.**

The claim being made is that creative reasoning capacity exists latently in Gemma 4 already. Structured prompting surfaces it. Fine-tuning then *internalises* that structure so the model reasons this way without scaffolding.

Using a smarter model to generate data would be distillation — a known technique, but not what this project proves. This constraint is what makes PSYCHE scientifically honest and technically distinct.

### The Data Pipeline

```
Carefully engineered prompt templates
             │
             ▼
       Vanilla Gemma 4
             │
             ▼
  Full loop reasoning chains
  (Curiosity → Creativity → Critic)
             │
             ▼
  QA + curation (filter shallow chains)
             │
             ▼
       JSONL training data
             │
             ▼
     Fine-tune Gemma 4
```

### Prompt Engineering (Before Scale)

Before running at scale, prompt templates that reliably trigger deep branching, genuine curiosity, and rigorous self-critique in vanilla Gemma 4 must be locked via rapid playground iteration (~30 min).

| Template | Purpose |
|---|---|
| *"Generate five completely different approaches to [task]. For each, explain what assumption it makes about the problem."* | Forces structural branch diversity |
| *"What's the most unexpected way to solve [task]? What assumption does that break?"* | Anomaly forcing |
| *"How would a [domain A expert] approach [task]? A [domain B expert]? What emerges if you combine them?"* | Cross-domain mapping |
| *"Before solving [task], list five things you don't know that would change your answer. Then solve it anyway."* | Curiosity seeding |
| *"Generate three mediocre solutions. Find the one insight from each that, combined, produces something genuinely new."* | Recombination forcing |

**Selection criteria:** branching depth, structural diversity, unexpected cross-domain connections, output novelty vs. vanilla prompting. Lock the top 2–3 per module before proceeding.

---

## Domain Space for Training Data

Domains are chosen not for breadth of human knowledge, but specifically to exercise **curiosity, creativity, and critical evaluation** as intensely as possible. Each domain is selected because it has a high ceiling for novelty, a clear failure mode (convergence to the obvious), and enough inherent complexity to require genuine branching.

### The Eight Core Domains

**1. Creative Naming & Branding**
The canonical creativity stress test. The space of "good names" is small, the space of "obvious names" is huge, and the gap requires genuine cross-domain synthesis. Easy to evaluate — novel vs. generic is obvious to humans. Exercises all three modules hard.
*Inputs:* name a biotech startup, tagline for an AI ethics nonprofit, rebrand a legacy institution

**2. Product & System Ideation**
Requires Curiosity (what problem is actually being solved?), Creativity (what hasn't been tried?), and Critic (is this genuinely differentiated?). Directly maps to the Education and Health competition tracks.
*Inputs:* design an offline literacy tool for rural classrooms, propose a novel approach to vaccine distribution tracking

**3. Scientific Hypothesis Generation**
Pure Curiosity + Creativity domain. The Critic maps directly to scientific rigour — is this testable, is it actually novel, does it contradict known evidence? Exercises anomaly detection and cross-domain synthesis most intensely.
*Inputs:* generate hypotheses for why this drug trial failed, propose an alternative mechanism for this observed phenomenon

**4. Philosophical Thought Experiments**
Highest-purity Curiosity domain. Forces the model to question assumptions at the root level before generating anything. Critic evaluates logical consistency and genuine novelty of framing.
*Inputs:* construct a thought experiment challenging the hard problem of consciousness, reframe the trolley problem using a non-Western ethical framework

**5. Cross-Domain Analogy Construction**
Exercises combinatory synthesis more than any other domain. The model must hold two distant domains in parallel and find structural correspondences. Produces the highest-novelty chains.
*Inputs:* find a deep analogy between immune systems and market regulation, map jazz improvisation onto software architecture

**6. Narrative & Character Design**
High creativity ceiling, clear Critic criteria (originality vs. tropes, internal consistency, emotional resonance). Branching is most natural here — character can be approached from backstory, contradiction, archetype subversion, or cognition.
*Inputs:* create a villain whose worldview is internally coherent and sympathetic, design a non-human character with genuinely alien cognition

**7. Strategic Problem Reframing**
Forces Curiosity to find the *actual* problem underneath the stated one — often the most valuable creative move. Critic evaluates whether the reframe is genuinely insightful or just lateral noise.
*Inputs:* reframe "how do we increase school attendance?" as a systems problem, find three alternative definitions of "productivity" that change what you'd optimise for

**8. Speculative & Future Design**
Exercises creative synthesis and Future Vision simultaneously. Critic must evaluate plausibility and second-order effects — not just novelty. Strong signal for the competition's Global Resilience track.
*Inputs:* design a society that has solved loneliness as a public health problem, what does education look like when AI tutors are ubiquitous?

### Why These Eight

Together they cover the full range of each module's cognitive modes:

| Module | Modes Covered | Domains That Exercise Them Most |
|---|---|---|
| Curiosity | Assumption-surfacing, domain-gap detection, unknown-unknown mapping | 4, 7, 3 |
| Creativity | Combinatory synthesis, cross-domain mapping, constraint inversion | 5, 1, 6 |
| Critic | Novelty evaluation, logical consistency, feasibility, trope detection | 3, 4, 2 |

A model trained across all eight generalises the *reasoning pattern* — not the domain knowledge. That generalisation is the goal.

### Volume Targets

| Chain Type | Per Domain | Domains | Total |
|---|---|---|---|
| Curiosity only | 60–80 | 8 | ~560 |
| Creativity only | 60–80 | 8 | ~560 |
| Critic only | 60–80 | 8 | ~560 |
| Full loop (all three modules) | 30–40 | 8 | ~280 |
| **Total** | | | **~1,960 examples** |

Quality over quantity. Weak or shallow chains are filtered during QA before fine-tuning.

### Full Loop Training Schema

```json
{
  "domain": "Creative Naming & Branding",
  "input": "Generate names for an AI-powered creative studio",
  "loop": [
    {
      "iteration": 1,
      "curiosity": {
        "hidden_assumptions": ["Studio = visual design only", "Name must be English"],
        "unexplored_domains": ["Romanian etymology", "material science", "evolutionary biology"],
        "questions": [
          {
            "id": "Q1",
            "question": "What is the studio's core transformation — of what into what?",
            "why_this_unlocks": "Names built around a transformation verb are more ownable than noun-compounds"
          }
        ],
        "branch_seeds": ["transformation", "material tension", "biological evolution", "linguistic roots"]
      },
      "creativity": {
        "research": ["Existing: Midjourney, Runway, ElevenLabs — abstract or motion-based. Gap: transformation-as-process names."],
        "branches": [
          { "id": "B1", "frame": "Latin roots for transformation", "candidates": ["Arvolve", "Mutare", "Evolvere"] },
          { "id": "B2", "frame": "Material + mineral tension", "candidates": ["Tecton", "Obsidian Studio"] },
          { "id": "B3", "frame": "Biological evolution", "candidates": ["Scion", "Morphex"] }
        ],
        "pruned": [{ "id": "B2", "reason": "Industrial feel mismatches creative positioning" }],
        "combinations": [
          { "from": ["B1", "B3"], "result": "Evolvere — evolution as continuous verb", "novelty_note": "Unreachable from either branch alone" }
        ],
        "dead_ends": ["Tech-prefix names (AI-, Gen-, Hyper-) — all generic and crowded"],
        "output": ["Arvolve", "Morphex", "Evolvere", "Tecton", "Scion"]
      },
      "critic": {
        "scores": [
          { "candidate": "Arvolve", "novelty": 9, "relevance": 9, "notes": "Art + evolve compound. Ownable, evolution-tied, not generic tech." },
          { "candidate": "Morphex", "novelty": 7, "relevance": 8, "notes": "Strong but slightly clinical." }
        ],
        "verdict": "PASS",
        "unexplored_directions": [],
        "feedback_for_curiosity": []
      }
    }
  ],
  "final_output": ["Arvolve", "Morphex", "Evolvere", "Tecton", "Scion"]
}
```

---

## Fine-Tuning

- **Base model:** Gemma 4 (26B or 31B depending on resource constraints)
- **Method:** Supervised Fine-Tuning (SFT) on full loop chains — input to complete reasoning trajectory to final output
- **No architectural changes** — data does the work
- **Training format:** JSONL with full Curiosity → Creativity → Critic chain as the output sequence
- **Tooling:** Unsloth (see links at top)

### What the Model Learns

After fine-tuning, when given any open-ended creative task, Gemma 4 should:
1. Surface hidden assumptions before generating
2. Branch into genuinely distinct framings
3. Prune convergent branches
4. Cross-pollinate survivors into novel combinations
5. Evaluate its own outputs critically
6. Iterate if outputs are still generic

This is a reasoning architecture change, not a style change.

---

## Validation

### The Kaggle Challenge as Benchmark

The competition challenge is used as the evaluation benchmark — not a training target. After fine-tuning, the model is tested: *"What project ideas would you generate for the Gemma 4 Good Hackathon?"*

If the fine-tuned model produces demonstrably better, more original, more strategically differentiated ideas than vanilla Gemma 4 — the system is proven.

| Tier | Setup | Expected |
|---|---|---|
| 1 | Vanilla Gemma 4, no prompting | Baseline |
| 2 | Vanilla Gemma 4 + engineered pipeline prompts | Better |
| 3 | Fine-tuned Gemma 4, standalone | Best |

Tier 3 > Tier 2 > Tier 1 = success.

### Competition Submission

- **Track:** Education (primary), Global Resilience (secondary)
- **Impact story:** Students and creators globally lack access to Socratic teachers and creative mentors. PSYCHE provides both — a model that teaches you *how* to think, not *what* to think.
- **Technical differentiator:** Training on full reasoning loops (Curiosity → Creativity → Critic), not input/output pairs. Proves that latent creative capacity can be surfaced and internalised from the base model alone.
- **Demo:** Live side-by-side — vanilla Gemma 4 vs fine-tuned PSYCHE on 5 benchmark creative tasks.

---

## Timeline

| Week | Milestone |
|---|---|
| **Week 1 ✅** | Curiosity, Creativity, and Critic modules built and tested. Pipeline validated. |
| **Week 2** | Lock prompt templates (playground iteration). Run full data generation across 8 domains. QA. ~2,000 training examples. |
| **Week 3** | Fine-tune Gemma 4 on Unsloth. Evaluate across all three tiers. Iterate. |
| **Week 4** | Build demo. Write submission docs + video. Submit. |

**Hard deadline: May 18, 2026**

---

## Success Criteria

1. Fine-tuned Gemma 4 produces demonstrably more original outputs than vanilla Gemma 4 on held-out prompts
2. Model shows Curiosity behaviour: surfaces assumptions before generating
3. Model shows Creativity behaviour: branches into structurally distinct framings and combines them
4. Model shows Critic behaviour: evaluates its own outputs and flags generic ones
5. All three behaviours emerge without external orchestration — from the weights alone

---

## Longer Vision

PSYCHE is a proof of concept for a broader thesis: **curiosity, creativity, and critical thinking are not personality traits — they are reasoning architectures that can be learned.** A model trained to think this way doesn't just name companies better. It approaches any open-ended problem — in education, in science, in design, in strategy — with structural originality.

That generalises. That's the direction of AGI.

> *"Everything is a remix. The question is whether you're aware of what you're remixing."*