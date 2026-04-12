# Psyche — Architecture Specification

> *Structured cognition over raw generation.*

---

## 1. Overview

Psyche is a cognitive super-structure for AI agents. It sits above Model Context Protocol as an orchestration and intelligence layer — not a prompt wrapper, but a structured system of interacting cognitive agents that together produce reasoned, creative, and contextually aware outputs.

**The premise:** Current LLM systems are reactive and stateless. Psyche makes them proactive, structured, and capable of extended autonomous cognition by decomposing intelligence into discrete, composable modules — each with its own internal pipeline.

**Part of:** ArX (Alin's modular AI engine) · Parent company: Arvolve

---

## 2. Top-Level Architecture

```
INPUT → PsycheGen ↔ Orchestrator ↔ [Cognitive Agents] → PsycheGen → OUTPUT
```

### PsycheGen

The system's entry/exit interface. Receives raw input (user messages, events, scheduled triggers), routes into Orchestrator via Perception, and surfaces the final response via Persona.

- All Orchestrator→agent calls are made via **structured output** (typed JSON schemas)
- All agents share access to: **selected tools** + **shared agent memory**

### Orchestrator

Central routing and coordination hub. Formerly labeled MCP (Model Context Protocol) in early sketches. Responsibilities:

- Parse incoming context (via Perception)
- Dispatch to cognitive agents based on task type
- Aggregate and synthesise agent responses
- Route output back through Persona to PsycheGen

---

## 3. Module Definitions

### 3.1 Cognitive Agents

All cognitive agents tagged `(TREE)` share the same internal pipeline structure (see Section 4).

| Agent | Role | Pipeline | Status |
|---|---|---|---|
| **Creativity** | Generative ideation | Research → Branch → Select → Mix → Output? | Defined |
| **Rationality** | Logical reasoning + planning | Research → Branch → Distil → Validate → Rationale | Defined |
| **Future Vision** | Forward projection | Same as Creativity, optimised for prediction | Defined |
| **Critic** | Recursive evaluation | OR-gated multi-criteria loop | Partially defined |
| **Emotion** | Affective modulation | TBD | Label only |
| **Curiosity** | Exploration drive / research trigger | TBD | Label only |
| **Dreamer** | Speculative/imaginative output generation | TBD | Label only |
| **Heartbeat** | Persistent pulse / scheduled tick | CronJob or Daemon | Defined |

---

#### Creativity (TREE)

> *"Everything is a remix."*

5-stage pipeline:

1. **Research** — Current SOTA, what exists
2. **Branched Areas** — Lateral expansion across domains `(BREADTH)`
3. **Selection** — Prune to highest-potential branches `(DEPTH)`
4. **Combinatory Mixing** — Cross-connect and recombine selected nodes
5. **Original Ideas?** — Output with explicit novelty uncertainty marker

The `?` is architecturally honest: the system produces *remixed* outputs. True novelty is not guaranteed.

---

#### Rationality (TREE)

4-stage beam-search pipeline:

1. **Idea Start + Research** — Entry point, grounds in evidence
2. **Branched Directions of Attack** — Parallel reasoning paths fan out from hub
3. **Distillation** — Convergence via logical heuristic / mathematical scoring
4. **Selection & Validation Loop → Final Rationale** — Output

**Loop condition:** If distillation fails, returns to Stage 1 with new research `(LOOP:D)`.

---

#### Future Vision (TREE)

Same structural pipeline as Creativity.

- Objective function: **future prediction**, not ideation
- Instructed and optimised for forward state modeling
- Different system prompt / reward signal, same tree executor

---

#### Critic

Recursive evaluator. Design decision **unresolved:**

- **Simple Loops** — Single-pass, cheaper, predictable
- **Branching Loops** — Multi-path, more powerful, requires RL training

The OR-gated node structure allows evaluation across multiple criteria simultaneously. Resolution of simple vs branching likely depends on whether RL-trained specialist models are pursued (see Section 6).

---

#### Heartbeat

Persistent background tick. Keeps the system alive between interactions. Triggers scheduled cognitive processes. Implemented as a CronJob or Daemon.

---

### 3.2 I/O Layer

| Module | Role | Direction | Function |
|---|---|---|---|
| **Perception** | Environmental input parsing | Bidirectional loop with Orchestrator | Reads and structures incoming context: messages, events, tool results |
| **Persona** | Identity and voice consistency | Bidirectional loop with Orchestrator | Ensures output coherence with defined system character/voice |

Both loop between Orchestrator and PsycheGen.

---

### 3.3 Execution Layer

| Module | Role | Pattern |
|---|---|---|
| **Factory** | Code writing and execution | ReAct: Reflection → Action |
| **JSON & Vision** | Structured I/O + multimodal input processing | Tool within Factory |
| **Agentic Loop** | Per-agent execution wrapper | ReAct / RE-ACT loop |

Factory is the execution primitive. JSON & Vision handles structured data and image input. The Agentic Loop wraps each cognitive agent's execution cycle.

---

### 3.4 Alignment Layer

| Module | Role | Direction |
|---|---|---|
| **Guardrails** | Alignment and safety constraints | Hard constraint input to Orchestrator |
| **Critic (Ethics)** | Ethical evaluation loop | Looping evaluator on Orchestrator outputs |

---

## 4. Shared Tree Primitive

All three `(TREE)` agents — Creativity, Rationality, Future Vision — share the same structural pattern:

```
expand → filter → collapse
```

| Stage | Rationality | Creativity | Future Vision |
|---|---|---|---|
| Seed | Idea Start + Research | Research / What Exists | Input + Context |
| Expand | Branched Directions (beam) | Branched Areas (breadth) | Branched Scenarios |
| Filter | Distillation (heuristic/maths) | Selection (depth) | Probability Scoring |
| Synthesise | Selection & Validation Loop | Combinatory Mixing | Trend Extrapolation |
| Output | Final Rationale | Original Ideas? | Predicted State |

**Architectural implication:** A single `TreeExecutor` primitive can instantiate all three trees with:

- Different **search objective** (rationale quality / creative novelty / future accuracy)
- Different **selection criteria** (logical scoring / recombination potential / probability)
- Different **loop condition** (re-enter on failure vs emit output)

This reduces the implementation surface significantly. Build one tree executor, configure three agents.

---

## 5. Implementation Plan

### Phase 1 — MVP

1. Modify `TextGen` to support **structured vision** function (Ollama · OpenAI · TextGen · AgentsGen)
2. Adjust `AgentsGen` as needed for the new structured output pattern
3. Implement MCP between agents as **JSON schema** (typed structured output per agent)
4. Iterate all modules to ensure interoperability
5. Ensure local-run support via **Ollama**

### Phase 2 — Learning & Fine-Tuning

- Backpropagation through agent responses via **importance/contribution scoring**
- Training at architectural level — embedded, not meta-level orchestration
- RL training for Critic (resolves simple vs branching loop decision)
- Evaluation: can architectural-level training produce emergent behaviours beyond individual module capability?

---

## 6. Open Questions

| Question | Priority | Notes |
|---|---|---|
| Specialised trained models vs prompted generalists? | High | MVP uses prompted; RL deferred to Phase 2 |
| Simple vs branching loops for Critic? | High | Resolves with RL decision |
| Memory architecture sufficient with agentic in-context? | High | May need persistent retrieval layer at scale |
| Can combinatory mixing produce genuine novelty? | Philosophical | Current model is honest: it produces remixes |
| How is rationality tested / scored heuristically? | Design | Maths + logical heuristics — implementation TBD |
| How does curiosity trigger research vs passive waiting? | Design | Drive signal implementation TBD |
| What is the Dreamer's output format and destination? | Design | Undefined — likely feeds back into Creativity |
| How does genius emerge in AI systems? | Research | Likely interaction of Curiosity + Future Vision + Creativity at scale |

---

## 7. Naming Reference

| Term | Meaning |
|---|---|
| **Psyche** | The super-structure. Named for the totality of mind. |
| **PsycheGen** | The generative entry/exit interface of the system |
| **Orchestrator** | Central hub (formerly MCP in early sketches) |
| **TreeExecutor** | Proposed shared implementation class for TREE agents |
| **Heartbeat** | Persistent system pulse / scheduler |
| **Factory** | Code execution layer |
| **ArX** | The parent modular AI engine this is built within |
| **Arvolve** | Parent company |

---

*Source of truth: this document. Downstream: implementation files, agent configs, system prompts.*
