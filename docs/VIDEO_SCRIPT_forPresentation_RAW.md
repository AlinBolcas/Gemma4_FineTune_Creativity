Creativity has been the central focus of this project. My entry for the Gemma 4 2026 competition explores creative reasoning — specifically, the limitations current language models face when attempting to generate genuinely original ideas.

At their core, LLMs are constrained by their training data, post-training structure, and stochastic sampling process. Through extensive experimentation over the past several years, one thing became consistently clear to me: language models are not inherently creative. They can recombine existing patterns remarkably well, but they rarely produce ideas that feel fundamentally novel or deeply innovative. Even simple tasks like generating original names often collapse into predictable outputs.

That observation led me to investigate creativity itself as a computational process. I believe this is one of the most important unsolved problems in AI. If we can meaningfully improve creative reasoning, we unlock a path toward systems capable of deeper innovation, self-improvement, and far more adaptive forms of intelligence.

To approach this problem, I started by studying human creative reasoning from first principles. I spent a significant amount of time analyzing my own thought process and comparing it with broader theories around human creativity and cognition.

From that analysis, I arrived at a two-branch framework.

The first branch is curiosity.

Curiosity can be viewed as an evolutionary mechanism — an exploratory predisposition that pushes organisms toward unknown spaces where potential opportunities for survival or adaptation may exist.

Applied computationally, this becomes a process of mapping curiosity domains around an input problem.

For example, if we ask a model to invent a new form of renewable energy, the first step is not generating answers directly. The first step is identifying all relevant conceptual domains connected to the problem: physics, biology, thermodynamics, materials science, environmental systems, infrastructure, and so on.

From there, the system expands exploratory questions within each domain. Those questions are then distilled into the most interesting and cognitively challenging directions. The result is a structured curiosity context designed to pressure the model into exploring less conventional conceptual territory.

The second branch is creativity itself.

This branch begins with grounded knowledge: facts, research, and established principles. From that foundation, the system branches into multiple lines of reasoning, exploring alternative possibilities and developing independent trains of thought.

An important aspect of this process is selective pruning.

Human thinking is not just expansion — it is also elimination. We naturally abandon weak directions and reinforce promising ones. So the system attempts to replicate that process by filtering out branches that lack coherence, novelty, or productive momentum.

The final stage is combinatorial mixing.

This is the core mechanism of the architecture.

Creativity often emerges when ideas from unrelated domains collide and recombine into new structures. The system therefore attempts to merge concepts across different branches, producing hybrids that would not emerge from a single linear reasoning path.

I believe this combinatorial synthesis is one of the fundamental mechanisms underlying human creativity.

The brain itself operates as a massively interconnected system where different regions continuously influence one another through highly complex emergent interactions. While that process is extraordinarily difficult to model directly, we can approximate aspects of it computationally through structured stochastic recombination.

In this pipeline, the language model acts as the synthesis layer that bridges concepts across branches and generates new conceptual combinations.

That becomes the heart of the system.

All preceding layers — curiosity expansion, branching, pruning, and exploration — exist to enrich the final synthesis stage. The end result is a filtered collection of ideas that have survived multiple layers of exploration and evaluation before converging into a final output.

The entire pipeline was implemented using Gemma 4.

The project itself is structured around several components, including inference architecture, LLM integration, and fine-tuning experiments centered on self-distillation.

The core hypothesis behind the fine-tuning process was this:

If we train models on reasoning traces that mimic structured creative cognition, we may be able to improve the model’s ability to generate more creative outputs over time.

Initially, I attempted this through self-distillation alone. However, the results were inconclusive due to the limited amount of synthetic training data I could generate locally. The first dataset contained roughly 300 examples, which trained successfully but produced no significant behavioral shift.

I then scaled the dataset substantially using externally generated synthetic data. While the results still do not fully reproduce the structure of the pipeline itself, the outputs appeared measurably more exploratory and creatively diverse.

Evaluating creativity remains inherently difficult because the metric is subjective. However, humans still possess a strong intuitive sense for what feels genuinely creative versus statistically derivative.

Another important observation involved temperature sampling.

Higher temperature values consistently increased exploratory behavior and produced more unconventional outputs. However, excessive temperature also introduced instability and incoherence.

In practice, the most effective range appeared to be between 0.6 and 0.8. Below that range, outputs became generic and overly deterministic. Above 1.0, coherence degraded significantly.

Ultimately, this research matters because creativity sits at the center of nearly everything we want advanced intelligence systems to achieve.

A language model capable of generating genuinely original ideas would have enormous implications across science, engineering, art, research, and autonomous problem solving.

That is why I believe creative reasoning is one of the highest-leverage directions in AI research today.

That concludes the presentation. Thank you.
