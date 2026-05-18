# Video Transcript — Gemma 4 Hackathon Entry

**Source:** `2026-05-18 12-55-31.mp4`  
**Duration:** ~19 min 30 sec  
**Transcribed via:** OpenAI Whisper (`whisper-1`)  
**Date:** 2026-05-18

---

## Full Transcript

**[0:00]** Creativity — that's what the main focus of the project has been. The focus of my entry for the Gemma 4 2026 competition has been creative reasoning. As we know, creativity is one thing that the current LLMs are lacking, or finding difficulty in being performant or competitive with humans.

**[1:10]** This may likely be due to the fact that the outputs that an LLM gives out are only as good as its original training data and the way the post-training is structured, and simply due to its stochastic nature. Through empirical testing we could conclude that language models are not incredibly creative. That is to say, no matter how I've used them the past four or five years, they did not manage to come up with an original idea that was truly innovative — even for simple things as coming up with names, it always failed to provide something that is truly impressive.

**[3:12]** With that in mind, I set out on a course of trying to investigate and research that area. This is because the moment we crack that, I think it's the one most impactful advancement that LLMs need in order to push themselves beyond their current capabilities. That's why I chose it as a main topic of focus — because I think it hits on all the branches that the projects are judged upon, and it's essentially the lock to super intelligence and self-improvement.

**[4:20]** So the way I approach this has been taking inspiration from human creative reasoning and human thinking. That is, asking myself and trying to have a very introspective analysis or reflection about my own creative thinking, and also just the general theory on how humans come up with creative ideas.

**[5:04]** So based on that, we have concluded that it's a two-way process — a process involving two branches. As you can see in this diagram, a part of it, or the beginning of it, is curiosity. Curiosity from an evolutionary perspective — you might call it the exploratory predisposition. A bias or a range within which being exploratory and being curious about things that are unknown can increase the chances of survival.

**[6:06]** And within that landscape, we can ask questions, we can explore the mental models. Like we say in this first bit, it is mapping out curiosity domains. So first it is: okay, what can we be curious about in respect to the input? Which kind of domains — if I say "invent a new type of renewable energy or a new way through which we can procure energy" — then we can think of all the different domains or sectors which could be relatively associated.

**[6:59]** And then we branch on to those, expanding a set of questions for each of the domains. And then it's a case of distilling the question set and arriving at some really interesting and the best questions that are relevant to our thought process. And that becomes the curiosity kind of recipe or context, which is designed to challenge the creativity and force the arrival at a truly innovative set of ideas.

**[8:06]** So on the second branch, we have creativity, which is essentially starting from research. The fundamentals of what we know, the facts — and on top of those, we branch into domains of thinking of different options and alternatives. And then we develop each of those branches. We follow the trains of thoughts for each of those.

**[8:45]** And then the important organic aspect — which I think happens in humans — is the natural selection of those branches. And more importantly, it's a natural selection of pruning some of the branches which are not more relevant or are ones that are not leading anywhere.

**[9:13]** And then the final part is the combinatorial mixing. So this is truly the organic element where we want multi-topic or multidisciplinary vision on top of those branches and see how they associate with each other. I think this is really the essence of being creative — being able to combine ideas from different areas and merging them, creating a remix, creating a hybrid, which essentially bridges or gives birth to any idea.

**[10:00]** Now, this is very similar to — if you will — procreation. So the DNA of a mother and a father gets mixed in a sort of pseudo-random biological way, giving birth to a completely new person with a mixed DNA recipe. Very similar is as well, if we think about the brain and how all of the neural processes that happen within our skull are very kind of tightly knit and compact.

**[10:53]** And this is very similar to what we see in a 3D sort of topological structure where different parts of the brain communicate in a very organic way, but also defined by the evolutionary process that led to the shape and form and structure that the brain currently has. Which means the way in which different neural signals influence each other is, for one, very difficult to model and discern and categorize.

**[11:36]** But we can at least say that it is a form of organic combinatory mixing, which we could simplify through a simple stochastic LLM call — which is attempting to bridge these ideas and acts as the agent or layer which creates new ideas from different branches by combining them.

**[12:16]** So that's the heart of the system, I could say, but all of the layers add to the overall output. So the final synthesis is essentially just putting together all the ideas that are making it through, or surviving all of these filters, and are processed through or prepared through this pipeline. So this is where we arrive. And that has been the pipeline, which creates it. We've used Gemma 4 for this.

**[13:04]** I've structured the project in a very readable way, I should say. And we've got inference where we have the LLM integration. I won't jump through the code too much, but we've also essentially fine-tuned. So if I just check my notes here — we've got the architecture. The second part is the self-distillation idea.

**[13:58]** So from the original training data, I'm just trying to essentially prove that the pipeline in itself — the way in which we prompt the model to think deeply about a problem in ways that mimic the categorization that I came up with for the way in which we think creatively — can lead to models improving their own creativity. So I think that is a key statement that I attempted to make.

**[14:29]** Although the results are inconclusive, I should say, due to the limitations of being able to generate enough training data. I unfortunately had to resort to a third-party API-based language model to generate enough training data to visibly have a successful fine-tune. So running just the self-distillation process, I arrived at around 300 data points, which successfully trained, but it didn't give any noticeable results.

**[15:38]** So the third part, or the next set, was when I increased the data set about tenfolds or even a hundredfolds. And from that, I can say we still haven't mimicked the pattern of the pipeline or the structure itself, but based on our analysis, we figured that it is relatively more creative. Although, obviously, that's why it's a very difficult topic to research — because the assessment is incredibly subjective. But regardless, all of us can say what is creative and what is not.

**[16:45]** And there's a caveat to this: the temperature parameter could yield more creative outputs. Now, to my understanding, and based on what I tested, the temperature seems to increase the likelihood of outputting more bold or extravagant thoughts or tokens. And in essence, it overall creates a more noisy result, I would say, but it definitely helps with being more exploratory.

**[17:44]** So the temperature that I normally gravitate towards is obviously past 0.5, but somewhere between 0.6 to 0.8 seems to be a good range. Anything past 1 is not really usable. And when we go to 0, it's just the most kind of baseline, generic result.

**[18:16]** So with that in mind — why this matters is already mentioned. This stands at the core of everything we're building, and the impacts of having a creative LLM, a truly fresh idea sparking LLM, is tremendous. So that's why I believe that it is the best winner-takes-all situation or direction to push towards.

**[19:14]** And that's about it. When it comes to the write-up, we've got this. That's it. Thank you.

---

## Key Points Summary

- **Core thesis:** LLMs lack genuine creativity. Fine-tuning Gemma 4 on a structured creativity pipeline can push models toward authentic creative output.
- **Two-branch architecture:**
  - **Curiosity branch** — maps curiosity domains, generates and distils best questions to challenge the problem space
  - **Creativity branch** — researches facts, branches into options, follows trains of thought, applies natural selection (pruning dead branches), then does combinatorial mixing across domains
- **Combinatorial mixing** is the heart: merging ideas from different branches via a stochastic LLM call simulates the organic mixing that produces genuinely novel ideas in human cognition
- **Fine-tuning approach:** Self-distillation first (300 data points — inconclusive). Scaled to 10-100x more data via external API — modest improvement, still subjective to evaluate
- **Temperature insight:** 0.6-0.8 is the sweet spot for exploratory output without noise collapse
- **Why it matters:** Creative LLMs are the unlock to superintelligence and self-improvement
