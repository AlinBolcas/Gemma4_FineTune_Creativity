"""
enrich_seed_prompts.py - Expand seed prompts to large-scale training coverage.

Run:
    python src/II_dataGen/enrich_seed_prompts.py
"""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from itertools import product
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = REPO_ROOT / "data" / "input" / "seed_prompts.json"
TARGET_PER_DOMAIN = 500
RANDOM_SEED = 20260427


COMMON_CONSTRAINTS = [
    "with no reliable internet",
    "with almost no budget",
    "where trust is low",
    "where the usual incentive is backwards",
    "where the obvious solution would make the problem worse",
    "where the people affected have little formal power",
    "where success must be visible within one week",
    "where privacy matters more than convenience",
    "where the system must work across languages",
    "where the strongest user is not the person being served",
    "where the solution must feel voluntary",
    "where the intervention has to survive boredom",
    "where social status distorts behavior",
    "where measurement changes the thing measured",
    "where failure must be useful rather than hidden",
    "where the design must work for both experts and beginners",
    "where the main bottleneck is attention, not information",
    "where the cultural norm punishes directness",
    "where the right answer depends on timing",
    "where the system must improve when resources shrink",
]

AVOIDS = [
    "apps, dashboards, or generic reminders",
    "gamification cliches",
    "surveillance or forced compliance",
    "more information as the main solution",
    "a simple marketplace model",
    "productivity-hack language",
    "therapy language",
    "charity framing",
    "AI magic as the entire answer",
    "purely individual behavior change",
    "generic community-building language",
    "obvious metaphors",
]


def _load_seed_file() -> dict:
    with open(SEED_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write_seed_file(data: dict) -> None:
    backup_path = SEED_PATH.with_name(
        f"{SEED_PATH.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{SEED_PATH.suffix}"
    )
    shutil.copy2(SEED_PATH, backup_path)
    with open(SEED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Backup saved: {backup_path.relative_to(REPO_ROOT)}")


def _clean(text: str) -> str:
    text = " ".join(text.strip().split())
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _dedupe_keep_order(prompts: list[str]) -> list[str]:
    seen = set()
    result = []
    for prompt in prompts:
        cleaned = _clean(prompt)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _build_from_templates(spec: dict, seed: int) -> list[str]:
    rng = random.Random(seed)
    prompts = []
    for template, dimensions in spec["templates"]:
        values = [spec[name] for name in dimensions]
        combos = list(product(*values))
        rng.shuffle(combos)
        for combo in combos:
            payload = dict(zip(dimensions, combo))
            prompts.append(_clean(template.format(**payload)))
    return _dedupe_keep_order(prompts)


DOMAIN_SPECS = {
    "everyday_creative": {
        "situations": [
            "two roommates who avoid conflict", "a family recovering from a hard year",
            "neighbors who recognize each other but never speak", "a couple with opposite social energy",
            "a teenager who finds sincerity embarrassing", "an adult child caring for an aging parent",
            "a friend group drifting apart after graduation", "a person rebuilding confidence after failure",
            "a workplace team that only talks through tasks", "siblings who love each other but compete",
            "a new immigrant family learning local customs", "someone whose creativity disappears under stress",
            "a household where chores become moral arguments", "a person who wants solitude without isolation",
            "friends separated by time zones", "a child afraid of transitions",
            "a person returning to a hometown that changed", "a family with no shared hobbies",
            "a quiet person being celebrated publicly", "someone ending a habit they still enjoy",
        ],
        "artifacts": [
            "ritual", "game", "conversation format", "gift", "household system", "weekly practice",
            "farewell ceremony", "shared challenge", "decision rule", "memory practice",
            "celebration", "repair process", "micro-tradition", "room arrangement", "story prompt",
        ],
        "success": [
            "connection feel earned rather than forced", "the awkwardness become part of the design",
            "the second attempt work better than the first", "people reveal something without confessing",
            "the format protect dignity", "small failures become useful material",
            "participation feel optional but attractive", "the quietest person gain influence",
            "meaning emerge through action rather than explanation", "the design improve with repetition",
        ],
        "templates": [
            ("Design a {artifacts} for {situations} {constraint}, and make {success}.", ["artifacts", "situations", "constraint", "success"]),
            ("Invent a {artifacts} for {situations}, but avoid {avoid} and make {success}.", ["artifacts", "situations", "avoid", "success"]),
            ("Create a low-pressure {artifacts} that helps {situations} while {constraint}.", ["artifacts", "situations", "constraint"]),
            ("Redesign an ordinary moment for {situations} so that {success}, without using {avoid}.", ["situations", "success", "avoid"]),
        ],
        "constraint": COMMON_CONSTRAINTS,
        "avoid": AVOIDS,
    },
    "naming_branding": {
        "ventures": [
            "a civic data trust for small towns", "a grief coaching service for practical people",
            "a repair-first electronics brand", "a cooperative grocery for mixed-income neighborhoods",
            "a privacy-preserving health journal", "a professional network for career restarters",
            "an offline learning lab for rural schools", "a slow news organization",
            "a climate adaptation studio for coastal families", "a library that functions as social infrastructure",
            "a peer mentorship platform for failed founders", "a mental health service for tradespeople",
            "a public-interest AI auditing firm", "a sleep platform with no tracking",
            "a local-history archive for ordinary people", "a food system using surplus harvests",
            "a financial service for irregular incomes", "a child-care cooperative for shift workers",
            "a science museum for adults and children together", "a civic ritual design studio",
        ],
        "tone": [
            "dignified but warm", "precise but not sterile", "serious without bureaucracy",
            "local without nostalgia", "premium without indulgence", "radical without sounding angry",
            "trustworthy without moralizing", "modern without futurist cliches",
            "humane without sentimentality", "quietly memorable",
        ],
        "task": [
            "name", "rebrand", "create a naming system for", "design a brand identity for",
            "invent a category name for", "create a public-facing identity for",
        ],
        "templates": [
            ("{task} {ventures} that feels {tone}, and avoid {avoid}.", ["task", "ventures", "tone", "avoid"]),
            ("Create five names for {ventures}; each should imply a different worldview while staying {tone}.", ["ventures", "tone"]),
            ("Reframe the category language for {ventures} so it feels {tone} rather than generic.", ["ventures", "tone"]),
            ("Build a brand concept for {ventures} where the name, tagline, and central metaphor all resist {avoid}.", ["ventures", "avoid"]),
        ],
        "avoid": AVOIDS,
    },
    "teaching_explanation": {
        "concepts": [
            "recursion", "emergence", "entropy", "opportunity cost", "confirmation bias",
            "compound interest", "feedback loops", "second-order effects", "path dependency",
            "network effects", "statistical significance", "working memory limits", "externalities",
            "systems thinking", "risk literacy", "asymmetric information", "implicit bias",
            "algorithmic ranking", "natural selection", "epistemic humility",
            "cognitive load", "causal inference", "incentive design", "collective action",
            "Bayesian updating", "precision versus accuracy", "marginal utility", "game theory",
        ],
        "learners": [
            "a 10-year-old who hates school", "teenagers who think the topic is obvious",
            "adults with no technical background", "a skeptical small business owner",
            "new voters", "parents at a community workshop", "nurses on a short break",
            "students with weak reading confidence", "apprentices in a trade school",
            "teachers with no prep time", "refugee learners in a multilingual classroom",
            "artists who distrust math", "engineers who overtrust models", "children using only household objects",
        ],
        "mediums": [
            "a 10-minute lesson", "a tactile demonstration", "a classroom game",
            "a story-based explanation", "a physical notation system", "a debate format",
            "a kitchen-table activity", "a no-slides workshop", "a two-person exercise",
            "a mistake-first activity", "a role-free simulation", "a drawing exercise",
        ],
        "templates": [
            ("Design {mediums} that teaches {concepts} to {learners}, but makes them feel the concept before naming it.", ["mediums", "concepts", "learners"]),
            ("Explain {concepts} to {learners} using a fresh analogy that avoids {avoid}.", ["concepts", "learners", "avoid"]),
            ("Create {mediums} for {learners} that reveals the hidden assumption behind {concepts}.", ["mediums", "learners", "concepts"]),
            ("Teach {concepts} through a deliberate misconception first, then design the moment where {learners} correct themselves.", ["concepts", "learners"]),
        ],
        "avoid": AVOIDS,
    },
    "product_service_ideation": {
        "users": [
            "first-generation university students", "rural teachers with overcrowded classrooms",
            "shift workers with unstable schedules", "elderly people preserving autonomy",
            "families managing chronic illness", "informal workers in dense cities",
            "caregivers returning to paid work", "new immigrants learning social norms",
            "students in low-bandwidth schools", "frontline healthcare workers",
            "freelancers who underprice themselves", "communities after natural disasters",
            "people leaving high-control communities", "parents of children with rare diseases",
            "small farmers facing volatile weather", "local governments with low civic trust",
            "isolated wealthy neighborhoods", "incarcerated learners", "trade apprentices",
            "people with ADHD facing paperwork", "adult children coordinating elder care",
        ],
        "systems": [
            "tool", "service", "physical product", "peer network", "coordination system",
            "training model", "financial product", "public-interest platform", "local archive",
            "offline-first workflow", "civic mechanism", "support infrastructure",
        ],
        "outcomes": [
            "preserves dignity", "turns a bottleneck into a feature", "works without centralized trust",
            "reduces coordination cost", "creates resilience without savings",
            "makes invisible labor legible", "builds confidence without surveillance",
            "lets expertise travel without hierarchy", "keeps agency with the user",
            "turns waiting time into valuable time",
        ],
        "templates": [
            ("Design a {systems} for {users} that {outcomes} {constraint}.", ["systems", "users", "outcomes", "constraint"]),
            ("Invent a {systems} for {users}, but avoid {avoid} and make the hardest constraint the source of value.", ["systems", "users", "avoid"]),
            ("Create a service concept for {users} where {outcomes}, and identify the failure mode it must resist.", ["users", "outcomes"]),
            ("Design a product for {users} that becomes more useful when {constraint}, not less.", ["users", "constraint"]),
        ],
        "constraint": COMMON_CONSTRAINTS,
        "avoid": AVOIDS,
    },
    "scientific_hypothesis": {
        "phenomena": [
            "some patients respond oppositely to the same treatment", "urban wildlife adapts faster than rural wildlife",
            "loneliness predicts inflammation better than stress", "some schools improve after losing funding",
            "creative insight appears after deliberate interruption", "sleep loss harms imagination before logic",
            "microbiome diversity changes risk preferences", "myopia rises fastest in wealthy cities",
            "social rejection changes pain thresholds", "placebos vary by delivery ritual",
            "some coral reefs survive repeated heat shocks", "bilingual children transfer skills beyond language",
            "certain memories become less accurate as confidence grows", "some ecosystems recover only after repeated disturbance",
            "public rankings lower real quality in expert domains", "long-term grief alters time perception",
            "mathematical beauty tracks later scientific usefulness", "cities produce innovation beyond population effects",
            "low-grade inflammation predicts depression onset", "some languages resist simplification under contact",
            "teams become less creative after adding high performers", "AI assistance reduces learning in some learners",
        ],
        "angles": [
            "hidden mediator", "selection effect", "measurement artifact", "adaptive tradeoff",
            "delayed feedback loop", "threshold effect", "context-dependent mechanism",
            "social signaling mechanism", "ecological niche shift", "developmental window",
        ],
        "tests": [
            "a clean discriminating experiment", "the data that would falsify each hypothesis",
            "a natural experiment", "a longitudinal study design", "a minimal causal model",
            "a rival prediction for each mechanism", "a measurement strategy that avoids circularity",
        ],
        "templates": [
            ("Generate competing hypotheses for why {phenomena}; make each imply {tests}.", ["phenomena", "tests"]),
            ("Propose rival mechanisms for {phenomena}, including one based on a {angles}.", ["phenomena", "angles"]),
            ("Design a research program around {phenomena} that separates a {angles} from an obvious explanation.", ["phenomena", "angles"]),
            ("Invent three non-obvious explanations for {phenomena}, then describe what observation would make each collapse.", ["phenomena"]),
        ],
    },
    "strategic_reframing": {
        "problems": [
            "increase school attendance", "reduce hospital readmissions", "fight misinformation",
            "make AI safer", "solve burnout", "improve public transit", "reduce loneliness",
            "fix the housing crisis", "make workplaces inclusive", "reduce youth unemployment",
            "improve public health", "address food insecurity", "make science more accessible",
            "reduce screen time", "improve corporate culture", "reduce violence in schools",
            "make cities sustainable", "help people exercise more", "reduce organizational silos",
            "build better climate adaptation", "improve elder care", "make democracy work",
            "repair trust in institutions", "reduce teacher turnover", "prepare workers for automation",
            "improve disaster response", "reduce medical errors", "make online spaces healthier",
        ],
        "lenses": [
            "from the viewpoint of the least powerful stakeholder", "by changing the unit of success",
            "without treating individuals as the main causal unit", "as an incentive design failure",
            "as a measurement problem", "as a status problem", "as a time-horizon conflict",
            "as a coordination failure", "as a legitimacy problem", "as a hidden definition dispute",
        ],
        "outputs": [
            "three conflicting definitions of success", "a new problem statement and one dangerous implication",
            "the assumptions the old framing protects", "a strategy that would look wrong under the old frame",
            "the stakeholder who benefits from keeping the old frame", "a metric that would reverse priorities",
        ],
        "templates": [
            ("Reframe 'how do we {problems}?' {lenses}, and produce {outputs}.", ["problems", "lenses", "outputs"]),
            ("Find three alternative definitions of '{problems}' that would change what gets funded first.", ["problems"]),
            ("Turn '{problems}' into a systems problem where the obvious solution becomes a symptom.", ["problems"]),
            ("Reframe '{problems}' by questioning who gets to define the problem and what their definition hides.", ["problems"]),
        ],
    },
    "cross_domain_analogy": {
        "source": [
            "immune tolerance", "river deltas", "jazz improvisation", "forest succession",
            "legal appeals", "coral reefs", "mycelial networks", "fermentation",
            "chess endgames", "glacier movement", "version control merges", "tidal pools",
            "black holes", "seed dormancy", "orchestral conducting", "translation of poetry",
            "fire ecology", "dead reckoning", "metamorphosis", "search and rescue",
            "peer review", "invasive species", "taxonomic classification", "vaccination",
        ],
        "target": [
            "organizational change", "software architecture", "market regulation", "team recovery",
            "scientific replication", "urban planning", "knowledge management", "startup ecosystems",
            "customer support escalation", "AI alignment", "public health messaging",
            "education reform", "platform governance", "community organizing", "database reliability",
            "diplomatic negotiation", "creative collaboration", "supply chain resilience",
            "legal reform", "climate adaptation", "family systems", "institutional trust",
        ],
        "focus": [
            "where the analogy breaks", "the hidden causal structure", "one misleading overlap",
            "the role of feedback loops", "the failure mode the analogy predicts",
            "what the analogy makes newly measurable", "what the analogy hides about power",
        ],
        "templates": [
            ("Find a deep structural analogy between {source} and {target}, focusing on {focus}.", ["source", "target", "focus"]),
            ("Map the process of {source} onto {target}, then identify the point where the analogy fails.", ["source", "target"]),
            ("Use {source} to redesign how we think about {target}, but avoid surface-level similarity.", ["source", "target"]),
            ("Draw a structural analogy between {source} and {target}; explain one insight and one dangerous distortion.", ["source", "target"]),
        ],
    },
    "narrative_worldbuilding": {
        "rules": [
            "memory is taxed like property", "sleep is public", "names expire every decade",
            "lies are physically impossible", "privacy is illegal but shame remains",
            "cause and effect occasionally reverse", "aging reverses after midlife",
            "all laws must be sung", "children legally raise their parents",
            "art cannot have individual authors", "success is treated as suspicious",
            "dreams are reviewable public records", "work is assigned by unresolved grief",
            "people inherit the consequences of choices they refused", "silence functions as currency",
            "weather responds to collective emotion", "citizens can loan years of life",
            "all contracts are remembered orally", "every adult has a legally assigned contradiction",
            "the dead can vote on one issue per year",
        ],
        "story_tasks": [
            "build the society's political economy", "create a protagonist shaped by the rule",
            "design a villain whose ethics make sense", "trace the second-order institutions",
            "show the new form of poverty", "design a conflict only this world could create",
            "create the social class that games the rule", "describe the bureaucracy that maintains it",
        ],
        "twists": [
            "without making the world dystopian by default", "while preserving one genuinely beautiful consequence",
            "and show why reform is morally ambiguous", "through architecture rather than exposition",
            "through family conflict rather than war", "from the viewpoint of someone who benefits from it",
            "by making the compassionate choice dangerous", "without using a chosen-one plot",
        ],
        "templates": [
            ("Build a world where {rules}; {story_tasks} {twists}.", ["rules", "story_tasks", "twists"]),
            ("Create a character in a society where {rules}, and make their strength become harmful in one context.", ["rules"]),
            ("Design a story conflict around a world where {rules}, focusing on the institution that would emerge.", ["rules"]),
            ("Invent a culture shaped by the rule that {rules}; show the status hierarchy it creates.", ["rules"]),
        ],
    },
    "philosophical_reasoning": {
        "assumptions": [
            "empathy is always good", "transparency improves truth", "more choice increases freedom",
            "responsibility belongs to individuals", "justice and fairness converge",
            "authenticity is morally superior to performance", "longer life is more valuable",
            "coherence is a virtue", "intentions matter morally", "privacy concerns shared information",
            "democracy requires equal votes", "science is value-free", "merit can be measured fairly",
            "education expands freedom", "punishment tracks desert", "trust should be verified",
            "consent can authorize future harm", "helping is distinct from controlling",
            "ownership applies cleanly to ideas", "moral progress is linear",
        ],
        "contexts": [
            "AI-mediated relationships", "irreversible medical decisions", "intergenerational climate debt",
            "collective memory", "predictive policing", "genetic risk markets",
            "automated education", "public health mandates", "digital identity systems",
            "synthetic companions", "life extension", "post-scarcity work",
            "cultural preservation", "algorithmic governance", "memory editing",
        ],
        "moves": [
            "Construct a thought experiment", "Build a paradox", "Create a scenario",
            "Design a case where the opposite seems ethically superior", "Reveal the hidden premise",
            "Separate two concepts that are usually fused", "Show where the concept becomes circular",
        ],
        "templates": [
            ("{moves} in order to challenge the assumption that {assumptions}, using {contexts} as the setting.", ["moves", "assumptions", "contexts"]),
            ("Design a thought experiment where {assumptions} becomes harmful rather than virtuous.", ["assumptions"]),
            ("Construct an argument that makes '{assumptions}' depend on a prior agreement nobody actually made.", ["assumptions"]),
            ("Create a philosophical case about {contexts} that makes the boundary between helping and controlling unstable.", ["contexts"]),
        ],
    },
    "systems_future_design": {
        "futures": [
            "AI tutors are free but attention is scarce", "routine surgery is dangerous because antibiotics fail",
            "500 million people are climate-stateless", "local manufacturing replaces global shipping",
            "citizens sell personal data as income", "memories can be legally deleted",
            "physical labor becomes a luxury activity", "AI-generated science floods journals",
            "carbon sequestration is the main property right", "cities are food and energy self-sufficient",
            "human lifespan reaches 200 years", "sleep can be compressed to two hours",
            "political parties are replaced by real-time issue systems", "orbital infrastructure is privately controlled",
            "translation removes practical language barriers", "personalized medicine makes every treatment unique",
            "the most scarce resource is human attention", "digital inheritance includes decades of behavior data",
            "robots have legal personhood", "climate volatility changes every 48 hours",
            "facial recognition is perfect and universal", "education cannot keep pace with skill change",
        ],
        "systems": [
            "governance model", "education system", "healthcare infrastructure", "property rights system",
            "labor market", "food distribution network", "credibility system", "immigration framework",
            "scientific review process", "mental health infrastructure", "electoral system",
            "public utility model", "water governance system", "conflict resolution system",
        ],
        "analysis": [
            "trace the second-order failure mode", "identify who resists it most",
            "show the new inequality it creates", "design the legitimacy mechanism",
            "describe the coordination bottleneck", "map the new black market",
            "show the cultural adaptation after 20 years", "identify the metric that would corrupt it",
        ],
        "templates": [
            ("Design a {systems} for a future where {futures}, then {analysis}.", ["systems", "futures", "analysis"]),
            ("Imagine a future where {futures}; design the institution society would need and its new failure mode.", ["futures"]),
            ("Create a transition plan toward a world where {futures}, focusing on who loses power first.", ["futures"]),
            ("Design safeguards for a future where {futures}, but make one safeguard create a new systemic risk.", ["futures"]),
        ],
    },
}


def enrich() -> None:
    data = _load_seed_file()
    domains = data.get("domains", {})
    summary = {}

    for idx, (domain_key, domain) in enumerate(domains.items(), 1):
        prompts = _dedupe_keep_order(domain.get("prompts", []))
        spec = DOMAIN_SPECS.get(domain_key)
        if not spec:
            summary[domain_key] = {"before": len(prompts), "after": len(prompts), "added": 0, "status": "no spec"}
            continue

        candidates = _build_from_templates(spec, RANDOM_SEED + idx)
        existing = {prompt.lower() for prompt in prompts}
        for candidate in candidates:
            if len(prompts) >= TARGET_PER_DOMAIN:
                break
            if candidate.lower() in existing:
                continue
            prompts.append(candidate)
            existing.add(candidate.lower())

        if len(prompts) < TARGET_PER_DOMAIN:
            raise RuntimeError(f"{domain_key} only reached {len(prompts)} prompts")

        before = len(domain.get("prompts", []))
        domain["prompts"] = prompts[:TARGET_PER_DOMAIN]
        summary[domain_key] = {
            "before": before,
            "after": len(domain["prompts"]),
            "added": len(domain["prompts"]) - before,
            "status": "ok",
        }

    data["_note"] = (
        "This seed curriculum is scaled for broad self-distillation. Each existing domain now has "
        f"{TARGET_PER_DOMAIN} diverse prompts designed to stress ambiguity, constraints, branching, "
        "recombination, reframing, and self-critique. The hackathon-style prompt remains held out."
    )
    _write_seed_file(data)

    print("\nSeed prompt enrichment complete:")
    for key, item in summary.items():
        print(f"  {key}: {item['before']} -> {item['after']} ({item['status']})")
    total = sum(len(domain.get("prompts", [])) for domain in domains.values())
    print(f"\nTotal seed prompts: {total}")


if __name__ == "__main__":
    enrich()
