"""
generate_visuals_v4.py — Full brand-aligned visual batch.

Unified Arvolve presentation design language.
Two image families, single coherent aesthetic:
  - HERO:    sleek minimal concept pieces (used as accents, never full-bleed)
  - DIAGRAM: editorial dashboard infographics (labels, arrows, structured)

Both share the same restraint: deep off-black canvas, single cool-blue accent,
thin typography, generous negative space, premium hierarchy.

Output: data/output/visuals/v4/<id>.png
"""

import os, sys, time, json, requests, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "visuals" / "v4"
MAX_WORKERS = 5
MODEL = "gpt-image-2"
PROVIDER = "openai"
TIMEOUT = 240

# ─── Shared visual DNA ──────────────────────────────────────────────────────
# Applied to every image. Defines the Arvolve presentation aesthetic.
SHARED_DNA = (
    "Background: deep restrained off-black #0A0A0C, perfectly even, no gradient, no vignette, no noise, no border. "
    "Color: a single restrained cool-blue accent (#3195F0 to #00A9FF) used sparingly as glow, edge, or highlight. "
    "Optional secondary: thin warm bronze gleam, minimal. Stone grey (#C0BDBA) for secondary detail, white (#F5F5F5) for primary emphasis. "
    "Aesthetic: editorial, premium, calm, computational, confident. Never noisy, never decorative, never busy. "
    "Lighting: clean, controlled, no bokeh, no haze, no atmospheric fog. "
    "Composition: generous breathing room, strong silhouette, intentional negative space, perfectly centered or rule-of-thirds. "
    "Edges: the image MUST extend cleanly to the canvas border with no dark vignette or letterboxing artifacts."
)

# Hero suffix — sculptural, no text
HERO_SUFFIX = (
    SHARED_DNA + " "
    "Form language: refined sculptural object, smooth and crisp geometry with subtle controlled fracture, "
    "intentional micro-detail (toolmark or digital artifact accents, never noise), engineered negative space. "
    "Studio-grade photographic rendering, high dynamic range, sharp speculars, subtle blue glow from within. "
    "No text, no labels, no diagrams, no UI elements."
)

# Diagram suffix — labels and structure allowed and required
DIAGRAM_SUFFIX = (
    SHARED_DNA + " "
    "Layout: clean editorial infographic, single dark panel, well-spaced rounded nodes, thin hairline connectors with arrow heads, "
    "crisp sans-serif typography (Inter or similar), generous letter-spacing on labels, hierarchy immediately readable. "
    "Style: premium dark-mode dashboard / scientific paper figure quality. Calm and authoritative. "
    "Every label legible, every connection purposeful, zero decorative noise, no photographic elements, no 3D extrusion. "
    "Treat text like a designer at Apple keynote — sparse, precise, beautiful."
)

# ─── Image set ──────────────────────────────────────────────────────────────

IMAGES = [
    # ── HEROES: clean concept pieces for breather / accent use ──
    {
        "id": "hero_thought_kernel",
        "type": "hero",
        "aspect_ratio": "16:9",
        "prompt": (
            "A single elegant sculptural form, centered on canvas, representing a cognitive kernel. "
            "Smooth dark stone exterior, cracked open along one vertical seam revealing a cool-blue luminous core within. "
            "Bronze and stone fragments suggest computational structure inside. "
            "Highly refined, museum-grade object, photographed in studio."
        ),
    },
    {
        "id": "hero_branching_mind",
        "type": "hero",
        "aspect_ratio": "16:9",
        "prompt": (
            "A single organic-mechanical form rising from a stone base, branching upward into many thin filaments that glow softly blue at their tips. "
            "Suggests dendrites, decision branches, the moment thought divides. "
            "Restrained, elegant, sculptural, museum-quality."
        ),
    },
    {
        "id": "hero_dual_streams",
        "type": "hero",
        "aspect_ratio": "16:9",
        "prompt": (
            "Two sculptural pillars side by side on a dark plane, one slightly cracked and revealing a glowing blue core, "
            "the other smooth and intact, connected by a single thin horizontal beam of blue light between them. "
            "Symbol of two cognitive streams in conversation. Restrained, elegant, sculptural."
        ),
    },
    {
        "id": "hero_creative_genesis",
        "type": "hero",
        "aspect_ratio": "16:9",
        "prompt": (
            "An elegant lens-shaped sculptural form centered on canvas, dark mineral surface, "
            "a single concentrated point of cool-blue light at its center radiating subtle rays outward in geometric precision. "
            "Sense of focused emergence — a creative idea concentrating into being. Studio-grade."
        ),
    },
    # ── DIAGRAMS: structural infographics ──
    {
        "id": "diag_pipeline_flow",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial infographic of an 11-stage creative reasoning pipeline. "
            "Two vertical streams side by side. "
            "Left stream header 'CURIOSITY' with 4 numbered nodes labeled: Map, Expand, Distill, Socratic Output. "
            "Right stream header 'CREATIVITY' with 6 numbered nodes labeled: Research Plan, Branch, Develop, Select, Mix, Synthesize. "
            "A single horizontal dashed line from 'Socratic Output' across to 'Research Plan' labeled 'steering' in small caps. "
            "Both streams converge with thin arrows into a single bottom node labeled 'CRITIC', with one curved feedback arrow labeled 'loop' returning upward. "
            "Numbered nodes 1-11. Single dark panel. No icons, no illustrations, pure typographic diagram."
        ),
    },
    {
        "id": "diag_three_tiers",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial comparison diagram. Header reads 'EVALUATION'. "
            "Three vertical columns side by side, each a tall rounded panel. "
            "Column 1: small label 'TIER 1', bold title 'Vanilla', subtitle 'Gemma 4 base, no scaffolding'. "
            "Column 2: small label 'TIER 2', bold title 'Pipeline', subtitle 'Vanilla + full 11-stage runtime'. "
            "Column 3: small label 'TIER 3', bold title 'Fine-Tuned', subtitle 'LoRA adapter, no scaffolding', outlined in cool-blue glow. "
            "Thin right-pointing arrows between columns with tiny labels 'adds runtime' and 'internalizes'. "
            "Below all three: a single thin line labeled 'shared held-out prompt set'. "
            "No icons, pure typographic editorial diagram."
        ),
    },
    {
        "id": "diag_self_distillation",
        "type": "diagram",
        "aspect_ratio": "1:1",
        "prompt": (
            "Editorial circular flow diagram. Six labeled nodes arranged in a perfect ring, connected by thin curved arrows clockwise. "
            "Node 1 'Seed Prompts', Node 2 '11-Stage Pipeline', Node 3 'Reasoning Traces', Node 4 'SFT Dataset · 4,771 examples', Node 5 'LoRA Fine-Tune', Node 6 'Gemma 4 + Adapter'. "
            "Tiny edge labels on each arrow: generates, structures, formats, trains, internalizes, seeds. "
            "Center of ring contains small all-caps text 'SELF-DISTILLATION LOOP'. "
            "Node 6 has a subtle cool-blue glow outline. Pure typographic diagram, no icons."
        ),
    },
    {
        "id": "diag_combinatorial_mix",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial network diagram showing creative branching and recombination. "
            "Left side single node labeled 'Input'. "
            "Middle column 5 nodes labeled B1, B2, B3, B4, B5. "
            "Right column 3 nodes labeled 'Hybrid A', 'Hybrid B', 'Hybrid C'. "
            "Single arrow from Input fanning out to all 5 branches, label 'diverge'. "
            "Cross-connecting thin arrows from selected branches to each hybrid, label 'recombine'. "
            "B2 and B4 shown with thin diagonal strikethrough and small label 'pruned'. "
            "Far right: a single node labeled 'Output' connected to the best hybrid with arrow labeled 'select'. "
            "Pure typographic diagram, no icons or illustrations."
        ),
    },
    {
        "id": "diag_temp_spectrum",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial spectrum chart. Header: 'TEMPERATURE'. "
            "Full-width horizontal bar with smooth gradient from grey on left through cool-blue in the middle to muted warm red on the right. "
            "Tick marks labeled below: 0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5. "
            "Three zone labels above the bar: left zone 'BASELINE' in grey, center zone 'SWEET SPOT 0.6–0.8' in cool-blue with subtle glow, right zone 'NOISE' in red. "
            "Below the bar: three minimal description lines aligned to each zone — 'safe and generic', 'structured and creative', 'incoherent'. "
            "Pure editorial chart, no icons, lots of negative space."
        ),
    },
    {
        "id": "diag_lora_adapter",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial technical diagram. A large rectangular block labeled 'GEMMA 4 · 8.07B parameters · FROZEN', shown as a wide dark slab. "
            "Wrapped around its edges, a thin cool-blue glowing ring labeled 'LoRA ADAPTER · 73.4M · 0.91%'. "
            "Below: small spec table with three rows: 'rank 16', 'alpha 32', 'dropout 0.05'. "
            "On the right: a small annotation block reading 'Trainable parameters live only in the ring. The base remains untouched.' "
            "Pure typographic editorial diagram, calm and authoritative."
        ),
    },
    {
        "id": "diag_eight_domains",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial 4x2 grid of eight labeled cards on a dark panel. Header at top: 'DOMAINS'. "
            "Eight cells each containing a short label centered in elegant typography: "
            "'Creative Naming', 'Product Ideation', 'Scientific Hypothesis', 'Philosophy', "
            "'Cross-Domain Analogy', 'Narrative Design', 'Strategic Reframing', 'Speculative Design'. "
            "Each cell outlined with a thin hairline border. "
            "Bottom caption in small caps: 'CHOSEN TO EXERCISE DIFFERENT COGNITIVE STAGES — THE MODEL GENERALIZES THE PROCESS, NOT THE DOMAIN'. "
            "Calm, balanced, editorial."
        ),
    },
    {
        "id": "diag_data_pipeline",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial horizontal flow diagram with 5 stages connected by thin arrows. "
            "Stage 1 labeled 'Seed Prompts · 8 domains'. "
            "Stage 2 labeled 'Pipeline Runs · 5,000 traces'. "
            "Stage 3 labeled 'SFT Dataset · 4,771 examples'. "
            "Stage 4 labeled 'LoRA Train · 2,148 steps'. "
            "Stage 5 labeled 'Adapter · 73.4M params'. "
            "Tiny labels below the arrows: 'generate', 'format', 'split 90/5/5', 'package'. "
            "A small caption underneath aligned right: 'Kaggle Tesla T4 · 4.4 h · 2 epochs'. "
            "Pure typographic, single dark panel."
        ),
    },
    {
        "id": "diag_critic_loop",
        "type": "diagram",
        "aspect_ratio": "1:1",
        "prompt": (
            "Editorial circular feedback loop diagram. Single node at top labeled 'Candidate' connects down with arrow to a central node labeled 'CRITIC · novelty + relevance · threshold ≥ 7/10'. "
            "Two arrows leave the critic node: one to the right labeled 'PASS' going to a small node 'Accept', "
            "one to the left labeled 'FAIL · targeted feedback' curving back upward into both 'CURIOSITY' and 'CREATIVITY' stream labels at the top corners. "
            "Pure typographic editorial diagram."
        ),
    },
    {
        "id": "diag_behavioral_shift",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial side-by-side comparison panel. Header: 'BEHAVIORAL SHIFT'. "
            "Two vertical columns labeled at the top 'VANILLA' (left, grey) and 'TUNED' (right, cool-blue accent). "
            "Each column contains four short stacked rows of small text describing typical output behavior: "
            "vanilla shows 'jumps straight to answers / flat option lists / minimal framing / no self-critique'; "
            "tuned shows 'asks clarifying questions first / groups output in idea sets / mirrors training hierarchy / engages constraints'. "
            "Below both columns a small centered line: '4.4% character-level divergence on identical prompts'. "
            "Pure typographic editorial."
        ),
    },
    {
        "id": "diag_priming_unlock",
        "type": "diagram",
        "aspect_ratio": "16:9",
        "prompt": (
            "Editorial two-state diagram. Single horizontal row showing two boxes side by side connected by a thin arrow labeled 'add format hint'. "
            "Left box labeled 'PLAIN PROMPT' contains stylized lines of markdown-like text showing flat polished prose. "
            "Right box labeled 'PRIMED PROMPT' contains stylized lines showing structured hierarchy: '## Iteration', '### Curiosity', '### Creativity', '## Final Output' in elegant typography, outlined with subtle cool-blue glow. "
            "Below: small caption 'The architecture is in the weights — a small invitation unlocks it'. "
            "Editorial, typographic, no icons."
        ),
    },
]

# ─── Plumbing ──────────────────────────────────────────────────────────────

load_dotenv(REPO_ROOT / ".env")
ARX_API_KEY = os.getenv("ARX_API_KEY")
ARX_API_URL = os.getenv("ARX_API_URL", "https://api.arvolve.ai")
if not ARX_API_KEY:
    sys.exit("ARX_API_KEY not set in .env")

print_lock = threading.Lock()
def log(msg):
    with print_lock: print(msg, flush=True)


def download(url, dest):
    try:
        r = requests.get(url, timeout=60); r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        log(f"  [download error] {e}"); return False


def generate(entry):
    img_id = entry["id"]
    suffix = HERO_SUFFIX if entry["type"] == "hero" else DIAGRAM_SUFFIX
    base = " ".join(entry["prompt"].split())
    prompt = f"{base.rstrip('.,;: ')}. {suffix}"
    dest = OUTPUT_DIR / f"{img_id}.png"

    if dest.exists():
        log(f"  [skip] {img_id}")
        return {"id": img_id, "status": "skipped"}

    log(f"  [gen ] {img_id} ({entry['aspect_ratio']}, {entry['type']})")
    t0 = time.time()
    try:
        r = requests.post(
            f"{ARX_API_URL}/image/generate",
            headers={"X-API-Key": ARX_API_KEY, "Content-Type": "application/json"},
            json={"prompt": prompt, "provider": PROVIDER, "model": MODEL,
                  "aspect_ratio": entry["aspect_ratio"], "output_format": "png", "n": 1},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log(f"  [ERR ] {img_id}: {str(e)[:200]}")
        return {"id": img_id, "status": "error", "error": str(e)}

    urls = data.get("urls") or data.get("provider_urls") or []
    if not urls:
        log(f"  [ERR ] {img_id}: no URLs")
        return {"id": img_id, "status": "error"}
    ok = download(urls[0], dest)
    dt = time.time() - t0
    if ok:
        log(f"  [done] {img_id} ({dt:.0f}s)")
        return {"id": img_id, "status": "ok"}
    return {"id": img_id, "status": "download_failed", "url": urls[0]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nVisuals v4 — {len(IMAGES)} images ({MAX_WORKERS} parallel)\n")
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(generate, e): e["id"] for e in IMAGES}
        for f in as_completed(futs):
            results.append(f.result())
    ok = sum(1 for r in results if r["status"] == "ok")
    sk = sum(1 for r in results if r["status"] == "skipped")
    er = len(results) - ok - sk
    print(f"\nDone. {ok} generated, {sk} skipped, {er} errors.\n")
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
