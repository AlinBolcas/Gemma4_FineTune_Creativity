"""
generate_diagrams_test.py — 5 test diagram images via ArX API.

Direction: Arvolve-palette infographic / diagram aesthetic.
Dark background, blue accents, legible text labels and arrows.
NOT abstract sculpture — useful visual communication.

Usage:
    python src/V_utility/generate_diagrams_test.py

Env vars required (from .env):
    ARX_API_KEY
    ARX_API_URL
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "visuals" / "diagrams"
MODEL = "gpt-image-2"
PROVIDER = "openai"
TIMEOUT = 180

# Diagram-permissive suffix: Arvolve palette + readable layout
ARVOLVE_DIAGRAM_SUFFIX = (
    "Design: sleek dark-mode infographic on deep off-black #0A0A0C background. "
    "Color palette: electric cobalt #3195F0 and sky blue #00A9FF for highlights and connectors, "
    "cool midnight blue #103662 for filled panels and node backgrounds, "
    "stone grey #C0BDBA for secondary labels, clean white #F5F5F5 for primary text. "
    "Typography: crisp sans-serif (Inter-style), generous letter-spacing, clearly legible at all sizes. "
    "Layout: clean grid or flow structure, explicit arrows and connectors, "
    "well-spaced nodes and boxes with rounded corners, hierarchy immediately readable. "
    "Style: premium dark-mode dashboard aesthetic, calm and authoritative, "
    "no photographic elements, no gradients on backgrounds, "
    "thin glowing outlines on key nodes, subtle ambient blue glow on connectors. "
    "Every label legible, every connection purposeful, zero decorative noise."
)

TEST_IMAGES = [
    {
        "id": "diag_pipeline_flow",
        "aspect_ratio": "16:9",
        "prompt": (
            "Flowchart diagram: 11-stage creative reasoning pipeline. "
            "Two parallel vertical streams side by side. "
            "Left stream labeled 'CURIOSITY' with 4 boxes: Map, Expand, Distill, Socratic Output. "
            "Right stream labeled 'CREATIVITY' with 6 boxes: Research Plan, Branch, Develop, Select Prune, Combinatory Mix, Synthesize. "
            "Both streams converge at bottom into a single 'CRITIC' box with a feedback loop arrow pointing back up to both streams. "
            "Horizontal dashed arrow from Socratic Output to Creativity stream labeled 'steering signal'."
        ),
    },
    {
        "id": "diag_three_tiers",
        "aspect_ratio": "16:9",
        "prompt": (
            "Three-column comparison diagram with header row 'Evaluation Tiers'. "
            "Column 1 labeled 'Tier 1 / Vanilla Gemma 4' — icon: plain document, description: 'Baseline behavior. No scaffolding.' "
            "Column 2 labeled 'Tier 2 / Pipeline Scaffolded' — icon: layered stack, description: 'Full 11-stage pipeline at runtime.' "
            "Column 3 labeled 'Tier 3 / Fine-Tuned' — icon: neural weight chip, description: 'Trained weights, no scaffolding. Architecture internalized.' "
            "Right-pointing arrows between columns labeled 'adds pipeline' and 'internalizes'. "
            "Bottom row: 'Held-out prompt set evaluated across all three.' "
            "Column 3 subtly highlighted as the key result."
        ),
    },
    {
        "id": "diag_self_distillation",
        "aspect_ratio": "1:1",
        "prompt": (
            "Circular data flow diagram. "
            "Large ring connecting six labeled nodes clockwise: "
            "'Seed Prompts' → '11-Stage Pipeline' → 'Reasoning Traces' → 'SFT JSONL 334 examples' → 'LoRA Fine-Tune' → 'Gemma 4 Adapted' → back to 'Seed Prompts'. "
            "Center of ring: small text 'Self-Distillation Loop'. "
            "Each arrow labeled with the action: 'generates', 'structures', 'formats', 'trains', 'improves', 'seeds'. "
            "The Gemma 4 node highlighted with subtle glow."
        ),
    },
    {
        "id": "diag_combinatorial_mix",
        "aspect_ratio": "16:9",
        "prompt": (
            "Network graph diagram showing the branch-and-recombine creative process. "
            "Left column: single node 'Input Prompt'. "
            "Middle column: 5 nodes labeled B1, B2, B3, B4, B5 representing distinct creative branches. "
            "Right column: 3 hybrid combination nodes labeled 'Hybrid A', 'Hybrid B', 'Hybrid C'. "
            "Arrows from Input to all 5 branches, labeled 'diverge'. "
            "Cross-arrows from multiple branches to each hybrid, labeled 'recombine'. "
            "Far right: single 'Final Output' node with arrow from best hybrid, labeled 'select'. "
            "B2 and B4 crossed out with faint red X labeled 'pruned'."
        ),
    },
    {
        "id": "diag_temp_spectrum",
        "aspect_ratio": "16:9",
        "prompt": (
            "Horizontal spectrum bar chart. Title: 'Sampling Temperature Effect'. "
            "Full-width gradient bar from left (0.0) to right (1.5). "
            "Left zone 0.0–0.5 colored grey, labeled 'BASELINE — collapses to safe generic output'. "
            "Center zone 0.6–0.8 colored bright blue with glow, outlined box labeled 'SWEET SPOT — structured creative output'. "
            "Right zone 1.0–1.5 colored red-orange, labeled 'UNUSABLE — incoherent noise'. "
            "Tick marks at 0.0, 0.3, 0.5, 0.6, 0.8, 1.0, 1.3, 1.5. "
            "Below the bar: three example output descriptors aligned to each zone."
        ),
    },
]

load_dotenv(REPO_ROOT / ".env")

ARX_API_KEY = os.getenv("ARX_API_KEY")
ARX_API_URL = os.getenv("ARX_API_URL", "https://api.arvolve.ai")

if not ARX_API_KEY:
    sys.exit("ARX_API_KEY not set in .env")


def download_image(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  [download error] {url} → {e}", flush=True)
        return False


def generate_one(entry: dict) -> dict:
    img_id = entry["id"]
    aspect_ratio = entry.get("aspect_ratio", "16:9")
    base = " ".join(entry["prompt"].split())
    prompt = f"{base.rstrip('.,;: ')}. {ARVOLVE_DIAGRAM_SUFFIX}"

    dest_path = OUTPUT_DIR / f"{img_id}.png"
    if dest_path.exists():
        print(f"  [skip] {img_id} already exists", flush=True)
        return {"id": img_id, "status": "skipped"}

    print(f"  [generating] {img_id} ({aspect_ratio}) ...", flush=True)
    t0 = time.time()

    try:
        resp = requests.post(
            f"{ARX_API_URL}/image/generate",
            headers={"X-API-Key": ARX_API_KEY, "Content-Type": "application/json"},
            json={
                "prompt": prompt,
                "provider": PROVIDER,
                "model": MODEL,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "n": 1,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.HTTPError as e:
        print(f"  [error] {img_id}: HTTP {e.response.status_code} — {e.response.text[:300]}", flush=True)
        return {"id": img_id, "status": "error", "error": str(e)}
    except Exception as e:
        print(f"  [error] {img_id}: {e}", flush=True)
        return {"id": img_id, "status": "error", "error": str(e)}

    urls = data.get("urls", []) or data.get("provider_urls", [])
    if not urls:
        print(f"  [error] {img_id}: no URLs — {json.dumps(data)[:300]}", flush=True)
        return {"id": img_id, "status": "error", "error": "no URLs returned"}

    ok = download_image(urls[0], dest_path)
    elapsed = time.time() - t0

    if ok:
        print(f"  [done] {img_id} → diagrams/{img_id}.png ({elapsed:.1f}s)", flush=True)
        return {"id": img_id, "status": "ok", "path": str(dest_path)}
    else:
        sidecar = OUTPUT_DIR / f"{img_id}.url.txt"
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(urls[0])
        print(f"  [partial] {img_id}: generated but download failed — URL saved", flush=True)
        return {"id": img_id, "status": "download_failed", "url": urls[0]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDiagram Test — 5 images")
    print(f"  Output : data/output/visuals/diagrams/")
    print(f"  Model  : {MODEL} via {ARX_API_URL}")
    print()

    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(generate_one, e): e["id"] for e in TEST_IMAGES}
        for future in as_completed(futures):
            results.append(future.result())

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] not in ("ok", "skipped")]
    print(f"\nDone. {len(ok)}/5 generated.")
    if errors:
        for r in errors:
            print(f"  FAIL {r['id']}: {r.get('error', r['status'])}")


if __name__ == "__main__":
    main()
