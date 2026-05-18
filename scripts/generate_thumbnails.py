"""
generate_thumbnails.py — 8 YouTube thumbnail variants via ArX / gpt-image-2.

All 16:9, 1280x720 intent. Bold, viral-thumbnail aesthetic layered over
the Arvolve off-black + blue palette.

Usage:
    python scripts/generate_thumbnails.py
"""

import os, sys, time, json, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

REPO_ROOT  = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "visuals" / "thumbnails"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(REPO_ROOT / ".env")
ARX_API_URL = os.getenv("ARX_API_URL", "https://api.arvolve.ai")
ARX_API_KEY = os.getenv("ARX_API_KEY")
if not ARX_API_KEY:
    sys.exit("ERROR: ARX_API_KEY not set in .env")

MODEL    = "gpt-image-2"
PROVIDER = "openai"
TIMEOUT  = 240

# ── Shared thumbnail DNA ───────────────────────────────────────────────────────
# YouTube thumbnails need: high contrast, bold readable text, strong focal point
BASE = (
    "16:9 YouTube thumbnail. Deep off-black background #0A0A0C. "
    "Single electric-blue accent (#00A9FF to #3195F0). "
    "Photorealistic or hyper-real CG quality, extremely high contrast, "
    "bold and immediately readable at small size. "
    "Premium dark-mode aesthetic. No letterboxing, no border, image fills frame edge to edge. "
    "Text must be legible and large — minimum 10% of frame height for headline. "
    "Composition: strong single focal object, generous negative space. "
    "Style: between a Netflix title card and a Nature magazine cover. "
    "Lighting: controlled studio-grade, single key light, subtle blue rim."
)

THUMBNAILS = [
    {
        "id": "thumb_01_unlock",
        "prompt": (
            BASE +
            " Central object: an ancient stone vault door cracked open revealing blinding blue-white light pouring through. "
            "The crack forms the shape of a human neural network / branching tree. "
            "Bold white all-caps headline at top: 'I TAUGHT AI TO BE CREATIVE'. "
            "Subtitle below in electric blue: 'Gemma 4 Fine-Tune'. "
            "Dramatic, cinematic, awe-inspiring."
        ),
    },
    {
        "id": "thumb_02_brain_break",
        "prompt": (
            BASE +
            " A geometric mechanical brain made of dark stone and bronze, cracking open from the inside — "
            "blue light erupting through fracture lines, crystalline ideas forming at the cracks. "
            "Top-left corner: bold white text 'WHAT IF AI COULD ACTUALLY THINK?'. "
            "Bottom-right: small blue text 'Gemma 4 · Creative Reasoning'. "
            "Cinematic tension, feels revelatory."
        ),
    },
    {
        "id": "thumb_03_before_after",
        "prompt": (
            BASE +
            " Split composition: left half desaturated grey — a plain dull stone monolith labeled 'VANILLA' in muted grey. "
            "Right half vibrant — same monolith but cracked open with electric-blue branching crystalline growth exploding outward, "
            "labeled 'FINE-TUNED' in bright white. "
            "Sharp vertical dividing line between halves. "
            "Bold headline centered at top: 'THIS CHANGES EVERYTHING'. "
            "The contrast between the two halves is extreme and striking."
        ),
    },
    {
        "id": "thumb_04_11_stages",
        "prompt": (
            BASE +
            " Large bold number '11' dominating the left two-thirds of the frame, rendered in brushed dark steel "
            "with electric-blue light tracing the inner edges. "
            "Right side: a vertical chain of glowing nodes, each connected by thin blue lines, representing pipeline stages. "
            "Headline in bright white at top: '11 STAGES TO TEACH AI CREATIVITY'. "
            "Subtext in electric blue at bottom: 'Gemma 4 Fine-Tune · Kaggle'. "
            "Bold, graphic, infographic-meets-cinematic."
        ),
    },
    {
        "id": "thumb_05_winner_takes_all",
        "prompt": (
            BASE +
            " A single dramatic spotlight illuminating a lone polished stone obelisk on a vast dark plane, "
            "electric-blue vein of light running vertically through its center. "
            "Other stone obelisks visible in the far dark background, unlit. "
            "The lit obelisk is clearly the winner. "
            "Bold headline at top in white: 'CREATIVITY = THE UNLOCK TO AGI'. "
            "Bottom: 'Winner Takes All · Gemma 4 Research'. "
            "Cinematic, lonely, powerful. High-contrast spotlight effect."
        ),
    },
    {
        "id": "thumb_06_pipeline_visual",
        "prompt": (
            BASE +
            " Raw dark ore enters a sleek bronze-and-glass industrial machine on the left. "
            "Glowing blue crystalline gems emerge from the right output. "
            "Above the machine: clean editorial labels 'CURIOSITY → CREATIVITY → CRITIC'. "
            "Bold white headline at top: 'THE CREATIVE PIPELINE'. "
            "Subtitle: 'How I Fine-Tuned Gemma 4 to Think'. "
            "The machine is dramatic and beautiful — equal parts sci-fi and editorial."
        ),
    },
    {
        "id": "thumb_07_self_improvement",
        "prompt": (
            BASE +
            " A serpent of electric-blue light forming a perfect circle (Ouroboros), eating its own tail. "
            "Inside the circle: the words 'SELF-IMPROVEMENT' in large bold white. "
            "Outside top: 'THE MISSING PIECE IN AI'. "
            "Outside bottom in blue: 'Creative Reasoning · Gemma 4'. "
            "The ouroboros is made of flowing data streams and crystalline nodes. "
            "Mystical, technical, and visually arresting."
        ),
    },
    {
        "id": "thumb_08_kaggle_submission",
        "prompt": (
            BASE +
            " Heroic wide-angle shot of a single figure standing before an enormous glowing portal made of blue light, "
            "branching neural patterns radiating outward from the portal's edges. "
            "The figure is small, silhouetted, looking up — sense of scale and discovery. "
            "Bold headline at top: 'GEMMA 4 CREATIVE FINE-TUNE'. "
            "Subtitle in electric blue: 'Kaggle Hackathon 2026 · Alin Bolcas'. "
            "Epic, cinematic, adventurous. Wide establishing shot feel."
        ),
    },
]

# ── API call ───────────────────────────────────────────────────────────────────

def generate(thumb: dict) -> dict:
    tid   = thumb["id"]
    dest  = OUTPUT_DIR / f"{tid}.png"
    if dest.exists():
        print(f"  [skip] {tid}")
        return {"id": tid, "status": "skipped"}

    print(f"  [gen]  {tid}", flush=True)
    t0 = time.time()

    try:
        r = requests.post(
            f"{ARX_API_URL}/image/generate",
            headers={"X-API-Key": ARX_API_KEY, "Content-Type": "application/json"},
            json={"prompt": thumb["prompt"], "provider": PROVIDER, "model": MODEL,
                  "aspect_ratio": "16:9", "output_format": "png", "n": 1},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"  [err]  {tid}: {e}")
        return {"id": tid, "status": "error", "error": str(e)}

    urls = data.get("urls") or data.get("provider_urls", [])
    if not urls:
        print(f"  [err]  {tid}: no URL — {json.dumps(data)[:200]}")
        return {"id": tid, "status": "error"}

    dl = requests.get(urls[0], timeout=120)
    dest.write_bytes(dl.content)
    print(f"  [ok]   {tid} — {dest.stat().st_size//1024} KB  {time.time()-t0:.1f}s")
    return {"id": tid, "status": "ok", "path": dest}


def main():
    print(f"\nGenerating {len(THUMBNAILS)} YouTube thumbnails (16:9)\n")
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(generate, t): t for t in THUMBNAILS}
        for f in as_completed(futures):
            results.append(f.result())

    ok  = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    err = sum(1 for r in results if r["status"] == "error")
    print(f"\nDone: {ok} ok  {err} errors")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
