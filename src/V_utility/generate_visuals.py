"""
generate_visuals.py — Batch image generation via ArX API.

Reads src/V_utility/image_prompts.yaml and generates all images
in parallel using gpt-image-2 through the ArX /image/generate endpoint.
Saves results to data/output/visuals/<category>/<id>.png

Usage:
    python src/V_utility/generate_visuals.py

Env vars required (from .env):
    ARX_API_KEY
    ARX_API_URL
"""

import os
import sys
import time
import json
import yaml
import requests
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# ── Config ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROMPTS_FILE = REPO_ROOT / "src" / "V_utility" / "image_prompts.yaml"
OUTPUT_DIR = REPO_ROOT / "data" / "output" / "visuals"
MAX_WORKERS = 5           # parallel requests (ArX API rate limit headroom)
MODEL = "gpt-image-2"
PROVIDER = "openai"
TIMEOUT = 180             # seconds per request

# Canonical ARVOLVE aesthetic preset (mirrors ArX promptTemplates.IMAGE_PRESETS["ARVOLVE"])
# Appended to every prompt for unified brand-aligned output.
ARVOLVE_SUFFIX = (
    "Style: premium, futuristic, minimal, high-end, clean, computational elegance, calm confidence. "
    "Form language: smooth and crisp geometry, subtle flow with controlled fracture, "
    "intentional micro-detail (toolmark and digital artifact accents, not noise), "
    "engineered negative space, strong silhouette and structure. "
    "Palette: deep restrained off-black base, neutral metallic and mineral tones, "
    "sparse cool-blue energy accents as glow or edge-light or core, "
    "optional tiny warm bronze metallic glints as secondary accents, minimal. "
    "Lighting: crisp studio lighting, high dynamic range, clean speculars, "
    "controlled bloom on energy accents only, no bokeh, no heavy depth of field. "
    "Composition: uncluttered, generous breathing room, premium hierarchy, "
    "no text, no labels, no diagrams."
)

load_dotenv(REPO_ROOT / ".env")

ARX_API_KEY = os.getenv("ARX_API_KEY")
ARX_API_URL = os.getenv("ARX_API_URL", "https://api.arvolve.ai")

if not ARX_API_KEY:
    sys.exit("ARX_API_KEY not set in .env")

# ── Helpers ─────────────────────────────────────────────────────────────────

print_lock = threading.Lock()

def log(msg: str):
    with print_lock:
        print(msg, flush=True)


def download_image(url: str, dest: Path) -> bool:
    """Download image from URL to dest path. Returns True on success."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        log(f"  [download error] {url} → {e}")
        return False


def generate_one(entry: dict) -> dict:
    """Generate a single image. Returns result dict."""
    img_id = entry["id"]
    category = entry.get("category", "misc")
    aspect_ratio = entry.get("aspect_ratio", "1:1")
    base = " ".join(entry["prompt"].split())  # collapse whitespace
    prompt = f"{base.rstrip('.,;: ')}. {ARVOLVE_SUFFIX}"

    dest_dir = OUTPUT_DIR / category
    dest_path = dest_dir / f"{img_id}.png"

    if dest_path.exists():
        log(f"  [skip] {img_id} already exists")
        return {"id": img_id, "status": "skipped", "path": str(dest_path)}

    log(f"  [generating] {img_id} ({aspect_ratio}) ...")
    t0 = time.time()

    try:
        resp = requests.post(
            f"{ARX_API_URL}/image/generate",
            headers={
                "X-API-Key": ARX_API_KEY,
                "Content-Type": "application/json",
            },
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
        log(f"  [error] {img_id}: HTTP {e.response.status_code} — {e.response.text[:200]}")
        return {"id": img_id, "status": "error", "error": str(e)}
    except Exception as e:
        log(f"  [error] {img_id}: {e}")
        return {"id": img_id, "status": "error", "error": str(e)}

    urls = data.get("urls", [])
    if not urls:
        # fallback: try provider_urls
        urls = data.get("provider_urls", [])

    if not urls:
        log(f"  [error] {img_id}: no URLs in response — {json.dumps(data)[:200]}")
        return {"id": img_id, "status": "error", "error": "no URLs returned"}

    image_url = urls[0]
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = download_image(image_url, dest_path)
    elapsed = time.time() - t0

    if ok:
        log(f"  [done] {img_id} → {dest_path.relative_to(REPO_ROOT)} ({elapsed:.1f}s)")
        return {"id": img_id, "status": "ok", "path": str(dest_path), "url": image_url}
    else:
        # save the URL in a sidecar so we can retry manually
        sidecar = dest_dir / f"{img_id}.url.txt"
        sidecar.write_text(image_url)
        log(f"  [partial] {img_id}: generated but download failed — URL saved to {sidecar.name}")
        return {"id": img_id, "status": "download_failed", "url": image_url}


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    if not PROMPTS_FILE.exists():
        sys.exit(f"Prompts file not found: {PROMPTS_FILE}")

    with open(PROMPTS_FILE) as f:
        config = yaml.safe_load(f)

    entries = config.get("images", [])
    if not entries:
        sys.exit("No images defined in prompts file.")

    print(f"\nGemma 4 Visuals Generator")
    print(f"  Prompts : {PROMPTS_FILE.relative_to(REPO_ROOT)}")
    print(f"  Output  : {OUTPUT_DIR.relative_to(REPO_ROOT)}")
    print(f"  Model   : {MODEL} via {ARX_API_URL}")
    print(f"  Images  : {len(entries)} total, {MAX_WORKERS} parallel")
    print()

    # Optional: filter by category or id via args
    if len(sys.argv) > 1:
        filter_val = sys.argv[1].lower()
        entries = [e for e in entries if filter_val in e["id"].lower() or filter_val in e.get("category","").lower()]
        print(f"  Filter  : '{filter_val}' → {len(entries)} matched")
        print()

    if not entries:
        sys.exit("No matching entries after filter.")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, e): e["id"] for e in entries}
        for future in as_completed(futures):
            results.append(future.result())

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] not in ("ok", "skipped")]

    print()
    print(f"Done.")
    print(f"  Generated : {len(ok)}")
    print(f"  Skipped   : {len(skipped)} (already exist)")
    print(f"  Errors    : {len(errors)}")
    if errors:
        print()
        print("Failed:")
        for r in errors:
            print(f"  {r['id']}: {r.get('error', r['status'])}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Manifest  : {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
