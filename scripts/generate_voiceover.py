"""
generate_voiceover.py — Per-slide TTS via Replicate (resemble-ai/chatterbox-turbo).

Voice-clones from docs/voice_reference.mp3 so the output sounds like the presenter.

Process per slide:
  1. Read segment text from VOICEOVER_SCRIPT.md
  2. Split into ≤490-char chunks at sentence boundaries (API limit: 500 chars)
  3. Run one replicate prediction per chunk
  4. Concatenate MP3 bytes → presentation/audio/slide_XX.mp3

Usage:
    pip install replicate
    python scripts/generate_voiceover.py

Env vars (.env):
    REPLICATE_API_TOKEN   required
"""

import os
import re
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPT_FILE = REPO_ROOT / "VOICEOVER_SCRIPT.md"
VOICE_REF   = REPO_ROOT / "docs" / "voice_reference.mp3"
AUDIO_DIR   = REPO_ROOT / "presentation" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(REPO_ROOT / ".env")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_TOKEN:
    sys.exit("ERROR: REPLICATE_API_TOKEN not set in .env")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_TOKEN

try:
    import replicate
except ImportError:
    sys.exit("ERROR: replicate not installed — run: pip install replicate")

MODEL      = "resemble-ai/chatterbox-turbo"
MAX_CHARS  = 490   # stay under the 500-char API limit
TEMPERATURE = 0.7  # natural, not too varied
TOP_K      = 1000
TOP_P      = 0.95
REP_PENALTY = 1.2


# ── Text utils ─────────────────────────────────────────────────────────────────

def split_chunks(text: str) -> list[str]:
    """Split at sentence boundaries keeping each chunk ≤ MAX_CHARS."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current = [], ""
    for s in sentences:
        if not current:
            current = s
        elif len(current) + 1 + len(s) <= MAX_CHARS:
            current += " " + s
        else:
            chunks.append(current)
            current = s
    if current:
        chunks.append(current)
    return chunks


# ── Script parsing ─────────────────────────────────────────────────────────────

def parse_segments(md_path: Path) -> list[dict]:
    text   = md_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n## Slide ", text)
    segs   = []
    for block in blocks[1:]:
        header, _, body = block.partition("\n")
        m = re.match(r"(\d+)\s*[—-]\s*(.+)", header.strip())
        if not m:
            continue
        segs.append({"num": int(m.group(1)), "title": m.group(2).strip(), "text": body.strip()})
    return sorted(segs, key=lambda s: s["num"])


# ── Reference audio upload ─────────────────────────────────────────────────────

def get_ref_url() -> str:
    """Upload voice_reference.mp3 to Replicate once, return public URL."""
    if not VOICE_REF.exists():
        sys.exit(f"ERROR: voice reference not found at {VOICE_REF}")

    print(f"  Uploading voice reference ({VOICE_REF.stat().st_size // 1024} KB)...")
    try:
        with open(VOICE_REF, "rb") as f:
            file_obj = replicate.files.create(f, filename="voice_reference.mp3")
        url = file_obj.urls["get"]
        print(f"  Reference URL: {url[:60]}...")
        return url
    except AttributeError:
        # Older replicate library — fall back to returning the local path
        # replicate will handle the upload per-call when given a file object
        print("  [note] replicate.files.create not available — will upload per call")
        return None


# ── Single TTS chunk ───────────────────────────────────────────────────────────

def tts_chunk(text: str, ref_url: str | None) -> bytes:
    """Run one Replicate prediction, return MP3 bytes."""
    inp = {
        "text":               text,
        "top_k":              TOP_K,
        "top_p":              TOP_P,
        "temperature":        TEMPERATURE,
        "repetition_penalty": REP_PENALTY,
    }
    if ref_url:
        inp["reference_audio"] = ref_url
    else:
        # Pass file object directly — library uploads it
        inp["reference_audio"] = open(VOICE_REF, "rb")

    output = replicate.run(MODEL, input=inp)

    # output is a FileOutput with .read() method
    if hasattr(output, "read"):
        return output.read()
    # Or a URL string
    r = requests.get(str(output), timeout=60)
    r.raise_for_status()
    return r.content


# ── Per-slide generation ───────────────────────────────────────────────────────

def generate_slide(seg: dict, ref_url: str | None) -> dict:
    num   = seg["num"]
    title = seg["title"]
    text  = seg["text"]
    dest  = AUDIO_DIR / f"slide_{num:02d}.wav"

    if dest.exists():
        print(f"  [skip] slide {num:02d} — already exists")
        return {"num": num, "status": "skipped"}

    chunks = split_chunks(text)
    print(f"  [tts]  slide {num:02d}: {title} ({len(chunks)} chunk{'s' if len(chunks) > 1 else ''})", flush=True)

    t0    = time.time()
    parts = []
    for i, chunk in enumerate(chunks, 1):
        if len(chunks) > 1:
            print(f"         chunk {i}/{len(chunks)}: {len(chunk)} chars", flush=True)
        try:
            audio_bytes = tts_chunk(chunk, ref_url)
            parts.append(audio_bytes)
        except Exception as e:
            print(f"  [error] slide {num:02d} chunk {i}: {e}")
            return {"num": num, "status": "error", "error": str(e)}

    # Concatenate MP3 frames — byte-level concat works for MP3
    dest.write_bytes(b"".join(parts))
    elapsed = time.time() - t0
    size_kb = dest.stat().st_size // 1024
    print(f"  [ok]   slide {num:02d}: {size_kb} KB — {elapsed:.1f}s")
    return {"num": num, "status": "ok"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not SCRIPT_FILE.exists():
        sys.exit(f"ERROR: {SCRIPT_FILE} not found")

    segs = parse_segments(SCRIPT_FILE)
    if not segs:
        sys.exit("ERROR: no ## Slide XX sections found in VOICEOVER_SCRIPT.md")

    print(f"\nGenerating voiceover — {len(segs)} slides")
    print(f"Model: {MODEL}  Voice-clone: {VOICE_REF.name}")
    print(f"Output: {AUDIO_DIR}\n")

    ref_url = get_ref_url()
    print()

    results = [generate_slide(seg, ref_url) for seg in segs]

    ok     = [r for r in results if r["status"] in ("ok", "skipped")]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'─'*50}")
    print(f"Done: {len(ok)} ok  {len(errors)} errors")
    if errors:
        for e in errors:
            print(f"  Slide {e['num']:02d}: {e.get('error')}")
    else:
        print("All audio files ready.")
        print(f"\nNext: open  http://localhost:8000?mode=record")


if __name__ == "__main__":
    main()
