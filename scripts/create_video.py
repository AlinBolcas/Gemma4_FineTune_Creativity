"""
create_video.py — Automated presentation → MP4.

1. Starts a local HTTP server
2. Screenshots every slide via headless Chromium (Playwright)
3. Combines each screenshot + its WAV voiceover with ffmpeg
4. Concatenates segments into a final MP4

Usage:
    python scripts/create_video.py

Output:
    data/output/video/gemma4_creative_presentation.mp4
"""

import asyncio
import shutil
import subprocess
import time
import wave
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
AUDIO_DIR    = REPO_ROOT / "presentation" / "audio"
OUTPUT_DIR   = REPO_ROOT / "data" / "output" / "video"
TEMP_DIR     = OUTPUT_DIR / "_tmp"
OUTPUT_MP4   = OUTPUT_DIR / "gemma4_creative_presentation.mp4"
PORT         = 8766
TOTAL_SLIDES = 15
PAUSE_END    = 0.8   # silence after each slide's audio (seconds)
TRANSITION   = 0.9   # wait after ArrowRight for CSS transition to finish (seconds)


# ── Helpers ────────────────────────────────────────────────────────────────────

def wav_duration(path: Path) -> float:
    """Use ffprobe for robust duration — handles all WAV subtypes."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 5.0


def check_deps():
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found — brew install ffmpeg")
    try:
        from playwright.async_api import async_playwright  # noqa
    except ImportError:
        raise SystemExit("playwright not installed — pip install playwright && playwright install chromium")


# ── Screenshot capture ─────────────────────────────────────────────────────────

async def capture_slides() -> list[Path]:
    from playwright.async_api import async_playwright

    screenshots = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        url = f"http://localhost:{PORT}/presentation/"
        print(f"  Loading {url}")
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(1.0)   # let JS charts + animations settle

        # Dismiss the record overlay so the presentation is visible
        await page.click("#rec-skip")
        await asyncio.sleep(0.5)

        for i in range(1, TOTAL_SLIDES + 1):
            path = TEMP_DIR / f"slide_{i:02d}.png"
            await page.screenshot(path=str(path), full_page=False)
            screenshots.append(path)
            print(f"  [screenshot] slide {i:02d}", flush=True)

            if i < TOTAL_SLIDES:
                await page.keyboard.press("ArrowRight")
                await asyncio.sleep(TRANSITION)

        await browser.close()
    return screenshots


# ── Video assembly ─────────────────────────────────────────────────────────────

def build_segment(i: int, img: Path, audio: Path | None, duration: float) -> Path:
    seg = TEMP_DIR / f"seg_{i:02d}.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-t", str(duration), "-i", str(img),
    ]
    if audio and audio.exists():
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k",
                "-af", f"apad=pad_dur={PAUSE_END}"]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", "30",
        # fit to 1920×1080, letterbox if needed
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,"
               "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
        str(seg),
    ]
    subprocess.run(cmd, check=True)
    return seg


def concat_segments(segments: list[Path]) -> None:
    concat_txt = TEMP_DIR / "concat.txt"
    concat_txt.write_text("\n".join(f"file '{s.resolve()}'" for s in segments))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_txt),
        "-c", "copy", str(OUTPUT_MP4),
    ], check=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    check_deps()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Start HTTP server (serve from repo root so ../data paths resolve)
    print(f"\nStarting server on :{PORT}...")
    server = subprocess.Popen(
        ["python3", "-m", "http.server", str(PORT), "--bind", "127.0.0.1"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)

    try:
        # ── Screenshots ──
        print("\nCapturing slides (headless Chromium)...\n")
        screenshots = asyncio.run(capture_slides())

        # ── Build segments ──
        print("\nBuilding video segments (ffmpeg)...\n")
        segments = []
        for i, img in enumerate(screenshots, 1):
            audio    = AUDIO_DIR / f"slide_{i:02d}.wav"
            duration = wav_duration(audio) + PAUSE_END if audio.exists() else 5.0
            seg      = build_segment(i, img, audio, duration)
            segments.append(seg)
            print(f"  [video] segment {i:02d}: {duration:.1f}s", flush=True)

        # ── Concat ──
        print("\nConcatenating into final MP4...")
        concat_segments(segments)

        size_mb = OUTPUT_MP4.stat().st_size / 1_000_000
        print(f"\n{'─'*50}")
        print(f"Done!  {OUTPUT_MP4.name}  ({size_mb:.1f} MB)")
        print(f"Path:  {OUTPUT_MP4}")

    finally:
        server.terminate()
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
