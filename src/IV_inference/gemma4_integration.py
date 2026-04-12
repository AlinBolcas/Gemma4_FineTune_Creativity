"""
gemma4_integration.py — Gemma 4 wrapper with memory, thinking, and multimodal.

Class: Gemma4
    .chat(msg)                        — text, with memory
    .chat(msg, images=["url/path"])   — image(s) + text, with memory
    .chat(msg, audio="path")          — audio + text, with memory (E2B/E4B only)
    .generate(system, user)           — stateless text (pipeline use)
    .generate(system, user, images=)  — stateless multimodal
    .structured(prompt)               — returns dict
    .generate_fn()                    — returns callable for runner.py

Thinking toggle:
    g = Gemma4("e2b", thinking=True)          # on by default
    g.thinking = False                         # toggle at runtime
    g.chat("explain this", thinking=True)      # override per call

Run as chat TUI:
    python src/IV_inference/gemma4_integration.py
"""

import os
import json
import torch
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "e2b": {"id": "google/gemma-4-E2B-it", "ram_gb": 8,  "multimodal": True,  "description": "Smallest, fastest. Text + image + audio."},
    "e4b": {"id": "google/gemma-4-E4B-it", "ram_gb": 16, "multimodal": True,  "description": "Best quality/speed. Text + image + audio."},
    "26b": {"id": "google/gemma-4-26B-A4B-it", "ram_gb": 40, "multimodal": False, "description": "MoE. Text + image. A100."},
    "31b": {"id": "google/gemma-4-31B-it", "ram_gb": 60, "multimodal": False, "description": "Dense. Highest quality. A100."},
}
DEFAULT_MODEL = "e2b"


def resolve_model_id(alias_or_id: str) -> str:
    key = alias_or_id.lower().strip()
    return MODELS[key]["id"] if key in MODELS else alias_or_id


def list_models() -> str:
    lines = [f"{'Alias':<6}  {'RAM':>5}  {'MM':>4}  {'HF Model ID':<35}  Description", "-" * 88]
    for alias, info in MODELS.items():
        mark = " <-- default" if alias == DEFAULT_MODEL else ""
        mm = "yes" if info["multimodal"] else " - "
        lines.append(f"{alias:<6}  {info['ram_gb']:>4}G  {mm:>4}  {info['id']:<35}  {info['description']}{mark}")
    return "\n".join(lines)


def download_model(alias_or_id: str) -> str:
    model_id = resolve_model_id(alias_or_id)
    hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
    from huggingface_hub import snapshot_download
    print(f"  Downloading {model_id}...")
    path = snapshot_download(repo_id=model_id, token=hf_token)
    print(f"  Cached: {path}")
    return path


# Backward compat: returns a callable like before
def load_gemma4(model: str = DEFAULT_MODEL, **kwargs):
    g = Gemma4(model, **kwargs)
    return g.generate_fn()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def _detect_dtype(device: str):
    if device == "cuda": return torch.bfloat16
    if device == "mps":  return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Gemma4 class
# ---------------------------------------------------------------------------

class Gemma4:
    """
    Gemma 4 wrapper — text, image, audio, thinking mode, conversation memory.

    Args:
        model:          alias (e2b, e4b, 26b, 31b) or full HF model ID
        system:         default system message
        thinking:       enable Gemma 4 thinking mode globally (toggle per call too)
        device:         override device detection
        max_new_tokens: generation length cap
        temperature:    sampling temperature
        max_history:    max stored messages (auto-trims oldest turns)
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system: str = "You are a helpful assistant.",
        thinking: bool = False,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 64,
        max_history: int = 40,
    ):
        self.model_id = resolve_model_id(model)
        self.alias = next((k for k, v in MODELS.items() if v["id"] == self.model_id), model)
        self.system_message = system
        self.thinking = thinking
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_history = max_history
        self.history: list[dict] = []

        hf_token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)

        dev = device or _detect_device()
        dtype = _detect_dtype(dev)
        model_info = MODELS.get(self.alias, {})
        is_multimodal = model_info.get("multimodal", False)

        print(f"\n  Model : {self.alias} ({self.model_id})")
        print(f"  Device: {dev}  |  dtype: {dtype}  |  multimodal: {is_multimodal}  |  thinking: {thinking}")

        from transformers import AutoProcessor

        print("  Loading processor...")
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=hf_token)

        print("  Loading weights...")
        load_kwargs = {"dtype": dtype, "token": hf_token}

        # Use multimodal model class for E2B/E4B (supports image+audio)
        # Use CausalLM for 26B/31B (text+image, no audio)
        if is_multimodal:
            from transformers import AutoModelForMultimodalLM
            if dev == "cuda":
                load_kwargs["device_map"] = "auto"
            self.model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **load_kwargs)
        else:
            from transformers import AutoModelForCausalLM
            if dev == "cuda":
                load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

        if dev == "mps":
            self.model = self.model.to("mps")
        self.model.eval()

        if dev == "cuda":
            print(f"  VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Ready: {self.alias}\n")

    # -------------------------------------------------------------------------
    # Core generation (stateless, no memory)
    # -------------------------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        images: Optional[list] = None,
        audio: Optional[str] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        """
        Stateless generation. Used by the pipeline runner.

        Args:
            system:  system prompt
            user:    user message text
            images:  list of image URLs or local paths (optional)
            audio:   path to audio file (optional, E2B/E4B only)
            thinking: override thinking mode for this call
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _build_content(user, images, audio)},
        ]
        return self._run(messages, thinking=thinking)

    def generate_fn(self):
        """Return a stateless callable(system, user) -> str for pipeline.runner."""
        def fn(system: str, user: str) -> str:
            return self.generate(system, user)
        fn.model_id = self.model_id
        fn.alias = self.alias
        return fn

    # -------------------------------------------------------------------------
    # Chat with memory
    # -------------------------------------------------------------------------

    def chat(
        self,
        message: str,
        images: Optional[list] = None,
        audio: Optional[str] = None,
        system: Optional[str] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        """
        Send a message with conversation memory.

        Args:
            message:  text message
            images:   list of image URLs or local paths (optional)
            audio:    path to audio file (optional, E2B/E4B only)
            system:   override system message for this call
            thinking: override thinking mode for this call
        """
        messages = [{"role": "system", "content": system or self.system_message}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": _build_content(message, images, audio)})

        response = self._run(messages, thinking=thinking)

        # Store text-only in history (images not re-sent on follow-ups)
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return response

    def clear(self):
        """Clear conversation history."""
        self.history.clear()

    def get_history(self) -> list[dict]:
        return list(self.history)

    # -------------------------------------------------------------------------
    # Structured JSON output
    # -------------------------------------------------------------------------

    def structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        images: Optional[list] = None,
    ) -> dict:
        """Generate structured JSON. Optionally pass a schema hint."""
        sys_msg = (system or self.system_message) + "\n\nReturn ONLY valid JSON. No markdown, no explanation."
        if schema:
            sys_msg += f"\n\nExpected shape:\n{json.dumps(schema, indent=2)}"
        raw = self.generate(sys_msg, prompt, images=images, thinking=False)
        return _extract_json(raw) or {}

    # -------------------------------------------------------------------------
    # Internal: run inference
    # -------------------------------------------------------------------------

    def _run(self, messages: list[dict], thinking: Optional[bool] = None) -> str:
        enable_thinking = thinking if thinking is not None else self.thinking
        has_media = _messages_have_media(messages)

        if has_media:
            # Multimodal path: apply_chat_template tokenizes directly
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        else:
            # Text-only path: apply_chat_template → processor
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
            )

        raw_response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # parse_response extracts {"thinking": ..., "content": ...}
        try:
            parsed = self.processor.parse_response(raw_response)
            if isinstance(parsed, dict):
                return parsed.get("content", parsed.get("thinking", str(parsed)))
            return str(parsed)
        except Exception:
            # fallback: strip special tokens manually
            return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def __repr__(self):
        return f"Gemma4(alias={self.alias}, thinking={self.thinking}, history={len(self.history)} msgs)"


# ---------------------------------------------------------------------------
# Content builder (text + images + audio)
# ---------------------------------------------------------------------------

def _build_content(text: str, images: Optional[list] = None, audio: Optional[str] = None):
    """Build message content: str for text-only, list for multimodal."""
    if not images and not audio:
        return text
    content = []
    # Images before text (Gemma 4 convention)
    if images:
        for img in images:
            if img.startswith(("http://", "https://")):
                content.append({"type": "image", "url": img})
            else:
                content.append({"type": "image", "image": img})
    if audio:
        content.append({"type": "audio", "audio": audio})
    content.append({"type": "text", "text": text})
    return content


def _messages_have_media(messages: list[dict]) -> bool:
    """Check if any message contains non-text content."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            return True
    return False


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for marker in ("```json", "```"):
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass
    brace = text.find("{")
    if brace >= 0:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# Interactive TUI
# ---------------------------------------------------------------------------

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"


def _tui():
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Gemma 4 Chat{RESET}  — with memory, thinking, multimodal")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    # Model
    print(f"\n{YELLOW}Model:{RESET}")
    aliases = list(MODELS.keys())
    for i, alias in enumerate(aliases, 1):
        info = MODELS[alias]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        mm = f" {GREY}[mm]{RESET}" if info["multimodal"] else ""
        print(f"  [{i}] {alias:<5} {info['ram_gb']:>3}GB{mm}  {info['description']}{default}")
    print(f"  [d] Download only")
    raw = input(f"\n  Choice [1]: ").strip() or "1"

    if raw.lower() == "d":
        print(f"\n{YELLOW}Download which?{RESET}")
        for i, a in enumerate(aliases, 1): print(f"  [{i}] {a}")
        print(f"  [a] all")
        dl = input("  > ").strip().lower() or "1"
        if dl == "a":
            for a in aliases: download_model(a)
        elif dl.isdigit() and 1 <= int(dl) <= len(aliases):
            download_model(aliases[int(dl) - 1])
        return

    model_alias = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]

    # Thinking
    print(f"\n{YELLOW}Thinking mode? [y/N]{RESET}")
    print(f"  {GREY}Gemma 4 reasons step-by-step before answering when enabled.{RESET}")
    thinking = input("  > ").strip().lower() in ("y", "yes")

    g = Gemma4(model_alias, thinking=thinking)

    thinking_status = f"{GREEN}ON{RESET}" if g.thinking else f"{GREY}off{RESET}"
    print(f"{GREEN}Ready.{RESET}  thinking={thinking_status}  |  multimodal: {MODELS.get(g.alias, {}).get('multimodal', '?')}")
    print(f"{GREY}Commands:")
    print(f"  /think      — toggle thinking on/off")
    print(f"  /clear      — clear memory")
    print(f"  /system     — change system message")
    print(f"  /history    — show turns in memory")
    print(f"  /image <url or path> <prompt>  — send image + text")
    print(f"  /json <prompt>                 — structured JSON output")
    print(f"  q           — quit{RESET}")

    while True:
        try:
            user_input = input(f"\n  {BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

        if user_input == "/think":
            g.thinking = not g.thinking
            status = f"{GREEN}ON{RESET}" if g.thinking else f"{GREY}off{RESET}"
            print(f"  Thinking mode: {status}")
            continue

        if user_input == "/clear":
            g.clear()
            print(f"  {GREY}History cleared.{RESET}")
            continue

        if user_input == "/system":
            new_sys = input("  New system message: ").strip()
            if new_sys:
                g.system_message = new_sys
                print(f"  {GREEN}Updated.{RESET}")
            continue

        if user_input == "/history":
            if not g.history:
                print(f"  {GREY}(empty){RESET}")
            else:
                for msg in g.history:
                    col = CYAN if msg["role"] == "user" else GREEN
                    print(f"  {col}[{msg['role']}]{RESET} {str(msg['content'])[:120]}")
            continue

        if user_input.startswith("/image "):
            # /image <url_or_path> <prompt text>
            rest = user_input[7:].strip()
            parts = rest.split(" ", 1)
            img_src = parts[0]
            prompt = parts[1] if len(parts) > 1 else "Describe this image."
            print(f"  {GREY}Sending image...{RESET}")
            response = g.chat(prompt, images=[img_src])
            print(f"\n  {CYAN}[{g.alias}]{RESET} {response}")
            print(f"  {GREY}({len(g.history)//2} turns){RESET}")
            continue

        if user_input.startswith("/json "):
            prompt = user_input[6:].strip()
            result = g.structured(prompt)
            print(f"\n  {CYAN}{json.dumps(result, indent=2)}{RESET}")
            continue

        # Normal chat with memory + current thinking setting
        response = g.chat(user_input)
        thinking_tag = f" {GREY}[thinking]{RESET}" if g.thinking else ""
        print(f"\n  {CYAN}[{g.alias}]{thinking_tag}{RESET} {response}")
        print(f"  {GREY}({len(g.history)//2} turns in memory){RESET}")

    print(f"\n  {GREY}Bye.{RESET}")


if __name__ == "__main__":
    _tui()
