"""
ollama_integration.py - Ollama mirror of the Gemma4 wrapper.

Same shape as src/IV_inference/gemma4_integration.py, but routed through
the local Ollama server instead of Hugging Face transformers.

Class:
    OllamaGemma4
        .chat(msg)
        .generate(system, user)
        .structured(prompt)
        .generate_fn()
        .clear()
        .get_history()

Run:
    python src/IV_inference/ollama_integration.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
import ollama

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# ---------------------------------------------------------------------------
# Ollama model registry
# ---------------------------------------------------------------------------

# These are convenience defaults. If your local tags differ, just type them.
MODELS = {
    "e2b": {
        "id": "gemma4:e2b",
        "multimodal": True,
        "description": "Fast local iteration. Adjust if your local Ollama tag differs.",
    },
    "e4b": {
        "id": "gemma4:e4b",
        "multimodal": True,
        "description": "Better quality local Gemma 4. Proven in archive references.",
    },
    "26b": {
        "id": "gemma4:26b",
        "multimodal": True,
        "description": "Large local model if available in your Ollama setup.",
    },
    "31b": {
        "id": "gemma4:31b",
        "multimodal": True,
        "description": "Largest local Gemma 4 tag if available.",
    },
}
DEFAULT_MODEL = "e2b"


def resolve_model_id(alias_or_id: str) -> str:
    key = alias_or_id.lower().strip()
    return MODELS[key]["id"] if key in MODELS else alias_or_id


def list_models() -> str:
    lines = [f"{'Alias':<6}  {'Local Tag':<22}  Description", "-" * 80]
    for alias, info in MODELS.items():
        mark = " <-- default" if alias == DEFAULT_MODEL else ""
        lines.append(f"{alias:<6}  {info['id']:<22}  {info['description']}{mark}")
    return "\n".join(lines)


def pull_model(alias_or_id: str):
    model_id = resolve_model_id(alias_or_id)
    print(f"  Pulling {model_id} ...")
    for progress in ollama.pull(model_id):
        status = getattr(progress, "status", None) or progress.get("status") if isinstance(progress, dict) else None
        if status:
            print(f"  {status}")


def load_ollama_gemma4(model: str = DEFAULT_MODEL, **kwargs):
    wrapper = OllamaGemma4(model=model, **kwargs)
    return wrapper.generate_fn()


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class OllamaGemma4:
    """
    Ollama-based mirror of the Gemma4 wrapper.

    Notes:
    - Thinking mode is a prompt-level toggle, not a native effort slider.
    - Images are supported through Ollama's `images` field.
    - Audio is not wired here yet; calls with `audio=` raise clearly.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system: str = "You are a helpful assistant.",
        thinking: bool = False,
        use_memory: bool = True,
        temperature: float = 0.618,
        max_new_tokens: int = 4096,
        top_p: float = 0.95,
        top_k: int = 64,
        max_history: int = 40,
        keep_alive: str = "5m",
        auto_pull: bool = False,
    ):
        self.model_id = resolve_model_id(model)
        self.alias = next((k for k, v in MODELS.items() if v["id"] == self.model_id), model)
        self.system_message = system
        self.thinking = thinking
        self.use_memory = use_memory
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.max_history = max_history
        self.keep_alive = keep_alive
        self.history: list[dict] = []

        self.available_models = self._list_local_models()
        if auto_pull and self.model_id not in self.available_models:
            pull_model(self.model_id)
            self.available_models = self._list_local_models()

        print(f"\n  Model : {self.alias} ({self.model_id})")
        print(f"  Backend: ollama  |  thinking: {thinking}  |  memory: {use_memory}")
        if self.available_models:
            print(f"  Local models detected: {len(self.available_models)}")
        print("  Ready.\n")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def generate(
        self,
        system: str,
        user: str,
        images: Optional[list] = None,
        audio: Optional[str] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": self._thinking_system(system, thinking)},
            self._build_user_message(user, images=images, audio=audio),
        ]
        return self._chat(messages)

    def generate_fn(self):
        def fn(system: str, user: str) -> str:
            return self.generate(system, user)
        fn.model_id = self.model_id
        fn.alias = self.alias
        fn.owner = self
        return fn

    def chat(
        self,
        message: str,
        images: Optional[list] = None,
        audio: Optional[str] = None,
        system: Optional[str] = None,
        thinking: Optional[bool] = None,
    ) -> str:
        messages = [{"role": "system", "content": self._thinking_system(system or self.system_message, thinking)}]
        messages.extend(self.history)
        messages.append(self._build_user_message(message, images=images, audio=audio))

        response = self._chat(messages)

        if self.use_memory:
            # Keep stored history text-only and compact.
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]

        return response

    def structured(
        self,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[dict] = None,
        images: Optional[list] = None,
    ) -> dict:
        """
        Structured JSON output using Ollama's `format` support.
        """
        system_prompt = (system or self.system_message) + (
            "\n\nRespond only with valid JSON. No markdown. No explanation."
        )
        if schema:
            system_prompt += f"\n\nExpected shape:\n{json.dumps(schema, indent=2)}"

        message = self._build_user_message(prompt, images=images)
        messages = [
            {"role": "system", "content": system_prompt},
            message,
        ]

        response = ollama.chat(
            model=self.model_id,
            messages=messages,
            stream=False,
            format="json",
            options=self._options(temperature=0.2),
            keep_alive=self.keep_alive,
        )
        content = response["message"]["content"].strip()
        return _extract_json(content) or {}

    def clear(self):
        self.history.clear()

    def get_history(self) -> list[dict]:
        return list(self.history)

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _list_local_models(self) -> list[str]:
        try:
            response = ollama.list()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            names = []
            for model in models or []:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model")
                else:
                    name = getattr(model, "name", None) or getattr(model, "model", None)
                if name:
                    names.append(str(name).strip())
            return sorted(dict.fromkeys(names))
        except Exception:
            return []

    def _thinking_system(self, system: str, thinking: Optional[bool]) -> str:
        enabled = self.thinking if thinking is None else thinking
        if not enabled:
            return system
        return (
            system
            + "\n\nThink step by step inside <think></think> tags."
            + "\nKeep the final answer outside those tags."
            + "\nBe concise in both thinking and final answer."
        )

    def _options(self, temperature: Optional[float] = None) -> dict:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _chat(self, messages: list[dict]) -> str:
        response = ollama.chat(
            model=self.model_id,
            messages=messages,
            stream=False,
            options=self._options(),
            keep_alive=self.keep_alive,
        )
        content = response["message"]["content"].strip()
        return _strip_think_tags(content)

    def _build_user_message(self, text: str, images: Optional[list] = None, audio: Optional[str] = None) -> dict:
        if audio:
            raise NotImplementedError("Audio is not wired in the Ollama wrapper yet.")
        if not images:
            return {"role": "user", "content": text}

        encoded_images = []
        for img in images:
            encoded_images.append(_image_to_base64(img))
        return {"role": "user", "content": text, "images": encoded_images}

    def __repr__(self):
        return (
            f"OllamaGemma4(alias={self.alias}, thinking={self.thinking}, "
            f"use_memory={self.use_memory}, history={len(self.history)} msgs)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode("utf-8")

    with open(path_or_url, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _strip_think_tags(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text


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
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ---------------------------------------------------------------------------
# TUI
# ---------------------------------------------------------------------------

CYAN = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
GREY = "\033[90m"; BOLD = "\033[1m"; RESET = "\033[0m"


def _tui():
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"  {BOLD}Ollama Gemma 4 Chat{RESET}")
    print(f"  Local Ollama backend with memory, thinking, and images")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    aliases = list(MODELS.keys())
    print(f"\n{YELLOW}Model:{RESET}")
    for i, alias in enumerate(aliases, 1):
        info = MODELS[alias]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        print(f"  [{i}] {alias:<5} {info['id']:<14}  {info['description']}{default}")
    print("  [c] custom local tag")
    print("  [p] pull model")
    raw = input("\n  Choice [1]: ").strip().lower() or "1"

    if raw == "p":
        model_name = input("  Model/tag to pull [gemma4:e2b]: ").strip() or "gemma4:e2b"
        pull_model(model_name)
        return

    if raw == "c":
        model_name = input("  Custom local Ollama tag: ").strip()
    else:
        model_name = aliases[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(aliases) else aliases[0]

    print(f"\n{YELLOW}Thinking mode? [y/N]{RESET}")
    thinking = input("  > ").strip().lower() in ("y", "yes")

    print(f"\n{YELLOW}Memory? [Y/n]{RESET}")
    use_memory = input("  > ").strip().lower() not in ("n", "no")

    g = OllamaGemma4(model_name, thinking=thinking, use_memory=use_memory)
    print(f"{GREEN}Ready.{RESET}")
    print(f"{GREY}Commands:{RESET}")
    print(f"{GREY}  /think                  toggle thinking{RESET}")
    print(f"{GREY}  /clear                  clear memory{RESET}")
    print(f"{GREY}  /history                show memory{RESET}")
    print(f"{GREY}  /system                 change system prompt{RESET}")
    print(f"{GREY}  /image <url/path> <q>   send image + question{RESET}")
    print(f"{GREY}  /json <prompt>          structured JSON{RESET}")
    print(f"{GREY}  q                       quit{RESET}")

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
            print(f"  Thinking: {status}")
            continue

        if user_input == "/clear":
            g.clear()
            print(f"  {GREY}History cleared.{RESET}")
            continue

        if user_input == "/history":
            if not g.history:
                print(f"  {GREY}(empty){RESET}")
            else:
                for msg in g.history:
                    col = CYAN if msg["role"] == "user" else GREEN
                    print(f"  {col}[{msg['role']}]{RESET} {str(msg['content'])[:120]}")
            continue

        if user_input == "/system":
            new_sys = input("  New system message: ").strip()
            if new_sys:
                g.system_message = new_sys
                print(f"  {GREEN}Updated.{RESET}")
            continue

        if user_input.startswith("/image "):
            rest = user_input[7:].strip()
            parts = rest.split(" ", 1)
            img_src = parts[0]
            prompt = parts[1] if len(parts) > 1 else "Describe this image."
            response = g.chat(prompt, images=[img_src])
            print(f"\n  {CYAN}[{g.alias}]{RESET} {response}")
            print(f"  {GREY}({len(g.history)//2} turns in memory){RESET}")
            continue

        if user_input.startswith("/json "):
            prompt = user_input[6:].strip()
            result = g.structured(prompt)
            print(f"\n  {CYAN}{json.dumps(result, indent=2)}{RESET}")
            continue

        response = g.chat(user_input)
        thinking_tag = f" {GREY}[thinking]{RESET}" if g.thinking else ""
        print(f"\n  {CYAN}[{g.alias}]{thinking_tag}{RESET} {response}")
        print(f"  {GREY}({len(g.history)//2} turns in memory){RESET}")

    print(f"\n  {GREY}Bye.{RESET}")


if __name__ == "__main__":
    _tui()
