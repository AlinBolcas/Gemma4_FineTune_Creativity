"""
openai_integration.py - OpenAI Responses API mirror of the local generators.

Same shape as src/IV_inference/ollama_integration.py, but routed through
OpenAI for faster remote data generation.

Class:
    OpenAIGenerator
        .chat(msg)
        .generate(system, user)
        .structured(prompt)
        .generate_fn()
        .clear()
        .get_history()

Run:
    python src/IV_inference/openai_integration.py
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

# Up-to-date model capabilities and IDs are published here:
# https://developers.openai.com/api/docs/models/all
# We do not hardcode a fixed catalog: we list models from the API for this key
# and filter to text-generation-appropriate names.

# ---------------------------------------------------------------------------
# OpenAI model registry (API-backed)
# ---------------------------------------------------------------------------

MODELS_DOCS_URL = "https://developers.openai.com/api/docs/models/all"
PRICING_DOCS_URL = "https://developers.openai.com/api/docs/pricing#text-tokens"

# Legacy aliases (older code / scripts). Prefer real model ids from the live list.
LEGACY_ALIASES: dict[str, str] = {
    "fast": "gpt-4.1-mini",
    "strong": "gpt-4.1",
    "cheap": "gpt-4.1-nano",
    "omni": "gpt-4o-mini",
    "gpt5mini": "gpt-5-mini",
    "gpt5": "gpt-5",
}

# If list_models() fails, still allow a run with common ids (may 404 for some keys).
FALLBACK_MODEL_ROWS: list[dict[str, str]] = [
    {
        "id": "gpt-4.1-mini",
        "description": "Fallback: API list failed. Install/refresh openai, check OPENAI_API_KEY, then retry.",
    },
    {"id": "gpt-4.1", "description": "Fallback row."},
    {"id": "gpt-5.4", "description": "Fallback row (if your key has 5.4 access)."},
    {"id": "gpt-5.5", "description": "Fallback row (if your key has 5.5 access)."},
]

_MODEL_LIST_CACHE: tuple[float, list[dict[str, str]]] | None = None
_CACHE_TTL_SEC = 300.0
# How many models to show in TUIs; full set still comes from the API for this key.
MODEL_PICKER_DISPLAY_CAP = 50

# Prices are USD per 1M tokens from the OpenAI pricing docs above.
OPENAI_TEXT_PRICING_PER_1M: dict[str, dict[str, dict[str, float | None]]] = {
    "standard": {
        "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00, "long_input": 10.00, "long_cached_input": 1.00, "long_output": 45.00},
        "gpt-5.5-pro": {"input": 30.00, "cached_input": None, "output": 180.00, "long_input": 60.00, "long_cached_input": None, "long_output": 270.00},
        "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00, "long_input": 5.00, "long_cached_input": 0.50, "long_output": 22.50},
        "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-pro": {"input": 30.00, "cached_input": None, "output": 180.00, "long_input": 60.00, "long_cached_input": None, "long_output": 270.00},
    },
    "batch": {
        "gpt-5.5": {"input": 2.50, "cached_input": 0.25, "output": 15.00, "long_input": 5.00, "long_cached_input": 0.50, "long_output": 22.50},
        "gpt-5.5-pro": {"input": 15.00, "cached_input": None, "output": 90.00, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4": {"input": 1.25, "cached_input": 0.13, "output": 7.50, "long_input": 2.50, "long_cached_input": 0.25, "long_output": 11.25},
        "gpt-5.4-mini": {"input": 0.375, "cached_input": 0.0375, "output": 2.25, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.625, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-pro": {"input": 15.00, "cached_input": None, "output": 90.00, "long_input": 30.00, "long_cached_input": None, "long_output": 135.00},
    },
    "flex": {
        "gpt-5.5": {"input": 2.50, "cached_input": 0.25, "output": 15.00, "long_input": 5.00, "long_cached_input": 0.50, "long_output": 22.50},
        "gpt-5.5-pro": {"input": 15.00, "cached_input": None, "output": 90.00, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4": {"input": 1.25, "cached_input": 0.13, "output": 7.50, "long_input": 2.50, "long_cached_input": 0.25, "long_output": 11.25},
        "gpt-5.4-mini": {"input": 0.375, "cached_input": 0.0375, "output": 2.25, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-nano": {"input": 0.10, "cached_input": 0.01, "output": 0.625, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-pro": {"input": 15.00, "cached_input": None, "output": 90.00, "long_input": 30.00, "long_cached_input": None, "long_output": 135.00},
    },
    "priority": {
        "gpt-5.5": {"input": 12.50, "cached_input": 1.25, "output": 75.00, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4": {"input": 5.00, "cached_input": 0.50, "output": 30.00, "long_input": None, "long_cached_input": None, "long_output": None},
        "gpt-5.4-mini": {"input": 1.50, "cached_input": 0.15, "output": 9.00, "long_input": None, "long_cached_input": None, "long_output": None},
    },
}

PRICED_FLAGSHIP_MODELS = tuple(OPENAI_TEXT_PRICING_PER_1M["standard"].keys())


def _get_api_key() -> str | None:
    return (os.getenv("OPENAI_API_KEY") or "").strip() or None


def normalize_priced_model_id(model_id: str) -> str | None:
    """Map dated model ids like gpt-5.5-2026-04-23 to their priced family."""
    m = (model_id or "").strip().lower()
    for base in sorted(PRICED_FLAGSHIP_MODELS, key=len, reverse=True):
        if m == base or m.startswith(f"{base}-"):
            return base
    return None


def get_model_pricing(
    model_id: str,
    *,
    service_tier: str = "standard",
    long_context: bool = False,
) -> dict[str, float | None] | None:
    family = normalize_priced_model_id(model_id)
    if not family:
        return None
    row = OPENAI_TEXT_PRICING_PER_1M.get(service_tier, {}).get(family)
    if not row:
        return None
    if not long_context:
        return row
    if row.get("long_input") is None or row.get("long_output") is None:
        return None
    return {
        "input": row.get("long_input"),
        "cached_input": row.get("long_cached_input"),
        "output": row.get("long_output"),
    }


def estimate_cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    *,
    service_tier: str = "standard",
    long_context: bool = False,
) -> float | None:
    rates = get_model_pricing(model_id, service_tier=service_tier, long_context=long_context)
    if not rates:
        return None
    cached = max(0, min(int(cached_input_tokens or 0), int(input_tokens or 0)))
    uncached = max(0, int(input_tokens or 0) - cached)
    cached_rate = rates.get("cached_input")
    input_rate = rates.get("input")
    output_rate = rates.get("output")
    if input_rate is None or output_rate is None:
        return None
    cached_cost = cached * (cached_rate if cached_rate is not None else input_rate)
    uncached_cost = uncached * input_rate
    output_cost = max(0, int(output_tokens or 0)) * output_rate
    return (cached_cost + uncached_cost + output_cost) / 1_000_000


def format_usd(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.2f}"


def price_brief(model_id: str) -> str:
    rates = get_model_pricing(model_id)
    if not rates:
        return "price unknown"
    return f"${rates['input']}/M in, ${rates['output']}/M out"


def _usage_value(obj: object, *names: str) -> int:
    for name in names:
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def _cached_input_tokens(usage: object) -> int:
    details = None
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details")
    else:
        details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
    return _usage_value(details or {}, "cached_tokens", "cached_input_tokens")


def _is_text_suitable_model_id(model_id: str) -> bool:
    """Heuristic: include chat/completion models, drop embeddings, audio, image gen, etc."""
    m = (model_id or "").strip().lower()
    if not m:
        return False
    # Hard excludes (id substring)
    for bad in (
        "embed",
        "embedding",
        "whisper",
        "moderation",
        "tts-",
        "tts-1",
        "dall",
        "sora-",
        "transcrib",
        "diarize",
        "text-embedding",
        "babbage",
        "davinci",
        "omni-moderation",
        "search-preview",
        "computer-use",
        "deep-research",
        "instruct-",
        "chat-latest",
    ):
        if bad in m:
            return False
    if m.startswith("chatgpt"):
        return False
    if m.startswith(("gpt-image", "dall", "sora", "text-embed", "omni-moderation")):
        return False
    if m.startswith("gpt-audio") or m.startswith("gpt-realtime") or "realtime" in m:
        return False
    # Includes (per current catalog families on developers.openai.com)
    for ok in (
        "gpt-5",
        "gpt-4",
        "gpt-3.5",
        "gpt-3",
        "gpt-oss-",
        "o1",
        "o3",
        "o4",
    ):
        if m.startswith(ok) or m == ok.rstrip("-"):
            return True
    return False


def _model_sort_key(model_id: str) -> tuple[int, int, str]:
    """Frontier-ish models first, then 4.1, 4o, 3.5, then the rest A-Z."""
    m = (model_id or "").lower()
    if m.startswith("gpt-5.5"):
        g = 0
    elif m.startswith("gpt-5.4"):
        g = 1
    elif m.startswith("gpt-5.3"):
        g = 2
    elif m.startswith("gpt-5.2"):
        g = 3
    elif m.startswith("gpt-5.1"):
        g = 4
    elif m.startswith("gpt-5") and not m.startswith("gpt-5."):
        g = 5
    elif m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        g = 6
    elif m.startswith("gpt-4.1"):
        g = 10
    elif m.startswith("gpt-4o"):
        g = 11
    elif m.startswith("gpt-4"):
        g = 12
    elif m.startswith("gpt-3.5"):
        g = 20
    elif m.startswith("gpt-oss"):
        g = 30
    else:
        g = 40
    return (g, -len(m), m)


def _description_for_model_id(model_id: str) -> str:
    m = (model_id or "").lower()
    price = price_brief(model_id) if normalize_priced_model_id(model_id) else ""
    if "mini" in m and "o1" not in m and "o3" not in m and "o4" not in m:
        return f"Smaller, lower latency / cost; {price}" if price else "Smaller, lower latency / cost"
    if "nano" in m:
        return f"Fastest, highest volume, lowest cost in family; {price}" if price else "Fastest, highest volume, lowest cost in family"
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "Reasoning-style model; sampling may be constrained"
    if m.startswith("gpt-5.5") or m.startswith("gpt-5.4") or m.startswith("gpt-5.3"):
        return f"Recent GPT-5 class; {price}" if price else "Recent GPT-5 class (see model docs for limits)"
    if m.startswith("gpt-5"):
        return "GPT-5 class (see model docs for capabilities)"
    if m.startswith("gpt-4.1"):
        return "GPT-4.1 (non-reasoning)"
    if m.startswith("gpt-4o"):
        return "4o class multimodal text (vision-capable; we use text here)"
    if m.startswith("gpt-3.5"):
        return "Legacy 3.5 (cheap baseline)"
    if m.startswith("gpt-oss-"):
        return "Open-weight OSS model (if your key can call it)"
    return "Text model (from live API list)"


def _row_from_api_obj(obj: object) -> dict[str, str] | None:
    mid = getattr(obj, "id", None) if not isinstance(obj, dict) else obj.get("id")
    if not mid or not isinstance(mid, str):
        return None
    if not _is_text_suitable_model_id(mid):
        return None
    owned = getattr(obj, "owned_by", None) if not isinstance(obj, dict) else obj.get("owned_by")
    desc = _description_for_model_id(mid)
    if owned and str(owned).strip():
        desc = f"{desc}  owner={owned}"
    return {"id": mid, "description": desc[:200]}


def fetch_text_generation_models(*, use_cache: bool = True) -> list[dict[str, str]]:
    """
    Return rows {id, description} for models the API exposes for this key, filtered
    for this repo's text pipeline. Uses a short in-memory cache.
    """
    global _MODEL_LIST_CACHE

    if use_cache and _MODEL_LIST_CACHE is not None:
        ts, rows = _MODEL_LIST_CACHE
        if time.time() - ts < _CACHE_TTL_SEC and rows:
            return rows

    if not _get_api_key():
        return list(FALLBACK_MODEL_ROWS)

    openai_module = importlib.import_module("openai")
    try:
        client = openai_module.OpenAI()
        listed = client.models.list()
    except Exception:
        return list(FALLBACK_MODEL_ROWS)

    raw = getattr(listed, "data", None) or []
    rows: list[dict[str, str]] = []
    for obj in raw:
        row = _row_from_api_obj(obj)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: _model_sort_key(r["id"]))

    if not rows:
        return list(FALLBACK_MODEL_ROWS)

    _MODEL_LIST_CACHE = (time.time(), rows)
    return rows


def clear_model_list_cache() -> None:
    """For tests: force a fresh models.list on next fetch."""
    global _MODEL_LIST_CACHE
    _MODEL_LIST_CACHE = None


def get_default_model_id() -> str:
    rows = fetch_text_generation_models()
    return rows[0]["id"] if rows else "gpt-4.1-mini"


def resolve_model_id(alias_or_id: str) -> str:
    raw = str(alias_or_id or "").strip()
    if not raw:
        return get_default_model_id()
    key = raw.lower()
    if key in LEGACY_ALIASES:
        return LEGACY_ALIASES[key]
    return raw


def list_models() -> str:
    lines = [f"{'#':<4}  {'Model ID':<32}  Description", "-" * 96, f"Catalog: {MODELS_DOCS_URL}"]
    rows = fetch_text_generation_models()
    for i, r in enumerate(rows[:MODEL_PICKER_DISPLAY_CAP], 1):
        mark = " <-- default" if i == 1 else ""
        lines.append(
            f"{i:<4}  {r['id']:<32}  {r.get('description', '')[:50]}{mark}"
        )
    if len(rows) > MODEL_PICKER_DISPLAY_CAP:
        lines.append(
            f"... +{len(rows) - MODEL_PICKER_DISPLAY_CAP} more (type [c] in TUI, or set cap MODEL_PICKER_DISPLAY_CAP)"
        )
    return "\n".join(lines)


def load_openai_generator(model: str | None = None, **kwargs):
    wrapper = OpenAIGenerator(model=model, **kwargs)
    return wrapper.generate_fn()


# Backwards-friendly alias for callers that expect the provider-specific name.
load_openai_gemma4 = load_openai_generator


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class OpenAIGenerator:
    """
    OpenAI-based mirror of the Ollama wrapper.

    Notes:
    - Uses the Responses API.
    - The pipeline calls are independent, so generation should normally use
      use_memory=False.
    - Reasoning models may ignore sampling controls; unsupported params are
      stripped automatically.
    """

    def __init__(
        self,
        model: str | None = None,
        system: str = "You are a helpful assistant.",
        thinking: bool = False,
        use_memory: bool = True,
        temperature: float = 0.618,
        max_new_tokens: int = 4096,
        top_p: float = 0.95,
        max_history: int = 40,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        store: bool = False,
    ):
        self.model_id = resolve_model_id(model if model is not None else "")
        self.alias = self.model_id
        self.system_message = system
        self.thinking = thinking
        self.use_memory = use_memory
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.max_history = max_history
        self.timeout = timeout
        self.store = store
        self.history: list[dict] = []
        self._usage_lock = threading.Lock()
        self._usage_totals = {
            "requests": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY was not found. Add it to .env or your shell environment.")
        openai_module = importlib.import_module("openai")
        self.client = openai_module.OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)

        print(f"\n  Model : {self.model_id}")
        print(f"  Backend: openai  |  thinking: {thinking}  |  memory: {use_memory}")
        print(f"  Tokens: max_output={max_new_tokens}  |  store={store}")
        print(f"  Price : {price_brief(self.model_id)}  |  docs: {PRICING_DOCS_URL}")
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
        if audio:
            raise NotImplementedError("Audio is not wired in the OpenAI generator yet.")
        messages = [
            {"role": "system", "content": self._thinking_system(system, thinking)},
            self._build_user_message(user, images=images),
        ]
        return self._response(messages)

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
        if audio:
            raise NotImplementedError("Audio is not wired in the OpenAI generator yet.")
        messages = [{"role": "system", "content": self._thinking_system(system or self.system_message, thinking)}]
        messages.extend(self.history)
        messages.append(self._build_user_message(message, images=images))

        response = self._response(messages)

        if self.use_memory:
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
        system_prompt = (system or self.system_message) + (
            "\n\nRespond only with valid JSON. No markdown. No explanation."
        )
        if schema:
            system_prompt += f"\n\nExpected shape:\n{json.dumps(schema, indent=2, ensure_ascii=False)}"

        messages = [
            {"role": "system", "content": system_prompt},
            self._build_user_message(prompt, images=images),
        ]

        params = self._params(messages, temperature=0.2)
        params["text"] = {"format": {"type": "json_object"}}

        response = self.client.responses.create(**self._sanitize_params(params))
        self._record_response_usage(response)
        return _extract_json(getattr(response, "output_text", "") or "") or {}

    def clear(self):
        self.history.clear()

    def get_history(self) -> list[dict]:
        return list(self.history)

    def get_usage_snapshot(self) -> dict:
        with self._usage_lock:
            return dict(self._usage_totals)

    def reset_usage(self) -> None:
        with self._usage_lock:
            for key in self._usage_totals:
                self._usage_totals[key] = 0.0 if key == "cost_usd" else 0

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _thinking_system(self, system: str, thinking: Optional[bool]) -> str:
        enabled = self.thinking if thinking is None else thinking
        if not enabled:
            return system
        return (
            system
            + "\n\nThink step by step privately before answering."
            + "\nReturn only the final useful answer."
        )

    def _params(self, messages: list[dict], temperature: Optional[float] = None) -> dict:
        return {
            "model": self.model_id,
            "input": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "store": self.store,
        }

    def _response(self, messages: list[dict]) -> str:
        response = self.client.responses.create(**self._sanitize_params(self._params(messages)))
        self._record_response_usage(response)
        return (getattr(response, "output_text", "") or "").strip()

    def _record_response_usage(self, response: object) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or (input_tokens + output_tokens)
        cached_tokens = _cached_input_tokens(usage)
        cost = estimate_cost_usd(
            self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_tokens,
        )
        with self._usage_lock:
            self._usage_totals["requests"] += 1
            self._usage_totals["input_tokens"] += input_tokens
            self._usage_totals["cached_input_tokens"] += cached_tokens
            self._usage_totals["output_tokens"] += output_tokens
            self._usage_totals["total_tokens"] += total_tokens
            if cost is not None:
                self._usage_totals["cost_usd"] += cost

    def _sanitize_params(self, params: dict) -> dict:
        model = str(params.get("model") or "").lower()
        if model.startswith(("o1", "o3", "o4", "gpt-5")):
            # Reasoning-family models can reject classic sampling knobs.
            params.pop("temperature", None)
            params.pop("top_p", None)
        return params

    def _build_user_message(self, text: str, images: Optional[list] = None) -> dict:
        if not images:
            return {"role": "user", "content": text}

        content = [{"type": "input_text", "text": text}]
        for img in images:
            content.append({"type": "input_image", "image_url": _image_to_data_url(img)})
        return {"role": "user", "content": content}

    def __repr__(self):
        return (
            f"OpenAIGenerator(alias={self.alias}, thinking={self.thinking}, "
            f"use_memory={self.use_memory}, history={len(self.history)} msgs)"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_url(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url

    suffix = Path(path_or_url).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix == "jpg" else suffix
    with open(path_or_url, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
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
        in_string = False
        escaped = False
        for i in range(brace, len(text)):
            char = text[i]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
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
    print(f"  {BOLD}OpenAI Generator Chat{RESET}")
    print("  Responses API backend with memory and structured JSON")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

    rows = fetch_text_generation_models()
    priced_rows = [row for row in rows if normalize_priced_model_id(row["id"])]
    shown = (priced_rows or rows)[:MODEL_PICKER_DISPLAY_CAP]
    default_id = shown[0]["id"] if shown else get_default_model_id()

    print(f"\n{YELLOW}Model:{RESET} {GREY}(5.5/5.4 priced flagship models for your API key){RESET}")
    print(f"  {GREY}Catalog reference: {MODELS_DOCS_URL}{RESET}")
    print(f"  {GREY}Pricing reference: {PRICING_DOCS_URL}{RESET}")
    for i, row in enumerate(shown, 1):
        desc = price_brief(row["id"]) if normalize_priced_model_id(row["id"]) else (row.get("description") or "")[:52]
        default = f" {GREY}<- default{RESET}" if i == 1 else ""
        print(f"  [{i}] {row['id']:<26}  {desc}{default}")
    if priced_rows and len(rows) > len(priced_rows):
        print(f"  {GREY}Showing only priced 5.5/5.4 choices. Use [c] to type another live id.{RESET}")
    print("  [c] custom model id")
    raw = input("\n  Choice [1]: ").strip().lower() or "1"

    if raw == "c":
        model_name = input("  Custom OpenAI model id: ").strip() or default_id
    else:
        idx = int(raw) - 1 if raw.isdigit() and 1 <= int(raw) <= len(shown) else 0
        model_name = shown[idx]["id"] if shown else default_id

    print(f"\n{YELLOW}Thinking mode? [y/N]{RESET}")
    thinking = input("  > ").strip().lower() in ("y", "yes")

    print(f"\n{YELLOW}Memory? [Y/n]{RESET}")
    use_memory = input("  > ").strip().lower() not in ("n", "no")

    g = OpenAIGenerator(model_name, thinking=thinking, use_memory=use_memory)
    print(f"{GREEN}Ready.{RESET}")
    print(f"{GREY}Commands: /think, /clear, /history, /system, /json <prompt>, q{RESET}")

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

        if user_input.startswith("/json "):
            result = g.structured(user_input[6:].strip())
            print(f"\n  {CYAN}{json.dumps(result, indent=2)}{RESET}")
            continue

        response = g.chat(user_input)
        thinking_tag = f" {GREY}[thinking]{RESET}" if g.thinking else ""
        print(f"\n  {CYAN}[{g.alias}]{thinking_tag}{RESET} {response}")
        print(f"  {GREY}({len(g.history)//2} turns in memory){RESET}")

    print(f"\n  {GREY}Bye.{RESET}")


if __name__ == "__main__":
    _tui()
