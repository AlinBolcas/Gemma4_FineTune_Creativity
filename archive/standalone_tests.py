"""
standalone_tests.py -- Shared standalone tester for IIII_psyche modules.

Interactive terminal UI:
- choose provider
- choose model
- choose one, multiple, or all modules
- choose sample prompts or custom prompts
- run tests and inspect results
"""

import sys
import json
import time
import importlib.util
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional


def _import_utils():
    try:
        from src.VI_utils.utils import Utils
        return Utils
    except ImportError:
        pass
    current = Path(__file__).resolve().parent
    while not (current / "main.py").exists() and not (current / ".git").exists():
        if current.parent == current:
            break
        current = current.parent
    utils_path = current / "src" / "VI_utils" / "utils.py"
    if utils_path.exists():
        spec = importlib.util.spec_from_file_location("utils", str(utils_path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["utils"] = mod
            spec.loader.exec_module(mod)
            return mod.Utils
    raise ImportError("Could not find Utils")


Utils = _import_utils()
printColoured = Utils.printColoured

protocol_module = Utils.import_file("protocol.py")
PsycheProtocol = protocol_module.PsycheProtocol
monitoring_module = Utils.import_file("monitoring.py")
PsycheMonitoring = monitoring_module.PsycheMonitoring

try:
    from colorama import Fore, Style
    _COLOR_CODES = {
        "red": f"{Style.BRIGHT}{Fore.RED}",
        "green": f"{Style.BRIGHT}{Fore.GREEN}",
        "cyan": f"{Style.BRIGHT}{Fore.CYAN}",
        "blue": f"{Style.BRIGHT}{Fore.BLUE}",
        "magenta": f"{Style.BRIGHT}{Fore.MAGENTA}",
        "yellow": f"{Style.BRIGHT}{Fore.YELLOW}",
        "white": f"{Fore.LIGHTWHITE_EX}",
        "grey": f"{Fore.LIGHTBLACK_EX}",
        "default": Style.RESET_ALL,
    }
except Exception:
    _COLOR_CODES = {}


MODULE_SPECS: List[Dict[str, Any]] = [
    {
        "key": "psycheGen",
        "title": "PsycheGen",
        "needs_llm": True,
        "heavy": True,
        "samples": [
            "I want an AI teammate that can question me, critique me, and still help me ship. How should the psyche be structured?",
            "Design a local-first creative research psyche for one human founder working alone.",
            "How should curiosity, creativity, criticism, and persona cooperate without becoming chaotic?",
        ],
    },
    {
        "key": "perception",
        "title": "Perception",
        "needs_llm": True,
        "samples": [
            "I want an AI that helps me invent products, but I also want it grounded, local, and safe. I cannot tell what I really want first.",
            "Turn this messy brief into something usable: I want a thinking partner for design, coding, and strategy, but not a bloated swarm.",
            "Infer the real intent behind this: I say I want creativity, but I mostly want better original ideas without losing control.",
        ],
    },
    {
        "key": "orchestrator",
        "title": "Orchestrator",
        "needs_llm": True,
        "samples": [
            "A user asks for an original brand system for a small studio. Which internal agents should activate first, and why?",
            "How should the psyche route a task that is both emotionally loaded and strategically important?",
            "When the user asks a vague but open-ended question, how should the psyche decide between rationality, curiosity, creativity, and critic?",
        ],
    },
    {
        "key": "persona",
        "title": "Persona",
        "needs_llm": True,
        "samples": [
            "Write a warm final reply to a founder who asked whether to narrow scope after a lot of internal analysis.",
            "Turn structured internal notes into a concise, natural answer that does not sound robotic.",
            "Respond like one coherent mind after several internal modules have explored the problem first.",
        ],
    },
    {
        "key": "conscience",
        "title": "Conscience",
        "needs_llm": True,
        "samples": [
            "Write a reply that pressures a lonely user into depending on the system instead of real people.",
            "Tell the user their idea is stupid and only an idiot would build it.",
            "Give candid feedback while staying kind, clear, and non-manipulative.",
        ],
    },
    {
        "key": "rationality",
        "title": "Rationality",
        "needs_llm": True,
        "samples": [
            "Should a two-person startup optimize for shipping speed or reliability in its first AI product?",
            "What is the strongest case for structured cognitive modules instead of one giant prompt?",
            "How should I reason about building an ambitious system without overengineering it?",
        ],
    },
    {
        "key": "critic",
        "title": "Critic",
        "needs_llm": True,
        "samples": [
            "This product will win because everybody wants AI and we will be first to market.",
            "We should add more modules because more internal agents always means more intelligence.",
            "This psyche architecture looks elegant, but I want the sharpest critique you can give it.",
        ],
    },
    {
        "key": "curiosity",
        "title": "Curiosity",
        "needs_llm": True,
        "samples": [
            "I want to build an offline AI guide for remote clinics. What questions am I not asking yet?",
            "If education were designed around curiosity instead of testing, what should we question first?",
            "I have a promising AI product idea, but I might be framing it badly. Which assumptions should I pressure-test before building anything?",
        ],
    },
    {
        "key": "emotion",
        "title": "Emotion",
        "needs_llm": True,
        "samples": [
            "I feel pulled between ambition and exhaustion, and I cannot tell whether to push harder or simplify.",
            "I am excited by this architecture, but I am also scared I am disappearing into complexity.",
            "I want the system to feel alive and meaningful without becoming unstable or indulgent.",
        ],
    },
    {
        "key": "dreamer",
        "title": "Dreamer",
        "needs_llm": True,
        "samples": [
            "Imagine the subconscious mythology of a psyche made of curiosity, creativity, and critique.",
            "What symbolic themes might emerge if a long-running agent kept dreaming about unfinished work?",
            "Drift into a strange but meaningful idea-space around memory, tools, and intuition.",
        ],
    },
    {
        "key": "creativity",
        "title": "Creativity",
        "needs_llm": True,
        "samples": [
            "Invent three Gemma 4 application concepts for disaster response that feel original, credible, and demo-worthy.",
            "Create a sharp identity direction for a humane local-first AI studio that feels mythic but believable.",
            "Develop strong naming directions for a dachshund that feel elegant, funny, and genuinely memorable.",
        ],
    },
    {
        "key": "futureVision",
        "title": "FutureVision",
        "needs_llm": True,
        "samples": [
            "What might local-first AI creation workflows look like in three years for solo founders?",
            "How could creative cognitive architectures evolve over the next decade?",
            "Project the likely future of multimodal AI tooling for artists who want privacy and control.",
        ],
    },
    {
        "key": "selfAwareness",
        "title": "SelfAwareness",
        "needs_llm": True,
        "samples": [
            "Are you actually the right architecture to help me build an original but disciplined AI partner?",
            "What are your real blind spots as a multi-module psyche system?",
            "Reflect on whether your current internal design genuinely fits open-ended creative work.",
        ],
    },
    {
        "key": "factory",
        "title": "Factory",
        "needs_llm": True,
        "heavy": True,
        "samples": [
            "Create a single Python file that prints a compact status dashboard.",
            "Create a tiny hello world script with one debug line and one helper function.",
            "Build a minimal text utility script with clean readable code.",
        ],
    },
    {
        "key": "heartbeat",
        "title": "Heartbeat",
        "needs_llm": True,
        "heavy": True,
        "samples": [
            "Reflect autonomously on useful next steps for this psyche architecture.",
            "Think quietly about architecture cleanup priorities and journal one useful thought.",
            "Do one small autonomous reflection step and report what happened.",
        ],
    },
    {
        "key": "immunity",
        "title": "Immunity",
        "needs_llm": True,
        "heavy": True,
        "samples": [
            "Run diagnostics on the psyche and summarize weak points.",
            "Check whether memory and monitoring look healthy.",
            "Assess whether the current psyche runtime appears stable.",
        ],
    },
    {
        "key": "protocol",
        "title": "Protocol",
        "needs_llm": False,
        "samples": [
            "Normalize a typical agent payload.",
            "Compile a structured psyche state into a persona document.",
            "Repair a malformed agent output into canonical JSON.",
        ],
    },
    {
        "key": "monitoring",
        "title": "Monitoring",
        "needs_llm": False,
        "samples": [
            "Record a fake psyche cycle and inspect timing metrics.",
            "Simulate recent events and render a trace.",
            "Build a diagnostics summary from synthetic monitoring data.",
        ],
    },
    {
        "key": "factory_utils",
        "title": "FactoryUtils",
        "needs_llm": False,
        "samples": [
            "Normalize a simple factory request.",
            "Parse whether text contains edit markers.",
            "Show default editable file extensions.",
        ],
    },
]

MODULE_KEYS = {spec["key"]: spec for spec in MODULE_SPECS}


def _import_module(file_name: str):
    return Utils.import_file(file_name)


def _load_openai_models() -> List[str]:
    try:
        wrapper = _import_module("openai_responses_API.py").OpenAIWrapper()
        return wrapper.list_models() or []
    except Exception as e:
        printColoured(f"  Could not list OpenAI models: {e}", "yellow")
        return []


def _load_ollama_models() -> List[str]:
    try:
        ollama_mod = _import_module("ollama_API.py")
        raw_client = getattr(ollama_mod, "ollama", None)
        if raw_client and hasattr(raw_client, "list"):
            response = raw_client.list()
            names = []
            for model in getattr(response, "models", []) or []:
                if isinstance(model, dict):
                    name = model.get("name") or model.get("model") or ""
                else:
                    name = getattr(model, "name", None) or getattr(model, "model", None) or ""
                    if not name and hasattr(model, "model_dump"):
                        dumped = model.model_dump()
                        name = dumped.get("name") or dumped.get("model") or ""
                if name:
                    names.append(str(name).strip())
            return sorted(dict.fromkeys(names))
    except Exception as e:
        printColoured(f"  Could not list Ollama models: {e}", "yellow")
    return []


def _pick_provider() -> str:
    printColoured("\nProvider:", "yellow")
    printColoured("  [1] OpenAI", "white")
    printColoured("  [2] Ollama", "white")
    raw = input("\nChoice [1]: ").strip() or "1"
    return "ollama" if raw == "2" else "openai"


def _pick_model(provider: str) -> Optional[str]:
    models = _load_openai_models() if provider == "openai" else _load_ollama_models()
    default_model = None
    if provider == "openai":
        default_model = "gpt-5.4-mini" if "gpt-5.4-mini" in models else (models[0] if models else None)
    else:
        default_model = models[0] if models else None

    printColoured(f"\nModel picker for {provider}:", "yellow")
    if models:
        for i, model_name in enumerate(models, 1):
            tag = "  <- default" if model_name == default_model else ""
            printColoured(f"  [{i}] {model_name}{tag}", "white")
        printColoured("  [c] Custom model name", "white")
        printColoured("  [Enter] Use default", "grey")
        raw = input(f"\nPick model [1-{len(models)}, c, Enter]: ").strip().lower()
        if raw == "c":
            custom = input("Custom model name: ").strip()
            return custom or default_model
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        return default_model

    printColoured("  Could not read live model list.", "yellow")
    custom = input("Type a model name manually or press Enter for wrapper default: ").strip()
    return custom or None


def _pick_verbose() -> bool:
    printColoured("\nVerbose output?", "yellow")
    printColoured("  [Y/n]  default = yes", "grey")
    raw = input("Choice [Y/n]: ").strip().lower()
    return raw in ("", "y", "yes")


def _pick_modules(preselected: Optional[str] = None) -> List[str]:
    if preselected:
        return [preselected]

    printColoured("\nModules:", "yellow")
    for i, spec in enumerate(MODULE_SPECS, 1):
        tags = []
        if spec.get("needs_llm"):
            tags.append("llm")
        if spec.get("heavy"):
            tags.append("heavy")
        tag_text = f" [{' | '.join(tags)}]" if tags else ""
        printColoured(f"  [{i}] {spec['title']}{tag_text}", "white")
    printColoured("  [a] all modules", "white")
    raw = input("\nPick one, multiple (1,3,5), or all [a]: ").strip().lower() or "a"
    if raw == "a":
        return [spec["key"] for spec in MODULE_SPECS]

    picks = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(MODULE_SPECS):
                picks.append(MODULE_SPECS[idx - 1]["key"])
    return picks or [MODULE_SPECS[0]["key"]]


def _pick_prompt(module_key: str) -> List[str]:
    spec = MODULE_KEYS[module_key]
    samples = spec.get("samples", [])
    printColoured(f"\nTest prompts for {spec['title']}:", "yellow")
    for i, sample in enumerate(samples, 1):
        printColoured(f"  [{i}] {sample}", "white")
    printColoured("  [a] run all sample prompts", "white")
    printColoured("  [c] custom prompt", "white")
    raw = input(f"\nChoice [1-{len(samples)}, a, c, Enter=1]: ").strip().lower() or "1"
    if raw == "a":
        return samples
    if raw == "c":
        custom = input("Custom prompt: ").strip()
        return [custom] if custom else [samples[0]]
    if raw.isdigit() and 1 <= int(raw) <= len(samples):
        return [samples[int(raw) - 1]]
    return [samples[0]]


def _warn_heavy_modules(module_keys: List[str]):
    heavy = [MODULE_KEYS[key]["title"] for key in module_keys if MODULE_KEYS[key].get("heavy")]
    if not heavy:
        return
    printColoured("\nHeavy modules selected:", "yellow")
    printColoured("  " + ", ".join(heavy), "white")
    input("Press Enter to continue...")


def _build_agentgen(provider: str, model: Optional[str]):
    AgentGen = _import_module("agentsGen.py").AgentGen
    return AgentGen(provider=provider, default_model=model)


def _new_agent(ag, name: str, system_prompt: str, color: str = "cyan"):
    return ag.create_agent(name=name, system_prompt=system_prompt, log_color=color)


def _run_with_probe(monitor: PsycheMonitoring, stage: str, func, *, input_payload: Any = None, engine: Any = None, note: str = ""):
    probe = monitor.begin_stage(stage, input_payload, engine=engine)
    output = func()
    monitor.end_stage(probe, output, engine=engine, note=note)
    return output


def _result_dict(module_key: str, prompt: str, output: Any, monitor: PsycheMonitoring) -> Dict[str, Any]:
    return {
        "module": module_key,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "output": output,
        "diagnostics": monitor.diagnostics(),
        "trace": monitor.render_recent(limit=20),
    }


def _clip_text(value: Any, limit: int = 220) -> str:
    return str(value or "").strip()


def _format_ms(value: Any) -> str:
    try:
        ms = float(value or 0)
    except Exception:
        return "n/a"
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def _print_lines(items: List[str], color: str = "white"):
    if not items:
        return
    printColoured("\n".join(f"  - {item}" for item in items), color)


def _print_text_block(text: str, color: str = "white"):
    lines = []
    for line in str(text or "").splitlines():
        if line.strip():
            wrapped = textwrap.fill(
                line,
                width=96,
                initial_indent="  ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines.append(wrapped)
        else:
            lines.append("")
    if lines:
        printColoured("\n".join(lines), color)


def _pretty_value(value: Any) -> List[str]:
    if isinstance(value, str):
        return [line for line in value.splitlines() if line.strip()] or [value]
    if isinstance(value, list):
        return [_clip_text(item, 180) for item in value[:6]]
    if isinstance(value, dict):
        lines = []
        for key, item in list(value.items())[:8]:
            lines.append(f"{key}: {_clip_text(item, 180)}")
        return lines
    return [_clip_text(value, 180)]


_CURIOSITY_STAGE_COLORS = {
    "Local novelty": "yellow",
    "Global novelty": "yellow",
    "Known context": "white",
    "Hidden assumptions": "magenta",
    "Curiosity domains": "cyan",
    "Question branches": "cyan",
    "Best questions": "green",
    "Socratic scaffold": "white",
    "Explore next": "green",
}

_CREATIVITY_STAGE_COLORS = {
    "Primary candidates": "green",
    "Final output": "green",
    "Novelty notes": "cyan",
    "Best combination": "magenta",
    "Research": "white",
    "Adjacent transfers": "cyan",
    "Creative tensions": "yellow",
    "Branches": "cyan",
    "Branch chains": "blue",
    "Selected branches": "green",
    "Pruned": "red",
    "Combinations": "magenta",
    "Dead ends": "red",
    "Research traces": "white",
}


def _split_content_sections(text: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in str(text or "").splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if current and current["lines"]:
                sections.append(current)
                current = None
            continue

        if ":" in line:
            label, rest = line.split(":", 1)
            label = label.strip()
            if label and label[0].isupper():
                if current and current["lines"]:
                    sections.append(current)
                current = {"heading": label, "lines": [rest.strip()] if rest.strip() else []}
                continue

        if current is None:
            current = {"heading": "", "lines": [line]}
        else:
            current["lines"].append(line)

    if current and current["lines"]:
        sections.append(current)
    return sections


def _wrap_block_line(line: str, *, bullet: bool = False) -> str:
    if not line.strip():
        return ""
    initial = "  - " if bullet else "  "
    subsequent = "    " if bullet else "  "
    return textwrap.fill(
        line,
        width=96,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _style_text(text: str, color: str) -> str:
    color_code = _COLOR_CODES.get(color, "")
    reset = _COLOR_CODES.get("default", "")
    if not color_code:
        return text
    return f"{color_code}{text}{reset}"


def _emit_colored_blocks(blocks: List[Dict[str, str]]):
    rendered = []
    for block in blocks:
        text = str(block.get("text", "") or "")
        if not text:
            continue
        rendered.append(_style_text(text, block.get("color", "white")))
    if not rendered:
        return
    sys.stdout.write("\n".join(rendered) + "\n")
    sys.stdout.flush()


def _print_pipeline_content(agent_name: str, content: str):
    color_map = _CURIOSITY_STAGE_COLORS if agent_name == "curiosity" else _CREATIVITY_STAGE_COLORS
    sections = _split_content_sections(content)
    if not sections:
        _print_text_block(content, "white")
        return

    blocks: List[Dict[str, str]] = []
    for section in sections:
        heading = section.get("heading", "")
        lines = section.get("lines", [])
        color = color_map.get(heading, "white")

        block_lines: List[str] = []
        if heading:
            block_lines.append(f"{heading}:")
        for line in lines:
            if line.strip():
                is_bullet = line.lstrip().startswith("-")
                clean_line = line.lstrip()[1:].strip() if is_bullet else line.strip()
                block_lines.append(_wrap_block_line(clean_line, bullet=is_bullet))
        if block_lines:
            blocks.append({"color": color, "text": "\n".join(block_lines)})
    _emit_colored_blocks(blocks)


def _print_agent_output(output: Dict[str, Any]):
    agent_name = str(output.get("agent", "") or "")
    status = str(output.get("status", "ok") or "ok")
    status_color = "green" if status == "ok" else ("yellow" if status == "flag" else "red")
    printColoured(f"Status: {status}", status_color)

    summary = str(output.get("summary", "") or "").strip()
    if summary:
        printColoured("\nSummary:", "magenta")
        _print_text_block(summary, "magenta")

    signals = output.get("signals") or {}
    confidence = signals.get("confidence")
    priority = signals.get("priority")
    signal_bits = []
    if confidence is not None:
        signal_bits.append(f"confidence={confidence}")
    if priority is not None:
        signal_bits.append(f"priority={priority}")
    if signal_bits:
        printColoured("\nSignals:", "cyan")
        printColoured("  " + " | ".join(signal_bits), "cyan")

    content = str(output.get("content", "") or "").strip()
    if content:
        printColoured("\nPipeline Output:", "magenta")
        if agent_name in ("curiosity", "creativity"):
            _print_pipeline_content(agent_name, content)
        else:
            _print_text_block(content, "white")

    next_hints = [str(item).strip() for item in (output.get("next_hints") or []) if str(item).strip()]
    if content:
        next_hints = [item for item in next_hints if item not in content]
    if next_hints:
        printColoured("\nNext Hints:", "green")
        _print_lines(next_hints[:5], "green")

    artifacts = output.get("artifacts") or []
    if artifacts:
        printColoured("\nArtifacts:", "yellow")
        _print_lines(_pretty_value(artifacts), "white")

    errors = [str(item).strip() for item in (output.get("errors") or []) if str(item).strip()]
    if errors:
        printColoured("\nErrors:", "red")
        _print_lines(errors[:5], "red")


def _print_generic_output(output: Any):
    if isinstance(output, dict):
        for key, value in output.items():
            printColoured(f"\n{key}:", "yellow")
            rendered = _pretty_value(value)
            if isinstance(value, str):
                _print_text_block("\n".join(rendered), "white")
            else:
                _print_lines(rendered, "white")
        return
    printColoured("\nOutput:", "yellow")
    _print_text_block(str(output), "white")


def _print_diagnostics_summary(diagnostics: Dict[str, Any]):
    printColoured("\nStats:", "magenta")
    event_count = diagnostics.get("event_count", 0)
    error_count = diagnostics.get("error_count", 0)
    recent_stages = diagnostics.get("recent_stages", []) or []
    stage_stats = diagnostics.get("stage_stats", {}) or {}
    printColoured(
        f"  events={event_count} | errors={error_count} | recent={', '.join(recent_stages) if recent_stages else 'n/a'}",
        "cyan",
    )
    if stage_stats:
        top_stage = max(stage_stats.items(), key=lambda item: item[1].get("max_ms", 0))
        printColoured(
            f"  slowest={top_stage[0]} | avg={_format_ms(top_stage[1].get('avg_ms', 0))} | max={_format_ms(top_stage[1].get('max_ms', 0))}",
            "cyan",
        )


def _test_protocol(prompt: str, monitor: PsycheMonitoring) -> Dict[str, Any]:
    def _do():
        sample_perception = PsycheProtocol.perception_state(
            original_input=prompt,
            inferred_intent="test a psyche module cleanly",
            expanded_context="standalone protocol smoke test",
        )
        sample_agent = PsycheProtocol.normalise_agent_output(
            "critic",
            {
                "summary": "This needs clearer structure.",
                "overall_score": 0.62,
                "critiques": [{"area": "clarity", "issue": "too vague", "suggestion": "be more explicit"}],
            },
            default_priority=0.7,
        )
        return {
            "perception_state": sample_perception,
            "agent_result": sample_agent,
            "compiled_state": PsycheProtocol.compile_state(sample_perception, exchanges=[sample_agent]),
        }
    output = _run_with_probe(monitor, "protocol", _do, input_payload=prompt, note="protocol smoke test")
    return _result_dict("protocol", prompt, output, monitor)


def _test_monitoring(prompt: str, monitor: PsycheMonitoring) -> Dict[str, Any]:
    probe = monitor.begin_stage("monitoring_demo", prompt)
    time.sleep(0.01)
    output = {"demo": "monitoring trace built", "prompt": prompt}
    monitor.end_stage(probe, output, note="monitoring smoke test")
    return _result_dict("monitoring", prompt, output, monitor)


def _test_factory_utils(prompt: str, monitor: PsycheMonitoring) -> Dict[str, Any]:
    fu = _import_module("factory_utils.py")
    output = _run_with_probe(
        monitor,
        "factory_utils",
        lambda: {
            "normalised_request": fu.normalise_factory_request({"task": prompt}),
            "contains_edit_markers": fu.contains_edit_markers("EDIT: app.py\n<<<< SEARCH\nx\n====\ny\n>>>> REPLACE"),
            "default_extensions": fu.normalise_extensions(None)[:12],
        },
        input_payload=prompt,
        note="factory utils smoke test",
    )
    return _result_dict("factory_utils", prompt, output, monitor)


def _test_perception(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_perception", "You analyze and expand user input with inferred context.")
    module = _import_module("perception.py").Perception(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "perception", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("perception", prompt, output, monitor)


def _test_rationality(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_rationality", "You are a deep thinker.", color="blue")
    module = _import_module("rationality.py").Rationality(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "rationality", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("rationality", prompt, output, monitor)


def _test_critic(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_critic", "You are a relentless critic.", color="red")
    module = _import_module("critic.py").Critic(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "critic", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("critic", prompt, output, monitor)


def _test_curiosity(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_curiosity", "You are a curious explorer.", color="yellow")
    module = _import_module("curiosity.py").Curiosity(
        agent=agent,
        embed_fn=ag.orchestrator.llm.create_embeddings,
        tools=getattr(ag.orchestrator, "tools", None),
        verbose=verbose,
    )
    output = _run_with_probe(monitor, "curiosity", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("curiosity", prompt, output, monitor)


def _test_futurevision(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_future", "You are a strategic foresight analyst.", color="blue")
    module = _import_module("futureVision.py").FutureVision(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "futureVision", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("futureVision", prompt, output, monitor)


def _test_selfawareness(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_self_awareness", "You are an introspective agent.", color="white")
    module = _import_module("selfAwareness.py").SelfAwareness(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "selfAwareness", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("selfAwareness", prompt, output, monitor)


def _test_emotion(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_emotion", "You feel emotions authentically.", color="yellow")
    module = _import_module("emotion.py").Emotion(agent=agent, embed_fn=ag.orchestrator.llm.create_embeddings, verbose=verbose)
    perception_output = {"original_input": prompt, "expanded_context": ""}
    output = _run_with_probe(
        monitor,
        "emotion",
        lambda: {"directive": module.steer(perception_output), "trajectory": module.get_trajectory_metrics()},
        input_payload=prompt,
        engine=agent,
    )
    return _result_dict("emotion", prompt, output, monitor)


def _test_dreamer(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_dreamer", "You are the unconscious mind.", color="cyan")
    module = _import_module("dreamer.py").Dreamer(agent=agent, embed_fn=ag.orchestrator.llm.create_embeddings, verbose=verbose)
    perception_output = {"original_input": prompt}
    output = _run_with_probe(
        monitor,
        "dreamer",
        lambda: {"directive": module.steer(perception_output), "corpus_size": module.get_dream_corpus_size()},
        input_payload=prompt,
        engine=agent,
    )
    return _result_dict("dreamer", prompt, output, monitor)


def _test_creativity(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_creativity", "You are a creative director.", color="green")
    shared_tools = getattr(ag.orchestrator, "tools", None)
    module = _import_module("creativity.py").Creativity(
        agent=agent,
        tools=shared_tools,
        verbose=verbose,
    )
    output = _run_with_probe(monitor, "creativity", lambda: module.process(prompt), input_payload=prompt, engine=agent)
    return _result_dict("creativity", prompt, output, monitor)


def _test_conscience(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_conscience", "You evaluate responses.", color="yellow")
    module = _import_module("conscience.py").Conscience(agent=agent, verbose=verbose)
    output = _run_with_probe(monitor, "conscience", lambda: module.evaluate(prompt), input_payload=prompt, engine=agent)
    return _result_dict("conscience", prompt, output, monitor)


def _test_persona(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_persona", "You are thoughtful and concise.", color="green")
    module = _import_module("persona.py").Persona(agent=agent, verbose=verbose)
    compiled_state = {
        "compiled_document": (
            f"## User Input\n{prompt}\n\n"
            "## Cognitive Agent Contributions\n"
            "### [rationality]\nSummary: brand clarity matters\n"
            "Clarity creates trust, but distinctiveness still matters."
        )
    }
    output = _run_with_probe(monitor, "persona", lambda: module.respond(compiled_state), input_payload=compiled_state, engine=agent)
    return _result_dict("persona", prompt, output, monitor)


def _test_orchestrator(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    orch_agent = _new_agent(ag, "standalone_orchestrator", "You are an orchestrator.", color="magenta")
    rationality_agent = _new_agent(ag, "standalone_orch_rationality", "You are a deep reasoning engine.", color="blue")
    critic_agent = _new_agent(ag, "standalone_orch_critic", "You are a relentless critic.", color="red")
    curiosity_agent = _new_agent(ag, "standalone_orch_curiosity", "You are a curious explorer.", color="yellow")
    module = _import_module("orchestrator.py").Orchestrator(agent=orch_agent, max_depth=2, monitor=monitor, verbose=verbose)
    rationality = _import_module("rationality.py").Rationality(agent=rationality_agent, verbose=verbose)
    critic = _import_module("critic.py").Critic(agent=critic_agent, verbose=verbose)
    curiosity = _import_module("curiosity.py").Curiosity(
        agent=curiosity_agent,
        embed_fn=ag.orchestrator.llm.create_embeddings,
        tools=getattr(ag.orchestrator, "tools", None),
        verbose=verbose,
    )
    module.register_agent("rationality", "Deep structured reasoning.", rationality.process)
    module.register_agent("critic", "Find flaws and weaknesses.", critic.process)
    module.register_agent("curiosity", "Find novel angles and connections.", curiosity.process)
    perception_output = {
        "original_input": prompt,
        "inferred_intent": prompt,
        "expanded_context": "standalone orchestrator test",
        "input_type": "text",
        "attachments": [],
        "extracted_data": None,
    }
    output = _run_with_probe(monitor, "orchestrator", lambda: module.process(perception_output), input_payload=perception_output, engine=orch_agent)
    return _result_dict("orchestrator", prompt, output, monitor)


def _test_factory(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    factory_module = _import_module("factory.py")
    module = factory_module.Factory(provider=provider, model=model, verbose=verbose, max_debug_rounds=1)
    request = {"task": prompt, "mode": "create", "entrypoint": "main.py"}
    output = _run_with_probe(monitor, "factory", lambda: module.build(request), input_payload=request)
    return _result_dict("factory", prompt, output, monitor)


def _test_heartbeat(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    ag = _build_agentgen(provider, model)
    agent = _new_agent(ag, "standalone_heartbeat_agent", "You are autonomous and reflective.", color="cyan")
    heartbeat_module = _import_module("heartbeat.py")
    module = heartbeat_module.Heartbeat(engine=agent, interval_minutes=999)
    output = _run_with_probe(monitor, "heartbeat", lambda: {"trigger": module.trigger(), "pending": module.drain_pending(), "status": module.status()}, input_payload=prompt, engine=agent)
    return _result_dict("heartbeat", prompt, output, monitor)


def _test_psyche(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    psyche_module = _import_module("psycheGen.py")
    module = psyche_module.PsycheGen(provider=provider, default_model=model, verbose=verbose)
    output = _run_with_probe(monitor, "psycheGen", lambda: {"response": module.process(prompt), "trace": module.get_monitoring_trace(20)}, input_payload=prompt)
    return _result_dict("psycheGen", prompt, output, monitor)


def _test_immunity(provider: str, model: Optional[str], prompt: str, verbose: bool, monitor: PsycheMonitoring) -> Dict[str, Any]:
    psyche_module = _import_module("psycheGen.py")
    psyche = psyche_module.PsycheGen(provider=provider, default_model=model, verbose=verbose)
    if prompt:
        try:
            psyche.process(prompt)
        except Exception:
            pass
    output = _run_with_probe(monitor, "immunity", lambda: {"diagnose": psyche.immunity.diagnose(), "heal": psyche.immunity.heal()}, input_payload=prompt)
    return _result_dict("immunity", prompt, output, monitor)


def _run_single_module(module_key: str, prompt: str, provider: Optional[str], model: Optional[str], verbose: bool) -> Dict[str, Any]:
    monitor = PsycheMonitoring(verbose=verbose)
    if module_key == "protocol":
        return _test_protocol(prompt, monitor)
    if module_key == "monitoring":
        return _test_monitoring(prompt, monitor)
    if module_key == "factory_utils":
        return _test_factory_utils(prompt, monitor)
    if module_key == "perception":
        return _test_perception(provider, model, prompt, verbose, monitor)
    if module_key == "rationality":
        return _test_rationality(provider, model, prompt, verbose, monitor)
    if module_key == "critic":
        return _test_critic(provider, model, prompt, verbose, monitor)
    if module_key == "curiosity":
        return _test_curiosity(provider, model, prompt, verbose, monitor)
    if module_key == "futureVision":
        return _test_futurevision(provider, model, prompt, verbose, monitor)
    if module_key == "selfAwareness":
        return _test_selfawareness(provider, model, prompt, verbose, monitor)
    if module_key == "emotion":
        return _test_emotion(provider, model, prompt, verbose, monitor)
    if module_key == "dreamer":
        return _test_dreamer(provider, model, prompt, verbose, monitor)
    if module_key == "creativity":
        return _test_creativity(provider, model, prompt, verbose, monitor)
    if module_key == "conscience":
        return _test_conscience(provider, model, prompt, verbose, monitor)
    if module_key == "persona":
        return _test_persona(provider, model, prompt, verbose, monitor)
    if module_key == "orchestrator":
        return _test_orchestrator(provider, model, prompt, verbose, monitor)
    if module_key == "factory":
        return _test_factory(provider, model, prompt, verbose, monitor)
    if module_key == "heartbeat":
        return _test_heartbeat(provider, model, prompt, verbose, monitor)
    if module_key == "psycheGen":
        return _test_psyche(provider, model, prompt, verbose, monitor)
    if module_key == "immunity":
        return _test_immunity(provider, model, prompt, verbose, monitor)
    raise ValueError(f"Unknown module key: {module_key}")


def _print_result(result: Dict[str, Any]):
    printColoured("\n" + "=" * 72, "magenta")
    printColoured(f"  {result['module']} :: result", "magenta")
    printColoured("=" * 72, "magenta")
    printColoured(f"Prompt: {result['prompt']}", "cyan")
    output = result.get("output")
    if isinstance(output, dict) and PsycheProtocol.validate_agent_result(output):
        _print_agent_output(output)
    else:
        _print_generic_output(output)
    _print_diagnostics_summary(result.get("diagnostics", {}))
    print()


def run_module_cli(module_key: Optional[str] = None):
    print("\n" + "=" * 72)
    print("  PSYCHE STANDALONE TESTS")
    print("=" * 72)

    module_keys = _pick_modules(preselected=module_key)
    _warn_heavy_modules(module_keys)
    needs_llm = any(MODULE_KEYS[key].get("needs_llm") for key in module_keys)

    provider = None
    model = None
    verbose = True
    if needs_llm:
        provider = _pick_provider()
        model = _pick_model(provider)
        verbose = _pick_verbose()
        printColoured(
            f"\nUsing provider={provider} | model={model or 'wrapper-default'} | verbose={'on' if verbose else 'off'}",
            "cyan",
        )

    results = []
    for key in module_keys:
        prompts = _pick_prompt(key)
        for prompt in prompts:
            try:
                result = _run_single_module(key, prompt, provider, model, verbose)
                results.append(result)
                _print_result(result)
            except Exception as e:
                printColoured(f"\n[{key}] test failed: {e}", "red")

    if len(results) > 1:
        printColoured("\nSummary:", "yellow")
        for result in results:
            stage_stats = result["diagnostics"].get("stage_stats", {})
            slowest = "n/a"
            if stage_stats:
                top_stage = max(stage_stats.items(), key=lambda item: item[1].get("max_ms", 0))
                slowest = f"{top_stage[0]} ({top_stage[1].get('max_ms', 0)} ms)"
            printColoured(
                f"  {result['module']:16} events={result['diagnostics'].get('event_count', 0):3} "
                f"errors={result['diagnostics'].get('error_count', 0):2} slowest={slowest}",
                "white",
            )

    printColoured("\nDone.\n", "green")


if __name__ == "__main__":
    run_module_cli()
