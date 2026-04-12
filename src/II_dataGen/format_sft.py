"""
format_sft.py - Convert full-loop JSONL into SFT chat-format training data.

Takes raw pipeline output and converts it into the Gemma 4 chat template
format expected by SFTTrainer / Unsloth.

The training signal:
  user:      the creative task
  assistant: the full visible reasoning trace (curiosity -> creativity -> critic -> output)
"""

import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def format_trace_as_text(example: dict) -> str:
    """Convert a full-loop example into the assistant's visible reasoning output."""
    parts = []

    for iteration in example.get("loop", []):
        it_num = iteration.get("iteration", "?")
        parts.append(f"## Iteration {it_num}")

        # Curiosity
        cur = iteration.get("curiosity", {})
        parts.append("### Curiosity")
        if cur.get("hidden_assumptions"):
            parts.append("Hidden assumptions:")
            for a in cur["hidden_assumptions"]:
                parts.append(f"- {a}")
        if cur.get("questions"):
            parts.append("Key questions:")
            for q in cur["questions"]:
                qtext = q.get("question", "") if isinstance(q, dict) else str(q)
                parts.append(f"- {qtext}")
        if cur.get("branch_seeds"):
            parts.append(f"Branch seeds: {', '.join(cur['branch_seeds'])}")

        # Creativity
        cre = iteration.get("creativity", {})
        parts.append("### Creativity")
        if cre.get("research"):
            parts.append("Research:")
            for r in cre["research"][:4]:
                parts.append(f"- {r}")
        if cre.get("branches"):
            parts.append("Branches:")
            for b in cre["branches"]:
                bid = b.get("id", "?")
                frame = b.get("frame", "")
                candidates = ", ".join(b.get("candidates", [])[:3])
                parts.append(f"- {bid}: {frame} [{candidates}]")
        if cre.get("pruned"):
            parts.append("Pruned:")
            for p in cre["pruned"]:
                parts.append(f"- {p.get('id', '?')}: {p.get('reason', '')}")
        if cre.get("combinations"):
            parts.append("Combinations:")
            for c in cre["combinations"]:
                parts.append(f"- {c.get('from', [])} -> {c.get('result', '')} ({c.get('novelty_note', '')})")
        if cre.get("output"):
            parts.append("Candidates:")
            for o in cre["output"]:
                parts.append(f"- {o}")

        # Critic
        cri = iteration.get("critic", {})
        parts.append("### Critic")
        verdict = cri.get("verdict", "?")
        parts.append(f"Verdict: {verdict}")
        if cri.get("scores"):
            for s in cri["scores"]:
                parts.append(f"- {s.get('candidate', '?')}: novelty={s.get('novelty', '?')}, relevance={s.get('relevance', '?')} | {s.get('notes', '')}")
        if verdict == "FAIL" and cri.get("feedback_for_curiosity"):
            parts.append("Feedback for next pass:")
            for fb in cri["feedback_for_curiosity"]:
                parts.append(f"- {fb}")

        parts.append("")

    # Final output
    final = example.get("final_output", [])
    if final:
        parts.append("## Final Output")
        for f in final:
            parts.append(f"- {f}")

    return "\n".join(parts)


def convert_to_chat_format(example: dict) -> dict:
    """Convert a full-loop example to a chat-format training row."""
    task = example.get("input", "")
    trace_text = format_trace_as_text(example)

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a creative reasoning engine. When given a task, you think through it "
                    "using a structured Curiosity -> Creativity -> Critic loop. You surface hidden "
                    "assumptions, explore diverse branches, cross-pollinate ideas, and critically "
                    "evaluate your own output before presenting final candidates."
                ),
            },
            {"role": "user", "content": task},
            {"role": "assistant", "content": trace_text},
        ]
    }


def process_jsonl(input_path: Path, output_path: Path):
    """Read raw pipeline JSONL, convert to SFT chat format, write output."""
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            chat = convert_to_chat_format(raw)
            examples.append(chat)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Converted {len(examples)} examples -> {output_path}")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Convert pipeline JSONL to SFT chat format")
    parser.add_argument("input", type=str, help="Input JSONL from pipeline")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: input_sft.jsonl)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_sft.jsonl")

    process_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
