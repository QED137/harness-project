"""Step 8: turn stored runs into documentation.

    python -m tsagent.analyze results  --runs eval/results/loop/runs.regraded.jsonl --out docs/results.md
    python -m tsagent.analyze failures --runs eval/results/loop/runs.regraded.jsonl \
        --traces eval/results/loop/traces --out docs/failures.md

Both read only files on disk: no model is called and nothing is paid for.

The failure report extracts, for every failed run, the code the agent actually ran.
Categories are suggested from the grading reason, but they are a starting point for
reading the traces, not a substitute for it.
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .evaluate import load_rows
from .metrics import breakdown, hardest_questions, summarize
from .questions import BY_ID

Row = dict[str, Any]

# reason text -> suggested failure category
CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("no answer was submitted", "no_answer"),
    ("harness error", "harness_error"),
    ("does not count", "unit_label"),
    ("not convertible", "unit_label"),
    ("expected a number", "wrong_type"),
    ("cannot answer", "answered_unanswerable"),
    ("declined an answerable", "wrongly_declined"),
    ("off by", "wrong_value"),
    ("expected ", "wrong_value"),
)


def categorize(row: Row) -> str:
    reason = (row.get("reason") or "").lower()
    for needle, category in CATEGORY_RULES:
        if needle in reason:
            return category
    return "other"


def trace_path(traces_dir: Path, row: Row) -> Path:
    return traces_dir / row["config"] / f"{row['question_id']}_{row['run_index']}.json"


def load_trace(traces_dir: Path | None, row: Row) -> dict[str, Any] | None:
    if traces_dir is None:
        return None
    path = trace_path(traces_dir, row)
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def agent_code(trace: dict[str, Any]) -> list[str]:
    """Every piece of Python the agent ran, in order."""
    out = []
    for call in trace.get("tool_calls", []):
        if call.get("tool") != "run_python":
            continue
        try:
            out.append(json.loads(call.get("raw_arguments") or "{}").get("code", ""))
        except json.JSONDecodeError:
            out.append(call.get("raw_arguments", ""))
    return [c for c in out if c]


# ------------------------------------------------------------------ results
def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def results_markdown(rows: list[Row]) -> str:
    configs = sorted({r["config"] for r in rows})
    models = sorted({str(r["model"]) for r in rows if r.get("model")})
    summaries = {c: summarize([r for r in rows if r["config"] == c]) for c in configs}

    out = ["# Results", ""]
    out.append(
        f"{len(rows)} runs, {len({r['question_id'] for r in rows})} questions, "
        f"model(s): {', '.join(models) or 'unknown'}."
    )
    out.append("")
    out.append("## Summary per configuration")
    out.append("")
    metric_rows = []
    for c in configs:
        s = summaries[c]
        metric_rows.append(
            [
                c,
                str(s["runs"]),
                f"{s['pass@1']:.3f}",
                f"{s.get('pass@5', float('nan')):.3f}",
                f"{s.get('pass@10', float('nan')):.3f}",
                f"{s['consistency']:.3f}" if s.get("consistency") is not None else "-",
                f"{s['median_latency_s']:.1f}",
                f"{s['median_input_tokens']:.0f} / {s['median_output_tokens']:.0f}",
            ]
        )
    out.append(
        _table(
            ["config", "runs", "pass@1", "pass@5", "pass@10", "consistency", "median s", "tokens in/out"],
            metric_rows,
        )
    )

    for field in ("category", "difficulty"):
        out += ["", f"## pass@1 by {field}", ""]
        keys = sorted({r[field] for r in rows})
        table_rows = []
        for key in keys:
            line = [key]
            for c in configs:
                subset = [r for r in rows if r["config"] == c and r[field] == key]
                line.append(f"{breakdown(subset, field)[key]['pass@1']:.3f}" if subset else "-")
            table_rows.append(line)
        out.append(_table([field, *configs], table_rows))

    out += ["", "## Hardest questions", ""]
    out.append(
        _table(
            ["question", "category", "pass@1", "most common failure"],
            [
                [q["question_id"], q["category"] or "-", f"{q['pass@1']:.2f}", (q["top_reason"] or "-")[:60]]
                for q in hardest_questions(rows, limit=10)
            ],
        )
    )

    verifier_rows = [r for r in rows if r["config"] in ("verifier", "full")]
    if verifier_rows:
        s = summarize(verifier_rows)
        out += [
            "",
            "## Verifier",
            "",
            f"- correct catches (rejected a wrong answer): {s['verifier_correct_catches']}",
            f"- false rejections (rejected a correct answer): {s['verifier_false_rejections']}",
        ]

    events: dict[str, int] = {}
    for r in rows:
        for kind, n in (r.get("sandbox_violations") or {}).items():
            events[kind] = events.get(kind, 0) + n
        for kind, n in (r.get("sandbox_limits") or {}).items():
            events[kind] = events.get(kind, 0) + n
    out += ["", "## Sandbox events", ""]
    out.append(
        _table(["event", "count"], [[k, str(v)] for k, v in sorted(events.items())])
        if events
        else "None recorded."
    )

    tool_totals = summarize(rows)
    out += [
        "",
        "## Tool calls",
        "",
        f"- total tool calls: {tool_totals['tool_calls']}",
        f"- schema-level validation failures: {tool_totals['tool_schema_violation_rate']:.4f} per call",
        f"- rule-level validation failures: {tool_totals['tool_rule_violation_rate']:.4f} per call",
        f"- invalid JSON: {tool_totals['tool_invalid_json_rate']:.4f} per call",
        "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ failures
def failures_markdown(rows: list[Row], traces_dir: Path | None, max_per_question: int = 2) -> str:
    failed = [r for r in rows if not r["correct"]]
    out = ["# Failure analysis", ""]
    if not failed:
        return "\n".join([*out, "No failed runs in this result set."])

    counts: dict[str, int] = {}
    for r in failed:
        c = categorize(r)
        counts[c] = counts.get(c, 0) + 1
    out.append(f"{len(failed)} failed runs out of {len(rows)} ({len(failed) / len(rows):.1%}).")
    out += ["", "## Failure categories (suggested from the grading reason)", ""]
    out.append(
        _table(["category", "runs"], [[k, str(v)] for k, v in sorted(counts.items(), key=lambda kv: -kv[1])])
    )

    by_question: dict[str, list[Row]] = {}
    for r in failed:
        by_question.setdefault(r["question_id"], []).append(r)

    out += ["", "## Failed runs by question", ""]
    for qid, runs in sorted(by_question.items(), key=lambda kv: -len(kv[1])):
        question = BY_ID.get(qid)
        out += [f"### {qid} ({len(runs)} failed)", ""]
        if question is not None:
            out += [f"*{question.text}*", ""]
            if question.traps:
                out += [f"Intended trap: {', '.join(question.traps)}", ""]
        for r in runs[:max_per_question]:
            out += [
                f"- run {r['run_index']} ({r['config']}): expected `{r.get('expected')}`, "
                f"got `{r.get('got')}` {r.get('answer_unit') or ''} - {r['reason']}"
            ]
            trace = load_trace(traces_dir, r)
            if trace is None:
                continue
            if trace.get("plan") and trace["plan"].get("interpretation"):
                out += ["", f"  Plan interpretation: {trace['plan']['interpretation']}"]
            for code in agent_code(trace)[-2:]:
                out += ["", "```python", code.strip(), "```"]
            for v in trace.get("verifications", []):
                if not v.get("ok"):
                    checks = [c["name"] for c in v["verdict"]["checks"] if c["status"] == "fail"]
                    out.append(f"  Verifier rejected {v['answer']['value']}: {', '.join(checks)}")
        out.append("")
    out += [
        "## Notes",
        "",
        "Categories above are derived from the grading reason and are only a starting point.",
        "Read the code of each failed run before drawing conclusions.",
        "",
    ]
    return "\n".join(out)


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None, out: Callable[[str], None] = print) -> int:
    p = argparse.ArgumentParser(
        prog="python -m tsagent.analyze", description="Documentation from stored runs."
    )
    p.add_argument("what", choices=["results", "failures"])
    p.add_argument("--runs", type=Path, required=True, help="a runs.jsonl file")
    p.add_argument("--traces", type=Path, help="the traces directory (failures only)")
    p.add_argument("--out", type=Path, help="write to this file instead of printing")
    args = p.parse_args(argv)

    rows = load_rows(args.runs)
    if not rows:
        print(f"no runs found in {args.runs}", file=sys.stderr)
        return 2
    if args.traces is not None and not args.traces.is_dir():
        print(f"traces directory not found: {args.traces}", file=sys.stderr)
        return 2

    text = results_markdown(rows) if args.what == "results" else failures_markdown(rows, args.traces)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        out(f"wrote {args.out} ({len(text.splitlines())} lines)")
    else:
        try:
            out(text)
        except BrokenPipeError:  # e.g. piped into `head`
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
