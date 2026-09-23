"""Step 7: the evaluation harness.

    python -m tsagent.evaluate --dry-run                  what would run, and what it costs
    python -m tsagent.evaluate --configs loop full --runs 3 --limit 5
    python -m tsagent.evaluate                            all configs, all questions, 10 runs
    python -m tsagent.evaluate --report eval/results/<run>/runs.jsonl   metrics only, no API calls

Every finished run is appended to runs.jsonl immediately and re-running skips what is
already there, so an interruption never wastes the runs already paid for.

Each run gets its own sandbox and its own tool registry, so runs are independent
and can execute in parallel.
"""

import argparse
import json
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from .agent import AgentRun, run_agent
from .grading import Grade, canonical_text, grade
from .llm import LLMClient, OpenAIClient, load_dotenv, openai_config_problem
from .metrics import breakdown, format_summary, hardest_questions, summarize
from .questions import BY_ID, QUESTIONS, Question, ground_truth, load_dataset
from .sandbox import DockerSandbox, SandboxConfig
from .schemas import SubmitAnswerArgs
from .tools import ToolRegistry

CONFIGS: dict[str, tuple[bool, bool]] = {  # name -> (use_planner, use_verifier)
    "loop": (False, False),
    "planner": (True, False),
    "verifier": (False, True),
    "full": (True, True),
}


@dataclass
class Task:
    config: str
    question: Question
    run_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.config, self.question.id, self.run_index)


def answer_key(question: Question, grade_result: Grade) -> str | None:
    """A comparable form of the answer, for the consistency metric: numbers rounded to
    the question's tolerance, text canonicalised."""
    got = grade_result.got
    if got is None:
        return None
    if isinstance(got, int | float):
        step = question.tolerance if question.tolerance > 0 else 1e-9
        return f"{round(float(got) / step) * step:.6g}"
    return canonical_text(str(got))


def row_from_run(task: Task, run: AgentRun, expected: float | str, grade_result: Grade) -> dict[str, Any]:
    """One JSONL row: everything needed for the metrics, without the full transcript."""
    tool_status: dict[str, int] = {}
    violations: dict[str, int] = {}
    limits: dict[str, int] = {}
    for record in run.tool_calls:
        tool_status[record.status.value] = tool_status.get(record.status.value, 0) + 1
        if record.sandbox is not None:
            for v in record.sandbox.violations:
                violations[v.kind.value] = violations.get(v.kind.value, 0) + 1
            for lim in record.sandbox.limits_hit:
                limits[lim.value] = limits.get(lim.value, 0) + 1
    rejected = [
        {
            "value": v.answer.value,
            "unit": v.answer.unit,
            "was_correct": grade(task.question, expected, v.answer).correct,
            "failed_checks": [c.name for c in v.verdict.failures()],
        }
        for v in run.verifications
        if not v.ok
    ]
    return {
        "config": task.config,
        "question_id": task.question.id,
        "run_index": task.run_index,
        "category": task.question.category,
        "difficulty": task.question.difficulty,
        "correct": grade_result.correct,
        "reason": grade_result.reason,
        "expected": expected,
        "got": grade_result.got,
        "raw_value": run.answer.value if run.answer else None,
        "got_number": grade_result.got if isinstance(grade_result.got, int | float) else None,
        "answer_key": answer_key(task.question, grade_result),
        "unit_converted": grade_result.unit_converted,
        "answer_unit": run.answer.unit if run.answer else None,
        "stop_reason": run.stop_reason.value if run.stop_reason else None,
        "plan_status": run.plan_status.value if run.plan_status else None,
        "turns": run.turns,
        "tool_status_counts": tool_status,
        "sandbox_violations": violations,
        "sandbox_limits": limits,
        "rejected_answers": rejected,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "total_s": run.total_s,
        "model": run.model,
        "error": run.error,
    }


def default_sandbox(data_dir: Path) -> DockerSandbox:
    return DockerSandbox(SandboxConfig(data_dir=data_dir, preload="/data/weather.parquet"))


def execute(
    task: Task,
    expected: float | str,
    llm: LLMClient,
    data_dir: Path,
    max_turns: int,
    traces_dir: Path | None,
    sandbox_factory: Callable[[Path], Any] = default_sandbox,
) -> dict[str, Any]:
    use_planner, use_verifier = CONFIGS[task.config]
    registry = ToolRegistry.default(data_dir, sandbox_factory(data_dir))
    run = run_agent(
        task.question.text, llm, registry, max_turns, use_planner=use_planner, use_verifier=use_verifier
    )
    result = grade(task.question, expected, run.answer)
    if traces_dir is not None:
        path = traces_dir / task.config / f"{task.question.id}_{task.run_index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run.to_dict(), indent=1, ensure_ascii=False) + "\n")
    return row_from_run(task, run, expected, result)


def regrade(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Re-apply the current grading rules to stored runs, without calling any model.

    Used when a grading bug is found after an expensive evaluation: the agent's answers
    are already recorded, only the verdict changes. Runs whose answer was not recorded
    (errors, no answer) stay as they are.
    """
    changed = 0
    out = []
    for row in rows:
        question = BY_ID.get(row["question_id"])
        value = row.get("raw_value", row.get("got"))
        if question is None or value is None:
            out.append(row)
            continue
        try:
            answer = SubmitAnswerArgs(value=value, unit=row.get("answer_unit"), method_summary="regraded")
        except ValidationError:
            out.append(row)
            continue
        result = grade(question, row["expected"], answer)
        new = dict(row)
        new.update(
            correct=result.correct,
            reason=result.reason,
            got=result.got,
            unit_converted=result.unit_converted,
            answer_key=answer_key(question, result),
        )
        changed += new["correct"] != row["correct"]
        out.append(new)
    return out, changed


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def run_evaluation(
    tasks: list[Task],
    truths: dict[str, float | str],
    llm_factory: Callable[[], LLMClient],
    out_file: Path,
    data_dir: Path = Path("data"),
    max_turns: int = 15,
    concurrency: int = 4,
    traces_dir: Path | None = None,
    sandbox_factory: Callable[[Path], Any] = default_sandbox,
    progress: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Run every task not already present in out_file; append results as they finish."""
    from concurrent.futures import ThreadPoolExecutor

    done = load_rows(out_file)
    finished = {(r["config"], r["question_id"], r["run_index"]) for r in done}
    todo = [t for t in tasks if t.key not in finished]
    progress(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return done

    out_file.parent.mkdir(parents=True, exist_ok=True)
    llm = llm_factory()
    lock = threading.Lock()
    started = time.monotonic()
    completed = 0

    def work(task: Task) -> dict[str, Any]:
        nonlocal completed
        try:
            row = execute(
                task, truths[task.question.id], llm, data_dir, max_turns, traces_dir, sandbox_factory
            )
        except Exception as e:  # noqa: BLE001  one broken run must not stop the evaluation
            row = {
                "config": task.config,
                "question_id": task.question.id,
                "run_index": task.run_index,
                "category": task.question.category,
                "difficulty": task.question.difficulty,
                "correct": False,
                "reason": f"harness error: {type(e).__name__}: {e}",
                "expected": truths[task.question.id],
                "got": None,
                "got_number": None,
                "answer_key": None,
                "stop_reason": "harness_error",
                "turns": 0,
                "tool_status_counts": {},
                "sandbox_violations": {},
                "sandbox_limits": {},
                "rejected_answers": [],
                "input_tokens": 0,
                "output_tokens": 0,
                "total_s": 0.0,
                "model": "",
                "error": f"{type(e).__name__}: {e}",
            }
        with lock:  # append immediately: an interruption never loses finished work
            with out_file.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed += 1
            elapsed = time.monotonic() - started
            eta = elapsed / completed * (len(todo) - completed)
            mark = "ok " if row["correct"] else "MISS"
            progress(
                f"[{completed}/{len(todo)}] {mark} {task.config:8} {task.question.id:30} "
                f"{row['reason'][:40]:40} eta {eta / 60:.0f}m"
            )
        return row

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        new_rows = list(pool.map(work, todo))
    return done + new_rows


# ------------------------------------------------------------------ reporting
def report(rows: list[dict[str, Any]], out: Callable[[str], None] = print) -> None:
    configs = sorted({r["config"] for r in rows})
    out("\n=== summary ===")
    for config in configs:
        out(format_summary(config, summarize([r for r in rows if r["config"] == config])))

    for field in ("category", "difficulty"):
        out(f"\n=== pass@1 by {field} ===")
        header = f"{field:16}" + "".join(f"{c:>12}" for c in configs)
        out(header)
        keys = sorted({r[field] for r in rows})
        for key in keys:
            line = f"{key:16}"
            for config in configs:
                subset = [r for r in rows if r["config"] == config and r[field] == key]
                line += f"{(breakdown(subset, field)[key]['pass@1'] if subset else float('nan')):>12.3f}"
            out(line)

    out("\n=== hardest questions (all configs pooled) ===")
    for q in hardest_questions(rows, limit=10):
        out(f"{q['pass@1']:.2f}  {q['question_id']:30} {q['category']:15} {q['top_reason'] or ''}")

    verifier_rows = [r for r in rows if r["config"] in ("verifier", "full")]
    if verifier_rows:
        s = summarize(verifier_rows)
        out(
            f"\nverifier: {s['verifier_correct_catches']} correct catches, "
            f"{s['verifier_false_rejections']} false rejections"
        )


def estimate_cost(tasks: list[Task], price_in: float, price_out: float) -> tuple[float, str]:
    """Rough estimate from measured token use: ~3.2k in / 0.9k out per run, ~5k / 3.8k with planner."""
    total = 0.0
    for t in tasks:
        tokens_in, tokens_out = (5000, 3800) if CONFIGS[t.config][0] else (3200, 900)
        total += tokens_in / 1e6 * price_in + tokens_out / 1e6 * price_out
    return total, f"~${total:.2f} at ${price_in}/M in, ${price_out}/M out (rough, from measured runs)"


# ------------------------------------------------------------------ CLI
def build_tasks(configs: Iterable[str], questions: list[Question], runs: int) -> list[Task]:
    return [Task(c, q, i) for c in configs for q in questions for i in range(runs)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tsagent.evaluate", description="Run the evaluation.")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--runs", type=int, default=10, help="runs per question (default 10)")
    p.add_argument("--limit", type=int, help="use only the first N questions (for a smoke test)")
    p.add_argument("--questions", nargs="+", help="specific question ids")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, help="output directory (default eval/results/<timestamp>)")
    p.add_argument("--traces", action="store_true", help="also save the full transcript of every run")
    p.add_argument("--dry-run", action="store_true", help="show the plan and cost estimate, run nothing")
    p.add_argument("--report", type=Path, help="only recompute metrics from an existing runs.jsonl")
    p.add_argument(
        "--regrade",
        type=Path,
        help="re-apply current grading rules to an existing runs.jsonl (no model calls); "
        "writes <name>.regraded.jsonl",
    )
    p.add_argument(
        "--price-in", type=float, default=0.05, help="$ per million input tokens, for the estimate"
    )
    p.add_argument("--price-out", type=float, default=0.40, help="$ per million output tokens")
    args = p.parse_args(argv)

    if args.regrade:
        rows = load_rows(args.regrade)
        if not rows:
            print(f"no runs found in {args.regrade}", file=sys.stderr)
            return 2
        regraded, changed = regrade(rows)
        target = args.regrade.with_suffix(".regraded.jsonl")
        target.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in regraded) + "\n")
        before = sum(r["correct"] for r in rows) / len(rows)
        after = sum(r["correct"] for r in regraded) / len(regraded)
        print(f"regraded {len(rows)} runs, {changed} verdicts changed: pass@1 {before:.3f} -> {after:.3f}")
        print(f"wrote {target}")
        report(regraded)
        return 0

    if args.report:
        rows = load_rows(args.report)
        if not rows:
            print(f"no runs found in {args.report}", file=sys.stderr)
            return 2
        report(rows)
        return 0

    selected = QUESTIONS
    if args.questions:
        ids = set(args.questions)
        selected = [q for q in QUESTIONS if q.id in ids]
        missing = ids - {q.id for q in selected}
        if missing:
            print(f"unknown question ids: {sorted(missing)}", file=sys.stderr)
            return 2
    if args.limit:
        selected = selected[: args.limit]

    tasks = build_tasks(args.configs, selected, args.runs)
    _, cost = estimate_cost(tasks, args.price_in, args.price_out)
    print(f"{len(args.configs)} configs x {len(selected)} questions x {args.runs} runs = {len(tasks)} runs")
    print(f"estimated cost: {cost}")
    if args.dry_run:
        return 0

    load_dotenv()
    problem = openai_config_problem()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    try:
        df = load_dataset(args.data_dir)
    except FileNotFoundError:
        print("dataset not found; run: python -m tsagent.dataset", file=sys.stderr)
        return 2

    truths = {q.id: ground_truth(q, df) for q in selected}
    out_dir = args.out or Path("eval/results") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "runs.jsonl"
    print(f"writing to {out_file}")

    import os

    rows = run_evaluation(
        tasks,
        truths,
        lambda: OpenAIClient(model=os.environ["OPENAI_MODEL"]),
        out_file,
        data_dir=args.data_dir,
        max_turns=args.max_turns,
        concurrency=args.concurrency,
        traces_dir=(out_dir / "traces") if args.traces else None,
    )
    report(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                c: summarize([r for r in rows if r["config"] == c])
                for c in sorted({r["config"] for r in rows})
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nresults: {out_file}\nsummary: {out_dir / 'summary.json'}")
    return 0


def dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convenience for exploring results in a notebook or a follow-up analysis."""
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raise SystemExit(main())
