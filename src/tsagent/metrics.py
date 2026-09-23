"""Metrics computed from evaluation runs.

Pure functions over plain dicts (the JSONL rows written by evaluate.py), so they
can be tested against hand-computed numbers without running any agent.

pass@1 is the answer a user actually gets; pass@k is the chance that at least one
of k attempts is right. Reporting both is deliberate: a system that is sometimes
right looks much better under pass@k than it is in practice.
"""

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

Row = dict[str, Any]


def pass_at_k(n: int, correct: int, k: int) -> float:
    """Unbiased estimate that at least one of k samples drawn from n runs is correct."""
    if k > n:
        raise ValueError(f"k={k} is larger than the number of runs n={n}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def by_question(rows: list[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        out[r["question_id"]].append(r)
    return dict(out)


def consistency(rows: list[Row]) -> float | None:
    """Mean share of runs per question that agree with that question's most common answer.
    1.0 means the agent always says the same thing (right or wrong)."""
    shares = []
    for runs in by_question(rows).values():
        keys = [r.get("answer_key") for r in runs if r.get("answer_key") is not None]
        if not keys:
            continue
        shares.append(Counter(keys).most_common(1)[0][1] / len(runs))
    return statistics.mean(shares) if shares else None


def answer_spread(rows: list[Row]) -> float | None:
    """Median standard deviation of the numeric answers within a question."""
    spreads = []
    for runs in by_question(rows).values():
        values = [r["got_number"] for r in runs if isinstance(r.get("got_number"), int | float)]
        if len(values) >= 2:
            spreads.append(statistics.stdev(values))
    return statistics.median(spreads) if spreads else None


def _rate(counts: dict[str, int], keys: tuple[str, ...]) -> float:
    total = sum(counts.values())
    return sum(counts.get(k, 0) for k in keys) / total if total else 0.0


def summarize(rows: list[Row], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, Any]:
    """All metrics for one configuration."""
    if not rows:
        return {"runs": 0}
    grouped = by_question(rows)
    runs_per_question = {q: len(rs) for q, rs in grouped.items()}
    n = min(runs_per_question.values())

    tool_counts: Counter[str] = Counter()
    plan_counts: Counter[str] = Counter()
    stop_counts: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    limit_counts: Counter[str] = Counter()
    good_catches = bad_catches = 0
    for r in rows:
        tool_counts.update(r.get("tool_status_counts", {}))
        stop_counts[r.get("stop_reason") or "none"] += 1
        if r.get("plan_status"):
            plan_counts[r["plan_status"]] += 1
        violation_counts.update(r.get("sandbox_violations", {}))
        limit_counts.update(r.get("sandbox_limits", {}))
        for rejected in r.get("rejected_answers", []):
            if rejected["was_correct"]:
                bad_catches += 1  # the verifier rejected a correct answer
            else:
                good_catches += 1

    summary: dict[str, Any] = {
        "runs": len(rows),
        "questions": len(grouped),
        "runs_per_question_min": n,
        "pass@1": statistics.mean(1.0 if r["correct"] else 0.0 for r in rows),
        "consistency": consistency(rows),
        "answer_spread_median": answer_spread(rows),
        "tool_calls": sum(tool_counts.values()),
        "tool_schema_violation_rate": _rate(tool_counts, ("schema_violation",)),
        "tool_rule_violation_rate": _rate(tool_counts, ("rule_violation",)),
        "tool_invalid_json_rate": _rate(tool_counts, ("invalid_json",)),
        "stop_reasons": dict(stop_counts),
        "plan_status": dict(plan_counts),
        "sandbox_violations": dict(violation_counts),
        "sandbox_limits": dict(limit_counts),
        "verifier_correct_catches": good_catches,
        "verifier_false_rejections": bad_catches,
        "median_latency_s": statistics.median(r["total_s"] for r in rows),
        "median_input_tokens": statistics.median(r["input_tokens"] for r in rows),
        "median_output_tokens": statistics.median(r["output_tokens"] for r in rows),
        "total_input_tokens": sum(r["input_tokens"] for r in rows),
        "total_output_tokens": sum(r["output_tokens"] for r in rows),
    }
    for k in ks:
        if k <= n:
            summary[f"pass@{k}"] = statistics.mean(
                pass_at_k(len(rs), sum(r["correct"] for r in rs), k) for rs in grouped.values()
            )
    return summary


def breakdown(rows: list[Row], field: str) -> dict[str, dict[str, Any]]:
    """pass@1 split by question category or difficulty: where does it fail?"""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.get(field, "?")].append(r)
    return {
        key: {
            "runs": len(rs),
            "questions": len({r["question_id"] for r in rs}),
            "pass@1": statistics.mean(1.0 if r["correct"] else 0.0 for r in rs),
        }
        for key, rs in sorted(groups.items())
    }


def hardest_questions(rows: list[Row], limit: int = 10) -> list[dict[str, Any]]:
    """Questions with the lowest pass rate: the starting point for failure analysis."""
    out = []
    for qid, rs in by_question(rows).items():
        correct = sum(r["correct"] for r in rs)
        reasons = Counter(r["reason"] for r in rs if not r["correct"])
        out.append(
            {
                "question_id": qid,
                "category": rs[0].get("category"),
                "difficulty": rs[0].get("difficulty"),
                "pass@1": correct / len(rs),
                "runs": len(rs),
                "top_reason": reasons.most_common(1)[0][0] if reasons else None,
            }
        )
    return sorted(out, key=lambda d: (d["pass@1"], d["question_id"]))[:limit]


def format_summary(name: str, s: dict[str, Any]) -> str:
    if not s.get("runs"):
        return f"{name}: no runs"
    parts = [
        f"{name:10} runs={s['runs']:<5} pass@1={s['pass@1']:.3f}",
        f"pass@5={s['pass@5']:.3f}" if "pass@5" in s else "",
        f"pass@10={s['pass@10']:.3f}" if "pass@10" in s else "",
        f"consistency={s['consistency']:.3f}" if s.get("consistency") is not None else "",
        f"latency={s['median_latency_s']:.1f}s",
        f"tokens={s['median_input_tokens']:.0f}in/{s['median_output_tokens']:.0f}out",
    ]
    return "  ".join(p for p in parts if p)
