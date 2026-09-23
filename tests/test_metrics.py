"""Metrics: hand-computed numbers, so a wrong metric cannot hide behind a plausible result."""

import math

import pytest

from tsagent.metrics import (
    answer_spread,
    breakdown,
    consistency,
    format_summary,
    hardest_questions,
    pass_at_k,
    summarize,
)


def row(qid="q1", correct=True, config="loop", **kw):
    base = {
        "config": config,
        "question_id": qid,
        "run_index": kw.pop("run_index", 0),
        "category": kw.pop("category", "aggregation"),
        "difficulty": kw.pop("difficulty", "easy"),
        "correct": correct,
        "reason": kw.pop("reason", "matches" if correct else "off by 1"),
        "got_number": kw.pop("got_number", 1.0),
        "answer_key": kw.pop("answer_key", "1"),
        "stop_reason": kw.pop("stop_reason", "answered"),
        "plan_status": kw.pop("plan_status", None),
        "turns": 3,
        "tool_status_counts": kw.pop("tool_status_counts", {"ok": 3}),
        "sandbox_violations": kw.pop("sandbox_violations", {}),
        "sandbox_limits": kw.pop("sandbox_limits", {}),
        "rejected_answers": kw.pop("rejected_answers", []),
        "input_tokens": kw.pop("input_tokens", 1000),
        "output_tokens": kw.pop("output_tokens", 200),
        "total_s": kw.pop("total_s", 10.0),
    }
    base.update(kw)
    return base


# ------------------------------------------------------------------ pass@k
@pytest.mark.parametrize(
    "n, correct, k, expected",
    [
        (10, 10, 1, 1.0),
        (10, 0, 1, 0.0),
        (10, 5, 1, 0.5),
        (10, 0, 5, 0.0),
        (10, 10, 5, 1.0),
        (10, 1, 10, 1.0),  # the one correct run is always in a sample of all 10
        (2, 1, 2, 1.0),
        (4, 2, 2, 1 - math.comb(2, 2) / math.comb(4, 2)),  # = 5/6
    ],
)
def test_pass_at_k(n, correct, k, expected):
    assert pass_at_k(n, correct, k) == pytest.approx(expected)


def test_pass_at_k_rewards_sometimes_right_systems():
    """The reason pass@1 is reported next to pass@k."""
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1)
    assert pass_at_k(10, 1, 5) == pytest.approx(0.5)


def test_k_larger_than_runs_is_an_error():
    with pytest.raises(ValueError, match="larger"):
        pass_at_k(3, 1, 5)


# ------------------------------------------------------------------ aggregates
def test_pass_at_1_is_the_share_of_correct_runs():
    rows = [row(correct=True), row(correct=True), row(correct=False), row(correct=False)]
    assert summarize(rows)["pass@1"] == 0.5


def test_consistency_and_spread():
    rows = [
        row(answer_key="1", got_number=1.0),
        row(answer_key="1", got_number=1.0),
        row(answer_key="2", got_number=3.0),
    ]
    assert consistency(rows) == pytest.approx(2 / 3)
    assert answer_spread(rows) == pytest.approx(1.1547, abs=1e-4)


def test_spread_needs_two_numeric_answers():
    assert answer_spread([row(got_number=None)]) is None


def test_validation_rates_are_per_tool_call():
    rows = [
        row(tool_status_counts={"ok": 3, "rule_violation": 1}),
        row(tool_status_counts={"ok": 4, "schema_violation": 2}),
    ]
    s = summarize(rows)
    assert s["tool_calls"] == 10
    assert s["tool_rule_violation_rate"] == pytest.approx(0.1)
    assert s["tool_schema_violation_rate"] == pytest.approx(0.2)


def test_verifier_catches_are_split_by_whether_the_answer_was_wrong():
    rows = [
        row(rejected_answers=[{"value": 1, "was_correct": False, "failed_checks": ["provenance"]}]),
        row(rejected_answers=[{"value": 2, "was_correct": True, "failed_checks": ["range"]}]),
    ]
    s = summarize(rows)
    assert s["verifier_correct_catches"] == 1 and s["verifier_false_rejections"] == 1


def test_sandbox_events_are_counted():
    rows = [row(sandbox_violations={"static_import": 2}, sandbox_limits={"soft_timeout": 1})]
    s = summarize(rows)
    assert s["sandbox_violations"] == {"static_import": 2} and s["sandbox_limits"] == {"soft_timeout": 1}


def test_pass_at_k_only_reported_when_enough_runs():
    rows = [row(qid="q1", run_index=i, correct=i < 2) for i in range(3)]
    s = summarize(rows)
    assert "pass@1" in s and "pass@5" not in s and s["runs_per_question_min"] == 3


def test_summary_of_no_runs_is_empty_not_an_error():
    assert summarize([])["runs"] == 0
    assert "no runs" in format_summary("loop", summarize([]))


# ------------------------------------------------------------------ breakdowns
def test_breakdown_by_category():
    rows = [
        row(qid="a", category="counting", correct=True),
        row(qid="b", category="counting", correct=False),
        row(qid="c", category="resampling", correct=True),
    ]
    out = breakdown(rows, "category")
    assert out["counting"]["pass@1"] == 0.5 and out["resampling"]["pass@1"] == 1.0
    assert out["counting"]["questions"] == 2


def test_hardest_questions_are_sorted_with_their_main_failure_reason():
    rows = [
        row(qid="easy_q", correct=True),
        row(qid="hard_q", correct=False, reason="off by 5"),
        row(qid="hard_q", correct=False, reason="off by 5", run_index=1),
    ]
    hardest = hardest_questions(rows)
    assert hardest[0]["question_id"] == "hard_q" and hardest[0]["pass@1"] == 0.0
    assert hardest[0]["top_reason"] == "off by 5"


def test_format_summary_is_one_readable_line():
    rows = [row(qid="q", run_index=i, correct=i < 8) for i in range(10)]
    text = format_summary("full", summarize(rows))
    assert "pass@1=0.800" in text and "pass@10=1.000" in text and "latency=10.0s" in text
