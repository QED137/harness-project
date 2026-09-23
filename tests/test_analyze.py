"""Documentation generation from stored runs and traces. Reads files only."""

import json

import pytest

from tsagent import analyze
from tsagent.analyze import agent_code, categorize, failures_markdown, results_markdown

from .test_metrics import row


def trace(code="result = df.temperature_2m.mean()", plan=None, verifications=()):
    return {
        "question": "q?",
        "tool_calls": [
            {"tool": "describe_dataset", "raw_arguments": "{}"},
            {"tool": "run_python", "raw_arguments": json.dumps({"code": code})},
        ],
        "plan": plan,
        "verifications": list(verifications),
    }


@pytest.fixture
def runs():
    return [
        row(qid="mean_temp_2023", correct=True),
        row(
            qid="dry_days_2023",
            correct=False,
            reason="off by 1 (tolerance 0.5 days)",
            run_index=3,
            category="counting",
            difficulty="medium",
        ),
        row(
            qid="berlin_temperature",
            correct=False,
            reason="answered a question the dataset cannot answer",
            category="not_answerable",
            difficulty="hard",
            run_index=7,
        ),
    ]


# ------------------------------------------------------------------ categories
@pytest.mark.parametrize(
    "reason, expected",
    [
        ("off by 1 (tolerance 0.5 days)", "wrong_value"),
        ("unit 'records' does not count hours", "unit_label"),
        ("answered a question the dataset cannot answer", "answered_unanswerable"),
        ("declined an answerable question", "wrongly_declined"),
        ("no answer was submitted", "no_answer"),
        ("harness error: RuntimeError: boom", "harness_error"),
        ("expected a number, got 'about eleven'", "wrong_type"),
        ("something new", "other"),
    ],
)
def test_categorize(reason, expected):
    assert categorize(row(correct=False, reason=reason)) == expected


# ------------------------------------------------------------------ results
def test_results_markdown_has_the_tables(runs):
    md = results_markdown(runs)
    assert md.startswith("# Results")
    assert "## Summary per configuration" in md and "| config |" in md
    assert "## pass@1 by category" in md and "## pass@1 by difficulty" in md
    assert "## Hardest questions" in md and "## Tool calls" in md
    assert "0.333" in md  # pass@1 of the three example runs


def test_results_markdown_compares_configs():
    rows = [row(qid="q", config="loop", correct=True), row(qid="q", config="full", correct=False)]
    md = results_markdown(rows)
    header = next(line for line in md.splitlines() if line.startswith("| config |"))
    assert "full" in md and "loop" in md and "runs" in header


def test_verifier_section_only_when_relevant(runs):
    assert "## Verifier" not in results_markdown(runs)
    with_verifier = [
        row(
            config="verifier",
            correct=False,
            rejected_answers=[{"value": 1, "was_correct": False, "failed_checks": ["provenance"]}],
        )
    ]
    assert "correct catches" in results_markdown(with_verifier)


# ------------------------------------------------------------------ failures
def test_failures_markdown_lists_questions_and_code(tmp_path, runs):
    traces = tmp_path / "traces" / "loop"
    traces.mkdir(parents=True)
    (traces / "dry_days_2023_3.json").write_text(
        json.dumps(trace(code="result = (df.precipitation.resample('D').sum() < 0.1).sum()"))
    )
    md = failures_markdown(runs, tmp_path / "traces")
    assert "2 failed runs out of 3 (66.7%)" in md
    assert "### dry_days_2023 (1 failed)" in md
    assert "resample('D')" in md and "```python" in md
    assert "wrong_value" in md and "answered_unanswerable" in md


def test_failures_markdown_without_traces(runs):
    md = failures_markdown(runs, None)
    assert "### berlin_temperature" in md and "```python" not in md


def test_failure_report_shows_plan_and_verifier_rejections(tmp_path, runs):
    traces = tmp_path / "traces" / "loop"
    traces.mkdir(parents=True)
    (traces / "dry_days_2023_3.json").write_text(
        json.dumps(
            trace(
                plan={"interpretation": "count days with less than 0.1 mm"},
                verifications=[
                    {
                        "ok": False,
                        "answer": {"value": 130.0},
                        "verdict": {"checks": [{"name": "provenance", "status": "fail", "detail": "x"}]},
                    }
                ],
            )
        )
    )
    md = failures_markdown(runs, tmp_path / "traces")
    assert "Plan interpretation: count days with less than 0.1 mm" in md
    assert "Verifier rejected 130.0: provenance" in md


def test_intended_trap_is_shown_for_known_questions(tmp_path):
    rows = [row(qid="mean_wind_2023_ms", correct=False, reason="off by 3.0", category="unit_conversion")]
    md = failures_markdown(rows, None)
    assert "Intended trap: data is in km/h" in md


def test_no_failures_says_so(runs):
    assert "No failed runs" in failures_markdown([r for r in runs if r["correct"]], None)


def test_agent_code_extracts_every_run_python_call():
    t = trace()
    t["tool_calls"].append({"tool": "run_python", "raw_arguments": json.dumps({"code": "result = 2"})})
    t["tool_calls"].append({"tool": "run_python", "raw_arguments": "not json"})
    assert agent_code(t) == ["result = df.temperature_2m.mean()", "result = 2", "not json"]


# ------------------------------------------------------------------ CLI
def test_cli_writes_files(tmp_path, runs, capsys):
    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    out = tmp_path / "docs" / "results.md"
    assert analyze.main(["results", "--runs", str(runs_file), "--out", str(out)]) == 0
    assert out.read_text().startswith("# Results")
    assert "wrote" in capsys.readouterr().out


def test_cli_prints_when_no_out_file(tmp_path, runs):
    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text("\n".join(json.dumps(r) for r in runs) + "\n")
    printed: list[str] = []
    assert analyze.main(["failures", "--runs", str(runs_file)], out=printed.append) == 0
    assert "# Failure analysis" in printed[0]


def test_cli_errors(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    assert analyze.main(["results", "--runs", str(empty)]) == 2
    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text(json.dumps(row()) + "\n")
    assert analyze.main(["failures", "--runs", str(runs_file), "--traces", str(tmp_path / "nope")]) == 2
    assert "traces directory not found" in capsys.readouterr().err


def test_broken_pipe_is_not_an_error(tmp_path):
    """e.g. `python -m tsagent.analyze results --runs ... | head`"""

    def raising(_text):
        raise BrokenPipeError

    runs_file = tmp_path / "runs.jsonl"
    runs_file.write_text(json.dumps(row()) + "\n")
    assert analyze.main(["results", "--runs", str(runs_file)], out=raising) == 0
