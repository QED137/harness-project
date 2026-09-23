"""The evaluation harness end to end, with a scripted model and a fake sandbox:
no Docker, no network, no cost."""

import json

import pytest

from tsagent import evaluate as ev
from tsagent.evaluate import CONFIGS, build_tasks, estimate_cost, load_rows, run_evaluation
from tsagent.questions import BY_ID
from tsagent.sandbox import SandboxResult, SandboxStatus

from .test_agent import ScriptedLLM, call
from .test_planner import plan_json
from .test_tools import FakeSandbox, data_dir  # noqa: F401  (data_dir is a fixture)

Q = BY_ID["mean_temp_2023"]  # numeric, unit °C
Q2 = BY_ID["warmest_month_2023"]  # text answer
TRUTHS = {Q.id: 11.3139, Q2.id: "2023-07"}

RUN_PY = call("run_python", '{"code": "result = df.temperature_2m.mean()"}')


def submit(value, unit="°C"):
    return call("submit_answer", json.dumps({"value": value, "unit": unit, "method_summary": "mean"}))


class Factory:
    """Builds a fresh scripted model per evaluation; records how often it was asked."""

    def __init__(self, steps, plans=()):
        self.steps, self.plans, self.built = steps, plans, 0

    def __call__(self):
        self.built += 1
        return ScriptedLLM(list(self.steps) * 100, list(self.plans) * 100)


@pytest.fixture
def sandbox_factory():
    return lambda _dir: FakeSandbox(
        SandboxResult(status=SandboxStatus.OK, result={"type": "number", "value": 11.3139}, total_s=0.1)
    )


# ------------------------------------------------------------------ planning the work
def test_build_tasks_is_configs_times_questions_times_runs():
    tasks = build_tasks(["loop", "full"], [Q, Q2], 3)
    assert len(tasks) == 12
    assert {t.key for t in tasks} >= {("loop", Q.id, 0), ("full", Q2.id, 2)}


def test_cost_estimate_is_higher_with_a_planner():
    cheap, _ = estimate_cost(build_tasks(["loop"], [Q], 10), 0.05, 0.40)
    expensive, _ = estimate_cost(build_tasks(["planner"], [Q], 10), 0.05, 0.40)
    assert expensive > cheap > 0


def test_all_four_configs_are_defined():
    assert CONFIGS == {
        "loop": (False, False),
        "planner": (True, False),
        "verifier": (False, True),
        "full": (True, True),
    }


# ------------------------------------------------------------------ running
def test_correct_answers_are_graded_and_written(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    out = tmp_path / "runs.jsonl"
    rows = run_evaluation(
        build_tasks(["loop"], [Q], 2),
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)]),
        out,
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda msg: None,
    )
    assert len(rows) == 2 and all(r["correct"] for r in rows)
    on_disk = load_rows(out)
    assert len(on_disk) == 2
    assert on_disk[0]["expected"] == 11.3139 and on_disk[0]["got"] == 11.3139
    assert on_disk[0]["config"] == "loop" and on_disk[0]["answer_key"] is not None


def test_wrong_answers_are_recorded_with_a_reason(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    rows = run_evaluation(
        build_tasks(["loop"], [Q], 1),
        TRUTHS,
        Factory([RUN_PY, submit(20.0)]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda msg: None,
    )
    assert not rows[0]["correct"] and "off by" in rows[0]["reason"]


def test_equivalent_unit_is_accepted_by_the_harness(tmp_path, data_dir):  # noqa: F811
    """The agent answers in K, the question asks for °C: grading converts."""

    def sandbox(_dir):
        result = {"type": "number", "value": 284.4639}
        return FakeSandbox(SandboxResult(status=SandboxStatus.OK, result=result, total_s=0.1))

    rows = run_evaluation(
        build_tasks(["loop"], [Q], 1),
        TRUTHS,
        Factory([RUN_PY, submit(284.4639, "K")]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox,
        progress=lambda msg: None,
    )
    assert rows[0]["correct"] and rows[0]["unit_converted"] is True


def test_resume_skips_finished_runs(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    out = tmp_path / "runs.jsonl"
    tasks = build_tasks(["loop"], [Q], 3)
    first = Factory([RUN_PY, submit(11.3139)])
    run_evaluation(
        tasks[:2],
        TRUTHS,
        first,
        out,
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    second = Factory([RUN_PY, submit(11.3139)])
    rows = run_evaluation(
        tasks,
        TRUTHS,
        second,
        out,
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    assert len(rows) == 3 and len(load_rows(out)) == 3  # no duplicates
    assert {r["run_index"] for r in rows} == {0, 1, 2}


def test_nothing_left_to_do_makes_no_model(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    out = tmp_path / "runs.jsonl"
    tasks = build_tasks(["loop"], [Q], 1)
    run_evaluation(
        tasks,
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)]),
        out,
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    factory = Factory([RUN_PY, submit(11.3139)])
    run_evaluation(
        tasks,
        TRUTHS,
        factory,
        out,
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    assert factory.built == 0  # not a single API client was created, so nothing was paid for


def test_a_crashing_run_is_recorded_and_the_others_continue(tmp_path, data_dir):  # noqa: F811
    def exploding(_dir):
        raise RuntimeError("docker is not running")

    rows = run_evaluation(
        build_tasks(["loop"], [Q], 2),
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=exploding,
        progress=lambda m: None,
    )
    assert len(rows) == 2
    assert all(not r["correct"] and r["stop_reason"] == "harness_error" for r in rows)
    assert "docker is not running" in rows[0]["reason"]


def test_planner_and_verifier_configs_are_actually_used(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    rows = run_evaluation(
        build_tasks(["full"], [Q], 1),
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)], plans=[plan_json()]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    assert rows[0]["plan_status"] == "ok"  # the planner ran
    assert rows[0]["correct"]


def test_verifier_rejections_are_recorded_with_correctness(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    rows = run_evaluation(
        build_tasks(["verifier"], [Q], 1),
        TRUTHS,
        Factory([RUN_PY, submit(99.0), submit(11.3139)]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    rejected = rows[0]["rejected_answers"]
    assert rejected and rejected[0]["value"] == 99.0 and rejected[0]["was_correct"] is False
    assert "provenance" in rejected[0]["failed_checks"] and rows[0]["correct"]


def test_traces_are_written_when_asked(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    traces = tmp_path / "traces"
    run_evaluation(
        build_tasks(["loop"], [Q], 1),
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=1,
        traces_dir=traces,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    written = list(traces.rglob("*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["question"] == Q.text


def test_runs_execute_in_parallel(tmp_path, data_dir, sandbox_factory):  # noqa: F811
    rows = run_evaluation(
        build_tasks(["loop"], [Q, Q2], 2),
        TRUTHS,
        Factory([RUN_PY, submit(11.3139)]),
        tmp_path / "runs.jsonl",
        data_dir=data_dir,
        concurrency=4,
        sandbox_factory=sandbox_factory,
        progress=lambda m: None,
    )
    assert len(rows) == 4 and len(load_rows(tmp_path / "runs.jsonl")) == 4  # no lost or duplicated lines


# ------------------------------------------------------------------ CLI
def test_dry_run_shows_the_plan_without_spending(capsys):
    assert ev.main(["--dry-run", "--configs", "loop", "planner", "--runs", "10", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "2 configs x 5 questions x 10 runs = 100 runs" in out and "estimated cost" in out


def test_unknown_question_id_is_rejected(capsys):
    assert ev.main(["--questions", "does_not_exist", "--dry-run"]) == 2
    assert "unknown question ids" in capsys.readouterr().err


def test_report_reads_an_existing_file(tmp_path, capsys):
    from .test_metrics import row

    path = tmp_path / "runs.jsonl"
    path.write_text("\n".join(json.dumps(row(qid=f"q{i}", correct=i % 2 == 0)) for i in range(4)) + "\n")
    assert ev.main(["--report", str(path)]) == 0
    out = capsys.readouterr().out
    assert "summary" in out and "pass@1=0.500" in out and "hardest questions" in out


def test_report_on_empty_file_is_an_error(tmp_path, capsys):
    (tmp_path / "empty.jsonl").write_text("")
    assert ev.main(["--report", str(tmp_path / "empty.jsonl")]) == 2


# ------------------------------------------------------------------ regrading
def stored_row(qid, value, unit, correct, expected, reason="old reason"):
    from .test_metrics import row as base_row

    r = base_row(qid=qid, correct=correct)
    r.update(expected=expected, got=value, raw_value=value, answer_unit=unit, reason=reason, got_number=value)
    return r


def test_regrade_fixes_verdicts_without_touching_the_model():
    """The count-unit bug found in the first real evaluation: answers labelled 'records'
    were graded wrong. Regrading re-scores stored runs, with no API calls."""
    rows = [
        stored_row("hours_in_2023", 8760.0, "records", False, 8760.0, "unit 'records' is not convertible"),
        stored_row("hours_in_2023", 8760.0, "hours", True, 8760.0),
        stored_row("hours_in_2023", 8000.0, "records", False, 8760.0),
    ]
    regraded, changed = ev.regrade(rows)
    assert changed == 1
    assert [r["correct"] for r in regraded] == [True, True, False]
    assert regraded[0]["reason"] == "within tolerance"


def test_regrade_keeps_rows_it_cannot_judge():
    rows = [stored_row("mean_temp_2023", None, None, False, 11.3), stored_row("gone", 1.0, "°C", False, 1.0)]
    regraded, changed = ev.regrade(rows)
    assert changed == 0 and [r["correct"] for r in regraded] == [False, False]


def test_regrade_cli_writes_a_new_file_and_reports(tmp_path, capsys):
    path = tmp_path / "runs.jsonl"
    rows = [stored_row("hours_in_2023", 8760.0, "records", False, 8760.0) for _ in range(2)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert ev.main(["--regrade", str(path)]) == 0
    out = capsys.readouterr().out
    assert "2 verdicts changed: pass@1 0.000 -> 1.000" in out
    regraded = load_rows(tmp_path / "runs.regraded.jsonl")
    assert len(regraded) == 2 and all(r["correct"] for r in regraded)
    assert load_rows(path) == rows  # the original file is never modified
