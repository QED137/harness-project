"""Verifier (step 5): units, evidence, every check, and integration with the loop.
No LLM, no Docker, no network."""

import json

import pytest

from tsagent.agent import StopReason, format_run, run_agent
from tsagent.sandbox import SandboxResult, SandboxStatus
from tsagent.schemas import SubmitAnswerArgs
from tsagent.tools import ToolRegistry
from tsagent.verifier import CheckStatus, Evidence, collect_evidence, normalize_unit, verify

from .test_agent import SUBMIT, ScriptedLLM, call
from .test_planner import plan_json
from .test_tools import FakeSandbox, data_dir  # noqa: F401  (data_dir is a fixture)

MEAN_2023 = 11.313892694063927


def number(v: float) -> SandboxResult:
    return SandboxResult(
        status=SandboxStatus.OK,
        result={"type": "number", "value": v, "is_nan": False, "is_inf": False},
        total_s=0.1,
    )


def answer(value, unit="°C", summary="mean of hourly temperature_2m in 2023"):
    return SubmitAnswerArgs(value=value, unit=unit, method_summary=summary)


def raw(value_literal: str, unit="°C") -> str:
    return '{"value": ' + value_literal + ', "unit": ' + json.dumps(unit) + ', "method_summary": "m"}'


def status_of(verdict, name):
    return next(c.status for c in verdict.checks if c.name == name)


EVIDENCE = collect_evidence([number(MEAN_2023)])


# ------------------------------------------------------------------ units
@pytest.mark.parametrize(
    "unit, expected",
    [
        ("°C", ("°C", False)),
        ("degC", ("°C", False)),
        (" Celsius ", ("°C", False)),
        ("kmh", ("km/h", False)),
        ("m s-1", ("m/s", False)),
        ("mbar", ("hPa", False)),
        ("h", ("hours", False)),
        ("mm/day", ("mm/day", True)),
        ("°C per decade", ("°C/decade", True)),
        ("furlongs", (None, False)),
        ("mm/furlong", (None, False)),
        (None, (None, False)),
    ],
)
def test_normalize_unit(unit, expected):
    assert normalize_unit(unit) == expected


# ------------------------------------------------------------------ evidence
def test_evidence_from_numbers_stdout_and_frames():
    series = SandboxResult(
        status=SandboxStatus.OK,
        result={"type": "series", "shape": [2], "n_missing": 1, "head": '{"2023-01":1.5,"2023-02":null}'},
        stdout="monthly max 21.4\n",
        total_s=0.1,
    )
    failed = SandboxResult(status=SandboxStatus.ERROR, error="boom 999", total_s=0.1)
    ev = collect_evidence([number(3.25), series, failed, None])
    assert ev.n_runs == 2  # the failed run does not count as evidence
    assert {3.25, 1.5, 21.4} <= set(ev.numbers) and 999.0 not in ev.numbers
    assert ev.saw_nan is True  # the series had missing values


def test_nan_result_is_noted_not_used_as_evidence():
    nan = SandboxResult(
        status=SandboxStatus.OK,
        result={"type": "number", "value": None, "is_nan": True, "is_inf": False},
        total_s=0.1,
    )
    ev = collect_evidence([nan])
    assert ev.saw_nan and ev.numbers == []


# ------------------------------------------------------------------ checks
def test_correct_answer_passes_all_checks():
    v = verify(answer(MEAN_2023), EVIDENCE, raw(repr(MEAN_2023)))
    assert v.ok and all(c.status in (CheckStatus.PASS, CheckStatus.SKIP) for c in v.checks)


@pytest.mark.parametrize(
    "literal, ok",
    [
        ("11.31", True),  # rounded to 2 decimals
        ("11.3", True),
        ("11", True),  # rounded to an integer: tolerance ±0.5
        ("11.0", False),  # claims 1 decimal of precision, but 11.0 != 11.3
        ("11.32", False),  # wrong in the second decimal
        ("12", False),
    ],
)
def test_provenance_respects_the_precision_the_model_wrote(literal, ok):
    v = verify(answer(float(literal)), EVIDENCE, raw(literal))
    assert v.ok is ok, v.checks


def test_guessed_value_is_rejected():
    v = verify(answer(12.5), EVIDENCE, raw("12.5"))
    assert not v.ok and status_of(v, "provenance") is CheckStatus.FAIL
    assert "12.5" in v.message_for_model() and "run_python" in v.message_for_model()


def test_mental_unit_conversion_is_rejected():
    kelvin = MEAN_2023 + 273.15  # converted "in the head", never computed in run_python
    v = verify(answer(round(kelvin, 2), unit="K"), EVIDENCE, raw(f"{kelvin:.2f}", "K"))
    assert not v.ok and status_of(v, "provenance") is CheckStatus.FAIL
    assert status_of(v, "range") is CheckStatus.PASS  # plausible in K: only provenance catches it


def test_gross_unit_error_is_caught_by_range():
    ev = collect_evidence([number(97_000.0)])  # Pa computed, labelled as hPa
    v = verify(answer(97000.0, unit="hPa"), ev, raw("97000.0", "hPa"))
    assert not v.ok and status_of(v, "range") is CheckStatus.FAIL


def test_range_cannot_tell_plausible_units_apart():
    """Documented limitation: 3.5 is plausible in both km/h and m/s."""
    ev = collect_evidence([number(3.5)])
    assert verify(answer(3.5, unit="km/h"), ev, raw("3.5", "km/h")).ok
    assert verify(answer(3.5, unit="m/s"), ev, raw("3.5", "m/s")).ok


def test_text_with_unit_is_a_type_error():
    ev = collect_evidence([number(11.3)])
    v = verify(answer("about eleven", unit="°C"), ev, raw('"about eleven"'))
    assert status_of(v, "type") is CheckStatus.FAIL


def test_unknown_unit_only_warns():
    ev = collect_evidence([number(4.0)])
    v = verify(answer(4.0, unit="furlongs"), ev, raw("4.0", "furlongs"))
    assert v.ok and status_of(v, "unit") is CheckStatus.WARN and status_of(v, "range") is CheckStatus.SKIP


def test_rate_units_pass_without_range_check():
    ev = collect_evidence([number(0.08)])
    v = verify(answer(0.08, unit="°C/year"), ev, raw("0.08", "°C/year"))
    assert v.ok and status_of(v, "unit") is CheckStatus.PASS and status_of(v, "range") is CheckStatus.SKIP


def test_text_answers_must_appear_in_output():
    ts = SandboxResult(
        status=SandboxStatus.OK,
        result={"type": "timestamp", "value": "2023-07-01T00:00:00+00:00", "is_nat": False},
        total_s=0.1,
    )
    ev = collect_evidence([ts])
    assert verify(answer("2023-07", unit=None), ev, raw('"2023-07"', None)).ok
    assert not verify(answer("2021-07", unit=None), ev, raw('"2021-07"', None)).ok


def test_answer_without_any_computation_is_rejected():
    v = verify(answer(11.3), Evidence(), raw("11.3"))
    assert not v.ok and "no successful run_python" in v.message_for_model()


def test_not_answerable_is_accepted_as_an_abstention():
    v = verify(answer("not answerable", unit=None), Evidence(), raw('"not answerable"', None))
    assert v.ok and [c.name for c in v.checks] == ["not_answerable"]


def test_silent_nan_warns_unless_mentioned():
    ev = collect_evidence([number(5.0)])
    ev.saw_nan = True
    assert status_of(verify(answer(5.0), ev, raw("5.0")), "silent_nan") is CheckStatus.WARN
    mentioned = answer(5.0, summary="mean after dropping missing values")
    assert status_of(verify(mentioned, ev, raw("5.0")), "silent_nan") is CheckStatus.PASS
    assert verify(answer(5.0), ev, raw("5.0")).ok  # a warning never blocks


def test_plan_unit_mismatch_warns_and_aliases_match():
    ev = collect_evidence([number(3.0)])
    v = verify(answer(3.0, unit="m/s"), ev, raw("3.0", "m/s"), expected_unit="km/h")
    assert v.ok and status_of(v, "plan_unit") is CheckStatus.WARN
    v = verify(answer(3.0, unit="°C"), ev, raw("3.0"), expected_unit="degC")
    assert status_of(v, "plan_unit") is CheckStatus.PASS


def test_exponent_notation_uses_relative_tolerance():
    ev = collect_evidence([number(0.0012345)])
    assert verify(answer(0.0012345, unit="mm"), ev, raw("1.2345e-3", "mm")).ok


def test_message_lists_only_failures():
    ev = collect_evidence([number(4.0)])
    v = verify(answer(99.0, unit="furlongs"), ev, raw("99.0", "furlongs"))
    msg = v.message_for_model()
    assert "provenance" in msg and "furlongs" not in msg.split("provenance")[0].split("\n", 1)[1]


# ------------------------------------------------------------------ integration with the loop
@pytest.fixture
def sandbox():
    return FakeSandbox()  # every run_python returns 9.5; SUBMIT submits 9.5 °C


@pytest.fixture
def registry(data_dir, sandbox):  # noqa: F811
    return ToolRegistry.default(data_dir, sandbox)  # type: ignore[arg-type]


RUN = call("run_python", '{"code": "result = df.temperature_2m.mean()"}')
GUESS = call("submit_answer", '{"value": 12.0, "unit": "°C", "method_summary": "estimate"}')


def test_verifier_off_by_default(registry):
    run = run_agent("q", ScriptedLLM([call("submit_answer", SUBMIT)]), registry)
    assert run.use_verifier is False and run.verifications == []
    assert run.stop_reason is StopReason.ANSWERED  # unverified answers are accepted as before


def test_verified_answer_is_accepted(registry):
    run = run_agent("q", ScriptedLLM([RUN, call("submit_answer", SUBMIT)]), registry, use_verifier=True)
    assert run.stop_reason is StopReason.ANSWERED and run.answer.value == 9.5
    assert [v.ok for v in run.verifications] == [True]


def test_rejected_answer_goes_back_and_model_can_correct(registry):
    llm = ScriptedLLM([RUN, GUESS, call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry, use_verifier=True)
    assert [v.ok for v in run.verifications] == [False, True]
    assert run.stop_reason is StopReason.ANSWERED and run.answer.value == 9.5
    sent_back = llm.requests[2]["input"][-1]["output"]
    assert sent_back.startswith("Error: the answer was not accepted") and "12.0" in sent_back


def test_repeated_rejections_end_the_run(registry):
    llm = ScriptedLLM([RUN, GUESS, GUESS, GUESS, call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry, use_verifier=True)
    assert run.stop_reason is StopReason.REJECTED
    assert run.answer is None and run.rejected_answer.value == 12.0
    assert len(run.verifications) == 3  # 2 corrections allowed, the 3rd rejection stops


def test_max_rejections_is_configurable(registry):
    run = run_agent("q", ScriptedLLM([RUN, GUESS]), registry, use_verifier=True, max_rejections=0)
    assert run.stop_reason is StopReason.REJECTED and len(run.verifications) == 1


def test_evidence_comes_only_from_the_current_run(registry, sandbox):
    sandbox.result = number(12.0)
    run_agent("first", ScriptedLLM([RUN, GUESS]), registry, use_verifier=True)  # 12.0 computed here
    second = run_agent("second", ScriptedLLM([GUESS, GUESS, GUESS]), registry, use_verifier=True)
    assert second.stop_reason is StopReason.REJECTED  # 12.0 from the first run is not evidence


def test_planner_expected_unit_reaches_the_verifier(registry):
    llm = ScriptedLLM([RUN, call("submit_answer", SUBMIT)], plans=[plan_json(expected_unit="K")])
    run = run_agent("q", llm, registry, use_planner=True, use_verifier=True)
    checks = {c.name: c.status for c in run.verifications[0].verdict.checks}
    assert checks["plan_unit"] is CheckStatus.WARN and run.stop_reason is StopReason.ANSWERED


def test_run_record_with_verifications_is_json_serializable(registry):
    run = run_agent(
        "q", ScriptedLLM([RUN, GUESS, call("submit_answer", SUBMIT)]), registry, use_verifier=True
    )
    data = json.loads(json.dumps(run.to_dict()))
    first = data["verifications"][0]
    assert first["ok"] is False and first["answer"]["value"] == 12.0
    assert any(c["name"] == "provenance" and c["status"] == "fail" for c in first["verdict"]["checks"])


def test_format_run_shows_verifier_outcome(registry):
    run = run_agent(
        "q", ScriptedLLM([RUN, GUESS, call("submit_answer", SUBMIT)]), registry, use_verifier=True
    )
    text = format_run(run)
    assert "verifier: accepted after 1 rejection(s)" in text and "rejected 12.0: provenance" in text


# ------------------------------------------------------------------ command line
@pytest.mark.parametrize(
    "flags, planner, verifier",
    [
        ([], False, False),
        (["--planner"], True, False),
        (["--verifier"], False, True),
        (["--planner", "--verifier"], True, True),
    ],
)
def test_cli_flags_reach_the_agent(monkeypatch, data_dir, flags, planner, verifier):  # noqa: F811
    from tsagent import agent as agent_mod
    from tsagent.agent import AgentRun

    seen = {}

    def fake_run_agent(question, llm, registry, max_turns, use_planner=False, use_verifier=False):
        seen.update(use_planner=use_planner, use_verifier=use_verifier)
        return AgentRun(question=question, model="m", stop_reason=StopReason.ANSWERED)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setattr(agent_mod, "OpenAIClient", lambda model: ScriptedLLM([]))
    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)
    assert agent_mod.main(["q", "--data-dir", str(data_dir), *flags]) == 0
    assert seen == {"use_planner": planner, "use_verifier": verifier}
