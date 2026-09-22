"""Planner (step 4): validation levels, prompt, and integration with the agent loop.
Uses the scripted fake model: no network, no cost."""

import json
from types import SimpleNamespace

import pytest

from tsagent.agent import PLAN_INTRO, StopReason, format_run, run_agent
from tsagent.llm import OpenAIClient
from tsagent.planner import PLANNER_PROMPT, PlanStatus, format_plan, make_plan, plan_schema
from tsagent.schemas import Plan
from tsagent.tools import ToolRegistry

from .test_agent import SUBMIT, ScriptedLLM, call
from .test_tools import FakeSandbox, data_dir  # noqa: F401  (data_dir is a fixture)

STEPS = [
    {"tool": "describe_dataset", "description": "check columns and units"},
    {"tool": "run_python", "description": "mean of temperature_2m for 2023 (UTC)"},
    {"tool": "submit_answer", "description": "submit the mean in °C"},
]


def plan_json(**overrides) -> str:
    plan = {
        "interpretation": "Mean of hourly 2 m temperature over calendar year 2023, in °C.",
        "assumptions": ["year boundaries in UTC"],
        "answerable": True,
        "expected_unit": "°C",
        "steps": STEPS,
    }
    plan.update(overrides)
    return json.dumps(plan)


@pytest.fixture
def registry(data_dir):  # noqa: F811
    return ToolRegistry.default(data_dir, FakeSandbox())  # type: ignore[arg-type]


# ------------------------------------------------------------------ make_plan
def test_valid_plan():
    llm = ScriptedLLM([], plans=[plan_json()])
    result = make_plan("mean temperature 2023?", llm, "DATASET DESCRIPTION")
    assert result.status is PlanStatus.OK
    assert [s.tool for s in result.plan.steps] == ["describe_dataset", "run_python", "submit_answer"]
    assert result.record.purpose == "planner"


def test_planner_sees_prompt_description_and_strict_schema():
    llm = ScriptedLLM([], plans=[plan_json()])
    make_plan("q", llm, "DATASET DESCRIPTION")
    req = llm.plan_requests[0]
    assert req["instructions"].startswith(PLANNER_PROMPT) and "DATASET DESCRIPTION" in req["instructions"]
    assert req["input"] == [{"role": "user", "content": "q"}]
    assert req["schema_name"] == "plan" and req["schema"] == plan_schema()
    assert "$ref" not in json.dumps(req["schema"])


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("not json", PlanStatus.INVALID_JSON),
        ('{"interpretation": "x"', PlanStatus.INVALID_JSON),
        # schema level: what provider strict mode should prevent
        (json.dumps({"interpretation": "x"}), PlanStatus.SCHEMA_VIOLATION),
        (plan_json(steps=[{"tool": "delete_everything", "description": "d"}]), PlanStatus.SCHEMA_VIOLATION),
        (plan_json(steps="do it"), PlanStatus.SCHEMA_VIOLATION),
        (plan_json(answerable="yes please"), PlanStatus.SCHEMA_VIOLATION),
        (plan_json(extra_field=1), PlanStatus.SCHEMA_VIOLATION),
        # rule level: only Pydantic knows these
        (plan_json(steps=[]), PlanStatus.RULE_VIOLATION),
        (plan_json(steps=STEPS[:2]), PlanStatus.RULE_VIOLATION),  # no submit_answer
        (plan_json(steps=[STEPS[2], STEPS[1]]), PlanStatus.RULE_VIOLATION),  # submit not last
        (plan_json(steps=[STEPS[2], STEPS[2]]), PlanStatus.RULE_VIOLATION),  # two submits
        (plan_json(steps=[STEPS[1]] * 8 + [STEPS[2]]), PlanStatus.RULE_VIOLATION),  # 9 steps
        (plan_json(interpretation=""), PlanStatus.RULE_VIOLATION),
        (plan_json(assumptions=["  "]), PlanStatus.RULE_VIOLATION),
        (plan_json(assumptions=["a"] * 6), PlanStatus.RULE_VIOLATION),
    ],
)
def test_every_plan_ends_in_exactly_one_classified_status(raw, expected):
    result = make_plan("q", ScriptedLLM([], plans=[raw]), "desc")
    assert result.status is expected, result.error
    assert result.raw == raw  # exactly what the model returned is kept for the trace
    if expected is not PlanStatus.OK:
        assert result.plan is None and result.error


def test_llm_error_is_recorded_not_raised():
    result = make_plan("q", ScriptedLLM([], plans=[TimeoutError("slow")]), "desc")
    assert result.status is PlanStatus.LLM_ERROR and "slow" in result.error


def test_format_plan():
    plan = Plan.model_validate_json(plan_json(answerable=False))
    text = format_plan(plan)
    assert "1. [describe_dataset]" in text and "3. [submit_answer]" in text
    assert "Expected unit: °C" in text and "NOT answerable" in text and "UTC" in text


# ------------------------------------------------------------------ integration with the loop
def test_planner_off_by_default_makes_no_planner_call(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry)
    assert llm.plan_requests == [] and run.plan_status is None and run.use_planner is False


def test_valid_plan_is_given_to_the_agent(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()])
    run = run_agent("mean 2023?", llm, registry, use_planner=True)
    first = llm.requests[0]["input"][0]["content"]
    assert first.startswith("mean 2023?") and PLAN_INTRO in first and "[run_python]" in first
    assert run.plan_status is PlanStatus.OK and run.stop_reason is StopReason.ANSWERED


def test_planner_sees_the_real_dataset_description(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()])
    run_agent("q", llm, registry, use_planner=True)
    assert '"variable_name": "df"' in llm.plan_requests[0]["instructions"]


def test_planner_description_is_not_counted_as_a_model_tool_call(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()])
    run = run_agent("q", llm, registry, use_planner=True)
    assert [r.tool for r in run.tool_calls] == ["submit_answer"]


def test_invalid_plan_is_recorded_and_agent_continues_without_plan(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json(steps=[])])
    run = run_agent("q", llm, registry, use_planner=True)
    assert run.plan_status is PlanStatus.RULE_VIOLATION and run.plan is None and run.plan_error
    assert llm.requests[0]["input"][0]["content"] == "q"  # no plan text, not silently repaired
    assert run.stop_reason is StopReason.ANSWERED


def test_planner_llm_error_does_not_stop_the_run(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[ConnectionError("down")])
    run = run_agent("q", llm, registry, use_planner=True)
    assert run.plan_status is PlanStatus.LLM_ERROR and run.stop_reason is StopReason.ANSWERED


def test_missing_dataset_is_a_setup_error(tmp_path):
    reg = ToolRegistry.default(tmp_path / "nope", FakeSandbox())  # type: ignore[arg-type]
    llm = ScriptedLLM([call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, reg, use_planner=True)
    assert run.plan_status is PlanStatus.SETUP_ERROR and llm.plan_requests == []


def test_planner_tokens_counted_separately_and_in_totals(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()])
    run = run_agent("q", llm, registry, use_planner=True)
    assert [c.purpose for c in run.llm_calls] == ["planner", "agent"]
    assert run.input_tokens == 50 + 100 and run.turns == 1  # the planner call is not a loop turn


def test_run_record_with_plan_is_json_serializable(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()])
    data = json.loads(json.dumps(run_agent("q", llm, registry, use_planner=True).to_dict()))
    assert data["plan_status"] == "ok" and data["plan"]["steps"][0]["tool"] == "describe_dataset"
    assert data["llm_calls"][0]["purpose"] == "planner"


def test_format_run_shows_plan_outcome(registry):
    ok = run_agent(
        "q", ScriptedLLM([call("submit_answer", SUBMIT)], plans=[plan_json()]), registry, use_planner=True
    )
    bad = run_agent(
        "q", ScriptedLLM([call("submit_answer", SUBMIT)], plans=["nope"]), registry, use_planner=True
    )
    assert "plan:     ok, 3 steps" in format_run(ok)
    assert "plan:     invalid_json (continued without a plan)" in format_run(bad)


# ------------------------------------------------------------------ OpenAI adapter
def test_openai_structured_call_sends_strict_json_schema():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        usage = SimpleNamespace(
            input_tokens=80, output_tokens=40, output_tokens_details=None, input_tokens_details=None
        )
        return SimpleNamespace(id="resp_p", output=[], output_text=plan_json(), usage=usage)

    llm = OpenAIClient(model="m", client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    r = llm.create_structured(instructions="sys", input=[], schema_name="plan", schema={"type": "object"})
    fmt = sent["text"]["format"]
    assert fmt == {"type": "json_schema", "name": "plan", "schema": {"type": "object"}, "strict": True}
    assert "tools" not in sent
    assert json.loads(r.text)["answerable"] is True
    assert r.record.purpose == "planner" and r.record.input_tokens == 80
