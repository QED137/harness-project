"""The agent loop with a scripted fake model. No network, no API key, no cost."""

import json
from types import SimpleNamespace

import pytest

from tsagent import agent as agent_mod
from tsagent.agent import SYSTEM_PROMPT, StopReason, run_agent
from tsagent.llm import (
    FunctionCall,
    LLMCallRecord,
    LLMResponse,
    OpenAIClient,
    StructuredResponse,
    load_dotenv,
    openai_config_problem,
)
from tsagent.sandbox import SandboxResult, SandboxStatus
from tsagent.tools import ToolCallStatus, ToolRegistry

from .test_tools import FakeSandbox, data_dir  # noqa: F401  (data_dir is a fixture)

SUBMIT = '{"value": 9.5, "unit": "°C", "method_summary": "mean of temperature_2m"}'


def call(name: str, arguments: str) -> tuple[str, str]:
    return (name, arguments)


class ScriptedLLM:
    """Returns one scripted step per call: a (tool, arguments) tuple, a text string,
    or an Exception to raise. Records what it was sent.
    `plans` scripts the planner calls the same way: a JSON string or an Exception."""

    model = "fake-model"

    def __init__(self, steps, plans=()):
        self.steps = list(steps)
        self.plans = list(plans)
        self.requests: list[dict] = []
        self.plan_requests: list[dict] = []

    def create_structured(self, *, instructions, input, schema_name, schema):
        self.plan_requests.append(
            {"instructions": instructions, "input": input, "schema_name": schema_name, "schema": schema}
        )
        step = self.plans.pop(0)
        if isinstance(step, Exception):
            raise step
        record = LLMCallRecord(
            model=self.model, latency_s=0.01, purpose="planner", input_tokens=50, output_tokens=10
        )
        return StructuredResponse(step, record)

    def create(self, *, instructions, input, tools):
        self.requests.append({"instructions": instructions, "input": list(input), "tools": tools})
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        record = LLMCallRecord(model=self.model, latency_s=0.01, input_tokens=100, output_tokens=20)
        if isinstance(step, str):
            return LLMResponse([{"type": "message", "text": step}], [], step, record)
        name, arguments = step
        call_id = f"call_{len(self.requests)}"
        item = {"type": "function_call", "name": name, "arguments": arguments, "call_id": call_id}
        return LLMResponse([item], [FunctionCall(name, arguments, call_id)], "", record)


@pytest.fixture
def registry(data_dir):  # noqa: F811
    return ToolRegistry.default(data_dir, FakeSandbox())  # type: ignore[arg-type]


# ------------------------------------------------------------------ loop behaviour
def test_happy_path(registry):
    llm = ScriptedLLM(
        [
            call("describe_dataset", "{}"),
            call("run_python", '{"code": "result = df.temperature_2m.mean()"}'),
            call("submit_answer", SUBMIT),
        ]
    )
    run = run_agent("mean temperature?", llm, registry)
    assert run.stop_reason is StopReason.ANSWERED
    assert run.answer.value == 9.5 and run.answer.unit == "°C"
    assert run.turns == 3 and len(run.llm_calls) == 3
    assert [r.tool for r in run.tool_calls] == ["describe_dataset", "run_python", "submit_answer"]
    assert run.input_tokens == 300 and run.output_tokens == 60


def test_every_request_has_prompt_and_tools(registry):
    llm = ScriptedLLM([call("submit_answer", SUBMIT)])
    run_agent("q", llm, registry)
    req = llm.requests[0]
    assert req["instructions"] == SYSTEM_PROMPT
    assert [t["name"] for t in req["tools"]] == ["describe_dataset", "run_python", "submit_answer"]
    assert req["input"] == [{"role": "user", "content": "q"}]


def test_tool_output_goes_back_with_matching_call_id(registry):
    llm = ScriptedLLM([call("describe_dataset", "{}"), call("submit_answer", SUBMIT)])
    run_agent("q", llm, registry)
    second_input = llm.requests[1]["input"]
    model_item, output_item = second_input[1], second_input[2]
    assert model_item["type"] == "function_call"  # the model's own output is passed back
    assert output_item["type"] == "function_call_output"
    assert output_item["call_id"] == model_item["call_id"]
    assert json.loads(output_item["output"])["variable_name"] == "df"


def test_invalid_arguments_are_reported_and_model_can_recover(registry):
    llm = ScriptedLLM(
        [
            call("run_python", '{"code": ""}'),
            call("submit_answer", '{"value": NaN, "unit": "°C", "method_summary": "m"}'),
            call("submit_answer", SUBMIT),
        ]
    )
    run = run_agent("q", llm, registry)
    assert run.stop_reason is StopReason.ANSWERED
    assert [r.status for r in run.tool_calls] == [
        ToolCallStatus.RULE_VIOLATION,
        ToolCallStatus.RULE_VIOLATION,
        ToolCallStatus.OK,
    ]
    error_sent = llm.requests[1]["input"][-1]["output"]
    assert error_sent.startswith("Error:")


def test_invalid_submit_is_not_terminal(registry):
    llm = ScriptedLLM([call("submit_answer", '{"value": 1}'), call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry)
    assert run.turns == 2 and run.stop_reason is StopReason.ANSWERED


def test_text_reply_gets_one_nudge_then_can_still_answer(registry):
    llm = ScriptedLLM(["The answer is about 10 degrees.", call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry)
    assert run.nudges == 1 and run.stop_reason is StopReason.ANSWERED
    assert "submit_answer" in llm.requests[1]["input"][-1]["content"]


def test_repeated_text_replies_stop_without_answer(registry):
    llm = ScriptedLLM(["about 10 degrees", "still about 10 degrees"])
    run = run_agent("q", llm, registry)
    assert run.stop_reason is StopReason.NO_SUBMIT and run.answer is None
    assert run.final_text == "still about 10 degrees"


def test_turn_budget(registry):
    llm = ScriptedLLM([call("describe_dataset", "{}")] * 5)
    run = run_agent("q", llm, registry, max_turns=3)
    assert run.stop_reason is StopReason.MAX_TURNS and run.turns == 3 and run.answer is None


def test_llm_error_is_recorded_not_raised(registry):
    llm = ScriptedLLM([call("describe_dataset", "{}"), ConnectionError("network down")])
    run = run_agent("q", llm, registry)
    assert run.stop_reason is StopReason.LLM_ERROR and "network down" in run.error
    assert len(run.tool_calls) == 1  # work done before the error is still recorded


def test_unknown_tool_does_not_stop_the_run(registry):
    llm = ScriptedLLM([call("rm_rf", "{}"), call("submit_answer", SUBMIT)])
    run = run_agent("q", llm, registry)
    assert run.tool_calls[0].status is ToolCallStatus.UNKNOWN_TOOL
    assert run.stop_reason is StopReason.ANSWERED


def test_run_records_only_its_own_tool_calls(registry):
    run_agent("first", ScriptedLLM([call("submit_answer", SUBMIT)]), registry)
    second = run_agent(
        "second", ScriptedLLM([call("describe_dataset", "{}"), call("submit_answer", SUBMIT)]), registry
    )
    assert len(second.tool_calls) == 2


def test_run_record_is_json_serializable(data_dir):  # noqa: F811
    sb = FakeSandbox(
        SandboxResult(status=SandboxStatus.OK, result={"type": "number", "value": 1.0}, total_s=0.1)
    )
    reg = ToolRegistry.default(data_dir, sb)  # type: ignore[arg-type]
    run = run_agent(
        "q", ScriptedLLM([call("run_python", '{"code": "result = 1"}'), call("submit_answer", SUBMIT)]), reg
    )
    data = json.loads(json.dumps(run.to_dict()))
    assert data["stop_reason"] == "answered" and data["answer"]["value"] == 9.5
    assert data["tool_calls"][0]["sandbox"]["status"] == "ok"
    assert data["input_tokens"] == 200


def test_format_run_mentions_answer_and_stats(registry):
    run = run_agent("q", ScriptedLLM([call("submit_answer", SUBMIT)]), registry)
    text = agent_mod.format_run(run)
    assert "9.5 °C" in text and "1 turns" in text


# ------------------------------------------------------------------ OpenAI adapter
def fake_openai_response():
    usage = SimpleNamespace(
        input_tokens=120,
        output_tokens=30,
        output_tokens_details=SimpleNamespace(reasoning_tokens=12),
        input_tokens_details=SimpleNamespace(cached_tokens=64),
    )
    output = [
        SimpleNamespace(type="reasoning", id="rs_1"),
        SimpleNamespace(type="function_call", name="describe_dataset", arguments="{}", call_id="call_abc"),
    ]
    return SimpleNamespace(id="resp_1", output=output, output_text="", usage=usage)


def test_openai_adapter_sends_expected_request_and_parses_response():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return fake_openai_response()

    fake_sdk = SimpleNamespace(responses=SimpleNamespace(create=create))
    llm = OpenAIClient(model="some-model", client=fake_sdk)
    r = llm.create(instructions="sys", input=[{"role": "user", "content": "q"}], tools=[{"name": "t"}])

    assert sent["model"] == "some-model" and sent["instructions"] == "sys"
    assert sent["parallel_tool_calls"] is False and sent["tools"] == [{"name": "t"}]
    assert r.function_calls == [FunctionCall("describe_dataset", "{}", "call_abc")]
    assert len(r.output_items) == 2  # reasoning item kept, so it can be passed back
    assert (r.record.input_tokens, r.record.output_tokens) == (120, 30)
    assert (r.record.reasoning_tokens, r.record.cached_tokens) == (12, 64)
    assert r.record.response_id == "resp_1"


# ------------------------------------------------------------------ configuration
def test_dotenv_reads_values_without_overriding(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nOPENAI_MODEL="model-from-file"\nOPENAI_API_KEY=sk-file\nEMPTY=\n')
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["OPENAI_MODEL"] == "model-from-file"
    assert os.environ["OPENAI_API_KEY"] == "sk-already-set"
    assert "EMPTY" not in os.environ


def test_missing_config_is_explained(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    problem = openai_config_problem()
    assert "OPENAI_API_KEY" in problem and "OPENAI_MODEL" in problem and ".env" in problem


def test_main_without_config_exits_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert agent_mod.main(["q"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
