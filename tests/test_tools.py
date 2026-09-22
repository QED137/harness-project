"""Tools: strict schemas, two-level validation, execution and call records.
The sandbox is replaced by a fake, so no Docker is needed."""

import json
import re

import pytest

from tsagent import dataset as ds
from tsagent.sandbox import DEFAULT_ALLOWED_MODULES, SandboxConfig, SandboxResult, SandboxStatus
from tsagent.tools import MAX_OUTPUT_CHARS, STRICT_KEYWORDS, ToolCallStatus, ToolRegistry, to_strict_schema

from .test_dataset import REQUEST, fake_payload


class FakeSandbox:
    def __init__(self, result: SandboxResult | None = None):
        self.codes: list[str] = []
        self.result = result or SandboxResult(
            status=SandboxStatus.OK, result={"type": "number", "value": 9.5}, total_s=0.1
        )

    def run(self, code: str) -> SandboxResult:
        self.codes.append(code)
        return self.result


@pytest.fixture
def data_dir(tmp_path):
    out = tmp_path / "data"
    ds.save(ds.to_dataframe(fake_payload()), fake_payload(), out, REQUEST)
    return out


@pytest.fixture
def sandbox():
    return FakeSandbox()


@pytest.fixture
def registry(data_dir, sandbox):
    return ToolRegistry.default(data_dir, sandbox)  # type: ignore[arg-type]


# ------------------------------------------------------------------ definitions
def _walk(node, path="parameters"):
    yield path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_three_tools_with_strict_definitions(registry):
    tools = registry.openai_tools()
    assert [t["name"] for t in tools] == ["describe_dataset", "run_python", "submit_answer"]
    for t in tools:
        assert t["type"] == "function" and t["strict"] is True and t["description"]


def test_every_object_meets_strict_mode_requirements(registry):
    for t in registry.openai_tools():
        for path, node in _walk(t["parameters"]):
            if isinstance(node, dict) and node.get("type") == "object":
                assert node["additionalProperties"] is False, path
                assert sorted(node["required"]) == sorted(node["properties"]), path


def test_only_supported_keywords_are_sent(registry):
    for t in registry.openai_tools():
        for path, node in _walk(t["parameters"]):
            if isinstance(node, dict) and not path.endswith(".properties"):
                assert set(node) <= STRICT_KEYWORDS, (path, set(node) - STRICT_KEYWORDS)


def test_nested_models_are_rejected_not_mishandled():
    with pytest.raises(ValueError, match="nested"):
        to_strict_schema({"type": "object", "$defs": {"X": {}}, "properties": {}})


def test_run_python_description_matches_real_sandbox_limits(registry):
    """The description is written by hand; this catches drift from the real config."""
    desc = registry.tools["run_python"].description
    for module in DEFAULT_ALLOWED_MODULES:
        assert module in desc, module
    seconds = int(SandboxConfig().timeout_s)
    assert re.search(rf"\b{seconds} second", desc)


# ------------------------------------------------------------------ validation levels
@pytest.mark.parametrize(
    "tool, raw, expected",
    [
        # describe_dataset
        ("describe_dataset", "{}", ToolCallStatus.OK),
        ("describe_dataset", "", ToolCallStatus.OK),
        ("describe_dataset", '{"x": 1}', ToolCallStatus.SCHEMA_VIOLATION),
        # run_python
        ("run_python", '{"code": "result = 1"}', ToolCallStatus.OK),
        ("run_python", "{}", ToolCallStatus.SCHEMA_VIOLATION),
        ("run_python", '{"code": 5}', ToolCallStatus.SCHEMA_VIOLATION),
        ("run_python", '{"code": "x", "extra": 1}', ToolCallStatus.SCHEMA_VIOLATION),
        ("run_python", '{"code": ""}', ToolCallStatus.RULE_VIOLATION),
        ("run_python", '{"code": "   \\n  "}', ToolCallStatus.RULE_VIOLATION),
        ("run_python", json.dumps({"code": "x" * 20_001}), ToolCallStatus.RULE_VIOLATION),
        ("run_python", "not json", ToolCallStatus.INVALID_JSON),
        ("run_python", '{"code": "result = 1"', ToolCallStatus.INVALID_JSON),
        # submit_answer
        ("submit_answer", '{"value": 12.5, "unit": "°C", "method_summary": "mean"}', ToolCallStatus.OK),
        (
            "submit_answer",
            '{"value": "2023-07", "unit": null, "method_summary": "idxmax"}',
            ToolCallStatus.OK,
        ),
        ("submit_answer", '{"value": 3, "unit": null, "method_summary": "count"}', ToolCallStatus.OK),
        ("submit_answer", '{"value": 1, "method_summary": "m"}', ToolCallStatus.SCHEMA_VIOLATION),
        (
            "submit_answer",
            '{"value": [1], "unit": null, "method_summary": "m"}',
            ToolCallStatus.SCHEMA_VIOLATION,
        ),
        (
            "submit_answer",
            '{"value": NaN, "unit": "°C", "method_summary": "m"}',
            ToolCallStatus.RULE_VIOLATION,
        ),
        (
            "submit_answer",
            '{"value": Infinity, "unit": "°C", "method_summary": "m"}',
            ToolCallStatus.RULE_VIOLATION,
        ),
        (
            "submit_answer",
            '{"value": " ", "unit": null, "method_summary": "m"}',
            ToolCallStatus.RULE_VIOLATION,
        ),
        ("submit_answer", '{"value": 1, "unit": " ", "method_summary": "m"}', ToolCallStatus.RULE_VIOLATION),
        (
            "submit_answer",
            json.dumps({"value": 1, "unit": None, "method_summary": "m" * 501}),
            ToolCallStatus.RULE_VIOLATION,
        ),
        # unknown
        ("delete_everything", "{}", ToolCallStatus.UNKNOWN_TOOL),
    ],
)
def test_every_call_ends_in_exactly_one_classified_status(registry, tool, raw, expected):
    outcome = registry.call(tool, raw)
    assert outcome.status is expected, outcome.output
    assert len(registry.records) == 1 and registry.records[0].status is expected
    if expected is not ToolCallStatus.OK:
        assert outcome.output.startswith("Error:")  # the model is told, nothing is raised


def test_error_message_names_the_field(registry):
    out = registry.call("submit_answer", '{"value": NaN, "unit": "°C", "method_summary": "m"}').output
    assert "value" in out and "finite" in out


def test_invalid_run_python_never_reaches_the_sandbox(registry, sandbox):
    registry.call("run_python", '{"code": ""}')
    registry.call("run_python", "not json")
    assert sandbox.codes == []


# ------------------------------------------------------------------ execution
def test_describe_dataset_reports_units_and_utc(registry):
    info = json.loads(registry.call("describe_dataset", "{}").output)
    assert info["variable_name"] == "df" and info["rows"] == 48
    assert "UTC" in info["index"]
    assert set(info["columns"]) == set(ds.VARIABLES)
    assert info["columns"]["temperature_2m"]["unit"] == "unit"


def test_run_python_passes_code_and_keeps_full_sandbox_result(registry, sandbox):
    outcome = registry.call("run_python", '{"code": "result = 9.5"}')
    assert sandbox.codes == ["result = 9.5"]
    assert json.loads(outcome.output)["result"]["value"] == 9.5
    assert outcome.record.sandbox is sandbox.result  # full details kept for the evaluation


def test_failed_code_is_ok_call_with_error_for_model(data_dir):
    """A tool call can be valid (status OK) while the code inside it fails."""
    failing = FakeSandbox(SandboxResult(status=SandboxStatus.ERROR, error="ZeroDivisionError", total_s=0.1))
    reg = ToolRegistry.default(data_dir, failing)  # type: ignore[arg-type]
    outcome = reg.call("run_python", '{"code": "1/0"}')
    assert outcome.status is ToolCallStatus.OK
    assert json.loads(outcome.output) == {"status": "error", "error": "ZeroDivisionError"}


def test_missing_result_is_explained_to_the_model(data_dir):
    sb = FakeSandbox(SandboxResult(status=SandboxStatus.OK, result_missing=True, total_s=0.1))
    out = ToolRegistry.default(data_dir, sb).call("run_python", '{"code": "x = 1"}').output  # type: ignore[arg-type]
    assert "result" in json.loads(out)["note"]


def test_long_output_is_truncated(data_dir):
    sb = FakeSandbox(SandboxResult(status=SandboxStatus.OK, stdout="x" * 50_000, total_s=0.1))
    out = ToolRegistry.default(data_dir, sb).call("run_python", '{"code": "print(1)"}').output  # type: ignore[arg-type]
    assert len(out) < MAX_OUTPUT_CHARS + 100 and "truncated" in out


def test_submit_answer_is_terminal_and_returns_validated_answer(registry):
    outcome = registry.call("submit_answer", '{"value": 12.5, "unit": " °C ", "method_summary": "mean"}')
    assert outcome.terminal is True
    assert outcome.answer.value == 12.5 and outcome.answer.unit == "°C"


def test_non_terminal_tools(registry):
    assert registry.call("describe_dataset", "{}").terminal is False
    assert registry.call("run_python", '{"code": "result = 1"}').terminal is False


def test_bug_in_tool_code_is_recorded_not_raised(tmp_path, sandbox):
    reg = ToolRegistry.default(tmp_path / "does-not-exist", sandbox)  # type: ignore[arg-type]
    outcome = reg.call("describe_dataset", "{}")
    assert outcome.status is ToolCallStatus.EXECUTION_ERROR
    assert "FileNotFoundError" in outcome.record.error


def test_counts_summarize_all_calls(registry):
    registry.call("describe_dataset", "{}")
    registry.call("run_python", "{}")
    registry.call("run_python", '{"code": ""}')
    registry.call("run_python", '{"code": "result = 1"}')
    assert registry.counts() == {"ok": 2, "schema_violation": 1, "rule_violation": 1}
    assert all(r.duration_s >= 0 for r in registry.records)
