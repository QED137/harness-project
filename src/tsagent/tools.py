"""The three tools, their definitions for the LLM, and precise accounting of every call.

    registry = ToolRegistry.default(data_dir=Path("data"), sandbox=DockerSandbox(...))
    registry.openai_tools()                    # definitions for the Responses API
    outcome = registry.call("run_python", '{"code": "result = df.temperature_2m.mean()"}')
    registry.records                           # one ToolCallRecord per call, for the evaluation

Every call ends in exactly one ToolCallStatus. Validation failures are split by level:
  INVALID_JSON         arguments are not valid JSON
  SCHEMA_VIOLATION     wrong fields or JSON types (what provider strict mode should prevent)
  RULE_VIOLATION       Pydantic-only rules: empty code, NaN answer, length limits
so the evaluation can report both separately.
"""

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from .dataset import MANIFEST_NAME, PARQUET_NAME
from .sandbox import DockerSandbox, SandboxResult
from .schemas import DescribeDatasetArgs, RunPythonArgs, SubmitAnswerArgs

MAX_OUTPUT_CHARS = 4_000  # what one tool result may add to the model's context

# JSON Schema keywords that are sent to the provider. Everything else (minLength,
# maxLength, title, default, ...) is removed and enforced only by Pydantic.
STRICT_KEYWORDS = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
    "enum",
    "anyOf",
    "items",
}

# Pydantic error types that the JSON schema sent to the provider also expresses.
_SCHEMA_ERROR_TYPES = {"missing", "extra_forbidden", "model_type", "dict_type"}
_SCHEMA_ERROR_PREFIXES = (
    "string_type",
    "float_type",
    "float_parsing",
    "int_type",
    "bool_type",
    "none_required",
)


class ToolCallStatus(StrEnum):
    OK = "ok"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_JSON = "invalid_json"
    SCHEMA_VIOLATION = "schema_violation"
    RULE_VIOLATION = "rule_violation"
    EXECUTION_ERROR = "execution_error"  # our tool code raised: a bug on our side, not the model's


@dataclass
class ToolCallRecord:
    tool: str
    status: ToolCallStatus
    raw_arguments: str
    duration_s: float
    error: str | None = None
    sandbox: SandboxResult | None = None


@dataclass
class ToolOutcome:
    status: ToolCallStatus
    output: str  # exactly what is sent back to the model
    terminal: bool = False  # True after a valid submit_answer
    answer: SubmitAnswerArgs | None = None
    record: ToolCallRecord | None = None


@dataclass
class ToolResult:
    output: str
    sandbox: SandboxResult | None = None  # kept in the call record, not sent to the model


@dataclass
class Tool:
    name: str
    args_model: type[BaseModel]
    run: Callable[[Any], ToolResult]
    terminal: bool = False

    @property
    def description(self) -> str:
        return " ".join((self.args_model.__doc__ or "").split())

    def parameters_schema(self) -> dict[str, Any]:
        return to_strict_schema(self.args_model.model_json_schema())

    def openai_tool(self) -> dict[str, Any]:
        """Tool definition in the OpenAI Responses API format, strict mode on."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema(),
            "strict": True,
        }


# ------------------------------------------------------------------ schema conversion
def to_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a Pydantic JSON schema to the subset required by provider strict mode:
    every object has additionalProperties false and lists all properties as required;
    only STRICT_KEYWORDS remain. Our argument models are flat, so $defs/$ref are not
    supported and rejected loudly rather than silently mishandled."""
    if "$defs" in schema or "$ref" in json.dumps(schema):
        raise ValueError("nested models ($defs/$ref) are not supported by to_strict_schema")

    def clean(node: Any) -> Any:
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        out = {k: clean(v) for k, v in node.items() if k in STRICT_KEYWORDS}
        if "properties" in node:  # properties is a name->schema map, clean each schema
            out["properties"] = {name: clean(sub) for name, sub in node["properties"].items()}
        if node.get("type") == "object":
            out.setdefault("properties", {})
            out["required"] = list(out["properties"])
            out["additionalProperties"] = False
        return out

    result = clean(copy.deepcopy(schema))
    result.pop("description", None)  # the model docstring is already the tool description
    return result


def _classify_validation_error(e: ValidationError) -> ToolCallStatus:
    for err in e.errors():
        t = err["type"]
        if t in _SCHEMA_ERROR_TYPES or t.startswith(_SCHEMA_ERROR_PREFIXES):
            return ToolCallStatus.SCHEMA_VIOLATION
    return ToolCallStatus.RULE_VIOLATION


def _format_validation_error(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        where = ".".join(str(p) for p in err["loc"]) or "(arguments)"
        parts.append(f"{where}: {err['msg']}")
    return "; ".join(parts)


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more characters]"


# ------------------------------------------------------------------ registry
class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}
        self.records: list[ToolCallRecord] = []

    @classmethod
    def default(cls, data_dir: Path, sandbox: DockerSandbox) -> "ToolRegistry":
        return cls(
            [
                Tool("describe_dataset", DescribeDatasetArgs, lambda args: describe_dataset(data_dir)),
                Tool("run_python", RunPythonArgs, lambda args: run_python(sandbox, args)),
                Tool("submit_answer", SubmitAnswerArgs, submit_answer, terminal=True),
            ]
        )

    def openai_tools(self) -> list[dict[str, Any]]:
        return [t.openai_tool() for t in self.tools.values()]

    def call(self, name: str, raw_arguments: str) -> ToolOutcome:
        start = time.monotonic()

        def finish(
            status: ToolCallStatus,
            output: str,
            error: str | None = None,
            sandbox: SandboxResult | None = None,
            **kw: Any,
        ) -> ToolOutcome:
            record = ToolCallRecord(name, status, raw_arguments, time.monotonic() - start, error, sandbox)
            self.records.append(record)
            return ToolOutcome(status, _truncate(output), record=record, **kw)

        tool = self.tools.get(name)
        if tool is None:
            msg = f"unknown tool '{name}'; available tools: {sorted(self.tools)}"
            return finish(ToolCallStatus.UNKNOWN_TOOL, f"Error: {msg}", msg)

        try:
            data = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as e:
            msg = f"arguments are not valid JSON: {e.msg} at position {e.pos}"
            return finish(ToolCallStatus.INVALID_JSON, f"Error: {msg}", msg)

        try:
            args = tool.args_model.model_validate(data)
        except ValidationError as e:
            msg = _format_validation_error(e)
            return finish(_classify_validation_error(e), f"Error: invalid arguments for {name}: {msg}", msg)

        try:
            result = tool.run(args)
        except Exception as e:  # noqa: BLE001  a bug in OUR tool code; recorded, never raised
            msg = f"{type(e).__name__}: {e}"
            return finish(ToolCallStatus.EXECUTION_ERROR, f"Error: tool {name} failed internally: {msg}", msg)

        answer = args if isinstance(args, SubmitAnswerArgs) else None
        return finish(
            ToolCallStatus.OK, result.output, sandbox=result.sandbox, terminal=tool.terminal, answer=answer
        )

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.status.value] = out.get(r.status.value, 0) + 1
        return out


# ------------------------------------------------------------------ tool implementations
def describe_dataset(data_dir: Path) -> ToolResult:
    manifest = json.loads((data_dir / MANIFEST_NAME).read_text())
    df = pd.read_parquet(data_dir / PARQUET_NAME)
    columns = {}
    for col in df.columns:
        s = df[col]
        columns[col] = {
            "unit": manifest["variables"].get(col),
            "min": round(float(s.min()), 3),
            "mean": round(float(s.mean()), 3),
            "max": round(float(s.max()), 3),
            "missing": int(s.isna().sum()),
        }
    info = {
        "variable_name": "df",
        "rows": len(df),
        "frequency": "hourly",
        "index": "time, pandas DatetimeIndex in UTC (not local time)",
        "time_range_utc": manifest["time_range"],
        "location": manifest["grid_cell"],
        "source": manifest["source"],
        "columns": columns,
    }
    return ToolResult(json.dumps(info, ensure_ascii=False, indent=1))


def run_python(sandbox: DockerSandbox, args: RunPythonArgs) -> ToolResult:
    r = sandbox.run(args.code)
    return ToolResult(format_sandbox_result(r), sandbox=r)


def format_sandbox_result(r: SandboxResult) -> str:
    """What the model sees: enough to fix its code, nothing it does not need."""
    out: dict[str, Any] = {"status": r.status.value}
    if r.result is not None:
        out["result"] = r.result
    if r.result_missing:
        out["note"] = "the code did not assign a variable named `result`"
    if r.stdout:
        out["stdout"] = r.stdout
    if r.error:
        out["error"] = r.error
    if r.violations:
        out["violations"] = [f"{v.kind.value}: {v.detail}" for v in r.violations]
    if r.limits_hit:
        out["limits_hit"] = [lim.value for lim in r.limits_hit]
    return json.dumps(out, ensure_ascii=False)


def submit_answer(args: SubmitAnswerArgs) -> ToolResult:
    return ToolResult("Answer recorded. The task is complete.")
