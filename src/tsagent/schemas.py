"""Argument models for the three tools.

Each model plays two roles:
  1. Its JSON schema (reduced to the strict-mode subset, see tools.to_strict_schema)
     is what the LLM sees as the function definition.
  2. The model itself validates the arguments the LLM sends back.

Constraints expressed with Field(...) and validators live ONLY in Pydantic and
are not sent to the provider. They are the "rule level" checks whose failures the
evaluation counts. The provider's strict mode only guarantees the "schema level"
(fields present, correct JSON types).
"""

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CODE_CHARS = 20_000


class _Args(BaseModel):
    # extra="forbid" produces additionalProperties: false, required by strict mode.
    model_config = ConfigDict(extra="forbid")


class DescribeDatasetArgs(_Args):
    """Describe the dataset: columns, units, time range and summary statistics.
    Call this first. It takes no arguments."""


class RunPythonArgs(_Args):
    """Run Python code in an isolated sandbox and return its `result`.

    Every call starts a fresh process: variables do NOT persist between calls, so
    each call must be self-contained. The hourly dataset is preloaded as a pandas
    DataFrame named `df` with a UTC DatetimeIndex. Assign your final value to a
    variable named `result`. Allowed imports: pandas, numpy, math, statistics,
    datetime. No network, no file writes, 10 second time limit."""

    code: str = Field(
        description="Self-contained Python code. Must assign the answer to `result`.",
        min_length=1,
        max_length=MAX_CODE_CHARS,
    )

    @field_validator("code")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("code is empty or whitespace only")
        return v


class SubmitAnswerArgs(_Args):
    """Submit the final answer. Call this exactly once, when you are confident.
    This ends the task."""

    value: float | str = Field(
        description="The answer: a number, or a short text such as a date ('2023-07') when the "
        "question asks for one."
    )
    unit: str | None = Field(
        description="Unit of the value, e.g. '°C', 'mm', 'km/h', 'hPa', '%', 'hours'. "
        "null if the value has no unit (e.g. a date or a count)."
    )
    method_summary: str = Field(
        description="One or two sentences: how the value was computed.",
        min_length=1,
        max_length=500,
    )

    @field_validator("value")
    @classmethod
    def finite_or_nonempty(cls, v: float | str) -> float | str:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("value must be a finite number, not NaN or infinity")
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("value is an empty string")
            if len(v) > 100:
                raise ValueError("text value is longer than 100 characters; give a short answer")
        return v

    @field_validator("unit")
    @classmethod
    def unit_not_blank(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("unit is an empty string; use null for values without a unit")
        return v.strip() if v is not None else None


# ------------------------------------------------------------------ planner (step 4)
ToolName = Literal["describe_dataset", "run_python", "submit_answer"]


class PlanStep(_Args):
    tool: ToolName = Field(description="The tool this step will use.")
    description: str = Field(
        description="What this step does, concretely (columns, period, aggregation).",
        min_length=1,
        max_length=300,
    )


class Plan(_Args):
    """A plan for answering one question about the dataset, written before any code runs."""

    interpretation: str = Field(
        description="The question restated precisely: quantity, period, aggregation, unit.",
        min_length=1,
        max_length=500,
    )
    assumptions: list[str] = Field(
        description="Choices made where the question is ambiguous, e.g. 'year 2023 in UTC', "
        "'mean of hourly values'. Empty list if there are none.",
        max_length=5,
    )
    answerable: bool = Field(description="false if the dataset cannot answer this question.")
    expected_unit: str | None = Field(description="Unit of the final answer, or null if it has none.")
    steps: list[PlanStep] = Field(
        description="Ordered steps. The last step must be submit_answer.",
        min_length=1,
        max_length=8,
    )

    @field_validator("assumptions")
    @classmethod
    def assumptions_short(cls, v: list[str]) -> list[str]:
        for a in v:
            if not a.strip() or len(a) > 200:
                raise ValueError("each assumption must be 1 to 200 characters")
        return v

    @field_validator("steps")
    @classmethod
    def ends_with_single_submit(cls, v: list[PlanStep]) -> list[PlanStep]:
        submits = [i for i, s in enumerate(v) if s.tool == "submit_answer"]
        if submits != [len(v) - 1]:
            raise ValueError("the plan must contain exactly one submit_answer step, and it must be last")
        return v
