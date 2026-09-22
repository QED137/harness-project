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
