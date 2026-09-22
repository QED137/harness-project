"""Step 4: the planner.

One LLM call before the agent loop. The model sees the question and the dataset
description and returns a Plan as JSON (provider structured output, strict mode),
which is then validated with Pydantic.

Every planning attempt ends in exactly one PlanStatus. An invalid plan is recorded
and the agent continues WITHOUT a plan: it is counted, never silently repaired.
"""

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ValidationError

from .llm import LLMCallRecord, LLMClient
from .schemas import Plan
from .tools import ToolCallStatus, classify_validation_error, format_validation_error, to_strict_schema

PLANNER_PROMPT = """\
You plan how to answer one quantitative question about an hourly weather dataset.
You do not run code yourself; another agent will follow your plan using these tools:
- describe_dataset: columns, units, time range
- run_python: runs pandas code on the dataset (preloaded as `df`, UTC index); each call is independent
- submit_answer: submits the final value and unit (always the last step)

Write a short, concrete plan:
- interpretation: restate the question precisely (quantity, period, aggregation, unit)
- assumptions: the choices you make where the question is ambiguous
- steps: 2 to 5 steps, each naming its tool; the last step is submit_answer
- expected_unit: the unit of the final answer, converting if the question asks for a different unit
  than the data has
- answerable: false if the dataset cannot answer the question

The dataset description follows.
"""

SCHEMA_NAME = "plan"


class PlanStatus(StrEnum):
    OK = "ok"
    INVALID_JSON = "invalid_json"
    SCHEMA_VIOLATION = "schema_violation"  # what provider strict mode should prevent
    RULE_VIOLATION = "rule_violation"  # Pydantic-only rules: step count, submit last, lengths
    LLM_ERROR = "llm_error"
    SETUP_ERROR = "setup_error"  # our side failed before the model was asked (e.g. no dataset)


@dataclass
class PlanResult:
    status: PlanStatus
    plan: Plan | None = None
    raw: str = ""  # exactly what the model returned
    error: str | None = None
    record: LLMCallRecord | None = None


def plan_schema() -> dict:
    return to_strict_schema(Plan.model_json_schema())


def make_plan(question: str, llm: LLMClient, dataset_description: str) -> PlanResult:
    try:
        response = llm.create_structured(
            instructions=PLANNER_PROMPT + "\n" + dataset_description,
            input=[{"role": "user", "content": question}],
            schema_name=SCHEMA_NAME,
            schema=plan_schema(),
        )
    except Exception as e:  # noqa: BLE001  recorded, not raised
        return PlanResult(PlanStatus.LLM_ERROR, error=f"{type(e).__name__}: {e}")

    raw, record = response.text, response.record
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return PlanResult(
            PlanStatus.INVALID_JSON, raw=raw, error=f"{e.msg} at position {e.pos}", record=record
        )

    try:
        plan = Plan.model_validate(data)
    except ValidationError as e:
        level = classify_validation_error(e)
        status = (
            PlanStatus.SCHEMA_VIOLATION
            if level is ToolCallStatus.SCHEMA_VIOLATION
            else PlanStatus.RULE_VIOLATION
        )
        return PlanResult(status, raw=raw, error=format_validation_error(e), record=record)

    return PlanResult(PlanStatus.OK, plan=plan, raw=raw, record=record)


def format_plan(plan: Plan) -> str:
    """The plan as the agent loop sees it, appended to the question."""
    lines = [f"Interpretation: {plan.interpretation}"]
    if plan.assumptions:
        lines.append("Assumptions: " + "; ".join(plan.assumptions))
    lines.append(f"Expected unit: {plan.expected_unit or 'none'}")
    if not plan.answerable:
        lines.append("The planner judged this question NOT answerable from the dataset.")
    lines += [f"{i}. [{s.tool}] {s.description}" for i, s in enumerate(plan.steps, 1)]
    return "\n".join(lines)
