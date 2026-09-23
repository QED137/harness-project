"""Step 3: the agent loop.

    python -m tsagent.agent "What was the mean temperature in 2023?"
    python -m tsagent.agent "..." --planner           # plan first (step 4), then run the loop
    python -m tsagent.agent "..." --verifier          # check answers before accepting them (step 5)
    python -m tsagent.agent "..." --trace run.json     # also save the full run record

Loop: send the question and the tool definitions to the model; execute the tool
call it asks for; send the result back; repeat until a valid submit_answer or a
stop condition. Every model call and every tool call is recorded in AgentRun.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from .llm import LLMCallRecord, LLMClient, OpenAIClient, load_dotenv, openai_config_problem
from .planner import PlanResult, PlanStatus, format_plan, make_plan
from .schemas import Plan, SubmitAnswerArgs
from .tools import ToolCallRecord, ToolRegistry
from .verifier import CheckStatus, Verdict, collect_evidence, verify

SYSTEM_PROMPT = """\
You answer quantitative questions about an hourly weather dataset by writing and running Python.

How to work:
1. Call describe_dataset first to see the columns, their units and the time range.
2. Compute the answer with run_python. The data is preloaded as `df`; timestamps are UTC.
   Each run_python call is independent: variables do not persist, so repeat any setup.
3. Check your result before answering: is the number plausible, is the unit right,
   did you use the period the question asks about? Convert units if the question asks for a
   different unit than the data has.
4. Call submit_answer exactly once with the value, its unit, and a short method summary.

Never guess or estimate from memory: every number you submit must come from run_python.
If the question cannot be answered from this dataset, say so in method_summary and submit
the value "not answerable".
"""

PLAN_INTRO = "A planning step produced this plan. Follow it, but correct it if the data shows it is wrong:"

DEFAULT_MAX_TURNS = 15
MAX_NUDGES = 1  # how often we remind a model that replied in text instead of calling a tool
MAX_REJECTIONS = 2  # verifier rejections the model may correct; the next one ends the run


class StopReason(StrEnum):
    ANSWERED = "answered"  # valid submit_answer
    MAX_TURNS = "max_turns"  # turn budget used up
    NO_SUBMIT = "no_submit"  # model kept replying in text, never submitted
    LLM_ERROR = "llm_error"  # the API call itself failed
    REJECTED = "rejected"  # the verifier rejected more than MAX_REJECTIONS answers


@dataclass
class VerificationRecord:
    answer: SubmitAnswerArgs
    verdict: Verdict
    ok: bool


@dataclass
class AgentRun:
    question: str
    model: str
    use_planner: bool = False
    plan_status: PlanStatus | None = None  # None when the planner is off
    plan: Plan | None = None
    plan_error: str | None = None
    use_verifier: bool = False
    verifications: list[VerificationRecord] = field(default_factory=list)
    rejected_answer: SubmitAnswerArgs | None = None  # last answer, when the run ended as REJECTED
    stop_reason: StopReason | None = None
    answer: SubmitAnswerArgs | None = None
    turns: int = 0
    nudges: int = 0
    error: str | None = None
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    final_text: str = ""
    total_s: float = 0.0

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable record of the whole run (used as an evaluation trace)."""

        def convert(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {f.name: convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj

        data = convert(self)
        data["input_tokens"] = self.input_tokens
        data["output_tokens"] = self.output_tokens
        return data


def _plan(question: str, llm: LLMClient, registry: ToolRegistry) -> PlanResult:
    """Run the planner. The dataset description is produced by calling the tool function
    directly (host side), so it is not counted as a tool call made by the model."""
    try:
        tool = registry.tools["describe_dataset"]
        description = tool.run(tool.args_model()).output
    except Exception as e:  # noqa: BLE001  our side failed: recorded, not raised
        return PlanResult(PlanStatus.SETUP_ERROR, error=f"{type(e).__name__}: {e}")
    return make_plan(question, llm, description)


def run_agent(
    question: str,
    llm: LLMClient,
    registry: ToolRegistry,
    max_turns: int = DEFAULT_MAX_TURNS,
    use_planner: bool = False,
    use_verifier: bool = False,
    max_rejections: int = MAX_REJECTIONS,
) -> AgentRun:
    run = AgentRun(question=question, model=llm.model, use_planner=use_planner, use_verifier=use_verifier)
    start = time.monotonic()
    first_record = len(registry.records)
    tools = registry.openai_tools()

    first_message = question
    if use_planner:
        result = _plan(question, llm, registry)
        run.plan_status, run.plan, run.plan_error = result.status, result.plan, result.error
        if result.record is not None:
            run.llm_calls.append(result.record)
        if result.plan is not None:  # an invalid plan is recorded; the agent continues without one
            first_message = f"{question}\n\n{PLAN_INTRO}\n{format_plan(result.plan)}"
    history: list[Any] = [{"role": "user", "content": first_message}]

    try:
        while run.turns < max_turns:
            run.turns += 1
            try:
                response = llm.create(instructions=SYSTEM_PROMPT, input=history, tools=tools)
            except Exception as e:  # noqa: BLE001  network, auth, rate limit: recorded, not raised
                run.stop_reason, run.error = StopReason.LLM_ERROR, f"{type(e).__name__}: {e}"
                break
            run.llm_calls.append(response.record)
            history += response.output_items  # includes reasoning items, required by reasoning models

            if not response.function_calls:
                run.final_text = response.text
                if run.nudges >= MAX_NUDGES:
                    run.stop_reason = StopReason.NO_SUBMIT
                    break
                run.nudges += 1
                history.append(
                    {"role": "user", "content": "Please use the tools: finish by calling submit_answer."}
                )
                continue

            for call in response.function_calls:
                outcome = registry.call(call.name, call.arguments)
                output, accepted = outcome.output, outcome.terminal
                if outcome.terminal and use_verifier and outcome.answer is not None:
                    verdict = _verify(outcome.answer, call.arguments, registry, first_record, run.plan)
                    run.verifications.append(VerificationRecord(outcome.answer, verdict, verdict.ok))
                    if not verdict.ok:
                        output, accepted = verdict.message_for_model(), False
                        if sum(not v.ok for v in run.verifications) > max_rejections:
                            run.stop_reason, run.rejected_answer = StopReason.REJECTED, outcome.answer
                history.append({"type": "function_call_output", "call_id": call.call_id, "output": output})
                if accepted:
                    run.stop_reason, run.answer = StopReason.ANSWERED, outcome.answer

            if run.stop_reason in (StopReason.ANSWERED, StopReason.REJECTED):
                break
        else:
            run.stop_reason = StopReason.MAX_TURNS
    finally:
        run.tool_calls = registry.records[first_record:]
        run.total_s = time.monotonic() - start
    return run


def _verify(
    answer: SubmitAnswerArgs, raw_arguments: str, registry: ToolRegistry, first_record: int, plan: Plan | None
) -> Verdict:
    """Verify against what run_python actually produced in THIS run only."""
    results = [r.sandbox for r in registry.records[first_record:] if r.tool == "run_python"]
    expected_unit = plan.expected_unit if plan is not None else None
    return verify(answer, collect_evidence(results), raw_arguments, expected_unit)


# ------------------------------------------------------------------ CLI
def format_run(run: AgentRun) -> str:
    lines = [f"question: {run.question}"]
    if run.use_planner:
        if run.plan is not None:
            lines.append(f"plan:     ok, {len(run.plan.steps)} steps; {run.plan.interpretation}")
        else:
            status = run.plan_status.value if run.plan_status else "unknown"
            lines.append(f"plan:     {status} (continued without a plan): {run.plan_error}")
    for i, rec in enumerate(run.tool_calls, 1):
        lines.append(f"  {i:>2}. {rec.tool:<17} {rec.status.value}")
    if run.use_verifier:
        rejected = sum(not v.ok for v in run.verifications)
        warnings = [c for v in run.verifications for c in v.verdict.checks if c.status is CheckStatus.WARN]
        if not run.verifications:
            lines.append("verifier: no answer submitted")
        else:
            accepted = "rejected, run stopped" if run.stop_reason is StopReason.REJECTED else "accepted"
            lines.append(f"verifier: {accepted} after {rejected} rejection(s)")
        for v in run.verifications:
            for c in v.verdict.failures():
                lines.append(f"          rejected {v.answer.value}: {c.name}: {c.detail}")
        for c in warnings:
            lines.append(f"          warning: {c.name}: {c.detail}")
    if run.answer is not None:
        unit = f" {run.answer.unit}" if run.answer.unit else ""
        lines.append(f"answer:   {run.answer.value}{unit}")
        lines.append(f"method:   {run.answer.method_summary}")
    else:
        lines.append(f"no answer ({run.stop_reason.value if run.stop_reason else 'unknown'})")
        if run.error:
            lines.append(f"error:    {run.error}")
    lines.append(
        f"stats:    {run.turns} turns, {len(run.tool_calls)} tool calls, "
        f"{run.input_tokens} in / {run.output_tokens} out tokens, {run.total_s:.1f}s, model {run.model}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .sandbox import DockerSandbox, SandboxConfig

    p = argparse.ArgumentParser(prog="python -m tsagent.agent", description="Ask the agent one question.")
    p.add_argument("question")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--planner", action="store_true", help="run the planner before the agent loop (step 4)")
    p.add_argument("--verifier", action="store_true", help="verify answers before accepting them (step 5)")
    p.add_argument("--trace", type=Path, help="write the full run record as JSON to this file")
    args = p.parse_args(argv)

    load_dotenv()
    problem = openai_config_problem()
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    try:
        sandbox = DockerSandbox(SandboxConfig(data_dir=args.data_dir, preload="/data/weather.parquet"))
    except ValueError as e:
        print(f"error: {e}\nDid you run: python -m tsagent.dataset ?", file=sys.stderr)
        return 2

    llm = OpenAIClient(model=os.environ["OPENAI_MODEL"])
    registry = ToolRegistry.default(args.data_dir, sandbox)
    run = run_agent(
        args.question, llm, registry, args.max_turns, use_planner=args.planner, use_verifier=args.verifier
    )
    print(format_run(run))
    if args.trace:
        args.trace.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n")
        print(f"trace:    {args.trace}")
    return 0 if run.stop_reason is StopReason.ANSWERED else 1


if __name__ == "__main__":
    raise SystemExit(main())
