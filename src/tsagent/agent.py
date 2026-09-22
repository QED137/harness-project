"""Step 3: the agent loop.

    python -m tsagent.agent "What was the mean temperature in 2023?"
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
from .schemas import SubmitAnswerArgs
from .tools import ToolCallRecord, ToolRegistry

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

DEFAULT_MAX_TURNS = 15
MAX_NUDGES = 1  # how often we remind a model that replied in text instead of calling a tool


class StopReason(StrEnum):
    ANSWERED = "answered"  # valid submit_answer
    MAX_TURNS = "max_turns"  # turn budget used up
    NO_SUBMIT = "no_submit"  # model kept replying in text, never submitted
    LLM_ERROR = "llm_error"  # the API call itself failed


@dataclass
class AgentRun:
    question: str
    model: str
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


def run_agent(
    question: str,
    llm: LLMClient,
    registry: ToolRegistry,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> AgentRun:
    run = AgentRun(question=question, model=llm.model)
    start = time.monotonic()
    first_record = len(registry.records)
    tools = registry.openai_tools()
    history: list[Any] = [{"role": "user", "content": question}]

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
                history.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": outcome.output}
                )
                if outcome.terminal:
                    run.stop_reason, run.answer = StopReason.ANSWERED, outcome.answer

            if run.stop_reason is StopReason.ANSWERED:
                break
        else:
            run.stop_reason = StopReason.MAX_TURNS
    finally:
        run.tool_calls = registry.records[first_record:]
        run.total_s = time.monotonic() - start
    return run


# ------------------------------------------------------------------ CLI
def format_run(run: AgentRun) -> str:
    lines = [f"question: {run.question}"]
    for i, rec in enumerate(run.tool_calls, 1):
        lines.append(f"  {i:>2}. {rec.tool:<17} {rec.status.value}")
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
    run = run_agent(args.question, llm, ToolRegistry.default(args.data_dir, sandbox), args.max_turns)
    print(format_run(run))
    if args.trace:
        args.trace.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n")
        print(f"trace:    {args.trace}")
    return 0 if run.stop_reason is StopReason.ANSWERED else 1


if __name__ == "__main__":
    raise SystemExit(main())
