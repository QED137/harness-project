"""Thin wrapper around the OpenAI Responses API.

The agent loop only depends on the small LLMClient protocol below, so tests can
replace OpenAI with a scripted fake and the loop never needs the network.

Every call is measured: latency and input / output / reasoning / cached tokens.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class FunctionCall:
    name: str
    arguments: str  # raw JSON string exactly as the model produced it
    call_id: str


@dataclass
class LLMCallRecord:
    model: str
    latency_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    response_id: str | None = None


@dataclass
class LLMResponse:
    output_items: list[Any]  # passed back verbatim on the next call (includes reasoning items)
    function_calls: list[FunctionCall]
    text: str
    record: LLMCallRecord


class LLMClient(Protocol):
    model: str

    def create(self, *, instructions: str, input: list[Any], tools: list[dict[str, Any]]) -> LLMResponse: ...


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class OpenAIClient:
    model: str
    client: Any = field(default=None, repr=False)  # an openai.OpenAI instance; created lazily

    def __post_init__(self) -> None:
        if self.client is None:
            from openai import OpenAI  # imported here so tests without the SDK still import this module

            self.client = OpenAI()

    def create(self, *, instructions: str, input: list[Any], tools: list[dict[str, Any]]) -> LLMResponse:
        start = time.monotonic()
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input,
            tools=tools,
            parallel_tool_calls=False,  # at most one tool call per turn: simple traces, exact counts
        )
        latency = time.monotonic() - start

        output = list(_get(response, "output", []) or [])
        calls = [
            FunctionCall(_get(item, "name"), _get(item, "arguments") or "", _get(item, "call_id"))
            for item in output
            if _get(item, "type") == "function_call"
        ]
        usage = _get(response, "usage")
        record = LLMCallRecord(
            model=self.model,
            latency_s=latency,
            input_tokens=_get(usage, "input_tokens", 0) or 0,
            output_tokens=_get(usage, "output_tokens", 0) or 0,
            reasoning_tokens=_get(_get(usage, "output_tokens_details"), "reasoning_tokens", 0) or 0,
            cached_tokens=_get(_get(usage, "input_tokens_details"), "cached_tokens", 0) or 0,
            response_id=_get(response, "id"),
        )
        return LLMResponse(output, calls, _get(response, "output_text", "") or "", record)


# ------------------------------------------------------------------ configuration
def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader: KEY=VALUE lines, # comments. Never overrides variables
    that are already set, so real environment variables always win."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def openai_config_problem() -> str | None:
    """Return a human-readable problem with the configuration, or None if it is fine."""
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_MODEL") if not os.environ.get(k)]
    if missing:
        return (
            f"missing {', '.join(missing)}. Copy .env.example to .env and fill in "
            "OPENAI_API_KEY and OPENAI_MODEL (the model name, e.g. from the OpenAI models page)."
        )
    return None
