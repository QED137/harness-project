"""End to end: real OpenAI model, real sandbox, real data. Costs a few API calls.

Never runs by accident: skipped unless pytest is started with --live (see conftest.py),
and skipped if not configured:
    pytest --live tests/test_agent_live.py
Needs OPENAI_API_KEY and OPENAI_MODEL (e.g. in .env), Docker, the sandbox image,
and the downloaded dataset.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from tsagent.agent import StopReason, run_agent
from tsagent.llm import OpenAIClient, load_dotenv, openai_config_problem
from tsagent.sandbox import DockerSandbox, SandboxConfig
from tsagent.tools import ToolRegistry

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data"
load_dotenv(ROOT / ".env")

pytestmark = [
    pytest.mark.live,
    pytest.mark.docker,
    pytest.mark.skipif(openai_config_problem() is not None, reason="OPENAI_API_KEY / OPENAI_MODEL not set"),
    pytest.mark.skipif(not (DATA / "weather.parquet").is_file(), reason="run: python -m tsagent.dataset"),
]


def test_first_real_question_matches_pandas():
    sandbox = DockerSandbox(SandboxConfig(data_dir=DATA, preload="/data/weather.parquet"))
    llm = OpenAIClient(model=os.environ["OPENAI_MODEL"])
    run = run_agent(
        "What was the mean 2 m air temperature over the whole year 2023, in °C?",
        llm,
        ToolRegistry.default(DATA, sandbox),
    )
    assert run.stop_reason is StopReason.ANSWERED, (run.stop_reason, run.error, run.final_text)
    expected = pd.read_parquet(DATA / "weather.parquet").loc["2023", "temperature_2m"].mean()
    assert float(run.answer.value) == pytest.approx(expected, abs=0.05)
