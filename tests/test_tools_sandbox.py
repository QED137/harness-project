"""run_python through the REAL sandbox on the REAL dataset: the tool's answer must
equal the same computation done directly on the host. Needs Docker and the data."""

import json
from pathlib import Path

import pandas as pd
import pytest

from tsagent.sandbox import DockerSandbox, SandboxConfig
from tsagent.tools import ToolCallStatus, ToolRegistry

DATA = Path(__file__).parents[1] / "data"

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not (DATA / "weather.parquet").is_file(), reason="run: python -m tsagent.dataset"),
]


def test_run_python_on_real_data_matches_host_pandas():
    sb = DockerSandbox(SandboxConfig(data_dir=DATA, preload="/data/weather.parquet", timeout_s=20))
    reg = ToolRegistry.default(DATA, sb)
    code = 'result = df.loc["2023", "temperature_2m"].mean()'
    outcome = reg.call("run_python", json.dumps({"code": code}))
    assert outcome.status is ToolCallStatus.OK, outcome.output
    expected = pd.read_parquet(DATA / "weather.parquet").loc["2023", "temperature_2m"].mean()
    assert outcome.record.sandbox.result["value"] == pytest.approx(expected)
