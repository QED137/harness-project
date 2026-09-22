"""The real downloaded dataset loads inside the real sandbox.

Needs Docker, the sandbox image, and `python -m tsagent.dataset` to have been run.
"""

import json
from pathlib import Path

import pytest

from tsagent.sandbox import DockerSandbox, SandboxConfig, SandboxStatus

DATA = Path(__file__).parents[1] / "data"

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        not (DATA / "weather.parquet").is_file(),
        reason="dataset not downloaded; run: python -m tsagent.dataset",
    ),
]


def test_dataset_loads_as_df_in_sandbox():
    manifest = json.loads((DATA / "manifest.json").read_text())
    sb = DockerSandbox(SandboxConfig(data_dir=DATA, preload="/data/weather.parquet", timeout_s=20))
    code = "result = f'{len(df)}|{df.index.tz}|{\",\".join(df.columns)}'"
    r = sb.run(code)
    assert r.status is SandboxStatus.OK, r.error
    rows, tz, columns = r.result["value"].split("|")
    assert int(rows) == manifest["rows"]
    assert tz == "UTC"
    assert columns.split(",") == list(manifest["variables"])
