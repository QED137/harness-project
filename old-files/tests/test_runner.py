"""The in-container runner, executed as a local subprocess: no Docker needed.
Tests runner LOGIC only (protocol, guard, timer, serialization), not isolation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = Path(__file__).parents[1] / "docker" / "sandbox" / "runner.py"
MARKER = "__TSAGENT_RESULT__"
NONCE = "testnonce"


def run(code, *, timeout_s=2.0, runtime_policy=True, allowed=("pandas", "numpy", "math"), preload=None):
    job = {
        "code": code,
        "nonce": NONCE,
        "allowed_modules": list(allowed),
        "runtime_policy": runtime_policy,
        "timeout_s": timeout_s,
        "preload": preload,
        "max_output_chars": 1000,
    }
    proc = subprocess.run(
        [sys.executable, "-I", "-B", str(RUNNER)],
        input=json.dumps(job),
        capture_output=True,
        text=True,
        timeout=timeout_s + 20,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith(MARKER + NONCE)]
    assert lines, f"no result line. stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return json.loads(lines[-1][len(MARKER + NONCE) :])


def test_number_result():
    p = run("result = 2 + 2")
    assert p["status"] == "ok"
    assert p["result"] == {
        "type": "number",
        "python_type": "int",
        "value": 4.0,
        "is_nan": False,
        "is_inf": False,
    }


def test_nan_is_flagged_never_silent():
    p = run("import math\nresult = math.nan")
    assert p["result"]["is_nan"] is True and p["result"]["value"] is None


def test_numpy_scalar_and_pandas_series():
    p = run("import numpy as np, pandas as pd\nresult = pd.Series([1.0, np.nan, 3.0])")
    assert p["result"]["type"] == "series" and p["result"]["n_missing"] == 1


def test_missing_result_flagged():
    assert run("x = 1")["result_missing"] is True


def test_runtime_import_guard_blocks_agent_imports():
    p = run("import os\nresult = 1")
    assert p["status"] == "error"
    assert p["violations"] == [{"kind": "runtime_import", "detail": "os"}]
    assert "not allowed" in p["error"]


def test_runtime_guard_does_not_break_library_internals():
    # pandas imports os, re, etc. internally; that must keep working.
    p = run("import pandas as pd\nresult = pd.to_datetime(['2024-01-01']).year[0]")
    assert p["status"] == "ok", p["error"]


def test_guard_applies_inside_agent_defined_functions():
    p = run("def f():\n    import socket\nf()")
    assert p["violations"][0]["detail"] == "socket"


def test_soft_timeout_not_swallowed_by_except_exception():
    p = run("while True:\n    try:\n        pass\n    except Exception:\n        pass", timeout_s=0.5)
    assert p["status"] == "timeout"


def test_traceback_shows_agent_line():
    p = run("x = 1\ny = x / 0")
    assert "line 2" in p["error"] and "ZeroDivisionError" in p["error"]
    assert "runner.py" not in p["error"]


def test_print_captured_and_truncated():
    p = run("print('x' * 5000)\nresult = 1")
    assert p["stdout_truncated"] is True and len(p["stdout"]) == 1000


def test_spoofed_marker_with_wrong_nonce_is_ignored():
    # Written to the REAL stdout (bypassing print capture), with the guard off.
    code = (
        "import sys\n"
        "sys.__stdout__.write('__TSAGENT_RESULT__wrongnonce' + chr(123) + chr(125) + chr(10))\n"
        "result = 7\n"
    )
    p = run(code, runtime_policy=False)
    assert p["result"]["value"] == 7.0


def test_system_exit_is_an_error():
    assert run("raise SystemExit(0)")["status"] == "error"


def test_preload_parquet(tmp_path):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "w.parquet"
    pd.DataFrame({"temp": [1.0, 2.0, 3.0]}).to_parquet(path)
    p = run("result = df['temp'].mean()", preload=str(path))
    assert p["result"]["value"] == 2.0
