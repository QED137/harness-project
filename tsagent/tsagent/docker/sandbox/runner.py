"""Runs INSIDE the sandbox container.

Protocol
--------
stdin : one JSON job  {code, nonce, allowed_modules, runtime_policy, timeout_s,
                       preload, max_output_chars}
stdout: anything, followed by exactly one line
        "__TSAGENT_RESULT__<nonce><json payload>"

The host takes the LAST line carrying the nonce, so agent code that prints a
fake marker line is ignored unless it also recovers the nonce (see threat model).

This file is NOT the security boundary. The container is. The import guard and
the in-process timeout exist to (a) give the model a useful error message and
(b) produce measurable "policy violation" events for the evaluation harness.
"""

import builtins
import io
import json
import linecache
import math
import signal
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout

MARKER = "__TSAGENT_RESULT__"
AGENT_MODULE = "__agent__"
AGENT_FILENAME = "<agent>"


class SandboxTimeout(BaseException):
    """BaseException, so a model's `except Exception:` cannot swallow it.
    A bare `except:` still can; the host-side hard kill covers that case."""


# --------------------------------------------------------------------- guards
def make_import_guard(allowed: set[str], violations: list[dict]):
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        # Only police imports whose calling frame belongs to agent code.
        # pandas/numpy importing their own internals have their own module
        # globals and pass straight through.
        if globals is not None and globals.get("__name__") == AGENT_MODULE:
            top = name.partition(".")[0]
            if level > 0 or top not in allowed:
                violations.append({"kind": "runtime_import", "detail": name})
                raise ImportError(
                    f"import of '{name}' is not allowed in this sandbox; allowed modules: {sorted(allowed)}"
                )
        return real_import(name, globals, locals, fromlist, level)

    return real_import, guarded_import


def _on_alarm(signum, frame):
    raise SandboxTimeout()


# -------------------------------------------------------------- serialization
def describe(value, max_chars: int) -> dict:
    """Turn the agent's `result` into JSON the verifier can reason about.
    Never lets NaN/inf pass silently: they become null plus an explicit flag."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:  # keeps the runner usable without the data stack
        np = pd = None

    if value is None:
        return {"type": "none"}
    if isinstance(value, bool) or (np is not None and isinstance(value, np.bool_)):
        return {"type": "bool", "value": bool(value)}
    if isinstance(value, (int, float)) or (np is not None and isinstance(value, np.number)):
        f = float(value)
        return {
            "type": "number",
            "python_type": type(value).__name__,
            "value": f if math.isfinite(f) else None,
            "is_nan": math.isnan(f),
            "is_inf": math.isinf(f),
        }
    if isinstance(value, str):
        return {"type": "str", "value": value[:max_chars], "truncated": len(value) > max_chars}
    if pd is not None:
        if isinstance(value, pd.Timestamp):
            return {"type": "timestamp", "value": value.isoformat(), "is_nat": bool(pd.isna(value))}
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return {
                "type": type(value).__name__.lower(),
                "shape": list(value.shape),
                "n_missing": int(value.isna().sum().sum() if value.ndim == 2 else value.isna().sum()),
                "head": value.head(20).to_json(date_format="iso")[:max_chars],
            }
    text = repr(value)
    return {
        "type": "other",
        "python_type": type(value).__name__,
        "repr": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def agent_traceback(exc: BaseException, max_chars: int) -> str:
    """Traceback restricted to agent frames: what the model needs, no runner noise."""
    tb = traceback.TracebackException.from_exception(exc)
    tb.stack = traceback.StackSummary.from_list([f for f in tb.stack if f.filename == AGENT_FILENAME])
    return "".join(tb.format())[-max_chars:]


# ----------------------------------------------------------------------- main
def main() -> None:
    real_stdout = sys.stdout
    job = json.loads(sys.stdin.read())
    nonce = job["nonce"]
    max_chars = int(job.get("max_output_chars", 10_000))
    code = job["code"]

    payload = {
        "status": "ok",
        "error": None,
        "result": None,
        "result_missing": False,
        "stdout": "",
        "stdout_truncated": False,
        "violations": [],
        "exec_s": None,
    }

    agent_globals = {"__name__": AGENT_MODULE, "__builtins__": builtins}

    # Preload data before any guard is installed, so it is not charged to the agent.
    if job.get("preload"):
        try:
            import pandas as pd

            agent_globals["df"] = pd.read_parquet(job["preload"])
        except Exception as e:  # noqa: BLE001
            payload.update(status="setup_error", error=f"preload failed: {e!r}")
            real_stdout.write("\n" + MARKER + nonce + json.dumps(payload) + "\n")
            return

    try:
        compiled = compile(code, AGENT_FILENAME, "exec")
    except SyntaxError as e:
        payload.update(status="syntax_error", error=f"{e.msg} (line {e.lineno})")
        real_stdout.write("\n" + MARKER + nonce + json.dumps(payload) + "\n")
        return

    # Make tracebacks show the agent's source lines.
    linecache.cache[AGENT_FILENAME] = (len(code), None, code.splitlines(True), AGENT_FILENAME)

    violations: list[dict] = []
    real_import, guarded_import = make_import_guard(set(job["allowed_modules"]), violations)
    captured = io.StringIO()

    signal.signal(signal.SIGALRM, _on_alarm)
    start = time.monotonic()
    try:
        if job.get("runtime_policy", True):
            builtins.__import__ = guarded_import
        signal.setitimer(signal.ITIMER_REAL, float(job["timeout_s"]))
        with redirect_stdout(captured), redirect_stderr(captured):
            exec(compiled, agent_globals)
    except SandboxTimeout:
        payload.update(status="timeout", error=f"execution exceeded {job['timeout_s']}s")
    except MemoryError:
        payload.update(status="memory_error", error="MemoryError")
    except BaseException as e:  # noqa: BLE001  (includes SystemExit)
        payload.update(status="error", error=agent_traceback(e, max_chars))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        builtins.__import__ = real_import
        payload["exec_s"] = round(time.monotonic() - start, 4)

    out = captured.getvalue()
    payload["stdout"] = out[:max_chars]
    payload["stdout_truncated"] = len(out) > max_chars
    payload["violations"] = violations

    if payload["status"] == "ok":
        if "result" in agent_globals:
            try:
                payload["result"] = describe(agent_globals["result"], max_chars)
            except Exception as e:  # noqa: BLE001
                payload.update(status="error", error=f"could not serialize result: {e!r}")
        else:
            payload["result_missing"] = True

    real_stdout.write("\n" + MARKER + nonce + json.dumps(payload, allow_nan=False) + "\n")
    real_stdout.flush()


if __name__ == "__main__":
    main()
