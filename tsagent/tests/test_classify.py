"""Host-side result classification, without Docker."""

import json

from tsagent.sandbox import DockerSandbox, LimitKind, SandboxConfig, SandboxStatus

MARKER = "__TSAGENT_RESULT__"
sb = DockerSandbox(SandboxConfig())


def classify(stdout="", **kw):
    args = dict(
        exit_code=0,
        stdout=stdout,
        stderr="",
        nonce="real",
        hard_killed=False,
        oom_killed=False,
        raw_truncated=False,
        name="c",
        total_s=1.0,
    )
    args.update(kw)
    return sb._classify(**args)


def payload(**kw):
    base = {
        "status": "ok",
        "error": None,
        "result": {"type": "number", "value": 1.0},
        "result_missing": False,
        "stdout": "",
        "stdout_truncated": False,
        "violations": [],
        "exec_s": 0.01,
    }
    base.update(kw)
    return json.dumps(base)


def test_last_line_with_real_nonce_wins_over_spoofs():
    out = "\n".join(
        [
            MARKER + "real" + payload(result={"type": "number", "value": 666.0}),  # earlier fake w/ nonce
            MARKER + "wrong" + payload(result={"type": "number", "value": 999.0}),
            MARKER + "real" + payload(),
        ]
    )
    assert classify(out).result["value"] == 1.0


def test_hard_kill_beats_everything():
    r = classify(MARKER + "real" + payload(), hard_killed=True)
    assert r.status is SandboxStatus.TIMEOUT and LimitKind.HARD_KILL in r.limits_hit


def test_oom():
    r = classify("", oom_killed=True, exit_code=137)
    assert r.status is SandboxStatus.OOM_KILLED and LimitKind.OOM in r.limits_hit


def test_no_payload_is_crash_or_docker_error():
    assert classify("", exit_code=1).status is SandboxStatus.RUNNER_CRASH
    assert classify("", exit_code=125, stderr="Unable to find image").status is SandboxStatus.DOCKER_ERROR


def test_soft_timeout_recorded():
    r = classify(MARKER + "real" + payload(status="timeout", result=None))
    assert r.status is SandboxStatus.TIMEOUT and LimitKind.SOFT_TIMEOUT in r.limits_hit


def test_result_marker_glued_to_agent_output_is_still_found():
    """Regression: agent output without a trailing newline (os.write) was glued to the
    front of the result line, and a line-start-only parser missed it."""
    out = "x" * 5000 + MARKER + "real" + payload() + "\n"
    r = classify(out)
    assert r.status is SandboxStatus.OK and r.result["value"] == 1.0


def test_garbage_after_marker_is_a_crash_not_an_exception():
    assert classify(MARKER + "real" + "{not json").status is SandboxStatus.RUNNER_CRASH
