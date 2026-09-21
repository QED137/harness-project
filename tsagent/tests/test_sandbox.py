"""Isolation tests against the real container.

Most tests turn OFF both Python-level policies (static + runtime) to simulate an
attacker who has already bypassed them. The container alone must then hold.

  docker build -t tsagent-sandbox:latest docker/sandbox
  pytest -m docker
"""

import shutil
import subprocess

import pytest

from tsagent.sandbox import (
    DockerSandbox,
    LimitKind,
    SandboxConfig,
    SandboxStatus,
    ViolationKind,
)

IMAGE = "tsagent-sandbox:latest"


def _image_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(not _image_available(), reason=f"docker or image {IMAGE} not available"),
]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    pd = pytest.importorskip("pandas")
    d = tmp_path_factory.mktemp("data")
    pd.DataFrame({"temp": [10.0, 12.0, 14.0]}).to_parquet(d / "weather.parquet")
    # pytest creates temp dirs as 0700 (owner only). The container runs as UID 65534,
    # so the mounted dir and file must be readable by "others".
    d.chmod(0o755)
    (d / "weather.parquet").chmod(0o644)
    return d


def sandbox(data_dir=None, **overrides) -> DockerSandbox:
    cfg = dict(timeout_s=3.0, startup_grace_s=20.0, data_dir=data_dir)
    cfg.update(overrides)
    return DockerSandbox(SandboxConfig(**cfg))


def unguarded(data_dir=None, **overrides) -> DockerSandbox:
    """Both Python-level policy layers disabled: tests the container alone."""
    return sandbox(data_dir, static_policy=False, runtime_policy=False, **overrides)


def probe(sb: DockerSandbox, body: str) -> str:
    """Run code that sets result to 'blocked' if the attempt failed."""
    code = (
        "result = 'allowed'\ntry:\n"
        + "\n".join("    " + line for line in body.splitlines())
        + "\nexcept Exception as e:\n    result = 'blocked: ' + type(e).__name__"
    )
    r = sb.run(code)
    assert r.status is SandboxStatus.OK, (r.status, r.error)
    return r.result["value"]


# ---------------------------------------------------------------- happy path
def test_reads_preloaded_data(data_dir):
    r = sandbox(data_dir, preload="/data/weather.parquet").run("result = df['temp'].mean()")
    assert r.status is SandboxStatus.OK, r.error
    assert r.result["value"] == 12.0


# ------------------------------------------------------ policy layers engage
def test_static_policy_rejects_before_container_starts():
    r = sandbox().run("import os")
    assert r.status is SandboxStatus.POLICY_REJECTED
    assert r.container_name is None  # no container was created
    assert r.violations[0].kind is ViolationKind.STATIC_IMPORT


def test_runtime_guard_catches_what_static_layer_missed():
    r = sandbox(static_policy=False).run("import socket")
    assert r.violations and r.violations[0].kind is ViolationKind.RUNTIME_IMPORT


# ------------------------------------------ container holds with policies off
def test_no_network():
    sb = unguarded()
    assert probe(sb, "import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)").startswith(
        "blocked"
    )


def test_no_dns():
    assert probe(unguarded(), "import socket\nsocket.getaddrinfo('example.com', 443)").startswith("blocked")


def test_data_mount_is_read_only(data_dir):
    assert probe(unguarded(data_dir), "open('/data/pwned', 'w').write('x')").startswith("blocked")


def test_root_filesystem_is_read_only():
    assert probe(unguarded(), "open('/sandbox/runner.py', 'a').write('x')").startswith("blocked")


def test_tmp_is_writable_but_noexec():
    sb = unguarded()
    code = (
        "import os, subprocess\n"
        "open('/tmp/x.sh', 'w').write('#!/bin/sh\\necho hi')\n"
        "os.chmod('/tmp/x.sh', 0o755)\n"
        "try:\n"
        "    subprocess.run(['/tmp/x.sh'], check=True)\n"
        "    result = 'executed'\n"
        "except PermissionError:\n"
        "    result = 'noexec'\n"
    )
    r = sb.run(code)
    assert r.result["value"] == "noexec", (r.status, r.error)


def test_runs_as_nobody_without_capabilities():
    r = unguarded().run("import os\nresult = os.getuid()")
    assert r.result["value"] == 65534.0
    assert probe(unguarded(), "import os\nos.setuid(0)").startswith("blocked")


def test_pid_limit_stops_thread_or_fork_bomb():
    code = (
        "import threading, time\n"
        "n = 0\n"
        "try:\n"
        "    while n < 10_000:\n"
        "        threading.Thread(target=time.sleep, args=(10,), daemon=True).start()\n"
        "        n += 1\n"
        "except RuntimeError:\n"
        "    pass\n"
        "result = n\n"
    )
    r = unguarded(pids_limit=64).run(code)
    assert r.status is SandboxStatus.OK, r.error
    assert r.result["value"] < 64


@pytest.mark.slow
def test_memory_limit():
    code = "blocks = []\nwhile True:\n    blocks.append(b'x' * 50_000_000)"
    r = unguarded(memory_mb=256).run(code)
    assert r.status in (SandboxStatus.OOM_KILLED, SandboxStatus.MEMORY_ERROR), (r.status, r.error)


def test_soft_timeout():
    r = sandbox(timeout_s=1.0).run("while True:\n    pass")
    assert r.status is SandboxStatus.TIMEOUT and LimitKind.SOFT_TIMEOUT in r.limits_hit


@pytest.mark.slow
def test_hard_kill_when_code_swallows_the_timer():
    code = (
        "while True:\n"
        "    try:\n"
        "        while True:\n"
        "            pass\n"
        "    except BaseException:\n"
        "        pass"
    )
    r = sandbox(timeout_s=1.0, startup_grace_s=5.0).run(code)
    assert r.status is SandboxStatus.TIMEOUT and LimitKind.HARD_KILL in r.limits_hit
    assert r.total_s < 15


def test_output_flood_on_raw_fd_is_bounded():
    code = "import os\nfor _ in range(200):\n    os.write(1, b'x' * 1_000_000)\nresult = 1"
    r = unguarded(max_raw_output_bytes=100_000).run(code)
    assert r.status is SandboxStatus.OK, (r.status, r.error)  # result line survives in the tail
    assert LimitKind.OUTPUT_TRUNCATED in r.limits_hit


def test_containers_are_cleaned_up():
    r = sandbox().run("result = 1")
    ps = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={r.container_name}", "-q"], capture_output=True, text=True
    )
    assert ps.stdout.strip() == ""
