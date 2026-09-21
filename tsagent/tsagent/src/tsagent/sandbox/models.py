from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_ALLOWED_MODULES = frozenset({"pandas", "numpy", "math", "statistics", "datetime"})


class SandboxStatus(StrEnum):
    OK = "ok"  # code ran; check result / result_missing
    ERROR = "error"  # agent code raised
    SYNTAX_ERROR = "syntax_error"
    POLICY_REJECTED = "policy_rejected"  # static check refused to run it
    TIMEOUT = "timeout"  # soft (runner) or hard (host kill)
    OOM_KILLED = "oom_killed"
    MEMORY_ERROR = "memory_error"  # Python-level MemoryError
    SETUP_ERROR = "setup_error"  # e.g. preload failed: our bug, not the agent's
    RUNNER_CRASH = "runner_crash"  # no result line: runner died or output was destroyed
    DOCKER_ERROR = "docker_error"  # infrastructure failure


class ViolationKind(StrEnum):
    STATIC_IMPORT = "static_import"
    STATIC_FORBIDDEN_NAME = "static_forbidden_name"
    STATIC_FORBIDDEN_ATTR = "static_forbidden_attr"
    RUNTIME_IMPORT = "runtime_import"


class LimitKind(StrEnum):
    SOFT_TIMEOUT = "soft_timeout"  # in-process timer fired
    HARD_KILL = "hard_kill"  # host killed the container (timer was evaded)
    OOM = "oom"
    OUTPUT_TRUNCATED = "output_truncated"


class Violation(BaseModel):
    kind: ViolationKind
    detail: str
    line: int | None = None


class SandboxConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    image: str = "tsagent-sandbox:latest"
    data_dir: Path | None = None  # mounted read-only at /data
    preload: str | None = None  # e.g. "/data/weather.parquet" -> `df`
    timeout_s: float = Field(10.0, gt=0)  # agent code execution budget
    startup_grace_s: float = Field(20.0, gt=0)  # container start + imports, before hard kill
    memory_mb: int = Field(512, ge=64)
    cpus: float = Field(1.0, gt=0)
    pids_limit: int = Field(64, ge=8)
    tmpfs_mb: int = Field(64, ge=1)
    max_output_chars: int = 10_000
    max_raw_output_bytes: int = 1_000_000  # host-side cap on what we read from the container
    allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES
    runtime: str | None = None  # "runsc" to use gVisor (Linux only)

    # Layer switches. Both True in production. Tests turn them off to prove the
    # container still holds when the Python-level policy is bypassed.
    static_policy: bool = True
    runtime_policy: bool = True


class SandboxResult(BaseModel):
    status: SandboxStatus
    result: dict[str, Any] | None = None
    result_missing: bool = False
    stdout: str = ""
    error: str | None = None
    violations: list[Violation] = []
    limits_hit: list[LimitKind] = []
    exit_code: int | None = None
    exec_s: float | None = None  # time inside agent code (from runner)
    total_s: float  # wall clock including container lifecycle
    container_name: str | None = None
