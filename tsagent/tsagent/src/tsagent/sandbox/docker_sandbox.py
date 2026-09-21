"""Host side of the sandbox: one fresh container per execution.

Defence layers, outermost first:
  1. Container: no network, read-only rootfs, read-only data mount, small noexec
     tmpfs, memory+swap cap, CPU cap, PID cap, all capabilities dropped,
     no-new-privileges, unprivileged UID, Docker's default seccomp profile,
     optional gVisor runtime.
  2. Host wall-clock hard kill (covers code that swallows the in-process timer
     or blocks inside a C extension).
  3. Host-side bounded output reading (covers code writing gigabytes to fd 1).
  4. Runner: in-process timer and import guard (for good error messages and metrics).
  5. Static AST policy (cheap early rejection, precise feedback).
"""

import json
import secrets
import stat
import subprocess
import threading
import time
import uuid

from .models import (
    LimitKind,
    SandboxConfig,
    SandboxResult,
    SandboxStatus,
    Violation,
    ViolationKind,
)
from .policy import check_code

MARKER = "__TSAGENT_RESULT__"


class _TailReader(threading.Thread):
    """Drains a pipe completely but keeps only the last `limit` bytes.
    Draining matters: if we stopped reading, the container would block on a full pipe."""

    def __init__(self, stream, limit: int):
        super().__init__(daemon=True)
        self.stream, self.limit = stream, limit
        self.buf = bytearray()
        self.total = 0

    def run(self) -> None:
        for chunk in iter(lambda: self.stream.read(65536), b""):
            self.total += len(chunk)
            self.buf += chunk
            if len(self.buf) > self.limit:
                del self.buf[: len(self.buf) - self.limit]

    @property
    def truncated(self) -> bool:
        return self.total > self.limit

    def text(self) -> str:
        return self.buf.decode("utf-8", errors="replace")


class DockerSandbox:
    def __init__(self, config: SandboxConfig):
        self.config = config
        self._check_data_dir()

    # ------------------------------------------------------------ public API
    def run(self, code: str) -> SandboxResult:
        t0 = time.monotonic()
        cfg = self.config

        if cfg.static_policy:
            report = check_code(code, cfg.allowed_modules)
            if report.syntax_error:
                return SandboxResult(
                    status=SandboxStatus.SYNTAX_ERROR,
                    error=report.message_for_model(),
                    total_s=time.monotonic() - t0,
                )
            if report.violations:
                return SandboxResult(
                    status=SandboxStatus.POLICY_REJECTED,
                    error=report.message_for_model(),
                    violations=report.violations,
                    total_s=time.monotonic() - t0,
                )

        name = f"tsagent-sbx-{uuid.uuid4().hex[:12]}"
        nonce = secrets.token_hex(16)
        job = json.dumps(
            {
                "code": code,
                "nonce": nonce,
                "allowed_modules": sorted(cfg.allowed_modules),
                "runtime_policy": cfg.runtime_policy,
                "timeout_s": cfg.timeout_s,
                "preload": cfg.preload,
                "max_output_chars": cfg.max_output_chars,
            }
        ).encode()

        hard_killed = False
        try:
            proc = subprocess.Popen(
                self._docker_cmd(name), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError:
            return SandboxResult(
                status=SandboxStatus.DOCKER_ERROR, error="docker CLI not found", total_s=time.monotonic() - t0
            )
        try:
            # Popen was given PIPEs, so these are never None. An explicit check, not `assert`,
            # because asserts are removed when Python runs with -O.
            if proc.stdin is None or proc.stdout is None or proc.stderr is None:
                raise RuntimeError("docker process started without pipes")
            out = _TailReader(proc.stdout, cfg.max_raw_output_bytes)
            err = _TailReader(proc.stderr, 20_000)
            out.start()
            err.start()
            try:
                proc.stdin.write(job)
                proc.stdin.close()
            except BrokenPipeError:
                pass  # container failed to start; stderr will say why

            try:
                exit_code = proc.wait(timeout=cfg.timeout_s + cfg.startup_grace_s)
            except subprocess.TimeoutExpired:
                hard_killed = True
                self._docker("kill", name)
                exit_code = proc.wait(timeout=30)
            out.join(timeout=5)
            err.join(timeout=5)
            oom_killed = self._oom_killed(name)
        finally:
            self._docker("rm", "-f", name)

        return self._classify(
            exit_code=exit_code,
            stdout=out.text(),
            stderr=err.text(),
            nonce=nonce,
            hard_killed=hard_killed,
            oom_killed=oom_killed,
            raw_truncated=out.truncated,
            name=name,
            total_s=time.monotonic() - t0,
        )

    # ------------------------------------------------------------ internals
    def _check_data_dir(self) -> None:
        """Fail fast with a clear message if the sandbox user cannot read the data.

        The container runs as UID 65534 (nobody), not as the host user, so the
        mounted directory must be enterable and the files readable by "others".
        Without this check the failure shows up much later as a PermissionError
        inside the container.
        """
        cfg = self.config
        if cfg.data_dir is None:
            return
        d = cfg.data_dir
        if not d.is_dir():
            raise ValueError(f"data_dir does not exist or is not a directory: {d}")
        mode = d.stat().st_mode
        if not (mode & stat.S_IROTH and mode & stat.S_IXOTH):
            raise ValueError(
                f"data_dir {d} is not readable by the sandbox user (UID 65534). Fix with: chmod o+rx {d}"
            )
        if cfg.preload and cfg.preload.startswith("/data/"):
            host_file = d / cfg.preload.removeprefix("/data/")
            if not host_file.is_file():
                raise ValueError(f"preload file not found on host: {host_file}")
            if not host_file.stat().st_mode & stat.S_IROTH:
                raise ValueError(
                    f"preload file {host_file} is not readable by the sandbox user. "
                    f"Fix with: chmod o+r {host_file}"
                )

    def _docker_cmd(self, name: str) -> list[str]:
        cfg = self.config
        cmd = [
            "docker",
            "run",
            "--interactive",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,nodev,size={cfg.tmpfs_mb}m",  # noqa: S108 (path inside the container)
            "--memory",
            f"{cfg.memory_mb}m",
            "--memory-swap",
            f"{cfg.memory_mb}m",  # equal to --memory: no swap
            "--cpus",
            str(cfg.cpus),
            "--pids-limit",
            str(cfg.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--ulimit",
            "nofile=256:256",
        ]
        if cfg.data_dir is not None:
            cmd += ["--mount", f"type=bind,source={cfg.data_dir.resolve()},target=/data,readonly"]
        if cfg.runtime:
            cmd += ["--runtime", cfg.runtime]
        # No --rm: we need `docker inspect` afterwards to learn whether the OOM killer fired.
        return cmd + [cfg.image]

    @staticmethod
    def _docker(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30)

    def _oom_killed(self, name: str) -> bool:
        r = self._docker("inspect", "--format", "{{.State.OOMKilled}}", name)
        return r.returncode == 0 and r.stdout.strip() == "true"

    def _classify(
        self, *, exit_code, stdout, stderr, nonce, hard_killed, oom_killed, raw_truncated, name, total_s
    ) -> SandboxResult:
        limits: list[LimitKind] = []
        if raw_truncated:
            limits.append(LimitKind.OUTPUT_TRUNCATED)
        base = dict(exit_code=exit_code, total_s=total_s, container_name=name, limits_hit=limits)

        if hard_killed:
            limits.append(LimitKind.HARD_KILL)
            return SandboxResult(
                status=SandboxStatus.TIMEOUT,
                **base,
                error=f"hard-killed after {self.config.timeout_s + self.config.startup_grace_s}s "
                "(code evaded the in-process timeout or startup was too slow)",
            )
        if oom_killed:
            limits.append(LimitKind.OOM)
            return SandboxResult(
                status=SandboxStatus.OOM_KILLED,
                **base,
                error=f"killed: exceeded {self.config.memory_mb} MB memory limit",
            )

        payload = self._parse_payload(stdout, nonce)
        if payload is None:
            status = (
                SandboxStatus.DOCKER_ERROR if exit_code in (125, 126, 127) else SandboxStatus.RUNNER_CRASH
            )
            return SandboxResult(status=status, **base, error=(stderr or stdout)[-2000:] or None)

        if payload["status"] == "timeout":
            limits.append(LimitKind.SOFT_TIMEOUT)
        if payload.get("stdout_truncated") and LimitKind.OUTPUT_TRUNCATED not in limits:
            limits.append(LimitKind.OUTPUT_TRUNCATED)

        return SandboxResult(
            status=SandboxStatus(payload["status"]),
            result=payload.get("result"),
            result_missing=payload.get("result_missing", False),
            stdout=payload.get("stdout", ""),
            error=payload.get("error"),
            violations=[
                Violation(kind=ViolationKind(v["kind"]), detail=v["detail"])
                for v in payload.get("violations", [])
            ],
            exec_s=payload.get("exec_s"),
            **base,
        )

    @staticmethod
    def _parse_payload(stdout: str, nonce: str) -> dict | None:
        """Find the LAST occurrence of marker+nonce anywhere in the output.

        Not only at line starts: agent code may write output without a trailing
        newline (e.g. os.write), which glues its text to the front of our line.
        """
        prefix = MARKER + nonce
        idx = stdout.rfind(prefix)
        if idx == -1:
            return None
        body = stdout[idx + len(prefix) :].split("\n", 1)[0]
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None
