# tsagent

An LLM agent that answers quantitative questions about a weather time series by writing Python and running it in a locked-down Docker sandbox, plus an evaluation harness that measures how often it is actually right.

The evaluation is the point of the project. Every question has a ground-truth answer computed directly with pandas, every question is run many times because the system is non-deterministic, and this README reports what the agent gets wrong, not only what it gets right.

> **Status: work in progress.** The sandbox is implemented and tested. The planner, tools, verifier and evaluation harness are designed but not yet built. Sections marked *planned* describe intended behaviour, not results. No accuracy numbers are reported until they have been measured.

---

## Contents

- [tsagent](#tsagent)
  - [Contents](#contents)
  - [How it works](#how-it-works)
  - [The sandbox](#the-sandbox)
    - [Design principle](#design-principle)
    - [Layers](#layers)
    - [Container configuration](#container-configuration)
    - [Result protocol](#result-protocol)
    - [Usage](#usage)
  - [Evaluation](#evaluation)
    - [Setup](#setup)
    - [Metrics](#metrics)
    - [Results](#results)
    - [Failure analysis](#failure-analysis)
  - [Quick start](#quick-start)
  - [Testing](#testing)
    - [Bugs found by the isolation tests](#bugs-found-by-the-isolation-tests)
  - [Development](#development)
  - [Repository layout](#repository-layout)
  - [Known limitations](#known-limitations)
  - [Roadmap](#roadmap)
    - [Achieved so far](#achieved-so-far)
    - [Next steps](#next-steps)
    - [Possible extensions (after the core is done)](#possible-extensions-after-the-core-is-done)
  - [License](#license)

---

## How it works

```
question
   │
   ▼
Planner ───────────── LLM call → ordered steps, validated as a Pydantic model   (planned)
   │
   ▼
Executor ◄──────────► Tools, each with a Pydantic argument schema               (planned)
   │                    ├─ describe_dataset
   │                    ├─ run_python ──► Docker sandbox                        (implemented)
   │                    └─ submit_answer
   ▼
Verifier ──────────── type, plausible range, units, no silent NaN               (planned)
   │
   ▼
answer
```

**Planner.** One LLM call decomposes the question into ordered steps, returned as structured output and validated against a Pydantic model. Invalid plans are counted, not silently repaired.

**Tools.** Each tool is defined by one Pydantic model. The same model generates the JSON schema the LLM sees and validates the arguments the LLM sends back, so the definition and the check cannot drift apart. Validation failures are returned to the model as errors and counted for evaluation.

**Sandbox.** Generated code runs in a fresh, isolated container per execution. See [below](#the-sandbox).

**Verifier.** A deterministic check that the returned value is sane before it is reported: correct type, within a physically plausible range for the quantity, correct unit, and not NaN or infinite. It is plain code, not a model.

A note on terminology: this is an orchestrated pipeline of LLM calls with deterministic components around them. It is not described as "multi-agent" because only the planner and executor are model-driven.

---

## The sandbox

The agent writes code that nobody reviews before it runs. The sandbox makes sure that whatever that code does, whether by mistake or because of instructions injected through the data, it cannot reach the network, change the data or the host, or consume unbounded resources. Every blocked attempt is recorded, so the sandbox is also a source of evaluation data.

### Design principle

**The container is the security boundary. The Python-level checks are not.**

Python cannot be sandboxed at the language level. For example, `pd.io.common.os.system(...)` reaches the operating system without any `import` statement. The static check and import guard exist to give the model precise, fast feedback and to produce countable policy events. Isolation comes from the container.

### Layers

| Layer | Where | Purpose | Can it be bypassed? |
|---|---|---|---|
| Static AST check | host, before start | reject disallowed imports, `eval`, `open`, reflection | yes; feedback and metrics only |
| Runtime import guard | inside container | block imports from agent code | yes; feedback and metrics only |
| Soft timeout (`SIGALRM`) | inside container | stop runaway Python loops cleanly | yes, via bare `except:` or long C calls |
| Hard kill | host | `docker kill` after the wall-clock budget | no |
| Output cap | host | keep only the tail of stdout | no |
| Container | Docker + kernel | network, filesystem, privileges, memory, CPU, PIDs | only via kernel or runtime exploits |

### Container configuration

| Setting | Effect |
|---|---|
| `--network none` | no network, no DNS |
| `--read-only` | root filesystem is read-only |
| `--mount type=bind,...,readonly` | dataset mounted read-only at `/data` |
| `--tmpfs /tmp:noexec,nosuid,nodev,size=64m` | small scratch space; nothing in it can be executed |
| `--memory 512m --memory-swap 512m` | memory cap with no swap |
| `--cpus 1` | CPU cap |
| `--pids-limit 64` | stops fork and thread bombs |
| `--cap-drop ALL`, `--security-opt no-new-privileges` | no Linux capabilities, no privilege escalation |
| `--user 65534:65534` | runs as `nobody` |
| `--runtime runsc` (optional) | gVisor, for a user-space kernel on Linux |

Code is passed to the container over stdin, so no writable host directory is shared.

### Result protocol

The runner writes exactly one result line, prefixed with a random per-execution nonce. The host takes the last line carrying that nonce, so agent code that prints a fake result line is ignored. NaN and infinity are never serialized silently: they become `null` with explicit `is_nan` / `is_inf` flags.

Each execution returns a `SandboxResult`:

```python
SandboxResult(
    status=SandboxStatus.OK,        # ok | error | syntax_error | policy_rejected | timeout
                                    # | oom_killed | memory_error | setup_error
                                    # | runner_crash | docker_error
    result={"type": "number", "value": 12.4, "is_nan": False, "is_inf": False, ...},
    violations=[],                  # policy events: static_import, runtime_import, ...
    limits_hit=[],                  # soft_timeout, hard_kill, oom, output_truncated
    exec_s=0.041,                   # time spent in agent code
    total_s=1.37,                   # wall clock including container start and teardown
)
```

Policy violations ("the model tried to import `requests`") and resource limits ("the model wrote a slow loop") are kept separate because they are different failure modes. Execution time and total time are kept separate so container cold start does not distort latency measurements.

### Usage

```python
from pathlib import Path
from tsagent.sandbox import DockerSandbox, SandboxConfig

sandbox = DockerSandbox(
    SandboxConfig(
        data_dir=Path("data"),
        preload="/data/weather.parquet",  # available to agent code as `df`
        timeout_s=10,
        memory_mb=512,
    )
)

r = sandbox.run("result = df['temperature'].resample('MS').mean().max()")
print(r.status, r.result, r.total_s)
```

**Data permissions.** The container runs as UID 65534, not as you, so the data directory must be readable by others (`chmod o+rx data && chmod o+r data/*.parquet`). `DockerSandbox` checks this at construction and tells you the exact command if it fails.

The full threat model, including accepted risks, is in [`docs/threat_model.md`](docs/threat_model.md).

---

## Evaluation

*Planned. The metrics below are defined; no results have been measured yet.*

### Setup

- About 50 questions about one frozen public weather dataset, each with a ground-truth answer computed directly in pandas (`eval/ground_truth.py`).
- Each question is run 10 times, because model sampling makes the system non-deterministic.
- Numeric answers are compared against ground truth with a per-question tolerance defined in `eval/questions.yaml` together with the expected unit.
- Every run is saved as a JSONL trace (plan, tool calls, code, sandbox results, tokens, timings), so any number in this README can be traced back to raw runs.

### Metrics

| Metric | Definition |
|---|---|
| pass@1 | fraction of individual runs that return a correct answer |
| pass@k | probability that at least one of k runs is correct (unbiased estimator) |
| consistency | per-question fraction of runs that agree with the majority answer |
| variance across runs | spread of numeric answers across the 10 runs of a question |
| tool-call validation failure rate | tool calls whose arguments failed Pydantic validation, per call |
| sandbox policy violations | static and runtime policy events, by kind |
| sandbox limits hit | soft timeouts, hard kills, OOM kills, output truncations |
| verifier rejections | answers stopped by the verifier, split into correct and incorrect catches |
| latency | median and p90 wall clock per task |
| cost | median input and output tokens per task |

Reporting pass@1 alongside pass@k matters: pass@k rewards a system that is sometimes right, while a user usually gets one answer.

### Results

*Not yet measured.* Each published result will state the model, date, dataset checksum and git commit it was produced with.

### Failure analysis

*Not yet written.* `docs/failures.md` will categorize failed runs with example traces, for example: wrong aggregation, off-by-one date ranges, unit errors, NaN handling, plan/execution mismatch, and answers the verifier failed to catch.

---

## Quick start

Requirements: Python 3.11+, Docker.

```bash
git clone https://github.com/<your-username>/tsagent.git
cd tsagent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

docker build -t tsagent-sandbox:latest docker/sandbox
```

---

## Testing

```bash
pytest                 # everything; container tests are skipped if Docker or the image is missing
pytest -m docker       # container isolation tests only
pytest -m "not slow"   # skip memory-exhaustion and hard-kill tests
```

| File | Needs Docker | Tests |
|---|---|---|
| `tests/test_policy.py` | no | static check accepts normal analysis code, rejects known patterns, documents a known bypass |
| `tests/test_runner.py` | no | runner protocol, import guard, soft timeout, NaN flags, tracebacks, output capture |
| `tests/test_classify.py` | no | host-side classification of results, timeouts, OOM and crashes |
| `tests/test_data_dir_check.py` | no | data directory and preload file permission checks |
| `tests/test_sandbox.py` | yes | container isolation |

Most tests in `test_sandbox.py` **disable both Python-level policy layers** to simulate code that has already bypassed them, then check that the container alone blocks:

- network and DNS access
- writes to the dataset and to the root filesystem
- executing files from `/tmp`
- privilege changes (`setuid(0)`)
- thread and fork bombs
- memory exhaustion
- output floods written directly to the file descriptor

One test deliberately swallows the in-process timeout with `except BaseException:` to check that the host-side hard kill still ends the run.

### Bugs found by the isolation tests

The first run of the container tests on a real Docker host found two bugs that the Docker-free tests could not:

1. **Data not readable inside the container.** The container runs as UID 65534 (`nobody`), not as the host user. A data directory with owner-only permissions (`0700`, which pytest uses for temp dirs) made the preload fail with a bare `PermissionError` deep inside the container. Fix: `DockerSandbox` now checks up front that the mounted directory and preload file are readable by "others" and raises a `ValueError` with the exact `chmod` command. Regression tests: `tests/test_data_dir_check.py`.
2. **Result line lost after unterminated output.** Agent code writing to the file descriptor without a trailing newline glued its output to the front of the runner's result line, and the parser only looked for the marker at line starts, so a successful run was reported as `runner_crash`. Fix: the runner always starts the result on a new line, and the host searches for the last marker anywhere in the output. Regression test: `tests/test_classify.py`.

Both are the kind of bug that only appears against the real system, which is why the container tests exist.

---

## Development

One-time setup:

```bash
pip install -e ".[dev]" pre-commit mypy pip-audit
pre-commit install          # checks now run on every git commit
cp .env.example .env        # add your API key; .env is git-ignored
```

Checks before pushing (CI runs the same on every push):

| Check | Command | Catches |
|---|---|---|
| Tests | `pytest` | broken behaviour |
| Lint + security lint | `ruff check .` | bugs, insecure patterns (`S` rules) |
| Formatting | `ruff format --check .` | inconsistent style |
| Types | `mypy src` | type errors, e.g. unchecked `None` |
| Secrets | `pre-commit run gitleaks --all-files` | API keys, tokens, private keys |
| Dependencies | `pip-audit -r docker/sandbox/requirements.txt` | packages with known vulnerabilities |
| Everything above except tests | `pre-commit run --all-files` | |

CI (`.github/workflows/ci.yml`) additionally scans the full git history for secrets, builds the sandbox image, runs the Docker isolation tests, and scans the image for known CVEs with Trivy.

If a secret is ever committed, **revoke it at the provider immediately**. Deleting the commit is not enough: it stays in git history and in any clone or fork.

---

## Repository layout

```
├── docker
│   └── sandbox
│       ├── Dockerfile
│       ├── requirements.txt
│       └── runner.py
├── docs
│   └── threat_model.md
├── pyproject.toml
├── README.md
├── src
│   └── tsagent
│       ├── __init__.py
│       ├── __pycache__
│       │   └── __init__.cpython-312.pyc
│       └── sandbox
│           ├── docker_sandbox.py
│           ├── __init__.py
│           ├── models.py
│           ├── policy.py
│           └── __pycache__
│               ├── docker_sandbox.cpython-312.pyc
│               ├── __init__.cpython-312.pyc
│               ├── models.cpython-312.pyc
│               └── policy.cpython-312.pyc
└── tests
    ├── __pycache__
    │   ├── test_classify.cpython-312-pytest-9.1.1.pyc
    │   ├── test_data_dir_check.cpython-312-pytest-9.1.1.pyc
    │   ├── test_policy.cpython-312-pytest-9.1.1.pyc
    │   ├── test_runner.cpython-312-pytest-9.1.1.pyc
    │   └── test_sandbox.cpython-312-pytest-9.1.1.pyc
    ├── test_classify.py
    ├── test_data_dir_check.py
    ├── test_policy.py
    ├── test_runner.py
    └── test_sandbox.py
```

---

## Known limitations

These are stated deliberately rather than left for a reader to discover.

- **Docker shares the host kernel.** A kernel exploit would escape the container. gVisor (`runtime="runsc"`) or a microVM such as Firecracker reduces that risk.
- **The Python-level policy is bypassable.** It is a feedback and measurement tool, not a security control. `tests/test_policy.py` includes a known bypass as an explicit test.
- **The result nonce is recoverable** from inside the container. Agent code could forge its own result, but it can only lie about its own answer, which the verifier and ground truth still check.
- **Agent code can read everything inside the container**, including the runner and the dataset. Nothing secret is placed there; API keys never enter the container.
- **Cold start per execution** adds roughly 0.5–2 s. It is measured separately (`total_s` vs `exec_s`). A warm container pool is a possible optimization, not a current feature.
- **Stateless execution.** Each `run_python` call starts a fresh container, so variables do not persist between calls and generated code must be self-contained.
- **Docker isolation tests have not yet been run on macOS or Windows**, where Docker Desktop can differ from Linux in OOM reporting and bind mounts.

---

## Roadmap

### Achieved so far

- **Docker sandbox** (`src/tsagent/sandbox/`, `docker/sandbox/`)
  - Fresh container per execution: no network, read-only filesystem and data, noexec tmpfs, memory/CPU/PID limits, no capabilities, unprivileged user
  - Two timeouts: in-process soft timeout plus host-side hard kill
  - Bounded output reading, OOM detection, nonce-protected result protocol, explicit NaN/inf flags
  - Structured `SandboxResult` separating policy violations from resource limits
- **Static policy check and runtime import guard**, used for model feedback and metrics
- **Test suite**
  - 41 tests for policy, runner, result classification and configuration checks (no Docker needed)
  - 15 container isolation tests that disable the Python-level layers and check the container alone holds. First run on Linux: 13 passed, 2 failed and exposed real bugs (see [Bugs found by the isolation tests](#bugs-found-by-the-isolation-tests)); both fixed with regression tests
- **Threat model** (`docs/threat_model.md`), including accepted risks and known bypasses
- **Quality and security checks**: ruff (including security rules), mypy, pre-commit hooks, gitleaks secret scanning, pip-audit, and a GitHub Actions pipeline that also runs the Docker isolation tests and a Trivy image scan

### Next steps

1. **Tools**: Pydantic argument schemas for `describe_dataset`, `run_python` and `submit_answer`, used both as the LLM function definition and as validation, with validation failures counted
2. **Planner**: one LLM call returning ordered steps as a validated Pydantic model
3. **Verifier**: type, plausible range, unit and NaN checks before an answer is returned
4. **Dataset**: download script, frozen copy and SHA-256 checksum
5. **Question set**: about 50 questions with pandas ground truth, tolerances and units
6. **Evaluation harness**: 10 runs per question, JSONL traces, pass@1, pass@k, consistency, validation failure rate, sandbox events, latency and token cost
7. **First results and failure analysis** (`docs/failures.md`)

### Possible extensions (after the core is done)

- Stateful sandbox sessions (one container per task, variables persist between steps)
- Runs under gVisor (`runsc`) and a comparison of overhead
- Warm container pool to reduce cold-start latency

---

## License

MIT 