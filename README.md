# Harness Project

An LLM agent that answers quantitative questions about a weather time series by writing Python and running it in a locked-down Docker sandbox, plus an evaluation harness that measures how often it is actually right.

The evaluation is the point of the project. Every question has a ground-truth answer computed directly with pandas, every question is run ten times because the system is non-deterministic, and the documentation reports what the agent gets wrong, not only what it gets right.

**Headline result** (56 questions x 10 runs = 560 runs, gpt-5-nano, agent loop without planner or verifier):

| pass@1 | pass@10 | consistency | median latency | median tokens | cost per question |
|---|---|---|---|---|---|
| **0.988** | 1.000 | 0.988 | 14.3 s | 3,549 in / 1,232 out | ~$0.0007 |

Full tables: [`docs/results.md`](docs/results.md) · failure-by-failure analysis: [`docs/failures.md`](docs/failures.md) · what the numbers do **not** show: [`docs/limitations.md`](docs/limitations.md) · sandbox threat model: [`docs/threat_model.md`](docs/threat_model.md).

---

## Contents

- [Harness Project](#harness-project)
  - [Contents](#contents)
  - [How it works](#how-it-works)
  - [Results](#results)
  - [Quick start](#quick-start)
    - [Choosing a model](#choosing-a-model)
  - [Using it](#using-it)
  - [The sandbox](#the-sandbox)
  - [Evaluation](#evaluation)
  - [Testing](#testing)
    - [Bugs the tests found](#bugs-the-tests-found)
  - [Development](#development)
  - [Repository layout](#repository-layout)
  - [Limitations](#limitations)
  - [Roadmap](#roadmap)
  - [License](#license)

---

## How it works

```
question
   │
   ▼
Planner (optional) ── one LLM call → ordered steps, validated as a Pydantic model
   │
   ▼
Agent loop ◄────────► Tools, each defined by one Pydantic model
   │                    ├─ describe_dataset   columns, units, time range
   │                    ├─ run_python ──────► Docker sandbox
   │                    └─ submit_answer      value + unit + method
   ▼
Verifier (optional) ── type, unit, plausible range, provenance, silent NaN
   │
   ▼
answer ──────────────► graded against pandas ground truth
```

**Tools.** Each tool is defined by one Pydantic model. The same model produces the JSON schema the LLM sees (strict mode) and validates the arguments the LLM sends back, so definition and check cannot drift apart. Failures are split into **schema level** (wrong fields or types, which the provider's strict mode should prevent) and **rule level** (empty code, NaN answer, length limits), and counted separately.

**Sandbox.** Generated code runs in a fresh, isolated container per call: no network, read-only filesystem, memory/CPU/PID limits, two independent timeouts.

**Planner.** One structured-output call turns the question into an interpretation, assumptions, an expected unit and ordered steps. An invalid plan is recorded and the agent continues without one: counted, never silently repaired.

**Verifier.** Deterministic Python, no LLM. Its strongest check is **provenance**: the submitted number must match a number that `run_python` actually produced in that run, at the precision the model wrote it. That enforces the rule "every number must come from code" and catches guessed values and conversions done "in the head".

A note on terminology: this is an orchestrated pipeline of LLM calls with deterministic components around it. It is not called "multi-agent", because only the planner and the loop are model-driven.

---

## Results

Measured with the agent loop alone (no planner, no verifier), gpt-5-nano, 56 questions, 10 runs each.

**pass@1 by category**

| category | pass@1 |
|---|---|
| aggregation | 1.000 |
| unit_conversion | 1.000 |
| resampling | 0.993 |
| counting | 0.988 |
| time_selection | 0.988 |
| not_answerable | 0.980 |
| multi_step | 0.925 |

**pass@1 by difficulty:** easy 0.993, medium 0.994, hard 0.950.

The pattern is consistent: simple aggregation and unit conversion are solved reliably; accuracy drops where a question chains two steps or hinges on a boundary condition.

**All 7 failures** (out of 560 runs):

| question | failed runs | what happened |
|---|---|---|
| `mean_temp_wettest_month_2023` | 2 / 10 | two-step question; identical error both times, so a systematic misreading rather than noise |
| `dry_days_2023` | 1 / 10 | `<=` used where the question says "below" 0.1 mm |
| `july_2023_precip` | 1 / 10 | off by 6.3 mm |
| `mean_temp_july_nights_2023` | 1 / 10 | hour window 22:00-04:00 interpreted slightly differently |
| `pressure_diff_jan_jul_2023` | 1 / 10 | off by 0.46 hPa |
| `berlin_temperature` | 1 / 10 | answered instead of declining a question the dataset cannot answer |

Each failure is shown in [`docs/failures.md`](docs/failures.md) with the exact Python the agent ran.

**A result about the evaluation itself.** The first evaluation reported pass@1 = 0.970. Ten of those seventeen "failures" were correct answers rejected by a bug in the grader: it treated the unit label `records` as an unknown physical unit, so `8760 records` was scored wrong for "how many hourly records". After fixing the grader and re-scoring the stored runs (`--regrade`, no new API calls), the true figure is 0.988. A grader is a measuring instrument and has to be validated like one.

**Cost.** The whole 560-run evaluation cost about **$0.40**. Prices change; check the provider's pricing page.

---

## Quick start

Requirements: Python 3.11+, Docker.

```bash
git clone <this repository>
cd <repository>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

docker build -t tsagent-sandbox:latest docker/sandbox
python -m tsagent.dataset          # download and freeze the dataset (~44k rows)
```

### Choosing a model

Configure in `.env` (copy `.env.example`; `.env` is never committed):

- **OpenAI** (paid per token): set `OPENAI_API_KEY` and `OPENAI_MODEL`. Set a monthly spending limit in the dashboard first.
- **Local model via Ollama** (free, much slower): measured at ~4 tokens/s on a laptop CPU, which is fine for development but impractical for a full evaluation.

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:4b
```

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3:4b
```

Only Ollama runs in Docker; the agent runs on the host because it starts sandbox containers itself. Ollama listens on `127.0.0.1` only, since it has no authentication.

---

## Using it

```bash
# one question
python -m tsagent.agent "What was the mean 2 m air temperature in 2023, in °C?"

# with planner and verifier, saving the full trace
python -m tsagent.agent "..." --planner --verifier --trace run.json

# the question set and its ground truth
python -m tsagent.questions
python -m tsagent.questions --check

# evaluation: plan and cost first, then run (resumable)
python -m tsagent.evaluate --dry-run
python -m tsagent.evaluate --configs loop --runs 10 --out eval/results/loop --traces

# free: re-score stored runs, and regenerate the documentation
python -m tsagent.evaluate --regrade eval/results/loop/runs.jsonl
python -m tsagent.analyze results  --runs eval/results/loop/runs.regraded.jsonl --out docs/results.md
python -m tsagent.analyze failures --runs eval/results/loop/runs.regraded.jsonl \
    --traces eval/results/loop/traces --out docs/failures.md
```

Example output:

```
question: What was the mean 2 m air temperature over the whole year 2023, in °C?
   1. describe_dataset  ok
   2. run_python        ok
   3. submit_answer     ok
verifier: accepted after 0 rejection(s)
answer:   11.313892694063927 °C
stats:    3 turns, 3 tool calls, 3354 in / 811 out tokens, 10.3s, model gpt-5-nano
```

---

## The sandbox

The agent writes code that nobody reviews before it runs. The sandbox ensures that whatever that code does, whether by mistake or through instructions injected via the data, it cannot reach the network, change the data or the host, or consume unbounded resources. Every blocked attempt is recorded, so the sandbox is also a source of evaluation data.

**The container is the security boundary. The Python-level checks are not.** Python cannot be sandboxed at the language level: `pd.io.common.os.system(...)` reaches the operating system with no `import` at all. The static check and the import guard exist to give the model precise feedback and to produce countable policy events.

| Layer | Where | Bypassable? |
|---|---|---|
| Static AST check | host, before start | yes; feedback and metrics only |
| Runtime import guard | inside container | yes; feedback and metrics only |
| Soft timeout (`SIGALRM`) | inside container | yes, via bare `except:` or long C calls |
| Hard kill (`docker kill`) | host | no |
| Output cap (tail only) | host | no |
| Container | Docker + kernel | only via kernel or runtime exploits |

Container settings: `--network none`, `--read-only`, read-only data mount, `--tmpfs /tmp:noexec,nosuid,nodev`, `--memory` with no swap, `--cpus`, `--pids-limit`, `--cap-drop ALL`, `--security-opt no-new-privileges`, `--user 65534:65534`, optional gVisor runtime. Code is passed over stdin, so no writable host directory is shared.

Each execution returns a structured `SandboxResult` that separates **policy violations** (the model tried to import `requests`) from **resource limits** (timeout, OOM, truncation), and execution time from total time, so container cold start does not distort latency numbers. NaN and infinity never serialize silently; they carry explicit flags.

**Data permissions.** The container runs as UID 65534, so the data directory must be readable by others (`chmod o+rx data && chmod o+r data/*.parquet`). This is checked at construction, with the exact command in the error message.

Full threat model and accepted risks: [`docs/threat_model.md`](docs/threat_model.md).

---

## Evaluation

**Data.** Hourly Open-Meteo reanalysis for one location, 2020-2024: 43,848 rows, stored in UTC (avoiding the missing and duplicated local hours at daylight-saving changes), validated before saving (every hour present exactly once, sorted, no unexpected timestamps) and frozen with a content checksum in `data/manifest.json`.

**Questions.** 56 questions, each precise enough to have exactly one correct answer, each with ground truth computed in pandas. Categories: aggregation, unit conversion, time selection, resampling, counting, multi-step, and questions the dataset **cannot** answer (no snow depth, no sunshine hours, another city, a year outside the data), where declining is the correct behaviour. Thirteen questions record the specific trap they test, such as "data is in km/h", "inclusive end date" or "resample before subtracting".

**Grading** is strict about substance and generous about form: an answer in an equivalent unit is converted first (2 m/s counts as 7.2 km/h), "July 2023" and "2023-07" both identify the month, counts may be labelled `records` or `hours`, but counts are never rescaled between hours and days.

**Metrics.** pass@1 and pass@k (unbiased estimator), consistency, spread of numeric answers, tool-call validation failures split by level, plan outcomes, sandbox violations and limits, verifier correct catches versus false rejections, latency and token cost. pass@1 is the honest headline: pass@k flatters a system that is only sometimes right.

**The harness** writes every finished run to JSONL immediately and skips completed runs when restarted, so an interruption never wastes runs already paid for. Runs execute in parallel, each in its own sandbox, and a crashed run is recorded as a failure instead of stopping the evaluation.

---

## Testing

```bash
pytest                 # everything except Docker and paid tests
pytest -m docker       # container isolation tests (needs the sandbox image)
pytest --live          # calls the real API and costs money; never runs otherwise
```

Over 300 automated tests. The ones that matter most:

| File | Needs Docker | Tests |
|---|---|---|
| `tests/test_sandbox.py` | yes | container isolation, with both Python-level layers disabled |
| `tests/test_policy.py`, `test_runner.py`, `test_classify.py` | no | static policy, runner protocol, result classification |
| `tests/test_tools.py` | no | strict schemas and a table of 23 valid/invalid tool calls |
| `tests/test_planner.py` | no | 15 plans, valid and invalid, each landing in exactly one status |
| `tests/test_verifier.py` | no | every check, plus loop integration |
| `tests/test_questions.py` | no | every ground truth computed on a synthetic dataset; trap questions must differ from the naive answer |
| `tests/test_grading.py`, `test_metrics.py` | no | unit conversion, text matching, pass@k against hand-computed values |
| `tests/test_evaluate.py` | no | resume, parallel safety, regrading, harness errors |

Most isolation tests **switch off both Python-level policy layers**, simulating code that has already bypassed them, and check that the container alone blocks network and DNS, writes to the data and the root filesystem, execution from `/tmp`, `setuid(0)`, thread and fork bombs, memory exhaustion and output floods. One test deliberately swallows the in-process timeout with `except BaseException:` to prove the host-side hard kill still ends the run.

### Bugs the tests found

1. **Data unreadable in the container.** The container runs as UID 65534, not as the host user, so an owner-only data directory failed with a bare `PermissionError` deep inside the container. Fixed with an upfront check that names the exact `chmod` command.
2. **Result line lost after unterminated output.** Output written to the file descriptor without a trailing newline glued itself to the front of the result line, and the parser only looked at line starts, so a successful run was reported as a crash. Fixed on both sides.
3. **Validation failures misclassified.** An invalid tool name in a plan was counted as a rule violation instead of a schema violation, which would have skewed the very metric the design separates. Caught by a table test over all plan error types.
4. **The grader's count-unit bug** described under [Results](#results), which understated accuracy by nearly two points.

Bugs 1 and 2 appear only against a real container; bug 4 appeared only against real model output. That is the argument for testing against the real system rather than only against mocks.

---

## Development

```bash
pip install -e ".[dev]" pre-commit mypy pip-audit
pre-commit install
cp .env.example .env
```

| Check | Command |
|---|---|
| Tests | `pytest` |
| Lint, including security rules | `ruff check .` |
| Formatting | `ruff format --check .` |
| Types | `mypy src` |
| Secrets | `pre-commit run gitleaks --all-files` |
| Dependency vulnerabilities | `pip-audit -r docker/sandbox/requirements.txt` |

CI (`.github/workflows/ci.yml`) runs all of these on every push, scans the full git history for secrets, builds the sandbox image, runs the Docker isolation tests, and scans the image with Trivy. The dependency audit already caught one real vulnerability in a pinned package.

If a secret is ever committed, **revoke it at the provider immediately**: deleting the commit is not enough.

---

## Repository layout

```
├── compose.yaml                  local model (Ollama) for development
├── docker/sandbox/               sandbox image: Dockerfile, pinned requirements, runner.py
├── docs/
│   ├── results.md                generated from the runs
│   ├── failures.md               generated from the runs and traces
│   ├── limitations.md            what the numbers do not show
│   └── threat_model.md           sandbox layers, accepted risks
├── eval/results/                 runs.jsonl, summary.json, traces
├── src/tsagent/
│   ├── sandbox/                  policy check, container lifecycle, result model
│   ├── dataset.py                download, validate, freeze, checksum
│   ├── schemas.py                Pydantic models for tools and plans
│   ├── tools.py                  tool registry, strict schemas, call accounting
│   ├── llm.py                    OpenAI Responses API wrapper, token/latency records
│   ├── agent.py                  the loop
│   ├── planner.py                structured-output planning
│   ├── verifier.py               deterministic answer checks
│   ├── questions.py              56 questions with pandas ground truth
│   ├── grading.py                answer comparison with unit conversion
│   ├── evaluate.py               evaluation harness (resumable, parallel)
│   ├── metrics.py                pass@k, consistency, rates
│   └── analyze.py                generates docs/results.md and docs/failures.md
└── tests/
```

---

## Limitations

The main ones, in full in [`docs/limitations.md`](docs/limitations.md):

- **One dataset, one location, one model.** Reanalysis data has no gaps, so the agent never has to handle missing values in the source.
- **Ten runs per question** gives a rough rate; 9/10 and 10/10 are not meaningfully different, and no confidence intervals are reported.
- **The grader checks the value, not the method.** A right answer reached by a wrong route is scored correct.
- **Questions and agent were written by the same person**, which risks unconsciously matching one to the other.
- **Docker shares the host kernel.** Isolation is strong against mistakes and prompt injection, not against a kernel exploit.
- **Only the plain agent loop has been evaluated.** The planner and verifier are implemented, tested and switchable, but the comparison across configurations has not been run.

---

## Roadmap

**Done:** dataset pipeline · tools with strict Pydantic schemas and two-level validation accounting · agent loop over the OpenAI Responses API with per-call token and latency records · planner with validated structured output · deterministic verifier · 56 questions with pandas ground truth · resumable parallel evaluation harness and metrics · generated results, failure analysis and limitations · Docker sandbox with layered isolation and per-layer tests · CI with secret scanning, dependency audit and image scanning.

**Next:**

1. Evaluate the `planner`, `verifier` and `full` configurations and compare them against the baseline (about $2 of API usage).
2. Harder questions: the current set is nearly saturated at 0.988, so differences between configurations cannot show up. Multi-step and ambiguous questions are where accuracy actually drops.
3. Compare models, including the local one, on the same questions.
4. Optional engineering: stateful sandbox sessions, gVisor runs, a warm container pool to cut the 0.5-2 s cold start.

---

## License

MIT