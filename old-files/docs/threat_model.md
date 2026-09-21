# Sandbox threat model

## What we defend against

The primary adversary is **not a skilled human attacker**. It is:

1. **A model making mistakes**: infinite loops, loading everything into memory, printing a 2 GB DataFrame, `import requests` to "download fresher data".
2. **Instructions injected via data or questions**: text in the dataset or question that persuades the model to exfiltrate data or probe the host.
3. **Bugs in our own policy code.**

Goal: whatever the generated code does, it cannot reach the network, modify the data or host, or consume unbounded resources, and every such attempt is **recorded as a measurable event**.

## Layers

| Layer | Enforced by | Stops | Bypassable? |
|---|---|---|---|
| Static AST policy | host, before start | disallowed imports, reflection, `open`, `eval` | Yes, trivially (e.g. `pd.io.common.os`). Feedback + metrics only. |
| Runtime import guard | runner, in-process | imports from agent frames | Yes (same routes). Feedback + metrics only. |
| Soft timeout | runner, `SIGALRM` | runaway pure-Python loops | Yes: bare `except:`, long C calls |
| Hard kill | host wall clock | everything the soft timeout misses | No (short of the container runtime) |
| Output cap | host tail reader | output floods | No |
| Container | Docker + kernel | network, writes, privileges, memory, CPU, PIDs | Kernel/runtime exploits only; use gVisor (`runtime="runsc"`) to shrink that surface |

Tests in `tests/test_sandbox.py` disable both Python-level layers to check the container holds **alone**.

## Known limitations (deliberately not hidden)

- **Shared kernel.** Plain Docker is not a VM. gVisor or a microVM (Firecracker) is the step up.
- **Nonce is recoverable.** Agent code can in principle find the result nonce in runner memory and forge a result line. It can only lie about its *own* result, which the verifier and ground truth still check. Accepted risk.
- **Reading inside the container is allowed.** Agent code can read `/etc/passwd`, the runner source, and the dataset. None of these are secret. Never mount secrets or pass API keys into the container.
- **Default seccomp profile only.** A custom, tighter profile is possible but not in scope.
- **Cold start per execution** (~0.5–2 s incl. pandas import). Measured and reported separately (`total_s` vs `exec_s`).
