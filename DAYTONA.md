# Daytona in GhostData

Daytona is the **execution backend**, not a second chat agent.

GhostData's controller thinks (pick WorldSpecs, rank results, promote a Ghost). Daytona is the isolated computer that actually runs each candidate: transform → customer checks → frozen model → JSON result.

HackSprint London (30 Aug 2026): **prize eligibility requires using Daytona**. Outpost is the event host, not a Daytona product.

This repo is already wired: Python SDK **0.207.0**, region **eu**, `.env` has `DAYTONA_API_KEY`. Smoke test: `python scripts/daytona_smoke.py`.

Official docs: [daytona.io/docs](https://www.daytona.io/docs/en/). The old `daytonaio/daytona` GitHub repo is frozen. Trust the SDK and docs, not that repo.

---

## What Daytona is now (confirmed)

The old product was a human Dev Environment Manager (Codespaces-like). The current product is:

> Secure, elastic, programmable computers for AI-generated / agent-executed code.

A sandbox is a remote isolated machine (default: Linux container) with its own kernel, filesystem, network, CPU, RAM, and disk. Create, run, snapshot, delete via SDK/API/CLI/MCP.

That framing is correct. GhostData's spec already matches it:

```text
1 candidate failure = 1 Daytona sandbox
1 search round = N independently measurable sandbox runs
concurrency = 4–6, not 50
```

---

## Features the essay missed (or under-weighted)

These are real in current docs / SDK 0.207, beyond lifecycle + files + exec + git + LSP + VNC:

| Feature | What it is | GhostData? |
|---|---|---|
| **Declarative Image builder** | `Image.debian_slim().pip_install([...])` then `snapshot.create` — no Dockerfile upload | **Should** — bake the runner once |
| **Stateful `code_interpreter`** | Persistent Python REPL with isolated contexts; distinct from stateless `process.code_run` | **Should** — checks then model in one world |
| **Matplotlib chart artifacts** | `code_run` returns structured `artifacts.charts` (PNG + metadata) | **Should** — "same values, different relationships" plot |
| **Volumes** | FUSE mounts shared across sandboxes; `subpath` for isolation | **Should** — share reference data + frozen model |
| **Secrets (egress proxy)** | Sandbox only sees a placeholder; proxy injects the real secret into HTTPS headers for allow-listed hosts | Optional; only if a world calls an external API |
| **Network limits** | `network_block_all`, CIDR allow list, domain allow list, outbound proxy | **Should** — candidate code is untrusted |
| **Warm pools** | Pre-created running sandboxes from a snapshot for instant claim | **Do not depend** — `GET /api/warm-pools` 404'd on this account |
| **Linked sandboxes** | Parent/child share an internal DNS network | No — not a multi-service app |
| **Sessions + log streaming** | Background shells; stream logs of long jobs | Optional — evals are short |
| **PTY** | Interactive terminal (Claude Code / Devin inside a sandbox) | No — we are not nesting another coding agent |
| **Signed preview URLs** | Expose a port with a token | Optional — Ghost report HTML if time |
| **SSH access API** | `create_ssh_access` | Debug only, not the product |
| **Sandbox metrics** | `get_metrics` | Optional — "6 worlds running" |
| **Labels + list query** | Filter sandboxes by labels | **Should** — `world_id`, `failure_class` |
| **WebSocket state streaming** | SDK ≥ 0.198 waits on pushed state, not polling | Already default; do not set `DAYTONA_USE_DEPRECATED_POLLING` |
| **Archive** | Stopped **container** filesystem → object storage | Not needed for ephemeral worlds |
| **Fork / pause-resume** | **VM only** (Linux VM / Windows) | No — use containers |
| **GPU / Windows / Computer Use / VNC** | Desktops, CUDA, mouse/keyboard/screenshot | **Do not use** for this MVP |
| **Docker-in-Docker / k3s in sandbox** | Nested containers / cluster | No |
| **Spot** | Preemptible cheaper compute | Skip for a live demo |
| **OpenTelemetry** | SDK traces | Skip during the sprint |
| **MCP server** | `daytona mcp init cursor` — tools for another agent | Debug aid only. GhostData's runner is our SDK, not MCP |

`daytona-small` already includes pandas, numpy, scikit-learn, matplotlib, scipy. The smoke test ran sklearn + pandas inside it. That is enough to start; a custom snapshot is the judge-facing upgrade.

---

## What GhostData should actually use

Squeeze Daytona on **search + isolation + artifacts**, not by turning on every product surface.

### Must (P0 runner)

These prove the sentence judges need to hear:

> Each candidate is an executable production-failure hypothesis running with the real checks and frozen model in an isolated Daytona sandbox.

1. **`CreateSandboxFromSnapshotParams`**
   - `snapshot="ghostdata-runner"` (declarative Image builder; fallback upload path uses `daytona-small`)
   - `language="python"`
   - `ephemeral=True` (delete on stop)
   - `labels={"project":"ghostdata","world_id":"W023","failure_class":"entity_alignment"}`
   - `auto_stop_interval` short (e.g. 10–15 min) so a crash does not leak credits
2. **Filesystem**
   - Upload WorldSpec JSON, transform module, check suite, frozen model (or mount a volume)
   - Download `report.json`, optional parquet, plots
3. **`process.exec` and/or `process.code_run`**
   - Run the compiled candidate end-to-end
   - Read exit code; never trust stdout without it
4. **Result contract**
   - World returns compact JSON (checks pass/fail, baseline vs candidate AUC, damage, artifact paths)
   - Controller never pulls the full dataframe unless the world is a finalist
5. **`daytona.delete(sandbox)` in `finally`**
   - Ephemeral is a backstop, not a substitute for explicit delete
6. **Controlled concurrency 4–6**
   - Spec is explicit. Do not fan out 30 sandboxes.

Region: **`DAYTONA_TARGET=eu`**. Keep it. London latency + data stay in EU.

### Should (makes Daytona look load-bearing, not a hello-world)

Do these if P0 runner works. They are the "we actually used Daytona" features.

| GhostData need | Daytona feature |
|---|---|
| Same env every world; no `pip install` in the demo loop | **Custom snapshot** via declarative builder: pandas, numpy, scipy, scikit-learn, joblib, pyarrow, matplotlib |
| Don't re-upload the  reference table + `model.joblib` 4–6 times | **Volume** mounted at `/data` (reference parquet + frozen model). Worlds only write `/tmp/world-*` |
| Candidate transform is untrusted | **`network_block_all=True`** on worker sandboxes (eval is local files + sklearn, no internet) |
| Search UI: 31 searched / 6 running / W021 RUNNING | **Labels** + `daytona.list(ListSandboxesQuery(labels=...))` |
| "Same values. Different relationships." | **`code_run` chart artifacts** — histogram overlap + income×debt scatter |
| Checks then model share in-memory frames | **`code_interpreter.run_code`** in one sandbox, one context |
| Repro fixture for "Promote to regression test" | Download `ghost_XXXX.py` + `ghost_XXXX.parquet` via `fs.download_file` |

Declarative snapshot (build once, not per world):

```python
from daytona import CreateSnapshotParams, Image

image = (
    Image.debian_slim("3.12")
    .pip_install(["numpy", "pandas", "scikit-learn", "pyarrow", "matplotlib", "joblib"])
    .workdir("/home/daytona")
)
daytona.snapshot.create(CreateSnapshotParams(name="ghostdata-runner", image=image))
```

Build this **before** the demo, not during the 90 seconds. Creating a world from a snapshot is the fast path.

### Judge-visible extras (only after P0)

- **Preview URL** serving a static Ghost report (histograms + AUC) on port 8000. Nice; not required.
- **`refresh_activity()`** if a world eval might idle-stop (should not, if evals are seconds).
- Dashboard playground during Q&A: show 4 live sandboxes with GhostData labels.

### Do not use tomorrow

Not because they are fake — because they fight this product:

- **Computer Use / VNC / Windows / mouse / keyboard** — GhostData is CI for tabular ML, not a desktop agent.
- **GPU** — frozen sklearn/xgboost on CPU. GPU sandboxes are ephemeral and quota-tight.
- **Linux VM fork/pause** — slower, not needed; container + snapshot is the GhostData clone story.
- **MCP as the product** — judges should see *our* search controller calling the SDK. MCP is for you debugging in Cursor, not the demo.
- **Git/LSP inside the sandbox** — we ship a WorldSpec + deterministic transform, not "agent edits a repo until tests pass".
- **Linked sandboxes / DinD / k8s** — one world is one process graph, not a mesh.
- **Warm pools** — documented, SDK client exists, **hosted list endpoint 404'd on this key**. Do not bet the demo on them.
- **Wrapping Daytona as another LLM agent** — forbidden by our own spec.

---

## How a world runs (the only architecture)

```text
Controller (this laptop / FastAPI)
    |
    |  WorldSpec JSON
    v
Daytona sandbox  (ephemeral, labeled, eu, ghostdata-runner snapshot)
    |
    |  /data  volume: reference.parquet, model.joblib, checks.py
    |  /work  uploaded: worldspec.json, transform.py
    |
    |  code_interpreter / code_run
    |     1. apply transform
    |     2. run existing checks   -> must PASS or REJECT
    |     3. score frozen model     -> AUC / damage
    |     4. write report.json + optional charts
    |
    v
Controller downloads report.json  ->  rank / refine / Ghost
    |
    finally: delete sandbox
```

Controller verbs (map 1:1 to SDK):

| Verb | SDK |
|---|---|
| Open a computer | `daytona.create(CreateSandboxFromSnapshotParams(...))` |
| Put files in | `sandbox.fs.upload_file` / `upload_files` |
| Share heavy assets | `volumes=[VolumeMount(volume_id=..., mount_path="/data")]` |
| Run the candidate | `process.code_run` or `code_interpreter.run_code` |
| Run a shell if needed | `process.exec` |
| Read the result | `fs.download_file("/work/report.json")` |
| Read the plot | `response.artifacts.charts` or download PNG |
| Isolate the transform | `network_block_all=True` |
| List live worlds | `daytona.list(...)` by labels |
| Destroy | `daytona.delete(sandbox)` |

Do **not** add: `git.clone`, LSP, PTY, `computer_use`, preview, GPU, unless a later stretch goal needs it.

---

## Demo script ↔ Daytona

| Second | What they see | What Daytona is doing |
|---|---|---|
| 0–12s | 27/27 PASS | Nothing yet (local reference eval, or one completed sandbox) |
| 12–35s | Worlds W021–W024 RUNNING | 4–6 ephemeral sandboxes, labels on, search UI polling `list` |
| 35–55s | Ghost AUC drop | One sandbox's `report.json` — real numbers, not invented |
| 55–70s | Overlapping histograms | Chart artifact from that sandbox |
| 70–84s | Promote to regression test | Download transform + parquet from the winning sandbox (or from the result store after download) |
| 84–90s | Pitch line | Isolation + search, not "we used an LLM" |

Never fake the AUC. If the Ghost is weak, show a small real drop.

---

## Credits, quotas, failure modes

- Each builder gets **$100 Daytona credits**. Ephemeral + small snapshot + 4–6 concurrency is the budget.
- Default sandbox: **1 vCPU / 1GiB / 3GiB**. Stay on `daytona-small` or the custom snapshot's default. Do not request GPU.
- **Auto-stop is idle-based.** If a world does only in-sandbox work with no further API calls, it can still be stopped. Keep evals short; `refresh_activity()` only if something runs > a few minutes.
- **Stop ≠ pause.** We use ephemeral containers: stop ≈ gone. That is what we want.
- Candidate transform can hang. Pass a **timeout** to `code_run` / `exec`.
- If create is slow, the snapshot is cold. Pre-create `ghostdata-runner` and ping one sandbox before the pitch.

Known gap on this account (2026-08-29): **warm pools API 404**. Everything else in the smoke test passed: snapshot list, volume list, secrets list, ephemeral create, exec, code_run (sklearn), stateful interpreter, fs round-trip, sessions, delete.

---

## What not to tell judges

Do not say Daytona is our model, our checks, or our product UI.

Say:

> GhostData searches for a plausible pipeline failure that passes the customer's own tests. Each hypothesis runs in its own Daytona sandbox with the real checks and the frozen model, then dies. A Ghost is a measured counterexample, not a story.

---

## Pointers

- SDK install: `pip install daytona` into `~/envs/cmbagent_env`
- Config: `.env` (`DAYTONA_API_KEY`, `DAYTONA_API_URL`, `DAYTONA_TARGET=eu`)
- Live probe: `python scripts/daytona_smoke.py`
- Product spec: `GhostData_Hackathon_Short_Spec.md` §7, §12, §14
- Docs: [Sandboxes](https://www.daytona.io/docs/en/sandboxes/), [Snapshots](https://www.daytona.io/docs/en/snapshots/), [Volumes](https://www.daytona.io/docs/en/volumes/), [Declarative builder](https://www.daytona.io/docs/en/declarative-builder/), [Network limits](https://www.daytona.io/docs/en/network-limits/), [Secrets](https://www.daytona.io/docs/en/secrets/), [Code execution](https://www.daytona.io/docs/en/process-code-execution/)
