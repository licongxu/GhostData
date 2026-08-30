# GhostData

**Find the data-pipeline failure your tests already passed.**

GhostData searches for a plausible preprocessing failure that **passes existing checks** and still **breaks the frozen model**. That measured counterexample is a **Ghost**.

Not a drift dashboard. Not Great Expectations. Not an LLM inventing scary stories.

## Architecture

Codex proposes. Daytona proves. The host never invents an AUC.

```mermaid
flowchart TB
  U["Labeled CSV + prompt"] --> P["Analyst proposes worlds<br/>Codex, or pandas if no login"]

  P --> S1["Daytona sandbox"]
  P --> S2["Daytona sandbox"]
  P --> S3["Daytona sandbox"]

  S1 --> T["Apply transform"]
  S2 --> T
  S3 --> T

  T --> K{"Existing checks<br/>schema · marginals · missingness"}
  K -->|fail| X["Reject"]
  K -->|pass| M{"Frozen model"}

  M -->|metric drops| G["Ghost"]
  M -->|no drop| H["Harmless"]

  G --> Pack["Ghost pack<br/>transform.py · ghost_dataset.csv<br/>model_report.json · regression_contract.py"]
```

Each world runs in its own ephemeral Daytona sandbox with the real checks and the frozen model, then the sandbox is deleted. A Ghost is only declared from **measured** evidence: checks stay green **and** the model metric drops.

## Quick start

Python 3.11+. Daytona key from [app.daytona.io](https://app.daytona.io) (this demo uses Daytona sandboxes). Codex login is optional.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DAYTONA_API_KEY; leave DAYTONA_TARGET=eu
python scripts/ensure_snapshot.py
PYTHONPATH=src uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Upload a labeled CSV (or pick a fixture) and a one-line task, e.g. `Predict credit default; SeriousDlqin2yrs is the label.` Wait for a Ghost, then download the four artifacts.

Without Codex: `GHOSTDATA_ANALYST=deterministic` in `.env` uses the pandas proposer. With Codex: `codex login` (ChatGPT), then leave `GHOSTDATA_ANALYST=auto`.

## Other commands

```bash
pytest -q
python scripts/demo_local.py
python scripts/run_discovery.py --backend daytona --dataset debug --max-specs 2
python scripts/run_redteam.py --csv data/build/givemesomecredit_debug_3k.csv \
  --prompt "Predict default; SeriousDlqin2yrs is the label."
```

A leftover-sandbox count of `0` after a Daytona run is expected.

## Layout

| Path | Role |
|---|---|
| `app/` | FastAPI + static UI |
| `src/ghostdata/` | contracts, planner, Daytona/local runners, evaluators |
| `demo/pipeline/` | sandbox worker scripts |
| `data/` | fixtures (credit is a fixture, not the product) |
| `scripts/` | snapshot, demo, discovery, red-team |
