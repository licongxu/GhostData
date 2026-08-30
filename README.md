# GhostData

**Adversarial CI for ML data pipelines.**

Your tests only catch failures you encoded. GhostData searches for the next one: a plausible pipeline failure that **passes existing checks** and still **breaks the frozen model**. That counterexample is a **Ghost**.

Not a drift dashboard. Not Great Expectations. Not an LLM inventing scary stories.

```text
CSV + prompt
  → Codex proposes failure worlds   (pandas fallback if Codex is not logged in)
  → Daytona sandboxes measure them  (real checks + frozen model, then delete)
  → host ranks evidence
  → Ghost pack: transform.py · ghost_dataset.csv · model_report.json · regression_contract.py
```

Codex proposes. Daytona proves. The host never invents an AUC.

## Judge demo

Python 3.11+. Daytona key from [app.daytona.io](https://app.daytona.io) (prize eligibility requires Daytona). Codex login is optional.

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
