# GhostData

**Find the data-pipeline failure your tests already passed.**

GhostData searches for a plausible preprocessing failure that **passes existing checks** and still **breaks the frozen model**. That measured counterexample is a **Ghost**.

Not a drift dashboard. Not Great Expectations. Not an LLM inventing scary stories.

```mermaid
flowchart LR
  A["CSV + prompt"] --> B["Analyst<br/>proposes worlds"]
  B --> C["Daytona sandboxes<br/>checks + frozen model"]
  C --> D["Host ranks<br/>measured evidence"]
  D --> E["Ghost pack"]
```

A world becomes a Ghost only if checks stay green **and** the frozen-model metric drops. Codex proposes (pandas fallback if not logged in). Daytona proves. The host never invents an AUC.

Ghost pack: `transform.py` · `ghost_dataset.csv` · `model_report.json` · `regression_contract.py`

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
