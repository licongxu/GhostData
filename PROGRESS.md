# GhostData Progress

Last updated: 2026-08-30

**Status: done for the hackathon.** Core product, Daytona isolation, Codex analyst, Ghost delivery, API, and tests are in. What remains is stretch, not a pitch blocker.

```text
CSV + prompt
  → Codex analyst (ChatGPT credits; pandas fallback if no login)
  → Daytona executor sandboxes (transform, existing checks, frozen model)
  → Evaluator ranks measured evidence
  → four Ghost artifacts
```

A Ghost is a measured counterexample: checks PASS, frozen-model metric drops.

## Shipped

- Table-agnostic loop. Credit is a fixture, not the product.
- Codex SDK proposer (`GHOSTDATA_ANALYST=auto|codex|deterministic`).
- Daytona: `ghostdata-runner` snapshot, `ghostdata-data` Volume, parallel ephemeral sandboxes, `network_block_all`, labels, `code_run` charts, delete-on-exit.
- Judge path `POST /api/runs`. Discovery and local demo endpoints.
- Ghost pack: `transform.py`, `ghost_dataset.csv`, `model_report.json`, `regression_contract.py`.
- 219 tests, 98.88% coverage.

## Stretch

Other evaluators; LLM discovery planner; queued jobs; durable run history.

## Pitch

```bash
source ~/envs/cmbagent_env/bin/activate
python scripts/ensure_snapshot.py
pytest -q
PYTHONPATH=src uvicorn app.backend.main:app --reload
```
