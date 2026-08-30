# GhostData Progress

Last updated: 2026-08-30

## Product direction

GhostData is an execution-based verification system for AI data-analysis agents.

An upstream agent submits an analysis bundle containing its inputs, code, artifacts,
metrics, claims, and tests. GhostData independently plans falsification experiments,
executes them in isolated environments, evaluates the resulting evidence, and returns
a `verified`, `not_verified`, or `inconclusive` verdict.

Credit data is a fixture. The proposer inspects any labeled table, fills a
`VerificationSpec`, and a separate executor applies that spec. The product is the
verification loop, not a credit-scoring script.

```text
labeled CSV
    -> proposer sandbox (inspect table, emit VerificationSpec)
    -> executor sandbox (transform, checks, frozen-model score)
    -> ExecutionEvidence
    -> Evaluator
    -> VerificationReport
```

A Ghost is defined as an executable counterexample to an AI agent's claim:
existing checks pass and the frozen model metric drops.

The discovery demo still uses four transparent simulated-agent profiles. They are
deterministic parameterized proposals, not LLM-generated code. Each proposal executes
in its own Daytona sandbox, and the strongest measured counterexample is rerun in a
separate promotion sandbox before client artifacts are published.

## Completed

### Core contracts

- `AnalysisBundle` standardizes agent inputs, outputs, claims, tests, and schema version.
- `Claim` selects an evaluator and carries evaluator-specific parameters.
- `VerificationSpec` describes a falsification experiment without embedding its verdict.
- `ExecutionEvidence` records execution facts only.
- `ExperimentVerdict`, `ClaimVerdict`, and `VerificationReport` separate interpretation
  from execution.
- Contract payloads validate required identities, reject non-JSON values, and protect
  caller-owned mappings from mutation.

### Planning

- Added a `VerificationPlanner` protocol.
- Added `KnownFailurePlanner`, the P0 fixed-library implementation used by unit tests
  and the discovery demo.
- Added `StructuredSpecPlanner`. It never writes transform code or Ghost CSVs. It
  profiles a labeled table and fills a `VerificationSpec` with `experiment_type`,
  `parameters`, `hypothesis`, `expected_invariants`, and `origin="sandbox_agent"`.
- The current operator library is `entity_alignment`. The proposer picks the numeric
  feature with the strongest absolute correlation to the label.
- If no LLM key is present, the sandbox still analyzes the table. The spec is data-driven,
  not a hardcoded `MonthlyIncome` experiment.
- Added a credit discovery planner with four simulated-agent profiles using mismatch
  fractions `0.10`, `0.25`, `0.50`, and `0.75` and stable verification identities.
- Counterexamples are ranked by independently measured AUC degradation, never by the
  requested mismatch fraction.

### Execution

- Added `VerificationRunner`, which returns raw evidence and cannot decide whether a
  claim is valid.
- Added `LocalVerificationRunner` for development and deterministic testing.
- Added a positional entity-alignment transform that supports duplicate DataFrame
  indexes while preserving marginal values.
- Added generic table loading, profiling, invariant checks, and frozen-feature scoring
  in `ghostdata.tabular`. Credit helpers remain only where the discovery fixture still
  needs them.
- Added `DaytonaVerificationRunner` with:
  - one ephemeral sandbox per verification experiment;
  - labels for project, role, bundle, claim, verification, and experiment identities;
  - `bundle.json` and `verification.json` uploads;
  - configurable snapshot, timeouts, network blocking, auto-stop, env vars, and optional
    Volume mount;
  - evidence download and strict identity validation;
  - sandbox deletion on every post-creation exit path;
  - unsafe path and reserved manifest protection.
- Added `DaytonaProposalRunner`. It creates a separate ephemeral sandbox with
  `role=proposer`, uploads the CSV and proposer script, optionally uses a code
  interpreter / process session when the SDK exposes them, downloads
  `verification.json` and `analysis.json`, then deletes the sandbox.
- The executor sandbox uses `role=executor` and `network_block_all`.
- The generic worker reads `task.json` for the label column and the spec for the
  target feature. It does not hardcode credit column names.
- Ran the complete 120,000-row credit dataset locally and in a live Daytona sandbox.
  Both produced identical measurements, and bounded polling confirmed sandbox cleanup.
- Added a discovery worker that fits a deterministic scikit-learn logistic regression
  model once per run and evaluates every candidate on the same stratified holdout.
- Ran four discovery sandboxes concurrently on the complete dataset, followed by one
  promotion sandbox for the winning counterexample.
- The sandbox runtime no longer imports the host-only Daytona adapter. This keeps the
  uploaded worker compatible with the Daytona snapshot's installed Python packages.

### Ghost delivery

- A promoted Ghost contains exactly four client artifacts:
  - `transform.py`: standalone deterministic reproduction code;
  - `ghost_dataset.csv`: the measured degraded 120,000-row dataset;
  - `model_report.json`: fitted-model configuration, real holdout AUCs, hashes,
    invariants, all agent outcomes, and winning evidence;
  - `regression_contract.py`: an executable contract that passes the reference dataset
    and exits nonzero for the promoted Ghost.
- Promotion happens in a fresh Daytona sandbox. All four files are downloaded before
  the ephemeral sandbox is deleted, then independently validated on the host.
- Publication is atomic: failed validation leaves no completed run directory.
- The verified live delivery is stored at
  `artifacts/discovery/full-daytona-demo/` and is exposed through
  `GET /api/discovery/runs/full-daytona-demo` plus its four artifact URLs.

### Evaluation and orchestration

- Added an evaluator plugin registry.
- Added `ModelMetricPreservationEvaluator` for the credit demo.
- Supports higher-is-better and lower-is-better metrics and explicit degradation
  tolerances.
- Failed execution, invalid evidence, missing evaluators, and violated experiment
  invariants produce `inconclusive`, never a false verification.
- Added concurrent orchestration with stable result ordering, individual experiment
  failure isolation, identity checking, and claim/report aggregation.

### Demo application

- Added a working FastAPI backend and static frontend.
- The demo path is table-agnostic. Credit is one fixture; German credit and a synthetic
  churn table use the same proposer and executor.
- `POST /api/demo/run` returns the verification report, generic chart payloads, and a
  `proposal` object (`origin`, experiment type, hypothesis, inspected columns).
- The frontend states that the agent proposed entity alignment after inspecting the
  table. Judge-facing copy does not name credit columns.
- Available endpoints:
  - `GET /health`
  - `GET /api/demo/bundle`
  - `GET /api/verifications`
  - `POST /api/demo/run?backend=local|daytona`
  - `POST /api/discovery/runs?backend=local|daytona&dataset=full|debug&agents=1..4`
  - `GET /api/discovery/runs`
  - `GET /api/discovery/runs/{discovery_id}`
  - `GET /api/discovery/runs/{discovery_id}/artifacts/{role}`
- Removed the old `/api/worlds` contract and the narrow `WorldSpec` / `CandidateResult`
  architecture.

### Learning notebook and environment

- Added `notebooks/01_real_credit_verification.ipynb` with executable explanations of
  bundle construction, planning, local execution, sandbox packaging, live Daytona
  execution, evidence evaluation, cleanup, and backend use.
- The notebook defaults to `data/build/givemesomecredit.csv` (120,000 rows) and has
  saved outputs from a successful live run.
- Installed and pinned JupyterLab, ipykernel, nbconvert, and nbformat in
  `~/envs/cmbagent_env`.
- Registered the `ghostdata` kernel as `Python (GhostData)`.

### Tests and verification

- 173 unit tests pass.
- Line coverage: 100%.
- Total coverage: 99.82% (three partial branches).
- The default `pytest` command enforces a minimum 98% coverage threshold.
- The suite covers bundle contracts, table profiling, structured-spec planning, local
  execution, Daytona proposer/executor isolation, evaluator boundaries, concurrent
  orchestration, API responses, credit as a fixture, and non-credit tables.

The notebook's original proxy-scorer run remains reproducible, but it is not described
as a trained-model result. The new full-dataset Daytona discovery uses a fitted logistic
model and produced:

```text
Verdict:        not_verified
Invariants:     3 / 3 pass
Discovery sandboxes: 4
Promotion sandboxes: 1
Winning agent:       relationship_hunter
Baseline AUC:        0.5680139229
Candidate AUC:       0.5079452497
Degradation:         0.0600686732
Affected rows:       86,215 / 120,000
Invariants:          3 / 3 pass
Remaining sandboxes: 0
```

On the debug credit fixture, the proposer now selects `NumberOfTimes90DaysLate`
(highest |corr| with the label), not the previously hardcoded `MonthlyIncome`.

## Not completed

- The custom `ghostdata-runner` snapshot is not built. Optional Volume mounting is
  wired, but the shared data/model Volume is not provisioned.
- Claim extraction is manual through `BundleClaimExtractor`; there is no LLM extractor.
- The proposer emits structured JSON from a table profile. There is no LLM-backed
  structured-output path yet because no model key is configured.
- Only model metric preservation has an evaluator. Schema, SQL correctness,
  reproducibility, relationships, subgroup metrics, and statistical evaluators are not
  implemented.
- Analysis bundle upload from external agents and durable database-backed verification
  history are not implemented. Completed discovery reports are currently read from the
  artifact directories.
- The frontend runs the example end to end, but it is not yet the planned Pipeline PR /
  Search / Ghost Found presentation flow.
- Discovery profiles are deterministic simulations. There is no LLM proposal agent on
  that path yet.
- The discovery endpoint is synchronous; queued execution and live progress events are
  not implemented yet.

## Next steps

1. Build the `ghostdata-runner` Daytona snapshot and shared data/model Volume so each
   run does not upload the package and dataset.
2. Move discovery to a queued backend job with live sandbox/progress status.
3. Build the three demo screens against the completed discovery and artifact APIs.
4. Connect an LLM structured-output proposer behind the existing `VerificationPlanner`
   boundary when a key is available.
5. Add the next evaluator only after the credit demo presentation flow is complete.

## Commands

```bash
source ~/envs/cmbagent_env/bin/activate

# Unit tests and enforced coverage
pytest -q

# Local measured verification demo
python scripts/demo_local.py

# Demo API and frontend
PYTHONPATH=src uvicorn app.backend.main:app --reload

# Learning notebook (select the Python (GhostData) kernel)
jupyter lab notebooks/01_real_credit_verification.ipynb

# Enable the notebook's live Daytona cell before launching Jupyter
GHOSTDATA_RUN_DAYTONA=1 jupyter lab notebooks/01_real_credit_verification.ipynb

# Four real Daytona discovery sandboxes + one promotion sandbox, full dataset
python scripts/run_discovery.py \
  --backend daytona \
  --dataset full \
  --agents 4

# Independently audit a published delivery
python scripts/verify_ghost_delivery.py \
  artifacts/discovery/<discovery_id> \
  --dataset full
```

## Design constraints to preserve

- Agents propose; execution produces evidence; evaluators decide.
- The proposer must not write transforms, Ghost CSVs, or verdicts.
- Daytona is the isolated execution substrate, not the intelligence layer.
- Execution evidence must never contain a precomputed verdict.
- An execution failure must never be counted as verification success.
- Credit is a fixture. Any labeled CSV must follow the same two-sandbox path.
