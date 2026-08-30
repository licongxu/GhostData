# GhostData Progress

Last updated: 2026-08-30

## Product direction

GhostData is an execution-based verification system for AI data-analysis agents.

An upstream agent submits an analysis bundle containing its inputs, code, artifacts,
metrics, claims, and tests. GhostData independently plans falsification experiments,
executes them in isolated environments, evaluates the resulting evidence, and returns
a `verified`, `not_verified`, or `inconclusive` verdict.

The credit preprocessing scenario remains the only demo. The product contracts are
general, but the implementation scope has not been expanded into a generic agent
platform.

```text
AI data agent
    -> AnalysisBundle
    -> Claim extraction
    -> Verification Planner
    -> VerificationSpec
    -> Local / Daytona execution
    -> ExecutionEvidence
    -> Evaluator
    -> VerificationReport
```

A Ghost is defined as an executable counterexample to an AI agent's claim.

The discovery demo now uses four transparent simulated-agent profiles. They are
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
- Added `KnownFailurePlanner`, the P0 fixed-library implementation.
- The current planner emits one deterministic entity-alignment experiment for a model
  metric preservation claim.
- Added a credit discovery planner with four simulated-agent profiles using mismatch
  fractions `0.10`, `0.25`, `0.50`, and `0.75` and stable verification identities.
- Counterexamples are ranked by independently measured AUC degradation, never by the
  requested mismatch fraction.
- Future AI or data-driven planners can implement the same protocol without changing
  execution or evaluation.

### Execution

- Added `VerificationRunner`, which returns raw evidence and cannot decide whether a
  claim is valid.
- Added `LocalVerificationRunner` for development and deterministic testing.
- Added a positional entity-alignment transform that supports duplicate DataFrame
  indexes while preserving marginal values.
- Added `DaytonaVerificationRunner` with:
  - one ephemeral sandbox per verification experiment;
  - labels for bundle, claim, verification, and experiment identities;
  - `bundle.json` and `verification.json` uploads;
  - configurable snapshot, timeouts, network blocking, and auto-stop;
  - evidence download and strict identity validation;
  - sandbox deletion on every post-creation exit path;
  - unsafe path and reserved manifest protection.
- Added the real credit worker bundle. It uploads the selected CSV, the checked-in
  worker, and the current `ghostdata` package into the isolated sandbox, then returns
  `evidence.json` to the host evaluator.
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

- Added a working FastAPI backend and static frontend for the credit example.
- Available endpoints:
  - `GET /health`
  - `GET /api/demo/bundle`
  - `GET /api/verifications`
  - `POST /api/demo/run?backend=local|daytona`
  - `POST /api/discovery/runs?backend=local|daytona&dataset=full|debug&agents=1..4`
  - `GET /api/discovery/runs`
  - `GET /api/discovery/runs/{discovery_id}`
  - `GET /api/discovery/runs/{discovery_id}/artifacts/{role}`
- Both frontend buttons were exercised against the running backend. The local and
  Daytona paths returned the expected counterexample.
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

- 153 unit tests pass.
- Line coverage: 100%.
- Branch coverage: 99.6%.
- The default `pytest` command enforces a minimum 98% coverage threshold.
- The suite covers bundle contracts, planning, local execution, Daytona cleanup and
  isolation, evaluator boundaries, concurrent orchestration, API responses, and the
  credit demo path.

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

## Not completed

- The custom `ghostdata-runner` snapshot and shared Daytona Volume are not built.
- Claim extraction is manual through `BundleClaimExtractor`; there is no LLM extractor.
- Planning uses one fixed-library experiment; there is no AI planner, parameter sweep,
  or iterative refinement.
- Only model metric preservation has an evaluator. Schema, SQL correctness,
  reproducibility, relationships, subgroup metrics, and statistical evaluators are not
  implemented.
- Analysis bundle upload from external agents and durable database-backed verification
  history are not implemented. Completed discovery reports are currently read from the
  artifact directories.
- The frontend runs the example end to end, but it is not yet the planned Pipeline PR /
  Search / Ghost Found presentation flow.
- Discovery profiles are deterministic simulations. There is no LLM proposal agent yet.
- The discovery endpoint is synchronous; queued execution and live progress events are
  not implemented yet.

## Next steps

1. Build the `ghostdata-runner` Daytona snapshot and shared data/model Volume so each
   run does not upload the package and dataset.
2. Move discovery to a queued backend job with live sandbox/progress status.
3. Build the three demo screens against the completed discovery and artifact APIs.
4. Connect an LLM proposal agent behind the existing `VerificationPlanner` boundary.
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
- Daytona is the isolated execution substrate, not the intelligence layer.
- Execution evidence must never contain a precomputed verdict.
- An execution failure must never be counted as verification success.
- Product contracts may be general, but the hackathon demo remains the single credit
  preprocessing scenario until it works live end to end.
