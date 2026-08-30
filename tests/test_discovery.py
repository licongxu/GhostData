import json
import shutil
from dataclasses import replace
from pathlib import Path
import pandas as pd
import pytest

import ghostdata.demo.discovery as discovery
from ghostdata.bundle import AgentOutput, AnalysisBundle, Claim
from ghostdata.demo.artifacts import (
    ARTIFACT_NAMES,
    build_credit_artifacts,
    validate_credit_artifacts,
)
from ghostdata.demo.credit import DEFAULT_DATA_PATH
from ghostdata.verification import (
    ClaimVerdict,
    ExecutionEvidence,
    ExperimentVerdict,
    VerificationReport,
    VerificationSpec,
)


@pytest.fixture(scope="module")
def published_run(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("published")
    report = discovery.run_credit_discovery(
        backend="local",
        data_path=DEFAULT_DATA_PATH,
        output_root=root,
        discovery_id="local-test",
    )
    return root, root / "local-test", report


def test_local_discovery_ranks_measured_damage_and_publishes_four_artifacts(
    published_run,
) -> None:
    root, run_dir, report = published_run

    assert report["status"] == "completed"
    assert report["label_column"] == "SeriousDlqin2yrs"
    assert report["winning_spec"]["origin"] == "sandbox_agent"
    assert report["winning_spec"]["parameters"]["target_feature"] != "SeriousDlqin2yrs"
    assert report["model"]["type"] == "sklearn.linear_model.LogisticRegression"
    assert len(report["model"]["features"]) > 1
    assert report["candidate_auc"] < report["baseline_auc"]
    assert len(report["agents"]) >= 2
    assert report["proposal"]["inspected_columns"]
    assert {path.name for path in run_dir.iterdir()} == set(ARTIFACT_NAMES.values())
    assert discovery.load_discovery_run("local-test", root) == report
    assert discovery.list_discovery_runs(root) == [report]
    assert discovery.discovery_artifact_path(
        "local-test", "degraded_dataset", root
    ) == run_dir / "ghost_dataset.csv"


def test_discovery_validation_and_empty_catalog(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="agent_id"):
        discovery.AgentProfile("", 0.5)
    with pytest.raises(ValueError, match="between"):
        discovery.AgentProfile("bad", 1.1)
    with pytest.raises(ValueError, match="at least one"):
        discovery.CreditDiscoveryPlanner((), "run", "local")
    duplicate = (
        discovery.AgentProfile("same", 0.2),
        discovery.AgentProfile("same", 0.3),
    )
    with pytest.raises(ValueError, match="unique"):
        discovery.CreditDiscoveryPlanner(duplicate, "run", "local")
    with pytest.raises(ValueError, match="unsafe"):
        discovery.prepare_credit_discovery(
            DEFAULT_DATA_PATH, discovery.DEFAULT_AGENT_PROFILES, "../bad", "local"
        )
    with pytest.raises(ValueError, match="unsafe"):
        discovery.run_table_discovery(
            DEFAULT_DATA_PATH,
            "SeriousDlqin2yrs",
            output_root=tmp_path,
            discovery_id="../bad",
        )
    with pytest.raises(ValueError, match="unsupported"):
        discovery.run_credit_discovery(
            "unknown", DEFAULT_DATA_PATH, tmp_path, discovery.DEFAULT_AGENT_PROFILES[:1]
        )

    assert discovery.list_discovery_runs(tmp_path / "missing") == []
    with pytest.raises(ValueError, match="unsafe"):
        discovery.load_discovery_run("../bad", tmp_path)
    with pytest.raises(FileNotFoundError):
        discovery.load_discovery_run("missing", tmp_path)
    with pytest.raises(KeyError):
        discovery.discovery_artifact_path("missing", "unknown", tmp_path)


def test_planner_ignores_unsupported_claim_and_rejects_external_claim() -> None:
    prepared = discovery.prepare_credit_discovery(
        DEFAULT_DATA_PATH,
        discovery.DEFAULT_AGENT_PROFILES[:1],
        "planner-test",
        "local",
    )
    external = Claim("outside", "x", "model_metric_preservation")
    with pytest.raises(ValueError, match="not part"):
        prepared.planner.propose(prepared.bundle, (external,))

    ignored = Claim("ignored", "x", "other_evaluator")
    bundle = AnalysisBundle("bundle", "task", {}, AgentOutput(), (ignored,))
    assert prepared.planner.propose(bundle, (ignored,)) == []


def test_discovery_without_counterexample_is_not_promoted(tmp_path: Path) -> None:
    from ghostdata.demo.table import FrozenSpecsPlanner
    from ghostdata.verification import VerificationSpec

    planner = FrozenSpecsPlanner(
        (
            VerificationSpec(
                "V001",
                "C001",
                "entity_alignment",
                "no damage",
                {
                    "target_feature": "MonthlyIncome",
                    "segment": {},
                    "mismatch_fraction": 0.0,
                    "seed": 7,
                    "agent_id": "noop",
                },
                ("schema", "marginal_distribution", "missing_rate"),
                "sandbox_agent",
            ),
        )
    )
    with pytest.raises(RuntimeError, match="without a promotable Ghost"):
        discovery.run_table_discovery(
            DEFAULT_DATA_PATH,
            "SeriousDlqin2yrs",
            backend="local",
            output_root=tmp_path,
            discovery_id="no-ghost",
            planner=planner,
        )
    assert not (tmp_path / "no-ghost").exists()


def test_winner_ignores_counterexample_without_numeric_damage() -> None:
    spec = discovery.prepare_credit_discovery(
        DEFAULT_DATA_PATH,
        discovery.DEFAULT_AGENT_PROFILES[:1],
        "invalid-damage",
        "local",
    ).specs[0]
    evidence = ExecutionEvidence(
        "credit-discovery-invalid-damage",
        spec.verification_id,
        spec.claim_id,
        spec.experiment_type,
        "completed",
    )
    report = VerificationReport(
        evidence.bundle_id,
        "not_verified",
        (
            ClaimVerdict(
                spec.claim_id,
                "not_verified",
                (
                    ExperimentVerdict(
                        spec.verification_id,
                        spec.claim_id,
                        "counterexample",
                        "invalid measurement",
                        {"degradation": "unknown"},
                    ),
                ),
            ),
        ),
        (evidence,),
    )
    with pytest.raises(RuntimeError, match="without a promotable Ghost"):
        discovery._winner(report, (spec,))


def test_daytona_discovery_and_promotion_use_separate_sandboxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []

    class FakeProposal:
        def __init__(self, settings=None, client=None) -> None:
            del settings, client

        def propose(self, bundle, files, label_column, claim_id, volume_files=None):
            del files, claim_id, volume_files
            from ghostdata.planner.agent import StructuredSpecPlanner
            from ghostdata.tabular import load_table

            planner = StructuredSpecPlanner(
                load_table(DEFAULT_DATA_PATH, label_column),
                label_column,
                max_specs=2,
            )
            return planner.propose(bundle, bundle.claims), dict(planner.last_analysis or {})

    class FakeDaytonaRunner:
        def __init__(
            self,
            job_factory,
            settings=None,
            client=None,
            artifact_sink=None,
        ) -> None:
            self.job_factory = job_factory
            self.artifact_sink = artifact_sink

        def run(self, bundle, spec):
            from ghostdata.demo.table import build_local_runner
            from ghostdata.tabular import load_table

            job = self.job_factory(bundle, spec)
            calls.append((spec.verification_id, bool(job.download_paths)))
            if "dataset.csv" in job.files:
                reference = pd.read_csv(__import__("io").BytesIO(job.files["dataset.csv"]))
                reference_path = DEFAULT_DATA_PATH
            else:
                reference = load_table(DEFAULT_DATA_PATH, "SeriousDlqin2yrs")
                reference_path = DEFAULT_DATA_PATH
            runner = build_local_runner(reference, "SeriousDlqin2yrs")
            evidence = runner.run(bundle, spec)
            if job.download_paths:
                workspace = tmp_path / "fake-sandbox"
                workspace.mkdir()
                output = workspace / "outputs"
                payload = json.loads(job.files["discovery_report.json"])
                from ghostdata.demo.artifacts import build_ghost_artifacts

                build_ghost_artifacts(
                    reference_path,
                    "SeriousDlqin2yrs",
                    bundle,
                    spec,
                    evidence,
                    payload,
                    output,
                )
                paths = self.artifact_sink(
                    bundle,
                    spec,
                    {
                        role: (output / filename).read_bytes()
                        for role, filename in ARTIFACT_NAMES.items()
                    },
                )
                evidence = replace(evidence, artifact_paths=paths)
            return evidence

    monkeypatch.setattr(discovery, "DaytonaProposalRunner", FakeProposal)
    monkeypatch.setattr(discovery, "DaytonaVerificationRunner", FakeDaytonaRunner)
    settings = discovery.DaytonaSettings(snapshot="daytona-small", volume_name=None)

    report = discovery.run_table_discovery(
        DEFAULT_DATA_PATH,
        "SeriousDlqin2yrs",
        "daytona",
        tmp_path,
        discovery_id="daytona-test",
        max_specs=2,
        daytona_settings=settings,
    )

    assert report["status"] == "completed"
    assert report["winning_spec"]["origin"] == "sandbox_agent"
    assert any(not download for _, download in calls)
    assert calls[-1][1] is True


def test_promotion_rejects_incomplete_artifact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = discovery.prepare_credit_discovery(
        DEFAULT_DATA_PATH,
        (discovery.AgentProfile("ghost", 0.75),),
        "bad-promotion",
        "daytona",
    )

    class BadPromotionRunner:
        def __init__(self, job_factory, settings=None, artifact_sink=None) -> None:
            self.artifact_sink = artifact_sink

        def run(self, bundle, spec):
            self.artifact_sink(bundle, spec, {"model_report": b"{}"})

    monkeypatch.setattr(discovery, "DaytonaVerificationRunner", BadPromotionRunner)
    with pytest.raises(ValueError, match="invalid artifact set"):
        discovery._promote_daytona(
            prepared.data_path,
            "SeriousDlqin2yrs",
            prepared.bundle,
            prepared.specs[0],
            {"agents": [], "verification_report": {}},
            tmp_path,
            None,
        )


def test_job_factory_contains_fitted_model_worker_and_package() -> None:
    prepared = discovery.prepare_credit_discovery(
        DEFAULT_DATA_PATH,
        discovery.DEFAULT_AGENT_PROFILES[:1],
        "job-test",
        "daytona",
    )
    job = discovery.build_discovery_job(
        prepared, prepared.bundle, prepared.specs[0]
    )

    assert job.files["worker.py"] == discovery.DISCOVERY_WORKER.read_bytes()
    assert job.files["dataset.csv"] == DEFAULT_DATA_PATH.read_bytes()
    assert "src/ghostdata/demo/discovery.py" in job.files


def _copy_delivery(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _rewrite_report(run_dir: Path, **updates: object) -> None:
    path = run_dir / ARTIFACT_NAMES["model_report"]
    report = json.loads(path.read_text())
    report.update(updates)
    path.write_text(json.dumps(report), encoding="utf-8")


def test_artifact_builder_rejects_nonempty_destination_and_bad_evidence(
    tmp_path: Path, published_run
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing").write_text("x")
    with pytest.raises(ValueError, match="empty"):
        build_credit_artifacts(DEFAULT_DATA_PATH, None, None, None, {}, occupied)

    _, _, report = published_run
    prepared = discovery.prepare_credit_discovery(
        DEFAULT_DATA_PATH,
        discovery.DEFAULT_AGENT_PROFILES,
        "local-test",
        "local",
    )
    spec = VerificationSpec.from_dict(report["winning_spec"])
    evidence = ExecutionEvidence.from_dict(report["winning_evidence"])
    bad_observations = dict(evidence.observations)
    bad_observations["model_metric"] = {
        "name": "roc_auc",
        "baseline": 0.0,
        "candidate": 0.0,
    }
    bad_evidence = replace(evidence, observations=bad_observations)
    with pytest.raises(ValueError, match="do not match"):
        build_credit_artifacts(
            DEFAULT_DATA_PATH,
            prepared.bundle,
            spec,
            bad_evidence,
            {
                "agents": report["agents"],
                "verification_report": report["verification_report"],
            },
            tmp_path / "bad-evidence",
        )


def test_artifact_validator_rejects_missing_schema_target_and_invariants(
    tmp_path: Path, published_run
) -> None:
    _, source, _ = published_run
    missing = _copy_delivery(source, tmp_path / "missing")
    (missing / "transform.py").unlink()
    with pytest.raises(ValueError, match="exactly four"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, missing)

    schema = _copy_delivery(source, tmp_path / "schema")
    frame = pd.read_csv(schema / "ghost_dataset.csv").iloc[:-1]
    frame.to_csv(schema / "ghost_dataset.csv", index=False)
    with pytest.raises(ValueError, match="schema and row count"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, schema)

    target = _copy_delivery(source, tmp_path / "target")
    frame = pd.read_csv(target / "ghost_dataset.csv")
    frame.loc[0, "SeriousDlqin2yrs"] = 1 - frame.loc[0, "SeriousDlqin2yrs"]
    frame.to_csv(target / "ghost_dataset.csv", index=False)
    with pytest.raises(ValueError, match="target labels"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, target)

    invariants = _copy_delivery(source, tmp_path / "invariants")
    frame = pd.read_csv(invariants / "ghost_dataset.csv")
    frame.loc[0, "MonthlyIncome"] = 999999999
    frame.to_csv(invariants / "ghost_dataset.csv", index=False)
    with pytest.raises(ValueError, match="invariants"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, invariants)


def test_artifact_validator_recomputes_hashes_auc_and_transform(
    tmp_path: Path, published_run
) -> None:
    _, source, _ = published_run
    source_hash = _copy_delivery(source, tmp_path / "source-hash")
    _rewrite_report(source_hash, source_dataset_sha256="bad")
    with pytest.raises(ValueError, match="source dataset hash"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, source_hash)

    ghost_hash = _copy_delivery(source, tmp_path / "ghost-hash")
    _rewrite_report(ghost_hash, ghost_dataset_sha256="bad")
    with pytest.raises(ValueError, match="Ghost dataset hash"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, ghost_hash)

    auc = _copy_delivery(source, tmp_path / "auc")
    _rewrite_report(auc, baseline_auc=0.0)
    with pytest.raises(ValueError, match="AUC recomputation"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, auc)

    broken = _copy_delivery(source, tmp_path / "broken-transform")
    (broken / "transform.py").write_text("raise RuntimeError('broken')")
    with pytest.raises(ValueError, match="transform artifact failed"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, broken)

    mismatch = _copy_delivery(source, tmp_path / "mismatch-transform")
    (mismatch / "transform.py").write_text(
        "import pandas as pd, sys\n"
        "pd.read_csv(sys.argv[1]).to_csv(sys.argv[2], index=False)\n"
    )
    with pytest.raises(ValueError, match="does not reproduce"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, mismatch)


def test_artifact_validator_executes_regression_contract(
    tmp_path: Path, published_run
) -> None:
    _, source, _ = published_run
    run_dir = _copy_delivery(source, tmp_path / "contract")
    (run_dir / "regression_contract.py").write_text("raise SystemExit(0)")
    with pytest.raises(ValueError, match="must pass reference and reject"):
        validate_credit_artifacts(DEFAULT_DATA_PATH, run_dir)


def test_discovery_does_not_overwrite_and_cleans_failed_publication(
    tmp_path: Path, published_run, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = published_run
    with pytest.raises(FileExistsError, match="already exists"):
        discovery.run_credit_discovery(
            "local",
            DEFAULT_DATA_PATH,
            root,
            discovery.DEFAULT_AGENT_PROFILES,
            "local-test",
        )

    def fail_build(*args, **kwargs):
        raise OSError("publication failed")

    monkeypatch.setattr(discovery, "_promote_local", fail_build)
    with pytest.raises(OSError, match="publication failed"):
        discovery.run_credit_discovery(
            "local",
            DEFAULT_DATA_PATH,
            tmp_path,
            (discovery.AgentProfile("ghost", 0.75),),
            "failed-publication",
        )
    assert list(tmp_path.iterdir()) == []
