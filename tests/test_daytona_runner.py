import json
from types import SimpleNamespace

import pytest

from ghostdata.bundle import AgentOutput, AnalysisBundle, Claim
from ghostdata.execution.daytona import (
    DaytonaJob,
    DaytonaSettings,
    DaytonaVerificationRunner,
)
from ghostdata.verification import ExecutionEvidence, VerificationSpec


CLAIM = Claim("C001", "Metric is preserved", "model_metric_preservation")
BUNDLE = AnalysisBundle(
    "B001", "Verify", {"dataset": "data.csv"}, AgentOutput(), (CLAIM,)
)
SPEC = VerificationSpec("V001", "C001", "entity_alignment", "Misalign")


def valid_evidence(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "bundle_id": "B001",
        "verification_id": "V001",
        "claim_id": "C001",
        "experiment_type": "entity_alignment",
        "status": "completed",
        "observations": {"invariants": {"schema": True}},
        "artifact_paths": {"report": "evidence.json"},
        "error": None,
    }
    payload.update(overrides)
    return payload


class FakeFileSystem:
    def __init__(self, payload: bytes, files: dict[str, bytes] | None = None) -> None:
        self.payload = payload
        self.files = dict(files or {})
        self.uploads: dict[str, bytes] = {}
        self.downloads: list[str] = []
        self.fail_upload = False
        self.fail_download = False

    def upload_file(self, contents: bytes, path: str) -> None:
        if self.fail_upload:
            raise OSError("upload failed")
        self.uploads[path] = contents
        self.files[path] = contents

    def download_file(self, path: str) -> bytes:
        if self.fail_download:
            raise OSError("download failed")
        self.downloads.append(path)
        if path in self.files:
            return self.files[path]
        return self.payload


class FakeProcess:
    def __init__(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code
        self.calls: list[tuple[str, dict[str, object]]] = []

    def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append((command, kwargs))
        return SimpleNamespace(exit_code=self.exit_code, result="worker output")


class SessionProcess(FakeProcess):
    def __init__(self, exit_code: int = 0) -> None:
        super().__init__(exit_code)
        self.sessions: list[str] = []
        self.deleted_sessions: list[str] = []

    def create_session(self, name: str) -> None:
        self.sessions.append(name)

    def delete_session(self, name: str) -> None:
        self.deleted_sessions.append(name)


class FakeClient:
    def __init__(self, payload: bytes | None = None, exit_code: int = 0) -> None:
        self.sandbox = SimpleNamespace(
            id="sandbox-1",
            state="started",
            labels={
                "bundle_id": "B001",
                "claim_id": "C001",
                "verification_id": "V001",
                "experiment_type": "entity_alignment",
            },
            fs=FakeFileSystem(payload or json.dumps(valid_evidence()).encode()),
            process=FakeProcess(exit_code),
        )
        self.created_params: object | None = None
        self.create_timeout: int | None = None
        self.create_error: Exception | None = None
        self.deleted: list[object] = []
        self.delete_wait: list[bool] = []
        self.listed_query: object | None = None
        self.listed_sandboxes: list[object] = [self.sandbox]

    def create(self, params: object, timeout: int) -> SimpleNamespace:
        if self.create_error:
            raise self.create_error
        self.created_params = params
        self.create_timeout = timeout
        return self.sandbox

    def delete(self, sandbox: object, wait: bool = False) -> None:
        self.deleted.append(sandbox)
        self.delete_wait.append(wait)

    def list(self, query: object) -> list[object]:
        self.listed_query = query
        return self.listed_sandboxes


def runner(
    client: FakeClient,
    job: DaytonaJob | None = None,
    settings: DaytonaSettings | None = None,
) -> DaytonaVerificationRunner:
    return DaytonaVerificationRunner(
        lambda bundle, spec: job
        or DaytonaJob("python worker.py", {"worker/worker.py": b"pass"}),
        settings=settings,
        client=client,
    )


def test_daytona_runner_uploads_both_manifests_and_forwards_isolation() -> None:
    client = FakeClient()
    settings = DaytonaSettings(
        "ghostdata-runner", "/work/ghostdata", 91, 37, 8, True
    )

    result = runner(client, settings=settings).run(BUNDLE, SPEC)

    params = client.created_params
    assert isinstance(result, ExecutionEvidence)
    assert client.create_timeout == 91
    assert params.snapshot == "ghostdata-runner"
    assert params.ephemeral is True
    assert params.network_block_all is True
    assert params.auto_stop_interval == 8
    assert params.labels == {
        "project": "ghostdata",
        "role": "executor",
        "bundle_id": "B001",
        "claim_id": "C001",
        "verification_id": "V001",
        "experiment_type": "entity_alignment",
    }
    uploads = client.sandbox.fs.uploads
    assert uploads["/work/ghostdata/bundle.json"] == BUNDLE.to_json().encode()
    assert uploads["/work/ghostdata/verification.json"] == SPEC.to_json().encode()
    assert uploads["/work/ghostdata/worker/worker.py"] == b"pass"
    assert client.sandbox.process.calls[-1] == (
        "python worker.py",
        {"cwd": "/work/ghostdata", "timeout": 37},
    )
    assert client.sandbox.fs.downloads == ["/work/ghostdata/evidence.json"]
    assert client.deleted == [client.sandbox]
    assert client.delete_wait == [True]


@pytest.mark.parametrize(
    "payload,error",
    [
        (b"not-json", json.JSONDecodeError),
        (json.dumps({"bundle_id": "B001"}).encode(), TypeError),
        (json.dumps(valid_evidence(bundle_id="wrong")).encode(), ValueError),
        (json.dumps(valid_evidence(verification_id="wrong")).encode(), ValueError),
        (json.dumps(valid_evidence(claim_id="wrong")).encode(), ValueError),
        (json.dumps(valid_evidence(experiment_type="wrong")).encode(), ValueError),
    ],
)
def test_invalid_evidence_is_rejected_and_sandbox_deleted(
    payload: bytes, error: type[Exception]
) -> None:
    client = FakeClient(payload)

    with pytest.raises(error):
        runner(client).run(BUNDLE, SPEC)

    assert client.deleted == [client.sandbox]


@pytest.mark.parametrize("stage", ["upload", "download"])
def test_filesystem_failure_deletes_sandbox(stage: str) -> None:
    client = FakeClient()
    setattr(client.sandbox.fs, f"fail_{stage}", True)

    with pytest.raises(OSError, match=f"{stage} failed"):
        runner(client).run(BUNDLE, SPEC)

    assert client.deleted == [client.sandbox]


def test_command_failure_deletes_sandbox() -> None:
    client = FakeClient(exit_code=1)

    with pytest.raises(RuntimeError, match="exited 1"):
        runner(client).run(BUNDLE, SPEC)

    assert client.deleted == [client.sandbox]


def test_creation_and_job_factory_fail_before_cleanup_is_needed() -> None:
    client = FakeClient()
    client.create_error = RuntimeError("create failed")
    with pytest.raises(RuntimeError, match="create failed"):
        runner(client).run(BUNDLE, SPEC)
    assert client.deleted == []

    fresh = FakeClient()

    def invalid_job(bundle: AnalysisBundle, spec: VerificationSpec) -> DaytonaJob:
        raise ValueError("job failed")

    with pytest.raises(ValueError, match="job failed"):
        DaytonaVerificationRunner(invalid_job, client=fresh).run(BUNDLE, SPEC)
    assert fresh.created_params is None


@pytest.mark.parametrize(
    "job",
    [
        DaytonaJob("run", {"../escape.py": b"bad"}),
        DaytonaJob("run", {"/tmp/escape.py": b"bad"}),
        DaytonaJob("run", {"bundle.json": b"overwrite"}),
        DaytonaJob("run", {"verification.json": b"overwrite"}),
        DaytonaJob("run", evidence_path="../evidence.json"),
        DaytonaJob("run", evidence_path="."),
        DaytonaJob("run", download_paths={"artifact": "../ghost.csv"}),
    ],
)
def test_unsafe_or_reserved_paths_fail_before_sandbox_creation(job: DaytonaJob) -> None:
    client = FakeClient()

    with pytest.raises(ValueError, match="path|reserved"):
        runner(client, job).run(BUNDLE, SPEC)

    assert client.created_params is None
    assert client.deleted == []


def test_list_executions_maps_labels_and_filters_project() -> None:
    client = FakeClient()
    client.listed_sandboxes.append(SimpleNamespace(id="sandbox-2", state="stopped", labels={}))

    executions = runner(client).list_executions()

    assert executions[0] == {
        "sandbox_id": "sandbox-1",
        "state": "started",
        "bundle_id": "B001",
        "claim_id": "C001",
        "verification_id": "V001",
        "experiment_type": "entity_alignment",
        "agent_id": "",
        "discovery_id": "",
        "role": "",
    }
    assert executions[1]["verification_id"] == ""
    assert client.listed_query.labels == {"project": "ghostdata"}


def test_daytona_runner_downloads_artifacts_before_cleanup() -> None:
    client = FakeClient()
    client.sandbox.fs.payload = json.dumps(valid_evidence()).encode()
    stored: dict[str, bytes] = {}

    def sink(
        bundle: AnalysisBundle,
        spec: VerificationSpec,
        artifacts: dict[str, bytes],
    ) -> dict[str, str]:
        stored.update(artifacts)
        return {"dataset": "/safe/ghost.csv"}

    job = DaytonaJob(
        "run",
        download_paths={"dataset": "outputs/ghost.csv"},
    )
    result = DaytonaVerificationRunner(
        lambda bundle, spec: job,
        client=client,
        artifact_sink=sink,
    ).run(BUNDLE, SPEC)

    assert stored == {"dataset": client.sandbox.fs.payload}
    assert result.artifact_paths == {"dataset": "/safe/ghost.csv"}
    assert client.sandbox.fs.downloads == [
        "/home/daytona/ghostdata/evidence.json",
        "/home/daytona/ghostdata/outputs/ghost.csv",
    ]
    assert client.deleted == [client.sandbox]


def test_artifact_download_validation_and_sink_failure_cleanup() -> None:
    client = FakeClient()
    with pytest.raises(ValueError, match="artifact sink"):
        runner(
            client,
            DaytonaJob("run", download_paths={"dataset": "ghost.csv"}),
        ).run(BUNDLE, SPEC)
    assert client.created_params is None

    empty_role = FakeClient()
    with pytest.raises(ValueError, match="download roles"):
        DaytonaVerificationRunner(
            lambda bundle, spec: DaytonaJob(
                "run", download_paths={"": "ghost.csv"}
            ),
            client=empty_role,
            artifact_sink=lambda bundle, spec, artifacts: {},
        ).run(BUNDLE, SPEC)
    assert empty_role.created_params is None

    failing = FakeClient()

    def fail_sink(
        bundle: AnalysisBundle,
        spec: VerificationSpec,
        artifacts: dict[str, bytes],
    ) -> dict[str, str]:
        raise OSError("persist failed")

    with pytest.raises(OSError, match="persist failed"):
        DaytonaVerificationRunner(
            lambda bundle, spec: DaytonaJob(
                "run", download_paths={"dataset": "ghost.csv"}
            ),
            client=failing,
            artifact_sink=fail_sink,
        ).run(BUNDLE, SPEC)
    assert failing.deleted == [failing.sandbox]


def test_daytona_runner_adds_agent_and_discovery_labels() -> None:
    client = FakeClient()
    spec = VerificationSpec(
        "V001",
        "C001",
        "entity_alignment",
        "Misalign",
        {"agent_id": "agent-1", "discovery_id": "run-1"},
    )

    runner(client).run(BUNDLE, spec)

    assert client.created_params.labels["agent_id"] == "agent-1"
    assert client.created_params.labels["discovery_id"] == "run-1"


def test_daytona_runner_forwards_role_env_volume_and_network_override() -> None:
    client = FakeClient()
    client.volume = SimpleNamespace(
        get=lambda name, create=True: SimpleNamespace(id="vol-1")
    )
    job = DaytonaJob(
        "python worker.py",
        role="executor",
        extra_labels={"stage": "demo", "": "skip", "count": 1},
        env_vars={"GHOSTDATA": "1"},
        network_block_all=False,
    )
    settings = DaytonaSettings(volume_name="ghostdata-data", volume_mount="/data")

    runner(client, job, settings).run(BUNDLE, SPEC)

    params = client.created_params
    assert params.network_block_all is False
    assert params.env_vars == {"GHOSTDATA": "1"}
    assert params.labels["role"] == "executor"
    assert params.labels["stage"] == "demo"
    assert "" not in params.labels
    assert params.volumes[0].volume_id == "vol-1"
    assert params.volumes[0].mount_path == "/data"

    skipped = FakeClient()
    runner(skipped, settings=DaytonaSettings(volume_name="ghostdata-data")).run(
        BUNDLE, SPEC
    )
    assert skipped.created_params.volumes is None


def test_daytona_proposal_runner_inspects_table_and_deletes_sandbox() -> None:
    from ghostdata.execution.daytona import DaytonaProposalRunner

    spec = SPEC.to_dict()
    spec["origin"] = "sandbox_agent"
    analysis = {"ranked_features": ["income"], "inspected_columns": ["income", "label"]}
    work = "/home/daytona/ghostdata"
    client = FakeClient()
    client.sandbox.process = SessionProcess()
    client.sandbox.code_interpreter = SimpleNamespace(ran=[])
    client.sandbox.code_interpreter.run_code = (
        lambda code: client.sandbox.code_interpreter.ran.append(code)
    )
    client.sandbox.fs.files = {
        f"{work}/verification.json": json.dumps(spec).encode(),
        f"{work}/analysis.json": json.dumps(analysis).encode(),
    }

    specs, observed = DaytonaProposalRunner(client=client).propose(
        BUNDLE, {"proposer.py": b"print(1)"}, "label", "C001"
    )

    assert specs[0].origin == "sandbox_agent"
    assert observed == analysis
    assert client.created_params.labels["role"] == "proposer"
    assert client.sandbox.code_interpreter.ran == ["print('ghostdata-proposer-ready')"]
    assert client.sandbox.process.sessions == ["proposer"]
    assert client.sandbox.process.deleted_sessions == ["proposer"]
    assert client.deleted == [client.sandbox]


def test_daytona_proposal_runner_rejects_unsafe_paths_and_failed_exec() -> None:
    from ghostdata.execution.daytona import DaytonaProposalRunner

    client = FakeClient()
    with pytest.raises(ValueError, match="work directory"):
        DaytonaProposalRunner(client=client).propose(
            BUNDLE, {"../escape.py": b"bad"}, "label", "C001"
        )
    assert client.created_params is None

    failing = FakeClient(exit_code=1)
    with pytest.raises(RuntimeError, match="proposer exited 1"):
        DaytonaProposalRunner(client=failing).propose(
            BUNDLE, {"proposer.py": b"pass"}, "label", "C001"
        )
    assert failing.deleted == [failing.sandbox]
