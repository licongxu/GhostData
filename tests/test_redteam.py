import json
from pathlib import Path

import pandas as pd
import pytest

import ghostdata.demo.redteam as redteam
from ghostdata.demo.redteam import (
    AnalystOutput,
    ProposedWorld,
    stub_analyst,
    run_redteam,
    start_run,
    get_run,
    _parse_worlds,
    _world_spec,
)
from ghostdata.execution.local import default_compiler
from ghostdata.verification import VerificationSpec


def _churn_bytes() -> bytes:
    frame = pd.DataFrame(
        {
            "churned": [0, 0, 0, 0, 1, 1, 1, 1] * 8,
            "tenure": [20, 18, 19, 17, 1, 2, 0, 3] * 8,
            "spend": [90, 80, 85, 75, 15, 20, 10, 25] * 8,
        }
    )
    return frame.to_csv(index=False).encode()


def test_generated_transform_permutes_and_rejects_bad_source() -> None:
    frame = pd.DataFrame({"y": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
    source = (
        "import numpy as np\n"
        "def transform(dataframe):\n"
        "    out = dataframe.copy()\n"
        "    out['x'] = np.roll(out['x'].to_numpy(), 1)\n"
        "    return out\n"
    )
    spec = VerificationSpec(
        "W001",
        "C001",
        "generated_transform",
        "permute x",
        {"transform_source": source},
    )
    result = default_compiler().execute(frame, spec)
    assert result.affected_fraction > 0
    assert sorted(result.dataframe["x"]) == sorted(frame["x"])

    with pytest.raises(ValueError, match="missing a transform"):
        default_compiler().execute(
            frame,
            VerificationSpec("W002", "C001", "generated_transform", "bad", {}),
        )


def test_stub_analyst_and_local_redteam_find_a_ghost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="local-churn",
        backend="local",
        analyst=stub_analyst,
    )
    assert payload["status"] == "completed"
    assert payload["label_column"] == "churned"
    assert payload["ghosts"] >= 1
    assert payload["winner"]["hypothesis"]
    assert (tmp_path / "local-churn" / "transform.py").is_file()


def test_parse_worlds_requires_transform_functions() -> None:
    analysis = {
        "hypotheses": [
            {"world_id": "W001", "title": "A", "hypothesis": "h"},
        ]
    }
    with pytest.raises(RuntimeError, match="did not write"):
        _parse_worlds(analysis, {"analysis.json": "{}"})
    worlds = _parse_worlds(
        analysis,
        {
            "worlds/W001/transform.py": "def transform(dataframe):\n    return dataframe\n",
            "worlds/W001/hypothesis.json": json.dumps(
                {
                    "world_id": "W001",
                    "title": "A",
                    "hypothesis": "h",
                    "target_feature": "tenure",
                    "mismatch_fraction": 0.5,
                    "seed": 8,
                }
            ),
            "worlds/W002/transform.py": "print('no transform')\n",
            "worlds/W003/hypothesis.json": "not-json",
            "worlds/W003/transform.py": "def transform(dataframe):\n    return dataframe\n",
        },
    )
    assert worlds[0].world_id == "W001"
    assert "def transform" in worlds[0].transform_source
    assert {item.world_id for item in worlds} >= {"W001"}
    with pytest.raises(RuntimeError, match="missing transform"):
        _parse_worlds(
            {},
            {"worlds/W001/transform.py": "print('nope')\n"},
        )
    listed = _parse_worlds(
        {"hypotheses": ["skip-me"]},
        {
            "worlds/W001/transform.py": "def transform(dataframe):\n    return dataframe\n",
            "worlds/W001/hypothesis.json": "[1,2]",
        },
    )
    assert listed[0].hypothesis == "Executable preprocessing failure."


def test_leftover_count_uses_daytona_list(monkeypatch: pytest.MonkeyPatch) -> None:
    import daytona as daytona_mod

    monkeypatch.setattr(daytona_mod, "Daytona", lambda *args, **kwargs: object())
    monkeypatch.setattr(daytona_mod, "DaytonaConfig", lambda *args, **kwargs: object())
    monkeypatch.setattr(redteam, "leftover_sandboxes", lambda client: [])
    assert redteam._leftover_count() == 0


def test_start_run_records_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    run_id = start_run(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        backend="local",
        analyst=stub_analyst,
    )
    import time

    deadline = time.time() + 30
    payload = get_run(run_id)
    while payload["status"] == "running" and time.time() < deadline:
        time.sleep(0.1)
        payload = get_run(run_id)
    payload = get_run(run_id)
    assert payload["status"] == "completed"
    assert payload["report"]["ghosts"] >= 1
    assert any("Dataset loaded" in event["text"] for event in payload["events"])


def test_daytona_backend_uses_injected_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(redteam, "_leftover_count", lambda: 0)

    class FakeRunner:
        def __init__(self, job_factory, settings=None, client=None, artifact_sink=None) -> None:
            self.job_factory = job_factory
            self.artifact_sink = artifact_sink

        def list_executions(self) -> list[dict[str, str]]:
            return []

        def run(self, bundle, spec):
            from ghostdata.demo.artifacts import ARTIFACT_NAMES, build_ghost_artifacts
            from ghostdata.demo.table import build_local_runner
            import io

            job = self.job_factory(bundle, spec)
            assert job.network_block_all is True
            assert "dataset.csv" not in job.files
            assert "src/ghostdata" not in "".join(job.files)
            csv = job.volume_files.get("dataset.csv") or job.files.get("dataset.csv") or _churn_bytes()
            reference = pd.read_csv(io.BytesIO(csv))
            if job.extra_labels.get("world_id"):
                assert job.extra_labels["world_id"] == spec.verification_id
            evidence = build_local_runner(reference, "churned").run(bundle, spec)
            if job.download_paths:
                workspace = tmp_path / "fake-sandbox"
                workspace.mkdir(exist_ok=True)
                output = workspace / "outputs"
                payload = json.loads(job.files["discovery_report.json"])
                csv_path = tmp_path / "promote.csv"
                csv_path.write_bytes(csv)
                build_ghost_artifacts(
                    csv_path,
                    "churned",
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
                from dataclasses import replace as replace_evidence

                evidence = replace_evidence(evidence, artifact_paths=paths)
            return evidence

    monkeypatch.setattr(redteam, "DaytonaVerificationRunner", FakeRunner)
    monkeypatch.setattr(
        "ghostdata.demo.discovery.DaytonaVerificationRunner", FakeRunner
    )
    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="daytona-churn",
        backend="daytona",
        analyst=stub_analyst,
    )
    assert payload["backend"] == "daytona"
    assert payload["leftover_sandboxes"] == 0
    assert payload["ghosts"] >= 1


def test_get_run_unknown_id_and_artifact_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(FileNotFoundError):
        get_run("missing")
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    with pytest.raises(KeyError):
        redteam.run_artifact_path("x", "unknown")
    with pytest.raises(FileNotFoundError):
        redteam.run_artifact_path("x", "transform_code")
    assert isinstance(redteam.list_runs(), list)


def test_daytona_analyst_parses_sandbox_files(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(csv_bytes, prompt, filename, on_event, settings=None):
        del csv_bytes, filename, settings
        on_event({"kind": "analyst", "text": "exec python"})
        return {
            "sandbox_id": "sbx-1",
            "files": {
                "analysis.json": json.dumps(
                    {
                        "label_column": "churned",
                        "hypotheses": [
                            {"world_id": "W001", "title": "t", "hypothesis": prompt}
                        ],
                    }
                ),
                "worlds/W001/transform.py": "def transform(dataframe):\n    return dataframe\n",
            },
            "metrics": {"cpu": 0.4},
        }

    monkeypatch.setattr("demo.pipeline.analyst.run_sandbox_analyst", fake_run)
    import asyncio

    events: list[dict] = []
    output = asyncio.run(
        redteam.daytona_analyst(b"x", "Predict churned", "f.csv", events.append)
    )
    assert output.label_column == "churned"
    assert output.sandbox_id == "sbx-1"
    assert output.worlds[0].world_id == "W001"
    assert output.analysis["sandbox_metrics"]["cpu"] == 0.4


def test_generated_transform_rejects_non_dataframe() -> None:
    frame = pd.DataFrame({"y": [0, 1], "x": [1, 2]})
    spec = VerificationSpec(
        "W001",
        "C001",
        "generated_transform",
        "bad",
        {"transform_source": "def transform(dataframe):\n    return 1\n"},
    )
    with pytest.raises(ValueError, match="must return a DataFrame"):
        default_compiler().execute(frame, spec)
    renamed = VerificationSpec(
        "W002",
        "C001",
        "generated_transform",
        "rename",
        {
            "transform_source": (
                "def transform(dataframe):\n"
                "    return dataframe.rename(columns={'x': 'z'})\n"
            )
        },
    )
    result = default_compiler().execute(frame, renamed)
    assert result.affected_fraction == 1.0
    gone = VerificationSpec(
        "W003",
        "C001",
        "generated_transform",
        "gone",
        {"transform_source": "def transform(dataframe):\n    return dataframe\ntransform = 3\n"},
    )
    with pytest.raises(ValueError, match="must define transform"):
        default_compiler().execute(frame, gone)
    from ghostdata.demo.artifacts import _transform_source

    sourced = VerificationSpec(
        "W004",
        "C001",
        "generated_transform",
        "ok",
        {
            "transform_source": (
                "def transform(dataframe):\n    return dataframe\n\n"
                "if __name__ == \"__main__\":\n    pass\n"
            )
        },
    )
    assert "pass" in _transform_source(sourced)


def test_daytona_analyst_requires_analysis_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(csv_bytes, prompt, filename, on_event, settings=None):
        del csv_bytes, prompt, filename, on_event, settings
        return {"sandbox_id": "s", "files": {}}

    monkeypatch.setattr("demo.pipeline.analyst.run_sandbox_analyst", fake_run)
    import asyncio

    with pytest.raises(RuntimeError, match="analysis.json"):
        asyncio.run(redteam.daytona_analyst(b"x", "p", "f.csv", lambda event: None))

    async def bad_object(csv_bytes, prompt, filename, on_event, settings=None):
        del csv_bytes, prompt, filename, on_event, settings
        return {"sandbox_id": "s", "files": {"analysis.json": "[]"}}

    monkeypatch.setattr("demo.pipeline.analyst.run_sandbox_analyst", bad_object)
    with pytest.raises(RuntimeError, match="not an object"):
        asyncio.run(redteam.daytona_analyst(b"x", "p", "f.csv", lambda event: None))

    async def no_label(csv_bytes, prompt, filename, on_event, settings=None):
        del csv_bytes, prompt, filename, on_event, settings
        return {"sandbox_id": "s", "files": {"analysis.json": "{}"}}

    monkeypatch.setattr("demo.pipeline.analyst.run_sandbox_analyst", no_label)
    with pytest.raises(RuntimeError, match="label_column"):
        asyncio.run(redteam.daytona_analyst(b"x", "p", "f.csv", lambda event: None))


def test_stub_analyst_falls_back_when_prompt_omits_column_names() -> None:
    output = stub_analyst(_churn_bytes(), "Find pipeline failures.", "churn.csv")
    assert output.label_column in {"churned", "tenure", "spend"}


def test_duplicate_run_id_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="dup",
        backend="local",
        analyst=stub_analyst,
    )
    with pytest.raises(FileExistsError):
        run_redteam(
            _churn_bytes(),
            "Predict churn. churned is the target.",
            "churn.csv",
            run_id="dup",
            backend="local",
            analyst=stub_analyst,
        )
    path = redteam.run_artifact_path("dup", "transform_code")
    assert path.name == "transform.py"


def test_start_run_records_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("analyst exploded")

    monkeypatch.setattr(redteam, "run_redteam", boom)
    run_id = start_run(b"a,b\n1,2\n", "task", "x.csv", backend="local", analyst=stub_analyst)
    import time

    deadline = time.time() + 5
    payload = get_run(run_id)
    while payload["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
        payload = get_run(run_id)
    assert payload["status"] == "failed"
    assert "analyst exploded" in (payload["error"] or "")


def test_analyst_sandbox_uses_snapshot_volume_interpreter_session_and_code_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import daytona as daytona_mod
    from demo.pipeline.analyst import _run_python_analyst

    class FakeFS:
        def __init__(self) -> None:
            self.uploads: dict[str, bytes] = {}

        def upload_file(self, contents: bytes, path: str) -> None:
            self.uploads[path] = contents

        def download_file(self, path: str) -> bytes:
            if path.endswith("analysis.json"):
                return json.dumps({"label_column": "y", "hypotheses": []}).encode()
            return b""

    class FakeProcess:
        def __init__(self) -> None:
            self.sessions: list[str] = []
            self.deleted: list[str] = []
            self.calls: list[str] = []

        def exec(self, command: str, **kwargs: object) -> SimpleNamespace:
            self.calls.append(command)
            if command.startswith("find "):
                return SimpleNamespace(
                    exit_code=0,
                    result="/home/daytona/workspace/analysis.json\n",
                )
            return SimpleNamespace(exit_code=0, result="ghostdata-analyst: loaded 8 rows")

        def code_run(self, code: str, timeout: int | None = None) -> SimpleNamespace:
            del timeout
            self.calls.append("code_run")
            return SimpleNamespace(exit_code=0, result="ghostdata-chart-ready")

        def create_session(self, name: str) -> None:
            self.sessions.append(name)

        def delete_session(self, name: str) -> None:
            self.deleted.append(name)

        def execute_session_command(self, session_id: str, req: object, timeout: int | None = None) -> SimpleNamespace:
            del timeout
            command = getattr(req, "command", str(req))
            self.calls.append(command)
            return SimpleNamespace(
                exit_code=0,
                output="ghostdata-analyst: loaded 8 rows",
                stdout="ghostdata-analyst: loaded 8 rows",
                result="ghostdata-analyst: loaded 8 rows",
            )

    sandbox = SimpleNamespace(
        id="sbx-analyst",
        fs=FakeFS(),
        process=FakeProcess(),
        code_interpreter=SimpleNamespace(ran=[]),
        get_metrics_latest=lambda: SimpleNamespace(cpu=0.2, memory=64),
    )
    sandbox.code_interpreter.run_code = (
        lambda code: sandbox.code_interpreter.ran.append(code)
    )

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.volume = SimpleNamespace(
                get=lambda name, create=True: SimpleNamespace(id="vol-1")
            )
            self.snapshot = SimpleNamespace(
                get=lambda name: SimpleNamespace(name=name, state="active")
            )
            self.created_params = None
            self.deleted: list[object] = []

        def create(self, params: object, timeout: object = None) -> SimpleNamespace:
            del timeout
            self.created_params = params
            return sandbox

        def delete(self, item: object, wait: bool = False) -> None:
            del wait
            self.deleted.append(item)

    client = FakeClient()
    monkeypatch.setattr(daytona_mod, "Daytona", lambda *args, **kwargs: client)
    events: list[dict] = []
    payload = _run_python_analyst(
        b"y,x\n0,1\n1,2\n",
        "Predict y",
        "upload.csv",
        events.append,
    )
    assert payload["sandbox_id"] == "sbx-analyst"
    assert payload["metrics"]["cpu"] == 0.2
    assert client.created_params.snapshot == "ghostdata-runner"
    assert client.created_params.ephemeral is True
    assert client.created_params.network_block_all is True
    assert client.created_params.labels["role"] == "analyst"
    assert client.created_params.volumes[0].volume_id == "vol-1"
    assert "ghostdata-interpreter" in sandbox.code_interpreter.ran[0]
    assert sandbox.process.sessions == ["analyst"]
    assert sandbox.process.deleted == ["analyst"]
    assert "code_run" in sandbox.process.calls
    assert any("sandbox_analyst.py" in str(item) for item in sandbox.process.calls)
    assert client.deleted == [sandbox]


def test_proposal_from_payload_records_codex_planner() -> None:
    output = redteam._proposal_from_payload(
        {
            "sandbox_id": "codex",
            "planner": "codex_sdk",
            "metrics": {"cpu": 1},
            "daytona_charts": [{"title": "x"}],
            "files": {
                "analysis.json": json.dumps(
                    {"label_column": "churned", "hypotheses": []}
                ),
                "worlds/W001/transform.py": "def transform(dataframe):\n    return dataframe\n",
            },
        },
        "Predict churned",
        "churn.csv",
    )
    assert output.sandbox_id == "codex"
    assert output.analysis["planner"] == "codex_sdk"
    assert output.analysis["sandbox_metrics"]["cpu"] == 1
    assert output.analysis["daytona_charts"][0]["title"] == "x"
    assert output.label_column == "churned"
    with pytest.raises(RuntimeError, match="analysis.json"):
        redteam._proposal_from_payload({"files": []}, "p", "f.csv")


def test_local_redteam_uses_codex_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        "ghostdata.demo.codex_analyst.try_run_codex_analyst",
        lambda *args, **kwargs: {
            "sandbox_id": "codex",
            "planner": "codex_sdk",
            "files": {
                "analysis.json": json.dumps(
                    {
                        "label_column": "churned",
                        "hypotheses": [
                            {"world_id": "W001", "title": "t", "hypothesis": "permute"}
                        ],
                    }
                ),
                "worlds/W001/transform.py": (
                    "import numpy as np\n"
                    "FEATURES = ['tenure', 'spend']\n"
                    "def transform(dataframe):\n"
                    "    out = dataframe.copy()\n"
                    "    rng = np.random.default_rng(7)\n"
                    "    for name in FEATURES:\n"
                    "        out[name] = rng.permutation(out[name].to_numpy(copy=True))\n"
                    "    return out\n"
                ),
            },
        },
    )
    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="codex-local",
        backend="local",
    )
    assert payload["analyst_sandbox_id"] == "codex"
    assert payload["analysis"]["planner"] == "codex_sdk"
    assert payload["ghosts"] >= 1


def test_daytona_backend_without_injected_analyst(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(redteam, "_leftover_count", lambda: 0)

    async def fake_analyst(csv_bytes, prompt, filename, on_event, settings=None):
        del settings
        on_event({"kind": "analyst", "status": "running", "text": "codex"})
        return stub_analyst(csv_bytes, prompt, filename)

    monkeypatch.setattr(redteam, "daytona_analyst", fake_analyst)
    monkeypatch.setattr(
        redteam,
        "_verify_daytona",
        lambda csv_path, csv_bytes, label, bundle, specs, on_event, settings: redteam._verify_local(
            csv_bytes, label, bundle, specs
        ),
    )
    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="daytona-default-analyst",
        backend="daytona",
    )
    assert payload["backend"] == "daytona"
    assert payload["ghosts"] >= 1


def test_inconclusive_world_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)

    def exploding(csv_bytes: bytes, prompt: str, filename: str) -> AnalystOutput:
        del csv_bytes, prompt, filename
        return AnalystOutput(
            "churned",
            {"label_column": "churned"},
            [
                ProposedWorld(
                    "W001",
                    "boom",
                    "broken transform",
                    "def transform(dataframe):\n    raise ValueError('nope')\n",
                )
            ],
        )

    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="inconclusive",
        backend="local",
        analyst=exploding,
    )
    assert payload["ghosts"] == 0
    assert any(event.get("kind") == "rejected" for event in payload["events"])


def test_world_spec_uses_entity_alignment_when_a_target_feature_is_set() -> None:
    aligned = _world_spec(
        ProposedWorld(
            "W002",
            "misalign tenure",
            "break the join",
            "def transform(dataframe):\n    return dataframe\n",
            target_feature="tenure",
        ),
        "run-9",
    )
    assert aligned.experiment_type == "entity_alignment"
    assert aligned.parameters["target_feature"] == "tenure"
    assert aligned.parameters["mismatch_fraction"] == 0.5
    assert aligned.parameters["seed"] == 7
    generated = _world_spec(
        ProposedWorld("W001", "raw", "h", "def transform(dataframe):\n    return dataframe\n"),
        "run-9",
    )
    assert generated.experiment_type == "generated_transform"


def test_local_identity_world_is_not_promoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(redteam, "OUTPUT_ROOT", tmp_path)

    def identity(csv_bytes: bytes, prompt: str, filename: str) -> AnalystOutput:
        del csv_bytes, prompt, filename
        return AnalystOutput(
            "churned",
            {"label_column": "churned"},
            [
                ProposedWorld(
                    "W001",
                    "noop",
                    "identity",
                    "def transform(dataframe):\n    return dataframe\n",
                )
            ],
        )

    payload = run_redteam(
        _churn_bytes(),
        "Predict churn. churned is the target.",
        "churn.csv",
        run_id="no-ghost",
        backend="local",
        analyst=identity,
    )
    assert payload["ghosts"] == 0
    assert payload["winner"] is None
    assert payload["artifacts"] is None
    assert any("little or no drop" in event["text"] for event in payload["events"])
