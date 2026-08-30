from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.backend.main as backend
from app.backend.main import app


client = TestClient(app)


def test_health_and_verification_catalog() -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["content-type"].startswith("application/json")
    assert health.json() == {"status": "ok"}

    response = client.get("/api/verifications")

    assert response.status_code == 200
    specs = response.json()
    assert len(specs) >= 2
    assert all(item["experiment_type"] == "entity_alignment" for item in specs)
    assert all(item["origin"] == "sandbox_agent" for item in specs)
    assert {item["parameters"]["target_feature"] for item in specs} != {"MonthlyIncome"} or len(specs) >= 2
    assert "SeriousDlqin2yrs" not in {
        item["parameters"]["target_feature"] for item in specs
    }


def test_demo_bundle_endpoint_exposes_agent_claim_contract() -> None:
    response = client.get("/api/demo/bundle")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["bundle_id"] == "credit-preprocessing-demo"
    assert payload["claims"][0]["evaluator"] == "model_metric_preservation"


def test_demo_run_endpoint_executes_real_local_backend() -> None:
    response = client.post("/api/demo/run?backend=local")

    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "not_verified"
    assert payload["counterexamples"] >= 1
    assert payload["proposal"]["origin"] == "sandbox_agent"
    assert payload["proposal"]["experiment_type"] == "entity_alignment"
    assert payload["proposal"]["inspected_columns"]
    assert payload["proposal"]["executed_spec_count"] >= 2
    assert payload["artifacts"]
    assert set(payload["artifacts"]) == {
        "transform_code",
        "degraded_dataset",
        "model_report",
        "regression_contract",
    }
    dataset = client.get(payload["artifacts"]["degraded_dataset"])
    assert dataset.status_code == 200
    assert dataset.headers["content-type"].startswith("text/csv")
    assert "transform.py" not in dataset.text[:80]
    assert client.get("/api/demo/packs/missing/artifacts/model_report").status_code == 404
    assert client.get("/api/demo/packs/credit/artifacts/unknown").status_code == 404
    visuals = payload["visuals"]
    assert visuals["headline"] == "Same values. Different relationships."
    marginal = next(chart for chart in visuals["charts"] if chart["id"] == "marginal")
    assert marginal["reference"] == marginal["ghost"]
    label = next(chart for chart in visuals["charts"] if chart["id"] == "label")
    assert label["reference"] != label["ghost"]

    invalid = client.post("/api/demo/run?backend=unknown")
    assert invalid.status_code == 422


def test_publish_demo_pack_skips_incomplete_runs() -> None:
    spec = type("Spec", (), {"verification_id": "V001"})()
    empty = type("Report", (), {"ghosts": (), "evidence": ()})()
    assert backend._publish_demo_pack("credit", Path("x.csv"), "y", empty, spec, {}) is None
    missing = type(
        "Report",
        (),
        {"ghosts": (object(),), "evidence": ()},
    )()
    assert backend._publish_demo_pack("credit", Path("x.csv"), "y", missing, spec, {}) is None
    matched = type(
        "Report",
        (),
        {
            "ghosts": (object(),),
            "evidence": (type("Ev", (), {"verification_id": "V001"})(),),
        },
    )()
    with pytest.raises(ValueError, match="invalid pack id"):
        backend._publish_demo_pack("not valid", Path("x.csv"), "y", matched, spec, {})


def test_frontend_route_serves_demo_shell() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<h1>GhostData</h1>" in response.text
    assert "Find when a data agent is wrong." in response.text
    assert "Try it" in response.text
    assert "Example tables" in response.text
    assert 'href="#about"' in response.text
    assert "Architecture" in response.text
    assert "Analysis agent" in response.text
    assert "What happens in a run" in response.text
    assert "Measure" in response.text
    assert "Ghost dataset" in response.text
    assert "ghost_dataset.csv" in response.text
    assert "Model report" in response.text
    assert "Regression contract" in response.text
    assert "source is not shown" in response.text
    assert "Run on Daytona" in response.text
    assert 'id="upload"' in response.text
    assert 'id="prompt"' in response.text
    assert 'id="progress"' in response.text
    assert 'fetch("/api/runs"' in response.text
    assert "GHOST FOUND" in response.text
    assert "Frozen model" in response.text
    assert "Give Me Some Credit" in response.text
    assert "Credit Approval" in response.text
    assert "German Credit" in response.text


def test_discovery_endpoints_return_run_and_artifact_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"discovery_id": "run-1", "status": "completed"}
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(backend, "run_table_discovery", fake_run)
    monkeypatch.setattr(backend, "list_discovery_runs", lambda root: [report])
    monkeypatch.setattr(
        backend, "load_discovery_run", lambda discovery_id, root: report
    )
    files = {}
    for role, filename in backend.ARTIFACT_NAMES.items():
        path = tmp_path / filename
        path.write_text(f"artifact:{role}", encoding="utf-8")
        files[role] = path
    monkeypatch.setattr(
        backend,
        "discovery_artifact_path",
        lambda discovery_id, role, root: files[role],
    )

    created = client.post(
        "/api/discovery/runs?backend=local&dataset=debug&agents=2"
    )
    assert created.status_code == 200
    assert created.json()["artifacts"]["model_report"].endswith(
        "/artifacts/model_report"
    )
    assert calls[0]["data_path"] == backend.DEFAULT_DATA_PATH
    assert calls[0]["max_specs"] == 2
    assert calls[0]["label_column"] == backend.TARGET_COLUMN
    assert client.get("/api/discovery/runs").json()[0]["discovery_id"] == "run-1"
    assert client.get("/api/discovery/runs/run-1").json()["status"] == "completed"

    expected_types = {
        "transform_code": "text/x-python",
        "degraded_dataset": "text/csv",
        "model_report": "application/json",
        "regression_contract": "text/x-python",
    }
    for role, media_type in expected_types.items():
        response = client.get(f"/api/discovery/runs/run-1/artifacts/{role}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)
        assert response.text == f"artifact:{role}"


def test_discovery_endpoints_hide_internal_failures_and_return_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(**kwargs):
        raise RuntimeError("secret provider detail")

    monkeypatch.setattr(backend, "run_table_discovery", fail_run)
    response = client.post("/api/discovery/runs")
    assert response.status_code == 503
    assert response.json() == {"detail": "discovery failed: RuntimeError"}
    assert "secret" not in response.text
    assert client.post("/api/discovery/runs?agents=0").status_code == 422

    monkeypatch.setattr(
        backend,
        "load_discovery_run",
        lambda discovery_id, root: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert client.get("/api/discovery/runs/missing").status_code == 404
    monkeypatch.setattr(
        backend,
        "discovery_artifact_path",
        lambda discovery_id, role, root: (_ for _ in ()).throw(KeyError(role)),
    )
    assert (
        client.get("/api/discovery/runs/missing/artifacts/unknown").status_code
        == 404
    )


def test_demo_run_accepts_an_uploaded_table(tmp_path: Path) -> None:
    csv_path = tmp_path / "churn.csv"
    csv_path.write_text(
        "churned,tenure,spend\n"
        + "\n".join(
            f"{int(index >= 8)},{20 - index},{10 + index * 3}"
            for index in range(16)
        )
        + "\n",
        encoding="utf-8",
    )
    missing = client.post(
        "/api/demo/run?backend=local",
        files={"file": ("churn.csv", csv_path.read_bytes(), "text/csv")},
    )
    assert missing.status_code == 400

    response = client.post(
        "/api/demo/run?backend=local&label_column=churned",
        files={"file": ("churn.csv", csv_path.read_bytes(), "text/csv")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "not_verified"
    assert "churned" in payload["proposal"]["inspected_columns"]
    assert payload["proposal"]["target_feature"] != "churned"


def test_runs_endpoint_is_daytona_and_requires_a_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, str, bytes]] = []

    def fake_start(csv_bytes, prompt, filename, backend="daytona", analyst=None):
        calls.append((prompt, filename, backend, csv_bytes))
        return "run-live"

    monkeypatch.setattr(backend, "start_run", fake_start)
    empty = client.post("/api/runs", data={"prompt": "   ", "dataset": "credit"})
    assert empty.status_code == 400
    created = client.post(
        "/api/runs",
        data={
            "prompt": "Predict default. SeriousDlqin2yrs is the target.",
            "dataset": "credit",
        },
    )
    assert created.status_code == 200
    assert created.json() == {"run_id": "run-live", "status": "running"}
    assert calls[0][2] == "daytona"
    uploaded = client.post(
        "/api/runs",
        data={"prompt": "Predict churned"},
        files={"file": ("mine.csv", b"churned,x\n0,1\n1,2\n", "text/csv")},
    )
    assert uploaded.status_code == 200
    assert calls[-1][1] == "mine.csv"
    assert calls[-1][3].startswith(b"churned")
    assert client.get("/api/runs").status_code == 200
    assert client.get("/api/runs/missing").status_code == 404
    empty_file = client.post(
        "/api/runs",
        data={"prompt": "Predict y"},
        files={"file": ("empty.csv", b"  ", "text/csv")},
    )
    assert empty_file.status_code == 400
    monkeypatch.setattr(
        backend,
        "run_artifact_path",
        lambda run_id, role: Path(__file__),
    )
    artifact = client.get("/api/runs/run-live/artifacts/transform_code")
    assert artifact.status_code == 200
    monkeypatch.setattr(
        backend,
        "run_artifact_path",
        lambda run_id, role: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert client.get("/api/runs/run-live/artifacts/transform_code").status_code == 404


def test_demo_run_uses_the_credit_approval_fixture() -> None:
    response = client.post("/api/demo/run?backend=local&dataset=approval")
    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "not_verified"
    assert "class" in payload["proposal"]["inspected_columns"]
    assert payload["proposal"]["executed_spec_count"] >= 2


def test_demo_run_uses_the_german_credit_fixture() -> None:
    response = client.post("/api/demo/run?backend=local&dataset=german")
    assert response.status_code == 200
    payload = response.json()
    assert payload["verdict"] == "not_verified"
    assert "class" in payload["proposal"]["inspected_columns"]
