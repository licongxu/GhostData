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
    assert response.json() == [
        {
            "verification_id": "V001",
            "claim_id": "C001",
            "experiment_type": "entity_alignment",
            "hypothesis": (
                "Valid MonthlyIncome values become attached to the wrong entities while "
                "the agent's stated invariants remain unchanged."
            ),
            "parameters": {
                "target_feature": "MonthlyIncome",
                "segment": {},
                "mismatch_fraction": 0.25,
                "seed": 7,
            },
            "expected_invariants": [
                "schema",
                "marginal_distribution",
                "missing_rate",
            ],
            "origin": "fixed_library",
        }
    ]


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
    assert payload["counterexamples"] == 1
    visuals = payload["visuals"]
    assert visuals["headline"] == "Same values. Different relationships."
    marginal = next(chart for chart in visuals["charts"] if chart["id"] == "marginal")
    assert marginal["reference"] == marginal["ghost"]
    label = next(chart for chart in visuals["charts"] if chart["id"] == "label")
    assert label["reference"] != label["ghost"]

    invalid = client.post("/api/demo/run?backend=unknown")
    assert invalid.status_code == 422


def test_frontend_route_serves_demo_shell() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<h1>GhostData</h1>" in response.text
    assert 'fetch("/api/verifications")' in response.text
    assert "Run in Daytona" in response.text
    assert "Same values. Different relationships." in response.text
    assert 'id="charts"' in response.text


def test_discovery_endpoints_return_run_and_artifact_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"discovery_id": "run-1", "status": "completed"}
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return report

    monkeypatch.setattr(backend, "run_credit_discovery", fake_run)
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
    assert len(calls[0]["profiles"]) == 2
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

    monkeypatch.setattr(backend, "run_credit_discovery", fail_run)
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
