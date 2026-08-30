"""Thin HTTP surface for the hackathon demo."""

from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from ghostdata.demo import prepare_credit_demo
from ghostdata.demo.charts import attach_visuals
from ghostdata.demo.credit import DEFAULT_DATA_PATH, TARGET_COLUMN
from ghostdata.demo.table import run_table_demo
from ghostdata.tabular import load_table
from ghostdata.demo.artifacts import ARTIFACT_NAMES
from ghostdata.demo.discovery import (
    DEFAULT_AGENT_PROFILES,
    DEFAULT_OUTPUT_ROOT,
    FULL_DATA_PATH,
    discovery_artifact_path,
    list_discovery_runs,
    load_discovery_run,
    run_credit_discovery,
)


app = FastAPI(title="GhostData", version="0.1.0")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
DISCOVERY_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
DISCOVERY_DATA_PATHS = {"full": FULL_DATA_PATH, "debug": DEFAULT_DATA_PATH}


def _public_run(report: dict[str, object]) -> dict[str, object]:
    discovery_id = str(report["discovery_id"])
    return {
        **report,
        "artifacts": {
            role: f"/api/discovery/runs/{discovery_id}/artifacts/{role}"
            for role in ARTIFACT_NAMES
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/bundle")
def bundle() -> dict[str, object]:
    return prepare_credit_demo().bundle.to_dict()


@app.get("/api/verifications")
def verifications() -> list[dict[str, object]]:
    return [spec.to_dict() for spec in prepare_credit_demo().specs]


@app.post("/api/demo/run")
def run_demo(
    backend: Literal["local", "daytona"] = "local",
) -> dict[str, object]:
    report, spec, analysis = run_table_demo(
        DEFAULT_DATA_PATH, TARGET_COLUMN, backend
    )
    payload = attach_visuals(
        report, load_table(DEFAULT_DATA_PATH, TARGET_COLUMN), spec
    )
    payload["proposal"] = {
        "origin": spec.origin,
        "experiment_type": spec.experiment_type,
        "hypothesis": spec.hypothesis,
        "inspected_columns": list(analysis.get("inspected_columns") or []),
    }
    return payload


@app.post("/api/discovery/runs")
def discover(
    backend: Literal["local", "daytona"] = "local",
    dataset: Literal["full", "debug"] = "full",
    agents: Annotated[int, Query(ge=1, le=len(DEFAULT_AGENT_PROFILES))] = len(
        DEFAULT_AGENT_PROFILES
    ),
) -> dict[str, object]:
    try:
        report = run_credit_discovery(
            backend=backend,
            data_path=DISCOVERY_DATA_PATHS[dataset],
            output_root=DISCOVERY_OUTPUT_ROOT,
            profiles=DEFAULT_AGENT_PROFILES[:agents],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"discovery failed: {type(exc).__name__}",
        ) from exc
    return _public_run(report)


@app.get("/api/discovery/runs")
def discovery_runs() -> list[dict[str, object]]:
    return [
        _public_run(report) for report in list_discovery_runs(DISCOVERY_OUTPUT_ROOT)
    ]


@app.get("/api/discovery/runs/{discovery_id}")
def discovery_run(discovery_id: str) -> dict[str, object]:
    try:
        report = load_discovery_run(discovery_id, DISCOVERY_OUTPUT_ROOT)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="discovery not found") from exc
    return _public_run(report)


@app.get("/api/discovery/runs/{discovery_id}/artifacts/{role}")
def discovery_artifact(discovery_id: str, role: str) -> FileResponse:
    try:
        path = discovery_artifact_path(
            discovery_id, role, DISCOVERY_OUTPUT_ROOT
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    media_types = {
        "transform_code": "text/x-python",
        "degraded_dataset": "text/csv",
        "model_report": "application/json",
        "regression_contract": "text/x-python",
    }
    return FileResponse(path, media_type=media_types[role], filename=path.name)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND)
