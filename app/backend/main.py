"""Thin HTTP surface for the hackathon demo."""

import asyncio
import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from ghostdata.demo import prepare_credit_demo
from ghostdata.demo.charts import attach_visuals
from ghostdata.demo.credit import DEFAULT_DATA_PATH, TARGET_COLUMN
from ghostdata.demo.table import build_table_bundle, run_table_demo
from ghostdata.tabular import load_table
from ghostdata.demo.artifacts import ARTIFACT_NAMES, build_ghost_artifacts
from ghostdata.demo.discovery import (
    DEFAULT_OUTPUT_ROOT,
    FULL_DATA_PATH,
    discovery_artifact_path,
    list_discovery_runs,
    load_discovery_run,
    run_table_discovery,
)
from ghostdata.demo.redteam import (
    get_run,
    list_runs,
    run_artifact_path,
    start_run,
)


app = FastAPI(title="GhostData", version="0.1.0")
FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
DISCOVERY_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
DEMO_PACK_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "demo"
PACK_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
APPROVAL_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "live" / "credit_approval.csv"
GERMAN_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "build" / "german_credit.csv"
FIXTURES = {
    "credit": (DEFAULT_DATA_PATH, TARGET_COLUMN),
    "approval": (APPROVAL_DATA_PATH, "class"),
    "german": (GERMAN_DATA_PATH, "class"),
}


def _pack_urls(pack_id: str) -> dict[str, str]:
    return {
        role: f"/api/demo/packs/{pack_id}/artifacts/{role}" for role in ARTIFACT_NAMES
    }


def _pack_ready(pack_id: str) -> bool:
    root = DEMO_PACK_ROOT / pack_id
    return all((root / filename).is_file() for filename in ARTIFACT_NAMES.values())


def _publish_demo_pack(
    pack_id: str,
    data_path: Path,
    label_column: str,
    report: object,
    spec: object,
    analysis: dict[str, object],
) -> dict[str, str] | None:
    ghosts = getattr(report, "ghosts", ()) or ()
    if not ghosts:
        return None
    evidence = next(
        (
            item
            for item in getattr(report, "evidence", ())
            if item.verification_id == spec.verification_id
        ),
        None,
    )
    if evidence is None:
        return None
    if not PACK_ID_RE.fullmatch(pack_id):
        raise ValueError("invalid pack id")
    if _pack_ready(pack_id):
        return _pack_urls(pack_id)
    destination = DEMO_PACK_ROOT / pack_id
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    reference = load_table(data_path, label_column)
    bundle, _, _ = build_table_bundle(
        reference,
        label_column,
        f"demo-pack-{pack_id}",
        "Verify an agent-generated preprocessing change.",
    )
    build_ghost_artifacts(
        data_path,
        label_column,
        bundle,
        spec,
        evidence,
        {
            "proposal": analysis,
            "verification_report": report.to_dict(),
        },
        destination,
    )
    return _pack_urls(pack_id)


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


@app.post("/api/runs")
async def create_redteam_run(
    prompt: str = Form(...),
    dataset: Literal["credit", "approval", "german"] = Form("credit"),
    file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    text = prompt.strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt is required")
    if file is not None and file.filename:
        csv_bytes = await file.read()
        filename = file.filename
        if not csv_bytes.strip():
            raise HTTPException(status_code=400, detail="uploaded CSV is empty")
    else:
        path, _label = FIXTURES[dataset]
        csv_bytes = path.read_bytes()
        filename = path.name
    run_id = start_run(csv_bytes, text, filename, backend="daytona")
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs")
def redteam_runs() -> list[dict[str, object]]:
    return list_runs()


@app.get("/api/runs/{run_id}")
def redteam_run(run_id: str) -> dict[str, object]:
    try:
        return get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


@app.get("/api/runs/{run_id}/artifacts/{role}")
def redteam_artifact(run_id: str, role: str) -> FileResponse:
    try:
        path = run_artifact_path(run_id, role)
    except (FileNotFoundError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    media_types = {
        "transform_code": "text/x-python",
        "degraded_dataset": "text/csv",
        "model_report": "application/json",
        "regression_contract": "text/x-python",
    }
    return FileResponse(path, media_type=media_types[role], filename=path.name)


def _execute_demo(
    backend: Literal["local", "daytona"],
    dataset: Literal["credit", "approval", "german"],
    label_column: str | None,
    csv_bytes: bytes | None,
    filename: str | None,
) -> dict[str, object]:
    if csv_bytes is not None and filename:
        if not label_column or not label_column.strip():
            raise ValueError("label_column is required for uploads")
        suffix = Path(filename).suffix or ".csv"
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / f"upload{suffix}"
            path.write_bytes(csv_bytes)
            column = label_column.strip()
            report, spec, analysis = run_table_demo(path, column, backend)
            payload = attach_visuals(report, load_table(path, column), spec)
            pack_id = uuid4().hex[:12]
            payload["artifacts"] = _publish_demo_pack(
                pack_id, path, column, report, spec, analysis
            )
    else:
        data_path, fixture_label = FIXTURES[dataset]
        column = (label_column or fixture_label).strip()
        report, spec, analysis = run_table_demo(
            data_path, column, backend, max_specs=2
        )
        payload = attach_visuals(report, load_table(data_path, column), spec)
        payload["artifacts"] = _publish_demo_pack(
            dataset, data_path, column, report, spec, analysis
        )
    payload["proposal"] = {
        "origin": spec.origin,
        "experiment_type": spec.experiment_type,
        "hypothesis": spec.hypothesis,
        "target_feature": spec.parameters.get("target_feature"),
        "inspected_columns": list(analysis.get("inspected_columns") or []),
        "hypotheses": list(analysis.get("hypotheses") or []),
        "executed_spec_count": analysis.get("executed_spec_count"),
        "winning_verification_id": analysis.get("winning_verification_id"),
    }
    return payload


@app.post("/api/demo/run")
async def run_demo(
    backend: Literal["local", "daytona"] = "local",
    dataset: Literal["credit", "approval", "german"] = "credit",
    label_column: str | None = None,
    file: UploadFile | None = File(default=None),
) -> dict[str, object]:
    csv_bytes = None
    filename = None
    if file is not None and file.filename:
        csv_bytes = await file.read()
        filename = file.filename
    try:
        return await asyncio.to_thread(
            _execute_demo, backend, dataset, label_column, csv_bytes, filename
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/demo/packs/{pack_id}/artifacts/{role}", response_model=None)
def demo_pack_artifact(
    pack_id: str, role: str, preview: bool = False
):
    if not PACK_ID_RE.fullmatch(pack_id) or role not in ARTIFACT_NAMES:
        raise HTTPException(status_code=404, detail="artifact not found")
    path = DEMO_PACK_ROOT / pack_id / ARTIFACT_NAMES[role]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    if preview:
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = "".join(handle.readline() for _ in range(16))
        return PlainTextResponse(lines)
    media_types = {
        "transform_code": "text/x-python",
        "degraded_dataset": "text/csv",
        "model_report": "application/json",
        "regression_contract": "text/x-python",
    }
    return FileResponse(path, media_type=media_types[role], filename=path.name)


@app.post("/api/discovery/runs")
def discover(
    backend: Literal["local", "daytona"] = "local",
    dataset: Literal["full", "debug", "approval"] = "full",
    agents: Annotated[int, Query(ge=1, le=6)] = 4,
) -> dict[str, object]:
    paths = {
        "full": (FULL_DATA_PATH, TARGET_COLUMN),
        "debug": (DEFAULT_DATA_PATH, TARGET_COLUMN),
        "approval": (APPROVAL_DATA_PATH, "class"),
    }
    data_path, label_column = paths[dataset]
    try:
        report = run_table_discovery(
            data_path=data_path,
            label_column=label_column,
            backend=backend,
            output_root=DISCOVERY_OUTPUT_ROOT,
            max_specs=agents,
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
