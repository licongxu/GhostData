"""Sandbox proposer: inspect a table and emit VerificationSpecs. No Ghost CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.tabular import DEFAULT_MAX_SPECS, propose_from_table


WORK_DIR = Path(__file__).resolve().parent


def _log(message: str) -> None:
    print(f"ghostdata-proposer: {message}", flush=True)


def main() -> None:
    task = json.loads((WORK_DIR / "task.json").read_text(encoding="utf-8"))
    label_column = str(task["label_column"])
    claim_id = str(task.get("claim_id", "C001"))
    max_specs = int(task.get("max_specs", DEFAULT_MAX_SPECS))
    dataset = Path(str(task["dataset"]))
    dataframe = pd.read_csv(dataset if dataset.is_absolute() else WORK_DIR / dataset)
    _log(f"loaded {len(dataframe)} rows, {len(dataframe.columns)} columns")
    _log(f"label_column={label_column}")
    payloads, analysis = propose_from_table(
        dataframe, label_column, claim_id, max_specs=max_specs
    )
    _log("inspected_columns=" + ",".join(analysis.get("inspected_columns") or []))
    _log("ranked_features=" + ",".join(analysis.get("ranked_features") or []))
    _log("missing_ranked=" + ",".join(analysis.get("missing_ranked") or []))
    for item in analysis.get("hypotheses") or []:
        _log(
            f"hypothesis {item.get('verification_id')}: "
            f"{item.get('target_feature')} fraction={item.get('mismatch_fraction')} "
            f"segment={item.get('segment')} :: {item.get('hypothesis')}"
        )
    _log(f"emitted {len(payloads)} VerificationSpec(s); no Ghost CSV written")
    (WORK_DIR / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )
    (WORK_DIR / "specs.json").write_text(
        json.dumps(payloads, indent=2, sort_keys=True, allow_nan=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
