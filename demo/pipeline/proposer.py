"""Sandbox proposer: inspect a table and emit a VerificationSpec. No Ghost CSV."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ghostdata.tabular import profile_table, spec_from_profile


WORK_DIR = Path(__file__).resolve().parent


def main() -> None:
    task = json.loads((WORK_DIR / "task.json").read_text(encoding="utf-8"))
    label_column = str(task["label_column"])
    claim_id = str(task.get("claim_id", "C001"))
    dataframe = pd.read_csv(WORK_DIR / task["dataset"])
    profile = profile_table(dataframe, label_column)
    spec = spec_from_profile(profile, claim_id)
    (WORK_DIR / "analysis.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (WORK_DIR / "verification.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
