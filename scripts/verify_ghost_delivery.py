"""Audit a published four-file Ghost delivery from any working directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo.artifacts import (  # noqa: E402
    validate_credit_artifacts,
    validate_ghost_artifacts,
)
from ghostdata.demo.credit import DEFAULT_DATA_PATH, TARGET_COLUMN  # noqa: E402
from ghostdata.demo.discovery import FULL_DATA_PATH  # noqa: E402


APPROVAL_PATH = ROOT / "data" / "live" / "credit_approval.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument(
        "--dataset", choices=("full", "debug", "approval"), default="full"
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--label-column")
    args = parser.parse_args()
    if args.csv is not None:
        if not args.label_column:
            parser.error("--label-column is required with --csv")
        report = validate_ghost_artifacts(
            args.csv, args.artifact_directory.resolve(), args.label_column
        )
    elif args.dataset == "approval":
        report = validate_ghost_artifacts(
            APPROVAL_PATH, args.artifact_directory.resolve(), "class"
        )
    else:
        reference = FULL_DATA_PATH if args.dataset == "full" else DEFAULT_DATA_PATH
        if args.label_column and args.label_column != TARGET_COLUMN:
            report = validate_ghost_artifacts(
                reference, args.artifact_directory.resolve(), args.label_column
            )
        else:
            report = validate_credit_artifacts(
                reference, args.artifact_directory.resolve()
            )
    print(
        json.dumps(
            {
                "status": "verified",
                "discovery_id": report["discovery_id"],
                "selected_agent": report["selected_agent"],
                "baseline_auc": report["baseline_auc"],
                "candidate_auc": report["candidate_auc"],
                "auc_drop": report["auc_drop"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
