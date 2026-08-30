"""Audit a published four-file Ghost delivery from any working directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghostdata.demo.artifacts import validate_credit_artifacts  # noqa: E402
from ghostdata.demo.credit import DEFAULT_DATA_PATH  # noqa: E402
from ghostdata.demo.discovery import FULL_DATA_PATH  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_directory", type=Path)
    parser.add_argument("--dataset", choices=("full", "debug"), default="full")
    args = parser.parse_args()
    reference = FULL_DATA_PATH if args.dataset == "full" else DEFAULT_DATA_PATH
    report = validate_credit_artifacts(reference, args.artifact_directory.resolve())
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
