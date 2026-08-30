"""Runs inside a Daytona analyst sandbox. Inspects the table and writes worlds.

No Ghost CSV, no model_report, no regression contract, no verdict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORK = Path(__file__).resolve().parent


def _log(message: str) -> None:
    print(f"ghostdata-analyst: {message}", flush=True)


def infer_label(prompt: str, columns: list[str], frame: pd.DataFrame) -> str:
    lowered = prompt.lower()
    ranked = sorted(columns, key=len, reverse=True)
    for name in ranked:
        if name.lower() in lowered:
            return name
    binaries = [
        name
        for name in columns
        if int(frame[name].nunique(dropna=True)) == 2
    ]
    if binaries:
        return binaries[-1]
    return columns[-1]


def ranked_numeric(frame: pd.DataFrame, label: str) -> list[tuple[str, float]]:
    y = pd.to_numeric(frame[label], errors="coerce")
    if y.nunique(dropna=True) < 2:
        codes, _ = pd.factorize(frame[label], sort=True)
        y = pd.Series(codes, index=frame.index, dtype=float)
        y = y.mask(y < 0)
    scored: list[tuple[str, float]] = []
    for name in frame.columns:
        if name == label:
            continue
        x = pd.to_numeric(frame[name], errors="coerce")
        if x.nunique(dropna=True) < 2:
            continue
        aligned = pd.concat([x, y], axis=1).dropna()
        if len(aligned) < 4:
            continue
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        if corr == corr:
            scored.append((name, corr))
    scored.sort(key=lambda item: abs(item[1]), reverse=True)
    return scored


def transform_source(feature: str, fraction: float, seed: int) -> str:
    return (
        "import numpy as np\n"
        "import pandas as pd\n\n"
        f"FEATURE = {feature!r}\n"
        f"FRACTION = {fraction!r}\n"
        f"SEED = {seed}\n\n"
        "def transform(dataframe: pd.DataFrame) -> pd.DataFrame:\n"
        "    out = dataframe.copy(deep=True)\n"
        "    n = len(out)\n"
        "    count = min(n, round(n * FRACTION))\n"
        "    if count < 2 or FEATURE not in out.columns:\n"
        "        return out\n"
        "    selected = np.random.default_rng(SEED).choice(n, size=count, replace=False)\n"
        "    position = out.columns.get_loc(FEATURE)\n"
        "    original = out.iloc[selected, position].to_numpy(copy=True)\n"
        "    out.iloc[selected, position] = np.roll(original, 1)\n"
        "    return out\n"
    )


def main() -> None:
    prompt = (WORK / "task.md").read_text(encoding="utf-8")
    volume = Path("/data/dataset.csv")
    workspace = WORK / "data" / "dataset.csv"
    frame = pd.read_csv(volume if volume.is_file() else workspace)
    columns = [str(name) for name in frame.columns]
    _log(f"loaded {len(frame)} rows, {len(columns)} columns")
    _log("inspected_columns=" + ",".join(columns))
    label = infer_label(prompt, columns, frame)
    _log(f"label_column={label}")
    ranked = ranked_numeric(frame, label)
    _log("ranked_features=" + ",".join(name for name, _corr in ranked[:8]))
    missing = [
        name
        for name in columns
        if name != label and float(frame[name].isna().mean()) > 0
    ]
    _log("missing_ranked=" + ",".join(missing[:8]))
    worlds_root = WORK / "worlds"
    worlds_root.mkdir(parents=True, exist_ok=True)
    hypotheses = []
    fractions = (0.50, 0.35, 0.75)
    features = ranked[:3] or [
        (name, 0.0) for name in columns if name != label
    ][:3]
    for index, ((feature, corr), fraction) in enumerate(
        zip(features, fractions), start=1
    ):
        world_id = f"W{index:03d}"
        hypothesis = (
            f"{feature} is associated with {label} (corr={corr:.3f}); "
            "permuting valid values onto the wrong rows should keep schema and "
            "marginals while the frozen model loses the relationship."
        )
        _log(f"hypothesis {world_id}: {hypothesis}")
        folder = worlds_root / world_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "transform.py").write_text(
            transform_source(feature, fraction, seed=7 + index),
            encoding="utf-8",
        )
        meta = {
            "world_id": world_id,
            "title": f"Entity misalignment of {feature}",
            "hypothesis": hypothesis,
            "target_feature": feature,
            "mismatch_fraction": fraction,
            "seed": 7 + index,
        }
        (folder / "hypothesis.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        hypotheses.append(meta)
    analysis = {
        "label_column": label,
        "inspected_columns": columns,
        "ranked_features": [name for name, _corr in ranked],
        "missing_ranked": missing,
        "fragile_assumptions": [
            "Checks that only look at schema, missingness, and marginals "
            "will not see entity misalignment."
        ],
        "hypotheses": hypotheses,
        "planner": "daytona_sandbox_python",
    }
    (WORK / "analysis.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _log(f"emitted {len(hypotheses)} worlds; no Ghost CSV written")


if __name__ == "__main__":
    main()
