import math

import pandas as pd
import pytest

from ghostdata.demo import charts
from ghostdata.demo.credit import prepare_credit_demo
from ghostdata.verification import (
    ClaimVerdict,
    ExecutionEvidence,
    ExperimentVerdict,
    VerificationReport,
)


def test_chart_numeric_helpers_handle_invalid_and_constant_values() -> None:
    assert charts._finite_float("bad") is None
    assert charts._finite_float(math.inf) is None
    edges = charts._bin_edges(pd.Series([5.0, 5.0, None]))
    assert edges[0] == 5.0
    assert edges[-1] == 6.0
    quantile_edges, labels = charts._quantile_bins(pd.Series([5.0, 5.0, 5.0]))
    assert len(quantile_edges) == 1
    assert labels == ()
    empty_edges, empty_labels = charts._quantile_bins(pd.Series([None, None]))
    assert list(empty_edges) == [0.0]
    assert empty_labels == ()
    assert charts._to_numeric(pd.Series([True, False])).tolist() == [1.0, 0.0]


def test_metric_and_visual_fallbacks_reject_missing_feature() -> None:
    empty_report = VerificationReport("bundle", "inconclusive", (), ())
    assert charts._metric_block(empty_report) == {
        "name": "metric",
        "baseline": None,
        "candidate": None,
        "degradation": None,
        "affected_fraction": 0.0,
    }

    evidence = ExecutionEvidence(
        "bundle",
        "V001",
        "C001",
        "entity_alignment",
        "completed",
        {"model_metric": "invalid", "invariants": "invalid"},
    )
    report = VerificationReport("bundle", "inconclusive", (), (evidence,))
    assert charts._metric_block(report)["baseline"] is None

    prepared = prepare_credit_demo()
    spec = prepared.specs[0]
    with pytest.raises(ValueError, match="perturbed feature missing"):
        charts.build_visuals(
            prepared.reference,
            prepared.reference.drop(columns=["MonthlyIncome"]),
            report,
            spec,
        )

    payload = charts.build_visuals(
        prepared.reference, prepared.reference.copy(), report, spec
    )
    assert payload["invariants"] == {}
    assert payload["perturbed_feature"] == "MonthlyIncome"


def test_metric_block_ignores_invalid_ghost_measurements() -> None:
    evidence = ExecutionEvidence(
        "bundle",
        "V001",
        "C001",
        "entity_alignment",
        "completed",
        {
            "model_metric": {
                "name": "roc_auc",
                "baseline": 0.5,
                "candidate": 0.4,
            }
        },
    )
    experiment = ExperimentVerdict(
        "V001",
        "C001",
        "counterexample",
        "bad measurements",
        {"baseline": "bad", "candidate": "bad", "degradation": "bad"},
    )
    report = VerificationReport(
        "bundle",
        "not_verified",
        (ClaimVerdict("C001", "not_verified", (experiment,)),),
        (evidence,),
    )

    assert charts._metric_block(report)["degradation"] == pytest.approx(0.1)
