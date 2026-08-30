from pathlib import Path

import pandas as pd
import pytest

from ghostdata.tabular import (
    dump_frozen_model,
    encode_label,
    feature_invariants,
    feature_score,
    fit_frozen_model,
    frozen_model_score,
    infer_feature_columns,
    load_frozen_model,
    load_table,
    profile_table,
    propose_from_table,
    spec_from_profile,
    specs_from_profile,
    table_invariants,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "churned": [0, 0, 0, 0, 1, 1, 1, 1],
            "tenure": [12, 10, 11, 9, 1, 2, 0, 3],
            "charges": [80, 70, 75, 65, 20, 25, 15, 30],
            "plan": ["pro", "pro", "basic", "pro", "basic", "basic", "basic", "pro"],
        }
    )


def test_load_table_rejects_missing_label_empty_and_single_class(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="label column missing"):
        load_table(missing, "y")

    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["y", "x"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="at least one row"):
        load_table(empty, "y")

    single = tmp_path / "single.csv"
    pd.DataFrame({"y": [1, 1], "x": [1, 2]}).to_csv(single, index=False)
    with pytest.raises(ValueError, match="at least two classes"):
        load_table(single, "y")


def test_encode_label_handles_bool_binary_and_categorical() -> None:
    assert encode_label(pd.Series([True, False, True])).tolist() == [1.0, 0.0, 1.0]
    assert encode_label(pd.Series([0, 1, 0])).tolist() == [0.0, 1.0, 0.0]
    encoded = encode_label(pd.Series(["bad", "good", "bad"]))
    assert set(encoded.dropna().unique().tolist()) == {0.0, 1.0}


def test_profile_picks_the_strongest_numeric_label_relationship() -> None:
    profile = profile_table(_frame(), "churned")
    spec = spec_from_profile(profile, "C001")

    ranked = profile["ranked_features"]
    assert ranked[0] in {"tenure", "charges"}
    assert spec["experiment_type"] == "entity_alignment"
    assert spec["origin"] == "sandbox_agent"
    assert spec["parameters"]["target_feature"] == ranked[0]
    assert spec["expected_invariants"] == [
        "schema",
        "marginal_distribution",
        "missing_rate",
    ]


def test_profile_skips_unusable_columns_and_requires_the_label() -> None:
    frame = pd.DataFrame(
        {
            "y": [1, 1, 1, 1],
            "const": [3, 3, 3, 3],
            "sparse": [1.0, 2.0, 3.0, None],
            "text": ["a", "b", "a", "b"],
        }
    )
    profile = profile_table(frame, "y")
    assert profile["ranked_features"] == []
    with pytest.raises(ValueError, match="label column missing"):
        profile_table(frame, "missing")


def test_encode_label_factorizes_non_binary_numeric_values() -> None:
    encoded = encode_label(pd.Series([2, 4, 2, 6]))
    assert encoded.nunique(dropna=True) == 3


def test_spec_from_profile_requires_a_ranked_feature() -> None:
    with pytest.raises(ValueError, match="no numeric feature"):
        spec_from_profile({"ranked_features": []}, "C001")
    with pytest.raises(ValueError, match="no numeric feature"):
        specs_from_profile({"ranked_features": []}, "C001")


def test_propose_from_table_emits_distinct_compilable_specs() -> None:
    payloads, analysis = propose_from_table(_frame(), "churned", "C001", max_specs=4)

    features = [item["parameters"]["target_feature"] for item in payloads]
    assert len(payloads) >= 2
    assert "tenure" in features
    assert "churned" not in features
    assert len({(item["parameters"]["target_feature"], item["parameters"]["mismatch_fraction"], tuple(sorted(item["parameters"]["segment"].items()))) for item in payloads}) == len(payloads)
    assert analysis["inspected_columns"] == ["churned", "tenure", "charges", "plan"]
    assert analysis["hypotheses"]
    assert all(item["experiment_type"] == "entity_alignment" for item in payloads)


def test_feature_invariants_and_score_detect_entity_alignment() -> None:
    frame = _frame()
    checks = feature_invariants("tenure")
    assert all(checks(frame, frame.copy()).values())

    shuffled = frame.copy()
    shuffled["tenure"] = shuffled["tenure"].iloc[::-1].to_numpy()
    assert checks(frame, shuffled)["marginal_distribution"]
    shuffled.loc[shuffled.index[0], "tenure"] = 999
    assert not checks(frame, shuffled)["marginal_distribution"]

    score = frozen_model_score(frame, "churned")
    assert score(frame) > 0.5
    with pytest.raises(ValueError, match="row count"):
        score(frame.iloc[:-1])
    assert table_invariants(frame, frame.copy())["marginal_distribution"]
    shuffled = frame.copy()
    shuffled["tenure"] = shuffled["tenure"].iloc[::-1].to_numpy()
    assert table_invariants(frame, shuffled)["marginal_distribution"]
    proxy = feature_score(frame, "churned", "tenure")
    assert proxy(frame) > 0.5
    payload = fit_frozen_model(frame, "churned")
    restored = load_frozen_model(dump_frozen_model(payload))
    assert restored["feature_columns"] == payload["feature_columns"]
    with pytest.raises(ValueError, match="no usable model"):
        infer_feature_columns(pd.DataFrame({"churned": [0, 1]}), "churned")
    with pytest.raises(ValueError, match="at least 1"):
        specs_from_profile({"ranked_features": ["tenure"]}, "C001", max_specs=0)
    assert infer_feature_columns(frame, "churned", allowlist=["tenure"]) == ["tenure"]
    mismatched = frame.copy()
    mismatched["extra"] = 1
    assert table_invariants(frame, mismatched)["schema"] is False
    import io
    import joblib

    buffer = io.BytesIO()
    joblib.dump(["not-a-mapping"], buffer)
    with pytest.raises(ValueError, match="invalid"):
        load_frozen_model(buffer.getvalue())
    single = specs_from_profile(
        {"ranked_features": ["tenure"], "correlations_with_label": {"tenure": 0.9}},
        "C001",
        max_specs=4,
    )
    assert len(single) >= 2
    inverted = pd.DataFrame(
        {"y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [9, 8, 7, 6, 1, 2, 3, 4]}
    )
    assert feature_score(inverted, "y", "x")(inverted) >= 0.5


def test_german_credit_table_profiles_without_credit_column_names() -> None:
    path = Path("data/build/german_credit.csv")
    frame = load_table(path, "class")
    spec = spec_from_profile(profile_table(frame, "class"), "C001")

    assert spec["parameters"]["target_feature"] in frame.columns
    assert spec["parameters"]["target_feature"] != "class"
    assert "MonthlyIncome" not in spec["hypothesis"]


def test_untrusted_or_unknown_payloads_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_specs(*args, **kwargs):
        del args, kwargs
        return [
            {
                "verification_id": "V001",
                "claim_id": "C001",
                "experiment_type": "entity_alignment",
                "hypothesis": "bad",
                "parameters": {"transform_code": "print(1)", "target_feature": "tenure"},
                "expected_invariants": ["schema"],
                "origin": "sandbox_agent",
            },
            {
                "verification_id": "V002",
                "claim_id": "C001",
                "experiment_type": "python_script",
                "hypothesis": "worse",
                "parameters": {},
                "expected_invariants": ["schema"],
                "origin": "sandbox_agent",
            },
        ]

    monkeypatch.setattr("ghostdata.tabular.specs_from_profile", fake_specs)
    payloads, analysis = propose_from_table(_frame(), "churned", "C001")
    assert payloads == []
    assert len(analysis["dropped"]) == 2


def test_credit_approval_emits_specs_from_that_table() -> None:
    path = Path("data/live/credit_approval.csv")
    frame = load_table(path, "class")
    payloads, analysis = propose_from_table(frame, "class", "C001")

    features = {item["parameters"]["target_feature"] for item in payloads}
    assert "class" in analysis["inspected_columns"]
    assert "MonthlyIncome" not in analysis["inspected_columns"]
    assert features <= set(frame.columns)
    assert "class" not in features
    assert len(payloads) >= 2
