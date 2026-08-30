from pathlib import Path

import pandas as pd
import pytest

from ghostdata.tabular import (
    encode_label,
    feature_invariants,
    feature_score,
    load_table,
    profile_table,
    spec_from_profile,
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


def test_feature_invariants_and_score_detect_entity_alignment() -> None:
    frame = _frame()
    checks = feature_invariants("tenure")
    assert all(checks(frame, frame.copy()).values())

    shuffled = frame.copy()
    shuffled["tenure"] = shuffled["tenure"].iloc[::-1].to_numpy()
    assert checks(frame, shuffled)["marginal_distribution"]
    shuffled.loc[shuffled.index[0], "tenure"] = 999
    assert not checks(frame, shuffled)["marginal_distribution"]

    score = feature_score(frame, "churned", "tenure")
    assert score(frame) > 0.5
    with pytest.raises(ValueError, match="row count"):
        score(frame.iloc[:-1])


def test_german_credit_table_profiles_without_credit_column_names() -> None:
    path = Path("data/build/german_credit.csv")
    frame = load_table(path, "class")
    spec = spec_from_profile(profile_table(frame, "class"), "C001")

    assert spec["parameters"]["target_feature"] in frame.columns
    assert spec["parameters"]["target_feature"] != "class"
    assert "MonthlyIncome" not in spec["hypothesis"]
