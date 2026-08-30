import pandas as pd
import pytest

from ghostdata.bundle import AgentOutput, AnalysisBundle
from ghostdata.execution.local import (
    ExperimentCompiler,
    LocalVerificationRunner,
    TransformResult,
    default_compiler,
)
from ghostdata.verification import VerificationSpec


def bundle() -> AnalysisBundle:
    return AnalysisBundle("B001", "Verify", {"dataset": "data.csv"}, AgentOutput(), ())


def spec(**parameter_overrides: object) -> VerificationSpec:
    parameters = {
        "target_feature": "income",
        "segment": {},
        "mismatch_fraction": 1.0,
        "seed": 7,
    }
    parameters.update(parameter_overrides)
    return VerificationSpec("V001", "C001", "entity_alignment", "Misalign", parameters)


def test_compiler_registers_custom_transform_and_rejects_duplicates() -> None:
    compiler = ExperimentCompiler()
    dataframe = pd.DataFrame({"income": [1, 2]})

    def identity(data: pd.DataFrame, experiment: VerificationSpec) -> TransformResult:
        return TransformResult(data.copy(), 0.0)

    compiler.register("custom", identity)
    custom = VerificationSpec("V", "C", "custom", "identity")

    assert compiler.execute(dataframe, custom).dataframe.equals(dataframe)
    with pytest.raises(ValueError, match="already registered"):
        compiler.register("custom", identity)


def test_compiler_rejects_unknown_experiment() -> None:
    unknown = VerificationSpec("V", "C", "unknown", "unknown")

    with pytest.raises(ValueError, match="unsupported experiment"):
        default_compiler().execute(pd.DataFrame({"income": [1, 2]}), unknown)


def test_entity_alignment_preserves_values_with_duplicate_indexes() -> None:
    dataframe = pd.DataFrame(
        {"income": [10, 20, 30, 40]},
        index=[1, 1, 2, 2],
    )

    result = default_compiler().execute(dataframe, spec(mismatch_fraction=0.75))

    assert result.dataframe.index.tolist() == dataframe.index.tolist()
    assert sorted(result.dataframe["income"]) == sorted(dataframe["income"])
    assert 0 < result.affected_fraction <= 0.75


def test_entity_alignment_only_changes_selected_segment() -> None:
    dataframe = pd.DataFrame(
        {"income": [10, 20, 30, 40], "employment": ["employee", "employee", "self", "self"]}
    )

    result = default_compiler().execute(
        dataframe, spec(segment={"employment": "self"})
    )

    assert result.dataframe["income"].tolist() == [10, 20, 40, 30]
    assert result.affected_fraction == 0.5


@pytest.mark.parametrize("fraction", [-0.1, 1.1, "nan"])
def test_entity_alignment_rejects_invalid_fraction(fraction: object) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        default_compiler().execute(
            pd.DataFrame({"income": [1, 2]}), spec(mismatch_fraction=fraction)
        )


@pytest.mark.parametrize(
    "dataframe,experiment,error",
    [
        (pd.DataFrame({"other": [1, 2]}), spec(), "missing target feature"),
        (
            pd.DataFrame({"income": [1, 2]}),
            spec(segment={"missing": "x"}),
            "missing segment column",
        ),
        (
            pd.DataFrame({"income": [1, 2]}),
            spec(segment="bad"),
            "segment must be a mapping",
        ),
    ],
)
def test_entity_alignment_rejects_invalid_columns_or_segment(
    dataframe: pd.DataFrame, experiment: VerificationSpec, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        default_compiler().execute(dataframe, experiment)


def test_entity_alignment_returns_copy_when_too_few_rows_are_selected() -> None:
    dataframe = pd.DataFrame({"income": [10, 20], "segment": ["yes", "no"]})

    result = default_compiler().execute(
        dataframe, spec(segment={"segment": "yes"})
    )
    result.dataframe.loc[0, "income"] = 999

    assert result.affected_fraction == 0
    assert dataframe.loc[0, "income"] == 10


def test_entity_alignment_handles_nan_and_rejects_duplicate_target_columns() -> None:
    dataframe = pd.DataFrame({"income": [None, 1.0, 2.0, 3.0]})
    result = default_compiler().execute(dataframe, spec())

    assert result.dataframe["income"].isna().sum() == 1
    assert result.affected_fraction == 1.0

    duplicates = pd.DataFrame([[1, 2], [3, 4]], columns=["income", "income"])
    with pytest.raises(ValueError, match="must be unique"):
        default_compiler().execute(duplicates, spec())


def test_local_runner_records_raw_evidence_without_deciding_verdict() -> None:
    reference = pd.DataFrame({"income": [1, 2, 3, 4]})

    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        lambda before, after: {
            "schema": tuple(before.columns) == tuple(after.columns)
        },
        lambda frame: float(frame["income"].iloc[0]),
        metric="score",
    )
    evidence = runner.run(bundle(), spec())

    assert evidence.observations["invariants"] == {"schema": True}
    assert evidence.observations["model_metric"] == {
        "name": "score",
        "baseline": 1.0,
        "candidate": 4.0,
    }
    assert "verdict" not in evidence.to_dict()


def test_local_runner_validates_metric_and_check_contract() -> None:
    reference = pd.DataFrame({"income": [1, 2]})
    with pytest.raises(ValueError, match="metric"):
        LocalVerificationRunner(reference, default_compiler(), lambda a, b: {}, lambda x: 1, "")

    runner = LocalVerificationRunner(
        reference,
        default_compiler(),
        lambda before, after: {"schema": 1},
        lambda frame: 1,
        "score",
    )
    with pytest.raises(ValueError, match="booleans"):
        runner.run(bundle(), spec())
