from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.math.probability.mutual_information import (
    MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
)
from jacobian.math.probability.values import (
    MAX_FINITE_JOINT_TABLE_CELLS,
    MAX_FINITE_JOINT_TABLE_COLUMNS,
    MAX_FINITE_JOINT_TABLE_ROWS,
    FiniteJointTable,
    MutualInformationResult,
)

_Q1 = {"num": "1", "den": "1"}


def test_request_accepts_json_array_wire_shapes_with_raw_bound_validator() -> None:
    payload = {
        "row_labels": ["row"],
        "column_labels": ["column"],
        "probabilities": [[_Q1]],
        "log_base": 2,
    }

    from_python = FiniteJointTable.model_validate_json(json.dumps(payload))
    from_json = FiniteJointTable.model_validate_json(
        json.dumps(payload),
        strict=True,
    )

    assert from_python == from_json


def test_request_rejects_oversized_outer_table_before_cell_parsing() -> None:
    payload = {
        "row_labels": ["only"],
        "column_labels": ["only"],
        "probabilities": [[{}] for _ in range(MAX_FINITE_JOINT_TABLE_ROWS + 1)],
        "log_base": 2,
    }

    with pytest.raises(ValidationError):
        FiniteJointTable.model_validate_json(json.dumps(payload))


def test_request_rejects_oversized_cell_product_before_cell_parsing() -> None:
    payload = {
        "row_labels": [str(index) for index in range(MAX_FINITE_JOINT_TABLE_ROWS)],
        "column_labels": [
            str(index)
            for index in range(
                MAX_FINITE_JOINT_TABLE_CELLS // MAX_FINITE_JOINT_TABLE_ROWS + 1
            )
        ],
        "probabilities": [
            [
                {}
                for _ in range(
                    MAX_FINITE_JOINT_TABLE_CELLS // MAX_FINITE_JOINT_TABLE_ROWS + 1
                )
            ]
            for _ in range(MAX_FINITE_JOINT_TABLE_ROWS)
        ],
        "log_base": 2,
    }

    with pytest.raises(ValidationError):
        FiniteJointTable.model_validate_json(json.dumps(payload))


def _candidate() -> dict[str, object]:
    return {
        "row_labels": ["row"],
        "column_labels": ["column"],
        "row_marginals": [_Q1],
        "column_marginals": [_Q1],
        "positive_support": [
            {
                "row_index": 0,
                "column_index": 0,
                "probability": _Q1,
                "row_marginal": _Q1,
                "column_marginal": _Q1,
                "likelihood_ratio": _Q1,
            }
        ],
        "log_base": 2,
        "exact_logarithmic_value": {
            "scale": "1",
            "product": _Q1,
            "identity": "SCALE_TIMES_I_EQUALS_LOG_BASE_OF_PRODUCT",
        },
        "exact_value": {"num": "0", "den": "1"},
        "sign": "ZERO",
        "zero_cell_convention": "ZERO_MASS_TERMS_OMITTED",
    }


def test_candidate_rejects_oversized_support_before_item_parsing() -> None:
    candidate = _candidate()
    candidate["positive_support"] = [
        {} for _ in range(MAX_FINITE_JOINT_TABLE_CELLS + 1)
    ]

    with pytest.raises(ValidationError):
        MutualInformationResult.model_validate(candidate)


def test_candidate_rejects_oversized_marginals_before_item_parsing() -> None:
    candidate = _candidate()
    candidate["row_marginals"] = [{} for _ in range(MAX_FINITE_JOINT_TABLE_ROWS + 1)]

    with pytest.raises(ValidationError):
        MutualInformationResult.model_validate(candidate)


def test_candidate_rejects_oversized_rational_components_before_item_parsing() -> None:
    candidate = _candidate()
    candidate["row_marginals"] = [
        {"num": "1" + "0" * MAX_CANONICAL_RATIONAL_DIGITS, "den": "1"}
    ]

    with pytest.raises(ValidationError):
        MutualInformationResult.model_validate(candidate)


def test_candidate_uses_a_separate_logarithmic_value_product_bound() -> None:
    candidate = _candidate()
    candidate["exact_logarithmic_value"] = {
        "scale": "1",
        "product": {
            "num": "1" + "0" * MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
            "den": "1",
        },
    }

    with pytest.raises(ValidationError):
        MutualInformationResult.model_validate(candidate)


def test_generated_schemas_publish_all_collection_bounds() -> None:
    request_schema = FiniteJointTable.model_json_schema()
    result_schema = MutualInformationResult.model_json_schema()

    assert (
        request_schema["properties"]["probabilities"]["maxItems"]
        == MAX_FINITE_JOINT_TABLE_ROWS
    )
    assert (
        result_schema["properties"]["row_marginals"]["maxItems"]
        == MAX_FINITE_JOINT_TABLE_ROWS
    )
    assert (
        result_schema["properties"]["column_marginals"]["maxItems"]
        == MAX_FINITE_JOINT_TABLE_COLUMNS
    )
    assert (
        result_schema["properties"]["positive_support"]["maxItems"]
        == MAX_FINITE_JOINT_TABLE_CELLS
    )


def test_canonical_joint_table_retains_zero_mass_axes_in_native_and_wire_result() -> (
    None
):
    from jacobian._exact import CanonicalRational
    from jacobian.math.probability import mutual_information
    from jacobian.math.probability._mutual_information import (
        MUTUAL_INFORMATION_OPERATION,
    )

    zero = CanonicalRational(num=0, den=1)
    half = CanonicalRational(num=1, den=2)
    table = FiniteJointTable(
        row_labels=("heads", "tails", "unused row"),
        column_labels=("observed heads", "observed tails", "unused column"),
        probabilities=((half, zero, zero), (zero, half, zero), (zero, zero, zero)),
    )
    decoded = FiniteJointTable.model_validate_json(table.model_dump_json())
    result = mutual_information(decoded)
    assert MUTUAL_INFORMATION_OPERATION.request_type is FiniteJointTable
    assert MUTUAL_INFORMATION_OPERATION.result_type is MutualInformationResult
    assert MUTUAL_INFORMATION_OPERATION.run(decoded) == result
    restored = MutualInformationResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert restored.row_labels == table.row_labels
    assert restored.column_labels == table.column_labels
    assert restored.row_marginals == (half, half, zero)
    assert restored.column_marginals == (half, half, zero)
    assert restored.exact_value == CanonicalRational(num=1, den=1)
    assert tuple(
        (term.row_index, term.column_index) for term in restored.positive_support
    ) == ((0, 0), (1, 1))


def test_joint_table_parsing_does_not_establish_normalization() -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.catalog.models import OperationDomainValidationError
    from jacobian.math.probability import mutual_information

    table = FiniteJointTable(
        row_labels=("a",),
        column_labels=("x",),
        probabilities=((CanonicalRational(num=1, den=2),),),
    )
    decoded = FiniteJointTable.model_validate_json(table.model_dump_json())
    with pytest.raises(OperationDomainValidationError, match="sum exactly to 1"):
        mutual_information(decoded)


def test_likelihood_ratio_is_a_claim_after_serialization() -> None:
    from jacobian._exact import CanonicalRational
    from jacobian.math.probability import MutualInformationTerm

    one = CanonicalRational(num=1, den=1)
    claimed = MutualInformationTerm(
        row_index=0,
        column_index=0,
        probability=one,
        row_marginal=one,
        column_marginal=one,
        likelihood_ratio=CanonicalRational(num=2, den=1),
    )
    decoded = MutualInformationTerm.model_validate_json(claimed.model_dump_json())
    assert decoded == claimed
    assert decoded.likelihood_ratio != one


@pytest.mark.parametrize("base, expected", [(2, "1/1"), (4, "1/2"), (3, None)])
def test_mutual_information_exact_log_identity(base: int, expected: str | None) -> None:
    from fractions import Fraction

    from jacobian._exact import CanonicalRational
    from jacobian.math.probability import mutual_information

    half = CanonicalRational(num=1, den=2)
    zero = CanonicalRational(num=0, den=1)
    result = mutual_information(
        FiniteJointTable(
            row_labels=("a", "b"),
            column_labels=("x", "y"),
            probabilities=((half, zero), (zero, half)),
            log_base=base,
        )
    )
    assert result.exact_logarithmic_value.scale == 2
    assert result.exact_logarithmic_value.product.as_fraction() == 4
    assert (
        None if result.exact_value is None else result.exact_value.as_fraction()
    ) == (None if expected is None else Fraction(expected))


def test_joint_table_rejects_ambiguous_axes_and_shape() -> None:
    from jacobian._exact import CanonicalRational

    one = CanonicalRational(num=1, den=1)
    with pytest.raises(ValidationError, match="unique"):
        FiniteJointTable(
            row_labels=("a", "a"), column_labels=("x",), probabilities=((one,), (one,))
        )
    with pytest.raises(ValidationError, match="column labels"):
        FiniteJointTable(
            row_labels=("a",), column_labels=("x", "y"), probabilities=((one,),)
        )
