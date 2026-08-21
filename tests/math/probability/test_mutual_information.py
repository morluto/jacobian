from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.math.probability._mutual_information import (
    FiniteJointTableMutualInformationRequest,
    FiniteJointTableMutualInformationResult,
)
from jacobian.math.probability.mutual_information import (
    MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
)

_Q1 = {"num": "1", "den": "1"}


def test_request_accepts_json_array_wire_shapes_with_raw_bound_validator() -> None:
    payload = {
        "row_labels": ["row"],
        "column_labels": ["column"],
        "probabilities": [[_Q1]],
        "log_base": 2,
    }

    from_python = FiniteJointTableMutualInformationRequest.model_validate(payload)
    from_json = FiniteJointTableMutualInformationRequest.model_validate_json(
        json.dumps(payload),
        strict=True,
    )

    assert from_python == from_json


def test_request_rejects_oversized_outer_table_before_cell_parsing() -> None:
    payload = {
        "row_labels": ["only"],
        "column_labels": ["only"],
        "probabilities": [[{}] for _ in range(17)],
        "log_base": 2,
    }

    with pytest.raises(ValidationError, match="bounded row count"):
        FiniteJointTableMutualInformationRequest.model_validate(payload)


def test_request_rejects_oversized_cell_product_before_cell_parsing() -> None:
    payload = {
        "row_labels": [str(index) for index in range(8)],
        "column_labels": [str(index) for index in range(9)],
        "probabilities": [[{} for _ in range(9)] for _ in range(8)],
        "log_base": 2,
    }

    with pytest.raises(ValidationError, match="bounded cell count"):
        FiniteJointTableMutualInformationRequest.model_validate(payload)


def _candidate() -> dict[str, object]:
    return {
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
        "log_product_certificate": {
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
    candidate["positive_support"] = [{} for _ in range(65)]

    with pytest.raises(ValidationError, match="positive_support exceeds"):
        FiniteJointTableMutualInformationResult.model_validate(candidate)


def test_candidate_rejects_oversized_marginals_before_item_parsing() -> None:
    candidate = _candidate()
    candidate["row_marginals"] = [{} for _ in range(17)]

    with pytest.raises(ValidationError, match="row_marginals exceeds"):
        FiniteJointTableMutualInformationResult.model_validate(candidate)


def test_candidate_rejects_oversized_rational_components_before_item_parsing() -> None:
    candidate = _candidate()
    candidate["row_marginals"] = [
        {"num": "1" + "0" * MAX_CANONICAL_RATIONAL_DIGITS, "den": "1"}
    ]

    with pytest.raises(ValidationError, match=r"row_marginals\[0\]"):
        FiniteJointTableMutualInformationResult.model_validate(candidate)


def test_candidate_uses_a_separate_certificate_product_bound() -> None:
    candidate = _candidate()
    candidate["log_product_certificate"] = {
        **candidate["log_product_certificate"],
        "product": {
            "num": "1" + "0" * MAX_MUTUAL_INFORMATION_PRODUCT_DIGITS,
            "den": "1",
        },
    }

    with pytest.raises(ValidationError, match="certificate product"):
        FiniteJointTableMutualInformationResult.model_validate(candidate)


def test_generated_schemas_publish_all_collection_bounds() -> None:
    request_schema = FiniteJointTableMutualInformationRequest.model_json_schema()
    result_schema = FiniteJointTableMutualInformationResult.model_json_schema()

    assert request_schema["properties"]["probabilities"]["maxItems"] == 16
    assert result_schema["properties"]["row_marginals"]["maxItems"] == 16
    assert result_schema["properties"]["column_marginals"]["maxItems"] == 16
    assert result_schema["properties"]["positive_support"]["maxItems"] == 64
