"""Consumer checks for source-bound majorization and decomposition claims."""

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.majorization import (
    RationalVector,
    birkhoff_decomposition,
    majorization_check,
    verify_birkhoff,
    verify_majorization,
)
from jacobian.math.matrices.values import rational_matrix_from_fractions


def test_wrong_slack_is_a_decodable_claim() -> None:
    vector = RationalVector(
        labels=("a", "b"),
        values=(
            CanonicalRational(num=1, den=1),
            CanonicalRational(num=0, den=1),
        ),
    )
    result = majorization_check(vector, vector)
    assert verify_majorization(
        type(result).model_validate_json(result.model_dump_json())
    )
    payload = json.loads(result.model_dump_json())
    payload["prefix_slacks"] = [{"num": "1", "den": "1"}]
    assert not verify_majorization(
        type(result).model_validate_json(json.dumps(payload))
    )


def test_decomposition_checks_axis_convexity_and_reconstruction() -> None:
    matrix = rational_matrix_from_fractions(((Fraction(1, 2), Fraction(1, 2)),) * 2)
    result = birkhoff_decomposition(matrix)
    assert verify_birkhoff(type(result).model_validate_json(result.model_dump_json()))
    payload = json.loads(result.model_dump_json())
    payload["terms"][0]["permutation"] = [0, 0]
    with pytest.raises(ValidationError):
        type(result).model_validate_json(json.dumps(payload))
    payload = json.loads(result.model_dump_json())
    payload["terms"][0]["weight"] = {"num": "-1", "den": "2"}
    assert not verify_birkhoff(type(result).model_validate_json(json.dumps(payload)))
    payload = json.loads(result.model_dump_json())
    payload["terms"][1]["permutation"] = payload["terms"][0]["permutation"]
    assert not verify_birkhoff(type(result).model_validate_json(json.dumps(payload)))


def test_empty_decomposition_round_trip() -> None:
    result = birkhoff_decomposition(rational_matrix_from_fractions(()))
    assert verify_birkhoff(type(result).model_validate_json(result.model_dump_json()))
