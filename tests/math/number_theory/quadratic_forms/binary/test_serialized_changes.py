"""Typed form-change witnesses remain claims until consumed."""

import pytest
from pydantic import ValidationError

from jacobian.canonical import encode_strict_json
from jacobian.math.number_theory.quadratic_forms.binary import (
    PrimitivePositiveDefiniteBinaryQuadraticForm,
    ProperFormChangeOfVariables,
    verify_change_of_variables,
    verify_reduction,
)
from jacobian.math.number_theory.quadratic_forms.binary._models import (
    BinaryQuadraticFormReduceRequest,
    ReducedBinaryQuadraticFormResult,
)
from jacobian.math.number_theory.quadratic_forms.binary._tools import compute_reduce


def test_reduction_witness_round_trip_and_forgery() -> None:
    form = PrimitivePositiveDefiniteBinaryQuadraticForm(a=5, b=6, c=2)
    result = compute_reduce(BinaryQuadraticFormReduceRequest(form=form))
    decoded = ReducedBinaryQuadraticFormResult.model_validate_json(
        result.model_dump_json()
    )
    assert verify_reduction(decoded)
    payload = decoded.model_dump(mode="json")
    payload["change"]["matrix"]["entries"] = [["1", "0"], ["0", "-1"]]
    assert not verify_reduction(
        ReducedBinaryQuadraticFormResult.model_validate_json(
            encode_strict_json(payload)
        )
    )


def test_change_checks_determinant_and_substitution_after_decode() -> None:
    form = PrimitivePositiveDefiniteBinaryQuadraticForm(a=1, b=0, c=1)
    for rows, expected in [
        (((1, 0), (0, 1)), True),
        (((1, 0), (0, -1)), False),
        (((0, 0), (0, 0)), False),
        (((1, 1), (0, 1)), False),
    ]:
        claim = ProperFormChangeOfVariables.from_rows(form, form, rows)
        assert (
            verify_change_of_variables(
                ProperFormChangeOfVariables.model_validate_json(claim.model_dump_json())
            )
            is expected
        )
    payload = claim.model_dump(mode="json")
    payload["matrix"]["entries"] = [["1"]]
    with pytest.raises(ValidationError):
        ProperFormChangeOfVariables.model_validate_json(encode_strict_json(payload))
