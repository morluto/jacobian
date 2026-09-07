"""Large semigroup values stay native while every exact coordinate round-trips."""

import pytest
from pydantic import ValidationError

from jacobian._exact import MAX_CANONICAL_INTEGER_DIGITS
from jacobian.math.number_theory.numerical_semigroups._factorization_models import (
    FactorizationComputeRequest,
    FactorizationComputeResult,
)
from jacobian.math.number_theory.numerical_semigroups._tools import (
    compute_factorizations,
)
from jacobian.math.number_theory.numerical_semigroups.operations import membership


def test_free_axis_large_factorization_composes_through_json() -> None:
    value = 2**53 + 1
    request = FactorizationComputeRequest(generators=(1,), value=value)
    result = compute_factorizations(request)
    assert result.value == value
    assert result.minimal_generators == (1,)
    assert result.factorizations == ((value,),)
    assert result.model_dump()["factorizations"] == ((value,),)
    assert result.model_dump(mode="json")["factorizations"] == [[str(value)]]
    assert (
        FactorizationComputeRequest.model_validate_json(request.model_dump_json())
        == request
    )
    assert (
        FactorizationComputeResult.model_validate_json(result.model_dump_json())
        == result
    )


def test_free_axis_admits_the_exact_output_envelope() -> None:
    value = 10**MAX_CANONICAL_INTEGER_DIGITS - 1
    result = membership((1,), value)
    assert result.in_semigroup
    assert result.value == value
    assert type(result).model_validate_json(result.model_dump_json()) == result
    for sign in (-1, 1):
        with pytest.raises(ValueError, match="output digit bound"):
            membership((1,), sign * (value + 1))


@pytest.mark.parametrize("value", ["42", True, 42.0])
def test_native_element_does_not_decode_or_coerce(value: object) -> None:
    with pytest.raises(ValidationError):
        FactorizationComputeRequest.model_validate({"generators": (1,), "value": value})
