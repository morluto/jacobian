"""Result-sensitive exact rational generating-function expansions."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics._recurrence_models import (
    RationalGeneratingFunctionCoefficientsRequest,
    RationalGeneratingFunctionCoefficientsResult,
)
from jacobian.math.combinatorics.operations import (
    rational_generating_function_coefficients,
)

ONE = CanonicalRational(num="1", den="1")
MINUS_ONE = CanonicalRational(num="-1", den="1")


def _expand(
    denominator: tuple[CanonicalRational, ...], order: int
) -> RationalGeneratingFunctionCoefficientsResult:
    return rational_generating_function_coefficients(
        (ONE,), denominator, "ASCENDING_POWERS_OF_X", "0", order
    )


def test_degree_32_recurrence_admits_a_prefix_above_the_old_ceiling() -> None:
    order = 4_096
    denominator = (ONE, *((MINUS_ONE,) * 32))
    request = RationalGeneratingFunctionCoefficientsRequest(
        numerator=(ONE,),
        denominator=denominator,
        coefficient_convention="ASCENDING_POWERS_OF_X",
        expansion_point="0",
        truncation_order=order,
    )

    result = _expand(request.denominator, request.truncation_order)
    values = tuple(value.as_fraction() for value in result.coefficients)
    denominator_values = tuple(value.as_fraction() for value in denominator)

    assert len(values) == order
    assert values[-1] == sum(values[-33:-1])
    assert all(
        sum(
            denominator_values[offset] * values[degree - offset]
            for offset in range(min(degree, len(denominator_values) - 1) + 1)
        )
        == (1 if degree == 0 else 0)
        for degree in range(order)
    )
    assert all(value.num == "0" for value in result.residual_coefficients)
    assert (
        RationalGeneratingFunctionCoefficientsResult.model_validate(result.model_dump())
        == result
    )


def test_denominator_degree_controls_the_recurrence_work_envelope() -> None:
    denominator = (ONE, *((MINUS_ONE,) * 32))

    with pytest.raises(OperationDomainValidationError, match="exact work bound"):
        _expand(denominator, 8_000)


def test_minimum_result_size_rejects_before_materializing_a_large_prefix() -> None:
    with pytest.raises(OperationDomainValidationError, match="bounded result limit"):
        _expand((ONE,), 250_000)
