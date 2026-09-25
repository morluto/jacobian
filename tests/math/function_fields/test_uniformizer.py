"""Uniformizers checked against independent polynomial valuation arithmetic."""

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.function_fields._models import (
    FiniteFunctionField,
    FunctionFieldPlace,
    PrimeFieldPolynomial,
    PrimeFieldRationalFunction,
)
from jacobian.math.function_fields.operations import function_field_place_uniformizer


def _field(p: int = 5) -> FiniteFunctionField:
    one = PrimeFieldPolynomial(characteristic=p, coefficients=(1,))
    return FiniteFunctionField(
        characteristic=p,
        defining_polynomial=(
            PrimeFieldRationalFunction(numerator=one, denominator=one),
        ),
    )


def _place(coefficients: tuple[int, ...]) -> FunctionFieldPlace:
    polynomial = PrimeFieldPolynomial(characteristic=5, coefficients=coefficients)
    return FunctionFieldPlace(
        field=_field(),
        kind="FINITE",
        prime_polynomial=polynomial,
        degree=polynomial.degree,
    )


def _independent_remainder(
    dividend: tuple[int, ...], divisor: tuple[int, ...], p: int
) -> tuple[int, ...]:
    """Naive coefficient elimination, independent of the operation kernel."""
    remainder = list(dividend)
    while len(remainder) >= len(divisor):
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
        if len(remainder) < len(divisor):
            break
        shift = len(remainder) - len(divisor)
        scale = remainder[-1] * pow(divisor[-1], -1, p) % p
        for i, coefficient in enumerate(divisor):
            remainder[shift + i] = (remainder[shift + i] - scale * coefficient) % p
    while len(remainder) > 1 and remainder[-1] == 0:
        remainder.pop()
    return tuple(remainder)


def _independent_order(poly: tuple[int, ...], q: tuple[int, ...], p: int) -> int:
    order = 0
    value = poly
    while value != (0,) and _independent_remainder(value, q, p) == (0,):
        # Exact quotient via coefficient elimination on the monic divisor.
        quotient = [0] * (len(value) - len(q) + 1)
        work = list(value)
        while len(work) >= len(q):
            while len(work) > 1 and work[-1] == 0:
                work.pop()
            if len(work) < len(q):
                break
            shift = len(work) - len(q)
            scale = work[-1] * pow(q[-1], -1, p) % p
            quotient[shift] = scale
            for i, coefficient in enumerate(q):
                work[shift + i] = (work[shift + i] - scale * coefficient) % p
        while len(quotient) > 1 and quotient[-1] == 0:
            quotient.pop()
        value = tuple(quotient)
        order += 1
    return order


@pytest.mark.parametrize("coefficients", [(2, 2), (4, 0, 2)])
def test_finite_uniformizer_has_independent_q_adic_order_one(
    coefficients: tuple[int, ...],
) -> None:
    result = function_field_place_uniformizer(_place(coefficients))
    q = result.place.prime_polynomial.coefficients
    rational = result.uniformizer.coordinates[0]
    assert _independent_order(rational.numerator.coefficients, q, 5) == 1
    assert _independent_order(rational.denominator.coefficients, q, 5) == 0


def test_infinite_uniformizer_has_degree_difference_one() -> None:
    place = FunctionFieldPlace(field=_field(), kind="INFINITE", degree=1)
    result = function_field_place_uniformizer(place)
    rational = result.uniformizer.coordinates[0]
    assert rational.numerator.coefficients == (1,)
    assert rational.denominator.coefficients == (0, 1)
    assert (
        len(rational.denominator.coefficients) - len(rational.numerator.coefficients)
        == 1
    )


def test_reducible_polynomial_is_not_a_place() -> None:
    with pytest.raises(OperationDomainValidationError, match="irreducible"):
        function_field_place_uniformizer(_place((1, 0, 1)))
