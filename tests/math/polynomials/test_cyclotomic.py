"""Exact FLINT cyclotomic polynomial operation."""

from flint import fmpz_poly

from jacobian.math.polynomials._cyclotomic import (
    CyclotomicRequest,
    CyclotomicResult,
    cyclotomic,
)


def ascending(result: CyclotomicResult) -> list[int]:
    return list(reversed(result.polynomial.coefficients))


def test_known_twelfth_cyclotomic() -> None:
    result = cyclotomic(CyclotomicRequest(index=12))
    assert result.source_index == 12
    assert result.totient == 4
    assert result.polynomial.coefficients == (1, 0, -1, 0, 1)


def test_divisor_product_identity_through_twenty() -> None:
    for index in range(1, 21):
        product = fmpz_poly([1])
        for divisor in range(1, index + 1):
            if index % divisor == 0:
                result = cyclotomic(CyclotomicRequest(index=divisor))
                product *= fmpz_poly(ascending(result))
        expected = fmpz_poly([-1] + [0] * (index - 1) + [1])
        assert product == expected
