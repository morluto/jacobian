"""Explicit polynomial/series maps compose canonical values unchanged."""

import pytest

from jacobian._exact import CanonicalRational
from jacobian.math.polynomials.series import from_polynomial, to_polynomial
from jacobian.math.polynomials.values import RationalPolynomial


@pytest.mark.parametrize(
    "terms",
    [
        [],
        [{"coefficient": {"num": 3, "den": 2}, "exponents": [0]}],
        [
            {"coefficient": {"num": 1, "den": 1}, "exponents": [32768]},
            {"coefficient": {"num": 1, "den": 1}, "exponents": [1]},
        ],
    ],
)
def test_canonical_sparse_projection(terms: list[dict[str, object]]) -> None:
    polynomial = RationalPolynomial.model_validate(
        {"variables": ["t"], "polynomial": {"terms": terms}}
    )
    projection = from_polynomial(polynomial, 3)
    decoded = type(projection).model_validate_json(projection.model_dump_json())
    lift = to_polynomial(decoded.result)
    assert lift.result.variables == ("t",)
    assert from_polynomial(lift.result, 3).result == decoded.result
    assert all(term.exponents[0] < 3 for term in lift.result.polynomial.terms)


def test_multivariate_conversion_requires_explicit_variable_map() -> None:
    polynomial = RationalPolynomial.model_validate(
        {"variables": ["x", "y"], "polynomial": {"terms": []}}
    )
    with pytest.raises(ValueError, match="univariate"):
        from_polynomial(polynomial, 3)


def test_degree_zero_series_keeps_its_order() -> None:
    polynomial = RationalPolynomial.model_validate(
        {"variables": ["x"], "polynomial": {"terms": []}}
    )
    projection = from_polynomial(polynomial, 512)
    assert projection.result.coefficients == (CanonicalRational(num=0, den=1),) * 512
    assert to_polynomial(projection.result).source.truncation_order == 512


@pytest.mark.parametrize("variable", ["X", "x_1", "a" * 32])
def test_variable_carrier_is_shared(variable: str) -> None:
    polynomial = RationalPolynomial.model_validate(
        {"variables": [variable], "polynomial": {"terms": []}}
    )
    assert to_polynomial(from_polynomial(polynomial, 1).result).result == polynomial
