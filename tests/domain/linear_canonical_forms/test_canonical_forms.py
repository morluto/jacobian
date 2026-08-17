from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian.contracts.canonical_forms import MonicPolynomial, SquareMatrixRequest
from jacobian.contracts.exact import CanonicalRational
from jacobian.contracts.matrices import RationalMatrix
from jacobian.domains.linear_canonical_forms.operations import (
    compute_minimal_polynomial,
    compute_primary_decomposition,
    compute_rational_canonical_form,
)

R = CanonicalRational


def _mat(*rows: tuple[str, ...]) -> SquareMatrixRequest:
    entries = tuple(tuple(R(num=n, den=d) for n, d in row) for row in rows)
    return SquareMatrixRequest(matrix=RationalMatrix(entries=entries))


def _pair(n: str, d: str) -> tuple[str, str]:
    return (n, d)


def test_nilpotent_jordan_block_minimal_polynomial_is_t_squared() -> None:
    """Matrix [[0,1],[0,0]] has minimal polynomial t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    coeffs = [c.as_fraction() for c in result.minimal_polynomial.coefficients]
    assert coeffs == [Fraction(0), Fraction(0), Fraction(1)]
    assert result.degree == 2


def test_diagonal_distinct_minimal_equals_characteristic() -> None:
    """diag(2,3) has minimal polynomial (t-2)(t-3) = t^2 - 5t + 6."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_minimal_polynomial(req)
    coeffs = [c.as_fraction() for c in result.minimal_polynomial.coefficients]
    assert coeffs == [Fraction(6), Fraction(-5), Fraction(1)]
    char_coeffs = [
        c.as_fraction() for c in result.characteristic_polynomial.coefficients
    ]
    assert char_coeffs == [Fraction(6), Fraction(-5), Fraction(1)]


def test_jordan_block_minimal_equals_characteristic() -> None:
    """[[2,1],[0,2]] has minimal polynomial (t-2)^2 = t^2 - 4t + 4."""
    req = _mat(
        (_pair("2", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("2", "1")),
    )
    result = compute_minimal_polynomial(req)
    coeffs = [c.as_fraction() for c in result.minimal_polynomial.coefficients]
    assert coeffs == [Fraction(4), Fraction(-4), Fraction(1)]
    char_coeffs = [
        c.as_fraction() for c in result.characteristic_polynomial.coefficients
    ]
    assert char_coeffs == [Fraction(4), Fraction(-4), Fraction(1)]


def test_identity_matrix_minimal_polynomial_is_t_minus_one() -> None:
    """2x2 identity has minimal polynomial t - 1."""
    req = _mat(
        (_pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "1")),
    )
    result = compute_minimal_polynomial(req)
    coeffs = [c.as_fraction() for c in result.minimal_polynomial.coefficients]
    assert coeffs == [Fraction(-1), Fraction(1)]


def test_irreducible_over_qq_minimal_polynomial() -> None:
    """[[0,-1],[1,0]] has minimal polynomial t^2 + 1 (irreducible over QQ)."""
    req = _mat(
        (_pair("0", "1"), _pair("-1", "1")),
        (_pair("1", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    coeffs = [c.as_fraction() for c in result.minimal_polynomial.coefficients]
    assert coeffs == [Fraction(1), Fraction(0), Fraction(1)]


def test_nilpotent_single_block_canonical_form() -> None:
    """[[0,1],[0,0]] has one invariant factor t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    coeffs = [c.as_fraction() for c in result.invariant_factors[0].factor.coefficients]
    assert coeffs == [Fraction(0), Fraction(0), Fraction(1)]
    assert result.invariant_factors[0].block_size == 2
    assert result.total_block_size == 2


def test_diagonal_distinct_single_factor_canonical_form() -> None:
    """diag(2,3) has one invariant factor (t-2)(t-3)."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    coeffs = [c.as_fraction() for c in result.invariant_factors[0].factor.coefficients]
    assert coeffs == [Fraction(6), Fraction(-5), Fraction(1)]


def test_identity_two_blocks_canonical_form() -> None:
    """2x2 identity has invariant factors (t-1), (t-1)."""
    req = _mat(
        (_pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 2
    assert result.total_block_size == 2
    for entry in result.invariant_factors:
        coeffs = [c.as_fraction() for c in entry.factor.coefficients]
        assert coeffs == [Fraction(-1), Fraction(1)]


def test_nilpotent_two_blocks_divisibility_chain() -> None:
    """Nilpotent with blocks of sizes 2 and 1: invariant factors t | t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 2
    sizes = [entry.block_size for entry in result.invariant_factors]
    assert sizes == [1, 2]


def test_primary_decomposition_distinct_linear_factors() -> None:
    """diag(2,3) decomposes into (t-2) and (t-3)."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 2
    for comp in result.components:
        coeffs = [c.as_fraction() for c in comp.coefficients]
        assert len(coeffs) == 2


def test_primary_decomposition_irreducible_power() -> None:
    """[[0,1],[0,0]] has minpoly t^2, primary decomposition is [t^2]."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 1
    coeffs = [c.as_fraction() for c in result.components[0].coefficients]
    assert coeffs == [Fraction(0), Fraction(0), Fraction(1)]


def test_contract_rejects_nonsquare() -> None:
    with pytest.raises(ValidationError, match="square"):
        SquareMatrixRequest(
            matrix=RationalMatrix(entries=((R(num="1", den="1"), R(num="0", den="1")),))
        )


def test_contract_rejects_non_monic_polynomial() -> None:
    with pytest.raises(ValidationError, match="monic"):
        MonicPolynomial(coefficients=(R(num="1", den="1"), R(num="2", den="1")))


def test_characteristic_equals_product_of_invariant_factors() -> None:
    """Product of invariant factors equals the characteristic polynomial."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    # Product of invariant factor degrees should equal matrix size (3)
    assert result.total_block_size == 3
    # Characteristic polynomial should be t^3
    char_coeffs = [
        c.as_fraction() for c in result.characteristic_polynomial.coefficients
    ]
    assert char_coeffs == [Fraction(0), Fraction(0), Fraction(0), Fraction(1)]


def test_method_tags() -> None:
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    assert compute_minimal_polynomial(req).method == "KRYLOV_NULLSPACE"
    assert compute_rational_canonical_form(req).method == "SMITH_NORMAL_FORM"
    assert compute_primary_decomposition(req).method == "FACTOR_LCM"
