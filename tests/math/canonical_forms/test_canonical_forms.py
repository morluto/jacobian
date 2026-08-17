from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.canonical_forms import (
    invariant_factors,
    minimal_polynomial,
    primary_decomposition,
)
from jacobian.math.canonical_forms._models import MonicPolynomial, SquareMatrixRequest
from jacobian.math.canonical_forms._operations import (
    compute_minimal_polynomial,
    compute_primary_decomposition,
    compute_rational_canonical_form,
)
from jacobian.math.matrices.values import RationalMatrix

R = CanonicalRational


def _mat(*rows: tuple[tuple[str, str], ...]) -> SquareMatrixRequest:
    entries = tuple(tuple(R(num=num, den=den) for num, den in row) for row in rows)
    return SquareMatrixRequest(matrix=RationalMatrix(entries=entries))


def _coeffs(poly: MonicPolynomial) -> list[Fraction]:
    return [coefficient.as_fraction() for coefficient in poly.coefficients]


def _pair(num: str, den: str) -> tuple[str, str]:
    return (num, den)


def _diagonal(*values: str) -> SquareMatrixRequest:
    entries = tuple(
        tuple(
            R(num=value if row == column else "0", den="1")
            for column, value in enumerate(values)
        )
        for row in range(len(values))
    )
    return SquareMatrixRequest(matrix=RationalMatrix(entries=entries))


def test_nilpotent_jordan_block_minimal_polynomial_is_t_squared() -> None:
    """Matrix [[0,1],[0,0]] has minimal polynomial t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(0), Fraction(0), Fraction(1)]
    assert result.degree == 2


def test_diagonal_distinct_minimal_equals_characteristic() -> None:
    """diag(2,3) has minimal polynomial (t-2)(t-3) = t^2 - 5t + 6."""
    req = _mat(
        (_pair("2", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("3", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]


def test_jordan_block_minimal_equals_characteristic() -> None:
    """[[2,1],[0,2]] has minimal polynomial (t-2)^2 = t^2 - 4t + 4."""
    req = _mat(
        (_pair("2", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("2", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [
        Fraction(4),
        Fraction(-4),
        Fraction(1),
    ]
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(4),
        Fraction(-4),
        Fraction(1),
    ]


def test_identity_matrix_minimal_polynomial_is_t_minus_one() -> None:
    """2x2 identity has minimal polynomial t - 1."""
    req = _mat(
        (_pair("1", "1"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(-1), Fraction(1)]


def test_irreducible_over_qq_minimal_polynomial() -> None:
    """[[0,-1],[1,0]] has minimal polynomial t^2 + 1 (irreducible over QQ)."""
    req = _mat(
        (_pair("0", "1"), _pair("-1", "1")),
        (_pair("1", "1"), _pair("0", "1")),
    )
    result = compute_minimal_polynomial(req)
    assert _coeffs(result.minimal_polynomial) == [Fraction(1), Fraction(0), Fraction(1)]


def test_nilpotent_single_block_canonical_form() -> None:
    """[[0,1],[0,0]] has one invariant factor t^2."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    assert _coeffs(result.invariant_factors[0].factor) == [
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]
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
    assert _coeffs(result.invariant_factors[0].factor) == [
        Fraction(6),
        Fraction(-5),
        Fraction(1),
    ]


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
        assert _coeffs(entry.factor) == [Fraction(-1), Fraction(1)]


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
    for component in result.components:
        assert len(_coeffs(component)) == 2


def test_primary_decomposition_irreducible_power() -> None:
    """[[0,1],[0,0]] has minpoly t^2, primary decomposition is [t^2]."""
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    result = compute_primary_decomposition(req)
    assert len(result.components) == 1
    assert _coeffs(result.components[0]) == [Fraction(0), Fraction(0), Fraction(1)]


def test_primary_decomposition_normalizes_rational_root_factors() -> None:
    """diag(1/2, 1/3) decomposes into the monic factors (t-1/2) and (t-1/3)."""
    req = _mat(
        (_pair("1", "2"), _pair("0", "1")),
        (_pair("0", "1"), _pair("1", "3")),
    )
    result = compute_primary_decomposition(req)
    assert sorted(_coeffs(component) for component in result.components) == [
        [Fraction(-1, 2), Fraction(1)],
        [Fraction(-1, 3), Fraction(1)],
    ]


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
    assert result.total_block_size == 3
    assert _coeffs(result.characteristic_polynomial) == [
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
    ]


def test_method_tags() -> None:
    req = _mat(
        (_pair("0", "1"), _pair("1", "1")),
        (_pair("0", "1"), _pair("0", "1")),
    )
    assert compute_minimal_polynomial(req).method == "KRYLOV_NULLSPACE"
    assert compute_rational_canonical_form(req).method == "SMITH_NORMAL_FORM"
    assert compute_primary_decomposition(req).method == "FACTOR_LCM"


def test_smith_normal_form_path_handles_larger_matrices() -> None:
    """A 6x6 diagonal matrix uses the maintained Smith form, not minor enumeration."""
    req = _diagonal("2", "3", "5", "7", "11", "13")
    result = compute_rational_canonical_form(req)
    assert len(result.invariant_factors) == 1
    assert result.invariant_factors[0].block_size == 6
    assert result.total_block_size == 6
    characteristic = _coeffs(result.characteristic_polynomial)
    assert len(characteristic) == 7
    assert characteristic[-1] == Fraction(1)
    assert characteristic[0] == Fraction(2 * 3 * 5 * 7 * 11 * 13)
    assert characteristic[-2] == Fraction(-(2 + 3 + 5 + 7 + 11 + 13))
    assert _coeffs(result.minimal_polynomial) == characteristic


def test_single_entry_matrix_canonical_forms() -> None:
    req = _mat((_pair("2", "1"),))
    assert _coeffs(compute_minimal_polynomial(req).minimal_polynomial) == [
        Fraction(-2),
        Fraction(1),
    ]

    rcf = compute_rational_canonical_form(req)
    assert len(rcf.invariant_factors) == 1
    assert rcf.invariant_factors[0].block_size == 1
    assert rcf.total_block_size == 1
    assert _coeffs(rcf.invariant_factors[0].factor) == [Fraction(-2), Fraction(1)]

    decomposition = compute_primary_decomposition(req)
    assert len(decomposition.components) == 1
    assert _coeffs(decomposition.components[0]) == [Fraction(-2), Fraction(1)]


def test_contract_rejects_oversized_and_wide_scalar_matrices() -> None:
    identity_17 = tuple(
        tuple(R(num="1" if row == column else "0", den="1") for column in range(17))
        for row in range(17)
    )
    with pytest.raises(ValidationError, match="16 x 16"):
        SquareMatrixRequest(matrix=RationalMatrix(entries=identity_17))

    wide_scalar = ((R(num="1" + "0" * 256, den="1"),),)
    with pytest.raises(ValidationError, match="256 decimal digits"):
        SquareMatrixRequest(matrix=RationalMatrix(entries=wide_scalar))


def test_public_kernels_return_monic_coefficient_lists() -> None:
    entries = (
        (Fraction(0), Fraction(1)),
        (Fraction(0), Fraction(0)),
    )
    assert minimal_polynomial(entries) == (Fraction(0), Fraction(0), Fraction(1))
    assert invariant_factors(entries) == ((Fraction(0), Fraction(0), Fraction(1)),)
    assert primary_decomposition(entries) == ((Fraction(0), Fraction(0), Fraction(1)),)
