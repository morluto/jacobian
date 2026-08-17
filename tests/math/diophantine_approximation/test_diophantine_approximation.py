"""Domain tests for exact Diophantine approximation operations."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from jacobian.canonical import parse_canonical_integer
from jacobian.math.diophantine_approximation import (
    continued_fraction,
    convergents,
    solve_pell,
)
from jacobian.math.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ConvergentRequest,
    PellEquationRequest,
)
from jacobian.math.diophantine_approximation._operations import (
    compute_continued_fraction,
    compute_convergents,
    compute_pell_equation,
)


def test_continued_fraction_sqrt_2() -> None:
    """sqrt(2) = [1; 2, 2, 2, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=5)
    )
    assert result.coefficients == (1, 2, 2, 2, 2)
    assert result.preperiod_length == 1
    assert result.period_length == 1
    assert result.method == "SYMPY_CONTINUED_FRACTION"


def test_continued_fraction_sqrt_3() -> None:
    """sqrt(3) = [1; 1, 2, 1, 2, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=3, term_count=6)
    )
    assert result.coefficients == (1, 1, 2, 1, 2, 1)
    assert result.preperiod_length == 1
    assert result.period_length == 2


def test_continued_fraction_sqrt_5() -> None:
    """sqrt(5) = [2; 4, 4, 4, ...]"""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=5, term_count=5)
    )
    assert result.coefficients[0] == 2
    assert all(c == 4 for c in result.coefficients[1:])


def test_continued_fraction_expands_period_to_max_terms() -> None:
    """A one-term period still produces every requested coefficient."""
    result = compute_continued_fraction(
        ContinuedFractionRequest(discriminant=2, term_count=500)
    )
    assert len(result.coefficients) == 500
    assert result.coefficients[0] == 1
    assert all(c == 2 for c in result.coefficients[1:])


def test_convergents_sqrt_2() -> None:
    """Convergents of sqrt(2): 1/1, 3/2, 7/5, 17/12, 41/29."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=5))
    assert len(result.convergents) == 5
    nums = [c.numerator for c in result.convergents]
    dens = [c.denominator for c in result.convergents]
    assert nums == ["1", "3", "7", "17", "41"]
    assert dens == ["1", "2", "5", "12", "29"]


def test_convergents_repeat_period_beyond_fixed_window() -> None:
    """Regression: a period of length one must expand for any convergent count."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=12))
    assert [c.index for c in result.convergents] == list(range(12))
    assert [c.numerator for c in result.convergents] == [
        "1",
        "3",
        "7",
        "17",
        "41",
        "99",
        "239",
        "577",
        "1393",
        "3363",
        "8119",
        "19601",
    ]
    assert [c.denominator for c in result.convergents] == [
        "1",
        "2",
        "5",
        "12",
        "29",
        "70",
        "169",
        "408",
        "985",
        "2378",
        "5741",
        "13860",
    ]


def test_convergents_expand_to_max_count() -> None:
    result = compute_convergents(
        ConvergentRequest(discriminant=2, convergent_count=500)
    )
    assert len(result.convergents) == 500
    assert [c.index for c in result.convergents] == list(range(500))


def test_convergents_are_best_approximations() -> None:
    """Each convergent p/q satisfies |p^2 - D*q^2| < 2*sqrt(D)."""
    discriminant = 2
    result = compute_convergents(
        ConvergentRequest(discriminant=discriminant, convergent_count=10)
    )
    for conv in result.convergents:
        p = parse_canonical_integer(conv.numerator)
        q = parse_canonical_integer(conv.denominator)
        assert abs(p**2 - discriminant * q**2) < 2 * math.sqrt(discriminant)


def test_pell_equation_sqrt_2() -> None:
    """x^2 - 2*y^2 = 1 has fundamental solution (3, 2)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=2))
    assert result.x == "3"
    assert result.y == "2"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 2 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_3() -> None:
    """x^2 - 3*y^2 = 1 has fundamental solution (2, 1)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=3))
    assert result.x == "2"
    assert result.y == "1"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 3 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_5() -> None:
    """x^2 - 5*y^2 = 1 has fundamental solution (9, 4)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=5))
    assert result.x == "9"
    assert result.y == "4"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 5 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_sqrt_13() -> None:
    """x^2 - 13*y^2 = 1 has fundamental solution (649, 180)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=13))
    assert result.x == "649"
    assert result.y == "180"
    assert (
        parse_canonical_integer(result.x) ** 2
        - 13 * parse_canonical_integer(result.y) ** 2
        == 1
    )


def test_pell_equation_all_verified() -> None:
    """Every Pell solution satisfies x^2 - D*y^2 = 1."""
    for discriminant in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        result = compute_pell_equation(PellEquationRequest(discriminant=discriminant))
        x = parse_canonical_integer(result.x)
        y = parse_canonical_integer(result.y)
        assert x**2 - discriminant * y**2 == 1


def test_pell_equation_large_discriminant() -> None:
    """The derived period bound reaches a large fundamental solution exactly."""
    result = compute_pell_equation(PellEquationRequest(discriminant=991))
    x = parse_canonical_integer(result.x)
    y = parse_canonical_integer(result.y)
    assert x**2 - 991 * y**2 == 1


def test_pell_equation_long_period() -> None:
    """The longest period below the bound still reaches the fundamental solution."""
    result = compute_pell_equation(PellEquationRequest(discriminant=9949))
    x = parse_canonical_integer(result.x)
    y = parse_canonical_integer(result.y)
    assert x**2 - 9949 * y**2 == 1


def test_contract_rejects_non_squarefree() -> None:
    with pytest.raises(ValidationError, match="squarefree"):
        ContinuedFractionRequest(discriminant=4, term_count=5)


def test_contract_rejects_perfect_square() -> None:
    with pytest.raises(ValidationError, match="squarefree"):
        ContinuedFractionRequest(discriminant=9, term_count=5)


def test_contract_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ContinuedFractionRequest(discriminant=1, term_count=5)


def test_public_kernels_reject_perfect_square() -> None:
    with pytest.raises(ValueError, match="perfect square"):
        continued_fraction(4, 5)
    with pytest.raises(ValueError, match="perfect square"):
        convergents(9, 3)
    with pytest.raises(ValueError, match="perfect square"):
        solve_pell(16)


def test_public_kernels_return_typed_values() -> None:
    assert continued_fraction(2, 3) == ([1, 2, 2], 1, 1)
    assert convergents(2, 3) == [(0, 1, 1), (1, 3, 2), (2, 7, 5)]
    assert solve_pell(2) == (3, 2)
