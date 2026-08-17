from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.contracts.diophantine_approximation import (
    ContinuedFractionRequest,
    ConvergentRequest,
    PellEquationRequest,
)
from jacobian.domains.diophantine_approximation.operations import (
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


def test_convergents_sqrt_2() -> None:
    """Convergents of sqrt(2): 1/1, 3/2, 7/5, 17/12, 41/29."""
    result = compute_convergents(ConvergentRequest(discriminant=2, convergent_count=5))
    assert len(result.convergents) == 5
    nums = [c.numerator for c in result.convergents]
    dens = [c.denominator for c in result.convergents]
    assert nums == [1, 3, 7, 17, 41]
    assert dens == [1, 2, 5, 12, 29]


def test_convergents_are_best_approximations() -> None:
    """Each convergent p/q satisfies |p^2 - D*q^2| < 2*sqrt(D)."""
    discriminant = 2
    result = compute_convergents(
        ConvergentRequest(discriminant=discriminant, convergent_count=10)
    )
    import math

    for conv in result.convergents:
        val = abs(conv.numerator**2 - discriminant * conv.denominator**2)
        assert val < 2 * math.sqrt(discriminant)


def test_pell_equation_sqrt_2() -> None:
    """x^2 - 2*y^2 = 1 has fundamental solution (3, 2)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=2))
    assert result.x == 3
    assert result.y == 2
    assert result.x**2 - 2 * result.y**2 == 1


def test_pell_equation_sqrt_3() -> None:
    """x^2 - 3*y^2 = 1 has fundamental solution (2, 1)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=3))
    assert result.x == 2
    assert result.y == 1
    assert result.x**2 - 3 * result.y**2 == 1


def test_pell_equation_sqrt_5() -> None:
    """x^2 - 5*y^2 = 1 has fundamental solution (9, 4)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=5))
    assert result.x == 9
    assert result.y == 4
    assert result.x**2 - 5 * result.y**2 == 1


def test_pell_equation_sqrt_13() -> None:
    """x^2 - 13*y^2 = 1 has fundamental solution (649, 180)."""
    result = compute_pell_equation(PellEquationRequest(discriminant=13))
    assert result.x == 649
    assert result.y == 180
    assert result.x**2 - 13 * result.y**2 == 1


def test_pell_equation_all_verified() -> None:
    """Every Pell solution satisfies x^2 - D*y^2 = 1."""
    for discriminant in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
        result = compute_pell_equation(PellEquationRequest(discriminant=discriminant))
        assert result.x**2 - discriminant * result.y**2 == 1


def test_contract_rejects_non_squarefree() -> None:
    with pytest.raises(ValidationError, match="squarefree"):
        ContinuedFractionRequest(discriminant=4, term_count=5)


def test_contract_rejects_perfect_square() -> None:
    with pytest.raises(ValidationError, match="squarefree"):
        ContinuedFractionRequest(discriminant=9, term_count=5)


def test_contract_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        ContinuedFractionRequest(discriminant=1, term_count=5)
