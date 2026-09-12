"""Exact continued fraction and Pell equation kernels backed by SymPy."""

from __future__ import annotations

from math import isqrt

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.arithmetic._integer_predicates import is_square_free
from jacobian.math.number_theory.diophantine_approximation._models import (
    _MAX_DISCRIMINANT,
    _MAX_TERMS,
    ContinuedFractionResult,
    ConvergentResult,
    ConvergentValue,
    PellEquationResult,
)

__all__ = ["continued_fraction", "convergents", "solve_pell"]


def _require_periodic_discriminant(discriminant: int) -> None:
    """Reject discriminants whose sqrt is not a periodic quadratic surd."""
    if type(discriminant) is not int or not 2 <= discriminant <= _MAX_DISCRIMINANT:
        raise OperationDomainValidationError(
            location=("discriminant",),
            code="diophantine.discriminant_out_of_range",
            message=(
                f"discriminant must be an integer between 2 and {_MAX_DISCRIMINANT}"
            ),
        )
    root = isqrt(discriminant)
    if root * root == discriminant:
        raise OperationDomainValidationError(
            location=("discriminant",),
            code="diophantine.discriminant_must_not_be_square",
            message="discriminant must not be a perfect square",
        )
    if not is_square_free(discriminant):
        raise OperationDomainValidationError(
            location=("discriminant",),
            code="diophantine.discriminant_must_be_squarefree",
            message="discriminant must be squarefree",
        )


def _cf_coefficients(discriminant: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (preperiod, period) of the continued fraction of sqrt(D)."""
    from sympy import continued_fraction_periodic

    _require_periodic_discriminant(discriminant)

    expansion = continued_fraction_periodic(0, 1, discriminant)
    head = expansion[0]
    preperiod = (
        tuple(int(item) for item in head) if isinstance(head, list) else (int(head),)
    )
    tail = expansion[1]
    period = (
        tuple(int(item) for item in tail) if isinstance(tail, list) else (int(tail),)
    )
    return preperiod, period


def _coefficients(
    preperiod: tuple[int, ...],
    period: tuple[int, ...],
    count: int,
) -> list[int]:
    """Return exactly count terms by repeating the period after the preperiod."""
    coefficients: list[int] = []
    for index in range(count):
        coefficients.append(
            preperiod[index]
            if index < len(preperiod)
            else period[(index - len(preperiod)) % len(period)]
        )
    return coefficients


def continued_fraction(
    discriminant: int,
    term_count: int,
) -> ContinuedFractionResult:
    """Return the continued fraction expansion of sqrt(D)."""
    if type(term_count) is not int or not 1 <= term_count <= _MAX_TERMS:
        raise OperationDomainValidationError(
            location=("term_count",),
            code="diophantine.term_count_out_of_range",
            message=f"term_count must be between 1 and {_MAX_TERMS}",
        )
    preperiod, period = _cf_coefficients(discriminant)
    return ContinuedFractionResult._from_kernel(
        discriminant=discriminant,
        term_count=term_count,
        coefficients=tuple(_coefficients(preperiod, period, term_count)),
        preperiod_length=len(preperiod),
        period_length=len(period),
    )


def convergents(discriminant: int, count: int) -> ConvergentResult:
    """Return the first count convergents (index, p_n, q_n) of sqrt(D)."""
    if type(count) is not int or not 1 <= count <= _MAX_TERMS:
        raise OperationDomainValidationError(
            location=("count",),
            code="diophantine.convergent_count_out_of_range",
            message=f"count must be between 1 and {_MAX_TERMS}",
        )
    preperiod, period = _cf_coefficients(discriminant)
    coefficients = _coefficients(preperiod, period, count)

    p_prev2, p_prev1 = 1, coefficients[0]
    q_prev2, q_prev1 = 0, 1

    values = [(0, p_prev1, q_prev1)]
    for index in range(1, count):
        coefficient = coefficients[index]
        p_current = coefficient * p_prev1 + p_prev2
        q_current = coefficient * q_prev1 + q_prev2
        p_prev2, p_prev1 = p_prev1, p_current
        q_prev2, q_prev1 = q_prev1, q_current
        values.append((index, p_prev1, q_prev1))

    return ConvergentResult._from_kernel(
        discriminant=discriminant,
        convergent_count=count,
        convergents=tuple(
            ConvergentValue(
                index=index,
                numerator=numerator,
                denominator=denominator,
            )
            for index, numerator, denominator in values
        ),
    )


def solve_pell(discriminant: int) -> PellEquationResult:
    """Return the fundamental solution (x, y) to x^2 - D*y^2 = 1.

    For a non-square positive integer D the continued fraction of sqrt(D) has
    period length r, and the fundamental solution is p_{r-1}/q_{r-1} when r is
    even and p_{2r-1}/q_{2r-1} when r is odd. Iterating the first 2r
    convergents therefore always reaches the first solution.
    """
    preperiod, period = _cf_coefficients(discriminant)
    period_length = len(period)
    convergents_needed = 2 * period_length

    coefficients = _coefficients(preperiod, period, convergents_needed)
    p_prev2, p_prev1 = 1, coefficients[0]
    q_prev2, q_prev1 = 0, 1

    if p_prev1**2 - discriminant * q_prev1**2 == 1:
        return PellEquationResult._from_kernel(
            discriminant=discriminant,
            x=p_prev1,
            y=q_prev1,
        )

    for index in range(1, convergents_needed):
        coefficient = coefficients[index]
        p_current = coefficient * p_prev1 + p_prev2
        q_current = coefficient * q_prev1 + q_prev2
        p_prev2, p_prev1 = p_prev1, p_current
        q_prev2, q_prev1 = q_prev1, q_current

        if p_prev1**2 - discriminant * q_prev1**2 == 1:
            return PellEquationResult._from_kernel(
                discriminant=discriminant,
                x=p_prev1,
                y=q_prev1,
            )

    raise ArithmeticError("Pell solution was not reached within the period bound")
