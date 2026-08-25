"""Typed wire contracts for exact Diophantine approximation operations."""

from __future__ import annotations

from math import isqrt
from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.arithmetic._integer_predicates import is_square_free

_MAX_DISCRIMINANT = 1_000_000
_MAX_TERMS = 5_000


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


def _convergent_component_digit_cap(count: int) -> int:
    """Conservative decimal-digit bound for any sqrt(D) convergent component.

    Every partial quotient of ``sqrt(D)`` after the initial term is at most
    ``2 * isqrt(D)``, so with ``D <= _MAX_DISCRIMINANT`` every coefficient is
    at most ``2 * isqrt(_MAX_DISCRIMINANT)`` and each continuant grows by a
    factor below ``2 * isqrt(_MAX_DISCRIMINANT) + 1`` per step.  Components of
    convergents with index ``< count`` therefore stay strictly below
    ``(2 * isqrt(_MAX_DISCRIMINANT) + 1) ** count``, which this returns as an
    exact digit count so oversized canonical strings are rejected by length
    before any bigint work.
    """
    growth = 2 * isqrt(_MAX_DISCRIMINANT) + 1
    return len(str(growth**count))


class SquarefreeRequest(StrictModel):
    """One positive squarefree integer D for sqrt(D) operations."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self


class ContinuedFractionRequest(StrictModel):
    """Request the continued fraction expansion of sqrt(D) up to n terms."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    term_count: StrictInt = Field(ge=1, le=_MAX_TERMS)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self


class ContinuedFractionResult(StrictModel):
    """The continued fraction [a_0; a_1, ...] of sqrt(D).

    Retains the complete request so validation replays the canonical
    periodic expansion of the same ``sqrt(D)``: the coefficient prefix is
    exactly the preperiod followed by repeats of the fundamental period, and
    the reported preperiod/period lengths match that expansion.  A prefix
    shorter than preperiod + period is an operationally truncated window
    (its requested count is retained), never a complete-period claim.
    """

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    term_count: StrictInt = Field(ge=1, le=_MAX_TERMS)
    coefficients: tuple[StrictInt, ...] = Field(min_length=1, max_length=_MAX_TERMS)
    preperiod_length: StrictInt = Field(ge=1)
    period_length: StrictInt = Field(ge=1)
    method: Literal["SYMPY_CONTINUED_FRACTION"] = "SYMPY_CONTINUED_FRACTION"

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self

    @model_validator(mode="after")
    def require_source_expansion(self) -> Self:
        from jacobian.math.diophantine_approximation.operations import (
            _cf_coefficients,
            _coefficients,
        )

        preperiod, period = _cf_coefficients(self.discriminant)
        if self.preperiod_length != len(preperiod) or self.period_length != len(period):
            raise _validation_error(
                "diophantine_approximation.period_metadata_mismatch",
                "preperiod/period metadata must match the canonical expansion "
                "of sqrt(discriminant)",
            )
        if self.term_count != len(self.coefficients):
            raise _validation_error(
                "diophantine_approximation.coefficient_count_mismatch",
                "coefficient count must equal the requested term_count",
            )
        expected = tuple(_coefficients(preperiod, period, self.term_count))
        if tuple(self.coefficients) != expected:
            raise _validation_error(
                "diophantine_approximation.coefficients_not_canonical",
                "coefficients must be the canonical continued fraction of "
                "sqrt(discriminant)",
            )
        return self


class ConvergentRequest(StrictModel):
    """Request the first n convergents p_n/q_n of sqrt(D)."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    convergent_count: StrictInt = Field(ge=1, le=_MAX_TERMS)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self


class ConvergentValue(StrictModel):
    """One convergent p_n/q_n with index n."""

    index: StrictInt = Field(ge=0)
    numerator: CanonicalInteger
    denominator: CanonicalInteger


class ConvergentResult(StrictModel):
    """Convergents of sqrt(D).

    Retains the complete request so validation replays the continuant
    recurrence from the canonical coefficient stream of the same ``sqrt(D)``:
    indices are contiguous from zero, denominators are positive, each
    numerator/denominator pair is reduced, and adjacent pairs satisfy
    ``p_n q_{n-1} - p_{n-1} q_n = (-1)^(n+1)``.  Canonical numerators and
    denominators carry no more digits than the geometric bound implied by
    the admitted discriminant/count envelope.
    """

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    convergent_count: StrictInt = Field(ge=1, le=_MAX_TERMS)
    convergents: tuple[ConvergentValue, ...] = Field(
        min_length=1, max_length=_MAX_TERMS
    )
    method: Literal["CONTINUED_FRACTION_RECURSION"] = "CONTINUED_FRACTION_RECURSION"

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self

    @model_validator(mode="after")
    def require_source_convergents(self) -> Self:
        from math import gcd

        from jacobian.canonical import parse_canonical_integer
        from jacobian.math.diophantine_approximation.operations import (
            _cf_coefficients,
            _coefficients,
        )

        if len(self.convergents) != self.convergent_count:
            raise _validation_error(
                "diophantine_approximation.convergent_count_mismatch",
                "convergent count must equal the requested count",
            )
        if tuple(value.index for value in self.convergents) != tuple(
            range(len(self.convergents))
        ):
            raise _validation_error(
                "diophantine_approximation.indices_not_contiguous",
                "convergent indices must be contiguous starting at zero",
            )
        digit_cap = _convergent_component_digit_cap(self.convergent_count)
        for value in self.convergents:
            if (
                len(value.numerator.lstrip("-")) > digit_cap
                or len(value.denominator.lstrip("-")) > digit_cap
            ):
                raise _validation_error(
                    "diophantine_approximation.component_digit_bound_exceeded",
                    "convergent numerators/denominators exceed the "
                    f"{digit_cap}-digit bound implied by the admitted request",
                )
            numerator = parse_canonical_integer(value.numerator)
            denominator = parse_canonical_integer(value.denominator)
            if denominator <= 0:
                raise _validation_error(
                    "diophantine_approximation.denominator_not_positive",
                    "convergent denominators must be positive",
                )
            if gcd(numerator, denominator) != 1:
                raise _validation_error(
                    "diophantine_approximation.pair_not_reduced",
                    "convergent numerator/denominator pairs must be reduced",
                )

        preperiod, period = _cf_coefficients(self.discriminant)
        coefficients = _coefficients(preperiod, period, self.convergent_count)
        p_prev2, p_prev1 = 1, coefficients[0]
        q_prev2, q_prev1 = 0, 1
        for index, claimed in enumerate(self.convergents):
            if index > 0:
                coefficient = coefficients[index]
                p_prev2, p_prev1 = p_prev1, coefficient * p_prev1 + p_prev2
                q_prev2, q_prev1 = q_prev1, coefficient * q_prev1 + q_prev2
            if (
                parse_canonical_integer(claimed.numerator) != p_prev1
                or parse_canonical_integer(claimed.denominator) != q_prev1
            ):
                raise _validation_error(
                    "diophantine_approximation.recurrence_mismatch",
                    "convergents must replay the continuant recurrence of the "
                    "canonical coefficient stream",
                )
        return self


class PellEquationRequest(StrictModel):
    """Solve x^2 - D*y^2 = 1 for the fundamental solution."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self


class PellEquationResult(StrictModel):
    """The fundamental solution (x, y) to x^2 - D*y^2 = 1."""

    discriminant: StrictInt = Field(ge=2, le=_MAX_DISCRIMINANT)
    x: CanonicalInteger
    y: CanonicalInteger
    method: Literal["CONTINUED_FRACTION_CONVERGENTS"] = "CONTINUED_FRACTION_CONVERGENTS"

    @model_validator(mode="after")
    def require_squarefree(self) -> Self:
        if not is_square_free(self.discriminant):
            raise _validation_error(
                "diophantine_approximation.discriminant_not_squarefree",
                "discriminant must be squarefree",
            )
        return self


__all__ = [
    "ContinuedFractionRequest",
    "ContinuedFractionResult",
    "ConvergentRequest",
    "ConvergentResult",
    "ConvergentValue",
    "PellEquationRequest",
    "PellEquationResult",
    "SquarefreeRequest",
]
