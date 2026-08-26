"""Typed wire contracts for exact Diophantine approximation operations."""

from __future__ import annotations

from math import ceil, isqrt, log10
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
    # Do not materialize the bound itself: at the admitted maximum this would
    # need more decimal digits than Python permits converting by default.
    return ceil(count * log10(growth)) + 1


def _pell_component_digit_cap() -> int:
    """Conservative digit bound for the admitted fundamental Pell solution.

    A continued-fraction period of ``sqrt(D)`` has length at most
    ``2 * isqrt(D)``.  The kernel examines at most twice that many
    convergents, and every recurrence coefficient is below
    ``2 * isqrt(_MAX_DISCRIMINANT) + 1``.  This bounds a supplied exact Pell
    certificate before its canonical decimal components are parsed.
    """

    return _convergent_component_digit_cap(4 * isqrt(_MAX_DISCRIMINANT))


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
    """A bounded continued-fraction prefix of ``sqrt(D)``.

    Parsing establishes only the retained request and bounded carrier shape.
    Use :func:`verify_continued_fraction_result` for an independently
    supplied claim; kernel-produced results use :meth:`_from_kernel`.
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
    def require_bounded_shape(self) -> Self:
        if self.term_count != len(self.coefficients):
            raise _validation_error(
                "diophantine_approximation.coefficient_count_mismatch",
                "coefficient count must equal the requested term_count",
            )
        coefficient_bound = 2 * isqrt(self.discriminant)
        if any(
            coefficient < 0 or coefficient > coefficient_bound
            for coefficient in self.coefficients
        ):
            raise _validation_error(
                "diophantine_approximation.coefficient_bound_exceeded",
                "continued-fraction coefficients exceed the admitted sqrt(D) bound",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        discriminant: int,
        term_count: int,
        coefficients: tuple[int, ...],
        preperiod_length: int,
        period_length: int,
    ) -> Self:
        """Build a result from the owner-local continued-fraction kernel."""

        return cls(
            discriminant=discriminant,
            term_count=term_count,
            coefficients=coefficients,
            preperiod_length=preperiod_length,
            period_length=period_length,
        )


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
    """A bounded carrier for convergents of ``sqrt(D)``.

    The model does not execute a continued-fraction backend.  The explicit
    verifier checks independently supplied values against the canonical
    stream within this already-admitted envelope.
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
    def require_bounded_shape(self) -> Self:
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
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        discriminant: int,
        convergent_count: int,
        convergents: tuple[ConvergentValue, ...],
    ) -> Self:
        """Build a result from the owner-local convergent kernel."""

        return cls(
            discriminant=discriminant,
            convergent_count=convergent_count,
            convergents=convergents,
        )


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

    @model_validator(mode="after")
    def require_bounded_positive_components(self) -> Self:
        if (
            len(self.x.lstrip("-")) > _pell_component_digit_cap()
            or len(self.y.lstrip("-")) > _pell_component_digit_cap()
        ):
            raise _validation_error(
                "diophantine_approximation.pell_component_digit_bound_exceeded",
                "Pell components exceed the digit bound implied by the admitted discriminant",
            )
        if (
            self.x.startswith("-")
            or self.x == "0"
            or self.y.startswith("-")
            or self.y == "0"
        ):
            raise _validation_error(
                "diophantine_approximation.pell_components_not_positive",
                "Pell solution components must be positive",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, discriminant: int, x: CanonicalInteger, y: CanonicalInteger
    ) -> Self:
        """Build a result from the owner-local Pell kernel."""

        return cls(discriminant=discriminant, x=x, y=y)


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
