"""Exact cubic resolvents for monic quartics over QQ."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.values import (
    MonicPolynomial,
    monic_polynomial_from_coefficients,
)

MAX_QUARTIC_RESOLVENT_INPUT_DIGITS = 256
MAX_QUARTIC_RESOLVENT_OUTPUT_DIGITS = 4_096
MAX_QUARTIC_RESOLVENT_WORK = 3_000_000


def _digits(value: int) -> int:
    return len(format_canonical_integer(abs(value)))


def _coefficient_digits(value: CanonicalRational) -> int:
    return max(_digits(value.num), _digits(value.den))


class QuarticCubicResolventRequest(StrictModel):
    polynomial: MonicPolynomial = Field(
        description=(
            "Monic univariate polynomial over QQ of degree exactly four. Each "
            f"coefficient numerator and denominator is limited to "
            f"{MAX_QUARTIC_RESOLVENT_INPUT_DIGITS} decimal digits."
        )
    )

    @model_validator(mode="after")
    def require_bounded_quartic(self) -> QuarticCubicResolventRequest:
        source_terms = self.polynomial.polynomial.terms
        if (
            len(self.polynomial.variables) != 1
            or not source_terms
            or source_terms[0].exponents[0] != 4
        ):
            raise PydanticCustomError(
                "polynomial.quartic_resolvent.degree",
                "cubic resolvent requires a degree-four source polynomial",
            )
        if any(
            _coefficient_digits(coefficient) > MAX_QUARTIC_RESOLVENT_INPUT_DIGITS
            for coefficient in (term.coefficient for term in source_terms)
        ):
            raise PydanticCustomError(
                "polynomial.quartic_resolvent.input_digits",
                "quartic coefficient components exceed the 256-digit input bound",
            )
        return self


class QuarticCubicResolventResult(StrictModel):
    """The cubic and its source, retaining the documented pair-product choice."""

    source: MonicPolynomial
    resolvent: MonicPolynomial
    convention: Literal["MONIC_QUARTIC_ROOT_PAIR_PRODUCTS"] = (
        "MONIC_QUARTIC_ROOT_PAIR_PRODUCTS"
    )

    @model_validator(mode="after")
    def require_source_and_output_degrees(self) -> QuarticCubicResolventResult:
        source_terms = self.source.polynomial.terms
        output_terms = self.resolvent.polynomial.terms
        if (
            len(self.source.variables) != 1
            or not source_terms
            or source_terms[0].exponents[0] != 4
            or len(self.resolvent.variables) != 1
            or not output_terms
            or output_terms[0].exponents[0] != 3
        ):
            raise PydanticCustomError(
                "polynomial.quartic_resolvent.result_shape",
                "result must bind a monic quartic source to a monic cubic",
            )
        return self


def _preflight(
    source: MonicPolynomial,
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    # Re-establish the request's structural and component bounds for native callers.
    try:
        parsed = QuarticCubicResolventRequest.model_validate(
            {"polynomial": source.model_dump(mode="python", warnings=False)}
        ).polynomial
    except (AttributeError, RecursionError, TypeError, ValueError) as exc:
        raise OperationDomainValidationError(
            location=("polynomial",),
            code="polynomial.quartic_resolvent.source",
            message="source must be a bounded monic quartic over QQ",
        ) from exc
    coefficients = parsed.coefficients
    # Four fixed-degree products and two additions: charge conservatively before
    # constructing any products. At 256 input digits the largest unreduced
    # numerator/denominator envelope is below 2,100 digits.
    digits = max(_coefficient_digits(value) for value in coefficients)
    work_bound = 32 * digits * digits
    output_bound = 8 * digits + 32
    if (
        work_bound > MAX_QUARTIC_RESOLVENT_WORK
        or output_bound > MAX_QUARTIC_RESOLVENT_OUTPUT_DIGITS
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.quartic_resolvent.admission",
            message="quartic resolvent exact arithmetic exceeds admitted bounds",
        )
    a, b, c, d = (coefficient.as_fraction() for coefficient in coefficients[3::-1])
    return a, b, c, d


def compute_quartic_cubic_resolvent(
    polynomial: MonicPolynomial,
) -> QuarticCubicResolventResult:
    a, b, c, d = _preflight(polynomial)
    request_checkpoint("during quartic cubic resolvent computation")
    coefficients = (
        4 * b * d - a * a * d - c * c,
        a * c - 4 * d,
        -b,
        Fraction(1),
    )
    if any(
        max(_digits(value.numerator), _digits(value.denominator))
        > MAX_QUARTIC_RESOLVENT_OUTPUT_DIGITS
        for value in coefficients
    ):
        raise OperationResourceAdmissionError(
            location=("polynomial",),
            code="polynomial.quartic_resolvent.output_bound",
            message="quartic resolvent coefficients exceed the exact-output bound",
        )
    result = monic_polynomial_from_coefficients(
        tuple(CanonicalRational.from_fraction(value) for value in coefficients),
        variable="y",
    )
    return QuarticCubicResolventResult(
        source=polynomial,
        resolvent=result,
    )


__all__ = [
    "QuarticCubicResolventRequest",
    "QuarticCubicResolventResult",
    "compute_quartic_cubic_resolvent",
]
