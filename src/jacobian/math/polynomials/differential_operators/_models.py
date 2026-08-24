"""Typed contracts for exact differential-operator application."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials.differential_operators._bounds import (
    validate_application_envelope,
)
from jacobian.math.polynomials.differential_operators._flint import apply_with_flint
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
)
from jacobian.math.polynomials.values import RationalPolynomial


class DifferentialOperatorApplyRequest(StrictModel):
    """Apply one finite constant-coefficient operator power over ``QQ``.

    The source polynomial and operator must carry the same complete ordered
    variable axis. Admission bounds the operator-power paths, expanded
    derivative support, derivative work, rational growth, exact output terms,
    and the retained source-bound result before python-flint is invoked.
    """

    polynomial: RationalPolynomial = Field(
        description="Canonical sparse source polynomial over QQ."
    )
    operator: ConstantCoefficientDifferentialOperator
    iterations: StrictInt = Field(
        default=1,
        ge=0,
        le=(1 << 53) - 1,
        description=(
            "Finite exponent k in D^k(f), inside the strict-JSON "
            "interoperable integer range. Zero returns the source unchanged. "
            "Admission is derived per request rather than fixed: requests whose "
            "exact result needs no operator-power expansion - guaranteed "
            "annihilation, the identity operator, and one-term zeroth-order "
            "pure-scaling operators - are admitted at any k, and expanding "
            "requests are bounded by the derived expanded-support, work, "
            "coefficient-growth, and serialized-size budgets."
        ),
        examples=[2],
    )
    expected: RationalPolynomial | None = Field(
        default=None,
        description=(
            "Optional canonical polynomial on the same ordered axis. When supplied, "
            "the result retains it and reports exact equality with the output."
        ),
    )

    @model_validator(mode="after")
    def require_bounded_application(self) -> Self:
        validate_application_envelope(
            self.polynomial,
            self.operator,
            self.iterations,
            self.expected,
        )
        return self


class DifferentialOperatorApplyResult(DifferentialOperatorApplyRequest):
    """Exact output and decisions bound to the supplied finite application."""

    output: RationalPolynomial
    is_zero: StrictBool
    matches_expected: StrictBool | None = None

    @model_validator(mode="after")
    def bind_exact_application(self) -> Self:
        envelope = validate_application_envelope(
            self.polynomial,
            self.operator,
            self.iterations,
            self.expected,
        )
        replayed = apply_with_flint(
            self.polynomial,
            self.operator,
            self.iterations,
            envelope,
        )
        if self.output != replayed:
            raise ValueError(
                "differential-operator output is not bound to the supplied application"
            )
        if self.is_zero != (not self.output.polynomial.terms):
            raise ValueError("is_zero must match the exact output polynomial")
        if self.expected is None:
            if self.matches_expected is not None:
                raise ValueError(
                    "matches_expected must be null when no expected polynomial is supplied"
                )
        elif self.matches_expected != (self.output == self.expected):
            raise ValueError(
                "matches_expected must report exact equality with the expected polynomial"
            )
        return self


__all__ = [
    "DifferentialOperatorApplyRequest",
    "DifferentialOperatorApplyResult",
]
