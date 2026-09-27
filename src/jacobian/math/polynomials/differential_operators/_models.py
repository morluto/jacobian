"""Typed contracts for exact differential-operator application."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.differential_operators.values import (
    ConstantCoefficientDifferentialOperator,
)
from jacobian.math.polynomials.values import RationalPolynomial


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"polynomial.differential_operator_{reason}", message)


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
    def require_compatible_axes(self) -> Self:
        if self.polynomial.variables != self.operator.variables:
            raise _validation_error(
                "axis_mismatch",
                "polynomial and operator must use the same ordered variable axis",
            )
        if (
            self.expected is not None
            and self.expected.variables != self.polynomial.variables
        ):
            raise _validation_error(
                "expected_axis_mismatch",
                "expected polynomial must use the same ordered variable axis",
            )
        return self


class DifferentialOperatorApplyResult(DifferentialOperatorApplyRequest):
    """Exact output and decisions bound to the supplied finite application."""

    output: RationalPolynomial
    is_zero: StrictBool
    matches_expected: StrictBool | None = None

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        """Validate canonical result relations without re-executing FLINT.

        Kernel-produced results use ``_from_kernel`` so parsing does not repeat
        the defining computation.
        """
        if self.output.variables != self.polynomial.variables:
            raise _validation_error(
                "output_axis_mismatch",
                "output polynomial must use the same ordered variable axis as the source",
            )
        if self.is_zero != (not self.output.polynomial.terms):
            raise _validation_error(
                "zero_flag_mismatch", "is_zero must match the exact output polynomial"
            )
        if self.expected is None:
            if self.matches_expected is not None:
                raise _validation_error(
                    "expected_missing",
                    "matches_expected must be null when no expected polynomial is supplied",
                )
        elif self.matches_expected != (self.output == self.expected):
            raise _validation_error(
                "expected_mismatch",
                "matches_expected must report exact equality with the expected polynomial",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: DifferentialOperatorApplyRequest,
        output: RationalPolynomial,
    ) -> Self:
        """Construct a trusted result after the admitted FLINT kernel returns."""

        return cls.model_construct(
            polynomial=request.polynomial,
            operator=request.operator,
            iterations=request.iterations,
            expected=request.expected,
            output=output,
            is_zero=not output.polynomial.terms,
            matches_expected=(
                None if request.expected is None else output == request.expected
            ),
        )


__all__ = [
    "DifferentialOperatorApplyRequest",
    "DifferentialOperatorApplyResult",
]
