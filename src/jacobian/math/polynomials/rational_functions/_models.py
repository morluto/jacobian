"""Typed contracts for exact rational-function reductions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_sparse_polynomial_budget,
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.rational_function_contract", message)


MAX_HERMITE_NUMERATOR_DEGREE = 6
MAX_HERMITE_DENOMINATOR_DEGREE = 3
MAX_HERMITE_COEFFICIENT_DIGITS = 2
MAX_HERMITE_RESULT_COEFFICIENT_DIGITS = 128


def require_hermite_reduction_budget(function: RationalFunction) -> None:
    """Validate the shared native and catalog Hermite-reduction envelope."""

    if len(function.variables) != 1:
        raise _validation_error("Hermite reduction requires exactly one variable")
    require_sparse_polynomial_budget(
        function.numerator,
        maximum_terms=MAX_HERMITE_NUMERATOR_DEGREE + 1,
        maximum_exponent=MAX_HERMITE_NUMERATOR_DEGREE,
        maximum_coefficient_digits=MAX_HERMITE_COEFFICIENT_DIGITS,
        label="Hermite-reduction numerator",
    )
    require_sparse_polynomial_budget(
        function.denominator,
        maximum_terms=MAX_HERMITE_DENOMINATOR_DEGREE + 1,
        maximum_exponent=MAX_HERMITE_DENOMINATOR_DEGREE,
        maximum_coefficient_digits=MAX_HERMITE_COEFFICIENT_DIGITS,
        label="Hermite-reduction denominator",
    )


def _require_hermite_result_budget(
    rational_part: RationalFunction,
    remainder: RationalFunction,
) -> None:
    """Bound the certificate algebra accepted by the explicit verifier.

    A degree-six numerator over a degree-three denominator yields a
    zero-constant rational part with numerator degree at most six and
    denominator degree at most two.  The square-free residual denominator has
    degree at most three and its numerator is proper.  These derived limits
    keep independently supplied certificates within the same exact envelope
    as a kernel-produced result.
    """

    for label, polynomial, maximum_terms, maximum_exponent in (
        ("Hermite rational-part numerator", rational_part.numerator, 7, 6),
        ("Hermite rational-part denominator", rational_part.denominator, 3, 2),
        ("Hermite remainder numerator", remainder.numerator, 3, 2),
        ("Hermite remainder denominator", remainder.denominator, 4, 3),
    ):
        require_sparse_polynomial_budget(
            polynomial,
            maximum_terms=maximum_terms,
            maximum_exponent=maximum_exponent,
            maximum_coefficient_digits=MAX_HERMITE_RESULT_COEFFICIENT_DIGITS,
            label=label,
        )


class HermiteReductionRequest(StrictModel):
    """One conservatively bounded canonical element of ``QQ(x)``.

    The present envelope bounds polynomial division, denominator GCD, and a
    three-variable Horowitz--Ostrogradsky linear system before backend
    expansion. Two-digit rational components keep the exact Cramer/factor
    coefficient-height bound inside the canonical 128-digit result carrier.
    This is a scale limit, not a restriction on the mathematical domain.
    """

    function: RationalFunction = Field(
        description=(
            "A canonical univariate QQ(x) value with numerator degree at most 6, "
            "denominator degree at most 3, at most 7/4 respective terms, and "
            "at most two decimal digits in each rational coefficient component."
        )
    )

    @model_validator(mode="after")
    def require_univariate_reduction_budget(self) -> Self:
        require_hermite_reduction_budget(self.function)
        return self


class HermiteReductionResult(HermiteReductionRequest):
    """The canonical rational derivative and reduced logarithmic remainder."""

    rational_part: RationalFunction
    remainder: RationalFunction
    rational_primitive_status: Literal["RATIONAL_PRIMITIVE", "NO_RATIONAL_PRIMITIVE"]
    rational_primitive: RationalFunction | None

    @model_validator(mode="after")
    def require_structural_contract(self) -> Self:
        variables = self.function.variables
        if (
            self.rational_part.variables != variables
            or self.remainder.variables != variables
        ):
            raise _validation_error(
                "all Hermite-reduction values must use the source variable"
            )
        _require_hermite_result_budget(self.rational_part, self.remainder)
        has_primitive = not self.remainder.numerator.terms
        expected_status = (
            "RATIONAL_PRIMITIVE" if has_primitive else "NO_RATIONAL_PRIMITIVE"
        )
        if self.rational_primitive_status != expected_status:
            raise _validation_error(
                "rational-primitive status must match the Hermite remainder"
            )
        if has_primitive:
            if self.rational_primitive != self.rational_part:
                raise _validation_error(
                    "rational primitive must equal the canonical rational part"
                )
        elif self.rational_primitive is not None:
            raise _validation_error(
                "a nonzero Hermite remainder has no rational primitive"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        function: RationalFunction,
        rational_part: RationalFunction,
        remainder: RationalFunction,
    ) -> Self:
        """Build a trusted result from the owner-local Hermite kernel."""

        has_primitive = not remainder.numerator.terms
        return cls(
            function=function,
            rational_part=rational_part,
            remainder=remainder,
            rational_primitive_status=(
                "RATIONAL_PRIMITIVE" if has_primitive else "NO_RATIONAL_PRIMITIVE"
            ),
            rational_primitive=rational_part if has_primitive else None,
        )


__all__ = ["HermiteReductionRequest", "HermiteReductionResult"]
