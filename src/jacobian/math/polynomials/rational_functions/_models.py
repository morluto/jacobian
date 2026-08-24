"""Typed contracts for exact rational-function reductions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.polynomials._conversions import (
    rational_function_to_sympy,
    symbols_for_variables,
)
from jacobian.math.polynomials.values import (
    RationalFunction,
    require_sparse_polynomial_budget,
)

MAX_HERMITE_NUMERATOR_DEGREE = 6
MAX_HERMITE_DENOMINATOR_DEGREE = 3
MAX_HERMITE_COEFFICIENT_DIGITS = 2


def require_hermite_reduction_budget(function: RationalFunction) -> None:
    """Validate the shared native and catalog Hermite-reduction envelope."""

    if len(function.variables) != 1:
        raise ValueError("Hermite reduction requires exactly one variable")
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
    def bind_exact_reduction(self) -> Self:
        from sympy import Poly, cancel, diff, fraction

        variables = self.function.variables
        if (
            self.rational_part.variables != variables
            or self.remainder.variables != variables
        ):
            raise ValueError(
                "all Hermite-reduction values must use the source variable"
            )
        (variable,) = symbols_for_variables(variables)
        source = rational_function_to_sympy(self.function)
        rational_part = rational_function_to_sympy(self.rational_part)
        remainder = cancel(rational_function_to_sympy(self.remainder))
        if cancel(diff(rational_part, variable) + remainder - source) != 0:
            raise ValueError("Hermite reduction does not reconstruct the source")

        remainder_numerator, remainder_denominator = fraction(remainder)
        numerator = Poly(remainder_numerator, variable, domain="QQ")
        denominator = Poly(remainder_denominator, variable, domain="QQ")
        if not numerator.is_zero and numerator.degree() >= denominator.degree():
            raise ValueError("Hermite remainder must be proper")
        if not denominator.gcd(denominator.diff()).degree() == 0:
            raise ValueError("Hermite remainder denominator must be square-free")

        part_numerator, part_denominator = fraction(cancel(rational_part))
        quotient, _ = Poly(part_numerator, variable, domain="QQ").div(
            Poly(part_denominator, variable, domain="QQ")
        )
        if quotient.nth(0) != 0:
            raise ValueError("Hermite rational part must use zero additive constant")

        has_primitive = numerator.is_zero
        expected_status = (
            "RATIONAL_PRIMITIVE" if has_primitive else "NO_RATIONAL_PRIMITIVE"
        )
        if self.rational_primitive_status != expected_status:
            raise ValueError(
                "rational-primitive status must match the Hermite remainder"
            )
        if has_primitive:
            if self.rational_primitive != self.rational_part:
                raise ValueError(
                    "rational primitive must equal the canonical rational part"
                )
        elif self.rational_primitive is not None:
            raise ValueError("a nonzero Hermite remainder has no rational primitive")
        return self


__all__ = ["HermiteReductionRequest", "HermiteReductionResult"]
