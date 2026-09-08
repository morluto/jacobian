"""Typed contracts for exact rational-function reductions."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.polynomials.values import (
    RationalFunction,
    _require_rational_function_shapes,
    _require_rational_function_structural_normal_form,
    require_canonical_rational_function,
    require_sparse_polynomial_budget,
)


def _validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("polynomial.rational_function_contract", message)


MAX_HERMITE_POLYNOMIAL_DEGREE = 63
MAX_HERMITE_NUMERATOR_DEGREE = 6
MAX_HERMITE_DENOMINATOR_DEGREE = 3
MAX_HERMITE_COEFFICIENT_DIGITS = 2
MAX_HERMITE_RESULT_COEFFICIENT_DIGITS = 128


def require_hermite_reduction_budget(function: RationalFunction) -> None:
    """Validate the shared native and catalog Hermite-reduction envelope."""

    if len(function.variables) != 1:
        raise _validation_error("Hermite reduction requires exactly one variable")
    polynomial_source = all(
        not any(term.exponents) for term in function.denominator.terms
    )
    coefficient_digits = (
        MAX_HERMITE_RESULT_COEFFICIENT_DIGITS
        if polynomial_source
        else MAX_HERMITE_COEFFICIENT_DIGITS
    )
    numerator_degree = (
        MAX_HERMITE_POLYNOMIAL_DEGREE
        if polynomial_source
        else MAX_HERMITE_NUMERATOR_DEGREE
    )
    try:
        require_sparse_polynomial_budget(
            function.numerator,
            maximum_terms=numerator_degree + 1,
            maximum_exponent=numerator_degree,
            maximum_coefficient_digits=coefficient_digits,
            label="Hermite-reduction numerator",
        )
        require_sparse_polynomial_budget(
            function.denominator,
            maximum_terms=MAX_HERMITE_DENOMINATOR_DEGREE + 1,
            maximum_exponent=MAX_HERMITE_DENOMINATOR_DEGREE,
            maximum_coefficient_digits=coefficient_digits,
            label="Hermite-reduction denominator",
        )
    except ValueError as exc:
        raise OperationResourceAdmissionError(
            location=("function",),
            code="polynomial.hermite_reduction_budget",
            message=str(exc),
        ) from exc
    if polynomial_source:
        _require_rational_function_shapes(function)
        _require_rational_function_structural_normal_form(function)
        # The canonical unit denominator makes coprimality immediate. Each
        # primitive coefficient divides a source coefficient by its new degree;
        # its numerator cannot grow and this unreduced denominator is a bound.
        if any(
            term.coefficient.den * (term.exponents[0] + 1)
            >= 10**MAX_HERMITE_RESULT_COEFFICIENT_DIGITS
            for term in function.numerator.terms
        ):
            raise OperationResourceAdmissionError(
                location=("function",),
                code="polynomial.hermite_reduction_budget",
                message="Hermite polynomial primitive denominator exceeds the 128-digit bound",
            )
        return
    require_canonical_rational_function(
        function,
        maximum_terms=MAX_HERMITE_NUMERATOR_DEGREE + 1,
        maximum_exponent=MAX_HERMITE_NUMERATOR_DEGREE,
        maximum_coefficient_digits=coefficient_digits,
        label="Hermite-reduction function",
    )


def _require_hermite_result_budget(
    rational_part: RationalFunction,
    remainder: RationalFunction,
) -> None:
    """Bound the retained exact Hermite-reduction result.

    Polynomial inputs yield at most 64 nonzero primitive terms of degree
    at most 64. General rational inputs retain the degree-six/degree-three
    envelope, with rational-part denominator degree at most two and proper
    remainder denominator degree at most three.
    """

    for label, polynomial, maximum_terms, maximum_exponent in (
        ("Hermite rational-part numerator", rational_part.numerator, 64, 64),
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
    expansion. Polynomial sources permit degree 63 and 128-digit components
    when coefficientwise integration stays within the result carrier. For other
    rational functions, two-digit components bound Cramer/factor growth inside
    the 128-digit result carrier.
    This is a scale limit, not a restriction on the mathematical domain.
    """

    function: RationalFunction = Field(
        description=(
            "A canonical univariate QQ(x) value. General rational inputs allow "
            "numerator degree 6, denominator degree 3 and two-digit components. "
            "Polynomial inputs allow degree 63 and 128-digit components subject "
            "to primitive denominator growth."
        )
    )


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
        return cls.model_construct(
            function=function,
            rational_part=rational_part,
            remainder=remainder,
            rational_primitive_status=(
                "RATIONAL_PRIMITIVE" if has_primitive else "NO_RATIONAL_PRIMITIVE"
            ),
            rational_primitive=rational_part if has_primitive else None,
        )


__all__ = ["HermiteReductionRequest", "HermiteReductionResult"]
