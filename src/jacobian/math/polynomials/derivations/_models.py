"""Typed wire contracts for exact polynomial derivations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_DERIVATION_VARIABLES = 8
MAX_DERIVATION_IMAGE_TERMS = 256
MAX_DERIVATION_SOURCE_TERMS = 256
MAX_DERIVATION_EXPONENT = 64
MAX_DERIVATION_COEFFICIENT_DIGITS = 128
MAX_DERIVATION_CONTRIBUTION_CELLS = 4_096


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable error owned by polynomial-derivation contracts."""

    return PydanticCustomError(f"polynomial_derivation.{reason}", message)


def _require_derivation_polynomial(
    polynomial: RationalPolynomial, *, label: str, maximum_terms: int
) -> None:
    if len(polynomial.variables) > MAX_DERIVATION_VARIABLES:
        raise _validation_error(
            "variable_budget",
            f"{label} exceeds the {MAX_DERIVATION_VARIABLES}-variable budget",
        )
    require_polynomial_budget(
        polynomial,
        maximum_terms=maximum_terms,
        maximum_exponent=MAX_DERIVATION_EXPONENT,
        maximum_coefficient_digits=MAX_DERIVATION_COEFFICIENT_DIGITS,
        label=label,
    )
    if any(
        sum(term.exponents) > MAX_DERIVATION_EXPONENT
        for term in polynomial.polynomial.terms
    ):
        raise _validation_error(
            "total_degree", f"{label} exceeds total degree {MAX_DERIVATION_EXPONENT}"
        )


class PolynomialDerivation(StrictModel):
    """One QQ-derivation of a polynomial ring, bound by its generator images."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=1, max_length=MAX_DERIVATION_VARIABLES
    )
    images: tuple[RationalPolynomial, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_generator_images(self) -> Self:
        if len(set(self.variables)) != len(self.variables):
            raise _validation_error(
                "duplicate_variables", "derivation variables must be unique"
            )
        if len(self.images) != len(self.variables):
            raise _validation_error(
                "image_count",
                "a derivation needs exactly one image per ring generator",
            )
        if any(image.variables != self.variables for image in self.images):
            raise _validation_error(
                "ordered_ring",
                "derivation images must use the declared ordered ring",
            )
        return self


class DerivationApplyRequest(StrictModel):
    """One checked derivation applied to one polynomial of the same ring."""

    derivation: PolynomialDerivation
    polynomial: RationalPolynomial = Field(
        description=(
            "Source polynomial in the derivation's ordered QQ ring; it must "
            "share the derivation's ordered variables."
        )
    )

    @model_validator(mode="after")
    def require_shared_ring(self) -> Self:
        if self.polynomial.variables != self.derivation.variables:
            raise _validation_error(
                "ordered_ring",
                "the polynomial must use the derivation's ordered ring",
            )
        return self


class DerivationApplyResult(StrictModel):
    """The exact derivation image with its per-variable contribution ledger."""

    derivation: PolynomialDerivation
    polynomial: RationalPolynomial
    result: RationalPolynomial
    contributions: tuple[RationalPolynomial, ...] = Field(
        description=(
            "Per-variable products D(x_i) * partial_i(f) in generator order; "
            "they sum exactly to the result."
        )
    )

    @model_validator(mode="after")
    def require_ledger_shape(self) -> Self:
        variables = self.derivation.variables
        if len(self.contributions) != len(variables) or any(
            contribution.variables != variables
            for contribution in (*self.contributions, self.result, self.polynomial)
        ):
            raise _validation_error(
                "ledger_ring",
                "result ledger entries must use the derivation's ordered ring",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        derivation: PolynomialDerivation,
        polynomial: RationalPolynomial,
        *,
        result: RationalPolynomial,
        contributions: tuple[RationalPolynomial, ...],
    ) -> Self:
        return cls.model_construct(
            derivation=derivation,
            polynomial=polynomial,
            result=result,
            contributions=contributions,
        )
