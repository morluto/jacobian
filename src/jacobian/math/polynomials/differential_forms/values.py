"""Canonical sparse polynomial differential-form values."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import ExactInteger
from jacobian._models import StrictModel
from jacobian.math.polynomials.values import (
    MAX_POLYNOMIAL_VARIABLES,
    PolynomialVariable,
    RationalPolynomial,
    require_polynomial_budget,
)

MAX_DIFFERENTIAL_FORM_COMPONENTS = 256
MAX_DIFFERENTIAL_FORM_TERMS = 256
# One wedge product adds exponents from both operands.  The carrier allows
# repeated bounded compositions; the operation rejects a sum beyond this
# representation limit before expansion.
MAX_DIFFERENTIAL_FORM_EXPONENT = 256
MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS = 4_096


def _error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"differential_form.{reason}", message)


class FormComponent(StrictModel):
    """One coefficient in the ordered differential basis."""

    indices: tuple[int, ...] = Field(
        max_length=MAX_POLYNOMIAL_VARIABLES,
        description="Strictly increasing coordinate indices in the differential basis.",
    )
    coefficient: RationalPolynomial = Field(
        description=(
            "Sparse QQ coefficient on the form variable axis. Differential-form "
            f"coefficients admit at most {MAX_DIFFERENTIAL_FORM_TERMS} terms, "
            f"exponents at most {MAX_DIFFERENTIAL_FORM_EXPONENT}, and numerator "
            "or denominator components at most "
            f"{MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS} digits."
        ),
    )

    @model_validator(mode="after")
    def require_local_canonical_indices(self) -> Self:
        if any(index < 0 for index in self.indices):
            raise _error(
                "component_indices",
                "differential indices must be nonnegative",
            )
        if self.indices != tuple(sorted(set(self.indices))):
            raise _error(
                "component_indices",
                "differential indices must be strictly increasing",
            )
        return self


class PolynomialDifferentialForm(StrictModel):
    """A sparse polynomial k-form on an ordered affine QQ coordinate axis."""

    variables: tuple[PolynomialVariable, ...] = Field(
        min_length=0, max_length=MAX_POLYNOMIAL_VARIABLES
    )
    degree: ExactInteger = Field(ge=0)
    components: tuple[FormComponent, ...] = Field(
        default=(),
        max_length=MAX_DIFFERENTIAL_FORM_COMPONENTS,
        description=(
            "Ordered differential-basis components. Each coefficient is bounded "
            f"to {MAX_DIFFERENTIAL_FORM_TERMS} terms, exponent "
            f"{MAX_DIFFERENTIAL_FORM_EXPONENT}, and "
            f"{MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS}-digit rationals."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_form(self) -> Self:
        dimension = len(self.variables)
        if len(set(self.variables)) != dimension:
            raise _error("variable_axis", "form variables must be unique")
        if self.degree > dimension:
            # The zero of an over-dimensioned graded piece retains its target
            # degree so repeated wedge composition remains deterministic.
            if self.components:
                raise _error(
                    "degree",
                    "a form above the ambient dimension must be canonical zero",
                )
            return self
        indices = tuple(component.indices for component in self.components)
        if indices != tuple(sorted(indices)) or len(set(indices)) != len(indices):
            raise _error(
                "component_order",
                "form components must be unique and sorted by indices",
            )
        for component in self.components:
            if len(component.indices) != self.degree or any(
                index < 0 or index >= dimension for index in component.indices
            ):
                raise _error(
                    "component_basis",
                    "component indices must be increasing and lie on the variable axis",
                )
            if component.indices != tuple(sorted(set(component.indices))):
                raise _error("component_basis", "component indices must be increasing")
            if component.coefficient.variables != self.variables:
                raise _error(
                    "coefficient_axis",
                    "every form coefficient must use the complete form variable axis",
                )
            if not component.coefficient.polynomial.terms:
                raise _error("zero_component", "zero coefficients must be omitted")
            try:
                require_polynomial_budget(
                    component.coefficient,
                    maximum_terms=MAX_DIFFERENTIAL_FORM_TERMS,
                    maximum_exponent=MAX_DIFFERENTIAL_FORM_EXPONENT,
                    maximum_coefficient_digits=MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS,
                    label="differential-form coefficient",
                )
            except ValueError as exc:
                raise _error("coefficient_budget", str(exc)) from exc
        return self


__all__ = [
    "MAX_DIFFERENTIAL_FORM_COEFFICIENT_DIGITS",
    "MAX_DIFFERENTIAL_FORM_COMPONENTS",
    "MAX_DIFFERENTIAL_FORM_EXPONENT",
    "MAX_DIFFERENTIAL_FORM_TERMS",
    "FormComponent",
    "PolynomialDifferentialForm",
]
