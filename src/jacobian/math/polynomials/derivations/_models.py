"""Typed wire contracts for exact polynomial derivations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import ConfigDict, Field, model_validator
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
MAX_DERIVATION_ITERATE_COUNT = 32
MAX_DERIVATION_ITERATE_TERMS = 4_096
MAX_DERIVATION_CERTIFICATE_CHAIN = 32
MAX_DERIVATION_PARAMETER = "t"
MAX_GA_ACTION_VARIABLES = 7
MAX_GA_ACTION_OUTPUT_TERMS = 4_096
MAX_GA_ACTION_OUTPUT_BYTES = 2_000_000

GeneratorIterateChain = Annotated[
    tuple[RationalPolynomial, ...],
    Field(min_length=1, max_length=MAX_DERIVATION_CERTIFICATE_CHAIN),
]
GeneratorIterateChains = Annotated[
    tuple[GeneratorIterateChain, ...],
    Field(max_length=MAX_DERIVATION_VARIABLES),
]


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
    images: tuple[RationalPolynomial, ...] = Field(
        min_length=1, max_length=MAX_DERIVATION_VARIABLES
    )

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


class DerivationFromVectorFieldRequest(StrictModel):
    """Convert a polynomial vector field to its induced QQ-derivation."""

    components: Annotated[
        tuple[RationalPolynomial, ...], Field(min_length=1, max_length=8)
    ]

    @model_validator(mode="after")
    def require_vector_field_axis(self) -> Self:
        variables = self.components[0].variables
        if len(self.components) != len(variables):
            raise _validation_error(
                "vector_field_component_count",
                "a vector field needs one polynomial component per ordered variable",
            )
        if any(component.variables != variables for component in self.components):
            raise _validation_error(
                "ordered_ring", "vector-field components must share one ordered QQ ring"
            )
        return self


class DerivationIteratesRequest(StrictModel):
    derivation: PolynomialDerivation
    polynomial: RationalPolynomial
    bound: int = Field(ge=0, le=MAX_DERIVATION_ITERATE_COUNT)

    @model_validator(mode="after")
    def require_shared_ring(self) -> Self:
        if self.polynomial.variables != self.derivation.variables:
            raise _validation_error(
                "ordered_ring", "the polynomial must use the derivation's ordered ring"
            )
        return self


class DerivationIteratesResult(StrictModel):
    derivation: PolynomialDerivation
    source: RationalPolynomial
    iterates: tuple[RationalPolynomial, ...]
    status: Literal["FIRST_ZERO_ON_F", "NONZERO_THROUGH_BOUND"]
    first_zero_index: int | None = None

    @model_validator(mode="after")
    def require_profile(self) -> Self:
        if not self.iterates or self.iterates[0] != self.source:
            raise _validation_error(
                "iterate_profile", "the iterate family must begin with the source"
            )
        if self.status == "FIRST_ZERO_ON_F":
            if (
                self.first_zero_index is None
                or self.first_zero_index != len(self.iterates) - 1
            ):
                raise _validation_error(
                    "iterate_profile",
                    "first zero index must identify the final returned iterate",
                )
            if self.iterates[-1].polynomial.terms:
                raise _validation_error(
                    "iterate_profile", "the first-zero iterate must be zero"
                )
        elif self.first_zero_index is not None:
            raise _validation_error(
                "iterate_profile", "a nonzero bounded profile has no first-zero index"
            )
        return self


class LocallyNilpotentCertificate(StrictModel):
    derivation: PolynomialDerivation
    generator_iterates: GeneratorIterateChains

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if len(self.generator_iterates) != len(self.derivation.variables):
            raise _validation_error(
                "certificate_shape",
                "one complete iterate chain is required per generator",
            )
        for chain in self.generator_iterates:
            if not chain or any(
                value.variables != self.derivation.variables for value in chain
            ):
                raise _validation_error(
                    "certificate_shape",
                    "certificate chains must use the derivation ring",
                )
            if len(chain) > MAX_DERIVATION_CERTIFICATE_CHAIN:
                raise _validation_error(
                    "certificate_bound",
                    "generator iterate chain exceeds the admitted bound",
                )
        return self


class PolynomialGaAction(StrictModel):
    source_variables: tuple[PolynomialVariable, ...]
    parameter: PolynomialVariable = MAX_DERIVATION_PARAMETER
    generator_images: tuple[RationalPolynomial, ...]

    @model_validator(mode="after")
    def require_action_shape(self) -> Self:
        if not 1 <= len(self.source_variables) <= MAX_GA_ACTION_VARIABLES:
            raise _validation_error(
                "action_variable_budget",
                "an action needs 1 to 7 source variables because its parameter is an explicit polynomial axis",
            )
        if self.parameter in self.source_variables:
            raise _validation_error(
                "action_parameter",
                "the action parameter must be distinct from source variables",
            )
        if len(set(self.source_variables)) != len(self.source_variables):
            raise _validation_error(
                "action_variable_axis", "source action variables must be unique"
            )
        expected = (*self.source_variables, self.parameter)
        if len(self.generator_images) != len(self.source_variables) or any(
            image.variables != expected for image in self.generator_images
        ):
            raise _validation_error(
                "action_ring",
                "action generator images must use the source ring extended by the parameter",
            )
        term_count = sum(len(image.polynomial.terms) for image in self.generator_images)
        if term_count > MAX_GA_ACTION_OUTPUT_TERMS:
            raise _validation_error(
                "action_output_terms", "action images exceed the complete term envelope"
            )
        if any(
            any(exponent > MAX_DERIVATION_EXPONENT for exponent in term.exponents[:-1])
            or term.exponents[-1] >= MAX_DERIVATION_CERTIFICATE_CHAIN
            or sum(term.exponents)
            > MAX_DERIVATION_EXPONENT + MAX_DERIVATION_CERTIFICATE_CHAIN - 1
            or len(str(abs(term.coefficient.num))) > MAX_DERIVATION_COEFFICIENT_DIGITS
            or len(str(term.coefficient.den)) > MAX_DERIVATION_COEFFICIENT_DIGITS
            for image in self.generator_images
            for term in image.polynomial.terms
        ):
            raise _validation_error(
                "action_output_bound",
                "action term exceeds its degree or coefficient envelope",
            )
        return self


class DerivationCertificateRequest(StrictModel):
    derivation: PolynomialDerivation
    chains: GeneratorIterateChains


class GaActionRequest(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Construct exp(tD) from exact generator iterate chains. The "
                "explicit parameter axis limits the source ring to "
                f"{MAX_GA_ACTION_VARIABLES} variables; action output is bounded "
                f"to {MAX_GA_ACTION_OUTPUT_TERMS} terms, "
                f"{MAX_GA_ACTION_OUTPUT_BYTES} estimated bytes."
            ),
            "admission_limits": {
                "max_source_variables": MAX_GA_ACTION_VARIABLES,
                "max_action_output_terms": MAX_GA_ACTION_OUTPUT_TERMS,
                "max_action_output_bytes": MAX_GA_ACTION_OUTPUT_BYTES,
                "max_generator_chain_length": MAX_DERIVATION_CERTIFICATE_CHAIN,
            },
        }
    )

    derivation: PolynomialDerivation = Field(
        description=(
            "The derivation must have at most "
            f"{MAX_GA_ACTION_VARIABLES} ordered source variables so its "
            "explicit additive parameter fits the 8-axis polynomial carrier."
        )
    )
    chains: GeneratorIterateChains = Field(
        description=(
            "One complete exact iterate chain per generator, starting with "
            "that generator and ending at zero; each chain has at most "
            f"{MAX_DERIVATION_CERTIFICATE_CHAIN} polynomials."
        )
    )


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
