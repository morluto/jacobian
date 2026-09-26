"""Typed wire contracts for finite function-field operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.finite_fields.values import (
    FiniteFieldElement,
    FiniteFieldPresentation,
)

MAX_CHARACTERISTIC = 257
MAX_EXTENSION_DEGREE = 6
MAX_POLYNOMIAL_X_DEGREE = 12
MAX_POLYNOMIAL_COEFFICIENTS = MAX_POLYNOMIAL_X_DEGREE + 1
MAX_LEDGER_ROWS = 64
MAX_FIELD_ADMISSION_WORK = 4_000_000
MAX_BASE_EMBEDDING_VALUE_BYTES = 32_768
MAX_ELEMENT_ADDITION_WORK = 2_000_000
MAX_ELEMENT_VALUE_BYTES = 32_768
MAX_INVERSION_WORK = 2_000_000
MAX_MULTIPLICATION_WORK = 2_000_000
MAX_DIVISOR_MULTIPLICITY_BITS = 4096
# 2**4096 needs 1234 decimal digits; divisor degrees combine at most 256
# places of degree <= 12, needing at most 1237 digits.
MAX_DIVISOR_MULTIPLICITY_DIGITS = 1234
MAX_DIVISOR_DEGREE_DIGITS = 1240
MAX_RIEMANN_ROCH_BASIS_DIMENSION = MAX_POLYNOMIAL_X_DEGREE + 1
MAX_RIEMANN_ROCH_CONSTRUCTION_WORK = 4096
MAX_RIEMANN_ROCH_MEMBERSHIP_PROFILE_ROWS = 256 + 2 * MAX_POLYNOMIAL_X_DEGREE + 1
MAX_RIEMANN_ROCH_MEMBERSHIP_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_RIEMANN_ROCH_MEMBERSHIP_FACTOR_WORK = 5_000_000
MAX_RATIONAL_PLACE_DEGREE = 12
MAX_RATIONAL_PLACE_CANDIDATES = 16_384
MAX_RATIONAL_PLACE_OUTPUT = 16_385
MAX_RATIONAL_PLACE_WORK = 20_000_000

FieldVariable = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,15}$", strict=True)
]

DivisorMultiplicity = Annotated[int, DecimalIntegerEncoding(max_digits=1234)]
DivisorDegree = Annotated[int, DecimalIntegerEncoding(max_digits=1240)]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"function_field.{reason}", message)


class PrimeFieldPolynomial(StrictModel):
    """A canonical univariate polynomial over GF(characteristic).

    Coefficients are ascending and reduced to ``0..p-1``; the zero polynomial
    has the unique spelling ``[0]`` and every nonzero polynomial has a nonzero
    leading coefficient.
    """

    characteristic: int = Field(ge=2, le=MAX_CHARACTERISTIC)
    coefficients: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_POLYNOMIAL_COEFFICIENTS
    )

    @model_validator(mode="after")
    def require_canonical_coefficients(self) -> Self:
        if any(
            type(value) is not int or value < 0 or value >= self.characteristic
            for value in self.coefficients
        ):
            raise _validation_error(
                "polynomial_coefficients",
                "coefficients must be canonical residues of the characteristic",
            )
        if len(self.coefficients) > 1 and self.coefficients[-1] == 0:
            raise _validation_error(
                "polynomial_trailing_zero",
                "a nonzero polynomial cannot have a trailing zero coefficient",
            )
        return self

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def is_zero(self) -> bool:
        return self.coefficients == (0,)

    def is_one(self) -> bool:
        return self.coefficients == (1,)


class PrimeFieldRationalFunction(StrictModel):
    """A presentation ``numerator/denominator`` over GF(characteristic).

    Deserialization checks only the bounded shape and a nonzero denominator.
    Canonical reduced form (coprime numerator/denominator and monic
    denominator) is established by the consuming function-field admission.
    """

    numerator: PrimeFieldPolynomial
    denominator: PrimeFieldPolynomial

    @model_validator(mode="after")
    def require_common_nonzero_fraction(self) -> Self:
        if self.numerator.characteristic != self.denominator.characteristic:
            raise _validation_error(
                "rational_function_characteristic",
                "numerator and denominator must share one characteristic",
            )
        if self.denominator.is_zero():
            raise _validation_error(
                "rational_function_zero_denominator",
                "a rational function requires a nonzero denominator",
            )
        return self

    @property
    def characteristic(self) -> int:
        return self.numerator.characteristic


class FiniteFunctionField(StrictModel):
    """One monic separable extension ``GF(p)(x)[y]/(f)``."""

    characteristic: int = Field(ge=2, le=MAX_CHARACTERISTIC)
    variable: FieldVariable = Field(
        default="x", description="Name of the rational-function variable."
    )
    generator: FieldVariable = Field(
        default="y", description="Name of the extension generator y."
    )
    defining_polynomial: tuple[PrimeFieldRationalFunction, ...] = Field(
        min_length=1, max_length=MAX_EXTENSION_DEGREE + 1
    )

    @model_validator(mode="after")
    def require_monic_extension(self) -> Self:
        if self.variable == self.generator:
            raise _validation_error(
                "extension_variable_collision",
                "the rational variable and extension generator must be distinct",
            )
        if any(
            coefficient.characteristic != self.characteristic
            for coefficient in self.defining_polynomial
        ):
            raise _validation_error(
                "extension_characteristic",
                "every defining-polynomial coefficient must share the characteristic",
            )
        if len(self.defining_polynomial) == 1:
            if not (
                self.defining_polynomial[0].numerator.is_one()
                and self.defining_polynomial[0].denominator.is_one()
            ):
                raise _validation_error(
                    "rational_field_polynomial",
                    "the rational-field defining polynomial is 1",
                )
        elif not self.defining_polynomial[-1].numerator.is_one() or not (
            self.defining_polynomial[-1].denominator.is_one()
        ):
            raise _validation_error(
                "extension_monic",
                "the defining polynomial must be monic in the generator",
            )
        if len(self.defining_polynomial) - 1 > MAX_EXTENSION_DEGREE:
            raise _validation_error(
                "extension_degree_bound",
                "extension degree exceeds the admitted bound",
            )
        return self

    @property
    def degree(self) -> int:
        return max(1, len(self.defining_polynomial) - 1)


class FunctionFieldPlace(StrictModel):
    """A rational/infinite place bound to one exact function field."""

    field: FiniteFunctionField
    kind: Literal["FINITE", "INFINITE"]
    prime_polynomial: PrimeFieldPolynomial | None = None
    degree: int = Field(ge=1)

    @model_validator(mode="after")
    def require_place_shape(self) -> Self:
        if self.kind == "FINITE":
            if (
                self.prime_polynomial is None
                or self.prime_polynomial.characteristic != self.field.characteristic
                or self.prime_polynomial.is_zero()
                or self.prime_polynomial.degree < 1
            ):
                raise _validation_error(
                    "place_prime",
                    "a finite place requires a nonconstant prime polynomial of the field characteristic",
                )
            if self.degree != self.prime_polynomial.degree:
                raise _validation_error(
                    "place_degree", "place degree must match its prime polynomial"
                )
        elif self.prime_polynomial is not None or self.degree != 1:
            raise _validation_error(
                "infinite_place_shape",
                "the infinite place has degree one and no prime polynomial",
            )
        return self


class HyperellipticAffinePlace(StrictModel):
    """A rational affine point on a supported odd-characteristic y^2=f(x) model."""

    field: FiniteFunctionField
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    local_parameter: Literal["x_minus_x0", "y"]
    residue_field: FiniteFieldPresentation

    @model_validator(mode="after")
    def require_canonical_point_shape(self) -> Self:
        prime = self.field.characteristic
        if self.x >= prime or self.y >= prime:
            raise _validation_error(
                "affine_point_coordinates",
                "affine coordinates must be canonical residues of the characteristic",
            )
        if (
            self.residue_field.characteristic != prime
            or self.residue_field.modulus_coefficients != (0, 1)
            or self.residue_field.generator != "a"
        ):
            raise _validation_error(
                "affine_place_residue_parent",
                "a rational affine point has residue parent GF(p) with modulus z",
            )
        if (self.y == 0) != (self.local_parameter == "y"):
            raise _validation_error(
                "affine_place_uniformizer",
                "use y at a branch point and x-x0 when y is nonzero",
            )
        return self


class HyperellipticInfinityPlace(StrictModel):
    """The unique degree-one place at infinity on an odd-degree model."""

    field: FiniteFunctionField
    residue_field: FiniteFieldPresentation

    @model_validator(mode="after")
    def require_prime_constant_residue(self) -> Self:
        prime = self.field.characteristic
        if (
            self.residue_field.characteristic != prime
            or self.residue_field.modulus_coefficients != (0, 1)
        ):
            raise _validation_error(
                "infinity_place_residue_parent",
                "the odd-degree point at infinity is rational over GF(p)",
            )
        return self


class HyperellipticAffinePlaceValuationRequest(StrictModel):
    place: HyperellipticAffinePlace
    element: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_shared_parent(self) -> Self:
        from jacobian.math.function_fields.operations import _canonical_field

        try:
            place_field = FiniteFunctionField.model_validate(
                self.place.field.model_dump()
            )
            element_field = FiniteFunctionField.model_validate(
                self.element.field.model_dump()
            )
        except (TypeError, ValueError) as error:
            raise _validation_error(
                "affine_valuation_parent_malformed",
                "the point and function element fields must be valid",
            ) from error
        if _canonical_field(place_field) != _canonical_field(element_field):
            raise _validation_error(
                "affine_valuation_parent_mismatch",
                "the point and function element must share the exact function field",
            )
        return self


class HyperellipticInfinityPlaceValuationRequest(StrictModel):
    place: HyperellipticInfinityPlace
    element: FiniteFunctionFieldElement


class FunctionFieldFiniteValuation(StrictModel):
    """A finite integer valuation, including finite value zero."""

    kind: Literal["FINITE"]
    value: int


class FunctionFieldPositiveInfinityValuation(StrictModel):
    """The valuation of the zero field element; it carries no numeric value."""

    kind: Literal["POSITIVE_INFINITY"]


FunctionFieldValuation = Annotated[
    FunctionFieldFiniteValuation | FunctionFieldPositiveInfinityValuation,
    Field(discriminator="kind"),
]


class HyperellipticAffinePlaceValuationResult(StrictModel):
    place: HyperellipticAffinePlace
    element: FiniteFunctionFieldElement
    valuation: FunctionFieldValuation

    @model_validator(mode="after")
    def require_shared_parent(self) -> Self:
        if self.place.field != self.element.field:
            raise _validation_error(
                "affine_valuation_parent_mismatch",
                "the point and function element must retain the exact function field",
            )
        return self


class HyperellipticInfinityPlaceValuationResult(StrictModel):
    place: HyperellipticInfinityPlace
    element: FiniteFunctionFieldElement
    valuation: FunctionFieldValuation


class FunctionFieldDivisorTerm(StrictModel):
    place: FunctionFieldPlace
    multiplicity: DivisorMultiplicity


class FunctionFieldDivisor(StrictModel):
    field: FiniteFunctionField
    terms: tuple[FunctionFieldDivisorTerm, ...] = Field(max_length=256)

    @model_validator(mode="after")
    def require_divisor_parent(self) -> Self:
        if any(
            term.place.field != self.field or term.multiplicity == 0
            for term in self.terms
        ):
            raise _validation_error(
                "divisor_parent",
                "divisor places must belong to the field and have nonzero multiplicity",
            )
        if len({term.place.model_dump_json() for term in self.terms}) != len(
            self.terms
        ):
            raise _validation_error(
                "divisor_duplicate", "divisor support must be unique"
            )
        return self

    @property
    def degree(self) -> int:
        return sum(term.multiplicity * term.place.degree for term in self.terms)


class FunctionFieldPlaceValuationRequest(StrictModel):
    place: FunctionFieldPlace
    element: FiniteFunctionFieldElement


class FunctionFieldPlaceValuationResult(StrictModel):
    place: FunctionFieldPlace
    element: FiniteFunctionFieldElement
    valuation: FunctionFieldValuation


class FunctionFieldPrincipalDivisorRequest(StrictModel):
    field: FiniteFunctionField
    element: FiniteFunctionFieldElement


class FunctionFieldDivisorRequest(StrictModel):
    divisor: FunctionFieldDivisor


class FunctionFieldDivisorAddRequest(StrictModel):
    left: FunctionFieldDivisor
    right: FunctionFieldDivisor


class FunctionFieldDivisorScaleRequest(StrictModel):
    divisor: FunctionFieldDivisor
    scalar: DivisorMultiplicity


class FunctionFieldDivisorEffectivePartsResult(StrictModel):
    """The effective decomposition ``D = D_+ - D_-`` of a divisor."""

    divisor: FunctionFieldDivisor
    positive_part: FunctionFieldDivisor
    negative_part: FunctionFieldDivisor


class FunctionFieldDivisorDegreeResult(StrictModel):
    divisor: FunctionFieldDivisor
    degree: DivisorDegree


class FunctionFieldPrincipalDivisorResult(StrictModel):
    field: FiniteFunctionField
    element: FiniteFunctionFieldElement
    divisor: FunctionFieldDivisor
    degree: DivisorDegree


class FunctionFieldGenusRequest(StrictModel):
    field: FiniteFunctionField = Field(
        description=(
            "GF(p)(x), or a supported odd-characteristic quadratic "
            "hyperelliptic extension y^2=f(x) with squarefree polynomial f."
        )
    )


class FunctionFieldGenusResult(StrictModel):
    field: FiniteFunctionField
    genus: int = Field(ge=0)


class FunctionFieldPlaceEnumerationRequest(StrictModel):
    field: FiniteFunctionField
    maximum_degree: int = Field(ge=1, le=MAX_RATIONAL_PLACE_DEGREE)


class FunctionFieldPlaceEnumerationResult(StrictModel):
    """Complete finite/infinite places through one degree bound."""

    field: FiniteFunctionField
    maximum_degree: int = Field(ge=1, le=MAX_RATIONAL_PLACE_DEGREE)
    places: tuple[FunctionFieldPlace, ...] = Field(max_length=MAX_RATIONAL_PLACE_OUTPUT)


class FiniteFunctionFieldElement(StrictModel):
    """Reduced coordinates in the generator-power basis of one function field."""

    field: FiniteFunctionField
    coordinates: tuple[PrimeFieldRationalFunction, ...] = Field(
        min_length=1, max_length=MAX_EXTENSION_DEGREE
    )

    @model_validator(mode="after")
    def require_reduced_coordinates(self) -> Self:
        if len(self.coordinates) != self.field.degree:
            raise _validation_error(
                "element_coordinate_count",
                "coordinates must have exactly one entry per power-basis element",
            )
        if any(
            coordinate.characteristic != self.field.characteristic
            for coordinate in self.coordinates
        ):
            raise _validation_error(
                "element_characteristic",
                "every coordinate must share the field characteristic",
            )
        return self


class FunctionFieldResidueRequest(StrictModel):
    """Reduce a function regular at one rational-function-field place."""

    place: FunctionFieldPlace
    element: FiniteFunctionFieldElement


class FunctionFieldResidueResult(StrictModel):
    """The exact residue, bound to its source place and finite field."""

    place: FunctionFieldPlace
    element: FiniteFunctionFieldElement
    residue: FiniteFieldElement

    @model_validator(mode="after")
    def require_residue_parent(self) -> Self:
        presentation = self.residue.presentation
        expected_modulus = (
            self.place.prime_polynomial.coefficients
            if self.place.kind == "FINITE" and self.place.prime_polynomial is not None
            else (0, 1)
        )
        if (
            presentation.characteristic != self.place.field.characteristic
            or presentation.modulus_coefficients != expected_modulus
        ):
            raise _validation_error(
                "residue_characteristic",
                "residue presentation must be the exact quotient attached to the place",
            )
        if self.element.field != self.place.field:
            raise _validation_error(
                "residue_element_parent",
                "residue input must belong to the place function field",
            )
        return self


class FunctionFieldBaseEmbedding(StrictModel):
    """Canonical inclusion GF(p)(x) into one presented extension field."""

    source: FiniteFunctionField
    target: FiniteFunctionField
    variable_image: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_canonical_base_inclusion(self) -> Self:
        # The source must be the rational-field sentinel itself, not a merely
        # degree-one linear presentation, so that applying this embedding is
        # the canonical inclusion and never an implicit change of parent.
        rational_sentinel = len(self.source.defining_polynomial) == 1 and (
            self.source.defining_polynomial[0].numerator.is_one()
            and self.source.defining_polynomial[0].denominator.is_one()
        )
        if (
            not rational_sentinel
            or self.target.degree <= 1
            or self.source.characteristic != self.target.characteristic
            or self.source.variable != self.target.variable
            or self.source.generator != self.target.generator
        ):
            raise _validation_error(
                "base_embedding_parent",
                "a base embedding requires GF(p)(x) and an extension over the same GF(p)(x)",
            )
        if self.variable_image.field != self.target:
            raise _validation_error(
                "base_embedding_image_parent",
                "the rational-variable image must belong to the target field",
            )
        prime = self.target.characteristic
        expected_x = PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(characteristic=prime, coefficients=(0, 1)),
            denominator=PrimeFieldPolynomial(characteristic=prime, coefficients=(1,)),
        )
        expected_zero = PrimeFieldRationalFunction(
            numerator=PrimeFieldPolynomial(characteristic=prime, coefficients=(0,)),
            denominator=PrimeFieldPolynomial(characteristic=prime, coefficients=(1,)),
        )
        if self.variable_image.coordinates != (
            expected_x,
            *(expected_zero for _ in range(self.target.degree - 1)),
        ):
            raise _validation_error(
                "base_embedding_variable_image",
                "the base inclusion must send its rational variable to the target variable",
            )
        return self


class FunctionFieldBaseEmbeddingRequest(StrictModel):
    target: FiniteFunctionField


class FunctionFieldBaseEmbeddingResult(StrictModel):
    embedding: FunctionFieldBaseEmbedding


class FunctionFieldBaseEmbeddingApplyRequest(StrictModel):
    embedding: FunctionFieldBaseEmbedding
    element: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_source_element(self) -> Self:
        if self.element.field != self.embedding.source:
            raise _validation_error(
                "base_embedding_element_parent",
                "element must belong to the embedding source field",
            )
        return self


class FunctionFieldBaseEmbeddingApplyResult(StrictModel):
    embedding: FunctionFieldBaseEmbedding
    source_element: FiniteFunctionFieldElement
    image: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_bound_values(self) -> Self:
        if (
            self.source_element.field != self.embedding.source
            or self.image.field != self.embedding.target
        ):
            raise _validation_error(
                "base_embedding_result_parent",
                "source and image elements must match the embedding parents",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        embedding: FunctionFieldBaseEmbedding,
        source_element: FiniteFunctionFieldElement,
        image: FiniteFunctionFieldElement,
    ) -> Self:
        return cls.model_construct(
            embedding=embedding, source_element=source_element, image=image
        )


class FunctionFieldRiemannRochSpace(StrictModel):
    """A divisor-bound finite-dimensional space with an exact basis."""

    divisor: FunctionFieldDivisor
    dimension: int = Field(ge=0, le=MAX_RIEMANN_ROCH_BASIS_DIMENSION)
    basis: tuple[FiniteFunctionFieldElement, ...] = Field(
        max_length=MAX_RIEMANN_ROCH_BASIS_DIMENSION
    )

    @model_validator(mode="after")
    def require_basis_dimension_and_parent(self) -> Self:
        if self.dimension != len(self.basis):
            raise _validation_error(
                "riemann_roch_basis_dimension",
                "space dimension must equal the number of basis elements",
            )
        if any(element.field != self.divisor.field for element in self.basis):
            raise _validation_error(
                "riemann_roch_basis_parent",
                "every basis element must belong to the divisor function field",
            )
        return self


class FunctionFieldRiemannRochSpaceRequest(StrictModel):
    divisor: FunctionFieldDivisor = Field(
        description=(
            "A finite divisor over GF(p)(x), or m times the unique infinity "
            "place of an odd-degree squarefree hyperelliptic model y^2=f(x). "
            "Support is limited to 256 terms and multiplicities to 4096 bits. "
            "Positive-dimensional outputs are admitted only when their exact "
            "canonical basis fits the degree-12 coefficient envelope."
        )
    )


class FunctionFieldRiemannRochMembershipRequest(StrictModel):
    """A function and finite divisor whose exact Riemann-Roch membership is asked."""

    element: FiniteFunctionFieldElement
    divisor: FunctionFieldDivisor

    @model_validator(mode="after")
    def require_shared_parent(self) -> Self:
        if self.element.field != self.divisor.field:
            raise _validation_error(
                "riemann_roch_membership_parent",
                "element and divisor must belong to the same exact function field",
            )
        return self


class FunctionFieldRiemannRochMembershipRow(StrictModel):
    """One exact valuation inequality at a place in the complete support union."""

    place: FunctionFieldPlace
    element_valuation: int = Field(
        strict=True, ge=-MAX_POLYNOMIAL_X_DEGREE, le=MAX_POLYNOMIAL_X_DEGREE
    )
    divisor_multiplicity: DivisorMultiplicity
    sum: DivisorMultiplicity

    @model_validator(mode="after")
    def require_exact_sum(self) -> Self:
        if self.sum != self.element_valuation + self.divisor_multiplicity:
            raise _validation_error(
                "riemann_roch_membership_sum",
                "the returned sum must equal valuation plus divisor multiplicity",
            )
        if self.element_valuation == 0 and self.divisor_multiplicity == 0:
            raise _validation_error(
                "riemann_roch_membership_empty_row",
                "profile rows must belong to the union of nonzero supports",
            )
        return self


class FunctionFieldRiemannRochMembership(StrictModel):
    """Exact membership in ``L(D)`` with the complete support inequalities."""

    element: FiniteFunctionFieldElement
    divisor: FunctionFieldDivisor
    status: Literal["IN_SPACE", "NOT_IN_SPACE"]
    profile: tuple[FunctionFieldRiemannRochMembershipRow, ...] = Field(
        max_length=MAX_RIEMANN_ROCH_MEMBERSHIP_PROFILE_ROWS
    )

    @model_validator(mode="after")
    def require_complete_profile_shape(self) -> Self:
        if self.element.field != self.divisor.field:
            raise _validation_error(
                "riemann_roch_membership_parent",
                "element and divisor must retain one exact function field",
            )
        places = tuple(row.place.model_dump_json() for row in self.profile)
        if len(set(places)) != len(places) or places != tuple(sorted(places)):
            raise _validation_error(
                "riemann_roch_membership_profile_order",
                "membership profile places must be unique and canonically ordered",
            )
        if any(row.place.field != self.divisor.field for row in self.profile):
            raise _validation_error(
                "riemann_roch_membership_profile_parent",
                "every profile place must belong to the divisor function field",
            )
        zero = all(
            coordinate.numerator.is_zero() for coordinate in self.element.coordinates
        )
        if zero:
            if self.status != "IN_SPACE" or self.profile:
                raise _validation_error(
                    "riemann_roch_membership_zero_branch",
                    "zero belongs to every L(D) through its structural empty-profile branch",
                )
            return self
        in_space = all(row.sum >= 0 for row in self.profile)
        if (self.status == "IN_SPACE") != in_space:
            raise _validation_error(
                "riemann_roch_membership_status",
                "membership status must agree with every returned valuation inequality",
            )
        return self


class FunctionFieldElementMultiplyRequest(StrictModel):
    """Two reduced elements of one declared finite function field."""

    left: FiniteFunctionFieldElement
    right: FiniteFunctionFieldElement


class FunctionFieldElementAddRequest(StrictModel):
    """Two elements whose coordinates share one exact function-field parent."""

    left: FiniteFunctionFieldElement
    right: FiniteFunctionFieldElement


class FunctionFieldElementInverseRequest(StrictModel):
    """One element to invert in its presented function-field parent."""

    element: FiniteFunctionFieldElement


class FunctionFieldProductTerm(StrictModel):
    """One nonzero generator-power term of an exact product."""

    y_power: int = Field(ge=0, le=2 * MAX_EXTENSION_DEGREE)
    coefficient: PrimeFieldRationalFunction


class FunctionFieldReductionStep(StrictModel):
    """One exact reduction of a high generator power modulo the defining polynomial."""

    y_power: int = Field(ge=1, le=2 * MAX_EXTENSION_DEGREE)
    coefficient: PrimeFieldRationalFunction
    replacement_terms: tuple[FunctionFieldProductTerm, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )


class FunctionFieldElementMultiplyResult(StrictModel):
    """The canonical reduced product with its complete reduction ledger."""

    field: FiniteFunctionField
    left: FiniteFunctionFieldElement
    right: FiniteFunctionFieldElement
    raw_product_terms: tuple[FunctionFieldProductTerm, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )
    reduction_steps: tuple[FunctionFieldReductionStep, ...] = Field(
        max_length=MAX_LEDGER_ROWS
    )
    product: FiniteFunctionFieldElement

    @model_validator(mode="after")
    def require_bound_result(self) -> Self:
        if (
            self.left.field != self.field
            or self.right.field != self.field
            or self.product.field != self.field
        ):
            raise _validation_error(
                "result_field",
                "inputs and product must share the declared function field",
            )
        if tuple(step.y_power for step in self.reduction_steps) != tuple(
            sorted(step.y_power for step in self.reduction_steps)
        ):
            raise _validation_error(
                "reduction_order",
                "reduction steps must be ordered by generator power",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        field: FiniteFunctionField,
        left: FiniteFunctionFieldElement,
        right: FiniteFunctionFieldElement,
        raw_product_terms: tuple[FunctionFieldProductTerm, ...],
        reduction_steps: tuple[FunctionFieldReductionStep, ...],
        product: FiniteFunctionFieldElement,
    ) -> FunctionFieldElementMultiplyResult:
        return cls.model_construct(
            field=field,
            left=left,
            right=right,
            raw_product_terms=raw_product_terms,
            reduction_steps=reduction_steps,
            product=product,
        )


__all__ = [
    "MAX_BASE_EMBEDDING_VALUE_BYTES",
    "MAX_CHARACTERISTIC",
    "MAX_DIVISOR_DEGREE_DIGITS",
    "MAX_DIVISOR_MULTIPLICITY_DIGITS",
    "MAX_ELEMENT_ADDITION_WORK",
    "MAX_ELEMENT_VALUE_BYTES",
    "MAX_EXTENSION_DEGREE",
    "MAX_FIELD_ADMISSION_WORK",
    "MAX_INVERSION_WORK",
    "MAX_LEDGER_ROWS",
    "MAX_MULTIPLICATION_WORK",
    "MAX_POLYNOMIAL_COEFFICIENTS",
    "MAX_POLYNOMIAL_X_DEGREE",
    "DivisorDegree",
    "DivisorMultiplicity",
    "FiniteFunctionField",
    "FiniteFunctionFieldElement",
    "FunctionFieldBaseEmbedding",
    "FunctionFieldBaseEmbeddingApplyRequest",
    "FunctionFieldBaseEmbeddingApplyResult",
    "FunctionFieldBaseEmbeddingRequest",
    "FunctionFieldBaseEmbeddingResult",
    "FunctionFieldDivisor",
    "FunctionFieldDivisorDegreeResult",
    "FunctionFieldDivisorRequest",
    "FunctionFieldDivisorTerm",
    "FunctionFieldElementAddRequest",
    "FunctionFieldElementInverseRequest",
    "FunctionFieldElementMultiplyRequest",
    "FunctionFieldElementMultiplyResult",
    "FunctionFieldPlace",
    "FunctionFieldPlaceValuationRequest",
    "FunctionFieldPlaceValuationResult",
    "FunctionFieldPrincipalDivisorRequest",
    "FunctionFieldPrincipalDivisorResult",
    "FunctionFieldProductTerm",
    "FunctionFieldReductionStep",
    "FunctionFieldResidueRequest",
    "FunctionFieldResidueResult",
    "PrimeFieldPolynomial",
    "PrimeFieldRationalFunction",
]
