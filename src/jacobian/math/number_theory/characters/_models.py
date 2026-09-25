"""Typed wire contracts for exact bounded principal Dirichlet characters."""

from __future__ import annotations

from math import gcd, lcm
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, ExactInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.values import (
    MAX_CHARACTER_GROUP_MODULUS,
    MAX_PRINCIPAL_CHARACTER_MODULUS,
    CyclotomicValue,
    DirichletCharacter,
    DirichletCharacterGroup,
    PrincipalDirichletCharacter,
)
from jacobian.math.number_theory.sequences.core._models import (
    FiniteCyclotomicSequence,
    FiniteIntegerSequence,
    FiniteRationalSequence,
)

MAX_INTEGER_DIGITS = 256
MAX_GENERALIZED_BERNOULLI_INDEX = 32


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Dirichlet-character contracts."""

    return PydanticCustomError(f"dirichlet_character.{reason}", message)


def _require_bounded_digits(value: int) -> int:
    if len(format_canonical_integer(abs(value))) > MAX_INTEGER_DIGITS:
        raise _validation_error(
            "integer_digit_bound",
            f"integer exceeds the {MAX_INTEGER_DIGITS}-digit bound",
        )
    return value


DirichletCharacterInteger = Annotated[
    ExactInteger,
    AfterValidator(_require_bounded_digits),
]


class DirichletCharacterRequest(StrictModel):
    group: DirichletCharacterGroup
    coordinates: tuple[StrictInt, ...]


class DirichletCharacterValueRequest(StrictModel):
    character: DirichletCharacter
    integer: DirichletCharacterInteger


class DirichletCharacterOrderRequest(StrictModel):
    """Compute the multiplicative order of one exact character."""

    character: DirichletCharacter


class DirichletCharacterKernelRequest(StrictModel):
    """Compute the exact residue subgroup on which one character is trivial."""

    character: DirichletCharacter


class DirichletCharacterInflationRequest(StrictModel):
    """Inflate a character to a bounded multiple modulus."""

    character: DirichletCharacter
    target_modulus: StrictInt = Field(
        ge=1,
        le=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Target modulus must be a positive multiple of the source modulus; "
            "the target unit group is bounded to 2,048 residues."
        ),
    )


class DirichletCharacterRestrictionRequest(StrictModel):
    """Factor a character through reduction to one bounded divisor modulus."""

    character: DirichletCharacter
    target_modulus: StrictInt = Field(
        ge=1,
        le=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Target modulus must divide the source modulus; the complete target "
            "unit group and lift map are bounded to 2,048 residues."
        ),
    )


class DirichletCharacterOrderResult(StrictModel):
    """Exact character order bound to its finite dual-group parent."""

    character: DirichletCharacter
    order: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)

    @model_validator(mode="after")
    def require_group_exponent_bound(self) -> Self:
        expected = lcm(
            *(
                axis_order // gcd(coordinate, axis_order)
                for coordinate, axis_order in zip(
                    self.character.coordinates,
                    self.character.group.generator_orders,
                    strict=True,
                )
            )
        )
        if self.order != expected:
            raise _validation_error(
                "order_mismatch",
                "character order must equal the order induced by its dual coordinates",
            )
        return self


class DirichletCharacterParityRequest(StrictModel):
    """Determine the exact value of a character at minus one."""

    character: DirichletCharacter


class DirichletCharacterParityResult(StrictModel):
    """Exact even/odd character value, retaining its source character."""

    character: DirichletCharacter
    value: CyclotomicValue
    parity: Literal["EVEN", "ODD"]

    @model_validator(mode="after")
    def require_sign_root(self) -> Self:
        order = self.character.group.exponent
        if (
            self.value.order != order
            or (self.parity == "EVEN" and self.value.exponent != 0)
            or (
                self.parity == "ODD"
                and (order % 2 != 0 or self.value.exponent != order // 2)
            )
        ):
            raise _validation_error(
                "parity_value_mismatch",
                "parity must agree with the exact value of chi(-1)",
            )
        return self


class DirichletCharacterValueResult(StrictModel):
    character: DirichletCharacter
    integer: DirichletCharacterInteger
    canonical_residue: StrictInt
    is_unit: StrictBool
    value: CyclotomicValue | None


class DirichletCharacterProductRequest(StrictModel):
    left: DirichletCharacter
    right: DirichletCharacter


class DirichletCharacterPowerRequest(StrictModel):
    """Raise a character to a bounded signed integer power."""

    character: DirichletCharacter
    exponent: DirichletCharacterInteger


class DirichletCharacterConjugateRequest(StrictModel):
    """Conjugate one exact source-bound Dirichlet character."""

    character: DirichletCharacter


class DirichletCharacterInverseRequest(StrictModel):
    """Invert one exact source-bound Dirichlet character in its dual group."""

    character: DirichletCharacter


class DirichletCharacterOrthogonalityRequest(StrictModel):
    """Evaluate the exact conjugate-pairing sum of two characters."""

    left: DirichletCharacter
    right: DirichletCharacter


class DirichletCharacterOrthogonalityResult(StrictModel):
    """An exact character-pairing integer bound to its two source characters."""

    left: DirichletCharacter
    right: DirichletCharacter
    value: StrictInt = Field(
        ge=-MAX_CHARACTER_GROUP_MODULUS,
        le=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Exact sum of chi(a) conjugate(psi(a)) over residues modulo the "
            "shared modulus; its absolute value is at most phi(modulus)."
        ),
    )

    @model_validator(mode="after")
    def require_source_bound_pairing_shape(self) -> Self:
        if self.left.group != self.right.group:
            raise _validation_error(
                "orthogonality_parent_mismatch",
                "orthogonality characters must use the identical group parent",
            )
        if abs(self.value) > self.left.group.character_count:
            raise _validation_error(
                "orthogonality_value_bound",
                "pairing value cannot exceed the number of units in the group",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        left: DirichletCharacter,
        right: DirichletCharacter,
        value: int,
    ) -> Self:
        return cls.model_construct(left=left, right=right, value=value)


class DirichletCharacterOrthogonalityOverCharactersRequest(StrictModel):
    """Sum the full dual group over one source-bound residue pair."""

    group: DirichletCharacterGroup
    left_integer: DirichletCharacterInteger
    right_integer: DirichletCharacterInteger


class DirichletCharacterOrthogonalityOverCharactersResult(StrictModel):
    """Exact dual-group pairing, with canonical residue and parent binding."""

    group: DirichletCharacterGroup
    left_integer: DirichletCharacterInteger
    right_integer: DirichletCharacterInteger
    left_residue: StrictInt = Field(ge=0, le=MAX_CHARACTER_GROUP_MODULUS - 1)
    right_residue: StrictInt = Field(ge=0, le=MAX_CHARACTER_GROUP_MODULUS - 1)
    value: StrictInt = Field(ge=0, le=MAX_CHARACTER_GROUP_MODULUS)

    @model_validator(mode="after")
    def require_source_bound_residues(self) -> Self:
        if (
            self.left_residue != int(self.left_integer) % self.group.modulus
            or self.right_residue != int(self.right_integer) % self.group.modulus
            or self.value > self.group.character_count
        ):
            raise _validation_error(
                "dual_orthogonality_result_shape",
                "dual orthogonality result must retain canonical source residues and a bounded sum",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        group: DirichletCharacterGroup,
        left_integer: DirichletCharacterInteger,
        right_integer: DirichletCharacterInteger,
        left_residue: int,
        right_residue: int,
        value: int,
    ) -> Self:
        return cls.model_construct(
            group=group,
            left_integer=left_integer,
            right_integer=right_integer,
            left_residue=left_residue,
            right_residue=right_residue,
            value=value,
        )


class DirichletCharacterGeneralizedBernoulliRequest(StrictModel):
    """Compute one generalized Bernoulli number for a bounded index."""

    character: DirichletCharacter
    index: StrictInt = Field(
        ge=0,
        le=MAX_GENERALIZED_BERNOULLI_INDEX,
        description=(
            "Nonnegative Bernoulli index; the public computation is bounded "
            f"to {MAX_GENERALIZED_BERNOULLI_INDEX}."
        ),
    )


class DirichletCharacterGeneralizedBernoulliResult(StrictModel):
    """An exact generalized Bernoulli value bound to its character and index."""

    character: DirichletCharacter
    index: StrictInt = Field(ge=0, le=MAX_GENERALIZED_BERNOULLI_INDEX)
    value: RationalCyclotomicElement

    @model_validator(mode="after")
    def require_character_value_field(self) -> Self:
        character_order = 1
        for coordinate, axis_order in zip(
            self.character.coordinates,
            self.character.group.generator_orders,
            strict=True,
        ):
            character_order = lcm(
                character_order,
                axis_order // gcd(coordinate, axis_order),
            )
        if self.value.field.order != character_order:
            raise _validation_error(
                "generalized_bernoulli_field_mismatch",
                "value field order must equal the source character value order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        character: DirichletCharacter,
        index: int,
        value: RationalCyclotomicElement,
    ) -> Self:
        return cls.model_construct(character=character, index=index, value=value)


class DirichletCharacterGeneralizedBernoulliPrefixRequest(StrictModel):
    """Compute the generalized Bernoulli values from zero through an index."""

    character: DirichletCharacter
    maximum_index: StrictInt = Field(ge=0, le=MAX_GENERALIZED_BERNOULLI_INDEX)


class DirichletCharacterGeneralizedBernoulliPrefix(StrictModel):
    """Exact prefix B(0,chi), ..., B(maximum_index,chi)."""

    character: DirichletCharacter
    maximum_index: StrictInt = Field(ge=0, le=MAX_GENERALIZED_BERNOULLI_INDEX)
    values: tuple[RationalCyclotomicElement, ...]

    @model_validator(mode="after")
    def require_prefix_shape_and_fields(self) -> Self:
        if len(self.values) != self.maximum_index + 1:
            raise _validation_error(
                "generalized_bernoulli_prefix_length",
                "values must contain exactly one entry for each index from zero through maximum_index",
            )
        character_order = 1
        for coordinate, axis_order in zip(
            self.character.coordinates,
            self.character.group.generator_orders,
            strict=True,
        ):
            character_order = lcm(
                character_order, axis_order // gcd(coordinate, axis_order)
            )
        if any(value.field.order != character_order for value in self.values):
            raise _validation_error(
                "generalized_bernoulli_field_mismatch",
                "each value field order must equal the source character value order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        character: DirichletCharacter,
        maximum_index: int,
        values: tuple[RationalCyclotomicElement, ...],
    ) -> Self:
        return cls.model_construct(
            character=character, maximum_index=maximum_index, values=values
        )


class DirichletCharacterLValueNonpositiveRequest(StrictModel):
    """Compute L(1-k, chi) from one generalized Bernoulli value."""

    character: DirichletCharacter
    bernoulli_index: StrictInt = Field(
        ge=1,
        le=MAX_GENERALIZED_BERNOULLI_INDEX,
        description=(
            "k in L(1-k, chi) = -B(k, chi)/k; k is bounded to "
            f"1..{MAX_GENERALIZED_BERNOULLI_INDEX}."
        ),
    )


class DirichletCharacterLValueNonpositiveResult(StrictModel):
    """Exact algebraic L-value, bound to its character and integer argument."""

    character: DirichletCharacter
    bernoulli_index: StrictInt = Field(ge=1, le=MAX_GENERALIZED_BERNOULLI_INDEX)
    argument: StrictInt = Field(le=0)
    value: RationalCyclotomicElement

    @model_validator(mode="after")
    def require_argument_and_character_field(self) -> Self:
        if self.argument != 1 - self.bernoulli_index:
            raise _validation_error(
                "l_value_argument_mismatch",
                "L-value argument must equal 1 minus the Bernoulli index",
            )
        character_order = 1
        for coordinate, axis_order in zip(
            self.character.coordinates,
            self.character.group.generator_orders,
            strict=True,
        ):
            character_order = lcm(
                character_order,
                axis_order // gcd(coordinate, axis_order),
            )
        if self.value.field.order != character_order:
            raise _validation_error(
                "l_value_field_mismatch",
                "value field order must equal the source character value order",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        character: DirichletCharacter,
        bernoulli_index: int,
        value: RationalCyclotomicElement,
    ) -> Self:
        return cls.model_construct(
            character=character,
            bernoulli_index=bernoulli_index,
            argument=1 - bernoulli_index,
            value=value,
        )


class DirichletCharacterSequenceTwistRequest(StrictModel):
    """Twist one exact finite sequence at a bounded index origin."""

    sequence: Annotated[
        FiniteIntegerSequence | FiniteRationalSequence | FiniteCyclotomicSequence,
        Field(discriminator="domain"),
    ]
    character: DirichletCharacter
    index_origin: StrictInt | None = Field(default=None, ge=-(2**31), le=2**31 - 1)

    @model_validator(mode="after")
    def require_consistent_index_origin(self) -> Self:
        if isinstance(self.sequence, FiniteCyclotomicSequence):
            if (
                self.index_origin is not None
                and self.index_origin != self.sequence.index_origin
            ):
                raise _validation_error(
                    "sequence_twist.index_origin_mismatch",
                    "an existing cyclotomic sequence keeps its authored index origin",
                )
        elif self.index_origin is None:
            raise _validation_error(
                "sequence_twist.index_origin_required",
                "integer and rational sequences require an explicit index origin",
            )
        return self


class DirichletCharacterJacobiSumRequest(StrictModel):
    """Compute J(chi, psi) when both characters use one exact group parent."""

    left: DirichletCharacter
    right: DirichletCharacter


class DirichletCharacterJacobiSumResult(StrictModel):
    """Exact Jacobi sum in the canonical rational cyclotomic field."""

    left: DirichletCharacter
    right: DirichletCharacter
    value: RationalCyclotomicElement


class DirichletCharacterGaussSumRequest(StrictModel):
    """Compute the additive Gauss sum of one source-bound character."""

    character: DirichletCharacter


class DirichletCharacterGaussSumResult(StrictModel):
    """Exact Gauss sum bound to its source character."""

    character: DirichletCharacter
    value: RationalCyclotomicElement


class DirichletCharacterPrimitiveGaussNormResult(StrictModel):
    """Exact squared complex modulus of a primitive character Gauss sum."""

    character: DirichletCharacter
    conductor: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    gauss_sum: RationalCyclotomicElement
    norm_squared: RationalCyclotomicElement

    @model_validator(mode="after")
    def require_primitive_source_and_norm(self) -> Self:
        if self.conductor != self.character.group.modulus:
            raise _validation_error(
                "gauss_norm_not_primitive",
                "character modulus must equal its computed conductor",
            )
        if self.gauss_sum.field != self.norm_squared.field:
            raise _validation_error(
                "gauss_norm_field", "Gauss sum and norm must use one cyclotomic parent"
            )
        expected = (
            CanonicalRational.from_integer_ratio(self.conductor, 1),
            *(
                CanonicalRational.from_integer_ratio(0, 1)
                for _ in range(self.norm_squared.field.degree - 1)
            ),
        )
        if self.norm_squared.coefficients_ascending != expected:
            raise _validation_error(
                "gauss_norm_value", "squared complex modulus must equal the conductor"
            )
        return self


class DirichletCharacterPrimitiveGaussNormRequest(StrictModel):
    character: DirichletCharacter


class DirichletCharacterGeneralizedGaussSumRequest(StrictModel):
    """Compute an additive character sum at an explicit integer frequency."""

    character: DirichletCharacter
    frequency: DirichletCharacterInteger


class DirichletCharacterGeneralizedGaussSumResult(StrictModel):
    """Exact generalized Gauss sum bound to its character and frequency."""

    character: DirichletCharacter
    frequency: DirichletCharacterInteger
    frequency_residue: StrictInt = Field(ge=0)
    value: RationalCyclotomicElement

    @model_validator(mode="after")
    def require_frequency_and_cyclotomic_parent(self) -> Self:
        group = self.character.group
        value_order = 1
        for coordinate, generator_order in zip(
            self.character.coordinates, group.generator_orders, strict=True
        ):
            value_order = lcm(
                value_order,
                generator_order // gcd(coordinate, generator_order),
            )
        expected_order = lcm(value_order, group.modulus)
        if self.frequency_residue != self.frequency % group.modulus:
            raise _validation_error(
                "generalized_gauss_sum_frequency",
                "frequency residue must be the canonical reduction modulo the character modulus",
            )
        if self.value.field != RationalCyclotomicField(order=expected_order):
            raise _validation_error(
                "generalized_gauss_sum_field",
                "generalized Gauss sum must use the canonical character/additive cyclotomic parent",
            )
        return self


class DirichletCharacterConductorRequest(StrictModel):
    character: DirichletCharacter


class DirichletCharacterTableResult(StrictModel):
    character: DirichletCharacter
    residues: tuple[StrictInt, ...]
    values: tuple[CyclotomicValue | None, ...]


class DirichletCharacterConductorResult(StrictModel):
    """Primitive ancestor and exact conductor of a source character."""

    character: DirichletCharacter
    conductor: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    primitive_character: DirichletCharacter

    @model_validator(mode="after")
    def require_divisor_of_source_modulus(self) -> Self:
        if self.character.group.modulus % self.conductor:
            raise _validation_error(
                "conductor_not_divisor",
                "character conductor must divide the source modulus",
            )
        if self.primitive_character.group.modulus != self.conductor:
            raise _validation_error(
                "primitive_character_modulus_mismatch",
                "primitive character must be represented modulo its conductor",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        character: DirichletCharacter,
        conductor: int,
        primitive_character: DirichletCharacter,
    ) -> Self:
        return cls.model_construct(
            character=character,
            conductor=conductor,
            primitive_character=primitive_character,
        )


class PrincipalDirichletCharacterRequest(StrictModel):
    """Materialize the complete principal-character table for one modulus."""

    modulus: StrictInt = Field(
        ge=1,
        le=MAX_PRINCIPAL_CHARACTER_MODULUS,
        description=(
            "Positive modulus; the complete residue table has exactly this many "
            "entries and is bounded to 2,048 entries."
        ),
    )


class CharacterGroupRequest(StrictModel):
    """Compute the finite unit-group decomposition for one modulus."""

    modulus: StrictInt = Field(
        ge=1,
        le=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Positive modulus; the unit table and generator-coordinate maps "
            "cover exactly phi(modulus) unit residues within the 2,048-entry "
            "bound."
        ),
    )


class DirichletCharacterEnumerationRequest(StrictModel):
    """Enumerate the complete dual character family of one exact group."""

    group: DirichletCharacterGroup


class DirichletCharacterFourierMatrixRequest(StrictModel):
    """Build the complete exact character matrix on one unit group."""

    group: DirichletCharacterGroup


class DirichletCharacterResidueIndicatorRequest(StrictModel):
    """Expand the indicator of one canonical unit residue in the character basis."""

    group: DirichletCharacterGroup
    residue: StrictInt = Field(ge=0, le=MAX_CHARACTER_GROUP_MODULUS - 1)


class DirichletCharacterResidueIndicatorExpansion(StrictModel):
    """Fourier expansion of one unit point indicator on the finite unit group.

    Coefficient at character ``chi`` is ``zeta_E**coefficient_exponents[i] / h``,
    where ``h=phi(N)`` and ``E`` is the group's common cyclotomic order.
    """

    group: DirichletCharacterGroup
    residue: StrictInt = Field(ge=0, le=MAX_CHARACTER_GROUP_MODULUS - 1)
    character_coordinates: tuple[tuple[StrictInt, ...], ...]
    coefficient_exponents: tuple[StrictInt, ...]
    cyclotomic_order: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    denominator: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)

    @model_validator(mode="after")
    def require_canonical_character_axis(self) -> Self:
        if (
            self.cyclotomic_order != self.group.exponent
            or self.denominator != self.group.character_count
            or len(self.character_coordinates) != self.denominator
            or len(self.coefficient_exponents) != self.denominator
            or self.character_coordinates
            != tuple(sorted(set(self.character_coordinates)))
            or any(
                len(row) != len(self.group.generator_orders)
                or any(
                    coordinate < 0 or coordinate >= order
                    for coordinate, order in zip(
                        row, self.group.generator_orders, strict=False
                    )
                )
                for row in self.character_coordinates
            )
            or any(
                not 0 <= e < self.cyclotomic_order for e in self.coefficient_exponents
            )
        ):
            raise _validation_error(
                "residue_indicator_shape",
                "indicator expansion must use the complete canonical dual axis and coefficient parent",
            )
        return self


class PrincipalDirichletCharacterValueRequest(StrictModel):
    """Evaluate a canonical principal-character value at one exact integer."""

    character: PrincipalDirichletCharacter
    integer: DirichletCharacterInteger = Field(
        description="Canonical base-10 integer syntax, reduced modulo character.modulus."
    )

    def integer_value(self) -> int:
        return int(self.integer)


class PrincipalDirichletCharacterValueResult(StrictModel):
    """A source-bound principal-character evaluation with canonical residue data."""

    character: PrincipalDirichletCharacter
    integer: DirichletCharacterInteger
    canonical_residue: StrictInt = Field(ge=0, lt=MAX_PRINCIPAL_CHARACTER_MODULUS)
    is_unit: StrictBool
    value: Literal[0, 1]

    @model_validator(mode="after")
    def require_source_bound_residue(self) -> Self:
        """Validate only the inexpensive source-to-residue structural binding."""

        residue = int(self.integer) % self.character.modulus
        if self.canonical_residue != residue:
            raise _validation_error(
                "canonical_residue_mismatch",
                "canonical residue does not match the source integer",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        character: PrincipalDirichletCharacter,
        integer: DirichletCharacterInteger,
        canonical_residue: StrictInt,
        is_unit: StrictBool,
        value: Literal[0, 1],
    ) -> Self:
        """Build a result after its producer has established the evaluation."""

        return cls.model_construct(
            character=character,
            integer=integer,
            canonical_residue=canonical_residue,
            is_unit=is_unit,
            value=value,
        )


__all__ = [
    "MAX_GENERALIZED_BERNOULLI_INDEX",
    "MAX_INTEGER_DIGITS",
    "CharacterGroupRequest",
    "DirichletCharacterConductorRequest",
    "DirichletCharacterConductorResult",
    "DirichletCharacterConjugateRequest",
    "DirichletCharacterEnumerationRequest",
    "DirichletCharacterGaussSumRequest",
    "DirichletCharacterGaussSumResult",
    "DirichletCharacterGeneralizedBernoulliRequest",
    "DirichletCharacterGeneralizedBernoulliResult",
    "DirichletCharacterGeneralizedGaussSumRequest",
    "DirichletCharacterGeneralizedGaussSumResult",
    "DirichletCharacterInteger",
    "DirichletCharacterInverseRequest",
    "DirichletCharacterJacobiSumRequest",
    "DirichletCharacterJacobiSumResult",
    "DirichletCharacterKernelRequest",
    "DirichletCharacterOrderRequest",
    "DirichletCharacterOrderResult",
    "DirichletCharacterOrthogonalityRequest",
    "DirichletCharacterOrthogonalityResult",
    "DirichletCharacterParityRequest",
    "DirichletCharacterParityResult",
    "DirichletCharacterPowerRequest",
    "DirichletCharacterPrimitiveGaussNormRequest",
    "DirichletCharacterPrimitiveGaussNormResult",
    "DirichletCharacterProductRequest",
    "DirichletCharacterRequest",
    "DirichletCharacterRestrictionRequest",
    "DirichletCharacterSequenceTwistRequest",
    "DirichletCharacterTableResult",
    "DirichletCharacterValueRequest",
    "DirichletCharacterValueResult",
    "PrincipalDirichletCharacterRequest",
    "PrincipalDirichletCharacterValueRequest",
    "PrincipalDirichletCharacterValueResult",
]
