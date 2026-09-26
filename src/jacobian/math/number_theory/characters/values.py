"""Canonical exact values for bounded Dirichlet-character operations."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_PRINCIPAL_CHARACTER_MODULUS = 2_048
MAX_CHARACTER_GROUP_MODULUS = 2_048


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by Dirichlet-character values."""

    return PydanticCustomError(f"dirichlet_character.{reason}", message)


class CyclotomicValue(StrictModel):
    """An exact root of unity in the explicitly declared cyclotomic parent."""

    order: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    exponent: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def canonical_exponent(self) -> Self:
        if self.exponent >= self.order:
            raise _validation_error(
                "cyclotomic_exponent",
                "root exponent must be reduced modulo the cyclotomic order",
            )
        return self

    def multiply(self, other: CyclotomicValue) -> CyclotomicValue:
        if self.order != other.order:
            raise ValueError("cyclotomic parents differ")
        return CyclotomicValue(
            order=self.order, exponent=(self.exponent + other.exponent) % self.order
        )

    def conjugate(self) -> CyclotomicValue:
        return CyclotomicValue(order=self.order, exponent=(-self.exponent) % self.order)


class DirichletCharacter(StrictModel):
    """One exact character in a concrete finite unit-group coordinate system."""

    group: DirichletCharacterGroup
    coordinates: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def require_coordinate_shape(self) -> Self:
        if len(self.coordinates) != len(self.group.generator_orders) or any(
            c < 0 or c >= o
            for c, o in zip(self.coordinates, self.group.generator_orders, strict=True)
        ):
            raise _validation_error(
                "character_coordinates",
                "character coordinates must match the group's dual coordinate axes",
            )
        return self


class PrimitiveDirichletCharacter(StrictModel):
    """A character claimed primitive, bound to its proposed exact conductor.

    Construction records the mathematical claim and its modulus. Consumers
    relying on primitivity establish that claim at their operation boundary.
    """

    character: DirichletCharacter
    conductor: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)

    @model_validator(mode="after")
    def require_modulus_matches_conductor(self) -> Self:
        if self.character.group.modulus != self.conductor:
            raise _validation_error(
                "primitive_character_modulus_mismatch",
                "primitive character modulus must equal its claimed conductor",
            )
        return self


class DirichletCharacterKernel(StrictModel):
    """Complete kernel of one character as canonical unit residues."""

    character: DirichletCharacter
    residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    index: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)

    @model_validator(mode="after")
    def require_canonical_kernel_shape(self) -> Self:
        group = self.character.group
        if self.residues != tuple(sorted(set(self.residues))) or not set(
            self.residues
        ).issubset(group.unit_residues):
            raise _validation_error(
                "kernel_residue_shape",
                "kernel residues must be increasing distinct canonical units",
            )
        if self.index * len(self.residues) != group.character_count:
            raise _validation_error(
                "kernel_index_mismatch",
                "kernel index must equal the character image size",
            )
        return self


class DirichletCharacterInflation(StrictModel):
    """An inflated target character with its source and unit reduction map."""

    source: DirichletCharacter
    target: DirichletCharacter
    target_unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    source_unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )

    @model_validator(mode="after")
    def require_canonical_reduction_shape(self) -> Self:
        source_modulus = self.source.group.modulus
        target_group = self.target.group
        if target_group.modulus % source_modulus:
            raise _validation_error(
                "inflation_modulus_not_multiple",
                "target modulus must be a multiple of the source modulus",
            )
        if self.target_unit_residues != target_group.unit_residues:
            raise _validation_error(
                "inflation_target_units_mismatch",
                "target residues must list every target unit in canonical order",
            )
        expected_source_residues = tuple(
            residue % source_modulus for residue in self.target_unit_residues
        )
        if self.source_unit_residues != expected_source_residues or any(
            residue not in self.source.group.unit_residues
            for residue in self.source_unit_residues
        ):
            raise _validation_error(
                "inflation_reduction_mismatch",
                "source residues must be the canonical reductions of target units",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: DirichletCharacter,
        target: DirichletCharacter,
        target_unit_residues: tuple[int, ...],
        source_unit_residues: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            source=source,
            target=target,
            target_unit_residues=target_unit_residues,
            source_unit_residues=source_unit_residues,
        )


class DirichletCharacterRestrictionObstruction(StrictModel):
    """Two units in one reduction fiber with different character values."""

    target_group: DirichletCharacterGroup
    target_unit_residue: StrictInt = Field(ge=0)
    source_unit_residues: tuple[StrictInt, StrictInt]
    values: tuple[CyclotomicValue, CyclotomicValue]

    @model_validator(mode="after")
    def require_fiber_witness(self) -> Self:
        if self.target_unit_residue not in self.target_group.unit_residues:
            raise _validation_error(
                "restriction_obstruction_target_residue",
                "obstruction residue must be a canonical target unit",
            )
        first, second = self.source_unit_residues
        if (
            first == second
            or first % self.target_group.modulus != self.target_unit_residue
            or second % self.target_group.modulus != self.target_unit_residue
            or self.values[0] == self.values[1]
        ):
            raise _validation_error(
                "restriction_obstruction_invalid",
                "obstruction needs distinct source units in one fiber with distinct values",
            )
        if self.values[0].order != self.values[1].order:
            raise _validation_error(
                "restriction_obstruction_parent_mismatch",
                "obstruction values must use one exact cyclotomic parent",
            )
        return self


class DirichletCharacterRestrictionResult(StrictModel):
    """Exact factor-down result or an explicit same-fiber obstruction."""

    status: Literal["descended", "does_not_factor"]
    source: DirichletCharacter
    target_modulus: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    target: DirichletCharacter | None = None
    target_unit_residues: tuple[StrictInt, ...] | None = None
    source_unit_lifts: tuple[StrictInt, ...] | None = None
    obstruction: DirichletCharacterRestrictionObstruction | None = None

    @model_validator(mode="after")
    def require_exact_outcome_shape(self) -> Self:
        source_modulus = self.source.group.modulus
        if source_modulus % self.target_modulus:
            raise _validation_error(
                "restriction_modulus_not_divisor",
                "target modulus must divide the source modulus",
            )
        if self.status == "descended":
            if (
                self.target is None
                or self.target.group.modulus != self.target_modulus
                or self.target_unit_residues != self.target.group.unit_residues
                or self.source_unit_lifts is None
                or self.obstruction is not None
                or len(self.source_unit_lifts) != len(self.target_unit_residues)
                or any(
                    lift not in self.source.group.unit_residues
                    or lift % self.target_modulus != target_residue
                    for target_residue, lift in zip(
                        self.target_unit_residues,
                        self.source_unit_lifts,
                        strict=True,
                    )
                )
            ):
                raise _validation_error(
                    "restriction_success_shape",
                    "descended result needs a target character and complete aligned unit lifts",
                )
        elif (
            self.target is not None
            or self.target_unit_residues is not None
            or self.source_unit_lifts is not None
            or self.obstruction is None
            or self.obstruction.target_group.modulus != self.target_modulus
            or any(
                residue not in self.source.group.unit_residues
                for residue in self.obstruction.source_unit_residues
            )
            or any(
                value.order != self.source.group.exponent
                for value in self.obstruction.values
            )
        ):
            raise _validation_error(
                "restriction_obstruction_shape",
                "nonfactor result must carry only a matching obstruction witness",
            )
        return self

    @classmethod
    def _descended_from_kernel(
        cls,
        *,
        source: DirichletCharacter,
        target_modulus: int,
        target: DirichletCharacter,
        target_unit_residues: tuple[int, ...],
        source_unit_lifts: tuple[int, ...],
    ) -> Self:
        return cls.model_construct(
            status="descended",
            source=source,
            target_modulus=target_modulus,
            target=target,
            target_unit_residues=target_unit_residues,
            source_unit_lifts=source_unit_lifts,
            obstruction=None,
        )

    @classmethod
    def _obstructed_from_kernel(
        cls,
        *,
        source: DirichletCharacter,
        target_modulus: int,
        obstruction: DirichletCharacterRestrictionObstruction,
    ) -> Self:
        return cls.model_construct(
            status="does_not_factor",
            source=source,
            target_modulus=target_modulus,
            target=None,
            target_unit_residues=None,
            source_unit_lifts=None,
            obstruction=obstruction,
        )


class PrincipalDirichletCharacter(StrictModel):
    """The extension-by-zero principal character modulo one fixed modulus.

    ``values[a]`` is the exact value of the principal character at the
    canonical residue ``a``.  The unit residues and complete table bind this
    value to its modulus without relying on a backend-specific group basis.
    """

    modulus: StrictInt = Field(ge=1, le=MAX_PRINCIPAL_CHARACTER_MODULUS)
    unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_PRINCIPAL_CHARACTER_MODULUS
    )
    values: tuple[Literal[0, 1], ...] = Field(
        min_length=1, max_length=MAX_PRINCIPAL_CHARACTER_MODULUS
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        """Validate the bounded wire shape without proving the character."""

        if len(self.unit_residues) > self.modulus:
            raise _validation_error(
                "unit_residues_length",
                "unit residues cannot contain more entries than the modulus",
            )
        if any(
            residue < 0 or residue >= self.modulus for residue in self.unit_residues
        ):
            raise _validation_error(
                "unit_residue_range",
                "unit residues must be distinct canonical residues modulo modulus",
            )
        if self.unit_residues != tuple(sorted(set(self.unit_residues))):
            raise _validation_error(
                "unit_residue_order",
                "unit residues must be strictly increasing canonical residues",
            )
        if len(self.values) != self.modulus:
            raise _validation_error(
                "values_table_length",
                "values must contain exactly one entry for every canonical residue",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        unit_residues: tuple[StrictInt, ...],
        values: tuple[Literal[0, 1], ...],
    ) -> Self:
        """Build a character after its producer has established its table."""

        return cls.model_construct(
            modulus=modulus,
            unit_residues=unit_residues,
            values=values,
        )


class DirichletCharacterGroup(StrictModel):
    """The finite unit group modulo one fixed modulus with character coordinates.

    ``unit_residues`` lists every unit in increasing canonical order.
    ``generators`` is a tuple of canonical unit residues whose orders are
    ``generator_orders``; every unit's ``unit_coordinates`` row holds the
    discrete logarithms of that unit against those generators, in generator
    order. ``invariant_factors`` is the divisibility chain of the finite
    Abelian unit group. ``exponent`` is the common root-of-unity order of the
    dual character group, and ``character_count`` equals ``phi(modulus)``.
    """

    modulus: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    character_count: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    invariant_factors: tuple[Annotated[StrictInt, Field(ge=2)], ...] = Field(
        max_length=32,
        description=(
            "Divisibility chain of the finite Abelian unit group; empty exactly "
            "for the trivial group."
        ),
    )
    generators: tuple[StrictInt, ...] = Field(
        max_length=32,
        description="Canonical unit residues generating the unit group.",
    )
    generator_orders: tuple[StrictInt, ...] = Field(max_length=32)
    unit_coordinates: tuple[tuple[StrictInt, ...], ...] = Field(
        max_length=MAX_CHARACTER_GROUP_MODULUS,
        description=(
            "Discrete-logarithm rows aligned with unit_residues, in generator "
            "order; each entry is below the corresponding generator order."
        ),
    )
    exponent: StrictInt = Field(
        ge=1,
        description="Common root-of-unity order of the dual character group.",
    )

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:  # noqa: C901
        """Validate the bounded wire shape without proving the group."""

        if len(self.unit_residues) > self.modulus:
            raise _validation_error(
                "unit_residues_length",
                "unit residues cannot contain more entries than the modulus",
            )
        if any(
            residue < 0 or residue >= self.modulus for residue in self.unit_residues
        ):
            raise _validation_error(
                "unit_residue_range",
                "unit residues must be distinct canonical residues modulo modulus",
            )
        if self.unit_residues != tuple(sorted(set(self.unit_residues))):
            raise _validation_error(
                "unit_residue_order",
                "unit residues must be strictly increasing canonical residues",
            )
        if self.character_count != len(self.unit_residues):
            raise _validation_error(
                "character_count_mismatch",
                "character count must equal the number of unit residues",
            )
        if len(self.generators) != len(self.generator_orders):
            raise _validation_error(
                "generator_order_length",
                "generators and generator orders must align",
            )
        if any(
            generator < 0 or generator >= self.modulus for generator in self.generators
        ):
            raise _validation_error(
                "generator_range",
                "generators must be canonical residues modulo modulus",
            )
        if any(order < 1 for order in self.generator_orders):
            raise _validation_error(
                "generator_order_range",
                "generator orders must be positive",
            )
        if len(self.unit_coordinates) != len(self.unit_residues):
            raise _validation_error(
                "coordinate_row_count",
                "coordinate rows must align with unit residues",
            )
        for row in self.unit_coordinates:
            if len(row) != len(self.generators):
                raise _validation_error(
                    "coordinate_row_length",
                    "coordinate rows must align with generators",
                )
            for coordinate, order in zip(row, self.generator_orders, strict=True):
                if coordinate < 0 or coordinate >= order:
                    raise _validation_error(
                        "coordinate_range",
                        "coordinates must lie below their generator order",
                    )
        if self.invariant_factors != tuple(sorted(self.invariant_factors)):
            raise _validation_error(
                "invariant_factor_order",
                "invariant factors must be sorted increasingly",
            )
        for first, second in zip(
            self.invariant_factors, self.invariant_factors[1:], strict=False
        ):
            if second % first != 0:
                raise _validation_error(
                    "invariant_factor_divisibility",
                    "each invariant factor must divide the next",
                )
        if self.exponent < 1:
            raise _validation_error(
                "exponent_range",
                "the common root-of-unity exponent must be positive",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        modulus: int,
        unit_residues: tuple[int, ...],
        character_count: int,
        invariant_factors: tuple[int, ...],
        generators: tuple[int, ...],
        generator_orders: tuple[int, ...],
        unit_coordinates: tuple[tuple[int, ...], ...],
        exponent: int,
    ) -> Self:
        """Build a group after its producer has established the decomposition."""

        return cls.model_construct(
            modulus=modulus,
            unit_residues=unit_residues,
            character_count=character_count,
            invariant_factors=invariant_factors,
            generators=generators,
            generator_orders=generator_orders,
            unit_coordinates=unit_coordinates,
            exponent=exponent,
        )


class DirichletCharacterFamily(StrictModel):
    """The complete dual of one finite unit-group parent.

    ``coordinates`` lists each character once in lexicographic order on the
    group's canonical cyclic dual axes. The parent is serialized once, so this
    carrier remains bounded when a family is large.
    """

    group: DirichletCharacterGroup
    coordinates: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1,
        max_length=MAX_CHARACTER_GROUP_MODULUS,
    )

    @model_validator(mode="after")
    def require_complete_dual_coordinates(self) -> Self:
        if not isinstance(self.group, DirichletCharacterGroup):
            raise _validation_error(
                "family_parent", "family needs a canonical character group"
            )
        orders = self.group.generator_orders
        if type(self.group.character_count) is not int or type(orders) is not tuple:
            raise _validation_error(
                "family_parent", "family parent must have canonical finite dual axes"
            )
        if any(type(order) is not int or order < 1 for order in orders):
            raise _validation_error(
                "family_axis_order", "dual axes must have positive integer orders"
            )
        expected_count = 1
        for order in orders:
            if order > MAX_CHARACTER_GROUP_MODULUS // expected_count:
                raise _validation_error(
                    "family_axis_order",
                    "dual axes exceed the admitted character-family size",
                )
            expected_count *= order
        if (
            expected_count != self.group.character_count
            or len(self.coordinates) != expected_count
        ):
            raise _validation_error(
                "family_size",
                "the family must contain exactly one row per dual coordinate",
            )
        if any(
            len(row) != len(orders)
            or any(
                coordinate < 0 or coordinate >= order
                for coordinate, order in zip(row, orders, strict=False)
            )
            for row in self.coordinates
        ):
            raise _validation_error(
                "family_coordinate",
                "every character coordinate must lie on the declared dual axes",
            )
        if self.coordinates != tuple(sorted(set(self.coordinates))):
            raise _validation_error(
                "family_order",
                "dual coordinates must enumerate every character once in lexicographic order",
            )
        return self


class DirichletCharacterFourierMatrix(StrictModel):
    """Exact complete character table on the canonical unit residues.

    Entries are exponents in the shared cyclotomic parent: entry ``e`` means
    ``zeta_order**e``. Rows follow lexicographic dual coordinates and columns
    follow the group's increasing canonical unit residues.
    """

    group: DirichletCharacterGroup
    character_coordinates: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    unit_residues: tuple[StrictInt, ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )
    cyclotomic_order: StrictInt = Field(ge=1, le=MAX_CHARACTER_GROUP_MODULUS)
    entries: tuple[tuple[StrictInt, ...], ...] = Field(
        min_length=1, max_length=MAX_CHARACTER_GROUP_MODULUS
    )

    @model_validator(mode="after")
    def require_canonical_axes(self) -> Self:
        if self.cyclotomic_order != self.group.exponent:
            raise _validation_error(
                "fourier_parent", "matrix cyclotomic order must match its group"
            )
        if self.unit_residues != self.group.unit_residues:
            raise _validation_error(
                "fourier_unit_axis",
                "matrix columns must be the canonical unit residues",
            )
        rank = len(self.group.generator_orders)
        count = self.group.character_count
        if len(self.character_coordinates) != count or len(self.entries) != count:
            raise _validation_error(
                "fourier_row_count", "matrix needs one row per character"
            )
        if self.character_coordinates != tuple(sorted(set(self.character_coordinates))):
            raise _validation_error(
                "fourier_character_axis",
                "character rows must be unique and lexicographically ordered",
            )
        if any(
            len(row) != rank
            or any(
                c < 0 or c >= order
                for c, order in zip(row, self.group.generator_orders, strict=True)
            )
            for row in self.character_coordinates
        ):
            raise _validation_error(
                "fourier_character_axis",
                "character coordinates must lie on the declared dual axes",
            )
        if any(
            len(row) != count
            or any(value < 0 or value >= self.cyclotomic_order for value in row)
            for row in self.entries
        ):
            raise _validation_error(
                "fourier_entry_shape",
                "entries must be reduced cyclotomic exponents on the declared axes",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        group: DirichletCharacterGroup,
        character_coordinates: tuple[tuple[int, ...], ...],
        unit_residues: tuple[int, ...],
        cyclotomic_order: int,
        entries: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            group=group,
            character_coordinates=character_coordinates,
            unit_residues=unit_residues,
            cyclotomic_order=cyclotomic_order,
            entries=entries,
        )


__all__ = [
    "MAX_CHARACTER_GROUP_MODULUS",
    "MAX_PRINCIPAL_CHARACTER_MODULUS",
    "CyclotomicValue",
    "DirichletCharacter",
    "DirichletCharacterFamily",
    "DirichletCharacterGroup",
    "DirichletCharacterInflation",
    "DirichletCharacterKernel",
    "DirichletCharacterRestrictionObstruction",
    "DirichletCharacterRestrictionResult",
    "PrincipalDirichletCharacter",
]
