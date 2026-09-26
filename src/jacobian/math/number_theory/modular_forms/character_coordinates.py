"""Exact coordinate admission for bounded cyclotomic character spaces."""

from __future__ import annotations

from typing import NamedTuple

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.matrices.cyclic_linear._models import (
    MAX_CYCLIC_FIELD_ELEMENT_DIGITS,
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.values import (
    DirichletCharacter,
    DirichletCharacterGroup,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_dimensions import (
    character_space_dimensions,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

CHARACTER_RREF_BASIS_ID = "gamma0-cyclotomic-character-sturm-rref-v1"
LEGACY_CHARACTER_BASIS_ID = "gamma0-13-even-order6-character-sturm-v1"
MAX_CHARACTER_COORDINATE_DIMENSION = 32
MAX_CHARACTER_COORDINATE_EQUALITY_WORK = 1_000_000


def _bounded_int_tuple(
    values: object, *, maximum_length: int, maximum_value: int
) -> bool:
    return (
        type(values) is tuple
        and len(values) <= maximum_length
        and all(type(value) is int and 0 <= value <= maximum_value for value in values)
    )


def _admit_space_work(space: ModularFormSpace) -> None:
    """Bound caller-supplied group scans and coordinate comparisons up front."""
    level = space.level
    if type(level) is not int or level not in (13, 26, 39):
        raise OperationDomainValidationError(
            location=("space", "level"),
            code="modular_form.character_coordinates_space_level",
            message="character coordinates support levels 13, 26, and 39",
        )
    character = space.character
    if type(character) is not DirichletCharacter:
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.character_coordinates_character",
            message="character coordinates require an explicit Dirichlet character",
        )
    group = character.group
    if type(group) is not DirichletCharacterGroup or group.modulus != level:
        raise OperationDomainValidationError(
            location=("space", "character", "group"),
            code="modular_form.character_coordinates_character_group",
            message="the exact character group modulus must equal the space level",
        )
    if (
        type(group.character_count) is not int
        or not 0 <= group.character_count <= level
        or type(group.exponent) is not int
        or not 0 <= group.exponent <= level
        or not _bounded_int_tuple(
            group.unit_residues, maximum_length=level, maximum_value=level
        )
        or not _bounded_int_tuple(
            group.generator_orders, maximum_length=32, maximum_value=level
        )
        or not _bounded_int_tuple(
            group.generators, maximum_length=32, maximum_value=level
        )
        or not _bounded_int_tuple(
            group.invariant_factors, maximum_length=32, maximum_value=level
        )
        or type(group.unit_coordinates) is not tuple
        or len(group.unit_coordinates) > level
        or any(
            not _bounded_int_tuple(row, maximum_length=32, maximum_value=level)
            for row in group.unit_coordinates
        )
        or not _bounded_int_tuple(
            character.coordinates, maximum_length=32, maximum_value=level
        )
    ):
        raise OperationDomainValidationError(
            location=("space", "character"),
            code="modular_form.character_coordinates_character_shape",
            message="caller-supplied character coordinates exceed their exact bounded shape",
        )
    # Character-group reconstruction and Cohen--Oesterle sums are O(N^2)
    # on this fixed level family. Vector equality compares at most 32 elements
    # in the degree-two field, with the canonical element digit cap applied.
    admitted_work = (
        64 * level * level
        + 2 * MAX_CHARACTER_COORDINATE_DIMENSION * 2 * MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    )
    if admitted_work > MAX_CHARACTER_COORDINATE_EQUALITY_WORK:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_coordinates_equality_work",
            message="character coordinate admission exceeds its exact work envelope",
        )


class _CoordinateSpace(NamedTuple):
    space: ModularFormSpace
    field: RationalCyclotomicField
    dimension: int
    basis_id: str


def _admit_coordinate_space(space: object) -> _CoordinateSpace:
    if type(space) is not ModularFormSpace:
        raise OperationDomainValidationError(
            location=("space",),
            code="modular_form.character_coordinates_space_type",
            message="character coordinates require a canonical modular-form space",
        )
    _admit_space_work(space)
    # Equality admission has already checked the complete native space shape
    # above. Do not replay the level-13 PARI character admission here; the
    # generic coordinate contract uses the canonical field carried by space.
    field = space.coefficient_domain
    character = space.character
    if type(character) is not DirichletCharacter:
        raise OperationDomainValidationError(
            location=("form", "space", "character"),
            code="modular_form.character_coordinates_character",
            message="character coordinates require an explicit Dirichlet character",
        )
    cusp_dimension, full_dimension = character_space_dimensions(
        space.level, space.weight, character, field
    )
    dimension = cusp_dimension if space.kind == "S" else full_dimension
    legacy_parent = (
        space.level == 13
        and space.kind == "S"
        and character.coordinates in ((2,), (10,))
    )
    expected_basis_id = (
        LEGACY_CHARACTER_BASIS_ID if legacy_parent else CHARACTER_RREF_BASIS_ID
    )
    if dimension > MAX_CHARACTER_COORDINATE_DIMENSION:
        raise OperationResourceAdmissionError(
            location=("space",),
            code="modular_form.character_coordinates_dimension_bound",
            message="character coordinate dimension exceeds the admitted equality bound",
        )
    return _CoordinateSpace(space, field, dimension, expected_basis_id)


def _admit_coordinate_vector(
    form: ModularFormCoordinates,
    context: _CoordinateSpace,
    side: str,
) -> None:
    if type(form) is not ModularFormCoordinates:
        raise OperationDomainValidationError(
            location=(side,),
            code="modular_form.character_coordinates_type",
            message="character coordinates require a canonical ModularFormCoordinates value",
        )
    if type(form.space) is not ModularFormSpace or form.space != context.space:
        raise OperationDomainValidationError(
            location=(side, "space"),
            code="modular_form.character_coordinates_equality_parent",
            message="character coordinate equality requires the identical exact space",
        )
    if form.basis_id != context.basis_id:
        raise OperationDomainValidationError(
            location=(side, "basis_id"),
            code="modular_form.character_coordinates_basis",
            message="character coordinates must use the canonical basis identifier for their exact space",
        )
    if (
        type(form.coordinates) is not tuple
        or len(form.coordinates) != context.dimension
    ):
        raise OperationDomainValidationError(
            location=(side, "coordinates"),
            code="modular_form.character_coordinates_dimension",
            message="character coordinate count must equal the exact space dimension",
        )
    for index, value in enumerate(form.coordinates):
        if type(value) is not RationalCyclotomicElement or value.field != context.field:
            raise OperationDomainValidationError(
                location=(side, "coordinates", index),
                code="modular_form.character_coordinates_parent",
                message="each character coordinate must use the exact space coefficient field",
            )
        cyclotomic._validate_element(value)


def _generic_character_coordinates_equal(
    left: ModularFormCoordinates,
    right: ModularFormCoordinates,
) -> bool:
    """Compare coordinates in one exact canonical character-space basis.

    The q-Sturm RREF basis is unique for the represented space. Its producer
    checks the exact dimension and full Sturm rank; equality therefore reduces
    to exact coordinate equality after re-admitting both typed inputs. This
    handles the unique empty coordinate vector of a zero-dimensional space.
    """
    if (
        type(left) is not ModularFormCoordinates
        or type(right) is not ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.character_coordinates_type",
            message="character coordinate equality requires two canonical coordinate values",
        )
    if left.space != right.space:
        raise OperationDomainValidationError(
            location=("right", "space"),
            code="modular_form.character_coordinates_equality_parent",
            message="character coordinate equality requires the identical exact space",
        )
    context = _admit_coordinate_space(left.space)
    _admit_coordinate_vector(left, context, "left")
    _admit_coordinate_vector(right, context, "right")
    admitted_work = (
        2 * context.dimension * context.field.degree * MAX_CYCLIC_FIELD_ELEMENT_DIGITS
    )
    if admitted_work > MAX_CHARACTER_COORDINATE_EQUALITY_WORK:
        raise OperationResourceAdmissionError(
            location=(),
            code="modular_form.character_coordinates_equality_work",
            message="exact character-coordinate equality exceeds its work envelope",
        )
    return left.coordinates == right.coordinates


def modular_character_coordinates_equal(
    left: ModularFormCoordinates,
    right: ModularFormCoordinates,
) -> bool:
    """Dispatch to the generic vector contract or the established 1D slice."""
    if (
        type(left) is not ModularFormCoordinates
        or type(right) is not ModularFormCoordinates
    ):
        raise OperationDomainValidationError(
            location=(),
            code="modular_form.character_coordinates_type",
            message="character coordinate equality requires two canonical coordinate values",
        )
    if (
        left.basis_id == CHARACTER_RREF_BASIS_ID
        or right.basis_id == CHARACTER_RREF_BASIS_ID
    ):
        return _generic_character_coordinates_equal(left, right)
    from jacobian.math.number_theory.modular_forms.character_basis import (
        modular_character_coordinates_equal as legacy_character_coordinates_equal,
    )

    return legacy_character_coordinates_equal(left, right)


__all__ = ["modular_character_coordinates_equal"]
