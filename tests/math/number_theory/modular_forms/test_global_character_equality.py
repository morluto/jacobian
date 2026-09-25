"""Global equality across exact character spaces and field embeddings."""

from __future__ import annotations

from itertools import product
from math import gcd

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_coordinates_q_expansion,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
)
from jacobian.math.number_theory.modular_forms.global_equality.models import (
    CyclotomicFieldEmbedding,
)
from jacobian.math.number_theory.modular_forms.global_equality.operations import (
    _map_element,
    modular_form_coordinates_global_equal,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_SOURCE_FIELD = RationalCyclotomicField(order=6)
_TARGET_FIELD = RationalCyclotomicField(order=12)


def _inflated_character(level: int, coordinate: int):
    source = dirichlet_character(character_group(13), (coordinate,))
    target_group = character_group(level)
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        candidate = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source, residue).value
            for residue in range(level)
            if gcd(residue, level) == 1
        ):
            return candidate
    raise AssertionError("character inflation fixture was not found")


def _space(level: int, coordinate: int = 4, kind: str = "M", weight: int = 2):
    return ModularFormSpace(
        level=level,
        weight=weight,
        kind=kind,
        character=_inflated_character(level, coordinate),
        coefficient_domain=_SOURCE_FIELD,
    )


def _element(constant: int = 0, zeta: int = 0):
    return RationalCyclotomicElement(
        field=_SOURCE_FIELD,
        coefficients_ascending=(
            {"num": constant, "den": 1},
            {"num": zeta, "den": 1},
        ),
    )


def _form(space: ModularFormSpace, first_coordinate: int = 0):
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        _admit_coordinate_space,
    )

    context = _admit_coordinate_space(space)
    dimension = context.dimension
    coordinates = tuple(
        _element(int(index == 0) if first_coordinate else 0)
        for index in range(dimension)
    )
    return ModularFormCoordinates(
        space=space,
        basis_id=context.basis_id,
        coordinates=coordinates,
    )


def _basis_form(space: ModularFormSpace, basis_index: int):
    from jacobian.math.number_theory.modular_forms.character_coordinates import (
        _admit_coordinate_space,
    )

    context = _admit_coordinate_space(space)
    assert 0 <= basis_index < context.dimension
    return ModularFormCoordinates(
        space=space,
        basis_id=context.basis_id,
        coordinates=tuple(
            _element(int(index == basis_index)) for index in range(context.dimension)
        ),
    )


def _embedding():
    # The inclusion Q(zeta_6) -> Q(zeta_12) sends zeta_6 to zeta_12^2.
    image = RationalCyclotomicElement(
        field=_TARGET_FIELD,
        coefficients_ascending=tuple(
            {"num": int(index == 2), "den": 1} for index in range(4)
        ),
    )
    return CyclotomicFieldEmbedding(
        source_order=6, target_field=_TARGET_FIELD, generator_image=image
    )


def _conjugate_embedding():
    # zeta_6 maps to zeta_12^-2 = 1 - zeta_12^2.
    image = RationalCyclotomicElement(
        field=_TARGET_FIELD,
        coefficients_ascending=tuple(
            {"num": (1 if index == 0 else -1 if index == 2 else 0), "den": 1}
            for index in range(4)
        ),
    )
    return CyclotomicFieldEmbedding(
        source_order=6, target_field=_TARGET_FIELD, generator_image=image
    )


def test_global_equality_compares_nonzero_cross_embeddings():
    # Characters 2 and 10 are Galois conjugate. Mapping their generators by
    # opposite embeddings gives the same common Nebentypus and the same form.
    left = _basis_form(_space(13, 2, kind="S"), basis_index=0)
    right = _basis_form(_space(13, 10, kind="S"), basis_index=0)
    assert modular_form_coordinates_global_equal(
        left, _embedding(), right, _conjugate_embedding()
    )
    # Independently realize each source coordinate with the exact character
    # q-expansion API, then compare after applying the declared embeddings.
    left_q = modular_character_coordinates_q_expansion(left).coefficients
    right_q = modular_character_coordinates_q_expansion(right).coefficients
    left_image = _embedding().generator_image
    right_image = _conjugate_embedding().generator_image
    assert tuple(
        _map_element(value, left_image, _TARGET_FIELD) for value in left_q
    ) == tuple(_map_element(value, right_image, _TARGET_FIELD) for value in right_q)

    # A distinct nonzero scalar multiple is not equal to the first form.
    unequal_right = ModularFormCoordinates(
        space=right.space,
        basis_id=right.basis_id,
        coordinates=(_element(2),),
    )
    assert not modular_form_coordinates_global_equal(
        left, _embedding(), unequal_right, _conjugate_embedding()
    )
    unequal_q = modular_character_coordinates_q_expansion(unequal_right).coefficients
    assert tuple(
        _map_element(value, right_image, _TARGET_FIELD) for value in right_q
    ) != tuple(_map_element(value, right_image, _TARGET_FIELD) for value in unequal_q)

    # Distinct mapped characters cannot describe the same nonzero form.
    with pytest.raises(OperationDomainValidationError) as mismatch:
        modular_form_coordinates_global_equal(
            left, _embedding(), _basis_form(_space(13, 10, kind="S"), 0), _embedding()
        )
    assert mismatch.value.errors()[0]["type"] == (
        "modular_form.global_equality_character_mismatch"
    )

    with pytest.raises(OperationDomainValidationError) as same_map:
        modular_form_coordinates_global_equal(
            left, _embedding(), _basis_form(_space(13, 2, kind="S"), 0), _embedding()
        )
    assert same_map.value.errors()[0]["type"] == (
        "modular_form.global_equality_same_embedding"
    )

    wrong_weight = ModularFormCoordinates.model_construct(
        space=_space(13, 10, weight=4),
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_global_equal(
            left, _embedding(), wrong_weight, _conjugate_embedding()
        )
    assert error.value.errors()[0]["type"] == "modular_form.global_equality_weight"


def test_global_equality_handles_zero_spaces_and_unequal_levels():
    zero_left = _form(_space(13, 4, kind="S"))
    zero_right = _form(_space(13, 8, kind="S"))
    assert not zero_left.coordinates and not zero_right.coordinates
    assert modular_form_coordinates_global_equal(
        zero_left, _embedding(), zero_right, _conjugate_embedding()
    )

    # Same source map at both ends belongs to the adjacent same-character
    # transport/equality contract, including across unequal levels.
    lower = _form(_space(13, 2))
    upper = _form(_space(26, 2))
    with pytest.raises(OperationDomainValidationError) as same_map_error:
        modular_form_coordinates_global_equal(lower, _embedding(), upper, _embedding())
    assert (
        same_map_error.value.errors()[0]["type"]
        == "modular_form.global_equality_same_embedding"
    )


def test_global_equality_reaches_gamma1_78_sturm_boundary():
    # [SL2(Z):Gamma1(78)] = 4032, hence weight two needs 673 coefficients.
    # This exercises the high-precision worker lane and compares nonzero forms.
    left = _form(_space(26, 4), first_coordinate=1)
    right = _form(_space(39, 8), first_coordinate=1)
    assert not modular_form_coordinates_global_equal(
        left, _embedding(), right, _conjugate_embedding()
    )
