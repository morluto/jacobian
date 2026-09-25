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
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
)
from jacobian.math.number_theory.modular_forms.global_equality.models import (
    CyclotomicFieldEmbedding,
)
from jacobian.math.number_theory.modular_forms.global_equality.operations import (
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


def test_global_equality_compares_different_characters_and_rejects_weight_mismatch():
    left = _form(_space(13, 4), first_coordinate=1)
    right = _form(_space(13, 8), first_coordinate=1)
    embedding = _embedding()
    assert not modular_form_coordinates_global_equal(left, embedding, right, embedding)

    same_character_form = _form(_space(13, 4), first_coordinate=1)
    with pytest.raises(OperationDomainValidationError) as same_character_error:
        modular_form_coordinates_global_equal(
            same_character_form, embedding, same_character_form, embedding
        )
    assert (
        same_character_error.value.errors()[0]["type"]
        == "modular_form.global_equality_same_character"
    )
    wrong_weight = ModularFormCoordinates.model_construct(
        space=_space(13, 8, weight=4),
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(),
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_global_equal(left, embedding, wrong_weight, embedding)
    assert error.value.errors()[0]["type"] == "modular_form.global_equality_weight"


def test_global_equality_handles_zero_spaces_and_unequal_levels():
    embedding = _embedding()
    zero_left = _form(_space(13, 4, kind="S"))
    zero_right = _form(_space(13, 8, kind="S"))
    assert not zero_left.coordinates and not zero_right.coordinates
    assert modular_form_coordinates_global_equal(
        zero_left, embedding, zero_right, embedding
    )
    assert modular_form_coordinates_global_equal(
        zero_left, _conjugate_embedding(), zero_right, _conjugate_embedding()
    )

    # The explicit maps can change the transported characters independently.
    # Both positive-dimensional source forms here are zero.
    mapped_zero_left = _form(_space(13, 4))
    mapped_zero_right = _form(_space(13, 2))
    assert modular_form_coordinates_global_equal(
        mapped_zero_left, embedding, mapped_zero_right, _conjugate_embedding()
    )

    # Same Nebentypus at different levels belongs to the adjacent
    # same-character transport/equality contract, not this operation.
    lower = _form(_space(13, 4))
    upper = _form(_space(26, 4))
    with pytest.raises(OperationDomainValidationError) as same_character_error:
        modular_form_coordinates_global_equal(lower, embedding, upper, embedding)
    assert (
        same_character_error.value.errors()[0]["type"]
        == "modular_form.global_equality_same_character"
    )


def test_global_equality_reaches_gamma1_78_sturm_boundary():
    # [SL2(Z):Gamma1(78)] = 4032, hence weight two needs 673 coefficients.
    # This exercises the high-precision worker lane and compares nonzero forms.
    left = _form(_space(26, 4), first_coordinate=1)
    right = _form(_space(39, 8), first_coordinate=1)
    embedding = _embedding()
    assert not modular_form_coordinates_global_equal(left, embedding, right, embedding)
