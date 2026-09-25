"""Character-valued V maps preserve exact inflated Gamma0 targets."""

from __future__ import annotations

import pytest
from cypari import pari
from pydantic import TypeAdapter

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.cyclic_linear._models import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
)
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms._pari_basis_worker import (
    _as_cyclotomic_coordinates,
    _character_vector,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    CHARACTER_BASIS_ID,
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_degeneracy import (
    modular_character_coordinates_v_degeneracy,
)
from jacobian.math.number_theory.modular_forms.character_degeneracy_models import (
    ModularCharacterVDegeneracyRequest,
)
from jacobian.math.number_theory.modular_forms.pari_basis import (
    _pari_character_request,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormFieldQExpansion,
    ModularFormSpace,
)


def _space(coordinate: int = 2) -> ModularFormSpace:
    character = dirichlet_character(character_group(13), (coordinate,))
    return ModularFormSpace(
        level=13,
        weight=2,
        kind="S",
        character=character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )


def _target_space(coordinate: int, d: int) -> ModularFormSpace:
    source_character = _space(coordinate).character
    target_level = 13 * d
    target_group = character_group(target_level)
    source_group = source_character.group
    source_rows = dict(
        zip(source_group.unit_residues, source_group.unit_coordinates, strict=True)
    )
    target_coordinates = []
    for generator, axis_order in zip(
        target_group.generators, target_group.generator_orders, strict=True
    ):
        source_row = source_rows[generator % source_group.modulus]
        source_value_exponent = (
            sum(
                coordinate
                * (source_group.exponent // source_axis_order)
                * unit_coordinate
                for coordinate, source_axis_order, unit_coordinate in zip(
                    source_character.coordinates,
                    source_group.generator_orders,
                    source_row,
                    strict=True,
                )
            )
            % source_group.exponent
        )
        value_exponent = source_value_exponent * (
            target_group.exponent // source_group.exponent
        )
        axis_step = target_group.exponent // axis_order
        assert value_exponent % axis_step == 0
        target_coordinates.append((value_exponent // axis_step) % axis_order)
    target_character = dirichlet_character(target_group, tuple(target_coordinates))
    return ModularFormSpace(
        level=target_level,
        weight=2,
        kind="S",
        character=target_character,
        coefficient_domain=RationalCyclotomicField(order=6),
    )


def _form(coordinate: int = 2) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=_space(coordinate),
        basis_id=CHARACTER_BASIS_ID,
        coordinates=(
            RationalCyclotomicElement(
                field=RationalCyclotomicField(order=6),
                coefficients_ascending=(
                    {"num": 1, "den": 1},
                    {"num": 0, "den": 1},
                ),
            ),
        ),
    )


def _as_element(value: object) -> RationalCyclotomicElement:
    coordinates = _as_cyclotomic_coordinates(pari, value, 6)
    return RationalCyclotomicElement(
        field=RationalCyclotomicField(order=6),
        coefficients_ascending=tuple(
            {"num": int(numerator), "den": int(denominator)}
            for numerator, denominator in coordinates
        ),
    )


@pytest.mark.parametrize("coordinate", (2, 10))
@pytest.mark.parametrize("d", (2, 3))
def test_v_degeneracy_matches_pari_expanding_operator_and_inflates_target(
    coordinate: int, d: int
) -> None:
    form = _form(coordinate)
    target = _target_space(coordinate, d)
    request = ModularCharacterVDegeneracyRequest(form=form, target_space=target)
    result = modular_character_coordinates_v_degeneracy(request)

    assert result.space == target
    assert len(result.coefficients) == 2 * d + 1

    # Ask PARI for its independently implemented B_d(f)=f(q^d) operator. The
    # source basis is scaled to Jacobian's q-RREF normalization by a_1.
    source = _space(coordinate)
    character_request = _pari_character_request(source)
    pari_group, pari_character = _character_vector(
        pari, character_request["character"], 6
    )
    pari_space = pari.mfinit([13, 2, [pari_group, pari_character]], 1)
    pari_form = pari.mfbasis(pari_space)[0]
    jacobian_basis = modular_character_basis_q_expansions(source).elements[0]
    pari_a1 = _as_element(pari.mfcoef(pari_form, 1))
    scale = jacobian_basis.expansion.coefficients[1]
    scale = cyclotomic.divide(scale, pari_a1)
    pari_image = pari.mfbd(pari_form, d)
    oracle = tuple(
        cyclotomic.multiply(scale, _as_element(pari.mfcoef(pari_image, index)))
        for index in range(2 * d + 1)
    )
    assert result.coefficients == oracle

    restored = TypeAdapter(ModularFormFieldQExpansion).validate_json(
        result.model_dump_json()
    )
    assert restored == result


def test_character_v_degeneracy_is_discoverable_and_rejects_wrong_target() -> None:
    operation = Catalog.open().operation(
        "modular_form.character_coordinates.v_degeneracy.apply"
    )
    assert operation is not None
    with pytest.raises(OperationDomainValidationError):
        modular_character_coordinates_v_degeneracy(
            ModularCharacterVDegeneracyRequest(form=_form(), target_space=_space())
        )
