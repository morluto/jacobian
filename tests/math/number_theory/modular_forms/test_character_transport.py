"""Exact, explicitly parented character inflation and global equality tests."""

from __future__ import annotations

from itertools import product
from math import gcd

import pytest
from pydantic import TypeAdapter

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    CyclotomicCharacterMap,
    CyclotomicIdentityFieldMap,
    ModularCharacterCoordinatesTransportRequest,
    ModularCharacterEqualityRequest,
    ModularCharacterSpaceInclusion,
    ModularCharacterTransportedForm,
)
from jacobian.math.number_theory.modular_forms.character_transport import (
    modular_character_coordinates_equal_in_common_space,
    modular_character_coordinates_transport,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_FIELD = RationalCyclotomicField(order=6)
_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"
_LEGACY_BASIS = "gamma0-13-even-order6-character-sturm-v1"


def _inflate(source_character, target_level: int):
    target_group = character_group(target_level)
    for coordinates in product(
        *(range(order) for order in target_group.generator_orders)
    ):
        target_character = dirichlet_character(target_group, coordinates)
        if all(
            dirichlet_character_value(target_character, residue).value
            == dirichlet_character_value(source_character, residue).value
            for residue in range(target_level)
            if gcd(residue, target_level) == 1
        ):
            return target_character
    raise AssertionError("no exact character inflation fixture was found")


def _space(level: int, character) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=character,
        coefficient_domain=_FIELD,
    )


def _element(constant: int, zeta: int = 0) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=_FIELD,
        coefficients_ascending=(
            {"num": constant, "den": 1},
            {"num": zeta, "den": 1},
        ),
    )


def _inclusion(source_space: ModularFormSpace, target_space: ModularFormSpace):
    return ModularCharacterSpaceInclusion(
        source_space=source_space,
        target_space=target_space,
        character_map=CyclotomicCharacterMap(
            source=source_space.character,
            target=target_space.character,
        ),
        coefficient_field_map=CyclotomicIdentityFieldMap(
            source=source_space.coefficient_domain,
            target=target_space.coefficient_domain,
        ),
    )


def _level_13_form(space: ModularFormSpace) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=space,
        basis_id=_LEGACY_BASIS,
        coordinates=(_element(1),),
    )


def test_transport_retains_inflation_and_exact_target_sturm_prefix() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)
    inclusion = _inclusion(source_space, target_space)

    transported = modular_character_coordinates_transport(
        _level_13_form(source_space), inclusion
    )

    assert transported.inclusion == inclusion
    assert transported.source_form.space == source_space
    assert transported.target_form.space == target_space
    assert type(transported.target_form) is ModularFormCoordinates
    assert transported.target_form.basis_id == _GENERIC_BASIS
    assert transported.target_form.coordinates == (_element(1), _element(-1, -1))
    assert len(transported.target_q_expansion.coefficients) == 8
    assert [
        tuple(
            (coefficient.num, coefficient.den)
            for coefficient in term.coefficients_ascending
        )
        for term in transported.target_q_expansion.coefficients
    ] == [
        ((0, 1), (0, 1)),
        ((1, 1), (0, 1)),
        ((-1, 1), (-1, 1)),
        ((-2, 1), (2, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (-2, 1)),
        ((4, 1), (-2, 1)),
        ((0, 1), (0, 1)),
    ]
    assert (
        TypeAdapter(ModularCharacterTransportedForm).validate_json(
            transported.model_dump_json()
        )
        == transported
    )
    assert Catalog.open().operation(
        "modular_form.character_coordinates.transport.compute"
    )


def test_common_target_global_equality_equal_and_v2_unequal_forms() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)

    from_level_13 = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    target_form_same_f = modular_character_coordinates_transport(
        from_level_13.target_form, _inclusion(target_space, target_space)
    )
    equal_request = ModularCharacterEqualityRequest(
        left=from_level_13,
        right=target_form_same_f,
    )
    assert modular_character_coordinates_equal_in_common_space(
        equal_request.left, equal_request.right
    ).equal

    # The second target basis row is exactly V_2(f)=f(q^2) in this canonical
    # q-Sturm frame. Add it to f and compare at the full level-26 Sturm bound.
    f_plus_v2 = ModularFormCoordinates(
        space=target_space,
        basis_id=_GENERIC_BASIS,
        coordinates=(_element(1), _element(0, -1)),
    )
    target_form_f_plus_v2 = modular_character_coordinates_transport(
        f_plus_v2, _inclusion(target_space, target_space)
    )
    assert not modular_character_coordinates_equal_in_common_space(
        from_level_13, target_form_f_plus_v2
    ).equal
    assert Catalog.open().operation("modular_form.character.equal.check")


def test_global_equality_rejects_forged_retained_target_coordinates() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)
    left = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    forged = left.model_copy(
        update={
            "target_form": ModularFormCoordinates(
                space=target_space,
                basis_id=_GENERIC_BASIS,
                coordinates=(_element(2), _element(0)),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError, match="do not match"):
        modular_character_coordinates_equal_in_common_space(left, forged)


def test_global_equality_retains_revalidated_nested_target_dictionaries() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    target_space = _space(26, _inflate(source_character, 26))
    canonical = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    forged = ModularCharacterTransportedForm.model_construct(
        **canonical.model_dump(mode="python")
    )

    assert modular_character_coordinates_equal_in_common_space(forged, canonical).equal


def test_transport_rejects_wrong_inflation_and_nonidentity_field_map() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    wrong_target = _space(26, _inflate(source_character, 26))
    wrong_character = dirichlet_character(character_group(26), (10,))
    wrong_character_space = _space(26, wrong_character)
    bad_map = _inclusion(source_space, wrong_character_space)
    with pytest.raises(
        OperationDomainValidationError, match="not the explicit inflation"
    ):
        modular_character_coordinates_transport(_level_13_form(source_space), bad_map)

    larger_field = RationalCyclotomicField(order=12)
    mismatched_target = ModularFormSpace(
        level=26,
        weight=2,
        kind="S",
        character=wrong_target.character,
        coefficient_domain=larger_field,
    )
    bad_field_map = ModularCharacterSpaceInclusion.model_construct(
        source_space=source_space,
        target_space=mismatched_target,
        character_map=CyclotomicCharacterMap(
            source=source_character, target=wrong_target.character
        ),
        coefficient_field_map=CyclotomicIdentityFieldMap(
            source=_FIELD, target=larger_field
        ),
    )
    with pytest.raises(
        OperationDomainValidationError,
        match="explicit character-space inclusion is malformed",
    ):
        modular_character_coordinates_transport(
            _level_13_form(source_space), bad_field_map
        )


def test_transport_request_model_round_trip() -> None:
    character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, character)
    target_space = _space(26, _inflate(character, 26))
    request = ModularCharacterCoordinatesTransportRequest(
        form=_level_13_form(source_space),
        inclusion=_inclusion(source_space, target_space),
    )
    assert (
        TypeAdapter(ModularCharacterCoordinatesTransportRequest).validate_json(
            request.model_dump_json()
        )
        == request
    )


def test_transport_height_boundary_is_admitted_before_basis_materialization(
    monkeypatch,
) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis

    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)
    inclusion = _inclusion(source_space, target_space)

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("PARI basis work ran before height admission")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    for coefficient in (10**29, 10**42):
        oversized = ModularFormCoordinates(
            space=source_space,
            basis_id=_LEGACY_BASIS,
            coordinates=(_element(coefficient),),
        )
        with pytest.raises(
            OperationResourceAdmissionError,
            match="expansion or target solve exceeds",
        ):
            modular_character_coordinates_transport(oversized, inclusion)


def test_transport_accepts_admitted_height_boundary_through_exact_expansion() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)
    coefficient = 10**28  # 29 digits; the next digit is rejected above.
    form = ModularFormCoordinates(
        space=source_space,
        basis_id=_LEGACY_BASIS,
        coordinates=(_element(coefficient),),
    )

    transported = modular_character_coordinates_transport(
        form, _inclusion(source_space, target_space)
    )

    assert transported.target_form.coordinates == (
        _element(coefficient),
        _element(-coefficient, -coefficient),
    )


@pytest.mark.parametrize(
    ("level", "precision", "dimension"),
    [(13, 3, 1), (13, 8, 1), (13, 10, 1), (26, 8, 2), (26, 10, 2), (39, 10, 3)],
)
@pytest.mark.parametrize("character_coordinate", [2, 10])
def test_transport_sturm_basis_coefficient_envelope(
    level: int, precision: int, dimension: int, character_coordinate: int
) -> None:
    character = dirichlet_character(character_group(13), (character_coordinate,))
    character = _inflate(character, level)
    basis = modular_character_basis_q_expansions(
        _space(level, character), precision=precision
    )

    assert len(basis.elements) == dimension
    assert all(
        value.den == 1 and abs(int(value.num)) < 10
        for element in basis.elements
        for coefficient in element.expansion.coefficients
        for value in coefficient.coefficients_ascending
    )
