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
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    CyclotomicCharacterMap,
    CyclotomicIdentityFieldMap,
    ModularCharacterCoordinates,
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


def _sum_cyclotomic(values) -> RationalCyclotomicElement:
    total = _element(0)
    for value in values:
        total = cyclotomic.add(total, value)
    return total


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
    f_plus_v2 = ModularCharacterCoordinates(
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


def test_nonnested_26_39_forms_compare_in_their_level_78_common_space() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_13 = _space(13, source_character)
    source_26 = _space(26, _inflate(source_character, 26))
    source_39 = _space(39, _inflate(source_character, 39))
    target_78 = _space(78, _inflate(source_character, 78))

    form_26 = modular_character_coordinates_transport(
        _level_13_form(source_13), _inclusion(source_13, source_26)
    ).target_form
    form_39 = modular_character_coordinates_transport(
        _level_13_form(source_13), _inclusion(source_13, source_39)
    ).target_form
    left = modular_character_coordinates_transport(
        form_26, _inclusion(source_26, target_78)
    )
    right = modular_character_coordinates_transport(
        form_39, _inclusion(source_39, target_78)
    )
    equal = modular_character_coordinates_equal_in_common_space(left, right)

    assert left.target_form is None and right.target_form is None
    assert left.target_q_expansion.space == right.target_q_expansion.space == target_78
    assert len(left.target_q_expansion.coefficients) == 29
    assert len(right.target_q_expansion.coefficients) == 29
    for source_space, source_form, transported in (
        (source_26, form_26, left),
        (source_39, form_39, right),
    ):
        source_basis = modular_character_basis_q_expansions(
            source_space, precision=29
        )
        direct_prefix = tuple(
            _sum_cyclotomic(
                cyclotomic.multiply(
                    coordinate,
                    source_basis.elements[index].expansion.coefficients[q],
                )
                for index, coordinate in enumerate(source_form.coordinates)
            )
            for q in range(29)
        )
        assert transported.target_q_expansion.coefficients == direct_prefix
    assert equal.equal is True
    assert equal.equal is (
        left.target_q_expansion.coefficients == right.target_q_expansion.coefficients
    )
    assert (
        TypeAdapter(ModularCharacterTransportedForm).validate_json(
            left.model_dump_json()
        )
        == left
    )

    twice_39 = ModularCharacterCoordinates(
        space=source_39,
        basis_id=form_39.basis_id,
        coordinates=tuple(
            cyclotomic.multiply(value, _element(2)) for value in form_39.coordinates
        ),
    )
    unequal_right = modular_character_coordinates_transport(
        twice_39, _inclusion(source_39, target_78)
    )
    unequal = modular_character_coordinates_equal_in_common_space(left, unequal_right)

    assert unequal.equal is False
    assert unequal.equal is (
        left.target_q_expansion.coefficients
        == unequal_right.target_q_expansion.coefficients
    )
    assert (
        left.target_q_expansion.coefficients[1]
        != (unequal_right.target_q_expansion.coefficients[1])
    )


def test_level_78_common_prefix_admits_source_expansion_and_rejects_height(
    monkeypatch,
) -> None:
    from jacobian.math.number_theory.modular_forms import character_basis

    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    target_space = _space(78, _inflate(source_character, 78))
    transported = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    assert transported.target_form is None
    assert transported.target_q_expansion.precision == 29
    assert len(transported.target_q_expansion.coefficients) == 29

    source_39 = _space(39, _inflate(source_character, 39))
    boundary = ModularCharacterCoordinates(
        space=source_39,
        basis_id=_GENERIC_BASIS,
        coordinates=(_element(10**82), _element(0), _element(0)),
    )
    admitted = modular_character_coordinates_transport(
        boundary, _inclusion(source_39, target_space)
    )
    assert admitted.target_q_expansion.precision == 29

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("source basis backend ran after output-height rejection")

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    oversized = ModularCharacterCoordinates(
        space=source_39,
        basis_id=_GENERIC_BASIS,
        coordinates=(_element(10**83), _element(0), _element(0)),
    )
    with pytest.raises(
        OperationResourceAdmissionError, match="expansion or target solve exceeds"
    ):
        modular_character_coordinates_transport(
            oversized, _inclusion(source_39, target_space)
        )


def test_global_equality_rejects_forged_retained_target_coordinates() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    target_character = _inflate(source_character, 26)
    source_space = _space(13, source_character)
    target_space = _space(26, target_character)
    left = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    right = modular_character_coordinates_transport(
        left.target_form, _inclusion(target_space, target_space)
    )
    forged = left.model_copy(
        update={
            "target_form": ModularCharacterCoordinates(
                space=target_space,
                basis_id=_GENERIC_BASIS,
                coordinates=(_element(2), _element(0)),
            )
        }
    )
    with pytest.raises(OperationDomainValidationError, match="do not match"):
        modular_character_coordinates_equal_in_common_space(right, forged)


def test_global_equality_requires_the_least_common_source_level() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_13 = _space(13, source_character)
    target_26 = _space(26, _inflate(source_character, 26))
    first = modular_character_coordinates_transport(
        _level_13_form(source_13), _inclusion(source_13, target_26)
    )
    second = modular_character_coordinates_transport(
        first.target_form, _inclusion(target_26, target_26)
    )
    identity = modular_character_coordinates_transport(
        _level_13_form(source_13), _inclusion(source_13, source_13)
    )

    with pytest.raises(OperationDomainValidationError, match="least common multiple"):
        modular_character_coordinates_equal_in_common_space(first, first)
    assert identity.target_form is None
    assert len(identity.target_q_expansion.coefficients) == 3
    assert modular_character_coordinates_equal_in_common_space(
        identity, identity
    ).equal
    assert modular_character_coordinates_equal_in_common_space(second, second).equal


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


def test_transport_rejects_malformed_constructed_inclusion_with_typed_error() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    target_space = _space(26, _inflate(source_character, 26))
    inclusion = _inclusion(source_space, target_space)
    malformed_source = source_space.model_dump()
    malformed_source["level"] = "13"
    forged_inclusion = inclusion.model_copy(update={"source_space": malformed_source})

    with pytest.raises(OperationDomainValidationError) as error:
        modular_character_coordinates_transport(
            _level_13_form(source_space), forged_inclusion
        )

    assert error.value.errors()[0]["type"] == "modular_form.character_inclusion_invalid"


def test_transport_rejects_malformed_constructed_form_with_typed_error() -> None:
    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    target_space = _space(26, _inflate(source_character, 26))
    malformed_space = source_space.model_dump()
    malformed_space["level"] = "13"
    forged_form = _level_13_form(source_space).model_copy(
        update={"space": malformed_space}
    )

    with pytest.raises(OperationDomainValidationError) as error:
        modular_character_coordinates_transport(
            forged_form, _inclusion(source_space, target_space)
        )

    assert (
        error.value.errors()[0]["type"] == "modular_form.character_coordinates_invalid"
    )


def test_common_equality_rejects_malformed_constructed_transport_with_typed_error() -> (
    None
):
    source_character = dirichlet_character(character_group(13), (2,))
    source_space = _space(13, source_character)
    target_space = _space(26, _inflate(source_character, 26))
    transported = modular_character_coordinates_transport(
        _level_13_form(source_space), _inclusion(source_space, target_space)
    )
    malformed_space = source_space.model_dump()
    malformed_space["level"] = "13"
    forged = transported.model_copy(
        update={
            "source_form": {
                **transported.source_form.model_dump(),
                "space": malformed_space,
            }
        }
    )

    with pytest.raises(OperationDomainValidationError) as error:
        modular_character_coordinates_equal_in_common_space(forged, transported)

    assert (
        error.value.errors()[0]["type"]
        == "modular_form.character_transport_value_invalid"
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
    for coefficient in (10**121, 10**200):
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
    coefficient = 10**120  # 121 digits; the next digit is rejected above.
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
    ("level", "precision", "dimension", "coefficient_digits"),
    [
        (13, 3, 1, 1),
        (13, 8, 1, 1),
        (13, 10, 1, 1),
        (13, 29, 1, 1),
        (26, 8, 2, 1),
        (26, 10, 2, 1),
        (26, 29, 2, 1),
        (39, 10, 3, 1),
        (39, 29, 3, 1),
    ],
)
@pytest.mark.parametrize("character_coordinate", [2, 10])
def test_transport_sturm_basis_coefficient_envelope(
    level: int,
    precision: int,
    dimension: int,
    coefficient_digits: int,
    character_coordinate: int,
) -> None:
    character = dirichlet_character(character_group(13), (character_coordinate,))
    character = _inflate(character, level)
    basis = modular_character_basis_q_expansions(
        _space(level, character), precision=precision
    )

    assert len(basis.elements) == dimension
    assert all(
        len(str(abs(int(value.num)))) <= coefficient_digits
        and len(str(int(value.den))) <= coefficient_digits
        for element in basis.elements
        for coefficient in element.expansion.coefficients
        for value in coefficient.coefficients_ascending
    )
