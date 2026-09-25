"""Exact global equality for multidimensional cyclotomic character spaces."""

from __future__ import annotations

from itertools import product
from math import gcd

import pytest
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
    dirichlet_character_value,
)
from jacobian.math.number_theory.modular_forms.basis import (
    modular_form_coordinates_equal,
)
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_coordinates import (
    CHARACTER_RREF_BASIS_ID,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_FIELD = RationalCyclotomicField(order=6)


def _inflated_character(level: int, coordinate: int = 4):
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
    raise AssertionError("exact character inflation fixture was not found")


def _space(level: int = 13, kind: str = "M") -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind=kind,
        character=_inflated_character(level),
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


def _form(space: ModularFormSpace, *coordinates: RationalCyclotomicElement):
    return ModularFormCoordinates(
        space=space,
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=coordinates,
    )


def test_multidimensional_character_coordinates_decide_global_equality() -> None:
    """A q-Sturm RREF basis turns exact vectors into a global equality test."""
    space = _space()
    basis = modular_character_basis_q_expansions(space)
    assert len(basis.elements) == 2
    assert basis.precision == 3

    first = _form(space, _element(1), _element(0))
    second = _form(space, _element(0), _element(1))
    restored = TypeAdapter(ModularFormCoordinates).validate_json(
        first.model_dump_json()
    )
    assert restored == first
    restored_second = TypeAdapter(ModularFormCoordinates).validate_json(
        second.model_dump_json()
    )

    # The exact producer prefixes are an independent coefficient oracle for
    # the two distinct coordinate vectors in the Sturm-determining basis.
    assert basis.elements[0].expansion.coefficients != (
        basis.elements[1].expansion.coefficients
    )
    assert modular_form_coordinates_equal(first, first)
    assert not modular_form_coordinates_equal(first, second)

    tool = Catalog.open().operation("modular_form.equal.check")
    result = tool.run(tool.request_type(left=restored, right=restored_second))
    assert result.equal is False


def test_empty_coordinates_are_the_unique_form_in_a_zero_dimensional_space() -> None:
    space = _space(kind="S")
    assert modular_character_basis_q_expansions(space).elements == ()
    zero = _form(space)
    assert modular_form_coordinates_equal(zero, zero)

    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_equal(_form(space, _element(0)), zero)
    assert (
        error.value.errors()[0]["type"]
        == "modular_form.character_coordinates_dimension"
    )


def test_inflated_multidimensional_coordinates_keep_the_exact_parent() -> None:
    space = _space(level=26)
    basis = modular_character_basis_q_expansions(space)
    assert len(basis.elements) == 5
    coordinates = tuple(_element(index) for index in range(5))
    form = _form(space, *coordinates)
    assert modular_form_coordinates_equal(form, form)

    other_space = _space(level=39)
    other = _form(other_space, *(_element(index) for index in range(7)))
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_equal(form, other)
    assert error.value.errors()[0]["type"] == "modular_form.equality_parent_unsupported"


def test_character_coordinate_equality_does_not_replay_the_basis(monkeypatch) -> None:
    space = _space()
    form = _form(space, _element(1), _element(0))

    def backend_must_not_run(*args, **kwargs):
        raise AssertionError("coordinate equality must not replay basis mathematics")

    from jacobian.math.number_theory.modular_forms import character_basis

    monkeypatch.setattr(character_basis, "pari_character_basis", backend_must_not_run)
    assert modular_form_coordinates_equal(form, form)


def test_character_coordinates_reject_wrong_canonical_basis_and_parent() -> None:
    space = _space()
    valid = _form(space, _element(1), _element(0))
    wrong_basis = valid.model_copy(
        update={"basis_id": "gamma0-13-even-order6-character-sturm-v1"}
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_equal(wrong_basis, valid)
    assert error.value.errors()[0]["type"] == "modular_form.character_coordinates_basis"

    foreign_field_value = RationalCyclotomicElement(
        field=RationalCyclotomicField(order=3),
        coefficients_ascending=(
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
        ),
    )
    forged = ModularFormCoordinates.model_construct(
        space=space,
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(foreign_field_value, _element(0)),
    )
    with pytest.raises(OperationDomainValidationError) as parent_error:
        modular_form_coordinates_equal(forged, valid)
    assert (
        parent_error.value.errors()[0]["type"]
        == "modular_form.character_coordinates_parent"
    )


def test_oversized_character_claim_is_rejected_before_group_reconstruction(
    monkeypatch,
) -> None:
    space = _space()
    character = type(space.character).model_construct(
        group=space.character.group,
        coordinates=(0,) * 33,
    )
    forged_space = ModularFormSpace.model_construct(
        group=space.group,
        level=space.level,
        weight=space.weight,
        kind=space.kind,
        character=character,
        coefficient_domain=space.coefficient_domain,
    )
    forged_form = ModularFormCoordinates.model_construct(
        space=forged_space,
        basis_id=CHARACTER_RREF_BASIS_ID,
        coordinates=(_element(1), _element(0)),
    )

    from jacobian.math.number_theory.modular_forms import character_coordinates

    def reconstruction_must_not_run(*args, **kwargs):
        raise AssertionError("oversized caller data reached group reconstruction")

    monkeypatch.setattr(
        character_coordinates, "_require_basis_space", reconstruction_must_not_run
    )
    with pytest.raises(OperationDomainValidationError) as error:
        modular_form_coordinates_equal(forged_form, forged_form)
    assert (
        error.value.errors()[0]["type"]
        == "modular_form.character_coordinates_character_shape"
    )
