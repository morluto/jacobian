"""Exact bad-prime U actions on represented cyclotomic character spaces."""

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
from jacobian.math.number_theory.modular_forms import character_basis as basis_module
from jacobian.math.number_theory.modular_forms.character_basis import (
    modular_character_basis_q_expansions,
    modular_character_coordinates_u_prime,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
)

_FIELD = RationalCyclotomicField(order=6)
_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"


# Columns are U_p images of canonical basis vectors; rows are target
# coordinates in ascending cyclotomic power-basis order.
def _inflated_character(coordinate: int, level: int):
    source = dirichlet_character(character_group(13), (coordinate,))
    group = character_group(level)
    for candidate_coordinates in product(
        *(range(order) for order in group.generator_orders)
    ):
        candidate = dirichlet_character(group, candidate_coordinates)
        if all(
            dirichlet_character_value(candidate, residue).value
            == dirichlet_character_value(source, residue).value
            for residue in range(level)
            if gcd(residue, level) == 1
        ):
            return candidate
    raise AssertionError("the exact character inflation must exist")


def _space(coordinate: int, level: int) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=_inflated_character(coordinate, level),
        coefficient_domain=_FIELD,
    )


def _unit(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


def _zero(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 0, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


def _coordinates(space: ModularFormSpace, index: int) -> ModularFormCoordinates:
    basis = modular_character_basis_q_expansions(space)
    return ModularFormCoordinates(
        space=space,
        basis_id=_GENERIC_BASIS,
        coordinates=tuple(
            _unit(_FIELD) if coordinate == index else _zero(_FIELD)
            for coordinate in range(len(basis.elements))
        ),
    )


def test_u_prime_rejects_a_prime_outside_the_exact_level_action() -> None:
    space = _space(2, 26)
    with pytest.raises(OperationDomainValidationError):
        modular_character_coordinates_u_prime(_coordinates(space, 0), 3)


def test_u_prime_rejects_order_three_character_before_using_order_six_envelope() -> (
    None
):
    space = _space(4, 26)
    form = _coordinates(space, 0)

    with pytest.raises(
        OperationDomainValidationError,
        match="coefficient envelope is established only for order-six",
    ):
        modular_character_coordinates_u_prime(form, 2)


def test_u_prime_accepts_seven_digit_coordinate_without_compounding_growth() -> None:
    space = _space(2, 26)
    form = _coordinates(space, 0)
    large_scalar = RationalCyclotomicElement(
        field=_FIELD,
        coefficients_ascending=(
            {"num": 9_999_999, "den": 1},
            {"num": 0, "den": 1},
        ),
    )
    form = form.model_copy(update={"coordinates": (large_scalar, _zero(_FIELD))})

    result = modular_character_coordinates_u_prime(form, 2)

    assert result.coordinates == (
        _zero(_FIELD),
        RationalCyclotomicElement(
            field=_FIELD,
            coefficients_ascending=(
                {"num": 0, "den": 1},
                {"num": -19_999_998, "den": 1},
            ),
        ),
    )


def test_u_prime_result_height_allows_distinct_component_denominators() -> None:
    space = _space(2, 26)
    form = _coordinates(space, 0)
    form = form.model_copy(
        update={
            "coordinates": (
                RationalCyclotomicElement(
                    field=_FIELD,
                    coefficients_ascending=(
                        {"num": 1, "den": 10_007},
                        {"num": 1, "den": 10_009},
                    ),
                ),
                RationalCyclotomicElement(
                    field=_FIELD,
                    coefficients_ascending=(
                        {"num": 1, "den": 10_037},
                        {"num": 1, "den": 10_039},
                    ),
                ),
            )
        }
    )

    result = modular_character_coordinates_u_prime(form, 2)

    assert (
        max(
            len(str(coefficient.den))
            for value in result.coordinates
            for coefficient in value.coefficients_ascending
        )
        > 15
    )


@pytest.mark.parametrize("digits", [42, 43, 255])
def test_u_prime_output_bound_is_admitted_before_backend_expansion(
    monkeypatch, digits: int
) -> None:
    space = _space(2, 26)
    form = _coordinates(space, 0)
    oversized = RationalCyclotomicElement(
        field=_FIELD,
        coefficients_ascending=(
            {"num": int("9" * digits), "den": 1},
            {"num": 0, "den": 1},
        ),
    )
    form = form.model_copy(update={"coordinates": (oversized, _zero(_FIELD))})

    def backend_must_not_run(*_args, **_kwargs):
        raise AssertionError("backend started before output growth was admitted")

    monkeypatch.setattr(basis_module, "pari_character_basis", backend_must_not_run)
    from jacobian.catalog.models import OperationResourceAdmissionError

    with pytest.raises(OperationResourceAdmissionError):
        modular_character_coordinates_u_prime(form, 2)
