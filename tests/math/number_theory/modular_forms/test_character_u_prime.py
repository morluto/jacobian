"""Exact bad-prime U actions on represented cyclotomic character spaces."""

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
from jacobian.math.number_theory.modular_forms import character_basis as basis_module
from jacobian.math.number_theory.modular_forms import cyclotomic
from jacobian.math.number_theory.modular_forms.character_basis import (
    _character_sturm_precision,
    modular_character_basis_q_expansions,
    modular_character_coordinates_u_prime,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterCoordinates,
    ModularCharacterUPrimeRequest,
)
from jacobian.math.number_theory.modular_forms.values import ModularFormSpace

_FIELD = RationalCyclotomicField(order=6)
_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"

# Columns are U_p images of canonical basis vectors; rows are target
# coordinates in ascending cyclotomic power-basis order.
_EXPECTED = {
    (2, 26, 2): (((0, 0), (0, -2)), ((1, 0), (-1, -1))),
    (10, 26, 2): (((0, 0), (-2, 2)), ((1, 0), (-2, 1))),
    (2, 26, 13): (((-1, -3), (0, 0)), ((0, 0), (-1, -3))),
    (10, 26, 13): (((-4, 3), (0, 0)), ((0, 0), (-4, 3))),
    (2, 39, 3): (
        ((0, 0), (2, -1), (0, -3)),
        ((0, 0), (1, -1), (-2, -2)),
        ((1, 0), (-1, -1), (-2, 2)),
    ),
    (10, 39, 3): (
        ((0, 0), (1, 1), (-3, 3)),
        ((0, 0), (0, 1), (-4, 2)),
        ((1, 0), (-2, 1), (0, -2)),
    ),
    (2, 39, 13): (
        ((4, -1), (0, 0), (7, -5)),
        ((4, -1), (-1, -3), (3, -4)),
        ((0, 0), (0, 0), (-1, -3)),
    ),
    (10, 39, 13): (
        ((3, 1), (0, 0), (2, 5)),
        ((3, 1), (-4, 3), (-1, 4)),
        ((0, 0), (0, 0), (-4, 3)),
    ),
}


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


def _coordinates(space: ModularFormSpace, index: int) -> ModularCharacterCoordinates:
    basis = modular_character_basis_q_expansions(space)
    return ModularCharacterCoordinates(
        space=space,
        basis_id=_GENERIC_BASIS,
        coordinates=tuple(
            _unit(_FIELD) if coordinate == index else _zero(_FIELD)
            for coordinate in range(len(basis.elements))
        ),
    )


def _numerators(value: RationalCyclotomicElement) -> tuple[int, ...]:
    return tuple(int(coefficient.num) for coefficient in value.coefficients_ascending)


@pytest.mark.parametrize("character_coordinate,level,prime", tuple(_EXPECTED))
def test_u_prime_matches_independent_q_prefix_action_and_target_coordinates(
    character_coordinate: int, level: int, prime: int
) -> None:
    space = _space(character_coordinate, level)
    sturm_precision = _character_sturm_precision(space)
    source_precision = prime * (sturm_precision - 1) + 1
    source_basis = modular_character_basis_q_expansions(space, source_precision)
    target_basis = modular_character_basis_q_expansions(space)
    result_columns = []
    for basis_index in range(len(target_basis.elements)):
        form = _coordinates(space, basis_index)
        result = modular_character_coordinates_u_prime(form, prime)
        assert result.space == space
        assert result.basis_id == _GENERIC_BASIS
        assert (
            TypeAdapter(ModularCharacterCoordinates).validate_json(
                result.model_dump_json()
            )
            == result
        )
        result_columns.append(tuple(_numerators(value) for value in result.coordinates))

        source_coefficients = source_basis.elements[basis_index].expansion.coefficients
        expected_prefix = tuple(
            source_coefficients[prime * exponent] for exponent in range(sturm_precision)
        )
        # Expand the returned coordinates using the public exact target basis.
        target_prefix = tuple(
            _sum(
                cyclotomic.multiply(
                    result.coordinates[row],
                    target_basis.elements[row].expansion.coefficients[exponent],
                )
                for row in range(len(result.coordinates))
            )
            for exponent in range(sturm_precision)
        )
        assert target_prefix == expected_prefix

    assert tuple(result_columns) == _EXPECTED[(character_coordinate, level, prime)]
    tool = Catalog.open().operation("modular_form.character_coordinates.u_prime.apply")
    request = ModularCharacterUPrimeRequest(form=_coordinates(space, 0), prime=prime)
    assert tool.run(request).space == space


def _sum(values):
    result = _zero(_FIELD)
    for value in values:
        result = cyclotomic.add(result, value)
    return result


def test_u_prime_rejects_a_prime_outside_the_exact_level_action() -> None:
    space = _space(2, 26)
    with pytest.raises(OperationDomainValidationError):
        modular_character_coordinates_u_prime(_coordinates(space, 0), 3)


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
