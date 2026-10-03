"""Modular-form coordinate transport and u-prime catalog evidence."""

from __future__ import annotations

from itertools import product
from math import gcd

import pytest
from pydantic import TypeAdapter

from jacobian._exact import CanonicalRational
from jacobian.canonical import encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.cyclic_linear import (
    RationalCyclotomicElement,
    RationalCyclotomicField,
)
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_value,
)
from jacobian.math.number_theory.modular_forms import (
    cyclotomic,
    modular_character_coordinates_u_prime,
    modular_form_space_inclusion,
)
from jacobian.math.number_theory.modular_forms._tools import TOOLS
from jacobian.math.number_theory.modular_forms.character_basis import (
    _character_sturm_precision,
    modular_character_basis_q_expansions,
)
from jacobian.math.number_theory.modular_forms.character_basis_models import (
    ModularCharacterUPrimeRequest,
)
from jacobian.math.number_theory.modular_forms.values import (
    ModularFormCoordinates,
    ModularFormSpace,
    ModularFormSpaceInclusion,
)

_FIELD = RationalCyclotomicField(order=6)

_GENERIC_BASIS = "gamma0-cyclotomic-character-sturm-rref-v1"

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


def _space(coordinate: int, level: int) -> ModularFormSpace:
    return ModularFormSpace(
        level=level,
        weight=2,
        kind="S",
        character=_inflated_character(coordinate, level),
        coefficient_domain=_FIELD,
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


def _numerators(value: RationalCyclotomicElement) -> tuple[int, ...]:
    return tuple(int(coefficient.num) for coefficient in value.coefficients_ascending)


def _sum(values):
    result = _zero(_FIELD)
    for value in values:
        result = cyclotomic.add(result, value)
    return result


def _transport_coordinates(
    level: int, weight: int, kind: str, basis_id: str, values: tuple[int, ...]
) -> ModularFormCoordinates:
    return ModularFormCoordinates(
        space=ModularFormSpace(level=level, weight=weight, kind=kind),  # type: ignore[arg-type]
        basis_id=basis_id,  # type: ignore[arg-type]
        coordinates=tuple(CanonicalRational(num=value, den=1) for value in values),
    )


def _unit(field: RationalCyclotomicField) -> RationalCyclotomicElement:
    return RationalCyclotomicElement(
        field=field,
        coefficients_ascending=(
            {"num": 1, "den": 1},
            {"num": 0, "den": 1},
        ),
    )


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
            TypeAdapter(ModularFormCoordinates).validate_json(result.model_dump_json())
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
    assert "order-six" in tool.description
    request = ModularCharacterUPrimeRequest(form=_coordinates(space, 0), prime=prime)
    # Route through dispatch so operation-ID lookup, strict wire parsing, and
    # the serialized output envelope all have to accept this request.
    applied = invoke_operation(
        tool.operation_id, request.model_dump(mode="json"), Catalog.open()
    )
    assert tool.result_type.model_validate(applied.output).space == space


def test_transport_catalog_operation_round_trips_target_coordinates() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.coordinates.transport.compute"
    )
    source = _transport_coordinates(1, 4, "M", "level-one-e4-e6-monomials-v1", (1,))
    inclusion_tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "modular_form.space.inclusion.compute"
    )
    catalog = Catalog.open()
    inclusion_result = invoke_operation(
        inclusion_tool.operation_id,
        {
            "source_space": source.space.model_dump(mode="json"),
            "target_space": {"level": 2, "weight": 4, "kind": "M"},
        },
        catalog,
    )
    decoded = ModularFormSpaceInclusion.model_validate_json(
        encode_strict_json(inclusion_result.output)
    )
    assert decoded == modular_form_space_inclusion(
        source.space, ModularFormSpace(level=2, weight=4, kind="M")
    )

    result = invoke_operation(
        tool.operation_id,
        {"form": source.model_dump(mode="json"), "inclusion": inclusion_result.output},
        catalog,
    )

    assert result.output["space"]["level"] == 2
    assert (
        ModularFormCoordinates.model_validate_json(
            encode_strict_json(result.output)
        ).space.level
        == 2
    )
