from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.characters import _tools
from jacobian.math.number_theory.characters.operations import (
    character_group,
    dirichlet_character,
    dirichlet_character_orthogonality,
)

_ROOTS = {
    1: ((1, 0),),
    2: ((1, 0), (-1, 0)),
    4: ((1, 0), (0, 1), (-1, 0), (0, -1)),
}


def _exponent(character, residue: int) -> int | None:
    group = character.group
    try:
        row = group.unit_coordinates[group.unit_residues.index(residue)]
    except ValueError:
        return None
    return (
        sum(
            coordinate * (group.exponent // axis_order) * unit_coordinate
            for coordinate, axis_order, unit_coordinate in zip(
                character.coordinates,
                group.generator_orders,
                row,
                strict=True,
            )
        )
        % group.exponent
    )


def _direct_root_sum(left, right) -> tuple[int, int]:
    """Independent direct residue sum in Q, Q(i), or Q(zeta_2)."""

    order = left.group.exponent
    roots = _ROOTS[order]
    real_total = 0
    imaginary_total = 0
    for residue in range(left.group.modulus):
        left_exponent = _exponent(left, residue)
        right_exponent = _exponent(right, residue)
        if left_exponent is None or right_exponent is None:
            continue
        left_root = roots[left_exponent % order]
        right_root = roots[(-right_exponent) % order]
        real_total += left_root[0] * right_root[0] - left_root[1] * right_root[1]
        imaginary_total += left_root[0] * right_root[1] + left_root[1] * right_root[0]
    return real_total, imaginary_total


def test_orthogonality_equal_and_distinct_characters_match_direct_root_sum() -> None:
    group = character_group(5)
    trivial = dirichlet_character(group, (0,))
    quartic = dirichlet_character(group, (1,))
    quadratic = dirichlet_character(group, (2,))

    equal = dirichlet_character_orthogonality(trivial, trivial)
    assert _direct_root_sum(trivial, trivial) == (4, 0)
    assert equal.value == 4
    assert equal.left == trivial and equal.right == trivial

    distinct = dirichlet_character_orthogonality(quartic, quadratic)
    assert _direct_root_sum(quartic, quadratic) == (0, 0)
    assert distinct.value == 0
    assert distinct.left == quartic and distinct.right == quadratic


def test_orthogonality_of_nonprincipal_character_is_exact() -> None:
    group = character_group(3)
    character = dirichlet_character(group, (1,))
    result = dirichlet_character_orthogonality(character, character)
    assert _direct_root_sum(character, character) == (2, 0)
    assert result.value == 2


def test_orthogonality_handles_the_trivial_group_modulo_one() -> None:
    group = character_group(1)
    character = dirichlet_character(group, ())
    result = dirichlet_character_orthogonality(character, character)
    assert _direct_root_sum(character, character) == (1, 0)
    assert result.value == 1


def test_orthogonality_rejects_different_group_parents() -> None:
    left = dirichlet_character(character_group(3), (1,))
    right = dirichlet_character(character_group(5), (2,))
    with pytest.raises(OperationDomainValidationError) as error:
        dirichlet_character_orthogonality(left, right)
    assert error.value.errors()[0]["type"] == (
        "dirichlet_character.orthogonality.parent_mismatch"
    )


def test_orthogonality_catalog_example_executes_and_round_trips() -> None:
    tool = next(
        tool
        for tool in _tools.TOOLS
        if tool.operation_id == "dirichlet_character.orthogonality.compute"
    )
    request = tool.request_type.model_validate_json(json.dumps(tool.examples[0].input))
    result = tool.run(request)
    assert result.value == 2
    tool.result_type.model_validate_json(json.dumps(result.model_dump(mode="json")))
