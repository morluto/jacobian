"""Antidominant representatives against an independent finite-group oracle."""

from __future__ import annotations

from collections import deque

import pytest

from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import (
    weyl_antidominant_representative,
    weyl_dominant_representative,
    weyl_longest_element,
)


def _identity(rank: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(i == j) for j in range(rank)) for i in range(rank))


def _multiply(
    left: tuple[tuple[int, ...], ...], right: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    size = len(left)
    return tuple(
        tuple(sum(left[i][k] * right[k][j] for k in range(size)) for j in range(size))
        for i in range(size)
    )


def _reflection_pair(
    matrix: tuple[tuple[int, ...], ...], index: int
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    rank = len(matrix)
    root = [list(row) for row in _identity(rank)]
    weight = [list(row) for row in _identity(rank)]
    for column in range(rank):
        root[index][column] -= matrix[index][column]
    for row in range(rank):
        weight[row][index] -= matrix[row][index]
    return tuple(map(tuple, root)), tuple(map(tuple, weight))


def _oracle_group(
    matrix: tuple[tuple[int, ...], ...],
) -> set[
    tuple[
        tuple[tuple[int, ...], ...],
        tuple[tuple[int, ...], ...],
    ]
]:
    rank = len(matrix)
    generators = tuple(_reflection_pair(matrix, i) for i in range(rank))
    identity = (_identity(rank), _identity(rank))
    found = {identity}
    pending = deque([identity])
    while pending:
        root_action, weight_action = pending.popleft()
        for root_reflection, weight_reflection in generators:
            image = (
                _multiply(root_reflection, root_action),
                _multiply(weight_reflection, weight_action),
            )
            if image not in found:
                found.add(image)
                pending.append(image)
    return found


def _act(
    matrix: tuple[tuple[int, ...], ...], weight: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        sum(matrix[row][column] * weight[column] for column in range(len(weight)))
        for row in range(len(weight))
    )


@pytest.mark.parametrize(
    ("matrix", "weight"),
    [
        (((2, -1), (-1, 2)), (-1, 1)),
        (((2, -2), (-1, 2)), (2, -1)),
        (((2, -1, 0), (-1, 2, 0), (0, 0, 2)), (-1, 2, -3)),
    ],
)
def test_antidominant_representative_and_transporter_match_group_closure(
    matrix: tuple[tuple[int, ...], ...], weight: tuple[int, ...]
) -> None:
    group = _oracle_group(matrix)
    orbit = {
        _act(weight_action, weight): root_action for root_action, weight_action in group
    }
    antidominant = [
        candidate for candidate in orbit if all(value <= 0 for value in candidate)
    ]
    assert len(antidominant) == 1

    result = weyl_antidominant_representative(matrix, weight)

    assert result.antidominant_weight == antidominant[0]
    assert result.element.matrix.entries == matrix
    assert any(
        root_action == result.element.root_action.entries
        and _act(weight_action, weight) == result.antidominant_weight
        for root_action, weight_action in group
    )

    # The longest element carries the dominant representative to the unique
    # antidominant representative, including for reducible finite type.
    dominant = weyl_dominant_representative(matrix, weight)
    longest = weyl_longest_element(matrix)
    weight_action = _identity(len(weight))
    for index in longest.word:
        reflection = [list(row) for row in _identity(len(weight))]
        for row in range(len(weight)):
            reflection[row][index] -= matrix[row][index]
        weight_action = _multiply(tuple(map(tuple, reflection)), weight_action)
    assert _act(weight_action, dominant.dominant_weight) == result.antidominant_weight


@pytest.mark.parametrize("weight", [(-1, -2), (0, 0)])
def test_antidominant_input_preserves_identity_transporter(
    weight: tuple[int, int],
) -> None:
    result = weyl_antidominant_representative(((2, -1), (-1, 2)), weight)

    assert result.antidominant_weight == weight
    assert result.element.root_action.entries == _identity(2)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_antidominant_representative_is_a_published_operation() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "weyl_group.antidominant_representative.compute"
    )

    assert tool.request_type.model_validate(
        {"matrix": ((2, -1), (-1, 2)), "weight": (-1, 1)}
    )
