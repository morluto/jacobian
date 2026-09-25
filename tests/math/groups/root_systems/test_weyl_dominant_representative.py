"""Dominant orbit representatives against an independent finite-group oracle."""

from __future__ import annotations

from collections import deque

import pytest

from jacobian.math.groups.root_systems._tools import TOOLS
from jacobian.math.groups.root_systems.operations import weyl_dominant_representative


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


@pytest.mark.parametrize(
    ("matrix", "weight"),
    [
        (((2, -1), (-1, 2)), (-1, 1)),
        (((2, -2), (-1, 2)), (2, -1)),
        (((2, -1, 0), (-1, 2, 0), (0, 0, 2)), (-1, 2, -3)),
    ],
)
def test_dominant_representative_and_transporter_match_exhaustive_oracle(
    matrix: tuple[tuple[int, ...], ...], weight: tuple[int, ...]
) -> None:
    group = _oracle_group(matrix)
    orbit: dict[tuple[int, ...], set[tuple[tuple[int, ...], ...]]] = {}
    for root_action, action in group:
        image = tuple(
            sum(action[row][column] * weight[column] for column in range(len(weight)))
            for row in range(len(weight))
        )
        orbit.setdefault(image, set()).add(root_action)
    dominant = [
        candidate for candidate in orbit if all(value >= 0 for value in candidate)
    ]
    assert len(dominant) == 1

    result = weyl_dominant_representative(matrix, weight)

    assert result.dominant_weight.coordinates == dominant[0]
    assert result.element.matrix.entries == matrix
    assert result.element.root_action.entries in orbit[dominant[0]]


def test_already_dominant_identity_weight_returns_identity_element() -> None:
    result = weyl_dominant_representative(((2, -1), (-1, 2)), ((1 << 53) - 1, 0))

    assert result.dominant_weight.coordinates == ((1 << 53) - 1, 0)
    assert result.element.root_action.entries == _identity(2)
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_dominant_representative_is_a_published_operation() -> None:
    tool = next(
        tool
        for tool in TOOLS
        if tool.operation_id == "weyl_group.dominant_representative.compute"
    )

    assert tool.request_type.model_validate(
        {"matrix": ((2, -1), (-1, 2)), "weight": (-1, 1)}
    )
