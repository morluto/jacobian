"""Exact integral weight orbits, checked against finite matrix enumeration."""

from collections import deque

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems._models import MAX_WEIGHT_ORBIT_SIZE
from jacobian.math.groups.root_systems.operations import weyl_weight_orbit

Matrix = tuple[tuple[int, ...], ...]
Vector = tuple[int, ...]


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    size = len(left)
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _apply(matrix: Matrix, vector: Vector) -> Vector:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(len(vector)))
        for row in range(len(vector))
    )


def _independent_group_orbit(
    generators: tuple[Matrix, ...], weight: Vector
) -> tuple[Vector, ...]:
    """Enumerate distinct reflection matrices, independently of weight BFS."""
    rank = len(weight)
    identity = tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )
    group = {identity}
    pending = deque([identity])
    while pending:
        element = pending.popleft()
        for generator in generators:
            product = _matmul(generator, element)
            if product not in group:
                group.add(product)
                pending.append(product)
    return tuple(sorted({_apply(element, weight) for element in group}))


@pytest.mark.parametrize(
    ("name", "cartan", "generators", "weight", "group_order", "expected"),
    (
        (
            "A2",
            ((2, -1), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 1), (0, -1))),
            (1, 0),
            6,
            ((-1, 1), (0, -1), (1, 0)),
        ),
        (
            "B2",
            ((2, -2), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 2), (0, -1))),
            (1, 0),
            8,
            ((-1, 0), (-1, 1), (1, -1), (1, 0)),
        ),
        (
            "G2",
            ((2, -3), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 3), (0, -1))),
            (1, 0),
            12,
            ((-2, 1), (-1, 0), (-1, 1), (1, -1), (1, 0), (2, -1)),
        ),
    ),
)
def test_small_type_orbits_match_independent_group_matrix_enumeration(
    name: str,
    cartan: Matrix,
    generators: tuple[Matrix, ...],
    weight: Vector,
    group_order: int,
    expected: tuple[Vector, ...],
) -> None:
    del name
    actual = weyl_weight_orbit(cartan, weight)
    oracle = _independent_group_orbit(generators, weight)
    assert len(_independent_group_orbit(generators, (1, 1))) == group_order
    assert actual.orbit == expected == oracle
    assert actual.weight in actual.orbit


@pytest.mark.parametrize(
    ("cartan", "generators", "weights"),
    (
        (
            ((2, -1), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 1), (0, -1))),
            ((0, 0), (2, -1), (-3, 2), (1, 1)),
        ),
        (
            ((2, -2), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 2), (0, -1))),
            ((0, 0), (0, 1), (-2, 3)),
        ),
        (
            ((2, -3), (-1, 2)),
            (((-1, 0), (1, 1)), ((1, 3), (0, -1))),
            ((0, 0), (0, 1), (-2, 3)),
        ),
    ),
)
def test_varied_weights_match_oracle_and_orbit_stabilizer(
    cartan: Matrix, generators: tuple[Matrix, ...], weights: tuple[Vector, ...]
) -> None:
    for weight in weights:
        actual = weyl_weight_orbit(cartan, weight)
        assert actual.orbit == _independent_group_orbit(generators, weight)


def test_large_group_with_small_weight_orbit_is_admitted() -> None:
    # |W(A8)| = 9!, but a fundamental weight has only 9 orbit elements.
    a8 = tuple(
        tuple(2 if i == j else (-1 if abs(i - j) == 1 else 0) for j in range(8))
        for i in range(8)
    )
    result = weyl_weight_orbit(a8, (1, 0, 0, 0, 0, 0, 0, 0))
    assert len(result.orbit) == 9
    assert len(set(result.orbit)) == 9


def test_orbit_cardinality_is_rejected_before_expansion() -> None:
    a8 = tuple(
        tuple(2 if i == j else (-1 if abs(i - j) == 1 else 0) for j in range(8))
        for i in range(8)
    )
    regular = (1, 1, 1, 1, 1, 1, 1, 1)
    with pytest.raises(OperationDomainValidationError, match="complete weight orbit"):
        weyl_weight_orbit(a8, regular)
    assert MAX_WEIGHT_ORBIT_SIZE == 4096


def test_nonintegral_and_wrong_rank_weights_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError, match="fundamental-weight"):
        weyl_weight_orbit(((2, -1), (-1, 2)), (1,))
    with pytest.raises(OperationDomainValidationError, match="fundamental-weight"):
        weyl_weight_orbit(((2, -1), (-1, 2)), (1, 0.5))  # type: ignore[arg-type]
