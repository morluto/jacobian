"""Exact integral weight orbits, checked against finite matrix enumeration."""

from collections import deque

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.groups.root_systems import operations as rs_operations
from jacobian.math.groups.root_systems._cartan import (
    cartan_type_matrix,
    simple_reflection,
)
from jacobian.math.groups.root_systems._models import (
    MAX_POSITIVE_ROOTS,
    MAX_WEIGHT_ORBIT_SIZE,
)
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
    with pytest.raises(OperationDomainValidationError) as exc_info:
        weyl_weight_orbit(a8, regular)
    assert exc_info.value.errors()[0]["type"] == "root_system.weight_orbit_size_bound"
    assert MAX_WEIGHT_ORBIT_SIZE == 4096


def test_nonintegral_and_wrong_rank_weights_are_rejected() -> None:
    with pytest.raises(OperationDomainValidationError) as exc_info:
        weyl_weight_orbit(((2, -1), (-1, 2)), (1,))
    assert exc_info.value.errors()[0]["type"] == "root_system.invalid_integral_weight"
    with pytest.raises(OperationDomainValidationError) as exc_info:
        weyl_weight_orbit(((2, -1), (-1, 2)), (1, 0.5))  # type: ignore[arg-type]
    assert exc_info.value.errors()[0]["type"] == "root_system.invalid_integral_weight"


@pytest.mark.parametrize(
    ("cartan", "generators"),
    [
        (((2, -1), (-1, 2)), (((-1, 0), (1, 1)), ((1, 1), (0, -1)))),
        (((2, -2), (-1, 2)), (((-1, 0), (1, 1)), ((1, 2), (0, -1)))),
        (((2, -3), (-1, 2)), (((-1, 0), (1, 1)), ((1, 3), (0, -1)))),
        (((2, -1), (-3, 2)), (((-1, 0), (3, 1)), ((1, 1), (0, -1)))),
    ],
)
@pytest.mark.parametrize("indices", [(), (0,), (1,), (0, 1)])
def test_dual_preflight_matches_independent_weight_action_matrices(
    cartan: Matrix, generators: tuple[Matrix, ...], indices: tuple[int, ...]
) -> None:
    for weight in ((0, 0), (1, 1), (2, -1), (-3, 2)):
        orbit = _independent_group_orbit(tuple(generators[i] for i in indices), weight)
        expected = tuple(max(abs(image[j]) for image in orbit) for j in range(2))

        assert (
            rs_operations._admit_weight_orbit_coordinate_bounds(cartan, weight, indices)
            == expected
        )


def test_e8_dual_preflight_has_a_fixed_work_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = cartan_type_matrix("E", 8)
    indices = tuple(range(8))
    reflection_count = 0

    def recording_reflect(root: Vector, index: int, matrix: Matrix) -> Vector:
        nonlocal reflection_count
        reflection_count += 1
        return simple_reflection(root, index, matrix)

    monkeypatch.setattr(rs_operations, "_simple_reflection_kernel", recording_reflect)
    bounds = rs_operations._admit_weight_orbit_coordinate_bounds(
        rows, (1,) * 8, indices
    )

    # rho pairs with a coroot's height; E8 has highest coroot height h-1=29.
    assert bounds == (29,) * 8
    assert 0 < reflection_count <= len(rows) * 2 * MAX_POSITIVE_ROOTS * len(indices)
