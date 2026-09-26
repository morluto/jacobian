"""Exact parabolic weight orbits compared with independent subgroup closure."""

from collections import deque

import pytest

from jacobian.math.groups.root_systems._models import WeylParabolicWeightOrbitRequest
from jacobian.math.groups.root_systems.operations import (
    weight_lattice_vector,
    weyl_parabolic_weight_orbit,
)

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


def _independent_parabolic_orbit(
    generators: tuple[Matrix, ...], weight: Vector
) -> tuple[Vector, ...]:
    rank = len(weight)
    identity = tuple(
        tuple(int(row == column) for column in range(rank)) for row in range(rank)
    )
    elements = {identity}
    pending = deque([identity])
    while pending:
        element = pending.popleft()
        for generator in generators:
            product = _matmul(generator, element)
            if product not in elements:
                elements.add(product)
                pending.append(product)
    return tuple(sorted({_apply(element, weight) for element in elements}))


@pytest.mark.parametrize(
    ("cartan", "weight", "indices", "reflection_matrices"),
    (
        (
            ((2, -1), (-1, 2)),
            (1, 0),
            (0,),
            (((-1, 0), (1, 1)),),
        ),
        (
            ((2, -1), (-1, 2)),
            (1, 0),
            (0, 1),
            (((-1, 0), (1, 1)), ((1, 1), (0, -1))),
        ),
        (
            ((2, -2), (-1, 2)),
            (1, 0),
            (1,),
            (((1, 2), (0, -1)),),
        ),
        (
            ((2, -1), (-1, 2)),
            (2, -1),
            (),
            (),
        ),
    ),
)
def test_parabolic_orbits_match_independent_matrix_subgroup_closure(
    cartan: Matrix,
    weight: Vector,
    indices: tuple[int, ...],
    reflection_matrices: tuple[Matrix, ...],
) -> None:
    vector = weight_lattice_vector(cartan, weight)
    request = WeylParabolicWeightOrbitRequest(
        weight=vector, simple_root_indices=indices
    )

    result = weyl_parabolic_weight_orbit(request)

    assert result.orbit == _independent_parabolic_orbit(reflection_matrices, weight)
    assert result.simple_root_indices == indices
    assert result.weight == weight
    assert result.datum.cartan_matrix.entries == cartan
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_noncanonical_parabolic_index_order_is_rejected() -> None:
    vector = weight_lattice_vector(((2, -1), (-1, 2)), (1, 0))
    with pytest.raises(ValueError, match="strictly increasing subset"):
        WeylParabolicWeightOrbitRequest(weight=vector, simple_root_indices=(1, 0))
