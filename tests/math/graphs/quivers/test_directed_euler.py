"""Circuit evidence checks direction, closure, and every source arrow identity."""

import pytest

from jacobian.math.graphs.quivers import directed_euler_circuit
from jacobian.math.graphs.quivers._eulerian import (
    DirectedEulerCircuit,
    EulerCircuitFound,
)
from jacobian.math.graphs.quivers._models import FiniteQuiver


@pytest.mark.parametrize(
    "arrows",
    [
        (),
        ((0, 0),),
        ((0, 0), (0, 0), (0, 1), (0, 1), (1, 0), (1, 0)),
        ((0, 1), (1, 0)) * 866,
    ],
)
def test_directed_circuit_uses_each_arrow(arrows: tuple[tuple[int, int], ...]) -> None:
    quiver = FiniteQuiver(vertex_count=3, arrows=arrows)
    result = directed_euler_circuit(quiver)
    decoded = DirectedEulerCircuit.model_validate_json(result.model_dump_json())
    assert isinstance(decoded.outcome, EulerCircuitFound)
    tour = decoded.outcome
    assert sorted(tour.arrow_indices) == list(range(len(arrows)))
    vertex = tour.start_vertex
    for index in tour.arrow_indices:
        source, target = arrows[index]
        assert source == vertex
        vertex = target
    assert vertex == tour.start_vertex


@pytest.mark.parametrize(
    "arrows",
    [
        ((0, 1), (0, 2), (1, 2)),
        ((0, 1), (1, 0), (2, 3), (3, 2)),
        ((0, 1), (1, 0), (0, 1)),
    ],
)
def test_non_eulerian_support(arrows: tuple[tuple[int, int], ...]) -> None:
    result = directed_euler_circuit(FiniteQuiver(vertex_count=4, arrows=arrows))
    assert result.outcome.status == "NO_CIRCUIT"
