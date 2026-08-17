"""Typed contract for bounded exact graph distance matrices."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.graph_optimization._coloring_models import (
    ChromaticGraph,
    GraphVertex,
)

MAX_GRAPH_DISTANCE_MATRIX_ORDER = 64
MAX_GRAPH_DISTANCE_MATRIX_EDGES = 2_016
MAX_GRAPH_DISTANCE = MAX_GRAPH_DISTANCE_MATRIX_ORDER - 1


class GraphDistanceMatrixGraph(ChromaticGraph):
    """A simple graph bounded for polynomial-time all-source BFS replay."""

    vertices: tuple[GraphVertex, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )
    edges: tuple[tuple[GraphVertex, GraphVertex], ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_EDGES
    )


class GraphDistanceMatrixRequest(StrictModel):
    """One complete graph input for an exact distance matrix."""

    graph: GraphDistanceMatrixGraph


GraphDistance = (
    Annotated[
        StrictInt,
        Field(ge=0, le=MAX_GRAPH_DISTANCE),
    ]
    | None
)
GraphDistanceRow = Annotated[
    tuple[GraphDistance, ...],
    Field(max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER),
]


def _validate_distance_matrix_shape(
    vertices: tuple[GraphVertex, ...],
    distances: tuple[GraphDistanceRow, ...],
) -> int:
    order = len(vertices)
    if tuple(sorted(vertices)) != vertices or len(set(vertices)) != order:
        raise ValueError("distance-matrix vertices must be unique and sorted")
    if len(distances) != order or any(len(row) != order for row in distances):
        raise ValueError("distance matrix must be square on the declared vertices")
    return order


def _validate_distance_matrix_diagonal_and_symmetry(
    distances: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    for source in range(order):
        for target in range(order):
            distance = distances[source][target]
            if source == target:
                if distance != 0:
                    raise ValueError("distance-matrix diagonal must be zero")
            elif distance == 0:
                raise ValueError("off-diagonal distances must be positive or null")
            if distance != distances[target][source]:
                raise ValueError("undirected distance matrix must be symmetric")


def _validate_distance_matrix_triangle_inequality(
    distances: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    for source in range(order):
        for intermediate in range(order):
            left = distances[source][intermediate]
            if left is None:
                continue
            for target in range(order):
                right = distances[intermediate][target]
                if right is None:
                    continue
                direct = distances[source][target]
                if direct is None or direct > left + right:
                    raise ValueError(
                        "finite distances must satisfy component closure and "
                        "the triangle inequality"
                    )


class GraphDistanceMatrixResult(StrictModel):
    """All exact unweighted shortest-path distances in canonical vertex order."""

    semantics_version: Literal["unweighted-shortest-path-distance-matrix.v1"]
    vertex_ordering: Literal["LEXICOGRAPHIC_ASCENDING"]
    pair_coverage: Literal["ALL_ORDERED_VERTEX_PAIRS"]
    unreachable_representation: Literal["JSON_NULL"]
    vertices: tuple[GraphVertex, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )
    distances: tuple[GraphDistanceRow, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )
    connected: StrictBool

    @model_validator(mode="after")
    def bind_complete_metric(self) -> Self:
        order = _validate_distance_matrix_shape(self.vertices, self.distances)
        _validate_distance_matrix_diagonal_and_symmetry(self.distances, order)
        _validate_distance_matrix_triangle_inequality(self.distances, order)
        expected_connected = order > 0 and all(
            distance is not None for row in self.distances for distance in row
        )
        if self.connected != expected_connected:
            raise ValueError("connected must match all-pairs finite reachability")
        return self


__all__ = [
    "MAX_GRAPH_DISTANCE",
    "MAX_GRAPH_DISTANCE_MATRIX_EDGES",
    "MAX_GRAPH_DISTANCE_MATRIX_ORDER",
    "GraphDistance",
    "GraphDistanceMatrixGraph",
    "GraphDistanceMatrixRequest",
    "GraphDistanceMatrixResult",
    "GraphDistanceRow",
]
