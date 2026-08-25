"""Typed contract for bounded exact graph distance matrices."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._coloring_models import (
    GraphVertex,
    PolynomialTimeGraph,
)

MAX_GRAPH_DISTANCE_MATRIX_ORDER = 256
MAX_GRAPH_DISTANCE_MATRIX_EDGES = 32_640
MAX_GRAPH_DISTANCE = MAX_GRAPH_DISTANCE_MATRIX_ORDER - 1


class GraphDistanceMatrixGraph(PolynomialTimeGraph):
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


class GraphDistanceRow(StrictModel):
    """One labelled row of the canonical vertex-ordered distance matrix.

    ``source`` names the row's vertex and the ``distances`` cells are the
    distances from ``source`` to the result ``vertices`` in their declared
    order. Binding the label into every row keeps a dense matrix attached to
    its authoritative vertex axis, so numeric-looking identifiers such as
    ``"2"`` and ``"10"`` cannot be silently presented under a different
    natural-number ordering.
    """

    source: GraphVertex
    distances: tuple[GraphDistance, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )


def _validate_distance_matrix_shape(
    vertices: tuple[GraphVertex, ...],
    rows: tuple[GraphDistanceRow, ...],
) -> int:
    order = len(vertices)
    if tuple(sorted(vertices)) != vertices or len(set(vertices)) != order:
        raise PydanticCustomError(
            "graph.distance_matrix_vertices_must_be_unique_and_sort",
            "distance-matrix vertices must be unique and sorted",
        )
    if len(rows) != order:
        raise PydanticCustomError(
            "graph.distance_matrix_must_declare_one_labelled_row_pe",
            "distance matrix must declare one labelled row per vertex",
        )
    for index, row in enumerate(rows):
        if row.source != vertices[index]:
            raise PydanticCustomError(
                "graph.every_distance_matrix_row_carry_canonical_vertex",
                "every distance-matrix row must carry the canonical vertex "
                "label at its position",
            )
        if len(row.distances) != order:
            raise PydanticCustomError(
                "graph.distance_matrix_must_be_square_on_the_declared_v",
                "distance matrix must be square on the declared vertices",
            )
    return order


def _validate_distance_matrix_diagonal_and_symmetry(
    rows: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    for source in range(order):
        row = rows[source].distances
        for target in range(order):
            distance = row[target]
            if source == target:
                if distance != 0:
                    raise PydanticCustomError(
                        "graph.distance_matrix_diagonal_must_be_zero",
                        "distance-matrix diagonal must be zero",
                    )
            elif distance == 0:
                raise PydanticCustomError(
                    "graph.off_diagonal_distances_must_be_positive_or_null",
                    "off-diagonal distances must be positive or null",
                )
            if distance != rows[target].distances[source]:
                raise PydanticCustomError(
                    "graph.undirected_distance_matrix_must_be_symmetric",
                    "undirected distance matrix must be symmetric",
                )


def _validate_distance_matrix_triangle_inequality(
    rows: tuple[GraphDistanceRow, ...],
    order: int,
) -> None:
    matrix = tuple(row.distances for row in rows)
    for source in range(order):
        for intermediate in range(order):
            left = matrix[source][intermediate]
            if left is None:
                continue
            for target in range(order):
                right = matrix[intermediate][target]
                if right is None:
                    continue
                direct = matrix[source][target]
                if direct is None or direct > left + right:
                    raise PydanticCustomError(
                        "graph.finite_distances_satisfy_component_closure_triangle_inequality",
                        "finite distances must satisfy component closure and "
                        "the triangle inequality",
                    )


class GraphDistanceMatrixResult(StrictModel):
    """All exact unweighted shortest-path distances in canonical vertex order.

    ``vertices`` declares the canonical column axis and every row carries its
    own ``source`` label, so the dense positional matrix stays bound to the
    authoritative vertex sequence.
    """

    vertex_ordering: Literal["LEXICOGRAPHIC_ASCENDING"]
    pair_coverage: Literal["ALL_ORDERED_VERTEX_PAIRS"]
    unreachable_representation: Literal["JSON_NULL"]
    vertices: tuple[GraphVertex, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )
    rows: tuple[GraphDistanceRow, ...] = Field(
        max_length=MAX_GRAPH_DISTANCE_MATRIX_ORDER
    )
    connected: StrictBool

    @model_validator(mode="after")
    def bind_complete_metric(self) -> Self:
        order = _validate_distance_matrix_shape(self.vertices, self.rows)
        _validate_distance_matrix_diagonal_and_symmetry(self.rows, order)
        _validate_distance_matrix_triangle_inequality(self.rows, order)
        expected_connected = order > 0 and all(
            distance is not None for row in self.rows for distance in row.distances
        )
        if self.connected != expected_connected:
            raise PydanticCustomError(
                "graph.connected_must_match_all_pairs_finite_reachabili",
                "connected must match all-pairs finite reachability",
            )
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
