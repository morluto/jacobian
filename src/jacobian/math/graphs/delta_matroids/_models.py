"""Contracts for exact graph and delta-matroid interoperability."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.extra import (
    MAX_BINARY_GROUND,
    BinarySymmetricMatrix,
)
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid
from jacobian.math.graphs.values import MAX_GRAPH_LABEL_BYTES, LoopedSimpleGraph


class LoopedGraphDeltaMatroidRequest(StrictModel):
    """Request using the canonical looped graph encoding."""

    graph: LoopedSimpleGraph = Field(
        description="Canonical looped graph: unique nonempty NFC labels (<=64 UTF-8 bytes), unique declared off-diagonal edges oriented left < right, and unique declared loop labels. Computational admission is at most eight vertices."
    )


class LoopedGraphDeltaMatroidResult(StrictModel):
    graph: LoopedSimpleGraph
    matrix: BinarySymmetricMatrix
    delta_matroid: FiniteDeltaMatroid

    @model_validator(mode="after")
    def require_source_axes_and_adjacency(self) -> Self:
        n = len(self.graph.vertices)
        if (
            self.matrix.ground != self.graph.vertices
            or self.delta_matroid.ground != self.graph.vertices
        ):
            raise PydanticCustomError(
                "delta_matroid.graph_result_ground",
                "matrix and delta-matroid axes must equal the graph vertex axis",
            )
        edge_set = set(self.graph.edges)
        loop_set = set(self.graph.loops)
        expected = tuple(
            tuple(
                int(
                    (self.graph.vertices[i] in loop_set)
                    if i == j
                    else tuple(sorted((self.graph.vertices[i], self.graph.vertices[j])))
                    in edge_set
                )
                for j in range(n)
            )
            for i in range(n)
        )
        if self.matrix.entries != expected:
            raise PydanticCustomError(
                "delta_matroid.graph_result_matrix",
                "matrix entries must be the looped graph adjacency matrix",
            )
        return self


def admit_looped_graph(graph: LoopedSimpleGraph) -> LoopedSimpleGraph:
    """Revalidate the canonical carrier and enforce the operation work bound."""
    from pydantic import ValidationError
    from pydantic_core import PydanticSerializationError

    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )

    if type(graph) is not LoopedSimpleGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.looped_graph_invalid",
            message="graph must be a canonical LoopedSimpleGraph value",
        )
    # Preflight the raw container before serializing. A native caller can
    # bypass Pydantic with model_construct, and serializing an oversized or
    # malformed payload would copy the whole vertex and edge graph before the
    # O(1) length check below could refuse it.
    raw_vertices = getattr(graph, "vertices", None)
    if type(raw_vertices) is not tuple:
        raise OperationDomainValidationError(
            location=("graph", "vertices"),
            code="graph.looped_graph_invalid",
            message="graph vertices must be a canonical tuple of labels",
        )
    if len(raw_vertices) > MAX_BINARY_GROUND:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"),
            code="delta_matroid.binary_work",
            message=f"looped graph conversion supports at most {MAX_BINARY_GROUND} vertices",
        )
    # All serialized storage, including malformed nested rows and labels, must
    # be bounded. These cardinalities follow from a simple graph on this axis;
    # they do not restrict any canonical graph admitted by the vertex envelope.
    n = len(raw_vertices)

    def bounded_container(
        field: str, maximum: int
    ) -> tuple[object, ...] | list[object]:
        values = getattr(graph, field, None)
        if (type(values) is not tuple and type(values) is not list) or len(
            values
        ) > maximum:
            raise OperationDomainValidationError(
                location=("graph", field),
                code="graph.looped_graph_invalid",
                message=f"graph {field} must fit the canonical vertex axis",
            )
        return values

    raw_edges = bounded_container("edges", n * (n - 1) // 2)
    raw_loops = bounded_container("loops", n)
    labels = [*raw_vertices, *raw_loops]
    for index, edge in enumerate(raw_edges):
        if (type(edge) is not tuple and type(edge) is not list) or len(edge) != 2:
            raise OperationDomainValidationError(
                location=("graph", "edges", index),
                code="graph.looped_graph_invalid",
                message="graph edges must contain pairs of labels",
            )
        labels.extend(edge)
    for label in labels:
        if type(label) is not str or not 0 < len(label) <= MAX_GRAPH_LABEL_BYTES:
            raise OperationDomainValidationError(
                location=("graph",),
                code="graph.looped_graph_invalid",
                message="graph labels must be bounded canonical strings",
            )
    try:
        canonical = LoopedSimpleGraph.model_validate(graph.model_dump())
    except (
        ValidationError,
        AttributeError,
        TypeError,
        ValueError,
        PydanticSerializationError,
    ) as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.looped_graph_invalid",
            message="graph must be a canonical LoopedSimpleGraph value",
        ) from error
    if len(canonical.vertices) > MAX_BINARY_GROUND:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"),
            code="delta_matroid.binary_work",
            message=f"looped graph conversion supports at most {MAX_BINARY_GROUND} vertices",
        )
    return canonical


__all__ = [
    "LoopedGraphDeltaMatroidRequest",
    "LoopedGraphDeltaMatroidResult",
    "admit_looped_graph",
]
