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
from jacobian.math.graphs.values import LoopedSimpleGraph


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
                    else tuple(
                        sorted((self.graph.vertices[i], self.graph.vertices[j]))
                    )
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

    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )

    if type(graph) is not LoopedSimpleGraph:
        raise OperationDomainValidationError(
            location=("graph",), code="graph.looped_graph_invalid",
            message="graph must be a canonical LoopedSimpleGraph value",
        )
    try:
        canonical = LoopedSimpleGraph.model_validate(graph.model_dump())
    except (ValidationError, AttributeError) as error:
        raise OperationDomainValidationError(
            location=("graph",), code="graph.looped_graph_invalid",
            message="graph must be a canonical LoopedSimpleGraph value",
        ) from error
    if len(canonical.vertices) > MAX_BINARY_GROUND:
        raise OperationResourceAdmissionError(
            location=("graph", "vertices"), code="delta_matroid.binary_work",
            message=f"looped graph conversion supports at most {MAX_BINARY_GROUND} vertices",
        )
    return canonical


__all__ = ["LoopedGraphDeltaMatroidRequest", "LoopedGraphDeltaMatroidResult", "admit_looped_graph"]
