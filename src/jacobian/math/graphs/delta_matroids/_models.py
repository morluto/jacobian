"""Contracts for exact graph and delta-matroid interoperability."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.combinatorics.matroids.delta.extra import (
    MAX_BINARY_GROUND,
    BinarySymmetricMatrix,
)
from jacobian.math.combinatorics.matroids.delta.values import FiniteDeltaMatroid
from jacobian.math.graphs.values import LoopedSimpleGraph


class LoopedGraphDeltaMatroidRequest(StrictModel):
    graph: LoopedSimpleGraph

    @model_validator(mode="after")
    def require_binary_envelope(self) -> Self:
        if len(self.graph.vertices) > MAX_BINARY_GROUND:
            raise PydanticCustomError(
                "delta_matroid.binary_work",
                f"looped graph conversion supports at most {MAX_BINARY_GROUND} vertices",
            )
        return self


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


__all__ = ["LoopedGraphDeltaMatroidRequest", "LoopedGraphDeltaMatroidResult"]
