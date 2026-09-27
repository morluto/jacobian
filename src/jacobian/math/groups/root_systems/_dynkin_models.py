"""Typed exact values for finite Dynkin diagrams."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    FiniteCartanDatum,
    _validation_error,
)

MAX_DYNKIN_EDGES = MAX_RANK * (MAX_RANK - 1) // 2
MAX_DYNKIN_DIAGRAM_OUTPUT_CELLS = 16_384


class DynkinEdge(StrictModel):
    """One Cartan-labeled edge, oriented by the ordered simple-root axis.

    ``cartan_pairing`` is ``(A[i,j], A[j,i])`` for the increasing node pair
    ``(i,j)``. This ordered pair preserves the arrow direction as well as the
    edge multiplicity of a multiple bond.
    """

    simple_root_indices: tuple[StrictInt, StrictInt]
    cartan_pairing: tuple[StrictInt, StrictInt]
    edge_multiplicity: StrictInt = Field(ge=1, le=3)

    @model_validator(mode="after")
    def require_edge_order_and_labels(self) -> Self:
        first, second = self.simple_root_indices
        left, right = self.cartan_pairing
        if (
            first < 0
            or first >= second
            or left >= 0
            or right >= 0
            or left * right != self.edge_multiplicity
        ):
            raise _validation_error(
                "dynkin_edge_shape",
                "Dynkin edges require ordered distinct nodes, negative Cartan entries, and matching multiplicity",
            )
        return self


class FiniteDynkinDiagram(StrictModel):
    """Finite Dynkin graph retaining its exact ordered Cartan datum."""

    datum: FiniteCartanDatum
    simple_root_axis: tuple[StrictInt, ...] = Field(min_length=1, max_length=MAX_RANK)
    edges: tuple[DynkinEdge, ...] = Field(max_length=MAX_DYNKIN_EDGES)

    @model_validator(mode="after")
    def require_complete_cartan_labeled_graph(self) -> Self:
        matrix = self.datum.cartan_matrix.entries
        rank = len(matrix)
        expected_axis = tuple(range(rank))
        expected_pairs = tuple(
            (i, j)
            for i in range(rank)
            for j in range(i + 1, rank)
            if matrix[i][j] != 0 or matrix[j][i] != 0
        )
        observed_pairs = tuple(edge.simple_root_indices for edge in self.edges)
        if (
            self.simple_root_axis != expected_axis
            or observed_pairs != tuple(sorted(set(observed_pairs)))
            or observed_pairs != expected_pairs
            or any(
                edge.cartan_pairing != (matrix[i][j], matrix[j][i])
                or edge.edge_multiplicity != matrix[i][j] * matrix[j][i]
                for edge in self.edges
                for i, j in (edge.simple_root_indices,)
            )
        ):
            raise _validation_error(
                "dynkin_diagram_consistency",
                "diagram nodes and labeled edges must exactly reconstruct the datum's Cartan adjacency",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        datum: FiniteCartanDatum,
        edges: tuple[DynkinEdge, ...],
    ) -> Self:
        rank = len(datum.cartan_matrix)
        return cls.model_construct(
            datum=datum,
            simple_root_axis=tuple(range(rank)),
            edges=edges,
        )
