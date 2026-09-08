"""Partial symmetric rational matrices and their chordal completions."""

from itertools import combinations
from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.values import RationalMatrix, SparseRationalMatrixEntry


class PartialSymmetricRationalMatrix(StrictModel):
    """A QQ partial matrix, with graph vertex i bound to matrix axis i.

    Store each specified upper-triangular entry exactly once, in row-major
    order, including every diagonal and every graph edge. Explicit zeros
    are specified values; absent off-diagonal coordinates are unknown.
    """

    domain: Literal["QQ"] = "QQ"
    graph: IndexedSimpleUndirectedGraph
    specified_entries: tuple[SparseRationalMatrixEntry, ...] = Field(max_length=66560)

    @model_validator(mode="after")
    def require_specified_pattern(self) -> Self:
        coordinates = tuple((e.row, e.column) for e in self.specified_entries)
        expected = set(self.graph.edges) | {
            (i, i) for i in range(self.graph.vertex_count)
        }
        if coordinates != tuple(sorted(expected)):
            raise ValueError(
                "specified entries must be row-major sorted diagonals and graph edges"
            )
        return self


class ChordalPSDCompletionRequest(StrictModel):
    matrix: PartialSymmetricRationalMatrix


class ChordalPSDCompletionResult(StrictModel):
    """One exact completion, or a specified non-PSD principal clique.

    The retained source binds the completed axes and specified-entry relation.
    Parsing checks shape, never replays positivity or completion mathematics.
    """

    matrix: PartialSymmetricRationalMatrix
    outcome: Literal["COMPLETED", "INFEASIBLE"]
    completion: RationalMatrix | None = None
    obstruction_clique: tuple[int, ...] = Field(default=(), max_length=1024)

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        n = self.matrix.graph.vertex_count
        if self.outcome == "COMPLETED":
            if self.completion is None or self.obstruction_clique:
                raise ValueError("COMPLETED requires only a completion")
            if (self.completion.row_count, self.completion.column_count) != (n, n):
                raise ValueError("completion must retain the source axes")
        elif self.completion is not None or not self.obstruction_clique:
            raise ValueError("INFEASIBLE requires only an obstruction clique")
        if self.obstruction_clique != tuple(sorted(set(self.obstruction_clique))):
            raise ValueError("obstruction axes must be distinct and sorted")
        if any(not 0 <= i < n for i in self.obstruction_clique):
            raise ValueError("obstruction axes must belong to the source")
        specified_edges = set(self.matrix.graph.edges)
        if any(
            (i, j) not in specified_edges
            for i, j in combinations(self.obstruction_clique, 2)
        ):
            raise ValueError(
                "obstruction axes must form a clique of specified graph edges"
            )
        return self
