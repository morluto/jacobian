"""Partial symmetric rational matrices and their chordal completions."""

from itertools import combinations
from typing import Annotated, Literal, Self

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


class CompletedChordalPSDCompletion(StrictModel):
    """A completed rational PSD matrix on the retained source axes."""

    status: Literal["COMPLETED"] = "COMPLETED"
    completion: RationalMatrix


class InfeasibleChordalPSDCompletion(StrictModel):
    """A specified non-PSD principal clique of the retained source pattern."""

    status: Literal["INFEASIBLE"] = "INFEASIBLE"
    obstruction_clique: tuple[int, ...] = Field(min_length=1, max_length=1024)


ChordalPSDCompletionOutcome = Annotated[
    CompletedChordalPSDCompletion | InfeasibleChordalPSDCompletion,
    Field(discriminator="status"),
]


class ChordalPSDCompletionResult(StrictModel):
    """One exact completion, or a specified non-PSD principal clique.

    The retained source binds the completed axes and specified-entry relation.
    Parsing checks shape, never replays positivity or completion mathematics.
    """

    matrix: PartialSymmetricRationalMatrix
    outcome: ChordalPSDCompletionOutcome

    @model_validator(mode="after")
    def require_result_shape(self) -> Self:
        n = self.matrix.graph.vertex_count
        if isinstance(self.outcome, CompletedChordalPSDCompletion):
            completion = self.outcome.completion
            if (completion.row_count, completion.column_count) != (n, n):
                raise ValueError("completion must retain the source axes")
            return self
        clique = self.outcome.obstruction_clique
        if clique != tuple(sorted(set(clique))):
            raise ValueError("obstruction axes must be distinct and sorted")
        if any(not 0 <= i < n for i in clique):
            raise ValueError("obstruction axes must belong to the source")
        specified_edges = set(self.matrix.graph.edges)
        if any((i, j) not in specified_edges for i, j in combinations(clique, 2)):
            raise ValueError(
                "obstruction axes must form a clique of specified graph edges"
            )
        return self
