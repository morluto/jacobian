"""Exact clique-supported sparse PSD decompositions."""

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph
from jacobian.math.matrices.values import RationalMatrix


class ChordalPSDRequest(StrictModel):
    """A fully specified symmetric QQ matrix on graph vertices 0,...,n-1.

    The graph must be chordal and cover all nonzero off-diagonal entries.
    Extra graph edges are allowed; matrix zeros remain specified zeros.
    """

    matrix: RationalMatrix
    graph: IndexedSimpleUndirectedGraph


class CliquePSDTerm(StrictModel):
    """Local matrix with ascending original-axis indices defining its inclusion.

    PSD is established by the producing operation, not by structural parsing.
    """

    axes: tuple[Annotated[int, Field(ge=0, le=1023)], ...] = Field(
        min_length=1, max_length=1024
    )
    matrix: RationalMatrix

    @model_validator(mode="after")
    def require_shape(self) -> Self:
        if tuple(sorted(set(self.axes))) != self.axes or self.axes[0] < 0:
            raise ValueError("axes must be distinct ascending nonnegative indices")
        if self.matrix.row_count != len(self.axes) or self.matrix.column_count != len(
            self.axes
        ):
            raise ValueError("local matrix dimensions must equal the axis count")
        return self


class ChordalPSDDecomposition(StrictModel):
    """Sum of clique-supported PSD terms on the retained source axes.

    Embedding and summing all local matrices reproduces matrix exactly. Terms
    are in deterministic elimination order; zero pivots contribute no term.
    The zero matrix (including order zero) has an empty tuple of terms.
    """

    matrix: RationalMatrix
    graph: IndexedSimpleUndirectedGraph
    terms: tuple[CliquePSDTerm, ...] = Field(max_length=1024)

    @model_validator(mode="after")
    def require_axes(self) -> Self:
        n = self.graph.vertex_count
        if self.matrix.row_count != n or self.matrix.column_count != n:
            raise ValueError("source matrix dimensions must equal graph vertex count")
        if any(axis >= n for term in self.terms for axis in term.axes):
            raise ValueError("term axes must belong to the source matrix")
        return self
