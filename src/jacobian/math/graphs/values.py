"""Provider-independent values for finite simple undirected graphs."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator

from jacobian.contracts.graph_isomorphism import SimpleUndirectedGraph
from jacobian.contracts.results import ContractModel

GraphCompositionOperation = Literal[
    "DISJOINT_UNION",
    "JOIN",
    "COMPLEMENT",
    "LEXICOGRAPHIC_PRODUCT",
]


class GraphCompositionInput(ContractModel):
    """Two exact graphs and one explicit graph composition operation."""

    operation: GraphCompositionOperation
    left: SimpleUndirectedGraph
    right: SimpleUndirectedGraph | None = None

    @model_validator(mode="after")
    def require_operands(self) -> Self:
        if self.operation == "COMPLEMENT":
            if self.right is not None:
                raise ValueError("complement does not accept a right graph")
        elif self.right is None:
            raise ValueError(f"{self.operation} requires a right graph")
        return self


__all__ = [
    "GraphCompositionInput",
    "GraphCompositionOperation",
    "SimpleUndirectedGraph",
]
