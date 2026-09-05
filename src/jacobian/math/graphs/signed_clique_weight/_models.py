"""Typed contracts for signed clique-weight maximization."""

from __future__ import annotations

from typing import Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.math.graphs.optimization._models import RationalWeightedGraph

# A clique lies inside one biconnected block, so blocks are optimized
# independently and the best (value, witness) pair wins globally.


class SignedCliqueWeightRequest(StrictModel):
    """Request the maximum signed edge-weight over nontrivial cliques."""

    graph: RationalWeightedGraph


class SignedCliqueWeightResult(StrictModel):
    """Maximum induced edge-weight over cliques of order at least two.

    ``optimum_value`` is None with an empty clique exactly when the graph
    has no edges: an edgeless graph admits no eligible clique and the
    operation reports that explicitly instead of inventing maximum zero.
    Otherwise the attaining clique is bound in source-vertex order with
    ties broken toward the lexicographically least witness.
    """

    graph: RationalWeightedGraph
    optimum_value: CanonicalRational | None = None
    clique: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_coherent_optimum(self) -> Self:
        if (self.optimum_value is None) != (not self.clique):
            raise PydanticCustomError(
                "graph.signed_clique_weight.optimum_binding",
                "a missing optimum pairs exactly with an empty clique",
            )
        witness = set(self.clique)
        if len(witness) != len(self.clique):
            raise PydanticCustomError(
                "graph.signed_clique_weight.clique_distinct",
                "clique vertices must be distinct",
            )
        if self.clique != tuple(
            vertex for vertex in self.graph.vertices if vertex in witness
        ):
            raise PydanticCustomError(
                "graph.signed_clique_weight.clique_axis",
                "clique vertices must use source-vertex order",
            )
        if self.clique and len(self.clique) < 2:
            raise PydanticCustomError(
                "graph.signed_clique_weight.clique_trivial",
                "a reported clique has order at least two",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: RationalWeightedGraph,
        optimum_value: CanonicalRational | None,
        clique: tuple[str, ...],
    ) -> Self:
        """Construct an optimum established by the owner-local kernel."""

        return cls.model_construct(
            graph=graph, optimum_value=optimum_value, clique=clique
        )


__all__ = [
    "SignedCliqueWeightRequest",
    "SignedCliqueWeightResult",
]
