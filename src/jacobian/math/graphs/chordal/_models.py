"""Typed contracts for chordal graph recognition."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import (
    MAX_INDEXED_SIMPLE_GRAPH_VERTICES,
    SimpleUndirectedGraph,
)

# Recognition (MCS ordering plus perfect-elimination verification) and
# certificate extraction (triple-indexed shortest-path searches) are charged
# as independent phases: dense chordal graphs pass phase one cheaply while
# only sparse-enough failures reach phase two.
MAX_CHORDAL_ORDER_WORK = 16_777_216
MAX_CHORDAL_CERTIFICATE_WORK = 500_000_000


def _validation_error(code: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(code, message)


ChordalStatus = Literal["CHORDAL", "NONCHORDAL"]


class ChordalRecognitionRequest(StrictModel):
    """Request chordal recognition of a finite simple graph."""

    graph: SimpleUndirectedGraph


class ChordalRecognitionResult(StrictModel):
    """Chordality with a structural certificate in each direction.

    CHORDAL carries a perfect elimination ordering: a permutation of all
    vertices where every vertex's later neighbors form a clique. NONCHORDAL
    carries an ordered induced cycle of length at least four, rotated to
    start at its smallest label with the smaller second endpoint.
    """

    graph: SimpleUndirectedGraph
    status: ChordalStatus
    elimination_ordering: tuple[str, ...] = ()
    induced_cycle: tuple[str, ...] = ()

    @model_validator(mode="after")
    def require_certificate_shape(self) -> Self:
        vertices = self.graph.vertices
        if self.status == "CHORDAL":
            if self.induced_cycle:
                raise _validation_error(
                    "graph.chordal.unexpected_cycle",
                    "a chordal result carries no induced cycle",
                )
            if set(self.elimination_ordering) != set(vertices) or len(
                self.elimination_ordering
            ) != len(vertices):
                raise _validation_error(
                    "graph.chordal.ordering_not_permutation",
                    "the elimination ordering must permute all vertices",
                )
        else:
            if self.elimination_ordering:
                raise _validation_error(
                    "graph.chordal.unexpected_ordering",
                    "a nonchordal result carries no elimination ordering",
                )
            cycle = self.induced_cycle
            if len(cycle) < 4 or len(set(cycle)) != len(cycle):
                raise _validation_error(
                    "graph.chordal.cycle_shape",
                    "the induced cycle needs at least four distinct vertices",
                )
            if len(cycle) > MAX_INDEXED_SIMPLE_GRAPH_VERTICES:
                raise _validation_error(
                    "graph.chordal.cycle_shape",
                    "the induced cycle cannot exceed the graph vertex bound",
                )
            unknown = set(cycle) - set(vertices)
            if unknown:
                raise _validation_error(
                    "graph.chordal.cycle_vertex_unknown",
                    "the induced cycle must use declared graph vertices",
                )
            if cycle[0] != min(cycle):
                raise _validation_error(
                    "graph.chordal.cycle_canonical_start",
                    "the induced cycle must start at its smallest label",
                )
            if not cycle[1] < cycle[-1]:
                raise _validation_error(
                    "graph.chordal.cycle_canonical_direction",
                    "the induced cycle must use the smaller second endpoint",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        graph: SimpleUndirectedGraph,
        status: ChordalStatus,
        elimination_ordering: tuple[str, ...] = (),
        induced_cycle: tuple[str, ...] = (),
    ) -> Self:
        """Construct a recognition verdict established by the owner kernel."""

        return cls.model_construct(
            graph=graph,
            status=status,
            elimination_ordering=elimination_ordering,
            induced_cycle=induced_cycle,
        )


__all__ = [
    "MAX_CHORDAL_CERTIFICATE_WORK",
    "MAX_CHORDAL_ORDER_WORK",
    "ChordalRecognitionRequest",
    "ChordalRecognitionResult",
    "ChordalStatus",
]
