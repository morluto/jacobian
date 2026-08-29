"""Pseudomanifold decision contracts and decision kernel."""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    FiniteSimplicialComplex,
    Simplex,
    SimplicialComplexRequest,
    _validation_error,
)


class _PseudomanifoldDecision(NamedTuple):
    """The canonical decision for one finite simplicial complex."""

    is_pseudomanifold: bool
    is_closed: bool
    obstruction: str | None
    dimension: int
    num_facets: int


def pseudomanifold_decision(facets: tuple[Simplex, ...]) -> _PseudomanifoldDecision:
    """Return the exact codimension-one incidence decision for ``facets``."""

    facet_sets = [frozenset(facet) for facet in facets]
    dimension = max((len(facet) - 1 for facet in facet_sets), default=0)
    num_facets = len(facet_sets)
    is_pure = all(len(facet) - 1 == dimension for facet in facet_sets)
    if not is_pure:
        return _PseudomanifoldDecision(
            False,
            False,
            "not pure: facets have different dimensions",
            dimension,
            num_facets,
        )

    incidence: dict[frozenset[str], int] = {}
    for facet in facet_sets:
        for face in combinations(sorted(facet), len(facet) - 1):
            key = frozenset(face)
            incidence[key] = incidence.get(key, 0) + 1
    for codimension_one_face, count in incidence.items():
        if count > 2:
            return _PseudomanifoldDecision(
                False,
                False,
                f"codim-1 face {sorted(codimension_one_face)} is in {count} facets",
                dimension,
                num_facets,
            )

    is_closed = bool(incidence) and all(count == 2 for count in incidence.values())
    return _PseudomanifoldDecision(
        True,
        is_closed,
        None if is_closed else "pseudomanifold with boundary",
        dimension,
        num_facets,
    )


class PseudomanifoldRequest(StrictModel):
    """Decide whether a complex is a pseudomanifold."""

    complex: SimplicialComplexRequest


class PseudomanifoldResult(StrictModel):
    """Pseudomanifold decision result produced for one source complex."""

    complex: FiniteSimplicialComplex
    is_pseudomanifold: bool
    is_closed: bool
    dimension: int
    num_facets: int
    obstruction: str | None = None

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        if not self.is_pseudomanifold and self.is_closed:
            raise _validation_error(
                "topology.require_pseudomanifold_branch_1",
                "non-pseudomanifold cannot be closed",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, *, complex_: FiniteSimplicialComplex, decision: _PseudomanifoldDecision
    ) -> Self:
        return cls.model_construct(
            complex=complex_,
            is_pseudomanifold=decision.is_pseudomanifold,
            is_closed=decision.is_closed,
            dimension=decision.dimension,
            num_facets=decision.num_facets,
            obstruction=decision.obstruction,
        )


__all__ = ["PseudomanifoldRequest", "PseudomanifoldResult", "pseudomanifold_decision"]
