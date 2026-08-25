"""Pseudomanifold decision contracts and replay kernel."""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple, Self

from pydantic import model_validator

from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    Simplex,
    SimplicialComplexRequest,
    TopologyExactResult,
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


class PseudomanifoldResult(TopologyExactResult):
    """Pseudomanifold decision result bound to its source complex."""

    complex: SimplicialComplexRequest
    is_pseudomanifold: bool
    is_closed: bool
    dimension: int
    num_facets: int
    obstruction: str | None = None

    @model_validator(mode="after")
    def require_pseudomanifold_binding(self) -> Self:
        expected = pseudomanifold_decision(self.complex.facets)
        if (
            self.dimension != expected.dimension
            or self.num_facets != expected.num_facets
        ):
            raise _validation_error(
                "topology.require_pseudomanifold_binding_1",
                "dimension/num_facets must match source complex",
            )
        if self.is_pseudomanifold != expected.is_pseudomanifold:
            raise _validation_error(
                "topology.require_pseudomanifold_binding_2",
                f"is_pseudomanifold {self.is_pseudomanifold} does not match "
                f"expected {expected.is_pseudomanifold}",
            )
        if not expected.is_pseudomanifold:
            if self.is_closed:
                raise _validation_error(
                    "topology.require_pseudomanifold_binding_3",
                    "non-pseudomanifold cannot be closed",
                )
            if self.obstruction != expected.obstruction:
                raise _validation_error(
                    "topology.require_pseudomanifold_binding_4",
                    f"obstruction {self.obstruction!r} does not match replayed "
                    f"{expected.obstruction!r}",
                )
            return self
        if self.is_closed != expected.is_closed:
            raise _validation_error(
                "topology.require_pseudomanifold_binding_5",
                "is_closed must match codim-1 incidence",
            )
        if self.obstruction != expected.obstruction:
            raise _validation_error(
                "topology.require_pseudomanifold_binding_6",
                f"obstruction {self.obstruction!r} does not match expected "
                f"{expected.obstruction!r}",
            )
        return self


__all__ = ["PseudomanifoldRequest", "PseudomanifoldResult", "pseudomanifold_decision"]
