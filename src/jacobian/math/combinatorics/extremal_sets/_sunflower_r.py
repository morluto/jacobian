"""Complete bounded sunflower construction for any admitted petal count.

A sunflower of petal count ``r >= 2`` over the source family is an ``r``-member
subfamily whose pairwise intersections are all equal to one common core.  The
``r = 3`` slice already exists as ``set_system.sunflower_triple_hypergraph``;
this module generalizes the same postcondition to a declared ``r`` while
keeping the source, core, and hypergraph conventions identical.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.extremal_sets.values import (
    IndexedFiniteSetFamily,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)

MAX_SUNFLOWER_PETALS = 8
MAX_SUNFLOWER_INTERSECTION_WORK = 20_000_000
MAX_SUNFLOWER_CANDIDATES = 1_000_000


class SunflowerFamilyRequest(StrictModel):
    """One canonical indexed family plus a declared petal count."""

    source: IndexedFiniteSetFamily
    # Bounds are advertised in the schema, while execution owns the stable
    # operation diagnostic (including the distinction between domain and
    # resource rejection) before any candidate enumeration.
    petal_count: StrictInt = Field(
        json_schema_extra={
            "minimum": 2,
            "maximum": MAX_SUNFLOWER_PETALS,
        }
    )


class SunflowerFamily(StrictModel):
    """One complete sunflower row bound to its source members and exact core."""

    edge_id: str
    source_indices: tuple[StrictInt, ...] = Field(max_length=MAX_VERTICES)
    core: tuple[int, ...]


class SunflowerFamilyResult(StrictModel):
    """Every admitted sunflower row plus the derived count and edge projection."""

    source: IndexedFiniteSetFamily
    petal_count: int
    sunflowers: tuple[SunflowerFamily, ...] = Field(max_length=MAX_EDGES)
    sunflower_count: StrictInt = Field(ge=0, le=MAX_EDGES)
    sunflower_free: bool
    hypergraph_edges: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_EDGES
    )
    # A domain-owned projection, unlike ``hypergraph_edges`` which is retained
    # as a compact compatibility ledger for callers that only need rows.
    hypergraph: FiniteHypergraph


def construct_sunflower_family(
    request: SunflowerFamilyRequest,
) -> SunflowerFamilyResult:
    """Return every ``petal_count``-member sunflower with its exact common core."""

    petal_count = request.petal_count
    if petal_count < 2 or petal_count > MAX_SUNFLOWER_PETALS:
        raise OperationResourceAdmissionError(
            location=("petal_count",),
            code="set_system.sunflower.petal_count_bound",
            message=(
                f"petal_count must lie in [2, {MAX_SUNFLOWER_PETALS}] for a bounded "
                "complete construction"
            ),
        )
    source = request.source
    member_count = len(source.members)
    if member_count > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.vertex_bound",
            message=f"sunflower construction supports at most {MAX_VERTICES} members",
        )
    if member_count < petal_count:
        # No subfamily of the requested size exists; the complete family is
        # empty and the source is vacuously sunflower-free at this petal count.
        return SunflowerFamilyResult(
            source=source,
            petal_count=petal_count,
            sunflowers=(),
            sunflower_count=0,
            sunflower_free=True,
            hypergraph_edges=(),
            hypergraph=FiniteHypergraph(
                vertices=tuple(str(index) for index in range(member_count)),
                edges=(),
            ),
        )
    candidate_bound = comb(member_count, petal_count)
    if candidate_bound > MAX_SUNFLOWER_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.candidate_bound",
            message=(
                f"the {candidate_bound} candidate {petal_count}-subfamilies exceed "
                f"the {MAX_SUNFLOWER_CANDIDATES}-candidate exact-work bound"
            ),
        )
    maximum_size = max((len(member) for member in source.members), default=0)
    intersection_pairs = comb(petal_count, 2)
    if (
        intersection_pairs * maximum_size * candidate_bound
        > MAX_SUNFLOWER_INTERSECTION_WORK
    ):
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.intersection_work_bound",
            message=(
                "sunflower pairwise-intersection work exceeds its exact bound for the "
                "declared petal count"
            ),
        )
    sets = tuple(frozenset(member) for member in source.members)
    rows: list[SunflowerFamily] = []
    for indices in combinations(range(member_count), petal_count):
        core = sets[indices[0]] & sets[indices[1]]
        if all(
            sets[left] & sets[right] == core for left, right in combinations(indices, 2)
        ):
            if len(rows) >= MAX_EDGES:
                # Candidate work is admitted independently from the actual
                # result carrier.  Stop before constructing an overlarge public
                # result while preserving the all-or-fail complete contract.
                raise OperationResourceAdmissionError(
                    location=("source", "members"),
                    code="set_system.sunflower.output_bound",
                    message=(
                        f"the complete sunflower family exceeds the {MAX_EDGES}-edge "
                        "result carrier"
                    ),
                )
            rows.append(
                SunflowerFamily(
                    edge_id="sunflower_" + "_".join(str(i) for i in indices),
                    source_indices=indices,
                    core=tuple(sorted(core)),
                )
            )
    if len(rows) * petal_count > MAX_TOTAL_INCIDENCES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.incidence_bound",
            message="the sunflower edge projections exceed the incidence carrier limit",
        )
    hypergraph = FiniteHypergraph(
        vertices=tuple(str(index) for index in range(member_count)),
        edges=tuple(
            (row.edge_id, tuple(str(index) for index in row.source_indices))
            for row in rows
        ),
    )
    return SunflowerFamilyResult(
        source=source,
        petal_count=petal_count,
        sunflowers=tuple(rows),
        sunflower_count=len(rows),
        sunflower_free=not rows,
        hypergraph_edges=hypergraph.edges,
        hypergraph=hypergraph,
    )


__all__ = [
    "MAX_SUNFLOWER_CANDIDATES",
    "MAX_SUNFLOWER_PETALS",
    "SunflowerFamily",
    "SunflowerFamilyRequest",
    "SunflowerFamilyResult",
    "construct_sunflower_family",
]
