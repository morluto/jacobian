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
from typing import Self

from pydantic import Field, StrictBool, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._execution import request_checkpoint
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.extremal_sets.values import (
    IndexedFiniteSetFamily,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)

# Petal count is bounded by the shared finite-hypergraph vertex carrier, not
# by an arbitrary small slice.  Candidate, intersection, output, and
# allocation admission below still reject requests whose complete exact work
# cannot fit the operation envelope before enumeration begins.
MAX_SUNFLOWER_PETALS = MAX_VERTICES
MAX_SUNFLOWER_INTERSECTION_WORK = 20_000_000
MAX_SUNFLOWER_CANDIDATES = 1_000_000
# The source value is retained unchanged in every result, including the
# vacuous case. These operation-owned limits cover the ambient axis, aggregate
# membership inspection, and retained result allocation. One allocation unit
# conservatively reserves a scalar digit, label code point, container slot, or
# fixed record field; transports own encoded-byte limits separately.
MAX_SUNFLOWER_GROUND_SET_SIZE = 1_000_000
MAX_SUNFLOWER_MEMBERSHIPS = 1_000_000
MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS = 16 * 1024 * 1024


def _result_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"set_system.sunflower.{reason}", message)


def _admit_source(
    request: SunflowerFamilyRequest,
) -> tuple[IndexedFiniteSetFamily, int, int, int, int]:
    """Admit the retained source before any candidate or result expansion."""

    petal_count = request.petal_count
    if petal_count < 2:
        raise OperationDomainValidationError(
            location=("petal_count",),
            code="set_system.sunflower.petal_count_domain",
            message="petal_count must be at least 2",
        )
    if petal_count > MAX_SUNFLOWER_PETALS:
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
    if source.ground_set_size > MAX_SUNFLOWER_GROUND_SET_SIZE:
        raise OperationResourceAdmissionError(
            location=("source", "ground_set_size"),
            code="set_system.sunflower.ground_set_bound",
            message=(
                f"the source ground set exceeds the {MAX_SUNFLOWER_GROUND_SET_SIZE}-element "
                "sunflower admission envelope"
            ),
        )
    memberships = sum(len(member) for member in source.members)
    if memberships > MAX_SUNFLOWER_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.membership_bound",
            message=(
                f"the source has {memberships} memberships, exceeding the "
                f"{MAX_SUNFLOWER_MEMBERSHIPS}-membership bound"
            ),
        )
    source_work = 3 * memberships + member_count
    if source_work > MAX_SUNFLOWER_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.membership_work_bound",
            message=(
                f"source membership work of {source_work} exceeds the exact "
                f"{MAX_SUNFLOWER_INTERSECTION_WORK}-unit bound"
            ),
        )
    ground_digits = len(str(max(source.ground_set_size - 1, 0)))
    source_units = 16 + member_count + memberships * (ground_digits + 1)
    if source_units > MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="set_system.sunflower.source_output_bound",
            message=(
                f"the retained source requires {source_units} allocation units, "
                "exceeding the sunflower result bound"
            ),
        )
    return source, petal_count, member_count, source_work, source_units


def _admit_candidates(
    source: IndexedFiniteSetFamily,
    petal_count: int,
    member_count: int,
    source_work: int,
    source_units: int,
) -> int:
    """Admit candidate, intersection-work, and complete-output envelopes."""

    candidate_bound = (
        comb(member_count, petal_count) if member_count >= petal_count else 0
    )
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
    intersection_work = (intersection_pairs + 1) * maximum_size * candidate_bound
    total_work = source_work + intersection_work
    if total_work > MAX_SUNFLOWER_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.intersection_work_bound",
            message=(
                "sunflower pairwise-intersection work exceeds its exact bound for the "
                "declared petal count"
            ),
        )
    if (
        candidate_bound > MAX_EDGES
        or candidate_bound * petal_count > MAX_TOTAL_INCIDENCES
    ):
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.output_bound",
            message=(
                f"the complete candidate family may require {candidate_bound} edges "
                f"and {candidate_bound * petal_count} incidences, exceeding the "
                f"{MAX_EDGES}-edge/{MAX_TOTAL_INCIDENCES}-incidence output bound"
            ),
        )
    member_digits = len(str(max(member_count - 1, 0)))
    ground_digits = len(str(max(source.ground_set_size - 1, 0)))
    # Row IDs are bounded ordinals (``sunflower_<position>``) over the found
    # rows, so the longest ID fits the candidate count, not the petal width.
    edge_id_units = 10 + len(str(max(candidate_bound, 1)))
    row_units = (
        128
        + edge_id_units
        + petal_count * (member_digits + 2)
        + maximum_size * (ground_digits + 1)
    )
    edge_projection_units = 64 + edge_id_units + petal_count * (member_digits + 2)
    base_result_units = source_units + 1024 + member_count * (member_digits + 3)
    allocation_upper_bound = base_result_units + candidate_bound * (
        row_units + 2 * edge_projection_units
    )
    if allocation_upper_bound > MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.result_allocation_bound",
            message=(
                "the complete sunflower result may require "
                f"{allocation_upper_bound} allocation units, exceeding the "
                f"{MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS}-unit result bound"
            ),
        )
    return candidate_bound


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
    core: tuple[StrictInt, ...] = Field(max_length=MAX_SUNFLOWER_MEMBERSHIPS)


class SunflowerFamilyResult(StrictModel):
    """Every admitted sunflower row plus the derived count and edge projection."""

    source: IndexedFiniteSetFamily
    petal_count: StrictInt
    sunflowers: tuple[SunflowerFamily, ...] = Field(max_length=MAX_EDGES)
    sunflower_count: StrictInt = Field(ge=0, le=MAX_EDGES)
    sunflower_free: StrictBool
    hypergraph_edges: tuple[tuple[str, tuple[str, ...]], ...] = Field(
        max_length=MAX_EDGES
    )
    # A domain-owned projection, unlike ``hypergraph_edges`` which is retained
    # as a compact compatibility ledger for callers that only need rows.
    hypergraph: FiniteHypergraph

    @model_validator(mode="after")
    def require_coherent_projection(self) -> Self:
        """Bind cheap structural summaries without replaying intersections."""

        if not 2 <= self.petal_count <= MAX_SUNFLOWER_PETALS:
            raise _result_error(
                "petal_count", "petal_count lies outside the admitted range"
            )
        member_count = len(self.source.members)
        expected_vertices = tuple(str(index) for index in range(member_count))
        if self.hypergraph.vertices != expected_vertices:
            raise _result_error(
                "source_vertices",
                "hypergraph vertices must be the canonical source member IDs",
            )
        expected_edges: list[tuple[str, tuple[str, ...]]] = []
        seen_indices: set[tuple[int, ...]] = set()
        seen_edge_ids: set[str] = set()
        # Row IDs are one-based ordinals over the found rows in enumeration
        # order.  Index-list IDs would exceed the hypergraph label limit for
        # large petal counts, while ordinals stay bounded by the candidate
        # count and remain canonical because enumeration order is fixed.
        for position, row in enumerate(self.sunflowers, start=1):
            indices = tuple(row.source_indices)
            if len(indices) != self.petal_count or indices != tuple(sorted(indices)):
                raise _result_error(
                    "row_shape",
                    "sunflower rows must have sorted source indices of the declared petal count",
                )
            if len(set(indices)) != len(indices) or any(
                index < 0 or index >= member_count for index in indices
            ):
                raise _result_error(
                    "row_source", "sunflower rows must reference source members"
                )
            core = tuple(row.core)
            if core != tuple(sorted(core)) or len(set(core)) != len(core):
                raise _result_error(
                    "core_shape", "sunflower cores must be sorted distinct indices"
                )
            if any(index < 0 or index >= self.source.ground_set_size for index in core):
                raise _result_error(
                    "core_source", "sunflower cores must lie on the source ground set"
                )
            expected_edge_id = f"sunflower_{position}"
            if row.edge_id != expected_edge_id:
                raise _result_error(
                    "row_identity", "sunflower row IDs must be canonical ordinals"
                )
            if indices in seen_indices or row.edge_id in seen_edge_ids:
                raise _result_error(
                    "row_identity", "sunflower rows must have unique identities"
                )
            seen_indices.add(indices)
            seen_edge_ids.add(row.edge_id)
            # FiniteHypergraph canonicalizes string labels lexicographically;
            # retain that exact projection for multi-digit source IDs.
            expected_edges.append(
                (row.edge_id, tuple(sorted(str(index) for index in indices)))
            )
        canonical_edges = tuple(expected_edges)
        if self.sunflower_count != len(self.sunflowers):
            raise _result_error(
                "count", "sunflower_count must equal the returned row count"
            )
        if self.sunflower_free != (not self.sunflowers):
            raise _result_error(
                "status", "sunflower_free must agree with whether rows are present"
            )
        if (
            self.hypergraph_edges != canonical_edges
            or self.hypergraph.edges != canonical_edges
        ):
            raise _result_error(
                "projection",
                "hypergraph projections must equal the canonical sunflower rows",
            )
        return self


def construct_sunflower_family(
    request: SunflowerFamilyRequest,
) -> SunflowerFamilyResult:
    """Return every ``petal_count``-member sunflower with its exact common core."""

    source, petal_count, member_count, source_work, source_units = _admit_source(
        request
    )
    _admit_candidates(source, petal_count, member_count, source_work, source_units)
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
    sets = tuple(frozenset(member) for member in source.members)
    rows: list[SunflowerFamily] = []
    for candidate_index, indices in enumerate(
        combinations(range(member_count), petal_count), start=1
    ):
        if candidate_index % 256 == 0:
            request_checkpoint("during sunflower candidate enumeration")
        core = sets[indices[0]] & sets[indices[1]]
        if all(
            sets[left] & sets[right] == core for left, right in combinations(indices, 2)
        ):
            rows.append(
                SunflowerFamily(
                    edge_id=f"sunflower_{len(rows) + 1}",
                    source_indices=indices,
                    core=tuple(sorted(core)),
                )
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
    "MAX_SUNFLOWER_GROUND_SET_SIZE",
    "MAX_SUNFLOWER_MEMBERSHIPS",
    "MAX_SUNFLOWER_PETALS",
    "MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS",
    "SunflowerFamily",
    "SunflowerFamilyRequest",
    "SunflowerFamilyResult",
    "construct_sunflower_family",
]
