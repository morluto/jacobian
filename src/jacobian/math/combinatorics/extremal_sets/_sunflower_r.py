"""Complete bounded sunflower construction for any admitted petal count.

A sunflower of petal count ``r >= 2`` over the source family is an ``r``-member
subfamily whose pairwise intersections are all equal to one common core.  This
module is the atomic complete-construction owner for every admitted ``r``,
including the former specialized ``r = 3`` slice.
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
# Pairwise frozenset intersection cost tracks admitted element work, not scan
# count: one large pair can dominate, so a modulo-64 scan cadence can skip an
# entire admitted candidate.
SUNFLOWER_PAIRWISE_CHECKPOINT_WORK = 64


def _result_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"set_system.sunflower.{reason}", message)


def _admit_source(
    source: IndexedFiniteSetFamily,
    petal_count: int,
) -> tuple[IndexedFiniteSetFamily, int, int, int, int]:
    """Admit the retained source before any candidate or result expansion."""

    if not isinstance(source, IndexedFiniteSetFamily):
        raise OperationDomainValidationError(
            location=("source",),
            code="set_system.sunflower.source_type",
            message="sunflower construction requires an IndexedFiniteSetFamily",
        )
    if type(petal_count) is not int:
        raise OperationDomainValidationError(
            location=("petal_count",),
            code="set_system.sunflower.petal_count_type",
            message="petal_count must be an integer",
        )
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


def _intersection_search_work(sizes: tuple[int, ...], petal_count: int) -> int:
    """Charge pairwise work from participating member sizes, not a global max."""

    member_count = len(sizes)
    if member_count < petal_count:
        return 0
    pair_occurrences = comb(member_count - 2, petal_count - 2)
    pair_min_sum = 0
    max_pair_min = 0
    for left, right in combinations(range(member_count), 2):
        pair_min = min(sizes[left], sizes[right])
        pair_min_sum += pair_min
        if pair_min > max_pair_min:
            max_pair_min = pair_min
    candidate_bound = comb(member_count, petal_count)
    # Intersection and equality for every pair in every r-tuple, plus one
    # core materialization bounded by the largest participating pair min.
    return 2 * pair_occurrences * pair_min_sum + 2 * candidate_bound * max_pair_min


def _admit_candidates(
    source: IndexedFiniteSetFamily,
    petal_count: int,
    member_count: int,
    source_work: int,
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
    sizes = tuple(len(member) for member in source.members)
    search_work = _intersection_search_work(sizes, petal_count)
    total_work = source_work + search_work
    if total_work > MAX_SUNFLOWER_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.intersection_work_bound",
            message=(
                "sunflower pairwise-intersection work exceeds its exact bound for the "
                "declared petal count"
            ),
        )
    return candidate_bound


def _admit_qualifying_result(
    source: IndexedFiniteSetFamily,
    petal_count: int,
    member_count: int,
    source_units: int,
    row_count: int,
    maximum_size: int,
) -> None:
    """Admit retained rows after the exact qualifying plan is known."""

    if row_count > MAX_EDGES or row_count * petal_count > MAX_TOTAL_INCIDENCES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.output_bound",
            message=(
                f"the complete family requires {row_count} edges "
                f"and {row_count * petal_count} incidences, exceeding the "
                f"{MAX_EDGES}-edge/{MAX_TOTAL_INCIDENCES}-incidence output bound"
            ),
        )
    member_digits = len(str(max(member_count - 1, 0)))
    ground_digits = len(str(max(source.ground_set_size - 1, 0)))
    # Row IDs are bounded ordinals (``sunflower_<position>``) over the found
    # rows, so the longest ID fits the qualifying count, not the petal width.
    edge_id_units = 10 + len(str(max(row_count, 1)))
    row_units = (
        128
        + edge_id_units
        + petal_count * (member_digits + 2)
        + maximum_size * (ground_digits + 1)
    )
    edge_projection_units = 64 + edge_id_units + petal_count * (member_digits + 2)
    base_result_units = source_units + 1024 + member_count * (member_digits + 3)
    allocation_units = base_result_units + row_count * (
        row_units + 2 * edge_projection_units
    )
    if allocation_units > MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.result_allocation_bound",
            message=(
                "the complete sunflower result may require "
                f"{allocation_units} allocation units, exceeding the "
                f"{MAX_SUNFLOWER_RESULT_ALLOCATION_UNITS}-unit result bound"
            ),
        )


def _charge_pairwise_checkpoint(accumulated: int, pair_work: int) -> int:
    """Checkpoint before an intersection whose element work fills the stride."""

    accumulated += max(pair_work, 1)
    if accumulated >= SUNFLOWER_PAIRWISE_CHECKPOINT_WORK:
        request_checkpoint("during sunflower pairwise intersection")
        return 0
    return accumulated


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
        previous_indices: tuple[int, ...] | None = None
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
            if previous_indices is not None and indices <= previous_indices:
                raise _result_error(
                    "row_order",
                    "sunflower rows must appear in lexicographic source-index order",
                )
            previous_indices = indices
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
            expected_edges.append(
                (row.edge_id, tuple(sorted(str(index) for index in indices)))
            )
        if any(
            self.sunflowers[index].source_indices
            >= self.sunflowers[index + 1].source_indices
            for index in range(len(self.sunflowers) - 1)
        ):
            raise _result_error(
                "row_order",
                "sunflower rows must be strictly ordered by source_indices",
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        source: IndexedFiniteSetFamily,
        petal_count: int,
        sunflowers: tuple[SunflowerFamily, ...],
        hypergraph: FiniteHypergraph,
    ) -> Self:
        """Bind a kernel-established family without replaying core scans."""

        return cls.model_construct(
            source=source,
            petal_count=petal_count,
            sunflowers=sunflowers,
            sunflower_count=len(sunflowers),
            sunflower_free=not sunflowers,
            hypergraph_edges=hypergraph.edges,
            hypergraph=hypergraph,
        )


def construct_sunflower_family(
    source: IndexedFiniteSetFamily,
    petal_count: int,
) -> SunflowerFamilyResult:
    """Return every ``petal_count``-member sunflower with its exact common core."""

    source, petal_count, member_count, source_work, source_units = _admit_source(
        source, petal_count
    )
    _admit_candidates(source, petal_count, member_count, source_work)
    vertices = tuple(str(index) for index in range(member_count))
    if member_count < petal_count:
        _admit_qualifying_result(
            source, petal_count, member_count, source_units, 0, 0
        )
        return SunflowerFamilyResult._from_kernel(
            source=source,
            petal_count=petal_count,
            sunflowers=(),
            hypergraph=FiniteHypergraph(vertices=vertices, edges=()),
        )
    request_checkpoint("before sunflower member expansion")
    sets = tuple(frozenset(member) for member in source.members)
    sizes = tuple(len(member) for member in source.members)
    plan: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    maximum_size = max(sizes, default=0)
    pairwise_work = 0
    for candidate_index, indices in enumerate(
        combinations(range(member_count), petal_count), start=1
    ):
        if candidate_index % 256 == 0:
            request_checkpoint("during sunflower candidate enumeration")
        first_left = sets[indices[0]]
        first_right = sets[indices[1]]
        pairwise_work = _charge_pairwise_checkpoint(
            pairwise_work, min(len(first_left), len(first_right))
        )
        core = first_left & first_right
        is_sunflower = True
        for left, right in combinations(indices, 2):
            left_set = sets[left]
            right_set = sets[right]
            pairwise_work = _charge_pairwise_checkpoint(
                pairwise_work, min(len(left_set), len(right_set))
            )
            if left_set & right_set != core:
                is_sunflower = False
                break
        if is_sunflower:
            plan.append((indices, tuple(sorted(core))))
    _admit_qualifying_result(
        source,
        petal_count,
        member_count,
        source_units,
        len(plan),
        maximum_size,
    )
    rows = tuple(
        SunflowerFamily.model_construct(
            edge_id=f"sunflower_{position}",
            source_indices=indices,
            core=core,
        )
        for position, (indices, core) in enumerate(plan, start=1)
    )
    hypergraph = FiniteHypergraph(
        vertices=vertices,
        edges=tuple(
            (row.edge_id, tuple(str(index) for index in row.source_indices))
            for row in rows
        ),
    )
    return SunflowerFamilyResult._from_kernel(
        source=source,
        petal_count=petal_count,
        sunflowers=rows,
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
