"""Binary-union relation hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationResult,
    SunflowerHypergraphResult,
    SunflowerTriple,
    UnionRelationRow,
)
from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    construct_sunflower_family,
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

__all__ = [
    "construct_binary_union_relation",
    "construct_sunflower_family",
    "construct_sunflower_hypergraph",
]

MAX_BINARY_UNION_MEMBERSHIP_WORK = 20_000_000
MAX_SUNFLOWER_INTERSECTION_WORK = 20_000_000


@dataclass(frozen=True)
class _UnionRelationPlan:
    rows: tuple[tuple[int, int, int], ...]


def construct_binary_union_relation(
    source: IndexedFiniteSetFamily,
) -> BinaryUnionRelationResult:
    """Return every distinct-member equation ``S_i union S_j = S_k``."""

    plan = _admit_union_relation(source)
    rows = tuple(
        UnionRelationRow(
            edge_id=_edge_id(i, j, k),
            operand_i=i,
            operand_j=j,
            result_k=k,
        )
        for i, j, k in plan.rows
    )
    vertices = tuple(str(index) for index in range(len(source.members)))
    edges = tuple(
        (
            row.edge_id,
            tuple(sorted((str(row.operand_i), str(row.operand_j), str(row.result_k)))),
        )
        for row in rows
    )
    return BinaryUnionRelationResult(
        source=source,
        rows=rows,
        hypergraph=FiniteHypergraph(vertices=vertices, edges=edges),
    )


def construct_sunflower_hypergraph(
    source: IndexedFiniteSetFamily,
) -> SunflowerHypergraphResult:
    """Return every three-member sunflower with its exact common core."""

    member_count = len(source.members)
    if member_count > MAX_VERTICES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.vertex_bound",
            message=f"sunflower construction supports at most {MAX_VERTICES} members",
        )
    triple_bound = member_count * (member_count - 1) * (member_count - 2) // 6
    if triple_bound > MAX_EDGES:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.output_bound",
            message=f"the {triple_bound} candidate triples exceed the {MAX_EDGES}-edge output bound",
        )
    maximum_size = max((len(member) for member in source.members), default=0)
    if 3 * maximum_size * triple_bound > MAX_SUNFLOWER_INTERSECTION_WORK:
        raise OperationResourceAdmissionError(
            location=("source", "members"),
            code="set_system.sunflower.intersection_work_bound",
            message="sunflower pair-intersection work exceeds its exact bound",
        )
    sets = tuple(frozenset(member) for member in source.members)
    rows: list[SunflowerTriple] = []
    for first in range(member_count):
        for second in range(first + 1, member_count):
            first_core = sets[first] & sets[second]
            for third in range(second + 1, member_count):
                if (
                    first_core
                    == sets[first] & sets[third]
                    == sets[second] & sets[third]
                ):
                    edge_id = f"sunflower_{first}_{second}_{third}"
                    rows.append(
                        SunflowerTriple(
                            edge_id=edge_id,
                            source_indices=(first, second, third),
                            core=tuple(sorted(first_core)),
                        )
                    )
    return SunflowerHypergraphResult(
        source=source,
        sunflowers=tuple(rows),
        hypergraph=FiniteHypergraph(
            vertices=tuple(str(index) for index in range(member_count)),
            edges=tuple(
                (row.edge_id, tuple(str(index) for index in row.source_indices))
                for row in rows
            ),
        ),
    )


def _admit_union_relation(source: IndexedFiniteSetFamily) -> _UnionRelationPlan:
    if len(source.members) > MAX_VERTICES:
        raise OperationDomainValidationError(
            location=("source",),
            code="set_system.binary_union_relation.family_exceeds_carrier",
            message=f"the source family exceeds the {MAX_VERTICES}-vertex relation carrier",
        )
    member_sizes = tuple(len(member) for member in source.members)
    membership_work = (len(source.members) + 1) * sum(member_sizes)
    generated_union_work = sum(
        member_sizes[i] + member_sizes[j]
        for i in range(len(member_sizes))
        for j in range(i + 1, len(member_sizes))
    )
    # Source sizing and the final canonical serialization each inspect every
    # retained coordinate; charge both traversals before any pair work begins.
    source_delivery_work = 2 * sum(member_sizes)
    membership_work += generated_union_work + source_delivery_work
    if membership_work > MAX_BINARY_UNION_MEMBERSHIP_WORK:
        raise OperationDomainValidationError(
            location=("source",),
            code="set_system.binary_union_relation.work_exceeded",
            message=(
                f"binary-union membership work of {membership_work} exceeds the "
                f"{MAX_BINARY_UNION_MEMBERSHIP_WORK}-unit bound"
            ),
        )
    sets = tuple(frozenset(member) for member in source.members)
    source_index = {member: index for index, member in enumerate(sets)}
    rows: list[tuple[int, int, int]] = []
    for i, left in enumerate(sets):
        for j in range(i + 1, len(sets)):
            result = source_index.get(left | sets[j])
            if result is not None and result not in (i, j):
                rows.append((i, j, result))

    row_count = len(rows)
    if row_count > MAX_EDGES or 3 * row_count > MAX_TOTAL_INCIDENCES:
        raise OperationDomainValidationError(
            location=("source",),
            code="set_system.binary_union_relation.result_exceeds_carrier",
            message=(
                f"the exact relation has {row_count} rows, exceeding the "
                f"{MAX_EDGES}-edge or {MAX_TOTAL_INCIDENCES}-incidence carrier limit"
            ),
        )
    return _UnionRelationPlan(rows=tuple(rows))


def _edge_id(operand_i: int, operand_j: int, result_k: int) -> str:
    return f"union_{operand_i}_{operand_j}_to_{result_k}"
