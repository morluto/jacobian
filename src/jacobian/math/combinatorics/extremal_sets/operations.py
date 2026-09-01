"""Binary-union relation hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.codes.nonlinear._models import ToSetSystemResult
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationResult,
    UnionRelationRow,
)
from jacobian.math.combinatorics.extremal_sets.values import (
    MAX_FAMILY_SIZE,
    IndexedFiniteSetFamily,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)

__all__ = ["construct_binary_union_relation"]

MAX_BINARY_UNION_MEMBERSHIP_WORK = 20_000_000


@dataclass(frozen=True)
class _UnionRelationPlan:
    rows: tuple[tuple[int, int, int], ...]


def construct_binary_union_relation(
    source: IndexedFiniteSetFamily | ToSetSystemResult,
) -> BinaryUnionRelationResult:
    """Return every distinct-member equation ``S_i union S_j = S_k``."""

    source = _canonical_source(source)
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


def _canonical_source(
    source: IndexedFiniteSetFamily | ToSetSystemResult,
) -> IndexedFiniteSetFamily:
    """Convert the code-support producer's value to the shared family type."""
    if isinstance(source, ToSetSystemResult):
        if len(source.supports) > MAX_FAMILY_SIZE:
            raise OperationDomainValidationError(
                location=("source",),
                code="set_system.binary_union_relation.family_exceeds_carrier",
                message=(
                    f"the source family exceeds the {MAX_FAMILY_SIZE}-vertex "
                    "relation carrier"
                ),
            )
        if any(
            support != tuple(sorted(set(support)))
            or any(not 0 <= element < source.length for element in support)
            for support in source.supports
        ):
            raise OperationDomainValidationError(
                location=("source", "supports"),
                code="set_system.binary_union_relation.noncanonical_supports",
                message=(
                    "code supports must be strictly increasing distinct coordinates "
                    "within the declared axis"
                ),
            )
        expected_supports = tuple(
            tuple(index for index, bit in enumerate(word) if bit == 1)
            for word in source.source.codewords
        )
        if source.supports != expected_supports:
            raise OperationDomainValidationError(
                location=("source", "supports"),
                code="set_system.binary_union_relation.support_source_mismatch",
                message="code supports must equal the retained codeword supports",
            )
        return IndexedFiniteSetFamily(
            ground_set_size=source.length,
            members=expected_supports,
        )
    return source


def _admit_union_relation(source: IndexedFiniteSetFamily) -> _UnionRelationPlan:
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
