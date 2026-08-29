"""Binary-union relation hypergraph constructor."""

from __future__ import annotations

from dataclasses import dataclass

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationResult,
    UnionRelationRow,
)
from jacobian.math.combinatorics.extremal_sets.values import IndexedFiniteSetFamily
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)

__all__ = ["construct_binary_union_relation"]


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


def _admit_union_relation(source: IndexedFiniteSetFamily) -> _UnionRelationPlan:
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
