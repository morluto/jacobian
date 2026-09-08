"""Exact bitset union planning followed by complete source-pair expansion."""

from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations, groupby
from time import monotonic

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
)
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.colorings import (
    IndexedHyperedgeColoring,
)
from jacobian.math.combinatorics.finite_structures.hypergraphs.same_color_conflicts._models import (
    MAX_CONFLICT_PAIRS,
    SameColorConflictProvenance,
    SameColorConflictsResult,
)

__all__ = ["construct"]


def _checkpoint(deadline: float, stage: str) -> None:
    request_checkpoint(stage)
    if monotonic() >= deadline:
        raise OperationExecutionTimeoutError("same-colour conflict deadline expired")


def _reject(message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("coloring",),
        code="same_color_conflicts.resource_bound",
        message=message,
    )


@dataclass(frozen=True)
class _Plan:
    masks: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]
    unions: tuple[int, ...]


def _admit(coloring: IndexedHyperedgeColoring, deadline: float) -> _Plan:
    """Admit pairs, compute compressed unions, then admit exact output size.

    P=sum_c choose(|E_c|,2) bounds provenance and original pair expansion.
    Source incidences are <=36000, masks have <=256 bits, and P<=65536.
    Duplicate member sets within each colour are grouped before union planning:
    distinct-type pairs and repeated types together number at most P. Sorting
    these <=P masks avoids dependence on numeric-hash collision behaviour.
    Exact distinct-union counts/incidences are admitted before source pairs are
    expanded. This admits repeated sources whose many pairs have one union.

    All admitted work is polynomial: O(I + E log E + P log P) operations on
    <=256-bit integers, plus <=36000 output memberships and P provenance rows.
    Retained source labels/IDs and two source IDs per provenance row have the
    existing 64-character carrier bound; no unbounded label namespace is made.
    Even six encoded bytes per label character, all retained source incidences,
    assignments, graph incidences and provenance fit below 128 MiB. This is
    derived from the mathematical row/label bounds, not a transport setting.
    """
    _checkpoint(deadline, "before conflict admission")
    groups: list[list[int]] = [[] for _ in range(coloring.color_count)]
    for index, assignment in enumerate(coloring.assignments):
        groups[assignment.color_index].append(index)
    pair_count = sum(len(group) * (len(group) - 1) // 2 for group in groups)
    if pair_count > MAX_CONFLICT_PAIRS:
        _reject("complete same-colour source-pair provenance exceeds 65536 rows")
    positions = {
        label: index for index, label in enumerate(coloring.hypergraph.vertices)
    }
    masks = tuple(
        sum(1 << positions[label] for label in members)
        for _, members in coloring.hypergraph.edges
    )
    candidates: list[int] = []
    for group in groups:
        _checkpoint(deadline, "during compressed union planning")
        types = tuple(
            (mask, len(tuple(duplicates)))
            for mask, duplicates in groupby(sorted(masks[index] for index in group))
        )
        candidates.extend(mask for mask, multiplicity in types if multiplicity > 1)
        candidates.extend(left[0] | right[0] for left, right in combinations(types, 2))
    unions = tuple(mask for mask, _ in groupby(sorted(candidates)))
    if len(unions) > MAX_EDGES:
        _reject("distinct conflict unions exceed the 12000-edge carrier")
    if sum(mask.bit_count() for mask in unions) > MAX_TOTAL_INCIDENCES:
        _reject("distinct conflict unions exceed the 36000-incidence carrier")
    _checkpoint(deadline, "after conflict admission")
    return _Plan(masks, tuple(tuple(group) for group in groups), unions)


def construct(coloring: IndexedHyperedgeColoring) -> SameColorConflictsResult:
    """Return every union of distinct equally coloured source edges, with pairs."""
    execution = current_request_execution()
    deadline = (execution.started_at if execution is not None else monotonic()) + 60
    if execution is not None and execution.deadline is not None:
        deadline = min(deadline, execution.deadline)
    bind_request_deadline(deadline)
    plan = _admit(coloring, deadline)
    vertices = coloring.hypergraph.vertices
    edges = tuple(
        (
            f"c{index}",
            tuple(label for bit, label in enumerate(vertices) if mask >> bit & 1),
        )
        for index, mask in enumerate(plan.unions)
    )
    source_ids = tuple(edge_id for edge_id, _ in coloring.hypergraph.edges)
    pairs = sorted(
        (left, right, color)
        for color, group in enumerate(plan.groups)
        for left, right in combinations(group, 2)
    )
    provenance: list[SameColorConflictProvenance] = []
    for left, right, color in pairs:
        _checkpoint(deadline, "during conflict provenance expansion")
        union = plan.masks[left] | plan.masks[right]
        union_index = bisect_left(plan.unions, union)
        provenance.append(
            SameColorConflictProvenance(
                conflict_edge_id=f"c{union_index}",
                source_edge_ids=(source_ids[left], source_ids[right]),
                color_index=color,
            )
        )
    _checkpoint(deadline, "before conflict result construction")
    result = SameColorConflictsResult.model_construct(
        coloring=coloring,
        hypergraph=FiniteHypergraph.model_construct(vertices=vertices, edges=edges),
        provenance=tuple(provenance),
    )
    _checkpoint(deadline, "after conflict result construction")
    return result
