"""Deterministic H-minor-model checking kernel."""

from __future__ import annotations

from collections import deque

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.minors._models import (
    MAX_MINOR_BRANCH_MEMBERSHIPS,
    BranchConnectivityLedger,
    BranchSet,
    EdgeWitness,
    MinorModelCheckResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = ["check_minor_model", "verify_minor_model"]


def _require_minor_model_admission(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    branch_sets: tuple[BranchSet, ...],
    witnesses: tuple[EdgeWitness, ...],
) -> None:
    """Share the catalog checker envelope with native callers."""

    if not target.vertices:
        raise OperationDomainValidationError(
            location=("target",),
            code="graph.minor_model.empty_target",
            message="the target graph must declare at least one vertex",
        )
    memberships = sum(len(branch.members) for branch in branch_sets)
    if memberships > MAX_MINOR_BRANCH_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("branch_sets",),
            code="graph.minor_model.membership_bound",
            message="branch-set memberships exceed the admitted checker envelope",
        )
    if len(witnesses) > len(target.edges) + len(branch_sets):
        raise OperationResourceAdmissionError(
            location=("witnesses",),
            code="graph.minor_model.witness_bound",
            message="witness rows exceed the admitted checker envelope",
        )


def _ordered(edge: tuple[str, str]) -> tuple[str, str]:
    """Return one canonical endpoint order for a source edge pair."""

    left, right = edge
    return (left, right) if left < right else (right, left)


def _adjacency(graph: SimpleUndirectedGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {vertex: set() for vertex in graph.vertices}
    for edge in graph.edges:
        left, right = _ordered(edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return adjacency


def _connected_component(
    members: frozenset[str], adjacency: dict[str, set[str]]
) -> frozenset[str]:
    root = min(members)
    reached = {root}
    queue: deque[str] = deque([root])
    while queue:
        vertex = queue.popleft()
        for neighbor in adjacency[vertex] & members - reached:
            reached.add(neighbor)
            queue.append(neighbor)
    return frozenset(reached)


class _MinorObstructionError(Exception):
    """Deterministic first-obstruction signal for the checker stages."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _branch_sets_by_target(
    target: SimpleUndirectedGraph,
    branch_sets: tuple[BranchSet, ...],
) -> dict[str, BranchSet]:
    """Index branch sets, raising the first coverage obstruction."""

    by_target = {branch.target: branch for branch in branch_sets}
    if len(by_target) != len(branch_sets):
        seen: set[str] = set()
        for branch in branch_sets:
            if branch.target in seen:
                raise _MinorObstructionError(
                    "DUPLICATE_BRANCH_SET",
                    f"target {branch.target!r} carries two branch sets",
                )
            seen.add(branch.target)
    for vertex in target.vertices:
        if vertex not in by_target:
            raise _MinorObstructionError(
                "MISSING_BRANCH_SET", f"target {vertex!r} has no branch set"
            )
    for branch in branch_sets:
        if branch.target not in set(target.vertices):
            raise _MinorObstructionError(
                "UNKNOWN_BRANCH_TARGET",
                f"branch target {branch.target!r} is not a target vertex",
            )
    for vertex in target.vertices:
        if not by_target[vertex].members:
            raise _MinorObstructionError(
                "EMPTY_BRANCH_SET", f"branch set of target {vertex!r} is empty"
            )
    return by_target


def _member_ownership(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    by_target: dict[str, BranchSet],
) -> dict[str, str]:
    """Map owned source vertices to targets, raising geometry obstructions."""

    source_vertices = set(source.vertices)
    for vertex in target.vertices:
        for member in by_target[vertex].members:
            if member not in source_vertices:
                raise _MinorObstructionError(
                    "UNDECLARED_SOURCE_VERTEX",
                    f"branch member {member!r} is not a source vertex",
                )
    occupied: dict[str, str] = {}
    for vertex in target.vertices:
        for member in by_target[vertex].members:
            if member in occupied:
                raise _MinorObstructionError(
                    "OVERLAPPING_BRANCH_SETS",
                    f"source vertex {member!r} lies in branch sets "
                    f"{occupied[member]!r} and {vertex!r}",
                )
            occupied[member] = vertex
    adjacency = _adjacency(source)
    for vertex in target.vertices:
        members = frozenset(by_target[vertex].members)
        reached = _connected_component(members, adjacency)
        if reached != members:
            raise _MinorObstructionError(
                "DISCONNECTED_BRANCH_SET",
                f"branch set of target {vertex!r} splits off {sorted(reached)}",
            )
    return occupied


def _require_witnesses(
    target: SimpleUndirectedGraph,
    witnesses: tuple[EdgeWitness, ...],
    source_edges: set[tuple[str, str]],
    owner: dict[str, str],
) -> None:
    """Replay every witness, raising the first witness obstruction."""

    if len({witness.targets for witness in witnesses}) != len(witnesses):
        raise OperationDomainValidationError(
            location=("witnesses",),
            code="graph.minor_model.duplicate_witness",
            message="witness targets must be unique",
        )
    target_edge_keys = {_ordered(edge) for edge in target.edges}
    witness_by_targets = {_ordered(witness.targets): witness for witness in witnesses}
    for edge in target.edges:
        key = _ordered(edge)
        if key not in witness_by_targets:
            raise _MinorObstructionError(
                "MISSING_WITNESS", f"target edge {list(key)!r} has no witness"
            )
    for witness in witnesses:
        if _ordered(witness.targets) not in target_edge_keys:
            raise _MinorObstructionError(
                "SUPERFLUOUS_WITNESS",
                f"witness targets {list(witness.targets)!r} is not a target edge",
            )
        left, right = (witness.source_edge[0], witness.source_edge[1])
        if (left, right) not in source_edges:
            raise _MinorObstructionError(
                "WITNESS_EDGE_ABSENT",
                f"witness edge {[left, right]!r} is absent from the source graph",
            )
        first_owner, second_owner = owner.get(left), owner.get(right)
        first_target, second_target = witness.targets
        if {first_owner, second_owner} != {first_target, second_target} or (
            first_owner == second_owner
        ):
            raise _MinorObstructionError(
                "WITNESS_ENDPOINT_MISMATCH",
                f"witness edge {[left, right]!r} does not cross branch sets "
                f"{first_target!r} and {second_target!r}",
            )


def check_minor_model(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    branch_sets: tuple[BranchSet, ...],
    witnesses: tuple[EdgeWitness, ...],
) -> MinorModelCheckResult:
    """Check one candidate H-minor model deterministically, first obstruction wins."""

    _require_minor_model_admission(source, target, branch_sets, witnesses)

    def invalid(code: str, detail: str) -> MinorModelCheckResult:
        return MinorModelCheckResult._from_kernel(
            source=source,
            target=target,
            status="INVALID_MINOR_MODEL",
            branch_sets=branch_sets,
            witnesses=witnesses,
            connectivity=(),
            used_source_vertices=(),
            deleted_source_vertices=tuple(source.vertices),
            used_source_edges=(),
            deleted_source_edges=tuple(source.edges),
            obstruction_code=code,
            obstruction_detail=detail,
        )

    try:
        by_target = _branch_sets_by_target(target, branch_sets)
        occupied = _member_ownership(source, target, by_target)
        _require_witnesses(
            target,
            witnesses,
            {_ordered(edge) for edge in source.edges},
            occupied,
        )
    except _MinorObstructionError as obstruction:
        return invalid(obstruction.code, obstruction.detail)
    owner = occupied
    branch_list = tuple(
        BranchSet(target=vertex, members=tuple(sorted(by_target[vertex].members)))
        for vertex in target.vertices
    )
    ledgers = tuple(
        BranchConnectivityLedger(
            target=vertex,
            member_count=len(by_target[vertex].members),
            internal_edge_count=sum(
                1
                for edge in source.edges
                if owner.get(edge[0]) == vertex and owner.get(edge[1]) == vertex
            ),
            root=min(by_target[vertex].members),
        )
        for vertex in target.vertices
    )
    used_vertices = tuple(vertex for vertex in source.vertices if vertex in occupied)
    used_edges = tuple(
        edge
        for edge in source.edges
        if edge in {witness.source_edge for witness in witnesses}
    )
    return MinorModelCheckResult._from_kernel(
        source=source,
        target=target,
        status="VALID_MINOR_MODEL",
        branch_sets=branch_list,
        witnesses=tuple(
            EdgeWitness(targets=witness.targets, source_edge=witness.source_edge)
            for witness in witnesses
        ),
        connectivity=ledgers,
        used_source_vertices=used_vertices,
        deleted_source_vertices=tuple(
            vertex for vertex in source.vertices if vertex not in occupied
        ),
        used_source_edges=used_edges,
        deleted_source_edges=tuple(
            edge for edge in source.edges if edge not in set(used_edges)
        ),
        obstruction_code=None,
        obstruction_detail=None,
    )


def verify_minor_model(claim: MinorModelCheckResult) -> bool:
    """Replay a minor-model verdict against its retained graphs and candidate."""

    try:
        return (
            check_minor_model(
                claim.source, claim.target, claim.branch_sets, claim.witnesses
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
