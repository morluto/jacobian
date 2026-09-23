"""Deterministic H-minor-model checking kernel."""

from __future__ import annotations

from collections import deque
from itertools import pairwise, permutations, product

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.minors._models import (
    MAX_MINOR_BRANCH_MEMBERSHIPS,
    MAX_MINOR_FIND_SOURCE_VERTICES,
    MAX_MINOR_FIND_TARGET_VERTICES,
    MAX_TOPO_FIND_SOURCE_VERTICES,
    MAX_TOPO_FIND_TARGET_VERTICES,
    BranchConnectivityLedger,
    BranchSet,
    BranchVertex,
    EdgeWitness,
    MinorModelCheckResult,
    MinorModelFindBudget,
    MinorModelFindResult,
    SubdivisionPath,
    TopologicalMinorCheckResult,
    TopologicalMinorFindBudget,
    TopologicalMinorFindResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "check_minor_model",
    "check_topological_minor",
    "find_minor_model",
    "find_topological_minor",
    "verify_minor_model",
    "verify_topological_minor",
]


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
    target_vertices = set(target.vertices)
    for branch in branch_sets:
        if branch.target not in target_vertices:
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
    witness_edges = {witness.source_edge for witness in witnesses}
    used_edges = tuple(edge for edge in source.edges if edge in witness_edges)
    used_edges_set = set(used_edges)
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
            edge for edge in source.edges if edge not in used_edges_set
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


def _require_minor_find_admission(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    resource_budget: MinorModelFindBudget,
) -> None:
    """Share the catalog find envelope with native callers."""

    if not target.vertices:
        raise OperationDomainValidationError(
            location=("target",),
            code="graph.minor_model.find.empty_target",
            message="the target graph must declare at least one vertex",
        )
    if len(source.vertices) > MAX_MINOR_FIND_SOURCE_VERTICES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="graph.minor_model.find.source_order_bound",
            message=(
                "source order exceeds the admitted find envelope "
                f"({len(source.vertices)} > {MAX_MINOR_FIND_SOURCE_VERTICES})"
            ),
        )
    if len(target.vertices) > MAX_MINOR_FIND_TARGET_VERTICES:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="graph.minor_model.find.target_order_bound",
            message=(
                "target order exceeds the admitted find envelope "
                f"({len(target.vertices)} > {MAX_MINOR_FIND_TARGET_VERTICES})"
            ),
        )


def _empty_minor_find(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    resource_budget: MinorModelFindBudget,
    status: str,
    termination_reason: str,
    candidates_enumerated: int,
) -> MinorModelFindResult:
    return MinorModelFindResult._from_kernel(
        source=source,
        target=target,
        resource_budget=resource_budget,
        status=status,
        branch_sets=(),
        witnesses=(),
        candidates_enumerated=candidates_enumerated,
        termination_reason=termination_reason,
    )


def _crossing_witness(
    owner: dict[str, str],
    first: str,
    second: str,
    source_edge_list: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Return the first source edge crossing the two branches, if any."""

    for left, right in source_edge_list:
        first_owner, second_owner = owner.get(left), owner.get(right)
        if (first_owner == first and second_owner == second) or (
            first_owner == second and second_owner == first
        ):
            return (left, right)
    return None


def _test_branch_assignment(
    members: list[list[str]],
    targets: tuple[str, ...],
    target_edges: tuple[tuple[str, str], ...],
    adjacency: dict[str, set[str]],
    source_edge_list: list[tuple[str, str]],
) -> tuple[tuple[BranchSet, ...], tuple[EdgeWitness, ...]] | None:
    """Test one complete branch assignment, returning a checked candidate or None."""

    if any(not bucket for bucket in members):
        return None
    owner: dict[str, str] = {}
    for index, bucket in enumerate(members):
        frozen = frozenset(bucket)
        if _connected_component(frozen, adjacency) != frozen:
            return None
        for vertex in bucket:
            owner[vertex] = targets[index]
    witnesses: list[EdgeWitness] = []
    for first, second in target_edges:
        crossing = _crossing_witness(owner, first, second, source_edge_list)
        if crossing is None:
            return None
        witnesses.append(EdgeWitness(targets=(first, second), source_edge=crossing))
    branch_sets = tuple(
        BranchSet(target=targets[index], members=tuple(sorted(members[index])))
        for index in range(len(targets))
    )
    return branch_sets, tuple(witnesses)


def find_minor_model(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    resource_budget: MinorModelFindBudget,
) -> MinorModelFindResult:
    """Search for an H-minor model by bounded backtracking over branch assignments.

    Every source vertex is assigned to one target branch or to deletion, in
    source-declaration order; each complete assignment is one tested
    candidate. The first assignment whose branches are all nonempty and
    connected and whose target edges all keep a crossing source edge is
    replayed through the shipped ``check_minor_model`` kernel, and its
    canonical payload is returned. A truncated search returns UNKNOWN, never
    EXHAUSTED.
    """

    _require_minor_find_admission(source, target, resource_budget)
    if len(target.vertices) > len(source.vertices) or len(target.edges) > len(
        source.edges
    ):
        return _empty_minor_find(
            source, target, resource_budget, "EXHAUSTED", "SEARCH_EXHAUSTED", 0
        )
    source_vertices = tuple(source.vertices)
    targets = tuple(target.vertices)
    branch_count = len(targets)
    adjacency = _adjacency(source)
    source_edge_list = sorted({_ordered(edge) for edge in source.edges})
    target_edges = tuple(_ordered(edge) for edge in target.edges)
    budget = resource_budget.max_candidates
    tested = 0
    for assignment in product(range(branch_count + 1), repeat=len(source_vertices)):
        if tested >= budget:
            return _empty_minor_find(
                source,
                target,
                resource_budget,
                "UNKNOWN",
                "CANDIDATE_BUDGET_EXCEEDED",
                tested,
            )
        tested += 1
        members: list[list[str]] = [[] for _ in range(branch_count)]
        for position, label in enumerate(assignment):
            if label < branch_count:
                members[label].append(source_vertices[position])
        candidate = _test_branch_assignment(
            members, targets, target_edges, adjacency, source_edge_list
        )
        if candidate is None:
            continue
        branch_sets, witnesses = candidate
        verdict = check_minor_model(source, target, branch_sets, witnesses)
        if verdict.status != "VALID_MINOR_MODEL":
            continue
        return MinorModelFindResult._from_kernel(
            source=source,
            target=target,
            resource_budget=resource_budget,
            status="FOUND",
            branch_sets=verdict.branch_sets,
            witnesses=verdict.witnesses,
            candidates_enumerated=tested,
            termination_reason="WITNESS_FOUND",
        )
    return _empty_minor_find(
        source, target, resource_budget, "EXHAUSTED", "SEARCH_EXHAUSTED", tested
    )


def _require_topological_admission(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    branch_vertices: tuple[BranchVertex, ...],
    paths: tuple[SubdivisionPath, ...],
) -> None:
    """Share the catalog subdivision-checker envelope with native callers."""

    if not target.vertices:
        raise OperationDomainValidationError(
            location=("target",),
            code="graph.topological_minor.check.empty_target",
            message="the target graph must declare at least one vertex",
        )
    memberships = sum(len(path.vertices) for path in paths)
    if memberships > MAX_MINOR_BRANCH_MEMBERSHIPS:
        raise OperationResourceAdmissionError(
            location=("paths",),
            code="graph.topological_minor.check.membership_bound",
            message="subdivision path memberships exceed the admitted envelope",
        )
    if len(paths) > len(target.edges) + len(branch_vertices):
        raise OperationResourceAdmissionError(
            location=("paths",),
            code="graph.topological_minor.check.path_bound",
            message="path rows exceed the admitted checker envelope",
        )


def _branch_vertices_by_target(
    target: SimpleUndirectedGraph,
    branch_vertices: tuple[BranchVertex, ...],
) -> dict[str, BranchVertex]:
    """Index branch vertices, raising the first coverage obstruction."""

    by_target = {row.target: row for row in branch_vertices}
    if len(by_target) != len(branch_vertices):
        seen: set[str] = set()
        for row in branch_vertices:
            if row.target in seen:
                raise _MinorObstructionError(
                    "DUPLICATE_BRANCH_TARGET",
                    f"target {row.target!r} carries two branch vertices",
                )
            seen.add(row.target)
    for vertex in target.vertices:
        if vertex not in by_target:
            raise _MinorObstructionError(
                "MISSING_BRANCH_VERTEX", f"target {vertex!r} has no branch vertex"
            )
    target_vertices = set(target.vertices)
    for row in branch_vertices:
        if row.target not in target_vertices:
            raise _MinorObstructionError(
                "UNKNOWN_BRANCH_TARGET",
                f"branch target {row.target!r} is not a target vertex",
            )
    return by_target


def _bound_branch_sources(
    target: SimpleUndirectedGraph,
    branch_vertices: tuple[BranchVertex, ...],
    source_vertices: set[str],
) -> dict[str, str]:
    """Bind every target vertex to a distinct declared source vertex."""

    by_target = _branch_vertices_by_target(target, branch_vertices)
    for vertex in target.vertices:
        branch = by_target[vertex].source
        if branch not in source_vertices:
            raise _MinorObstructionError(
                "UNDECLARED_SOURCE_VERTEX",
                f"branch vertex {branch!r} is not a source vertex",
            )
    seen_sources: dict[str, str] = {}
    for vertex in target.vertices:
        branch = by_target[vertex].source
        if branch in seen_sources:
            raise _MinorObstructionError(
                "DUPLICATE_BRANCH_VERTEX",
                f"source vertex {branch!r} binds branch vertices "
                f"{seen_sources[branch]!r} and {vertex!r}",
            )
        seen_sources[branch] = vertex
    return {vertex: by_target[vertex].source for vertex in target.vertices}


def _index_path_witnesses(
    target: SimpleUndirectedGraph,
    paths: tuple[SubdivisionPath, ...],
) -> dict[tuple[str, str], SubdivisionPath]:
    """Index path witnesses, raising coverage obstructions first."""

    if len({path.targets for path in paths}) != len(paths):
        raise OperationDomainValidationError(
            location=("paths",),
            code="graph.topological_minor.duplicate_path",
            message="path targets must be unique",
        )
    target_edge_keys = {_ordered(edge) for edge in target.edges}
    path_by_targets = {_ordered(path.targets): path for path in paths}
    for edge in target.edges:
        if _ordered(edge) not in path_by_targets:
            raise _MinorObstructionError(
                "MISSING_PATH_WITNESS",
                f"target edge {list(_ordered(edge))!r} has no path witness",
            )
    for path in paths:
        if _ordered(path.targets) not in target_edge_keys:
            raise _MinorObstructionError(
                "SUPERFLUOUS_PATH_WITNESS",
                f"path targets {list(path.targets)!r} is not a target edge",
            )
    return path_by_targets


def _replay_subdivision_path(
    path: SubdivisionPath,
    branch_source: dict[str, str],
    source_vertices: set[str],
    source_edges: set[tuple[str, str]],
    branch_source_set: set[str],
    occupied_internals: set[str],
) -> frozenset[str]:
    """Replay one path witness, returning its internal vertices."""

    vertices = path.vertices
    if len(set(vertices)) != len(vertices):
        raise _MinorObstructionError(
            "NON_SIMPLE_PATH",
            f"path for target edge {list(path.targets)!r} repeats a vertex",
        )
    for vertex in vertices:
        if vertex not in source_vertices:
            raise _MinorObstructionError(
                "UNDECLARED_PATH_VERTEX",
                f"path vertex {vertex!r} is not a source vertex",
            )
    first_target, second_target = path.targets
    if (
        vertices[0] != branch_source[first_target]
        or vertices[-1] != branch_source[second_target]
    ):
        raise _MinorObstructionError(
            "PATH_ENDPOINT_MISMATCH",
            f"path for target edge {list(path.targets)!r} does not run "
            f"from {branch_source[first_target]!r} to "
            f"{branch_source[second_target]!r}",
        )
    for left, right in pairwise(vertices):
        if _ordered((left, right)) not in source_edges:
            raise _MinorObstructionError(
                "PATH_EDGE_ABSENT",
                f"path step {[left, right]!r} is absent from the source graph",
            )
    internals = frozenset(vertices[1:-1])
    if internals & branch_source_set:
        raise _MinorObstructionError(
            "PATH_INTERNAL_INTERSECTION",
            f"path for target edge {list(path.targets)!r} runs through a branch vertex",
        )
    if internals & occupied_internals:
        raise _MinorObstructionError(
            "PATH_INTERNAL_INTERSECTION",
            f"path for target edge {list(path.targets)!r} shares an "
            "internal vertex with an earlier path",
        )
    return internals


def check_topological_minor(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    branch_vertices: tuple[BranchVertex, ...],
    paths: tuple[SubdivisionPath, ...],
) -> TopologicalMinorCheckResult:
    """Check one candidate subdivision model deterministically, first obstruction wins."""

    _require_topological_admission(source, target, branch_vertices, paths)

    def invalid(code: str, detail: str) -> TopologicalMinorCheckResult:
        return TopologicalMinorCheckResult._from_kernel(
            source=source,
            target=target,
            status="INVALID_SUBDIVISION",
            branch_vertices=branch_vertices,
            paths=paths,
            used_source_vertices=(),
            deleted_source_vertices=tuple(source.vertices),
            used_source_edges=(),
            deleted_source_edges=tuple(source.edges),
            obstruction_code=code,
            obstruction_detail=detail,
        )

    try:
        branch_source = _bound_branch_sources(
            target, branch_vertices, set(source.vertices)
        )
        path_by_targets = _index_path_witnesses(target, paths)
        source_edges = {_ordered(edge) for edge in source.edges}
        branch_source_set = set(branch_source.values())
        occupied_internals: set[str] = set()
        for edge in target.edges:
            internals = _replay_subdivision_path(
                path_by_targets[_ordered(edge)],
                branch_source,
                set(source.vertices),
                source_edges,
                branch_source_set,
                occupied_internals,
            )
            occupied_internals |= internals
    except _MinorObstructionError as obstruction:
        return invalid(obstruction.code, obstruction.detail)
    ordered_branches = tuple(
        BranchVertex(target=vertex, source=branch_source[vertex])
        for vertex in target.vertices
    )
    ordered_paths = tuple(path_by_targets[_ordered(edge)] for edge in target.edges)
    used_vertices = {branch.source for branch in ordered_branches}
    used_edge_set: set[tuple[str, str]] = set()
    for path in ordered_paths:
        used_vertices.update(path.vertices)
        for left, right in pairwise(path.vertices):
            used_edge_set.add(_ordered((left, right)))
    used_edges = tuple(edge for edge in source.edges if _ordered(edge) in used_edge_set)
    return TopologicalMinorCheckResult._from_kernel(
        source=source,
        target=target,
        status="VALID_SUBDIVISION",
        branch_vertices=ordered_branches,
        paths=ordered_paths,
        used_source_vertices=tuple(
            vertex for vertex in source.vertices if vertex in used_vertices
        ),
        deleted_source_vertices=tuple(
            vertex for vertex in source.vertices if vertex not in used_vertices
        ),
        used_source_edges=used_edges,
        deleted_source_edges=tuple(
            edge for edge in source.edges if edge not in used_edge_set
        ),
        obstruction_code=None,
        obstruction_detail=None,
    )


def verify_topological_minor(claim: TopologicalMinorCheckResult) -> bool:
    """Replay a subdivision verdict against its retained graphs and candidate."""

    try:
        return (
            check_topological_minor(
                claim.source, claim.target, claim.branch_vertices, claim.paths
            )
            == claim
        )
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


class _TopologicalBudgetExhaustedError(Exception):
    """Private signal that the subdivision search spent its probe budget."""


def _require_topological_find_admission(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    resource_budget: TopologicalMinorFindBudget,
) -> None:
    """Share the catalog subdivision-find envelope with native callers."""

    if not target.vertices:
        raise OperationDomainValidationError(
            location=("target",),
            code="graph.topological_minor.find.empty_target",
            message="the target graph must declare at least one vertex",
        )
    if len(source.vertices) > MAX_TOPO_FIND_SOURCE_VERTICES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="graph.topological_minor.find.source_order_bound",
            message=(
                "source order exceeds the admitted find envelope "
                f"({len(source.vertices)} > {MAX_TOPO_FIND_SOURCE_VERTICES})"
            ),
        )
    if len(target.vertices) > MAX_TOPO_FIND_TARGET_VERTICES:
        raise OperationResourceAdmissionError(
            location=("target",),
            code="graph.topological_minor.find.target_order_bound",
            message=(
                "target order exceeds the admitted find envelope "
                f"({len(target.vertices)} > {MAX_TOPO_FIND_TARGET_VERTICES})"
            ),
        )


def _simple_paths(
    adjacency: dict[str, set[str]],
    start: str,
    goal: str,
    forbidden: frozenset[str],
) -> deque[tuple[str, ...]]:
    """Enumerate simple source paths from ``start`` to ``goal`` deterministically.

    Depth-first over sorted neighbors, so repeated calls replay the same
    candidate order. ``forbidden`` holds vertices no routed path may use
    internally; the endpoints are always allowed.
    """

    found: deque[tuple[str, ...]] = deque()
    stack: list[tuple[str, ...]] = [(start,)]
    while stack:
        trail = stack.pop()
        head = trail[-1]
        for neighbor in sorted(adjacency[head], reverse=True):
            if neighbor == goal:
                found.append((*trail, neighbor))
            elif neighbor not in trail and neighbor not in forbidden:
                stack.append((*trail, neighbor))
    return found


def find_topological_minor(
    source: SimpleUndirectedGraph,
    target: SimpleUndirectedGraph,
    resource_budget: TopologicalMinorFindBudget,
) -> TopologicalMinorFindResult:
    """Search for a subdivision by bounded backtracking over branch injections.

    Branch-vertex injections range over the source vertices in declaration
    order; each injection routes the target edges in canonical order through
    internally vertex-disjoint source paths. Every injection trial and every
    candidate path probe spends one probe of the budget. The first complete
    routing is replayed through the shipped ``check_topological_minor``
    kernel, and its canonical payload is returned. A truncated search returns
    UNKNOWN, never EXHAUSTED.
    """

    _require_topological_find_admission(source, target, resource_budget)

    def unresolved(status: str, reason: str, spent: int) -> TopologicalMinorFindResult:
        return TopologicalMinorFindResult._from_kernel(
            source=source,
            target=target,
            resource_budget=resource_budget,
            status=status,
            branch_vertices=(),
            paths=(),
            candidates_enumerated=spent,
            termination_reason=reason,
        )

    if len(target.vertices) > len(source.vertices) or len(target.edges) > len(
        source.edges
    ):
        return unresolved("EXHAUSTED", "SEARCH_EXHAUSTED", 0)
    targets = tuple(target.vertices)
    target_edges = tuple(tuple(sorted(edge)) for edge in target.edges)
    adjacency = _adjacency(source)
    budget = resource_budget.max_candidates
    spent = 0

    def charge() -> None:
        nonlocal spent
        if spent >= budget:
            raise _TopologicalBudgetExhaustedError
        spent += 1

    def route(
        edge_index: int,
        branch: dict[str, str],
        branch_set: frozenset[str],
        occupied: frozenset[str],
        chosen: tuple[SubdivisionPath, ...],
    ) -> tuple[SubdivisionPath, ...] | None:
        if edge_index == len(target_edges):
            return chosen
        first, second = target_edges[edge_index]
        start, goal = branch[first], branch[second]
        forbidden = (branch_set | occupied) - {start, goal}
        for trail in _simple_paths(adjacency, start, goal, frozenset(forbidden)):
            charge()
            internals = frozenset(trail[1:-1])
            if internals & occupied:
                continue
            outcome = route(
                edge_index + 1,
                branch,
                branch_set,
                occupied | internals,
                (
                    *chosen,
                    SubdivisionPath(targets=(first, second), vertices=trail),
                ),
            )
            if outcome is not None:
                return outcome
        return None

    try:
        for injection in permutations(source.vertices, len(targets)):
            charge()
            branch = dict(zip(targets, injection, strict=True))
            routed = route(0, branch, frozenset(injection), frozenset(), ())
            if routed is None:
                continue
            rows = tuple(
                BranchVertex(target=vertex, source=branch[vertex]) for vertex in targets
            )
            verdict = check_topological_minor(source, target, rows, routed)
            if verdict.status != "VALID_SUBDIVISION":
                continue
            return TopologicalMinorFindResult._from_kernel(
                source=source,
                target=target,
                resource_budget=resource_budget,
                status="FOUND",
                branch_vertices=verdict.branch_vertices,
                paths=verdict.paths,
                candidates_enumerated=spent,
                termination_reason="WITNESS_FOUND",
            )
    except _TopologicalBudgetExhaustedError:
        return unresolved("UNKNOWN", "CANDIDATE_BUDGET_EXCEEDED", spent)
    return unresolved("EXHAUSTED", "SEARCH_EXHAUSTED", spent)
