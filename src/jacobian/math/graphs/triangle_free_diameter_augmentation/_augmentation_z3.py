"""Private Z3 backend for triangle-free diameter augmentation."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.triangle_free_diameter_augmentation._models import (
    TriangleFreeDiameterAugmentationBudget,
    TriangleFreeDiameterAugmentationResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.process import (
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

_AUGMENTATION_WORKER = Path(__file__).with_name("_augmentation_worker.py")
_WORKER_OUTPUT_BYTES = 64 * 1024
_WORKER_ERROR_BYTES = 16_384
_WORKER_ADDRESS_SPACE_BYTES = 1_536 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1_024 * 1_024

# Conservative backend-specific envelope calibrated against Z3.
# These are mathematical work bounds, not transport limits.
HARD_MAX_ORDER = 12
HARD_MAX_TARGET_DIAMETER = 12
HARD_MAX_CANDIDATES = 55
HARD_MAX_TRIANGLE_CONSTRAINTS = 500
HARD_MAX_REACHABILITY_VARS = 1000  # n * n * r
HARD_MAX_OUTPUT_EDGES = 55


def _graph_from_value(graph: SimpleUndirectedGraph) -> Any:
    import networkx as nx

    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from(graph.vertices)
    g.add_edges_from(graph.edges)
    return g


def _is_connected(graph: SimpleUndirectedGraph) -> bool:
    import networkx as nx

    if not graph.vertices:
        return False
    g = _graph_from_value(graph)
    return bool(nx.is_connected(g))


def _is_triangle_free(graph: SimpleUndirectedGraph) -> bool:
    import networkx as nx

    g = _graph_from_value(graph)
    try:
        tri = sum(nx.triangles(g).values()) // 3  # type: ignore[union-attr]
    except Exception:
        return False
    return tri == 0


def _diameter(graph: SimpleUndirectedGraph) -> int | None:
    import networkx as nx

    g = _graph_from_value(graph)
    if not g or not nx.is_connected(g):
        return None
    return int(nx.diameter(g))


def _sorted_vertices(graph: SimpleUndirectedGraph) -> tuple[str, ...]:
    return tuple(sorted(graph.vertices))


def _derive_candidates(
    graph: SimpleUndirectedGraph,
) -> tuple[tuple[str, ...], list[tuple[str, str]], dict[tuple[int, int], int]]:
    vertices = _sorted_vertices(graph)
    n = len(vertices)
    index = {v: i for i, v in enumerate(vertices)}
    original_set = {tuple(sorted(e)) for e in graph.edges}
    # adjacency set per index
    adj: list[set[int]] = [set() for _ in range(n)]
    for left, right in graph.edges:
        li = index[left]
        ri = index[right]
        adj[li].add(ri)
        adj[ri].add(li)
    candidates: list[tuple[str, str]] = []
    cand_index: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            vi = vertices[i]
            vj = vertices[j]
            edge = (vi, vj) if vi < vj else (vj, vi)
            if edge in original_set:
                continue
            # common neighbor check in G
            common = adj[i] & adj[j]
            if common:
                continue
            cand_index[(i, j)] = len(candidates)
            candidates.append(edge)
    return vertices, candidates, cand_index


def _triangle_constraints(
    vertices: tuple[str, ...],
    candidates: list[tuple[str, str]],
    cand_index: dict[tuple[int, int], int],
    original_set: set[tuple[str, str]],
) -> list[tuple[int, ...]]:
    n = len(vertices)
    # map candidate edge label to idx for quick lookup via sorted labels
    # cand_index already uses (i,j) with sorted vertex order indices (by sorted vertices)
    # original set uses string labels sorted
    constraints: list[tuple[int, ...]] = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                pairs = [(a, b), (a, c), (b, c)]
                statuses: list[str] = []
                cand_ids: list[int] = []
                orig_count = 0
                for u, v in pairs:
                    # labels for original set lookup
                    lu = vertices[u]
                    lv = vertices[v]
                    edge_label = (lu, lv) if lu < lv else (lv, lu)
                    if edge_label in original_set:
                        statuses.append("orig")
                        orig_count += 1
                    elif (u, v) in cand_index:
                        statuses.append("cand")
                        cand_ids.append(cand_index[(u, v)])
                    else:
                        statuses.append("forbid")
                if orig_count == 1 and len(cand_ids) == 2:
                    # need both candidates to create triangle
                    constraints.append((cand_ids[0], cand_ids[1]))
                elif orig_count == 0 and len(cand_ids) == 3:
                    constraints.append((cand_ids[0], cand_ids[1], cand_ids[2]))
                # orig_count==2 cannot happen with cand because would have common neighbor filtered
                # orig_count==3 would be triangle in source, already rejected
    return constraints


def _require_admitted_request(
    graph: SimpleUndirectedGraph,
    target_diameter: int,
    budget: TriangleFreeDiameterAugmentationBudget,
) -> tuple[tuple[str, ...], list[tuple[str, str]], list[tuple[int, ...]]]:
    # basic semantic admission before heavy derivation
    n = len(graph.vertices)
    if n == 0:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.empty_graph",
            message="augmentation requires a nonempty graph",
        )
    if n > budget.max_order:
        raise OperationDomainValidationError(
            location=("resource_budget", "max_order"),
            code="graph.triangle_free_diameter_augmentation.max_order_budget",
            message="graph order exceeds the declared max_order budget",
        )
    if n > HARD_MAX_ORDER:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.order_bound",
            message=f"augmentation supports order at most {HARD_MAX_ORDER}",
        )
    if target_diameter < 1 or target_diameter > HARD_MAX_TARGET_DIAMETER:
        raise OperationDomainValidationError(
            location=("target_diameter",),
            code="graph.triangle_free_diameter_augmentation.diameter_bound",
            message=f"target diameter must be in 1..{HARD_MAX_TARGET_DIAMETER}",
        )
    # connected and triangle-free checks before backend
    if not _is_connected(graph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.not_connected",
            message="augmentation requires a connected graph",
        )
    if not _is_triangle_free(graph):
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.not_triangle_free",
            message="augmentation requires a triangle-free graph",
        )
    # derive candidates and constraints
    vertices, candidates, cand_index = _derive_candidates(graph)
    m = len(candidates)
    if m > HARD_MAX_CANDIDATES:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.candidate_bound",
            message=f"candidate non-edges {m} exceeds bound {HARD_MAX_CANDIDATES}",
        )
    if m > HARD_MAX_OUTPUT_EDGES:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.output_bound",
            message="output edge bound exceeded",
        )
    original_set2: set[tuple[str, str]] = {tuple(sorted(e)) for e in graph.edges}  # type: ignore[misc]
    constraints = _triangle_constraints(vertices, candidates, cand_index, original_set2)
    if len(constraints) > HARD_MAX_TRIANGLE_CONSTRAINTS:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.triangle_free_diameter_augmentation.triangle_constraint_bound",
            message="triangle-compatibility constraints exceed bound",
        )
    reach_vars = n * n * target_diameter
    if reach_vars > HARD_MAX_REACHABILITY_VARS:
        raise OperationDomainValidationError(
            location=("target_diameter",),
            code="graph.triangle_free_diameter_augmentation.reachability_bound",
            message=f"reachability encoding {reach_vars} exceeds bound {HARD_MAX_REACHABILITY_VARS}",
        )
    # also bound candidate non-edges * r product? Already covered.
    return vertices, candidates, constraints


def _solve_augmentation_kernel(  # noqa: C901
    graph: SimpleUndirectedGraph,
    target_diameter: int,
    budget: TriangleFreeDiameterAugmentationBudget,
) -> TriangleFreeDiameterAugmentationResult:
    started = time.monotonic()
    # admission (also derives vertices/candidates)
    vertices, candidates, triangle_constraints = _require_admitted_request(
        graph, target_diameter, budget
    )
    n = len(vertices)
    m = len(candidates)
    # quick zero case: original already satisfies diameter
    orig_diam = _diameter(graph)
    if orig_diam is not None and orig_diam <= target_diameter:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="EXACT",
            added_edges=(),
            augmented_diameter=orig_diam,
            detail="original graph already satisfies the target diameter",
        )
    # If no candidates, cannot augment (but diameter not satisfied => infeasible)
    if m == 0:
        # no legal edge to add, diameter remains > target => infeasible
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="INFEASIBLE",
            added_edges=(),
            augmented_diameter=None,
            detail="no triangle-free augmentation achieves the target diameter",
        )
    # Build Z3 model
    import z3

    # timeout handling helper
    def remaining_ms() -> int:
        return int((budget.wall_seconds - (time.monotonic() - started)) * 1000)

    if remaining_ms() <= 0:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="SOLVER_BUDGET_EXCEEDED",
            added_edges=(),
            augmented_diameter=None,
            detail="wall-clock budget expired before solver startup",
        )
    # variables for candidates
    xs = [z3.Bool(f"x_{i}") for i in range(m)]
    # index lookup for adjacency expression
    index = {v: i for i, v in enumerate(vertices)}
    original_set = {tuple(sorted(e)) for e in graph.edges}
    # cached candidate index for adjacency
    _cand_index_cache: dict[tuple[int, int], int] = {}
    for idx, (lu, lv) in enumerate(candidates):
        iu = index[lu]
        iv = index[lv]
        a, b = (iu, iv) if iu < iv else (iv, iu)
        _cand_index_cache[(a, b)] = idx

    def adj_expr(i: int, j: int) -> Any:
        if i == j:
            return z3.BoolVal(False)
        a, b = (i, j) if i < j else (j, i)
        lu = vertices[a]
        lv = vertices[b]
        edge_label = (lu, lv) if lu < lv else (lv, lu)
        if edge_label in original_set:
            return z3.BoolVal(True)
        if (a, b) in _cand_index_cache:
            return xs[_cand_index_cache[(a, b)]]
        return z3.BoolVal(False)

    # Create reachability variables reach[k][i][j] for k=1..r
    r = target_diameter
    reach: dict[tuple[int, int, int], Any] = {}
    for k in range(1, r + 1):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                reach[(k, i, j)] = z3.Bool(f"reach_{k}_{i}_{j}")

    solver = z3.Solver()
    for clause in triangle_constraints:
        if len(clause) == 2:
            a, b = clause
            solver.add(z3.Or(z3.Not(xs[a]), z3.Not(xs[b])))
        elif len(clause) == 3:
            a, b, c = clause
            solver.add(z3.Or(z3.Not(xs[a]), z3.Not(xs[b]), z3.Not(xs[c])))

    # k=1 constraints
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            var = reach[(1, i, j)]
            solver.add(var == adj_expr(i, j))
    # k>1 constraints with exact equality
    for k in range(2, r + 1):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                var = reach[(k, i, j)]
                prev = reach[(k - 1, i, j)]
                or_terms: list[Any] = [prev]
                for t in range(n):
                    if t == j:
                        continue
                    reach_it = z3.BoolVal(True) if i == t else reach[(k - 1, i, t)]
                    atj = adj_expr(t, j)
                    or_terms.append(z3.And(reach_it, atj))
                solver.add(
                    var == z3.Or(*or_terms) if len(or_terms) > 1 else var == or_terms[0]
                )

    # Diameter constraints: for all i != j, reach_r[i][j] must be True
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            solver.add(reach[(r, i, j)] == z3.BoolVal(True))

    # Quick feasibility check without cardinality (already diameter constraints present)
    # Set timeout
    ms = remaining_ms()
    if ms <= 0:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="SOLVER_BUDGET_EXCEEDED",
            added_edges=(),
            augmented_diameter=None,
            detail="wall-clock budget expired before feasibility check",
        )
    solver.set(timeout=max(1, ms))
    res = solver.check()
    if res == z3.unknown:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="SOLVER_BUDGET_EXCEEDED",
            added_edges=(),
            augmented_diameter=None,
            detail="solver did not settle feasibility within budget",
        )
    if res == z3.unsat:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="INFEASIBLE",
            added_edges=(),
            augmented_diameter=None,
            detail="no triangle-free augmentation achieves the target diameter",
        )
    # Feasible, now minimize cardinality via iterative bound
    # Use push/pop to test increasing k
    # Extract model for upper bound may give any feasible solution; but we need minimal.
    # We'll iterate k from 0..m
    # Note: if original already feasible, we already returned, so minimal k >=1 possibly
    # Need to protect monotonic feasibility via Sum <= k is monotone
    for k in range(m + 1):
        # check deadline
        if remaining_ms() <= 0:
            return TriangleFreeDiameterAugmentationResult._from_kernel(
                graph=graph,
                target_diameter=target_diameter,
                status="SOLVER_BUDGET_EXCEEDED",
                added_edges=(),
                augmented_diameter=None,
                detail="wall-clock budget expired during minimization",
            )
        solver.push()
        # cardinality constraint Sum(If(xs[i],1,0)) <= k
        card = z3.Sum([z3.If(xs[i], 1, 0) for i in range(m)]) if m > 0 else z3.IntVal(0)
        solver.add(card <= k)
        ms2 = remaining_ms()
        if ms2 <= 0:
            solver.pop()
            return TriangleFreeDiameterAugmentationResult._from_kernel(
                graph=graph,
                target_diameter=target_diameter,
                status="SOLVER_BUDGET_EXCEEDED",
                added_edges=(),
                augmented_diameter=None,
                detail="wall-clock budget expired before cardinality check",
            )
        solver.set(timeout=max(1, ms2))
        res2 = solver.check()
        if res2 == z3.sat:
            model = solver.model()
            selected: list[tuple[str, str]] = []
            for idx, edge in enumerate(candidates):
                val = model.eval(xs[idx], model_completion=True)
                if z3.is_true(val):
                    selected.append(edge)
            selected_t = tuple(sorted(selected))
            # Validate via NetworkX before returning
            import networkx as nx

            g_aug: nx.Graph[str] = nx.Graph()
            g_aug.add_nodes_from(graph.vertices)
            g_aug.add_edges_from(graph.edges)
            g_aug.add_edges_from(selected_t)
            # triangle-free check
            tri = sum(nx.triangles(g_aug).values()) // 3  # type: ignore[union-attr]
            if tri != 0:
                solver.pop()
                # should not happen due to constraints, treat as budget exceeded? but fail closed
                return TriangleFreeDiameterAugmentationResult._from_kernel(
                    graph=graph,
                    target_diameter=target_diameter,
                    status="SOLVER_BUDGET_EXCEEDED",
                    added_edges=(),
                    augmented_diameter=None,
                    detail="solver witness violates triangle-free invariant",
                )
            if not nx.is_connected(g_aug):
                solver.pop()
                return TriangleFreeDiameterAugmentationResult._from_kernel(
                    graph=graph,
                    target_diameter=target_diameter,
                    status="SOLVER_BUDGET_EXCEEDED",
                    added_edges=(),
                    augmented_diameter=None,
                    detail="solver witness is disconnected",
                )
            diam = int(nx.diameter(g_aug))
            if diam > target_diameter:
                solver.pop()
                return TriangleFreeDiameterAugmentationResult._from_kernel(
                    graph=graph,
                    target_diameter=target_diameter,
                    status="SOLVER_BUDGET_EXCEEDED",
                    added_edges=(),
                    augmented_diameter=None,
                    detail="solver witness exceeds target diameter",
                )
            solver.pop()
            return TriangleFreeDiameterAugmentationResult._from_kernel(
                graph=graph,
                target_diameter=target_diameter,
                status="EXACT",
                added_edges=selected_t,
                augmented_diameter=diam,
                detail="bounded Z3 optimization proved minimal cardinality",
            )
        elif res2 == z3.unknown:
            solver.pop()
            return TriangleFreeDiameterAugmentationResult._from_kernel(
                graph=graph,
                target_diameter=target_diameter,
                status="SOLVER_BUDGET_EXCEEDED",
                added_edges=(),
                augmented_diameter=None,
                detail="solver did not settle cardinality check within budget",
            )
        else:  # unsat, need larger k
            solver.pop()
            continue
    # If loop finishes without sat, infeasible (should have been caught)
    return TriangleFreeDiameterAugmentationResult._from_kernel(
        graph=graph,
        target_diameter=target_diameter,
        status="INFEASIBLE",
        added_edges=(),
        augmented_diameter=None,
        detail="no feasible augmentation within candidate bound",
    )


def _augmentation_worker_stdout_limit(graph: SimpleUndirectedGraph) -> int:
    projection = {
        "status": "EXACT",
        "target_diameter": 6,
        "added_edge_count": 12,
        "added_edges": [[f"v{i}", f"v{j}"] for i in range(6) for j in range(i + 1, 6)][
            :12
        ],
        "augmented_diameter": 2,
        "detail": "x" * 1024,
    }
    # Include graph dump size rough
    base = json.dumps(projection, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    graph_bytes = json.dumps(
        graph.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return len(base) + len(graph_bytes) + 1024


def solve_triangle_free_diameter_augmentation_values(
    graph: SimpleUndirectedGraph,
    target_diameter: int,
    budget: TriangleFreeDiameterAugmentationBudget,
) -> TriangleFreeDiameterAugmentationResult:
    """Run bounded augmentation in owner worker and decode result."""

    # This exact no-op path never constructs candidates or invokes Z3, so it
    # is admitted independently of the backend order envelope.
    if (
        len(graph.vertices) > budget.max_order
        or len(graph.vertices) > HARD_MAX_ORDER
    ) and _is_connected(graph) and _is_triangle_free(graph):
        original_diameter = _diameter(graph)
        if original_diameter is not None and original_diameter <= target_diameter:
            return TriangleFreeDiameterAugmentationResult._from_kernel(
                graph=graph,
                target_diameter=target_diameter,
                status="EXACT",
                added_edges=(),
                augmented_diameter=original_diameter,
                detail="original graph already satisfies the target diameter",
            )

    # Shared admission before worker (native and MCP parity)
    _require_admitted_request(graph, target_diameter, budget)

    def fallback(detail: str) -> TriangleFreeDiameterAugmentationResult:
        return TriangleFreeDiameterAugmentationResult._from_kernel(
            graph=graph,
            target_diameter=target_diameter,
            status="SOLVER_BUDGET_EXCEEDED",
            added_edges=(),
            augmented_diameter=None,
            detail=detail,
        )

    deadline = time.monotonic() + budget.wall_seconds
    try:
        with TemporaryDirectory(prefix="jacobian-graph-augmentation-") as directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return fallback("augmentation request expired before worker startup")
            completed = run_bounded_process(
                [sys.executable, str(_AUGMENTATION_WORKER)],
                input_bytes=json.dumps(
                    {
                        "graph": graph.model_dump(mode="json"),
                        "target_diameter": target_diameter,
                        "resource_budget": budget.model_dump(mode="json"),
                    },
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8"),
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_augmentation_worker_stdout_limit(graph),
                stderr_limit=_WORKER_ERROR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=max(1, math.ceil(budget.wall_seconds)),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=directory,
            )
    except OSError:
        return fallback("bounded augmentation worker could not be started")
    if (
        completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return fallback("bounded augmentation worker did not establish an outcome")
    if completed.returncode != 0:
        raise RuntimeError(
            "bounded augmentation worker failed before establishing an outcome"
        )
    if time.monotonic() >= deadline:
        return fallback("augmentation request expired before response validation")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
        # payload excludes graph; reattach
        result = TriangleFreeDiameterAugmentationResult.model_validate(
            {**payload, "graph": graph.model_dump(mode="json")}
        )
        # additional binding checks: status vs graph order already via model
        if result.target_diameter != target_diameter:
            raise ValueError("worker target mismatch")
        return (
            result
            if time.monotonic() < deadline
            else fallback("request expired during validation")
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return fallback("bounded augmentation worker returned malformed output")


# Re-export helper for tests
__all__ = [
    "HARD_MAX_CANDIDATES",
    "HARD_MAX_ORDER",
    "HARD_MAX_REACHABILITY_VARS",
    "HARD_MAX_TARGET_DIAMETER",
    "HARD_MAX_TRIANGLE_CONSTRAINTS",
    "_derive_candidates",
    "_triangle_constraints",
    "solve_triangle_free_diameter_augmentation_values",
]
