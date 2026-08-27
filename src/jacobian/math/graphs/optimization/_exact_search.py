"""Bounded Z3 encodings for finite-graph optimization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from jacobian.math.graphs.optimization._budget import remaining_ms as _remaining_ms
from jacobian.math.graphs.optimization._coloring_models import ChromaticGraph
from jacobian.math.graphs.optimization._models import (
    GraphDominationMinimumOutput,
    GraphInducedBipartiteMaximumOutput,
    GraphInducedForestMaximumOutput,
    GraphInducedTreeMaximumOutput,
    GraphMinimumMaximalMatchingOutput,
    GraphOptimizationBudget,
    OptimizationSearchStep,
    OptimizationTermination,
)

ThresholdRelation = Literal["AT_MOST", "AT_LEAST"]
type VertexWitness = tuple[str, ...]
type EdgeWitness = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _SearchResult[WitnessT: (VertexWitness, EdgeWitness)]:
    exact: bool
    incumbent: WitnessT
    incumbent_value: int
    lower_bound: int
    upper_bound: int
    tested: tuple[OptimizationSearchStep, ...]
    termination: OptimizationTermination


def _canonical_edges(graph: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (left, right) if left < right else (right, left)
            for left, right in graph.edges
        )
    )


def _search_thresholds[WitnessT: (VertexWitness, EdgeWitness)](
    *,
    direction: Literal["MINIMUM", "MAXIMUM"],
    lower_bound: int,
    upper_bound: int,
    incumbent: WitnessT,
    budget: GraphOptimizationBudget,
    solve: Callable[[int, int], tuple[object, WitnessT]],
    started: float,
) -> _SearchResult[WitnessT]:
    import z3  # type: ignore[import-untyped]

    incumbent_value = len(incumbent)
    relation: ThresholdRelation = "AT_MOST" if direction == "MINIMUM" else "AT_LEAST"
    thresholds = (
        range(lower_bound, incumbent_value)
        if direction == "MINIMUM"
        else range(upper_bound, incumbent_value, -1)
    )
    tested: list[OptimizationSearchStep] = []
    if _remaining_ms(started, budget.wall_seconds) <= 0:
        return _SearchResult(
            False,
            incumbent,
            incumbent_value,
            lower_bound,
            upper_bound,
            (),
            "WALL_TIME",
        )
    for bound in thresholds:
        if len(tested) >= budget.max_solver_calls:
            return _SearchResult(
                False,
                incumbent,
                incumbent_value,
                lower_bound,
                upper_bound,
                tuple(tested),
                "SOLVER_CALL_LIMIT",
            )
        remaining_ms = _remaining_ms(started, budget.wall_seconds)
        if remaining_ms <= 0:
            return _SearchResult(
                False,
                incumbent,
                incumbent_value,
                lower_bound,
                upper_bound,
                tuple(tested),
                "WALL_TIME",
            )
        status, candidate = solve(bound, remaining_ms)
        if status == z3.unknown:
            tested.append(
                OptimizationSearchStep(
                    bound=bound,
                    relation=relation,
                    status="UNKNOWN",
                )
            )
            return _SearchResult(
                False,
                incumbent,
                incumbent_value,
                lower_bound,
                upper_bound,
                tuple(tested),
                "SOLVER_UNKNOWN",
            )
        if status == z3.unsat:
            tested.append(
                OptimizationSearchStep(
                    bound=bound,
                    relation=relation,
                    status="UNSATISFIABLE",
                )
            )
            if direction == "MINIMUM":
                lower_bound = bound + 1
            else:
                upper_bound = bound - 1
            continue

        tested.append(
            OptimizationSearchStep(
                bound=bound,
                relation=relation,
                status="SATISFIABLE",
            )
        )
        candidate_value = len(candidate)
        return _SearchResult(
            True,
            candidate,
            candidate_value,
            candidate_value,
            candidate_value,
            tuple(tested),
            "OPTIMUM_ESTABLISHED",
        )

    return _SearchResult(
        True,
        incumbent,
        incumbent_value,
        incumbent_value,
        incumbent_value,
        tuple(tested),
        "BOUND_CONVERGENCE",
    )


def _vertex_model(
    solver: Any,
    variables: dict[str, Any],
) -> tuple[str, ...]:
    import z3

    model = solver.model()
    return tuple(
        sorted(
            vertex
            for vertex, variable in variables.items()
            if z3.is_true(model.eval(variable, model_completion=True))
        )
    )


def _check_after_encoding(
    solver: Any,
    *,
    started: float,
    budget: GraphOptimizationBudget,
) -> object:
    """Charge encoding and solver completion to one operation deadline."""

    import z3

    remaining_ms = _remaining_ms(started, budget.wall_seconds)
    if remaining_ms <= 0:
        return z3.unknown
    solver.set(timeout=max(1, remaining_ms))
    status = solver.check()
    return z3.unknown if _remaining_ms(started, budget.wall_seconds) <= 0 else status


def solve_domination(
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    started: float,
) -> GraphDominationMinimumOutput:
    vertices = tuple(source.vertices)
    if not vertices:
        return GraphDominationMinimumOutput(
            status="EXACT",
            order=0,
            optimum_value=0,
            incumbent_value=0,
            lower_bound=0,
            upper_bound=0,
            witness_vertices=(),
            tested=(),
            termination_reason="SPECIAL_CASE",
            detail="the empty graph is dominated by the empty set",
        )
    incumbent = tuple(sorted(vertices))

    def solve(bound: int, _timeout_ms: int) -> tuple[object, VertexWitness]:
        import z3

        solver = z3.Solver()
        selected = {
            vertex: z3.Bool(f"dom_{index}") for index, vertex in enumerate(vertices)
        }
        for vertex in vertices:
            solver.add(
                z3.Or(
                    selected[vertex],
                    *(selected[neighbor] for neighbor in graph.neighbors(vertex)),
                )
            )
        solver.add(
            z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in vertices]) <= bound
        )
        status = _check_after_encoding(solver, started=started, budget=budget)
        if status != z3.sat:
            return status, ()
        witness = _vertex_model(solver, selected)
        return (
            (z3.unknown, ())
            if _remaining_ms(started, budget.wall_seconds) <= 0
            else (status, witness)
        )

    result = _search_thresholds(
        direction="MINIMUM",
        lower_bound=1,
        upper_bound=len(incumbent),
        incumbent=incumbent,
        budget=budget,
        solve=solve,
        started=started,
    )
    return GraphDominationMinimumOutput(
        status="EXACT" if result.exact else "UNKNOWN",
        order=len(vertices),
        optimum_value=result.incumbent_value if result.exact else None,
        incumbent_value=result.incumbent_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_vertices=result.incumbent,
        tested=result.tested,
        termination_reason=result.termination,
        detail="bounded Z3 domination threshold search",
    )


def solve_minimum_maximal_matching(
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    started: float,
) -> GraphMinimumMaximalMatchingOutput:
    vertices = tuple(source.vertices)
    edges = _canonical_edges(graph)
    if not edges:
        return GraphMinimumMaximalMatchingOutput(
            status="EXACT",
            order=len(vertices),
            optimum_value=0,
            incumbent_value=0,
            lower_bound=0,
            upper_bound=0,
            witness_edges=(),
            tested=(),
            termination_reason="SPECIAL_CASE",
            detail="an edgeless graph has the empty maximal matching",
        )
    used: set[str] = set()
    greedy_edges: list[tuple[str, str]] = []
    for edge in edges:
        if edge[0] not in used and edge[1] not in used:
            greedy_edges.append(edge)
            used.update(edge)
    incumbent = tuple(greedy_edges)

    def solve(bound: int, _timeout_ms: int) -> tuple[object, EdgeWitness]:
        import z3

        solver = z3.Solver()
        chosen = {edge: z3.Bool(f"match_{index}") for index, edge in enumerate(edges)}
        incident = {
            vertex: tuple(edge for edge in edges if vertex in edge)
            for vertex in vertices
        }
        for vertex in vertices:
            solver.add(
                z3.Sum([z3.If(chosen[edge], 1, 0) for edge in incident[vertex]]) <= 1
            )
        for left, right in edges:
            solver.add(
                z3.Or(*(chosen[edge] for edge in set(incident[left] + incident[right])))
            )
        solver.add(z3.Sum([z3.If(chosen[edge], 1, 0) for edge in edges]) <= bound)
        status = _check_after_encoding(solver, started=started, budget=budget)
        if status != z3.sat:
            return status, ()
        model = solver.model()
        witness = tuple(
            sorted(
                edge
                for edge, variable in chosen.items()
                if z3.is_true(model.eval(variable, model_completion=True))
            )
        )
        return (
            (z3.unknown, ())
            if _remaining_ms(started, budget.wall_seconds) <= 0
            else (status, witness)
        )

    result = _search_thresholds(
        direction="MINIMUM",
        lower_bound=1,
        upper_bound=len(incumbent),
        incumbent=incumbent,
        budget=budget,
        solve=solve,
        started=started,
    )
    return GraphMinimumMaximalMatchingOutput(
        status="EXACT" if result.exact else "UNKNOWN",
        order=len(vertices),
        optimum_value=result.incumbent_value if result.exact else None,
        incumbent_value=result.incumbent_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_edges=result.incumbent,
        tested=result.tested,
        termination_reason=result.termination,
        detail="bounded Z3 minimum-maximal-matching threshold search",
    )


def _maximum_vertex_search(
    *,
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    kind: Literal["FOREST", "TREE", "BIPARTITE"],
    started: float,
) -> _SearchResult[VertexWitness]:
    import networkx as nx

    vertices = tuple(source.vertices)
    if not vertices:
        return _SearchResult(True, (), 0, 0, 0, (), "SPECIAL_CASE")
    incumbent: VertexWitness = (min(vertices),)
    if _remaining_ms(started, budget.wall_seconds) <= 0:
        return _SearchResult(
            False,
            incumbent,
            1,
            1,
            len(vertices),
            (),
            "WALL_TIME",
        )
    whole_valid = (
        nx.is_forest(graph)
        if kind == "FOREST"
        else nx.is_tree(graph)
        if kind == "TREE"
        else nx.is_bipartite(graph)
    )
    if whole_valid:
        witness = tuple(sorted(vertices))
        return _SearchResult(
            True,
            witness,
            len(vertices),
            len(vertices),
            len(vertices),
            (),
            "SPECIAL_CASE",
        )
    edges = _canonical_edges(graph)

    def solve(bound: int, _timeout_ms: int) -> tuple[object, VertexWitness]:
        import z3

        solver = z3.Solver()
        selected = {
            vertex: z3.Bool(f"sel_{index}") for index, vertex in enumerate(vertices)
        }
        cardinality = z3.Sum([z3.If(selected[vertex], 1, 0) for vertex in vertices])
        solver.add(cardinality >= bound)
        if kind == "BIPARTITE":
            colors = {
                vertex: z3.Bool(f"color_{index}")
                for index, vertex in enumerate(vertices)
            }
            for left, right in edges:
                solver.add(
                    z3.Implies(
                        z3.And(selected[left], selected[right]),
                        colors[left] != colors[right],
                    )
                )
        else:
            ranks = {
                vertex: z3.Int(f"rank_{index}") for index, vertex in enumerate(vertices)
            }
            solver.add(z3.Distinct(*ranks.values()))
            for rank in ranks.values():
                solver.add(rank >= 0, rank < len(vertices))
            for vertex in vertices:
                lower_neighbors = [
                    z3.If(
                        z3.And(selected[neighbor], ranks[neighbor] < ranks[vertex]),
                        1,
                        0,
                    )
                    for neighbor in graph.neighbors(vertex)
                ]
                solver.add(
                    z3.Implies(
                        selected[vertex],
                        z3.Sum(lower_neighbors) <= 1,
                    )
                )
            if kind == "TREE":
                selected_edges = z3.Sum(
                    [
                        z3.If(z3.And(selected[left], selected[right]), 1, 0)
                        for left, right in edges
                    ]
                )
                solver.add(cardinality >= 1, selected_edges == cardinality - 1)
        status = _check_after_encoding(solver, started=started, budget=budget)
        if status != z3.sat:
            return status, ()
        witness = _vertex_model(solver, selected)
        return (
            (z3.unknown, ())
            if _remaining_ms(started, budget.wall_seconds) <= 0
            else (status, witness)
        )

    return _search_thresholds(
        direction="MAXIMUM",
        lower_bound=1,
        upper_bound=len(vertices),
        incumbent=incumbent,
        budget=budget,
        solve=solve,
        started=started,
    )


def solve_induced_forest(
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    started: float,
) -> GraphInducedForestMaximumOutput:
    result = _maximum_vertex_search(
        graph=graph, source=source, budget=budget, kind="FOREST", started=started
    )
    return GraphInducedForestMaximumOutput(
        status="EXACT" if result.exact else "UNKNOWN",
        order=len(source.vertices),
        optimum_value=result.incumbent_value if result.exact else None,
        incumbent_value=result.incumbent_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_vertices=result.incumbent,
        tested=result.tested,
        termination_reason=result.termination,
        detail="bounded Z3 maximum induced-forest threshold search",
    )


def solve_induced_tree(
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    started: float,
) -> GraphInducedTreeMaximumOutput:
    result = _maximum_vertex_search(
        graph=graph, source=source, budget=budget, kind="TREE", started=started
    )
    return GraphInducedTreeMaximumOutput(
        status="EXACT" if result.exact else "UNKNOWN",
        order=len(source.vertices),
        optimum_value=result.incumbent_value if result.exact else None,
        incumbent_value=result.incumbent_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_vertices=result.incumbent,
        tested=result.tested,
        termination_reason=result.termination,
        detail="bounded Z3 maximum induced-tree threshold search",
    )


def solve_induced_bipartite(
    graph: Any,
    source: ChromaticGraph,
    budget: GraphOptimizationBudget,
    started: float,
) -> GraphInducedBipartiteMaximumOutput:
    result = _maximum_vertex_search(
        graph=graph,
        source=source,
        budget=budget,
        kind="BIPARTITE",
        started=started,
    )
    return GraphInducedBipartiteMaximumOutput(
        status="EXACT" if result.exact else "UNKNOWN",
        order=len(source.vertices),
        optimum_value=result.incumbent_value if result.exact else None,
        incumbent_value=result.incumbent_value,
        lower_bound=result.lower_bound,
        upper_bound=result.upper_bound,
        witness_vertices=result.incumbent,
        tested=result.tested,
        termination_reason=result.termination,
        detail="bounded Z3 maximum induced-bipartite threshold search",
    )
