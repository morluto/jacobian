"""NetworkX and Z3 composition for bounded graph optimization."""

from __future__ import annotations

import time
from typing import Any

from jacobian.contracts.graph_coloring import (
    ChromaticGraph,
    ChromaticSearchStep,
    GraphChromaticNumberOutput,
)
from jacobian.graphs.coloring_semantics import canonical_graph, coloring_cnf

__all__ = ["build_simple_graph", "solve_chromatic_number"]


def build_simple_graph(graph: ChromaticGraph) -> Any:
    """Build a networkx Graph from a validated :class:`ChromaticGraph`.

    The contract has already enforced uniqueness, no self-loops, and edge
    endpoints within the vertex set, so this is a thin structural
    projection.  It raises only on a contract-internal inconsistency.
    """

    import networkx as nx

    g: nx.Graph[Any] = nx.Graph()
    g.add_nodes_from(graph.vertices)
    for left, right in graph.edges:
        g.add_edge(left, right)
    return g


def solve_chromatic_number(
    networkx_graph: Any,
    *,
    graph: ChromaticGraph,
    vertices: tuple[str, ...],
    wall_seconds: int,
) -> GraphChromaticNumberOutput:
    """Bounded Z3 k-colorability search returning EXACT or UNKNOWN.

    The result is always contract-valid.  EXACT carries the chromatic
    number, matching bounds, and a witness coloring.  UNKNOWN carries the
    partial lower/upper bounds, the incumbent greedy coloring, and the
    tested k-colorability trace; it is never a negative conclusion.
    """

    import networkx as nx
    import z3  # type: ignore[import-untyped]

    n = len(vertices)
    if n == 0:
        return GraphChromaticNumberOutput(
            status="EXACT",
            vertices=vertices,
            order=0,
            chromatic_number=0,
            lower_bound=0,
            upper_bound=0,
            coloring={},
            solver_status="SPECIAL_CASE",
            tested=(),
            detail="the empty graph requires zero colors",
        )

    greedy = nx.coloring.greedy_color(
        networkx_graph,
        strategy="saturation_largest_first",
    )
    upper_bound = max(greedy.values(), default=-1) + 1
    lower_bound = 2 if networkx_graph.number_of_edges() else 1
    if upper_bound == lower_bound:
        return GraphChromaticNumberOutput(
            status="EXACT",
            vertices=vertices,
            order=n,
            chromatic_number=upper_bound,
            lower_bound=upper_bound,
            upper_bound=upper_bound,
            coloring={str(node): int(color) for node, color in greedy.items()},
            solver_status="SPECIAL_CASE",
            tested=(),
            detail="a maintained greedy coloring and graph edge bound coincide",
        )

    started = time.monotonic()
    tested: list[ChromaticSearchStep] = []
    encoded_graph = canonical_graph(graph)
    for colors in range(lower_bound, upper_bound + 1):
        remaining_ms = int((wall_seconds - (time.monotonic() - started)) * 1000)
        if remaining_ms <= 0:
            return _unknown_chromatic_result(
                vertices=vertices,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                coloring=greedy,
                tested=tested,
                detail="the chromatic-number wall-clock budget expired",
            )
        solver = z3.Solver()
        solver.set(timeout=max(1, remaining_ms))
        variable_names, clauses = coloring_cnf(encoded_graph, colors)
        variables = {
            index: z3.Bool(name) for index, name in enumerate(variable_names, start=1)
        }
        for clause in clauses:
            literals = tuple(
                variables[abs(literal)]
                if literal > 0
                else z3.Not(variables[abs(literal)])
                for literal in clause
            )
            solver.add(z3.Or(*literals))

        result = solver.check()
        if result == z3.unknown:
            tested.append(ChromaticSearchStep(colors=colors, status="UNKNOWN"))
            return _unknown_chromatic_result(
                vertices=vertices,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                coloring=greedy,
                tested=tested,
                detail=(
                    "Z3 did not settle the k-colorability decision within the "
                    "remaining wall-clock budget"
                ),
            )
        if result == z3.unsat:
            tested.append(ChromaticSearchStep(colors=colors, status="UNSATISFIABLE"))
            lower_bound = colors + 1
            continue

        tested.append(ChromaticSearchStep(colors=colors, status="SATISFIABLE"))
        model = solver.model()
        coloring = {
            node: next(
                color
                for color in range(colors)
                if z3.is_true(
                    model.eval(
                        variables[index * colors + color + 1],
                        model_completion=True,
                    )
                )
            )
            for index, node in enumerate(encoded_graph.vertices)
        }
        return GraphChromaticNumberOutput(
            status="EXACT",
            vertices=vertices,
            order=n,
            chromatic_number=colors,
            lower_bound=colors,
            upper_bound=colors,
            coloring={str(node): int(color) for node, color in coloring.items()},
            solver_status="SATISFIABLE",
            tested=tuple(tested),
            detail="Z3 found the first satisfying k after settling all smaller k",
        )

    return _unknown_chromatic_result(
        vertices=vertices,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        coloring=greedy,
        tested=tested,
        detail="the solver did not produce a coloring through the valid upper bound",
    )


def _unknown_chromatic_result(
    *,
    vertices: tuple[str, ...],
    lower_bound: int,
    upper_bound: int,
    coloring: dict[Any, int],
    tested: list[ChromaticSearchStep],
    detail: str,
) -> GraphChromaticNumberOutput:
    return GraphChromaticNumberOutput(
        status="UNKNOWN",
        vertices=vertices,
        order=len(vertices),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        coloring={str(node): int(color) for node, color in coloring.items()},
        solver_status="UNKNOWN",
        tested=tuple(tested),
        detail=detail,
    )
