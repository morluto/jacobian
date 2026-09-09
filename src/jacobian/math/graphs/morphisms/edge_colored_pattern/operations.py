"""Bounded edge-coloured monomorphism search."""

import time
from collections import Counter

from pydantic_core import PydanticCustomError

from jacobian._execution import (
    OperationWorkLedger,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.morphisms.operations import _graph_label_characters
from jacobian.math.graphs.values import ColoredUndirectedGraph

from ._models import EdgeColoredPatternResult, require_edge_colors

MAX_ASSIGNMENTS = 10_000_000
MAX_WORK = 50_000_000
MAX_RETAINED_CHARACTERS = 20_000_000


def _reject(reason: str, message: str) -> None:
    raise OperationResourceAdmissionError(
        location=("pattern", "host"),
        code=f"graph.edge_colored_pattern.{reason}",
        message=message,
    )


def edge_colored_subgraph_pattern_find(
    pattern: ColoredUndirectedGraph, host: ColoredUndirectedGraph
) -> EdgeColoredPatternResult:
    for name, value in (("pattern", pattern), ("host", host)):
        if not isinstance(value, ColoredUndirectedGraph):
            raise TypeError(f"{name} must be a ColoredUndirectedGraph")
        try:
            require_edge_colors(value)
        except PydanticCustomError as exc:
            raise OperationDomainValidationError(
                location=(name,),
                code="graph.edge_colored_pattern.color_domain",
                message=str(exc),
            ) from exc
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return edge_colored_subgraph_pattern_find(pattern, host)
    deadline = execution.started_at + 60
    bind_request_deadline(
        min(deadline, execution.deadline)
        if execution.deadline is not None
        else deadline
    )
    request_checkpoint("before colored embedding admission")
    k, n = len(pattern.graph.vertices), len(host.graph.vertices)
    if k > 64:
        _reject("pattern_order", "pattern must have at most 64 vertices")
    retained = (
        _graph_label_characters(pattern.graph)
        + _graph_label_characters(host.graph)
        + sum(map(len, pattern.edge_colors))
        + sum(map(len, host.edge_colors))
        + k * max(map(len, host.graph.vertices), default=0)
    )
    if retained > MAX_RETAINED_CHARACTERS:
        _reject(
            "output", "retained sources and witness exceed their label allocation bound"
        )
    # The canonical source envelope bounds these linear presolves independently
    # of search. An injective edge map preserves each color's edge cardinality.
    pattern_counts, host_counts = (
        Counter(pattern.edge_colors),
        Counter(host.edge_colors),
    )
    impossible = k > n or any(
        count > host_counts[color] for color, count in pattern_counts.items()
    )
    host_edges = dict(zip(host.graph.edges, host.edge_colors, strict=True))
    identity = set(pattern.graph.vertices) <= set(host.graph.vertices) and all(
        host_edges.get(edge) == color
        for edge, color in zip(pattern.graph.edges, pattern.edge_colors, strict=True)
    )
    setup_work = (
        retained
        + 3 * len(pattern.graph.edges)
        + 3 * len(host.graph.edges)
        + n * (k + 2 * len(pattern.graph.edges))
    )
    if setup_work > MAX_WORK:
        _reject("work", "colored embedding setup exceeds its admitted work bound")
    request_checkpoint("after colored embedding admission")
    found: tuple[str, ...] | None = None
    if identity and not impossible:
        found = pattern.graph.vertices
    elif not impossible:
        found = _search(
            pattern,
            host,
            candidate_ledger=OperationWorkLedger(MAX_ASSIGNMENTS),
            work_ledger=OperationWorkLedger(MAX_WORK, consumed=setup_work),
        )
    request_checkpoint("before colored embedding result construction")
    return EdgeColoredPatternResult(
        pattern=pattern,
        host=host,
        decision="EXISTS" if found is not None else "DOES_NOT_EXIST",
        vertex_map=found if found is not None else (),
    )


def _search(
    pattern: ColoredUndirectedGraph,
    host: ColoredUndirectedGraph,
    *,
    candidate_ledger: OperationWorkLedger | None = None,
    work_ledger: OperationWorkLedger | None = None,
) -> tuple[str, ...] | None:
    pattern_index = {label: i for i, label in enumerate(pattern.graph.vertices)}
    host_index = {label: i for i, label in enumerate(host.graph.vertices)}
    palette = {
        color: i
        for i, color in enumerate(
            dict.fromkeys((*pattern.edge_colors, *host.edge_colors))
        )
    }
    pattern_adjacency: list[dict[int, int]] = [{} for _ in pattern.graph.vertices]
    pattern_color_degrees: list[Counter[int]] = [
        Counter() for _ in pattern.graph.vertices
    ]
    for (u, v), color in zip(pattern.graph.edges, pattern.edge_colors, strict=True):
        u_index, v_index, color_index = (
            pattern_index[u],
            pattern_index[v],
            palette[color],
        )
        pattern_adjacency[u_index][v_index] = color_index
        pattern_adjacency[v_index][u_index] = color_index
        pattern_color_degrees[u_index][color_index] += 1
        pattern_color_degrees[v_index][color_index] += 1
    available = {
        (min(host_index[u], host_index[v]), max(host_index[u], host_index[v])): palette[
            color
        ]
        for (u, v), color in zip(host.graph.edges, host.edge_colors, strict=True)
    }
    host_color_degrees: list[Counter[int]] = [Counter() for _ in host.graph.vertices]
    for (u, v), color in zip(host.graph.edges, host.edge_colors, strict=True):
        u_index, v_index, color_index = host_index[u], host_index[v], palette[color]
        host_color_degrees[u_index][color_index] += 1
        host_color_degrees[v_index][color_index] += 1
    host_degrees = [sum(color_degrees.values()) for color_degrees in host_color_degrees]
    if candidate_ledger is None:
        candidate_ledger = OperationWorkLedger(MAX_ASSIGNMENTS)
    if work_ledger is None:
        work_ledger = OperationWorkLedger(MAX_WORK)
    candidate_domains = [
        tuple(
            host_vertex
            for host_vertex, host_color_degree in enumerate(host_color_degrees)
            if host_degrees[host_vertex] >= len(pattern_adjacency[pattern_vertex])
            and all(
                host_color_degree[color] >= count
                for color, count in pattern_color_degrees[pattern_vertex].items()
            )
        )
        for pattern_vertex in range(len(pattern.graph.vertices))
    ]
    pattern_order = sorted(
        range(len(pattern.graph.vertices)),
        key=lambda vertex: (
            len(candidate_domains[vertex]),
            -len(pattern_adjacency[vertex]),
            vertex,
        ),
    )
    assignment = [-1] * len(pattern.graph.vertices)
    used_host_vertices: set[int] = set()

    def backtrack(position: int) -> bool:
        if position == len(pattern_order):
            return True
        pattern_vertex = pattern_order[position]
        for host_vertex in candidate_domains[pattern_vertex]:
            candidate_ledger.charge()
            work_ledger.charge()
            request_checkpoint("during colored embedding search")
            if host_vertex in used_host_vertices:
                continue
            compatible = True
            for neighbor, color in pattern_adjacency[pattern_vertex].items():
                mapped_neighbor = assignment[neighbor]
                if mapped_neighbor == -1:
                    continue
                work_ledger.charge()
                if (
                    available.get(
                        (
                            min(host_vertex, mapped_neighbor),
                            max(host_vertex, mapped_neighbor),
                        )
                    )
                    != color
                ):
                    compatible = False
                    break
            if not compatible:
                continue
            assignment[pattern_vertex] = host_vertex
            used_host_vertices.add(host_vertex)
            if backtrack(position + 1):
                return True
            used_host_vertices.remove(host_vertex)
            assignment[pattern_vertex] = -1
        return False

    if not backtrack(0):
        return None
    return tuple(host.graph.vertices[index] for index in assignment)
