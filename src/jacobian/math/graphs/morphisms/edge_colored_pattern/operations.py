"""Exhaustive admitted edge-coloured monomorphism search."""

import time
from collections import Counter
from itertools import permutations
from math import perm

from pydantic_core import PydanticCustomError

from jacobian._execution import (
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
    assignments = 0 if impossible or identity else perm(n, k)
    # itertools constructs k-entry injections; each candidate checks at most
    # every pattern edge. Colors are interned once so search compares integers.
    work = (
        retained + len(host.graph.edges) + assignments * (k + len(pattern.graph.edges))
    )
    if assignments > MAX_ASSIGNMENTS or work > MAX_WORK:
        _reject(
            "search",
            "complete injective assignment search exceeds its admitted work bound",
        )
    request_checkpoint("after complete colored embedding admission")
    found: tuple[str, ...] | None = None
    if identity and not impossible:
        found = pattern.graph.vertices
    elif not impossible:
        found = _search(pattern, host)
    request_checkpoint("before colored embedding result construction")
    return EdgeColoredPatternResult(
        pattern=pattern,
        host=host,
        decision="EXISTS" if found is not None else "DOES_NOT_EXIST",
        vertex_map=found if found is not None else (),
    )


def _search(
    pattern: ColoredUndirectedGraph, host: ColoredUndirectedGraph
) -> tuple[str, ...] | None:
    pattern_index = {label: i for i, label in enumerate(pattern.graph.vertices)}
    host_index = {label: i for i, label in enumerate(host.graph.vertices)}
    palette = {
        color: i
        for i, color in enumerate(
            dict.fromkeys((*pattern.edge_colors, *host.edge_colors))
        )
    }
    required = tuple(
        (pattern_index[u], pattern_index[v], palette[color])
        for (u, v), color in zip(pattern.graph.edges, pattern.edge_colors, strict=True)
    )
    available = {
        (min(host_index[u], host_index[v]), max(host_index[u], host_index[v])): palette[
            color
        ]
        for (u, v), color in zip(host.graph.edges, host.edge_colors, strict=True)
    }
    for assignment in permutations(
        range(len(host.graph.vertices)), len(pattern.graph.vertices)
    ):
        request_checkpoint("during colored embedding search")
        if all(
            available.get(
                (min(assignment[u], assignment[v]), max(assignment[u], assignment[v]))
            )
            == color
            for u, v, color in required
        ):
            return tuple(host.graph.vertices[i] for i in assignment)
    return None
