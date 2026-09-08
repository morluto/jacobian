"""Independent finite-map checks for colored containment."""

import time
from itertools import combinations, permutations, product

import pytest
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.morphisms.edge_colored_pattern import (
    edge_colored_subgraph_pattern_find,
)
from jacobian.math.graphs.morphisms.edge_colored_pattern._models import (
    EdgeColoredPatternRequest,
)
from jacobian.math.graphs.values import ColoredUndirectedGraph, SimpleUndirectedGraph


def colored(
    vertices: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
    colors: tuple[str, ...],
) -> ColoredUndirectedGraph:
    return ColoredUndirectedGraph(
        graph=SimpleUndirectedGraph(vertices=vertices, edges=edges), edge_colors=colors
    )


def oracle(pattern: ColoredUndirectedGraph, host: ColoredUndirectedGraph) -> bool:
    host_edges = dict(zip(host.graph.edges, host.edge_colors, strict=True))
    return any(
        all(
            host_edges.get((min(mapping[u], mapping[v]), max(mapping[u], mapping[v])))
            == color
            for (u, v), color in zip(
                pattern.graph.edges, pattern.edge_colors, strict=True
            )
        )
        for image in permutations(host.graph.vertices, len(pattern.graph.vertices))
        for mapping in (dict(zip(pattern.graph.vertices, image, strict=True)),)
    )


def test_exhaustive_small_mixed_color_patterns() -> None:
    vertices = ("x", "y", "z")
    all_edges = tuple(combinations(vertices, 2))
    for pattern_colors in product(("red", "blue"), repeat=2):
        pattern = colored(("a", "b", "c"), (("a", "b"), ("b", "c")), pattern_colors)
        for edge_states in product((None, "red", "blue"), repeat=3):
            retained = tuple(
                (edge, color)
                for edge, color in zip(all_edges, edge_states, strict=True)
                if color is not None
            )
            if not retained:
                continue
            host = colored(
                vertices,
                tuple(edge for edge, _ in retained),
                tuple(color for _, color in retained),
            )
            result = edge_colored_subgraph_pattern_find(pattern, host)
            assert (result.decision == "EXISTS") == oracle(pattern, host)
            assert type(result).model_validate_json(result.model_dump_json()) == result
            if result.decision == "EXISTS":
                mapping = dict(
                    zip(pattern.graph.vertices, result.vertex_map, strict=True)
                )
                assert len(set(result.vertex_map)) == 3
                edges = dict(zip(host.graph.edges, host.edge_colors, strict=True))
                assert all(
                    edges[(min(mapping[u], mapping[v]), max(mapping[u], mapping[v]))]
                    == c
                    for (u, v), c in zip(
                        pattern.graph.edges, pattern.edge_colors, strict=True
                    )
                )


def test_triangle_color_change_and_edge_axis_permutation() -> None:
    pattern = colored(
        ("a", "b", "c"), (("a", "b"), ("a", "c"), ("b", "c")), ("red",) * 3
    )
    host = colored(("x", "y", "z"), (("x", "y"), ("x", "z"), ("y", "z")), ("red",) * 3)
    assert edge_colored_subgraph_pattern_find(pattern, host).decision == "EXISTS"
    changed = host.model_copy(update={"edge_colors": ("red", "blue", "red")})
    assert (
        edge_colored_subgraph_pattern_find(pattern, changed).decision
        == "DOES_NOT_EXIST"
    )
    reversed_axis = colored(
        host.graph.vertices,
        tuple(reversed(host.graph.edges)),
        tuple(reversed(host.edge_colors)),
    )
    assert (
        edge_colored_subgraph_pattern_find(pattern, reversed_axis).decision == "EXISTS"
    )


def test_color_domain_and_shared_deadline() -> None:
    valid = colored(("a", "b"), (("a", "b"),), ("red",))
    uncolored = ColoredUndirectedGraph(graph=valid.graph)
    with pytest.raises(ValidationError):
        EdgeColoredPatternRequest(pattern=uncolored, host=valid)
    with pytest.raises(OperationDomainValidationError):
        edge_colored_subgraph_pattern_find(uncolored, valid)
    with pytest.raises(ValidationError):
        colored(valid.graph.vertices, valid.graph.edges, ("red", "blue"))
    vertex_colored = valid.model_copy(update={"vertex_colors": ("red", "red")})
    with pytest.raises(OperationDomainValidationError):
        edge_colored_subgraph_pattern_find(valid, vertex_colored)
    with request_execution(time.monotonic()):
        bind_request_deadline(time.monotonic() - 1)
        with pytest.raises(OperationExecutionTimeoutError):
            edge_colored_subgraph_pattern_find(valid, valid)


def test_large_identity_presolve_and_unadmitted_search() -> None:
    vertices = tuple(f"v{i:02}" for i in range(64))
    edges = tuple((vertices[i], vertices[i + 1]) for i in range(63))
    pattern = colored(vertices, edges, ("red",) * 63)
    result = edge_colored_subgraph_pattern_find(pattern, pattern)
    assert result.vertex_map == vertices
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert edge_colored_subgraph_pattern_find(decoded.pattern, decoded.host) == result
    host_vertices = tuple(f"x{i:02}" for i in range(64))
    host = colored(
        host_vertices,
        tuple((host_vertices[i], host_vertices[i + 1]) for i in range(63)),
        ("red",) * 63,
    )
    with pytest.raises(OperationResourceAdmissionError):
        edge_colored_subgraph_pattern_find(pattern, host)
