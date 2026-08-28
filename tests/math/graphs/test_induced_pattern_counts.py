"""Defining-invariant tests for induced vertex-subset pattern counts."""

from __future__ import annotations

import json
import random
from itertools import combinations, pairwise, permutations
from math import comb

import networkx as nx
import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs import explicit_graph
from jacobian.math.graphs.patterns._models import (
    MAX_INDUCED_PATTERN_CANDIDATES,
    MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS,
    InducedVertexSubsetPatternCountRequest,
)
from jacobian.math.graphs.patterns._tools import TOOLS
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _graph(
    vertices: tuple[str, ...] | list[str],
    edges: tuple[tuple[str, str], ...] | list[tuple[str, str]],
) -> SimpleUndirectedGraph:
    return SimpleUndirectedGraph(
        vertices=tuple(vertices),
        edges=tuple(
            sorted(
                (left, right) if left < right else (right, left)
                for left, right in edges
            )
        ),
    )


def _path(order: int, prefix: str) -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{index:02d}" for index in range(order))
    return _graph(vertices, list(pairwise(vertices)))


def _cycle(order: int, prefix: str) -> SimpleUndirectedGraph:
    path = _path(order, prefix)
    if order < 3:
        raise ValueError("a simple cycle requires at least three vertices")
    return _graph(path.vertices, [*path.edges, (path.vertices[0], path.vertices[-1])])


def _complete(order: int, prefix: str) -> SimpleUndirectedGraph:
    vertices = tuple(f"{prefix}{index:02d}" for index in range(order))
    return _graph(vertices, list(combinations(vertices, 2)))


def _empty(order: int, prefix: str) -> SimpleUndirectedGraph:
    return _graph(tuple(f"{prefix}{index:02d}" for index in range(order)), [])


def _count(host: SimpleUndirectedGraph, pattern: SimpleUndirectedGraph) -> str:
    request = InducedVertexSubsetPatternCountRequest(host=host, pattern=pattern)
    return TOOLS[0].run(request).occurrence_count


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _brute_force_count(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
) -> int:
    """Replay the definition directly, without the production VF2++ backend."""

    if len(pattern.vertices) > len(host.vertices):
        return 0
    host_edges = set(host.edges)
    pattern_edges = set(pattern.edges)
    count = 0
    for subset in combinations(host.vertices, len(pattern.vertices)):
        for image in permutations(subset):
            mapping = dict(zip(pattern.vertices, image, strict=True))
            if all(
                (_edge_key(left, right) in pattern_edges)
                == (_edge_key(mapping[left], mapping[right]) in host_edges)
                for left, right in combinations(pattern.vertices, 2)
            ):
                count += 1
                break
    return count


@pytest.mark.parametrize(
    ("example_name", "expected"),
    (
        ("two_induced_p4_in_p5", "2"),
        ("c4_does_not_induce_p4", "0"),
        ("one_induced_two_edge_matching", "1"),
    ),
)
def test_public_known_answer_examples(example_name: str, expected: str) -> None:
    """The motivating P4, C4, and 2K2 cases stay public and executable."""

    operation = TOOLS[0]
    invocation = next(item for item in operation.examples if item.name == example_name)
    request = operation.request_type.model_validate(invocation.input)
    result = operation.run(request)

    assert result.occurrence_count == expected
    assert result.host == request.host
    assert result.pattern == request.pattern


def test_k4_contains_four_triangle_subsets_not_twenty_four_maps() -> None:
    assert _count(_complete(4, "h"), _complete(3, "p")) == "4"


def test_c4_contains_one_full_induced_c4_copy() -> None:
    assert _count(_cycle(4, "h"), _cycle(4, "p")) == "1"


@pytest.mark.parametrize(
    ("host", "pattern", "expected"),
    (
        (_complete(5, "h"), _complete(2, "p"), "10"),
        (_empty(5, "h"), _empty(2, "p"), "10"),
        (_complete(5, "h"), _empty(2, "p"), "0"),
        (_empty(5, "h"), _complete(2, "p"), "0"),
    ),
)
def test_complete_and_empty_hosts_distinguish_complement_patterns(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
    expected: str,
) -> None:
    assert _count(host, pattern) == expected


def test_isolated_pattern_vertex_is_semantically_significant() -> None:
    host = _path(4, "h")
    pattern = _graph(
        ("p0", "p1", "p2"),
        [("p0", "p1")],
    )

    assert _count(host, pattern) == "2"


def test_empty_pattern_has_one_occurrence_without_backend_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_isomorphism(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("empty-pattern count must not invoke NetworkX")

    monkeypatch.setattr(nx, "vf2pp_is_isomorphic", fail_isomorphism)
    assert _count(_empty(0, "h"), _empty(0, "p")) == "1"
    assert _count(_path(5, "h"), _empty(0, "p")) == "1"


def test_pattern_larger_than_host_returns_zero_without_backend_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_isomorphism(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("pattern-larger-than-host must not invoke NetworkX")

    monkeypatch.setattr(nx, "vf2pp_is_isomorphic", fail_isomorphism)
    assert _count(_path(3, "h"), _path(4, "p")) == "0"


def test_host_and_pattern_relabelling_and_row_order_preserve_count() -> None:
    original = _count(_path(5, "h"), _path(4, "p"))
    host = SimpleUndirectedGraph(
        vertices=("z", "v", "x", "u", "y"),
        edges=(("u", "v"), ("v", "x"), ("x", "y"), ("y", "z")),
    )
    pattern = SimpleUndirectedGraph(
        vertices=("delta", "alpha", "gamma", "beta"),
        edges=(("alpha", "beta"), ("beta", "gamma"), ("delta", "gamma")),
    )

    assert original == "2"
    assert _count(host, pattern) == original


def test_canonical_graph_producer_output_enters_request_unchanged() -> None:
    host = explicit_graph(
        ("e", "c", "a", "d", "b"),
        (("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")),
    )
    pattern = explicit_graph(
        ("z", "x", "w", "y"),
        (("w", "x"), ("x", "y"), ("y", "z")),
    )
    request = InducedVertexSubsetPatternCountRequest.model_validate(
        {
            "host": host.model_dump(mode="json"),
            "pattern": pattern.model_dump(mode="json"),
        }
    )

    assert request.host == host
    assert request.pattern == pattern
    assert TOOLS[0].run(request).occurrence_count == "2"


def test_small_random_graphs_match_independent_subset_and_permutation_oracle() -> None:
    rng = random.Random(2275)
    for host_order in range(7):
        host_vertices = tuple(f"h{index}" for index in range(host_order))
        for case in range(3):
            host = _graph(
                host_vertices,
                [
                    edge
                    for edge in combinations(host_vertices, 2)
                    if rng.random() < 0.45
                ],
            )
            pattern_order = rng.randrange(0, min(4, host_order + 1) + 1)
            pattern_vertices = tuple(
                f"p{case}_{index}" for index in range(pattern_order)
            )
            pattern = _graph(
                pattern_vertices,
                [
                    edge
                    for edge in combinations(pattern_vertices, 2)
                    if rng.random() < 0.45
                ],
            )

            assert int(_count(host, pattern)) == _brute_force_count(host, pattern)


def test_request_accepts_useful_case_near_subset_bound() -> None:
    # C(20, 4) = 4,845, just below the named candidate limit.
    request = InducedVertexSubsetPatternCountRequest(
        host=_empty(20, "h"),
        pattern=_complete(4, "p"),
    )
    assert TOOLS[0].run(request).occurrence_count == "0"


def test_dense_host_at_subset_bound_avoids_host_filtered_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_has_edge = nx.Graph.has_edge
    host_pair_probes = 0

    def fail_subgraph(*_args: object, **_kwargs: object) -> nx.Graph[object]:
        raise AssertionError("candidate construction must not scan a host graph view")

    def count_host_pair_probe(graph: nx.Graph[int], left: int, right: int) -> bool:
        nonlocal host_pair_probes
        if graph.number_of_nodes() == 100:
            host_pair_probes += 1
        return original_has_edge(graph, left, right)

    monkeypatch.setattr(nx.Graph, "subgraph", fail_subgraph)
    monkeypatch.setattr(nx.Graph, "has_edge", count_host_pair_probe)

    # C(100, 2) = 4,950, so this dense case exercises the useful candidate
    # boundary without allowing work to depend on the host vertices' degrees.
    assert _count(_complete(100, "h"), _complete(2, "p")) == "4950"
    assert host_pair_probes == comb(100, 2)


def test_request_rejects_next_graph_order_above_subset_bound() -> None:
    # C(21, 4) = 5,985, so rejection happens before enumeration.
    request = InducedVertexSubsetPatternCountRequest(
        host=_empty(21, "h"),
        pattern=_complete(4, "p"),
    )
    with pytest.raises(OperationDomainValidationError):
        TOOLS[0].run(request)


def test_request_bounds_per_subset_isomorphism_work() -> None:
    # A single order-eight comparison fits; the next order still has one
    # subset but exceeds the conservative VF2++ partial-injection work bound.
    assert _count(_empty(8, "h"), _empty(8, "p")) == "1"
    request = InducedVertexSubsetPatternCountRequest(
        host=_empty(9, "h"),
        pattern=_empty(9, "p"),
    )
    with pytest.raises(OperationDomainValidationError):
        TOOLS[0].run(request)


def test_canonical_graph_size_bound_rejects_257_vertices() -> None:
    with pytest.raises(ValidationError):
        InducedVertexSubsetPatternCountRequest.model_validate(
            {
                "host": {
                    "vertices": [f"h{index:03d}" for index in range(257)],
                    "edges": [],
                },
                "pattern": {"vertices": [], "edges": []},
            }
        )


def test_request_reserves_exact_output_headroom_for_retained_sources() -> None:
    limits = CanonicalLimits()
    pattern = _empty(0, "p")
    base_host = _graph(("x",), [])
    base_payload = {
        "host": base_host.model_dump(mode="json"),
        "pattern": pattern.model_dump(mode="json"),
    }
    base_size = len(encode_strict_json(base_payload))
    label_length = limits.max_input_bytes - base_size + 1
    host = _graph(("x" * label_length,), [])
    request_payload = {
        "host": host.model_dump(mode="json"),
        "pattern": pattern.model_dump(mode="json"),
    }
    assert len(encode_strict_json(request_payload)) == limits.max_input_bytes

    request = InducedVertexSubsetPatternCountRequest(host=host, pattern=pattern)
    with pytest.raises(OperationDomainValidationError):
        TOOLS[0].run(request)


def test_numeric_admission_caps_are_visible_in_schema_and_tool_description() -> None:
    schema = json.dumps(
        InducedVertexSubsetPatternCountRequest.model_json_schema(),
        sort_keys=True,
    )
    assert f"{MAX_INDUCED_PATTERN_CANDIDATES:,} candidate subsets" in schema
    assert f"{MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} total work units" in schema
    assert (
        f"{CanonicalLimits().max_output_bytes:,}-byte canonical output bound" in schema
    )
    assert "C(|V(host)|, |V(pattern)|)" in schema
    assert "C(|V(pattern)|, 2) direct host-edge probes" in schema
    assert "partial-injection state bound" in schema
    assert f"{MAX_INDUCED_PATTERN_CANDIDATES:,} subsets" in TOOLS[0].description
    assert (
        f"{MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS:,} work units" in TOOLS[0].description
    )
    assert "direct host-edge probes" in TOOLS[0].description
    assert "one local graph per subset" in TOOLS[0].examples[0].description
    assert "per-subset VF2++ work" in TOOLS[0].examples[0].description
