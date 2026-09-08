"""Cycle existence is local to biconnected cyclic blocks."""

import json
from itertools import pairwise

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.cycle_length_profile._models import CycleLengthProfileResult
from jacobian.math.graphs.cycle_length_profile._tools import TOOLS
from jacobian.math.graphs.cycle_length_profile.operations import verify_cycle_length_row
from jacobian.math.graphs.values import SimpleUndirectedGraph


def _run(graph: SimpleUndirectedGraph) -> CycleLengthProfileResult:
    tool = TOOLS[0]
    request = tool.request_type.model_validate_json(
        json.dumps({"graph": graph.model_dump(mode="json")})
    )
    result = tool.run(request)
    return CycleLengthProfileResult.model_validate_json(result.model_dump_json())


def test_triangle_cactus_retains_source_and_only_triangle_lengths() -> None:
    vertices = tuple(f"v{i:03}" for i in range(61))
    edges = tuple(
        sorted(
            {
                (min(vertices[a], vertices[b]), max(vertices[a], vertices[b]))
                for i in range(30)
                for a, b in (
                    (2 * i, 2 * i + 1),
                    (2 * i + 1, 2 * i + 2),
                    (2 * i + 2, 2 * i),
                )
            }
        )
    )
    for axis in (vertices, tuple(reversed(vertices))):
        graph = SimpleUndirectedGraph(vertices=axis, edges=edges)
        result = _run(graph)
        assert result.graph == graph
        assert tuple(row.cycle_length for row in result.rows) == (3,)
        assert all(verify_cycle_length_row(graph, row) for row in result.rows)


def test_large_chordless_cycle_uses_its_unique_cycle() -> None:
    vertices = tuple(f"v{i:03}" for i in range(256))
    graph = SimpleUndirectedGraph(
        vertices=vertices,
        edges=tuple(
            sorted(
                (min(a, b), max(a, b)) for a, b in pairwise((*vertices, vertices[0]))
            )
        ),
    )
    result = _run(graph)
    assert tuple(row.cycle_length for row in result.rows) == (256,)
    assert set(result.rows[0].witness) == set(vertices)
    assert verify_cycle_length_row(graph, result.rows[0])


def test_near_complete_block_still_refuses_exponential_search() -> None:
    vertices = tuple(f"v{i:02}" for i in range(16))
    edges = tuple(
        (a, b)
        for i, a in enumerate(vertices)
        for b in vertices[i + 1 :]
        if (a, b) != (vertices[0], vertices[1])
    )
    with pytest.raises(OperationDomainValidationError, match="work bound"):
        _run(SimpleUndirectedGraph(vertices=vertices, edges=edges))


def test_block_profile_agrees_with_exhaustive_small_cycle_oracle() -> None:
    from itertools import combinations, permutations

    vertices = tuple(str(i) for i in range(5))
    all_edges = tuple(combinations(vertices, 2))
    for mask in range(1 << len(all_edges)):
        edges = tuple(edge for i, edge in enumerate(all_edges) if mask & (1 << i))
        edge_set = set(edges)
        expected = set()
        for length in range(3, 6):
            for subset in combinations(vertices, length):
                if any(
                    all(
                        tuple(sorted(edge)) in edge_set
                        for edge in pairwise((subset[0], *order, subset[0]))
                    )
                    for order in permutations(subset[1:])
                ):
                    expected.add(length)
                    break
        result = _run(SimpleUndirectedGraph(vertices=vertices, edges=edges))
        assert {row.cycle_length for row in result.rows} == expected
        assert all(verify_cycle_length_row(result.graph, row) for row in result.rows)
