from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.word_cubes._models import (
    CombinatorialLineHypergraphRequest,
    CombinatorialLineHypergraphResult,
)
from jacobian.math.combinatorics.word_cubes.operations import (
    construct_combinatorial_line_hypergraph,
)


def _edges(
    result: CombinatorialLineHypergraphResult,
) -> list[tuple[str, tuple[str, ...]]]:
    return list(result.hypergraph.edges)


def test_q2_d1() -> None:
    """[2]^1 has 2 vertices and 1 combinatorial line."""
    result = construct_combinatorial_line_hypergraph(2, 1)
    assert len(result.hypergraph.vertices) == 2
    assert len(result.hypergraph.edges) == 1
    _, members = _edges(result)[0]
    assert set(members) == {"[0]", "[1]"}


def test_q3_d1() -> None:
    """[3]^1 has 3 vertices and 1 combinatorial line."""
    result = construct_combinatorial_line_hypergraph(3, 1)
    assert len(result.hypergraph.vertices) == 3
    assert len(result.hypergraph.edges) == 1
    _, members = _edges(result)[0]
    assert set(members) == {"[0]", "[1]", "[2]"}


def test_q2_d2_count() -> None:
    """[2]^2 has 4 vertices and 5 combinatorial lines ((3^2-2^2=5)."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    assert len(result.hypergraph.vertices) == 4
    assert len(result.hypergraph.edges) == 5


def test_q2_d2_fixture() -> None:
    """[2]^2 includes one-wildcard coordinate lines and the all-wildcard line."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    edge_member_sets = {frozenset(members) for _, members in result.hypergraph.edges}
    assert edge_member_sets == {
        frozenset({"[0,0]", "[1,0]"}),
        frozenset({"[0,1]", "[1,1]"}),
        frozenset({"[0,0]", "[0,1]"}),
        frozenset({"[1,0]", "[1,1]"}),
        frozenset({"[0,0]", "[1,1]"}),
    }


def test_q3_d2_count() -> None:
    """[3]^2 has 9 vertices and (3+1)^2 - 3^2 = 16-9 = 7 combinatorial lines."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    assert len(result.hypergraph.vertices) == 9
    assert len(result.hypergraph.edges) == 7


def test_q3_d2_fixture() -> None:
    """Standard Hales--Jewett lines have one common wildcard value."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    assert len(result.lines) == 7
    assert sum(line.wildcard_positions == (0, 1) for line in result.lines) == 1
    assert result.lines[-1].vertices == ((0, 0), (1, 1), (2, 2))


def test_count_identity() -> None:
    """Vertex count = q^d and edge count = (q+1)^d - q^d."""
    for q, d in [(2, 1), (2, 2), (3, 1), (3, 2), (2, 3), (4, 1)]:
        result = construct_combinatorial_line_hypergraph(q, d)
        assert len(result.hypergraph.vertices) == q**d
        expected_edges = (q + 1) ** d - q**d
        assert len(result.hypergraph.edges) == expected_edges, (
            f"q={q}, d={d}: got {len(result.hypergraph.edges)}, expected {expected_edges}"
        )


def test_vertex_labels_are_words() -> None:
    """Vertex labels are canonical word representations."""
    result = construct_combinatorial_line_hypergraph(2, 2)
    assert result.words == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert set(result.hypergraph.vertices) == {"[0,0]", "[0,1]", "[1,0]", "[1,1]"}


def test_edge_size_equals_q() -> None:
    """Each edge has exactly q vertices."""
    result = construct_combinatorial_line_hypergraph(3, 2)
    for _, members in result.hypergraph.edges:
        assert len(members) == 3


def test_q4_example() -> None:
    """q=4,d=1: 4 vertices, 1 edge."""
    result = construct_combinatorial_line_hypergraph(4, 1)
    assert len(result.hypergraph.vertices) == 4
    assert len(result.hypergraph.edges) == 1


def test_rejects_q_too_small() -> None:
    with pytest.raises(ValidationError):
        CombinatorialLineHypergraphRequest(alphabet_size=1, dimension=2)


def test_rejects_vertex_count_exceeds_bound() -> None:
    request = CombinatorialLineHypergraphRequest(alphabet_size=3, dimension=6)
    with pytest.raises(OperationDomainValidationError) as error:
        construct_combinatorial_line_hypergraph(
            request.alphabet_size, request.dimension
        )
    assert error.value.errors()[0]["type"] == "word_cube.vertex_count_exceeds_bound"


def test_native_entrypoint_uses_the_same_admission() -> None:
    with pytest.raises(OperationDomainValidationError):
        construct_combinatorial_line_hypergraph(2, 0)
    with pytest.raises(OperationDomainValidationError):
        construct_combinatorial_line_hypergraph(3, 6)


def test_exact_carrier_boundary_is_admitted() -> None:
    result = construct_combinatorial_line_hypergraph(2, 8)
    assert len(result.words) == 256
    assert len(result.lines) == 3**8 - 2**8


def test_independent_wildcard_enumeration() -> None:
    """Cross-check by independently enumerating all wildcard patterns."""
    q, d = 3, 2
    result = construct_combinatorial_line_hypergraph(q, d)
    actual_edges = {frozenset(members) for _, members in result.hypergraph.edges}

    expected_edges = set()
    for mask in range(1, 1 << d):
        wildcard_positions = [i for i in range(d) if mask & (1 << i)]
        fixed_positions = [i for i in range(d) if not (mask & (1 << i))]
        for fixed_values in product(range(q), repeat=len(fixed_positions)):
            edge = []
            for wildcard_val in range(q):
                word = [0] * d
                for i, pos in enumerate(fixed_positions):
                    word[pos] = fixed_values[i]
                for pos in wildcard_positions:
                    word[pos] = wildcard_val
                edge.append("[" + ",".join(str(x) for x in word) + "]")
            expected_edges.add(frozenset(edge))

    assert actual_edges == expected_edges


def test_pattern_provenance_reconstructs_each_line_uniquely() -> None:
    result = construct_combinatorial_line_hypergraph(3, 3)
    patterns = set()
    for line in result.lines:
        pattern = (line.wildcard_positions, line.fixed_coordinates)
        assert pattern not in patterns
        patterns.add(pattern)
        for value, word in enumerate(line.vertices):
            assert all(word[position] == value for position in line.wildcard_positions)
            assert all(
                word[position] == fixed for position, fixed in line.fixed_coordinates
            )
        edge = dict(result.hypergraph.edges)[line.edge_id]
        assert set(edge) == {
            "[" + ",".join(map(str, word)) + "]" for word in line.vertices
        }
