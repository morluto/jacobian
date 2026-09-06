from __future__ import annotations

import json

import pytest

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.axis_aligned_square_grid.operations import (
    construct_axis_aligned_square_grid,
    verify_axis_aligned_square_grid,
)


def test_n1() -> None:
    """N=1: 1 vertex, 0 squares."""
    result = construct_axis_aligned_square_grid(1)
    assert len(result.hypergraph.vertices) == 1
    assert len(result.hypergraph.edges) == 0


def test_n2() -> None:
    """N=2: 4 vertices, 1 square."""
    result = construct_axis_aligned_square_grid(2)
    assert len(result.hypergraph.vertices) == 4
    assert len(result.hypergraph.edges) == 1


def test_n3() -> None:
    """N=3: 9 vertices, 5 squares."""
    result = construct_axis_aligned_square_grid(3)
    assert len(result.hypergraph.vertices) == 9
    assert len(result.hypergraph.edges) == 5


def test_n9_count() -> None:
    """N=9: 81 vertices, 204 squares, 816 incidences."""
    result = construct_axis_aligned_square_grid(9)
    assert len(result.hypergraph.vertices) == 81
    assert len(result.hypergraph.edges) == 204
    total_inc = sum(len(members) for _, members in result.hypergraph.edges)
    assert total_inc == 816


def test_edge_size() -> None:
    """Every edge has exactly 4 vertices."""
    result = construct_axis_aligned_square_grid(3)
    for _, members in result.hypergraph.edges:
        assert len(members) == 4


def test_replay_squares() -> None:
    """Every edge is a valid axis-aligned square."""
    result = construct_axis_aligned_square_grid(3)
    for _, members in result.hypergraph.edges:
        coords = [tuple(int(s) for s in m.strip("()").split(",")) for m in members]
        xs = sorted({c[0] for c in coords})
        ys = sorted({c[1] for c in coords})
        assert len(xs) == 2
        assert len(ys) == 2
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        assert dx == dy
        assert dx >= 1


def test_exhaustive_comparison() -> None:
    """Compare against independent enumeration."""
    n = 4
    result = construct_axis_aligned_square_grid(n)
    expected = set()
    for y in range(n):
        for x in range(n):
            for d in range(1, n):
                if x + d <= n - 1 and y + d <= n - 1:
                    sq = frozenset(
                        [
                            f"({x},{y})",
                            f"({x + d},{y})",
                            f"({x},{y + d})",
                            f"({x + d},{y + d})",
                        ]
                    )
                    expected.add(sq)
    actual = {frozenset(members) for _, members in result.hypergraph.edges}
    assert actual == expected


def test_result_preserves_n() -> None:
    result = construct_axis_aligned_square_grid(3)
    assert result.side_length == 3


def test_n16_is_admitted_by_carrier_bounds() -> None:
    """The largest grid has 256 vertices and remains a bounded result."""
    result = construct_axis_aligned_square_grid(16)
    assert len(result.hypergraph.vertices) == 256
    assert len(result.hypergraph.edges) == 1240


def test_native_admission_rejects_n17_before_enumeration() -> None:
    with pytest.raises(OperationDomainValidationError, match="between 1 and 16"):
        construct_axis_aligned_square_grid(17)


def test_serialized_grid_claim_is_verifiable_and_forgery_is_structural() -> None:
    result = construct_axis_aligned_square_grid(2)
    decoded = type(result).model_validate_json(result.model_dump_json())
    assert verify_axis_aligned_square_grid(decoded)

    forged = result.model_dump(mode="json")
    forged["hypergraph"]["edges"][0][0] = "wrong_square"
    forged_decoded = type(result).model_validate_json(json.dumps(forged))
    assert not verify_axis_aligned_square_grid(forged_decoded)
