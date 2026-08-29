from __future__ import annotations

from jacobian.math.combinatorics.word_cube_hypergraph.operations import (
    compute_word_cube_hypergraph,
)


def test_q2_d2() -> None:
    """[2]^2 has 4 vertices and lines: 00-11, 01-10, 00-01, 10-11, 00-10, 01-11, 10-01..."""
    result = compute_word_cube_hypergraph(2, 2)
    assert len(result.hypergraph.vertices) == 4  # 2^2 = 4


def test_q3_d1() -> None:
    """[3]^1 has 3 vertices and 1 line: {0,1,2}."""
    result = compute_word_cube_hypergraph(3, 1)
    assert len(result.hypergraph.vertices) == 3
    # Only one line: the full set {0,1,2}


def test_q2_d1() -> None:
    result = compute_word_cube_hypergraph(2, 1)
    assert len(result.hypergraph.vertices) == 2


def test_result_preserves_source() -> None:
    result = compute_word_cube_hypergraph(2, 2)
    assert result.alphabet_size == 2
    assert result.dimension == 2
