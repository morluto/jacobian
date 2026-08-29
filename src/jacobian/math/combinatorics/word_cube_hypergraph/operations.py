"""Word-cube combinatorial-line hypergraph kernel."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.word_cube_hypergraph._models import (
    WordCubeResult,
)

__all__ = ["compute_word_cube_hypergraph"]


def compute_word_cube_hypergraph(
    alphabet_size: int,
    dimension: int,
) -> WordCubeResult:
    """Return the q-uniform hypergraph of combinatorial lines in [q]^d.

    A combinatorial line is specified by a pattern where some positions
    are "wild" (*) and the remaining positions are fixed. Each line has
    exactly q vertices, one for each possible substitution of the wild
    positions.
    """
    q = alphabet_size
    d = dimension

    # All words of length d over [q]
    all_words = list(product(range(q), repeat=d))
    word_to_str = {w: "".join(str(x) for x in w) for w in all_words}
    all_word_strs = [word_to_str[w] for w in all_words]
    vertex_labels = tuple(all_word_strs)

    # Enumerate all combinatorial lines
    # A line is specified by a pattern: a tuple where each position is either
    # a fixed value in [q] or a wildcard.
    # For each pattern with at least one wildcard, the line has q vertices.
    lines: list[tuple[str, ...]] = []

    # Generate all patterns (each position is either fixed or wild)
    # A pattern is a tuple where None means wildcard
    for pattern in _all_patterns(q, d):
        # Get the q vertices of this line
        vertices_in_line = []
        for v in range(q):
            word = tuple(v if p is None else p for p in pattern)
            vertices_in_line.append(word_to_str[word])
        lines.append(tuple(sorted(vertices_in_line)))

    # Deduplicate lines (different patterns may produce the same line)
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    edges = tuple((f"line_{i}", line) for i, line in enumerate(unique_lines))

    hypergraph = FiniteHypergraph(
        vertices=vertex_labels,
        edges=edges,
    )

    return WordCubeResult(
        alphabet_size=q,
        dimension=d,
        hypergraph=hypergraph,
    )


def _all_patterns(q: int, d: int) -> Iterator[tuple[int | None, ...]]:
    """Generate all patterns with at least one wildcard."""
    if d == 0:
        return
    # Each position can be 0..q-1 or None (wildcard)
    options = [*range(q), None]
    for pattern in product(options, repeat=d):
        if None in pattern:  # At least one wildcard
            yield pattern
