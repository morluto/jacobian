"""Canonical combinatorial-line hypergraph constructor for word cubes [q]^d."""

from __future__ import annotations

from itertools import product

from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    FiniteHypergraph,
)
from jacobian.math.combinatorics.word_cubes._models import (
    CombinatorialLineHypergraphResult,
)

__all__ = ["construct_combinatorial_line_hypergraph"]


def construct_combinatorial_line_hypergraph(
    alphabet_size: int,
    dimension: int,
) -> CombinatorialLineHypergraphResult:
    """Construct the combinatorial-line hypergraph of [q]^d.

    Vertices are all length-d words over the alphabet {0,...,q-1}.
    Edges correspond to combinatorial lines: patterns with at least
    one wildcard coordinate. Each edge has exactly q vertices.
    """
    q = alphabet_size
    d = dimension

    all_words = list(product(range(q), repeat=d))
    word_labels = [_word_label(w) for w in all_words]
    word_set = dict(zip(all_words, word_labels, strict=True))

    edges: list[tuple[str, tuple[str, ...]]] = []
    edge_index = 0

    for mask in range(1, 1 << d):
        wildcard_positions = [i for i in range(d) if mask & (1 << i)]
        fixed_positions = [i for i in range(d) if not (mask & (1 << i))]

        for fixed_values in product(range(q), repeat=len(fixed_positions)):
            edge_vertices = []
            for wildcard_val in range(q):
                word = [0] * d
                for i, pos in enumerate(fixed_positions):
                    word[pos] = fixed_values[i]
                for pos in wildcard_positions:
                    word[pos] = wildcard_val
                edge_vertices.append(word_set[tuple(word)])

            edge_id = f"line_{edge_index}"
            edges.append((edge_id, tuple(edge_vertices)))
            edge_index += 1

    hypergraph = FiniteHypergraph(
        vertices=tuple(word_labels),
        edges=tuple(edges),
    )
    return CombinatorialLineHypergraphResult(
        alphabet_size=q,
        dimension=d,
        hypergraph=hypergraph,
    )


def _word_label(word: tuple[int, ...]) -> str:
    """Canonical label for a word."""
    return "".join(str(d) for d in word)
