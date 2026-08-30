"""Canonical combinatorial-line hypergraph constructor for word cubes."""

from __future__ import annotations

from itertools import combinations, product

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.finite_structures.hypergraphs._models import (
    MAX_EDGES,
    MAX_TOTAL_INCIDENCES,
    MAX_VERTICES,
    FiniteHypergraph,
)
from jacobian.math.combinatorics.word_cubes._models import (
    MAX_ALPHABET_SIZE,
    MAX_DIMENSION,
    CombinatorialLine,
    CombinatorialLineHypergraphResult,
)

__all__ = ["construct_combinatorial_line_hypergraph"]


def construct_combinatorial_line_hypergraph(
    alphabet_size: int,
    dimension: int,
) -> CombinatorialLineHypergraphResult:
    """Return every standard Hales--Jewett line in ``[alphabet_size]^dimension``."""

    _admit_word_cube(alphabet_size, dimension)
    return _construct_admitted(alphabet_size, dimension)


def _construct_admitted(
    alphabet_size: int,
    dimension: int,
) -> CombinatorialLineHypergraphResult:
    q = alphabet_size
    d = dimension
    words = tuple(product(range(q), repeat=d))
    labels = tuple(_word_label(word) for word in words)
    labels_by_word = dict(zip(words, labels, strict=True))

    lines: list[CombinatorialLine] = []
    hyperedges: list[tuple[str, tuple[str, ...]]] = []
    positions = range(d)

    for wildcard_count in range(1, d + 1):
        for wildcard_positions in combinations(positions, wildcard_count):
            wildcard_set = set(wildcard_positions)
            fixed_positions = tuple(pos for pos in positions if pos not in wildcard_set)
            for fixed_values in product(range(q), repeat=len(fixed_positions)):
                fixed_coordinates = tuple(
                    zip(fixed_positions, fixed_values, strict=True)
                )
                line_words = tuple(
                    _instantiate(d, fixed_coordinates, wildcard_positions, value)
                    for value in range(q)
                )
                edge_id = f"line_{len(lines)}"
                lines.append(
                    CombinatorialLine(
                        edge_id=edge_id,
                        wildcard_positions=wildcard_positions,
                        fixed_coordinates=fixed_coordinates,
                        vertices=line_words,
                    )
                )
                hyperedges.append(
                    (edge_id, tuple(labels_by_word[word] for word in line_words))
                )

    return CombinatorialLineHypergraphResult(
        alphabet_size=q,
        dimension=d,
        words=words,
        lines=tuple(lines),
        hypergraph=FiniteHypergraph(vertices=labels, edges=tuple(hyperedges)),
    )


def _admit_word_cube(alphabet_size: int, dimension: int) -> None:
    if not 2 <= alphabet_size <= MAX_ALPHABET_SIZE:
        raise OperationDomainValidationError(
            location=("alphabet_size",),
            code="word_cube.alphabet_size",
            message=f"alphabet_size must be between 2 and {MAX_ALPHABET_SIZE}",
        )
    if not 1 <= dimension <= MAX_DIMENSION:
        raise OperationDomainValidationError(
            location=("dimension",),
            code="word_cube.dimension",
            message=f"dimension must be between 1 and {MAX_DIMENSION}",
        )
    vertices = alphabet_size**dimension
    edges = (alphabet_size + 1) ** dimension - vertices
    incidences = alphabet_size * edges
    bounds = (
        (vertices, MAX_VERTICES, "vertex_count"),
        (edges, MAX_EDGES, "edge_count"),
        (incidences, MAX_TOTAL_INCIDENCES, "incidence_count"),
    )
    for actual, limit, quantity in bounds:
        if actual > limit:
            raise OperationDomainValidationError(
                location=("alphabet_size", "dimension"),
                code=f"word_cube.{quantity}_exceeds_bound",
                message=f"the derived {quantity} {actual} exceeds the carrier limit {limit}",
            )


def _instantiate(
    dimension: int,
    fixed_coordinates: tuple[tuple[int, int], ...],
    wildcard_positions: tuple[int, ...],
    wildcard_value: int,
) -> tuple[int, ...]:
    word = [wildcard_value] * dimension
    for position, value in fixed_coordinates:
        word[position] = value
    return tuple(word)


def _word_label(word: tuple[int, ...]) -> str:
    """Return an injective, human-readable label for a coordinate word."""

    return "[" + ",".join(map(str, word)) + "]"
