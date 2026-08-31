"""Shared exact sources for rational-flat owner and dispatch regressions."""

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.matroids.rational_flats import (
    ClauseConstrainedRationalFlatProblem,
    RationalFlatRankInterval,
    RationalFlatSymmetryGenerator,
    RationalVectorConfiguration,
)
from jacobian.math.matrices.values import (
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
)


def _sparse(rows: tuple[tuple[int, ...], ...], *, columns: int) -> SparseRationalMatrix:
    return SparseRationalMatrix(
        row_count=len(rows),
        column_count=columns,
        entries=tuple(
            SparseRationalMatrixEntry(
                row=row_index,
                column=column_index,
                value=CanonicalRational.from_fraction(Fraction(value)),
            )
            for row_index, row in enumerate(rows)
            for column_index, value in enumerate(row)
            if value
        ),
    )


def _primitive_line(row: tuple[int, ...]) -> tuple[int, ...]:
    first_nonzero = next((value for value in row if value), 0)
    return tuple(-value for value in row) if first_nonzero < 0 else row


def _permuted_row(
    row: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    image = [0] * len(row)
    for source, target in enumerate(permutation):
        image[target] = row[source]
    return _primitive_line(tuple(image))


def seven_coordinate_source_problem() -> ClauseConstrainedRationalFlatProblem:
    """Return the motivating complete seven-coordinate classification source."""

    ambient_dimension = 7
    candidate_rows: list[tuple[int, ...]] = []
    candidate_supports: list[frozenset[int]] = []
    for support in combinations(range(ambient_dimension), 4):
        first, second, third, fourth = support
        for positive, negative in (
            ((first, second), (third, fourth)),
            ((first, third), (second, fourth)),
            ((first, fourth), (second, third)),
        ):
            row = [0] * ambient_dimension
            for coordinate in positive:
                row[coordinate] = 1
            for coordinate in negative:
                row[coordinate] = -1
            candidate_rows.append(_primitive_line(tuple(row)))
            candidate_supports.append(frozenset(support))

    candidates = tuple(candidate_rows)
    candidate_index = {row: index for index, row in enumerate(candidates)}
    assert len(candidates) == len(candidate_index) == 105
    clauses = tuple(
        tuple(
            index
            for index, support in enumerate(candidate_supports)
            if support <= frozenset(five_set)
        )
        for five_set in combinations(range(ambient_dimension), 5)
    )
    assert len(clauses) == 21
    assert {len(clause) for clause in clauses} == {15}

    forbidden_rows = tuple(
        tuple(
            1 if coordinate == first else -1 if coordinate == second else 0
            for coordinate in range(ambient_dimension)
        )
        for first, second in combinations(range(ambient_dimension), 2)
    )
    coordinate_generators = (
        (1, 0, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5, 6, 0),
    )
    symmetry_generators = tuple(
        RationalFlatSymmetryGenerator(
            coordinate_permutation=permutation,
            candidate_permutation=tuple(
                candidate_index[_permuted_row(row, permutation)] for row in candidates
            ),
        )
        for permutation in coordinate_generators
    )
    return ClauseConstrainedRationalFlatProblem(
        candidates=RationalVectorConfiguration(
            coordinate_axis=tuple(
                f"a{index + 1}" for index in range(ambient_dimension)
            ),
            vector_labels=tuple(
                f"equation_{index}" for index in range(len(candidates))
            ),
            vectors=_sparse(candidates, columns=ambient_dimension),
        ),
        clauses=clauses,
        forbidden_vectors=_sparse(forbidden_rows, columns=ambient_dimension),
        rank_interval=RationalFlatRankInterval(minimum=4, maximum=5),
        symmetry_generators=symmetry_generators,
    )


__all__ = ["seven_coordinate_source_problem"]
