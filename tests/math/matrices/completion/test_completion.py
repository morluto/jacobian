"""Independent exact evidence for chordal PSD completion."""

from collections.abc import Sequence
from fractions import Fraction
from itertools import combinations, permutations

import pytest
from pydantic import ValidationError
from sympy import Matrix, Rational

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.completion import (
    ChordalPSDCompletionResult,
    PartialSymmetricRationalMatrix,
    complete_chordal_psd,
)


def partial(
    rows: Sequence[Sequence[int | Fraction | None]],
) -> PartialSymmetricRationalMatrix:
    return PartialSymmetricRationalMatrix.model_validate(
        {
            "graph": {
                "vertex_count": len(rows),
                "edges": [
                    [i, j]
                    for i in range(len(rows))
                    for j in range(i + 1, len(rows))
                    if rows[i][j] is not None
                ],
            },
            "specified_entries": [
                {
                    "row": i,
                    "column": j,
                    "value": CanonicalRational.from_fraction(Fraction(str(rows[i][j]))),
                }
                for i in range(len(rows))
                for j in range(i, len(rows))
                if rows[i][j] is not None
            ],
        }
    )


def exact(result: ChordalPSDCompletionResult) -> Matrix:
    assert result.outcome == "COMPLETED"
    assert result.completion is not None
    answer = Matrix(
        [[Rational(q.num, q.den) for q in row] for row in result.completion.entries]
    )
    for e in result.matrix.specified_entries:
        assert answer[e.row, e.column] == Rational(e.value.num, e.value.den)
    assert answer == answer.T
    # All principal minors, not merely leading minors, characterize PSD.
    for size in range(1, answer.rows + 1):
        for axes in combinations(range(answer.rows), size):
            assert answer.extract(axes, axes).det() >= 0
    return answer


def test_missing_path_entry_is_not_zero() -> None:
    q = Fraction(4, 5)
    source = partial([[1, q, None], [q, 1, q], [None, q, 1]])
    completed = exact(complete_chordal_psd(source))
    assert completed[0, 2] == Rational(16, 25)
    assert completed.det() == Rational(81, 625)
    source_zero = partial([[1, q, 0], [q, 1, q], [0, q, 1]])
    rejected = complete_chordal_psd(source_zero)
    assert rejected.outcome == "INFEASIBLE"
    assert rejected.obstruction_clique == (0, 1, 2)
    for item in (source, source_zero):
        assert (
            PartialSymmetricRationalMatrix.model_validate_json(item.model_dump_json())
            == item
        )
    assert source.model_dump_json() != source_zero.model_dump_json()


def test_singular_separator_all_axis_permutations() -> None:
    rows: list[list[int | None]] = [
        [2, 1, 1, None],
        [1, 1, 1, 2],
        [1, 1, 1, 2],
        [None, 2, 2, 5],
    ]
    for perm in permutations(range(4)):
        source = partial([[rows[i][j] for j in perm] for i in perm])
        result = complete_chordal_psd(source)
        answer = exact(result)
        assert answer[perm.index(0), perm.index(3)] == 2
        assert answer.rank() == 3
        assert (
            ChordalPSDCompletionResult.model_validate_json(result.model_dump_json())
            == result
        )


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [[0]],
        [[-1]],
        [[0, 1], [1, 0]],
        [[1, 0, None], [0, 0, 0], [None, 0, 2]],
        [[1, None, None], [None, 0, None], [None, None, 2]],
        [[1, 1, None], [1, 1, None], [None, None, 0]],
        [[1, 2, None], [2, 1, 0], [None, 0, 1]],
    ],
)
def test_degenerate_and_obstructed_patterns(rows: list[list[int | None]]) -> None:
    result = complete_chordal_psd(partial(rows))
    if result.outcome == "COMPLETED":
        exact(result)
    else:
        axes = result.obstruction_clique
        obstruction = Matrix([[rows[i][j] for j in axes] for i in axes])
        assert obstruction.is_positive_semidefinite is False


def test_authored_obstruction_must_be_a_specified_clique() -> None:
    source = partial([[1, 1, None], [1, 1, 1], [None, 1, 1]])
    with pytest.raises(ValidationError, match="specified graph edges"):
        ChordalPSDCompletionResult(
            matrix=source,
            outcome="INFEASIBLE",
            obstruction_clique=(0, 2),
        )
    source = partial(
        [[1, 0, None, 0], [0, 1, 0, None], [None, 0, 1, 0], [0, None, 0, 1]]
    )
    with pytest.raises(OperationDomainValidationError, match="must be chordal"):
        complete_chordal_psd(source)


@pytest.mark.parametrize(
    "mutation", ["missing_diagonal", "extra_entry", "duplicate", "ordering"]
)
def test_structural_pattern_rejects_malformed_source(mutation: str) -> None:
    payload = partial([[1, None], [None, 1]]).model_dump(mode="json")
    if mutation == "missing_diagonal":
        payload["specified_entries"].pop()
    elif mutation == "extra_entry":
        payload["specified_entries"].insert(
            1, {"row": 0, "column": 1, "value": {"num": "0", "den": "1"}}
        )
    elif mutation == "duplicate":
        payload["specified_entries"].append(payload["specified_entries"][0])
    else:
        payload["elimination_ordering"] = [0, 1]
    with pytest.raises(ValidationError):
        PartialSymmetricRationalMatrix.model_validate(payload)


def test_useful_sparse_path_scale() -> None:
    n = 128
    q = Fraction(4, 5)
    source = partial(
        [
            [1 if i == j else q if abs(i - j) == 1 else None for j in range(n)]
            for i in range(n)
        ]
    )
    result = complete_chordal_psd(source)
    assert result.completion is not None
    # This Toeplitz covariance has LDL pivots 1, 1-q², ..., 1-q².
    for i, row in enumerate(result.completion.entries):
        assert all(
            value.as_fraction() == q ** abs(i - j) for j, value in enumerate(row)
        )


def test_excessive_dense_output_rejected_before_expansion() -> None:
    n = 257
    source = partial([[1 if i == j else None for j in range(n)] for i in range(n)])
    with pytest.raises(OperationDomainValidationError, match="output entries"):
        complete_chordal_psd(source)


def test_overlapping_gram_cliques_with_negative_correlations() -> None:
    vectors = Matrix([[1, 2], [2, -1], [0, 1], [-1, 3], [2, 2]])
    gram = vectors * vectors.T
    source = partial(
        [
            [int(gram[i, j]) if abs(i - j) <= 2 else None for j in range(5)]
            for i in range(5)
        ]
    )
    exact(complete_chordal_psd(source))


def test_large_singleton_has_no_separator_growth() -> None:
    source = partial([[10**3000]])
    result = complete_chordal_psd(source)
    assert result.completion is not None
    assert result.completion.entries[0][0].num == 10**3000


def test_independent_diagonal_denominators_do_not_accumulate() -> None:
    n = 128
    diagonal = [Fraction(1, 2**128 - 1 + 2 * i) for i in range(n)]
    source = partial(
        [[diagonal[i] if i == j else None for j in range(n)] for i in range(n)]
    )
    result = complete_chordal_psd(source)
    assert result.completion is not None
    assert all(
        result.completion.entries[i][i].as_fraction() == diagonal[i] for i in range(n)
    )


def test_coupled_rational_growth_rejection() -> None:
    large = 10**3000
    with pytest.raises(OperationDomainValidationError, match="bit budget"):
        complete_chordal_psd(
            partial([[large, 1, None], [1, large, 1], [None, 1, large]])
        )
