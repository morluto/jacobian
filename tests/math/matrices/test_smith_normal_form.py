"""Edge case tests for the SmithNormalForm validator."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.matrices.values import IntegerMatrix, SmithNormalForm


def _matrix(entries: list[list[int]]) -> IntegerMatrix:
    return IntegerMatrix(entries=tuple(tuple(str(v) for v in row) for row in entries))


def test_smith_normal_form_rejects_rank_exceeding_dimensions() -> None:
    with pytest.raises(ValidationError, match="rank cannot exceed"):
        SmithNormalForm(
            normal_form=_matrix([[1, 0], [0, 0]]),
            rank=3,
            invariant_factors=("1", "1", "1"),
        )


def test_smith_normal_form_rejects_non_positive_factors() -> None:
    with pytest.raises(ValidationError, match="must be positive"):
        SmithNormalForm(
            normal_form=_matrix([[0, 0], [0, 0]]),
            rank=1,
            invariant_factors=("0",),
        )


def test_smith_normal_form_rejects_non_divisibility_chain() -> None:
    with pytest.raises(ValidationError, match="must divide the next"):
        SmithNormalForm(
            normal_form=_matrix([[3, 0], [0, 2]]),
            rank=2,
            invariant_factors=("3", "2"),
        )


def test_smith_normal_form_rejects_mismatched_diagonal() -> None:
    with pytest.raises(ValidationError, match="leading diagonal"):
        SmithNormalForm(
            normal_form=_matrix([[4, 0], [0, 2]]),
            rank=2,
            invariant_factors=("2", "4"),
        )
