"""Tests for combinatorial-matrix operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.combinatorial_matrices import HadamardMatrix, SignMatrix
from jacobian.math.combinatorial_matrices._models import (
    DeterminantProfileRequest,
    GramProfileRequest,
    NormalizeRequest,
    SignProfileRequest,
    SylvesterRequest,
)
from jacobian.math.combinatorial_matrices._operations import (
    compute_determinant_profile,
    compute_gram_profile,
    compute_normalize,
    compute_sign_profile,
    compute_sylvester,
)
from jacobian.math.combinatorial_matrices._tools import TOOLS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2() -> SignMatrix:
    return SignMatrix(rows=((1, 1), (1, -1)))


def _non_hadamard() -> SignMatrix:
    """A sign matrix that is NOT Hadamard: all +1 2x2."""
    return SignMatrix(rows=((1, 1), (1, 1)))


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


def test_catalog_contains_only_audited_agent_outcomes() -> None:
    assert {tool.operation_id for tool in TOOLS} == {
        "matrix.sign.profile.compute",
        "matrix.hadamard.gram_profile.compute",
        "matrix.hadamard.normalize.compute",
        "matrix.hadamard.determinant_profile.compute",
        "matrix.hadamard.sylvester.compute",
    }


# ---------------------------------------------------------------------------
# Sign profile
# ---------------------------------------------------------------------------


class TestSignProfile:
    def test_h2_profile(self) -> None:
        result = compute_sign_profile(SignProfileRequest(matrix=_h2()))
        assert result.row_count == 2
        assert result.column_count == 2
        assert result.plus_one_count == 3
        assert result.minus_one_count == 1
        assert result.is_square is True
        assert result.row_sums == (2, 0)


# ---------------------------------------------------------------------------
# Gram profile
# ---------------------------------------------------------------------------


class TestGramProfile:
    def test_h2_is_hadamard(self) -> None:
        result = compute_gram_profile(GramProfileRequest(matrix=_h2()))
        assert result.order == 2
        assert result.is_hadamard is True
        assert result.gram == ((2, 0), (0, 2))
        assert result.diagonal_residuals == (0, 0)
        assert result.nonzero_off_diagonal == ()

    def test_non_hadamard(self) -> None:
        result = compute_gram_profile(GramProfileRequest(matrix=_non_hadamard()))
        assert result.is_hadamard is False
        assert result.gram == ((2, 2), (2, 2))

    def test_tall_matrix_retains_every_diagonal_residual(self) -> None:
        result = compute_gram_profile(
            GramProfileRequest(matrix={"rows": [[1], [1], [-1]]})
        )
        assert result.diagonal_residuals == (0, 0, 0)


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_h2_normalize_idempotent(self) -> None:
        result = compute_normalize(NormalizeRequest(matrix=_h2()))
        # H2 already has first row/column all +1.
        assert result.normalized.rows == ((1, 1), (1, -1))
        assert result.row_switches == (0, 0)
        assert result.column_switches == (0, 0)

    def test_normalize_flips(self) -> None:
        matrix = SignMatrix(rows=((-1, -1), (-1, 1)))
        result = compute_normalize(NormalizeRequest(matrix=matrix))
        assert result.normalized.rows == ((1, 1), (1, -1))
        assert result.column_switches == (1, 1)
        assert result.row_switches == (0, 0)


# ---------------------------------------------------------------------------
# Determinant profile
# ---------------------------------------------------------------------------


class TestDeterminantProfile:
    def test_h2_determinant(self) -> None:
        h = HadamardMatrix(rows=((1, 1), (1, -1)))
        result = compute_determinant_profile(DeterminantProfileRequest(matrix=h))
        assert result.order == 2
        assert result.determinant_magnitude == 2  # 2^(2/2) = 2
        assert result.gram_determinant == 4  # 2^2


# ---------------------------------------------------------------------------
# Sylvester
# ---------------------------------------------------------------------------


class TestSylvester:
    def test_sylvester_k0(self) -> None:
        result = compute_sylvester(SylvesterRequest(k=0))
        assert result.order == 1
        assert result.matrix.rows == ((1,),)

    def test_sylvester_k1(self) -> None:
        result = compute_sylvester(SylvesterRequest(k=1))
        assert result.order == 2
        assert result.matrix.rows == ((1, 1), (1, -1))

    def test_sylvester_k2_is_hadamard(self) -> None:
        result = compute_sylvester(SylvesterRequest(k=2))
        assert result.order == 4
        h = result.matrix
        assert len(h.rows) == 4


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_sign_entry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sign matrix entries"):
            SignMatrix(rows=((1, 0), (0, 1)))

    def test_non_hadamard_rejected(self) -> None:
        with pytest.raises(ValidationError, match="orthogonality"):
            HadamardMatrix(rows=((1, 1), (1, 1)))

    def test_non_square_hadamard_rejected(self) -> None:
        with pytest.raises(ValidationError, match="square"):
            HadamardMatrix(rows=((1, 1, 1), (1, -1, 1)))

    def test_unequal_row_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError, match="equal length"):
            SignMatrix(rows=((1, 1, 1), (1, -1)))
