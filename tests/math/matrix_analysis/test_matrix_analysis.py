"""Tests for matrix analysis operations."""

from jacobian.math.matrix_analysis._models import (
    FarkasCertificateRequest,
    MatrixEntry,
    SymmetricMatrixRequest,
)
from jacobian.math.matrix_analysis._operations import (
    check_farkas_certificate,
    compute_inertia,
)


class TestInertia:
    def test_identity(self):
        req = SymmetricMatrixRequest(
            dimension=3,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "1", "den": "1"}),
                MatrixEntry(row=2, col=2, value={"num": "1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 3
        assert result.n_negative == 0
        assert result.n_zero == 0
        assert result.definiteness == "positive_definite"

    def test_negative_identity(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "-1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "-1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 0
        assert result.n_negative == 2
        assert result.definiteness == "negative_definite"

    def test_indefinite(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(
                MatrixEntry(row=0, col=0, value={"num": "1", "den": "1"}),
                MatrixEntry(row=1, col=1, value={"num": "-1", "den": "1"}),
            ),
        )
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.definiteness == "indefinite"

    def test_off_diagonal_hyperbolic_pair(self):
        req = SymmetricMatrixRequest(
            dimension=2,
            entries=(MatrixEntry(row=0, col=1, value={"num": "1", "den": "1"}),),
        )
        result = compute_inertia(req)
        assert result.n_positive == 1
        assert result.n_negative == 1
        assert result.n_zero == 0
        assert result.definiteness == "indefinite"

    def test_rejects_conflicting_symmetric_entries(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="conflict"):
            SymmetricMatrixRequest(
                dimension=2,
                entries=(
                    MatrixEntry(row=0, col=1, value={"num": "1", "den": "1"}),
                    MatrixEntry(row=1, col=0, value={"num": "2", "den": "1"}),
                ),
            )


class TestFarkas:
    def test_valid_certificate(self):
        # System: x1 + x2 <= -1, x1 + x2 >= 1 is infeasible.
        # A = [[1, 1], [-1, -1]], b = [-1, -1]
        # y = (1, 1), y^T A = (1-1, 1-1) = (0, 0), y^T b = -1 + -1 = -2 < 0 => valid
        req = FarkasCertificateRequest(
            constraint_matrix=[
                ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                ({"num": "-1", "den": "1"}, {"num": "-1", "den": "1"}),
            ],
            rhs_vector=(
                {"num": "-1", "den": "1"},
                {"num": "-1", "den": "1"},
            ),
            multipliers=(
                {"num": "1", "den": "1"},
                {"num": "1", "den": "1"},
            ),
        )
        result = check_farkas_certificate(req)
        assert result.valid is True

    def test_rejects_nonrectangular_matrix(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="rectangular"):
            FarkasCertificateRequest(
                constraint_matrix=[
                    ({"num": "1", "den": "1"}, {"num": "1", "den": "1"}),
                    ({"num": "-1", "den": "1"},),
                ],
                rhs_vector=(
                    {"num": "-1", "den": "1"},
                    {"num": "-1", "den": "1"},
                ),
                multipliers=(
                    {"num": "1", "den": "1"},
                    {"num": "1", "den": "1"},
                ),
            )
