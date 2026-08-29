"""Tests for combinatorial-matrix operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import CanonicalLimits, encode_strict_json, loads_strict_json
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices.combinatorial import HadamardMatrix, SignMatrix
from jacobian.math.matrices.combinatorial._models import (
    DeterminantProfileRequest,
    GramProfileRequest,
    NormalizeRequest,
    SignProfileRequest,
    SylvesterRequest,
)
from jacobian.math.matrices.combinatorial._tools import (
    TOOLS,
    compute_determinant_profile,
    compute_gram_profile,
    compute_normalize,
    compute_sign_profile,
    compute_sylvester,
)
from jacobian.math.matrices.combinatorial.operations import (
    MAX_GRAM_PROFILE_AXIS,
    determinant_profile,
    gram_profile,
    kronecker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2() -> SignMatrix:
    return SignMatrix(rows=((1, 1), (1, -1)))


def _non_hadamard() -> SignMatrix:
    """A sign matrix that is NOT Hadamard: all +1 2x2."""
    return SignMatrix(rows=((1, 1), (1, 1)))


def _sylvester_rows(order: int) -> tuple[tuple[int, ...], ...]:
    rows: tuple[tuple[int, ...], ...] = ((1,),)
    while len(rows) < order:
        rows = tuple(row + row for row in rows) + tuple(
            row + tuple(-entry for entry in row) for row in rows
        )
    return rows


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
            GramProfileRequest.model_validate({"matrix": {"rows": [[1], [1], [-1]]}})
        )
        assert result.diagonal_residuals == (0, 0, 0)

    def test_flint_gram_profile_above_previous_order_boundary(self) -> None:
        order = 256
        result = gram_profile(SignMatrix(rows=_sylvester_rows(order)))

        assert result.is_hadamard is True
        assert result.gram[0] == (order,) + (0,) * (order - 1)
        assert result.gram[-1][-1] == order
        assert result.nonzero_off_diagonal == ()

    def test_worst_shape_result_stays_inside_canonical_output_boundary(self) -> None:
        order = MAX_GRAM_PROFILE_AXIS
        result = gram_profile(SignMatrix(rows=((1,) * order,) * order))
        actual = len(encode_strict_json(result.model_dump(mode="json")))

        assert actual <= CanonicalLimits().max_output_bytes

    def test_axis_above_gram_work_boundary_is_rejected_by_operation(self) -> None:
        matrix = SignMatrix(rows=((1,),) * (MAX_GRAM_PROFILE_AXIS + 1))

        with pytest.raises(OperationDomainValidationError) as exc_info:
            gram_profile(matrix)
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.gram_axis_budget"
        )


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

    def test_general_sign_normalization_round_trips_as_one_canonical_type(self) -> None:
        request = NormalizeRequest.model_validate(
            {"matrix": {"rows": [[1, 1], [1, 1]]}}
        )
        result = compute_normalize(request)
        restored = NormalizeRequest.model_validate(
            {
                "matrix": loads_strict_json(
                    encode_strict_json(result.normalized.model_dump(mode="json"))
                )
            }
        )

        assert type(request.matrix) is SignMatrix
        assert type(result.normalized) is SignMatrix
        assert restored.matrix == result.normalized


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

    def test_hadamard_validation_and_determinant_above_previous_boundary(self) -> None:
        order = 256
        matrix = HadamardMatrix(rows=_sylvester_rows(order))
        result = determinant_profile(matrix)

        assert result.order == order
        assert result.determinant_magnitude**2 == result.gram_determinant
        assert result.gram_determinant == order**order

    def test_flint_validator_rejects_corruption_above_previous_boundary(self) -> None:
        order = 256
        rows = [list(row) for row in _sylvester_rows(order)]
        rows[-1][-1] = -rows[-1][-1]

        with pytest.raises(ValidationError) as exc_info:
            DeterminantProfileRequest.model_validate({"matrix": {"rows": rows}})
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.orthogonality_violation"
        )


# ---------------------------------------------------------------------------
# Kronecker product
# ---------------------------------------------------------------------------


def test_kronecker_returns_a_canonical_hadamard_matrix() -> None:
    factor = HadamardMatrix(rows=_h2().rows)
    result = kronecker(factor, factor)

    assert isinstance(result.product, HadamardMatrix)
    assert result.row_map == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert result.column_map == result.row_map


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
        with pytest.raises(ValidationError) as exc_info:
            SignMatrix(rows=((1, 0), (0, 1)))
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.sign_entry_invalid"
        )

    def test_non_hadamard_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            HadamardMatrix(rows=((1, 1), (1, 1)))
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.orthogonality_violation"
        )

    def test_non_square_hadamard_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            HadamardMatrix(rows=((1, 1, 1), (1, -1, 1)))
        assert exc_info.value.errors()[0]["type"] == "combinatorial_matrix.not_square"

    def test_unequal_row_lengths_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SignMatrix(rows=((1, 1, 1), (1, -1)))
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.row_length_mismatch"
        )
