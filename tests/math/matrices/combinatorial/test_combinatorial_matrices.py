"""Tests for combinatorial-matrix operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    loads_strict_json,
    parse_canonical_integer,
)
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
    recognize_hadamard,
)
from jacobian.math.matrices.combinatorial.values import (
    MAX_MATERIALIZED_SIGN_MATRIX_AXIS,
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

    def test_tall_thin_gram_is_admitted_by_predicted_work(self) -> None:
        row_count = MAX_GRAM_PROFILE_AXIS + 1
        result = gram_profile(SignMatrix(rows=((1,),) * row_count))

        assert result.order == row_count
        assert result.gram == ((1,) * row_count,) * row_count
        assert result.diagonal_residuals == (0,) * row_count
        assert result.is_hadamard is False
        assert len(result.nonzero_off_diagonal) == row_count * (row_count - 1) // 2

    def test_square_gram_above_work_budget_is_rejected(self) -> None:
        order = MAX_GRAM_PROFILE_AXIS + 1
        matrix = SignMatrix(rows=((1,) * order,) * order)

        with pytest.raises(OperationDomainValidationError) as exc_info:
            gram_profile(matrix)
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.gram_work_budget"
        )

    def test_tall_gram_above_result_budget_is_rejected(self) -> None:
        matrix = SignMatrix(rows=((1,),) * MAX_MATERIALIZED_SIGN_MATRIX_AXIS)

        with pytest.raises(OperationDomainValidationError) as exc_info:
            gram_profile(matrix)
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.gram_result_budget"
        )

    def test_wide_thin_gram_is_admitted_by_predicted_work(self) -> None:
        columns = 1_024
        result = gram_profile(SignMatrix(rows=((1,) * columns,)))

        assert result.gram == ((columns,),)
        assert result.is_hadamard is False
        assert result.diagonal_residuals == (0,)

    def test_gram_profile_discovery_advertises_work_and_result_admission(self) -> None:
        tool = next(
            item
            for item in TOOLS
            if item.operation_id == "matrix.hadamard.gram_profile.compute"
        )

        assert "512" not in tool.description
        assert tool.description.endswith(
            "Row and column counts are admitted by Gram multiply-add work "
            "and exact-result size."
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
# Hadamard recognition
# ---------------------------------------------------------------------------


class TestRecognizeHadamard:
    def test_h2_returns_a_trusted_hadamard_matrix(self) -> None:
        recognized = recognize_hadamard(_h2())
        assert type(recognized) is HadamardMatrix
        assert recognized.rows == _h2().rows

    def test_non_square_sign_matrix_is_rejected(self) -> None:
        with pytest.raises(OperationDomainValidationError) as exc_info:
            recognize_hadamard(SignMatrix(rows=((1, 1, 1), (1, -1, 1))))
        assert exc_info.value.errors()[0]["type"] == "combinatorial_matrix.not_square"

    def test_square_recognition_above_work_budget_is_rejected(self) -> None:
        order = MAX_GRAM_PROFILE_AXIS + 1
        matrix = SignMatrix(rows=((1,) * order,) * order)

        with pytest.raises(OperationDomainValidationError) as exc_info:
            recognize_hadamard(matrix)
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.gram_work_budget"
        )


# ---------------------------------------------------------------------------
# Determinant profile
# ---------------------------------------------------------------------------


class TestDeterminantProfile:
    def test_h2_determinant(self) -> None:
        h = HadamardMatrix(rows=((1, 1), (1, -1)))
        result = compute_determinant_profile(DeterminantProfileRequest(matrix=h))
        assert result.order == 2
        assert result.determinant_magnitude == "2"  # 2^(2/2) = 2
        assert result.gram_determinant == "4"  # 2^2

    def test_hadamard_recognition_and_determinant_above_previous_boundary(self) -> None:
        order = 256
        matrix = HadamardMatrix(rows=_sylvester_rows(order))
        result = determinant_profile(matrix)

        assert result.order == order
        magnitude = parse_canonical_integer(result.determinant_magnitude)
        gram = parse_canonical_integer(result.gram_determinant)
        assert magnitude**2 == gram
        assert gram == order**order

    def test_request_parse_is_structural_for_corrupted_wide_hadamard(self) -> None:
        order = 256
        rows = [list(row) for row in _sylvester_rows(order)]
        rows[-1][-1] = -rows[-1][-1]

        request = DeterminantProfileRequest.model_validate({"matrix": {"rows": rows}})
        assert request.matrix.rows[-1][-1] == rows[-1][-1]

        with pytest.raises(OperationDomainValidationError) as exc_info:
            compute_determinant_profile(request)
        assert (
            exc_info.value.errors()[0]["type"]
            == "combinatorial_matrix.orthogonality_violation"
        )


# ---------------------------------------------------------------------------
# Kronecker product
# ---------------------------------------------------------------------------


def test_kronecker_returns_a_canonical_hadamard_matrix() -> None:
    factor = recognize_hadamard(_h2())
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

    def test_non_hadamard_is_structural_on_the_value_and_rejected_by_recognition(
        self,
    ) -> None:
        matrix = HadamardMatrix(rows=((1, 1), (1, 1)))
        assert matrix.rows == ((1, 1), (1, 1))

        with pytest.raises(OperationDomainValidationError) as exc_info:
            recognize_hadamard(SignMatrix(rows=matrix.rows))
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
