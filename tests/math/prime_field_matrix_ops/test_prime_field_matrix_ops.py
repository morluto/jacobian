"""Tests for prime-field matrix operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.prime_field_matrix_ops._models import (
    NullspaceRequest,
    RankRequest,
    RrefRequest,
)
from jacobian.math.prime_field_matrix_ops._operations import (
    compute_nullspace,
    compute_rank,
    compute_rref,
)
from jacobian.math.prime_field_matrix_ops._tools import TOOLS


class TestRank:
    def test_basic_rank_gf2(self) -> None:
        req = RankRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rank(req)
        assert result.rank == 2

    def test_characteristic_dependent_rank(self) -> None:
        """The same matrix can have different rank over different fields."""
        entries = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
        r2 = compute_rank(RankRequest(prime=2, entries=entries, columns=3))
        r3 = compute_rank(RankRequest(prime=3, entries=entries, columns=3))
        assert r2.rank == 2
        assert r3.rank == 3

    def test_full_rank(self) -> None:
        req = RankRequest(prime=5, entries=[[1, 0], [0, 1]], columns=2)
        result = compute_rank(req)
        assert result.rank == 2

    def test_zero_matrix_rank(self) -> None:
        req = RankRequest(prime=2, entries=[[0, 0], [0, 0]], columns=2)
        result = compute_rank(req)
        assert result.rank == 0

    def test_empty_matrix_rank(self) -> None:
        req = RankRequest(prime=2, entries=[], columns=3)
        result = compute_rank(req)
        assert result.rank == 0

    def test_nonprime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(prime=4, entries=[[1]], columns=1)

    def test_noncanonical_entries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(prime=2, entries=[[3]], columns=1)

    def test_ragged_rows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(prime=2, entries=[[1, 0], [0]], columns=2)


class TestRref:
    def test_basic_rref(self) -> None:
        req = RrefRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rref(req)
        assert result.rref_rows == ((1, 0, 1), (0, 1, 1))
        assert result.pivot_columns == (0, 1)

    def test_identity_matrix(self) -> None:
        req = RrefRequest(prime=7, entries=[[1, 0], [0, 1]], columns=2)
        result = compute_rref(req)
        assert result.rref_rows == ((1, 0), (0, 1))
        assert result.pivot_columns == (0, 1)

    def test_zero_matrix(self) -> None:
        req = RrefRequest(prime=3, entries=[[0, 0], [0, 0]], columns=2)
        result = compute_rref(req)
        assert result.pivot_columns == ()

    def test_pivot_column_order(self) -> None:
        req = RrefRequest(prime=5, entries=[[0, 0, 1], [1, 0, 0]], columns=3)
        result = compute_rref(req)
        assert result.pivot_columns == tuple(sorted(result.pivot_columns))


class TestNullspace:
    def test_basic_nullspace(self) -> None:
        req = NullspaceRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_nullspace(req)
        assert len(result.nullspace_rows) == 1
        # Verify A*v = 0 mod p
        ns = result.nullspace_rows[0]
        for row in req.entries:
            dot = sum(a * b for a, b in zip(row, ns, strict=False)) % req.prime
            assert dot == 0

    def test_full_rank_no_nullspace(self) -> None:
        req = NullspaceRequest(prime=5, entries=[[1, 0], [0, 1]], columns=2)
        result = compute_nullspace(req)
        assert result.nullspace_rows == ()
        # The genuinely empty basis is preserved so basis rows agree with
        # nullspace_rows=() instead of counting a spurious zero row.
        assert result.basis_matrix.entries == ()
        assert result.basis_matrix.columns == 2
        assert result.basis_matrix.prime == 5

    def test_nullity(self) -> None:
        """Nullity = columns - rank."""
        req = NullspaceRequest(prime=3, entries=[[1, 2, 0], [0, 0, 1]], columns=3)
        rank_req = RankRequest(prime=3, entries=[[1, 2, 0], [0, 0, 1]], columns=3)
        null_result = compute_nullspace(req)
        rank_result = compute_rank(rank_req)
        nullity = len(null_result.nullspace_rows)
        assert nullity == req.columns - rank_result.rank


class TestToolsAndExamples:
    def test_three_tools(self) -> None:
        assert len(TOOLS) == 3

    @pytest.mark.parametrize(
        "tool",
        TOOLS,
        ids=[t.operation_id for t in TOOLS],
    )
    def test_examples_run(self, tool) -> None:
        for ex in tool.examples:
            request = tool.request_type.model_validate(ex.input)
            result = tool.run(request)
            assert result is not None


class TestRankSourceBinding:
    def test_result_retains_and_replays_source_matrix(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import (
            RankRequest,
            RankResult,
        )
        from jacobian.math.prime_field_matrix_ops._operations import compute_rank

        request = RankRequest(prime=2, entries=((1, 0), (1, 1)), columns=2)
        result = compute_rank(request)
        assert (result.entries, result.columns) == (((1, 0), (1, 1)), 2)
        revalidated = RankResult.model_validate(result.model_dump())
        assert revalidated.rank == result.rank

    def test_forged_rank_is_rejected(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import RankResult

        with pytest.raises(ValidationError, match="exact rank"):
            RankResult(
                prime=2,
                entries=((1, 0), (1, 1)),
                columns=2,
                rank=7,
            )


class TestComputedMatrixContext:
    def test_computed_matrix_prime_must_match(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import (
            RankRequest,
            RrefResult,
        )
        from jacobian.math.prime_field_matrix_ops._operations import (
            compute_rref,
        )

        request = RankRequest(prime=2, entries=((1, 0), (1, 1)), columns=2)
        rref_request = __import__(
            "jacobian.math.prime_field_matrix_ops._models",
            fromlist=["RrefRequest"],
        ).RrefRequest(prime=2, entries=request.entries, columns=2)
        rref_result = compute_rref(rref_request)
        # A relayed payload pairing GF(2) rows with a GF(3) claim fails.
        with pytest.raises(ValidationError):
            RrefResult.model_validate(
                {
                    "prime": 3,
                    "entries": rref_result.entries,
                    "columns": 2,
                    "rref_rows": rref_result.rref_rows,
                    "pivot_columns": rref_result.pivot_columns,
                    "computed_matrix": {
                        "prime": 2,
                        "entries": rref_result.rref_rows,
                        "columns": 2,
                    },
                }
            )


class TestNullspaceContext:
    def test_nullspace_basis_carries_field_context(self) -> None:
        from jacobian.math.prime_field_matrix_ops._operations import (
            compute_nullspace,
        )

        request = NullspaceRequest(prime=5, entries=((1, 2),), columns=2)
        result = compute_nullspace(request)
        assert result.basis_matrix is not None
        assert result.basis_matrix.prime == 5
