"""Tests for prime-field matrix operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.prime_field_matrix_ops._models import (
    NullspaceRequest,
    RankRequest,
    RankResult,
    RrefRequest,
    RrefResult,
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

    def test_result_retains_source_matrix(self) -> None:
        req = RankRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rank(req)
        assert result.entries == ((1, 1, 0), (0, 1, 1))
        assert result.columns == 3
        assert result.prime == 2
        RankResult.model_validate(result.model_dump())

    def test_result_rejects_rank_unsupported_by_matrix(self) -> None:
        with pytest.raises(ValidationError, match="exact rank"):
            RankResult(
                prime=2,
                entries=((1, 0), (0, 0)),
                columns=2,
                rank=2,
            )

    def test_ragged_rows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(prime=2, entries=[[1, 0], [0]], columns=2)


class TestRref:
    def test_basic_rref(self) -> None:
        req = RrefRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rref(req)
        assert result.reduced_matrix.entries == ((1, 0, 1), (0, 1, 1))
        assert result.pivot_columns == (0, 1)

    def test_identity_matrix(self) -> None:
        req = RrefRequest(prime=7, entries=[[1, 0], [0, 1]], columns=2)
        result = compute_rref(req)
        assert result.reduced_matrix.entries == ((1, 0), (0, 1))
        assert result.pivot_columns == (0, 1)

    def test_zero_matrix(self) -> None:
        req = RrefRequest(prime=3, entries=[[0, 0], [0, 0]], columns=2)
        result = compute_rref(req)
        assert result.pivot_columns == ()

    def test_pivot_column_order(self) -> None:
        req = RrefRequest(prime=5, entries=[[0, 0, 1], [1, 0, 0]], columns=3)
        result = compute_rref(req)
        assert result.pivot_columns == tuple(sorted(result.pivot_columns))

    def test_reduced_matrix_composes_with_downstream_requests(self) -> None:
        """The serialized matrix value feeds rank/nullspace unchanged."""
        import json

        req = RrefRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rref(req)
        payload = json.loads(result.model_dump_json())["reduced_matrix"]
        assert set(payload) == {"prime", "entries", "columns"}
        assert NullspaceRequest.model_validate(payload).columns == 3
        assert compute_rank(RankRequest.model_validate(payload)).rank == 2

    def test_zero_row_reduction_keeps_column_axis(self) -> None:
        result = compute_rref(RrefRequest(prime=5, entries=[], columns=4))
        assert result.reduced_matrix.entries == ()
        assert result.reduced_matrix.columns == 4
        RrefResult.model_validate(result.model_dump())

    def test_result_round_trips(self) -> None:
        req = RrefRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_rref(req)
        RrefResult.model_validate(result.model_dump())

    def test_result_rejects_wrong_reduced_matrix(self) -> None:
        with pytest.raises(ValidationError, match="exact reduced row-echelon"):
            RrefResult(
                prime=2,
                entries=((1, 1, 0), (0, 1, 1)),
                columns=3,
                reduced_matrix={
                    "prime": 2,
                    "entries": ((1, 0, 0), (0, 1, 0)),
                    "columns": 3,
                },
                pivot_columns=(0, 1),
            )

    def test_result_rejects_foreign_prime_on_reduced_matrix(self) -> None:
        with pytest.raises(ValidationError):
            RrefResult(
                prime=2,
                entries=((1, 1),),
                columns=2,
                reduced_matrix={"prime": 3, "entries": ((1, 1),), "columns": 2},
                pivot_columns=(0,),
            )


class TestNullspace:
    def test_basic_nullspace(self) -> None:
        req = NullspaceRequest(prime=2, entries=[[1, 1, 0], [0, 1, 1]], columns=3)
        result = compute_nullspace(req)
        assert len(result.nullspace_matrix.entries) == 1
        # Verify A*v = 0 mod p
        ns = result.nullspace_matrix.entries[0]
        for row in req.entries:
            dot = sum(a * b for a, b in zip(row, ns, strict=False)) % req.prime
            assert dot == 0

    def test_full_rank_no_nullspace(self) -> None:
        req = NullspaceRequest(prime=5, entries=[[1, 0], [0, 1]], columns=2)
        result = compute_nullspace(req)
        assert result.nullspace_matrix.entries == ()

    def test_nullity(self) -> None:
        """Nullity = columns - rank."""
        req = NullspaceRequest(prime=3, entries=[[1, 2, 0], [0, 0, 1]], columns=3)
        rank_req = RankRequest(prime=3, entries=[[1, 2, 0], [0, 0, 1]], columns=3)
        null_result = compute_nullspace(req)
        rank_result = compute_rank(rank_req)
        nullity = len(null_result.nullspace_matrix.entries)
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


class TestSafePrimeBound:
    def test_prime_beyond_safe_integer_range_rejected(self) -> None:
        """2^61-1 cannot survive number-based JSON clients unrounded."""
        with pytest.raises(ValidationError):
            RankRequest(prime=2**61 - 1, entries=[[1]], columns=1)
        with pytest.raises(ValidationError):
            RrefRequest(prime=2**61 - 1, entries=[[1]], columns=1)
        with pytest.raises(ValidationError):
            NullspaceRequest(prime=2**61 - 1, entries=[[1]], columns=1)

    def test_largest_admissible_prime_accepted(self) -> None:
        largest_prime_below_2_53 = 9007199254740881
        req = RankRequest(prime=largest_prime_below_2_53, entries=[[1], [1]], columns=1)
        assert compute_rank(req).rank == 1


class TestCanonicalNullspaceValue:
    def test_nullspace_result_feeds_rank_unchanged(self):
        """The serialized basis crosses rank/rref consumer boundaries as-is."""
        from jacobian.math.prime_field_matrix_ops._models import RankRequest
        from jacobian.math.prime_field_matrix_ops._operations import compute_rank

        request = NullspaceRequest(
            prime=5,
            entries=((1, 2), (2, 4)),
            columns=2,
        )
        result = compute_nullspace(request)
        basis = result.nullspace_matrix
        assert (
            compute_rank(
                RankRequest(
                    prime=basis.prime,
                    entries=basis.entries,
                    columns=basis.columns,
                )
            ).rank
            == 1
        )

    def test_empty_basis_keeps_the_column_axis(self):

        request = NullspaceRequest(prime=2, entries=((1,), (1,)), columns=1)
        result = compute_nullspace(request)
        assert result.nullspace_matrix.entries == ()
        assert result.nullspace_matrix.columns == 1
        assert result.nullspace_matrix.prime == 2
