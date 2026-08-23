"""Tests for prime-field matrix operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix
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


def _matrix(prime, entries, columns) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)


class TestRank:
    def test_basic_rank_gf2(self) -> None:
        req = RankRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        result = compute_rank(req)
        assert result.rank == 2
        assert result.matrix is req.matrix

    def test_characteristic_dependent_rank(self) -> None:
        """The same matrix can have different rank over different fields."""
        entries = ((1, 1, 0), (0, 1, 1), (1, 0, 1))
        r2 = compute_rank(RankRequest(matrix=_matrix(2, entries, 3)))
        r3 = compute_rank(RankRequest(matrix=_matrix(3, entries, 3)))
        assert r2.rank == 2
        assert r3.rank == 3

    def test_full_rank(self) -> None:
        req = RankRequest(matrix=_matrix(5, ((1, 0), (0, 1)), 2))
        result = compute_rank(req)
        assert result.rank == 2

    def test_zero_matrix_rank(self) -> None:
        req = RankRequest(matrix=_matrix(2, ((0, 0), (0, 0)), 2))
        result = compute_rank(req)
        assert result.rank == 0

    def test_empty_matrix_rank(self) -> None:
        req = RankRequest(matrix=_matrix(2, (), 3))
        result = compute_rank(req)
        assert result.rank == 0

    def test_nonprime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(matrix=_matrix(4, ((1,),), 1))

    def test_oversized_prime_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(matrix=_matrix(1_000_003, ((1,),), 1))


class TestRref:
    def test_basic_rref(self) -> None:
        req = RrefRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        result = compute_rref(req)
        assert result.rref.entries == ((1, 0, 1), (0, 1, 1))
        assert result.rref.prime == 2
        assert result.rref.columns == 3
        assert result.pivot_columns == (0, 1)

    def test_identity_matrix(self) -> None:
        req = RrefRequest(matrix=_matrix(7, ((1, 0), (0, 1)), 2))
        result = compute_rref(req)
        assert result.rref.entries == ((1, 0), (0, 1))
        assert result.pivot_columns == (0, 1)

    def test_zero_matrix(self) -> None:
        req = RrefRequest(matrix=_matrix(3, ((0, 0), (0, 0)), 2))
        result = compute_rref(req)
        assert result.pivot_columns == ()
        assert result.rref.entries == ((0, 0), (0, 0))

    def test_empty_shape_retains_column_axis(self) -> None:
        """The canonical matrix value carries the empty-matrix column axis."""
        req = RrefRequest(matrix=_matrix(2, (), 3))
        result = compute_rref(req)
        assert result.rref.entries == ()
        assert result.rref.columns == 3

    def test_pivot_column_order(self) -> None:
        req = RrefRequest(matrix=_matrix(5, ((0, 0, 1), (1, 0, 0)), 3))
        result = compute_rref(req)
        assert result.pivot_columns == tuple(sorted(result.pivot_columns))


class TestNullspace:
    def test_basic_nullspace(self) -> None:
        req = NullspaceRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        result = compute_nullspace(req)
        assert len(result.nullspace_rows) == 1
        # Verify A*v = 0 mod p
        ns = result.nullspace_rows[0]
        for row in req.matrix.entries:
            dot = sum(a * b for a, b in zip(row, ns, strict=False)) % req.matrix.prime
            assert dot == 0

    def test_full_rank_no_nullspace(self) -> None:
        req = NullspaceRequest(matrix=_matrix(5, ((1, 0), (0, 1)), 2))
        result = compute_nullspace(req)
        assert result.nullspace_rows == ()

    def test_nullity(self) -> None:
        """Nullity = columns - rank."""
        entries = ((1, 2, 0), (0, 0, 1))
        null_result = compute_nullspace(NullspaceRequest(matrix=_matrix(3, entries, 3)))
        rank_result = compute_rank(RankRequest(matrix=_matrix(3, entries, 3)))
        nullity = len(null_result.nullspace_rows)
        assert nullity == 3 - rank_result.rank


class TestComposition:
    """Results feed downstream consumers as the same canonical value."""

    def test_rref_result_feeds_rank_and_nullspace_unchanged(self) -> None:
        source = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        rref_result = compute_rref(RrefRequest(matrix=source))

        rank_request = RankRequest(matrix=rref_result.rref)
        assert rank_request.matrix is rref_result.rref
        rank_of_rref = compute_rank(rank_request)
        assert rank_of_rref.rank == len(rref_result.pivot_columns)

        null_of_rref = compute_nullspace(NullspaceRequest(matrix=rref_result.rref))
        assert len(null_of_rref.nullspace_rows) == 3 - rank_of_rref.rank

    def test_serialized_result_round_trips_into_consumers(self) -> None:
        rref_result = compute_rref(
            RrefRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        )
        payload = rref_result.model_dump()
        revived = type(rref_result).model_validate(payload)
        assert revived == rref_result
        assert (
            compute_rank(RankRequest.model_validate({"matrix": payload["rref"]})).rank
            == 2
        )

    def test_forged_rref_value_rejected(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import RrefResult

        source = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        rref_result = compute_rref(RrefRequest(matrix=source))
        forged_rows = (
            tuple((value + 1) % 2 for value in rref_result.rref.entries[0]),
            *rref_result.rref.entries[1:],
        )
        with pytest.raises(ValidationError, match="reduced row-echelon form"):
            RrefResult(
                matrix=source,
                rref=_matrix(2, forged_rows, 3),
                pivot_columns=rref_result.pivot_columns,
            )


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
