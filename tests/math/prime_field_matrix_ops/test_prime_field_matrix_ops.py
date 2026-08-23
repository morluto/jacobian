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


def _matrix(
    prime: int, entries: tuple[tuple[int, ...], ...], columns: int
) -> PrimeFieldMatrix:
    return PrimeFieldMatrix(prime=prime, entries=entries, columns=columns)


class TestRank:
    def test_basic_rank_gf2(self) -> None:
        req = RankRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        result = compute_rank(req)
        assert result.rank == 2

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

    def test_noncanonical_entries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(matrix=_matrix(2, ((3,),), 1))

    def test_ragged_rows_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankRequest(matrix=_matrix(2, ((1, 0), (0,)), 2))


class TestRref:
    def test_basic_rref(self) -> None:
        req = RrefRequest(matrix=_matrix(2, ((1, 1, 0), (0, 1, 1)), 3))
        result = compute_rref(req)
        assert isinstance(result.rref, PrimeFieldMatrix)
        assert result.rref.entries == ((1, 0, 1), (0, 1, 1))
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

    def test_pivot_column_order(self) -> None:
        req = RrefRequest(matrix=_matrix(5, ((0, 0, 1), (1, 0, 0)), 3))
        result = compute_rref(req)
        assert result.pivot_columns == tuple(sorted(result.pivot_columns))

    def test_rref_value_enters_consumers_unchanged(self) -> None:
        """The canonical RREF value composes directly into rank/nullspace."""
        source = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        result = compute_rref(RrefRequest(matrix=source))
        rank_of_rref = compute_rank(RankRequest(matrix=result.rref))
        nullspace_of_rref = compute_nullspace(NullspaceRequest(matrix=result.rref))
        assert rank_of_rref.rank == len(result.pivot_columns) == 2
        assert rank_of_rref.matrix == result.rref
        assert nullspace_of_rref.nullspace_rows == ((1, 1, 1),)
        # The retained source matrix stays distinct from its RREF.
        assert result.matrix == source
        assert result.rref != source

    def test_wire_round_trip_through_the_canonical_value(self) -> None:
        """A serialized RREF value validates as a consumer request payload."""
        from pydantic import TypeAdapter

        source = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        result = compute_rref(RrefRequest(matrix=source))
        payload = {"matrix": TypeAdapter(PrimeFieldMatrix).dump_python(result.rref)}
        rank_request = RankRequest.model_validate(payload)
        assert compute_rank(rank_request).rank == 2
        nullspace_request = NullspaceRequest.model_validate(payload)
        assert compute_nullspace(nullspace_request).nullspace_rows == ((1, 1, 1),)


class TestNullspace:
    def test_basic_nullspace(self) -> None:
        matrix = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        result = compute_nullspace(NullspaceRequest(matrix=matrix))
        assert len(result.nullspace_rows) == 1
        # Verify A*v = 0 mod p
        ns = result.nullspace_rows[0]
        for row in matrix.entries:
            dot = sum(a * b for a, b in zip(row, ns, strict=False)) % matrix.prime
            assert dot == 0

    def test_full_rank_no_nullspace(self) -> None:
        req = NullspaceRequest(matrix=_matrix(5, ((1, 0), (0, 1)), 2))
        result = compute_nullspace(req)
        assert result.nullspace_rows == ()

    def test_nullity(self) -> None:
        """Nullity = columns - rank."""
        matrix = _matrix(3, ((1, 2, 0), (0, 0, 1)), 3)
        null_result = compute_nullspace(NullspaceRequest(matrix=matrix))
        rank_result = compute_rank(RankRequest(matrix=matrix))
        nullity = len(null_result.nullspace_rows)
        assert nullity == matrix.columns - rank_result.rank


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
            RankResult,
        )
        from jacobian.math.prime_field_matrix_ops._operations import compute_rank

        request = RankRequest(matrix=_matrix(2, ((1, 0), (1, 1)), 2))
        result = compute_rank(request)
        assert result.matrix == request.matrix
        revalidated = RankResult.model_validate(result.model_dump())
        assert revalidated.rank == result.rank
        assert revalidated.matrix == request.matrix

    def test_forged_rank_is_rejected(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import RankResult

        with pytest.raises(ValidationError, match="exact rank"):
            RankResult(
                matrix=_matrix(2, ((1, 0), (1, 1)), 2),
                rank=7,
            )

    def test_forged_rref_is_rejected(self) -> None:
        from jacobian.math.prime_field_matrix_ops._models import RrefResult

        source = _matrix(2, ((1, 1, 0), (0, 1, 1)), 3)
        forged = _matrix(2, ((1, 0, 0), (0, 1, 0)), 3)
        with pytest.raises(ValidationError, match="reduced row-echelon"):
            RrefResult(matrix=source, rref=forged, pivot_columns=(0, 1))
