"""Tests for prime-field matrix operations."""

import pytest
from pydantic import ValidationError

from jacobian.math.prime_field_matrix._models import (
    PrimeFieldMatrixRankResult,
    PrimeFieldMatrixRequest,
    PrimeFieldNullspaceResult,
    PrimeFieldRrefResult,
)
from jacobian.math.prime_field_matrix._operations import (
    compute_nullspace,
    compute_rank,
    compute_rref,
)


class TestRank:
    def test_full_rank_gf2(self):
        """Identity 3x3 over GF(2) has rank 3."""
        req = PrimeFieldMatrixRequest(
            prime=2, entries=((1, 0, 0), (0, 1, 0), (0, 0, 1))
        )
        result = compute_rank(req)
        assert result.rank == 3
        assert result.prime == 2
        assert len(result.source.entries) == 3
        assert len(result.source.entries[0]) == 3

    def test_zero_matrix_gf2(self):
        """Zero matrix over GF(2) has rank 0."""
        req = PrimeFieldMatrixRequest(prime=2, entries=((0, 0), (0, 0)))
        result = compute_rank(req)
        assert result.rank == 0

    def test_characteristic_dependent_rank(self):
        """The same integer matrix has different rank over different primes.

        This is the core motivation from issue #2183: the integer matrix
            [[2, 0],
             [0, 2]]
        is rank 0 over GF(2) (where 2 == 0) but rank 2 over GF(3).
        Since our API requires canonical residues, the caller reduces
        mod p before sending: over GF(2) this becomes [[0,0],[0,0]] (rank 0)
        and over GF(3) it stays [[2,0],[0,2]] (rank 2).
        """
        # Over GF(2), 2 mod 2 = 0.
        rank_gf2 = compute_rank(
            PrimeFieldMatrixRequest(prime=2, entries=((0, 0), (0, 0)))
        )
        assert rank_gf2.rank == 0
        # Over GF(3), 2 mod 3 = 2.
        rank_gf3 = compute_rank(
            PrimeFieldMatrixRequest(prime=3, entries=((2, 0), (0, 2)))
        )
        assert rank_gf3.rank == 2

    def test_rank_gf5(self):
        """3x3 matrix with one zero row over GF(5)."""
        req = PrimeFieldMatrixRequest(
            prime=5,
            entries=((1, 2, 3), (4, 0, 1), (0, 0, 0)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_dependency_rows(self):
        """Dependent rows produce lower rank."""
        req = PrimeFieldMatrixRequest(
            prime=7,
            entries=((1, 2, 3), (2, 4, 6), (1, 0, 1)),
        )
        result = compute_rank(req)
        # Row 2 is 2 * row 1 over GF(7), so rank 2.
        assert result.rank == 2

    def test_rectangular_tall(self):
        """Tall rectangular matrix."""
        req = PrimeFieldMatrixRequest(
            prime=2,
            entries=((1, 0), (0, 1), (1, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_rectangular_wide(self):
        """Wide rectangular matrix."""
        req = PrimeFieldMatrixRequest(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_invalid_non_prime(self):
        """Non-prime modulus should raise."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=4, entries=((1, 0), (0, 1)))

    def test_invalid_entry_out_of_range(self):
        """Entry >= prime should raise."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=2, entries=((2, 0), (0, 1)))

    def test_empty_matrix_rejected(self):
        """Empty entries should raise."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=2, entries=())

    def test_prime_2147483647(self):
        """Large prime from the issue example."""
        req = PrimeFieldMatrixRequest(
            prime=2147483647,
            entries=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 3


class TestRref:
    def test_identity_gf2(self):
        """Identity matrix is already in RREF."""
        req = PrimeFieldMatrixRequest(prime=2, entries=((1, 0), (0, 1)))
        result = compute_rref(req)
        assert result.rref == ((1, 0), (0, 1))
        assert result.pivot_columns == (0, 1)
        assert result.rank == 2

    def test_rref_gf2_dependent(self):
        """RREF of a matrix with a dependent row over GF(2)."""
        req = PrimeFieldMatrixRequest(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1), (1, 1, 0)),
        )
        result = compute_rref(req)
        assert result.rank == 2
        assert len(result.pivot_columns) == 2

    def test_rref_gf3(self):
        """RREF over GF(3)."""
        req = PrimeFieldMatrixRequest(
            prime=3,
            entries=((1, 1), (1, 1)),
        )
        result = compute_rref(req)
        assert result.rank == 1
        assert result.pivot_columns == (0,)

    def test_rref_entries_canonical(self):
        """RREF entries should be canonical residues."""
        req = PrimeFieldMatrixRequest(
            prime=5,
            entries=((3, 1), (2, 4)),
        )
        result = compute_rref(req)
        for row in result.rref:
            for value in row:
                assert 0 <= value < 5

    def test_rref_rank_equals_pivots(self):
        """Rank must equal the number of pivot columns."""
        req = PrimeFieldMatrixRequest(
            prime=11,
            entries=((1, 2, 3), (4, 5, 6), (7, 8, 9)),
        )
        result = compute_rref(req)
        assert result.rank == len(result.pivot_columns)


class TestNullspace:
    def test_full_rank_nullspace_empty(self):
        """Full rank matrix has empty nullspace."""
        req = PrimeFieldMatrixRequest(prime=2, entries=((1, 0), (0, 1)))
        result = compute_nullspace(req)
        assert result.nullity == 0
        assert result.nullspace == ()

    def test_nullspace_gf2(self):
        """Nullspace of [[1,0,1],[0,1,1]] over GF(2)."""
        req = PrimeFieldMatrixRequest(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1)),
        )
        result = compute_nullspace(req)
        assert result.nullity == 1
        assert len(result.nullspace) == 1
        # Nullspace vector should have 3 components.
        assert len(result.nullspace[0]) == 3
        # All entries should be in [0, prime).
        for v in result.nullspace:
            for x in v:
                assert 0 <= x < 2

    def test_zero_matrix_nullspace(self):
        """Zero matrix has full nullspace."""
        req = PrimeFieldMatrixRequest(
            prime=2,
            entries=((0, 0), (0, 0)),
        )
        result = compute_nullspace(req)
        assert result.nullity == 2

    def test_nullspace_entries_canonical(self):
        """Nullspace entries should be canonical residues."""
        req = PrimeFieldMatrixRequest(
            prime=5,
            entries=((1, 2, 3, 1), (4, 1, 0, 2)),
        )
        result = compute_nullspace(req)
        for v in result.nullspace:
            for x in v:
                assert 0 <= x < 5

    def test_nullity_plus_rank_equals_columns(self):
        """Rank-nullity theorem: rank + nullity = columns."""
        req = PrimeFieldMatrixRequest(
            prime=3,
            entries=((1, 0, 1), (0, 1, 1), (1, 1, 2)),
        )
        rank_result = compute_rank(req)
        null_result = compute_nullspace(req)
        assert rank_result.rank + null_result.nullity == 3


class TestRequestValidation:
    def test_non_prime_rejected(self):
        """4 is not prime."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=4, entries=((1, 0), (0, 1)))

    def test_jagged_rows_rejected(self):
        """Rows of different lengths should raise."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=2, entries=((1, 0, 1), (0, 1)))

    def test_negative_entry_rejected(self):
        """Negative entries are not canonical residues."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=2, entries=((-1, 0), (0, 1)))

    def test_entry_ge_prime_rejected(self):
        """Entries >= prime are not canonical."""
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(prime=2, entries=((2, 0), (0, 1)))


class TestResultReplay:
    """Serialized results must revalidate only when they match the source."""

    def test_forged_rank_rejected(self):
        request = PrimeFieldMatrixRequest(prime=2, entries=((1, 0), (0, 1)))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldMatrixRankResult(prime=2, source=request, rank=0)

    def test_forged_rref_rejected(self):
        request = PrimeFieldMatrixRequest(prime=2, entries=((1, 0), (0, 1)))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldRrefResult(
                prime=2,
                source=request,
                rref=((0, 0), (0, 0)),
                pivot_columns=(),
                rank=0,
            )

    def test_forged_nullspace_rejected(self):
        request = PrimeFieldMatrixRequest(prime=2, entries=((0, 0),))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldNullspaceResult(prime=2, source=request, nullspace=(), nullity=0)

    def test_prime_mismatch_rejected(self):
        request = PrimeFieldMatrixRequest(prime=2, entries=((1, 0), (0, 1)))
        result = compute_rank(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldMatrixRankResult(prime=3, source=result.source, rank=result.rank)
        rref_result = compute_rref(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldRrefResult(
                prime=3,
                source=rref_result.source,
                rref=rref_result.rref,
                pivot_columns=rref_result.pivot_columns,
                rank=rref_result.rank,
            )
        ns_result = compute_nullspace(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldNullspaceResult(
                prime=3,
                source=ns_result.source,
                nullspace=ns_result.nullspace,
                nullity=ns_result.nullity,
            )

    def test_genuine_results_round_trip(self):
        request = PrimeFieldMatrixRequest(prime=5, entries=((1, 2, 3), (2, 4, 1)))
        rank_result = compute_rank(request)
        assert rank_result.rank == 1
        rref_result = compute_rref(request)
        assert rref_result.rank == 1
        ns_result = compute_nullspace(request)
        assert len(ns_result.nullspace) == 2
