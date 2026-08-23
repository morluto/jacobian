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


def pfm(prime: int, entries: tuple[tuple[int, ...], ...]):
    """Build a request from raw rows using the canonical matrix value."""
    from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

    return PrimeFieldMatrixRequest(
        matrix=PrimeFieldMatrix(
            prime=prime,
            entries=entries,
            columns=len(entries[0]) if entries else 0,
        )
    )


class TestRank:
    def test_full_rank_gf2(self):
        """Identity 3x3 over GF(2) has rank 3."""
        req = pfm(prime=2, entries=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        result = compute_rank(req)
        assert result.rank == 3
        assert result.prime == 2
        assert len(result.source.matrix.entries) == 3
        assert len(result.source.matrix.entries[0]) == 3

    def test_zero_matrix_gf2(self):
        """Zero matrix over GF(2) has rank 0."""
        req = pfm(prime=2, entries=((0, 0), (0, 0)))
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
        rank_gf2 = compute_rank(pfm(prime=2, entries=((0, 0), (0, 0))))
        assert rank_gf2.rank == 0
        # Over GF(3), 2 mod 3 = 2.
        rank_gf3 = compute_rank(pfm(prime=3, entries=((2, 0), (0, 2))))
        assert rank_gf3.rank == 2

    def test_rank_gf5(self):
        """3x3 matrix with one zero row over GF(5)."""
        req = pfm(
            prime=5,
            entries=((1, 2, 3), (4, 0, 1), (0, 0, 0)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_dependency_rows(self):
        """Dependent rows produce lower rank."""
        req = pfm(
            prime=7,
            entries=((1, 2, 3), (2, 4, 6), (1, 0, 1)),
        )
        result = compute_rank(req)
        # Row 2 is 2 * row 1 over GF(7), so rank 2.
        assert result.rank == 2

    def test_rectangular_tall(self):
        """Tall rectangular matrix."""
        req = pfm(
            prime=2,
            entries=((1, 0), (0, 1), (1, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_rectangular_wide(self):
        """Wide rectangular matrix."""
        req = pfm(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 2

    def test_invalid_non_prime(self):
        """Non-prime modulus should raise."""
        with pytest.raises(ValidationError):
            pfm(prime=4, entries=((1, 0), (0, 1)))

    def test_invalid_entry_out_of_range(self):
        """Entry >= prime should raise."""
        with pytest.raises(ValidationError):
            pfm(prime=2, entries=((2, 0), (0, 1)))

    def test_empty_entries_with_explicit_columns_accepted(self):
        """The canonical empty matrix composes as a bounded request.

        Full-rank nullspace producers return `PrimeFieldMatrix(entries=(),
        columns=n)`; that value must enter rank, RREF, and nullspace
        unchanged instead of being rejected as non-composable.
        """
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        empty = PrimeFieldMatrix(prime=5, entries=(), columns=3)
        req = PrimeFieldMatrixRequest(matrix=empty)
        assert compute_rank(req).rank == 0
        rref_result = compute_rref(req)
        assert rref_result.rref_matrix.entries == ()
        assert rref_result.pivot_columns == ()
        assert rref_result.rank == 0
        ns_result = compute_nullspace(req)
        # The nullspace of a 0x3 system is all of GF(5)^3.
        assert ns_result.nullity == 3
        assert ns_result.nullspace_matrix.entries == ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def test_full_rank_nullspace_value_feeds_all_consumers(self):
        """Relay scenario from review: the empty nullspace basis is a request."""
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        source = PrimeFieldMatrix(prime=5, entries=((1, 2), (3, 4)), columns=2)
        ns_result = compute_nullspace(PrimeFieldMatrixRequest(matrix=source))
        assert ns_result.nullspace_matrix.entries == ()
        basis_request = PrimeFieldMatrixRequest(matrix=ns_result.nullspace_matrix)
        assert compute_rank(basis_request).rank == 0
        relayed_ns = compute_nullspace(basis_request)
        assert relayed_ns.nullspace_matrix.columns == 2
        assert relayed_ns.nullity == 2

    def test_prime_2147483647(self):
        """Large prime from the issue example."""
        req = pfm(
            prime=2147483647,
            entries=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        result = compute_rank(req)
        assert result.rank == 3


class TestRref:
    def test_identity_gf2(self):
        """Identity matrix is already in RREF."""
        req = pfm(prime=2, entries=((1, 0), (0, 1)))
        result = compute_rref(req)
        assert result.rref_matrix.entries == ((1, 0), (0, 1))
        assert result.pivot_columns == (0, 1)
        assert result.rank == 2

    def test_rref_gf2_dependent(self):
        """RREF of a matrix with a dependent row over GF(2)."""
        req = pfm(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1), (1, 1, 0)),
        )
        result = compute_rref(req)
        assert result.rank == 2
        assert len(result.pivot_columns) == 2

    def test_rref_gf3(self):
        """RREF over GF(3)."""
        req = pfm(
            prime=3,
            entries=((1, 1), (1, 1)),
        )
        result = compute_rref(req)
        assert result.rank == 1
        assert result.pivot_columns == (0,)

    def test_rref_entries_canonical(self):
        """RREF entries should be canonical residues."""
        req = pfm(
            prime=5,
            entries=((3, 1), (2, 4)),
        )
        result = compute_rref(req)
        for row in result.rref_matrix.entries:
            for value in row:
                assert 0 <= value < 5

    def test_rref_rank_equals_pivots(self):
        """Rank must equal the number of pivot columns."""
        req = pfm(
            prime=11,
            entries=((1, 2, 3), (4, 5, 6), (7, 8, 9)),
        )
        result = compute_rref(req)
        assert result.rank == len(result.pivot_columns)


class TestNullspace:
    def test_full_rank_nullspace_empty(self):
        """Full rank matrix has empty nullspace."""
        req = pfm(prime=2, entries=((1, 0), (0, 1)))
        result = compute_nullspace(req)
        assert result.nullity == 0
        assert result.nullspace_matrix.entries == ()

    def test_nullspace_gf2(self):
        """Nullspace of [[1,0,1],[0,1,1]] over GF(2)."""
        req = pfm(
            prime=2,
            entries=((1, 0, 1), (0, 1, 1)),
        )
        result = compute_nullspace(req)
        assert result.nullity == 1
        assert len(result.nullspace_matrix.entries) == 1
        # Nullspace vector should have 3 components.
        assert len(result.nullspace_matrix.entries[0]) == 3
        # All entries should be in [0, prime).
        for v in result.nullspace_matrix.entries:
            for x in v:
                assert 0 <= x < 2

    def test_zero_matrix_nullspace(self):
        """Zero matrix has full nullspace."""
        req = pfm(
            prime=2,
            entries=((0, 0), (0, 0)),
        )
        result = compute_nullspace(req)
        assert result.nullity == 2

    def test_nullspace_entries_canonical(self):
        """Nullspace entries should be canonical residues."""
        req = pfm(
            prime=5,
            entries=((1, 2, 3, 1), (4, 1, 0, 2)),
        )
        result = compute_nullspace(req)
        for v in result.nullspace_matrix.entries:
            for x in v:
                assert 0 <= x < 5

    def test_nullity_plus_rank_equals_columns(self):
        """Rank-nullity theorem: rank + nullity = columns."""
        req = pfm(
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
            pfm(prime=4, entries=((1, 0), (0, 1)))

    def test_jagged_rows_rejected(self):
        """Rows of different lengths should raise."""
        with pytest.raises(ValidationError):
            pfm(prime=2, entries=((1, 0, 1), (0, 1)))

    def test_negative_entry_rejected(self):
        """Negative entries are not canonical residues."""
        with pytest.raises(ValidationError):
            pfm(prime=2, entries=((-1, 0), (0, 1)))

    def test_entry_ge_prime_rejected(self):
        """Entries >= prime are not canonical."""
        with pytest.raises(ValidationError):
            pfm(prime=2, entries=((2, 0), (0, 1)))

    def test_canonical_object_composition_cannot_bypass_prime_bound(self):
        """Python-mode composition must not smuggle an out-of-domain field
        past the request bound: the request model owns the MAX_PRIME bound."""
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        canonical = PrimeFieldMatrix(prime=2**127 - 1, entries=((1,),), columns=1)
        with pytest.raises(ValidationError):
            PrimeFieldMatrixRequest(matrix=canonical)


class TestResultReplay:
    """Serialized results must revalidate only when they match the source."""

    def test_forged_rank_rejected(self):
        request = pfm(prime=2, entries=((1, 0), (0, 1)))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldMatrixRankResult(prime=2, source=request, rank=0)

    def test_forged_rref_rejected(self):
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        request = pfm(prime=2, entries=((1, 0), (0, 1)))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldRrefResult(
                prime=2,
                source=request,
                rref_matrix=PrimeFieldMatrix(
                    prime=2, entries=((0, 0), (0, 0)), columns=2
                ),
                pivot_columns=(),
                rank=0,
            )

    def test_forged_nullspace_rejected(self):
        from jacobian.math.prime_field_linear_algebra import PrimeFieldMatrix

        request = pfm(prime=2, entries=((0, 0),))
        with pytest.raises(ValidationError, match="recomputation"):
            PrimeFieldNullspaceResult(
                prime=2,
                source=request,
                nullspace_matrix=PrimeFieldMatrix(
                    prime=2, entries=((1, 0),), columns=2
                ),
                nullity=0,
            )

    def test_prime_mismatch_rejected(self):
        request = pfm(prime=2, entries=((1, 0), (0, 1)))
        result = compute_rank(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldMatrixRankResult(prime=3, source=result.source, rank=result.rank)
        rref_result = compute_rref(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldRrefResult(
                prime=3,
                source=rref_result.source,
                rref_matrix=rref_result.rref_matrix,
                pivot_columns=rref_result.pivot_columns,
                rank=rref_result.rank,
            )
        ns_result = compute_nullspace(request)
        with pytest.raises(ValidationError, match="prime"):
            PrimeFieldNullspaceResult(
                prime=3,
                source=ns_result.source,
                nullspace_matrix=ns_result.nullspace_matrix,
                nullity=ns_result.nullity,
            )

    def test_genuine_results_round_trip(self):
        request = pfm(prime=5, entries=((1, 2, 3), (2, 4, 1)))
        rank_result = compute_rank(request)
        assert rank_result.rank == 1
        rref_result = compute_rref(request)
        assert rref_result.rank == 1
        ns_result = compute_nullspace(request)
        assert len(ns_result.nullspace_matrix.entries) == 2


class TestCanonicalValueComposition:
    def test_oversized_prime_rejected_before_construction(self):
        """A huge characteristic is bounded before primality work."""
        with pytest.raises(ValidationError, match=r"\[2, 2147483647\]"):
            PrimeFieldMatrixRequest.model_validate(
                {
                    "matrix": {
                        # A ~3,000-digit Mersenne-style prime fits an int but
                        # must be rejected before any isprime/modular work.
                        "prime": 2**9941 - 1,
                        "entries": [[1]],
                        "columns": 1,
                    }
                }
            )

    def test_rref_and_nullspace_results_feed_rank_unchanged(self):
        """The canonical result matrices compose into consumers unchanged."""
        request = pfm(prime=5, entries=((1, 2), (2, 4)))
        rref_result = compute_rref(request)
        assert (
            compute_rank(PrimeFieldMatrixRequest(matrix=rref_result.rref_matrix)).rank
            == 1
        )
        ns_result = compute_nullspace(request)
        assert ns_result.nullity == 1
        relayed = PrimeFieldNullspaceResult.model_validate(ns_result.model_dump())
        assert relayed.nullspace_matrix == ns_result.nullspace_matrix
        assert (
            compute_rank(
                PrimeFieldMatrixRequest(matrix=ns_result.nullspace_matrix)
            ).rank
            == 1
        )

    def test_duplicate_family_removed_from_catalog(self):
        """The duplicate underscore-named family is gone from discovery."""
        import subprocess
        import sys

        code = (
            "from jacobian.catalog.builtins import BUILTIN_TOOLS\n"
            "ids = [t.operation_id for t in BUILTIN_TOOLS]\n"
            "assert not [i for i in ids if i.startswith('prime_field_matrix.')], ids\n"
            "assert 'prime_field.matrix.rank.compute' in ids\n"
        )
        import os

        env = dict(os.environ, PYTHONPATH="src")
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
