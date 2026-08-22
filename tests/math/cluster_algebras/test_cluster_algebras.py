"""Tests for cluster algebra operations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.cluster_algebras._models import (
    ExchangeMatrix,
    GVectorRequest,
    GVectorResult,
    SeedMutationRequest,
)
from jacobian.math.cluster_algebras._operations import (
    compute_g_vectors,
    mutate_seed,
)


class TestSeedMutation:
    """Test cluster seed mutation."""

    def test_a2_mutation_at_0(self):
        """Mutate the A2 seed at index 0."""
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        result = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=0))
        assert result.exchange_matrix.entries[0][1] == -1
        assert result.exchange_matrix.entries[1][0] == 1

    def test_a2_mutation_at_1(self):
        """Mutate the A2 seed at index 1."""
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        result = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=1))
        assert result.exchange_matrix.entries[0][1] == -1
        assert result.exchange_matrix.entries[1][0] == 1

    def test_mutation_involutive(self):
        """Double mutation at the same index returns to the original."""
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        result1 = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=0))
        result2 = mutate_seed(
            SeedMutationRequest(
                exchange_matrix=result1.exchange_matrix, mutation_index=0
            )
        )
        assert result2.exchange_matrix.entries == b.entries

    def test_3x3_mutation(self):
        """Mutate a 3x3 skew-symmetric matrix."""
        b = ExchangeMatrix(
            n=3,
            entries=((0, 1, 0), (-1, 0, 0), (0, 0, 0)),
            symmetrizer=(1, 1, 1),
        )
        result = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=1))
        # The mutated matrix should still be skew-symmetric
        new = result.exchange_matrix.entries
        for i in range(3):
            assert new[i][i] == 0

    def test_invalid_mutation_index(self):
        """Mutation index out of range should fail."""
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        with pytest.raises(ValueError, match="mutation_index"):
            SeedMutationRequest(exchange_matrix=b, mutation_index=2)

    def test_skew_symmetrizable(self):
        """A non-skew-symmetrizable matrix should fail."""
        with pytest.raises(ValueError, match="skew-symmetrizability"):
            ExchangeMatrix(n=2, entries=((0, 1), (1, 0)), symmetrizer=(1, 1))

    def test_zero_symmetrizer_rejected(self):
        """A symmetrizer with a zero entry is not an exchange matrix."""
        with pytest.raises(ValueError, match="strictly positive"):
            ExchangeMatrix(
                n=2, entries=((0, 1), (-1, 0)), symmetrizer=(0, 2)
            )

    def test_negative_symmetrizer_rejected(self):
        """A symmetrizer with a negative entry is rejected."""
        with pytest.raises(ValueError, match="strictly positive"):
            ExchangeMatrix(
                n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, -1)
            )


class TestCoefficientBounds:
    """Admitted seeds are bounded so mutation output stays bounded."""

    def test_request_rejects_entry_beyond_input_bound(self):
        b = ExchangeMatrix(
            n=2, entries=((0, 10**64), (-(10**64), 0)), symmetrizer=(1, 1)
        )
        with pytest.raises(ValidationError, match="64-digit bound"):
            SeedMutationRequest(exchange_matrix=b, mutation_index=0)

    def test_request_accepts_entry_at_input_bound(self):
        edge = 10**64 - 1
        b = ExchangeMatrix(n=2, entries=((0, edge), (-edge, 0)), symmetrizer=(1, 1))
        result = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=0))
        assert result.exchange_matrix.entries == ((0, -edge), (edge, 0))

    def test_mutation_near_input_bound_stays_within_result_ceiling(self):
        # Mutating a path matrix at the middle index forms an entry of
        # magnitude ~N^2; it must remain an admissible exchange matrix.
        edge = 10**63
        b = ExchangeMatrix(
            n=3,
            entries=((0, edge, 0), (-edge, 0, edge), (0, -edge, 0)),
            symmetrizer=(1, 1, 1),
        )
        result = mutate_seed(SeedMutationRequest(exchange_matrix=b, mutation_index=1))
        mutated = result.exchange_matrix.entries[0][2]
        assert abs(mutated) == edge * edge
        assert len(str(abs(mutated))) <= 129

    def test_rejects_symmetrizer_beyond_bound(self):
        # A zero row-pair keeps the matrix skew-symmetrizable so the
        # symmetrizer bound is what rejects the seed.
        with pytest.raises(ValidationError, match="symmetrizer coefficients"):
            ExchangeMatrix(
                n=2,
                entries=((0, 0), (0, 0)),
                symmetrizer=(1, 10**64),
            )

    def test_result_ceiling_rejects_oversized_entries(self):
        # Even skew-symmetrizable matrices cannot carry unbounded integers.
        huge = 10**130
        with pytest.raises(ValidationError, match="129-digit bound"):
            ExchangeMatrix(
                n=2, entries=((0, huge), (-huge, 0)), symmetrizer=(1, 1)
            )

    def test_gvector_request_rejects_unbounded_coefficients(self):
        b = ExchangeMatrix(
            n=2, entries=((0, 10**64 + 1), (-(10**64 + 1), 0)), symmetrizer=(1, 1)
        )
        with pytest.raises(ValidationError, match="64-digit bound"):
            GVectorRequest(exchange_matrix=b)


class TestGVectorBinding:
    """Exact g-vector results must replay against their retained source."""

    def test_result_binds_to_source_and_convention(self):
        b = ExchangeMatrix(n=3, entries=((0, 1, 0), (-1, 0, 1), (0, -1, 0)), symmetrizer=(1, 1, 1))
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.exchange_matrix == b
        assert result.g_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        assert result.convention == "FOMIN_ZELEVINSKY"
        GVectorResult.model_validate(result.model_dump())

    def test_result_rejects_non_identity_matrix(self):
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        with pytest.raises(ValidationError, match="identity"):
            GVectorResult(
                exchange_matrix=b,
                g_matrix=((1, 1), (0, 1)),
                convention="FOMIN_ZELEVINSKY",
            )

    def test_result_rejects_dimension_mismatch(self):
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        with pytest.raises(ValidationError, match="identity"):
            GVectorResult(
                exchange_matrix=b,
                g_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                convention="FOMIN_ZELEVINSKY",
            )

    def test_result_rejects_arbitrary_convention(self):
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        with pytest.raises(ValidationError):
            GVectorResult(
                exchange_matrix=b,
                g_matrix=((1, 0), (0, 1)),
                convention="anything",
            )

    def test_forged_empty_payload_rejected(self):
        b = ExchangeMatrix(n=3, entries=((0, 1, 0), (-1, 0, 1), (0, -1, 0)), symmetrizer=(1, 1, 1))
        with pytest.raises(ValidationError):
            GVectorResult(exchange_matrix=b, g_matrix=(), convention="anything")


class TestGVector:
    """Test g-vector computation."""

    def test_initial_g_vectors(self):
        """Initial g-vectors should be the identity."""
        b = ExchangeMatrix(n=2, entries=((0, 1), (-1, 0)), symmetrizer=(1, 1))
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.g_matrix == ((1, 0), (0, 1))

    def test_3x3_g_vectors(self):
        """g-vectors for a 3x3 seed should be identity."""
        b = ExchangeMatrix(
            n=3,
            entries=((0, 1, 0), (-1, 0, 0), (0, 0, 0)),
            symmetrizer=(1, 1, 1),
        )
        result = compute_g_vectors(GVectorRequest(exchange_matrix=b))
        assert result.g_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
