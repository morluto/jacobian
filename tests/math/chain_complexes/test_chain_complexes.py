"""Tests for chain complex operations."""

from __future__ import annotations

from jacobian.math.chain_complexes._models import (
    ChainComplex,
    HomologyRequest,
    MatrixEntry,
)
from jacobian.math.chain_complexes._operations import compute_homology


class TestHomology:
    """Test chain complex homology computation."""

    def test_zero_differentials(self):
        """With zero differentials, homology equals the chain groups."""
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(1, 2, 1),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 1
        assert result.groups[1].betti == 2
        assert result.groups[2].betti == 1

    def test_identity_differential(self):
        """A complex with an identity-like differential reduces homology."""
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=1,
            dimensions=(1, 2),
            differentials=(
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=1, col=0, value="1")),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 0
        assert result.groups[1].betti == 1

    def test_exact_sequence(self):
        """An exact sequence has zero homology everywhere."""
        # d: C_0 -> C_1 is full rank, d: C_1 -> C_2 is full rank
        # C_0 = 2, C_1 = 2, C_2 = 2 over GF(2)
        # d^0 is 2x2 identity, d^1 is 2x2 identity
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(2, 2, 2),
            differentials=(
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=1, col=1, value="1")),
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=1, col=1, value="1")),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        # d^0 has rank 2, d^1 has rank 2
        # H^0 = (2 - 2) - 0 = 0
        # H^1 = (2 - 2) - 2 = -2 → 0
        # H^2 = (2 - 0) - 2 = 0
        assert result.groups[0].betti == 0

    def test_single_degree(self):
        """A complex with one degree has homology equal to that degree."""
        cx = ChainComplex(
            prime=5,
            min_degree=0,
            max_degree=0,
            dimensions=(3,),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 3

    def test_prime_field(self):
        """Test with a different prime field."""
        cx = ChainComplex(
            prime=7,
            min_degree=0,
            max_degree=1,
            dimensions=(2, 2),
            differentials=(),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 2
        assert result.groups[1].betti == 2
