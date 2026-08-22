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
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=0, col=1, value="1")),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        assert result.groups[0].betti == 0
        assert result.groups[1].betti == 1

    def test_exact_sequence(self):
        """An exact sequence has zero homology everywhere."""
        # Valid chain complex with d^2=0: C0=1, C1=2, C2=1 over GF(2)
        # d1: C1 -> C0 is 1x2 [[1,1]] rank1, d2: C2 -> C1 is 2x1 [[1],[1]] rank1
        # d1*d2 = [[2]] = 0 mod2, so d^2=0.
        cx = ChainComplex(
            prime=2,
            min_degree=0,
            max_degree=2,
            dimensions=(1, 2, 1),
            differentials=(
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=0, col=1, value="1")),
                (MatrixEntry(row=0, col=0, value="1"), MatrixEntry(row=1, col=0, value="1")),
            ),
        )
        result = compute_homology(HomologyRequest(complex=cx))
        # H0 = 1 -1 =0, H1 = (2-1)-1=0, H2=1-1=0
        assert result.groups[0].betti == 0
        assert result.groups[1].betti == 0
        assert result.groups[2].betti == 0

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
