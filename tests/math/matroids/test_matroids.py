"""Tests for linear matroid operations over a prime field."""

from __future__ import annotations

import pytest

from jacobian.math.matroids._models import (
    LinearMatroid,
    MatroidClosureRequest,
    MatroidRankRequest,
)
from jacobian.math.matroids._operations import (
    compute_closure,
    compute_rank,
)


class TestMatroidRank:
    """Test matroid rank computation."""

    def test_full_rank(self):
        """U(2,3) has rank 2."""
        matroid = LinearMatroid(prime=5, num_rows=2, columns=((1, 0), (0, 1), (1, 1)))
        result = compute_rank(MatroidRankRequest(matroid=matroid))
        assert result.rank == 2

    def test_dependent_columns(self):
        """A matrix with dependent columns has lower rank."""
        matroid = LinearMatroid(
            prime=7,
            num_rows=3,
            columns=((1, 0, 0), (0, 1, 0), (1, 1, 0)),
        )
        result = compute_rank(MatroidRankRequest(matroid=matroid))
        assert result.rank == 2

    def test_single_column(self):
        """A single nonzero column has rank 1."""
        matroid = LinearMatroid(prime=5, num_rows=3, columns=((1, 2, 3),))
        result = compute_rank(MatroidRankRequest(matroid=matroid))
        assert result.rank == 1

    def test_zero_column(self):
        """A zero column contributes no rank."""
        matroid = LinearMatroid(prime=5, num_rows=2, columns=((0, 0), (1, 1)))
        result = compute_rank(MatroidRankRequest(matroid=matroid))
        assert result.rank == 1


class TestMatroidClosure:
    """Test matroid closure computation."""

    def test_closure_includes_span(self):
        """Closure of {0, 1} in U(2,3) includes element 2 (in the span)."""
        matroid = LinearMatroid(prime=5, num_rows=2, columns=((1, 0), (0, 1), (1, 1)))
        result = compute_closure(MatroidClosureRequest(matroid=matroid, subset=(0, 1)))
        assert result.closure == (0, 1, 2)

    def test_closure_of_singleton(self):
        """Closure of a single independent element is just that element."""
        matroid = LinearMatroid(prime=5, num_rows=2, columns=((1, 0), (0, 1), (1, 1)))
        result = compute_closure(MatroidClosureRequest(matroid=matroid, subset=(0,)))
        assert result.closure == (0,)

    def test_closure_empty(self):
        """Closure of the empty set is the set of loops (zero columns)."""
        matroid = LinearMatroid(
            prime=5,
            num_rows=2,
            columns=((0, 0), (1, 0), (0, 1)),
        )
        result = compute_closure(MatroidClosureRequest(matroid=matroid, subset=()))
        # The zero column is in the closure of the empty set
        assert 0 in result.closure

    def test_closure_is_flat(self):
        """The rank of the closure equals the rank of the subset."""
        matroid = LinearMatroid(prime=7, num_rows=3, columns=((1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)))
        result = compute_closure(MatroidClosureRequest(matroid=matroid, subset=(0, 1)))
        # The closure should include column 2 (in span of {0,1}) but not 3
        assert result.rank == 2
