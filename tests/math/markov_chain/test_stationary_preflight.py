"""Tests for Markov chain stationary-distribution rational-height preflight (#2067)."""

from __future__ import annotations

import pytest

from jacobian.math.markov_chain._models import TransitionMatrixRequest
from jacobian.math.markov_chain._operations import compute_stationary_distribution


def test_normal_chain_succeeds() -> None:
    """A normal 2-state chain succeeds and returns a stationary distribution."""
    request = TransitionMatrixRequest.model_validate({
        "matrix": [
            [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
            [{"num": "1", "den": "3"}, {"num": "2", "den": "3"}],
        ]
    })
    result = compute_stationary_distribution(request)
    assert result.unique


def test_extreme_chain_rejected_before_solving() -> None:
    """A chain with extreme rational heights is rejected before exact solving.

    Use 1/p^1000 and 1/q^1000 for distinct primes p, q to ensure
    the fractions are already reduced but the Hadamard bound exceeds
    the digit limit.
    """
    # Use large prime powers that have enough digits to exceed the
    # Hadamard bound (dimension^2 * max_digit_count > 32768) but fit
    # within the CanonicalRational scalar limit.
    import sys
    sys.set_int_max_str_digits(50000)
    p = 2**12000
    q = 3**12000
    with pytest.raises(Exception, match="rational height"):
        TransitionMatrixRequest.model_validate({
            "matrix": [
                [{"num": str(p - 1), "den": str(p)}, {"num": "1", "den": str(p)}],
                [{"num": "1", "den": str(q)}, {"num": str(q - 1), "den": str(q)}],
            ]
        })


def test_near_boundary_chain_succeeds() -> None:
    """A chain near the rational-height boundary still succeeds."""
    n = 50
    request = TransitionMatrixRequest.model_validate({
        "matrix": [
            [{"num": str(2**n - 1), "den": str(2**n)}, {"num": "1", "den": str(2**n)}],
            [{"num": "1", "den": str(3**n)}, {"num": str(3**n - 1), "den": str(3**n)}],
        ]
    })
    result = compute_stationary_distribution(request)
    assert result.unique
    assert len(result.extreme_distributions) == 1
