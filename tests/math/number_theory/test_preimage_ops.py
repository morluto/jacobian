"""Tests for k*sigma preimage and p-adic interval valuation profiles."""

from jacobian.math.number_theory._preimage_models import (
    KSigmaPreimageRequest,
    IntervalValuationProfileRequest,
)
from jacobian.math.number_theory._preimage_operations import (
    compute_ksigma_preimage,
    compute_interval_valuation_profile,
)


def test_ksigma_preimage_basic() -> None:
    """sigma(6) = 1+2+3+6 = 12, so 1*sigma(n)=12 should include n=6."""
    result = compute_ksigma_preimage(KSigmaPreimageRequest(k=1, target_value=12))
    assert 6 in result.preimages


def test_ksigma_preimage_none() -> None:
    """No n has sigma(n) = 2."""
    result = compute_ksigma_preimage(KSigmaPreimageRequest(k=1, target_value=2))
    assert len(result.preimages) == 0


def test_valuation_2_basic() -> None:
    result = compute_interval_valuation_profile(
        IntervalValuationProfileRequest(lower_bound=1, upper_bound=10, prime=2)
    )
    vals = [r.valuation for r in result.rows]
    assert vals == [0, 1, 0, 2, 0, 1, 0, 3, 0, 1]


def test_valuation_3_basic() -> None:
    result = compute_interval_valuation_profile(
        IntervalValuationProfileRequest(lower_bound=1, upper_bound=10, prime=3)
    )
    vals = [r.valuation for r in result.rows]
    assert vals == [0, 0, 1, 0, 0, 1, 0, 0, 2, 0]


def test_valuation_single_element() -> None:
    result = compute_interval_valuation_profile(
        IntervalValuationProfileRequest(lower_bound=8, upper_bound=8, prime=2)
    )
    assert result.rows[0].n == 8
    assert result.rows[0].valuation == 3
