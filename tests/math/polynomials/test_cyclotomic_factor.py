"""Cyclotomic-factor profile slice (#3724)."""

from __future__ import annotations

import pytest

from jacobian._execution import BackendFailureReason, OperationBackendError
from jacobian.math.polynomials._cyclotomic_factor import (
    MAX_FACTOR_SEARCH_INDEX,
    _convert_cyclotomic_candidate,
    cyclotomic_factor_profile,
)
from jacobian.math.polynomials._models import IntegerPolynomial


def test_malformed_cyclotomic_candidate_is_typed_backend_failure() -> None:
    with pytest.raises(OperationBackendError) as caught:
        _convert_cyclotomic_candidate(3, 2, object())
    assert caught.value.reason is BackendFailureReason.INVALID_OUTPUT


@pytest.mark.parametrize(
    ("coefficients", "expected_index"),
    [((1, 1, 1), 3), ((1, 0, 1), 4)],
)
def test_identified_cyclotomic_factors(
    coefficients: tuple[int, ...], expected_index: int
) -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=coefficients))
    assert result.status == "IDENTIFIED_CYCLOTOMIC"
    assert result.cyclotomic_index == expected_index
    assert result.searched_index_bound == MAX_FACTOR_SEARCH_INDEX


@pytest.mark.parametrize("coefficients", [(1, 1, -1), (2, 2, 2)])
def test_unidentified_cyclotomic_factors(coefficients: tuple[int, ...]) -> None:
    result = cyclotomic_factor_profile(IntegerPolynomial(coefficients=coefficients))
    assert result.status == "NOT_IDENTIFIED_IN_SUPPORTED_RANGE"
    assert result.cyclotomic_index is None
    assert result.searched_index_bound == MAX_FACTOR_SEARCH_INDEX
