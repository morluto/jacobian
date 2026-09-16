"""Exact and contract tests for Laurent valuation profiles."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series import (
    TruncatedLaurentWindow,
    laurent_valuation_profile,
)
from jacobian.math.polynomials.local_series._models import (
    ValuationProfileRequest,
    ValuationProfileResult,
)
from jacobian.math.polynomials.local_series._tools import compute_valuation_profile
from jacobian.math.polynomials.local_series.values import MAX_LOCAL_SERIES_TERMS


def _rational(numerator: int, denominator: int = 1) -> CanonicalRational:
    return CanonicalRational(num=numerator, den=denominator)


def _window(
    coefficients: tuple[CanonicalRational, ...],
    *,
    valuation_lower: int = 0,
    center: CanonicalRational | None = None,
) -> TruncatedLaurentWindow:
    return TruncatedLaurentWindow(
        variable="t",
        center=center if center is not None else _rational(0),
        valuation_lower=valuation_lower,
        precision=valuation_lower + len(coefficients),
        coefficients=coefficients,
    )


def test_simple_pole_known_answer() -> None:
    series = _window((_rational(1), _rational(2), _rational(3)), valuation_lower=-1)
    result = compute_valuation_profile(ValuationProfileRequest(series=series))

    assert result.status == "NONZERO"
    assert result.valuation == -1
    assert result.leading_coefficient == _rational(1)
    assert result.pole_order == 1
    assert result.zero_order == 0


def test_zero_window_reports_zero_at_precision() -> None:
    series = _window((_rational(0), _rational(0), _rational(0)), valuation_lower=-2)
    result = compute_valuation_profile(ValuationProfileRequest(series=series))

    assert result.status == "ZERO_AT_PRECISION"
    assert result.valuation is None
    assert result.leading_coefficient is None
    assert result.pole_order == 0
    assert result.zero_order == 0


def test_leading_zeros_normalize_to_true_valuation() -> None:
    series = _window((_rational(0), _rational(0), _rational(5, 3)), valuation_lower=-4)
    result = laurent_valuation_profile(series)

    assert result.status == "NONZERO"
    assert result.valuation == -2
    assert result.leading_coefficient == _rational(5, 3)
    assert result.pole_order == 2
    assert result.zero_order == 0


def test_zero_order_at_positive_valuation() -> None:
    series = _window((_rational(0), _rational(-2)), valuation_lower=2)
    result = laurent_valuation_profile(series)

    assert result.valuation == 3
    assert result.pole_order == 0
    assert result.zero_order == 3


def test_nonzero_center_is_retained() -> None:
    series = _window((_rational(1, 2),), center=_rational(3, 4))
    result = laurent_valuation_profile(series)

    assert result.series.center == _rational(3, 4)
    assert result.valuation == 0
    assert result.pole_order == 0
    assert result.zero_order == 0


def test_defining_invariant_leading_term_reconstruction() -> None:
    coefficients = (_rational(0), _rational(7, 5), _rational(-1, 2), _rational(0))
    series = _window(coefficients, valuation_lower=-3)
    result = laurent_valuation_profile(series)

    assert result.status == "NONZERO"
    assert result.valuation is not None
    assert result.leading_coefficient is not None
    index = result.valuation - series.valuation_lower
    assert series.coefficients[index] == result.leading_coefficient
    assert all(value.num == 0 for value in series.coefficients[:index])
    assert result.pole_order == max(-result.valuation, 0)
    assert result.zero_order == max(result.valuation, 0)


def test_native_and_catalog_results_agree() -> None:
    series = _window((_rational(2), _rational(0), _rational(1, 3)), valuation_lower=-1)

    native = laurent_valuation_profile(series)
    catalog = compute_valuation_profile(ValuationProfileRequest(series=series))

    assert catalog == native
    assert catalog.series == series
    assert (
        ValuationProfileResult.model_validate_json(catalog.model_dump_json()) == catalog
    )


def test_empty_window_is_rejected_structurally() -> None:
    with pytest.raises(ValidationError):
        TruncatedLaurentWindow(
            variable="t",
            center=_rational(0),
            valuation_lower=2,
            precision=2,
            coefficients=(),
        )


def test_ragged_window_is_rejected_structurally() -> None:
    with pytest.raises(ValidationError):
        TruncatedLaurentWindow(
            variable="t",
            center=_rational(0),
            valuation_lower=0,
            precision=3,
            coefficients=(_rational(1), _rational(2)),
        )


def test_window_above_envelope_is_refused_before_scan() -> None:
    series = TruncatedLaurentWindow.model_construct(
        variable="t",
        center=_rational(0),
        valuation_lower=0,
        precision=MAX_LOCAL_SERIES_TERMS + 1,
        coefficients=tuple(_rational(0) for _ in range(MAX_LOCAL_SERIES_TERMS + 1)),
    )

    with pytest.raises(OperationResourceAdmissionError):
        laurent_valuation_profile(series)


def test_window_at_envelope_is_accepted() -> None:
    series = _window(tuple(_rational(0) for _ in range(MAX_LOCAL_SERIES_TERMS)))

    assert laurent_valuation_profile(series).status == "ZERO_AT_PRECISION"


def test_native_rejects_a_non_window_value() -> None:
    with pytest.raises(OperationDomainValidationError):
        laurent_valuation_profile("not-a-window")  # type: ignore[arg-type]
