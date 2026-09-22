"""Exact native valuation profiles for truncated Laurent windows."""

from __future__ import annotations

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series._models import (
    NonzeroValuation,
    ValuationProfileResult,
    ZeroValuation,
)
from jacobian.math.polynomials.local_series.arithmetic import (
    add,
    change_scale,
    deramify,
    derivative,
    divide,
    integral,
    inverse,
    multiply,
    power,
    principal_part,
    ramify,
    residue,
    shift,
    subtract,
    truncate,
)
from jacobian.math.polynomials.local_series.values import (
    MAX_LOCAL_SERIES_TERMS,
    TruncatedLaurentWindow,
)


def laurent_valuation_profile(
    series: TruncatedLaurentWindow,
) -> ValuationProfileResult:
    """Return ZERO_AT_PRECISION or the exact valuation profile of a window."""

    if not isinstance(series, TruncatedLaurentWindow):
        raise OperationDomainValidationError(
            location=("series",),
            code="local_series.valuation_profile_series_type",
            message="series must be a truncated Laurent window value",
        )
    if len(series.coefficients) > MAX_LOCAL_SERIES_TERMS:
        raise OperationResourceAdmissionError(
            location=("series", "coefficients"),
            code="local_series.valuation_profile_window_bound",
            message=(
                "Laurent window length exceeds the "
                f"{MAX_LOCAL_SERIES_TERMS}-term valuation envelope"
            ),
        )
    for index, coefficient in enumerate(series.coefficients):
        if coefficient.as_fraction() != 0:
            valuation = series.valuation_lower + index
            return ValuationProfileResult._from_kernel(
                series,
                conclusion=NonzeroValuation(
                    status="NONZERO",
                    valuation=valuation,
                    leading_coefficient=coefficient,
                    pole_order=max(-valuation, 0),
                    zero_order=max(valuation, 0),
                ),
            )
    return ValuationProfileResult._from_kernel(
        series, conclusion=ZeroValuation(status="ZERO_AT_PRECISION")
    )


__all__ = [
    "add",
    "change_scale",
    "deramify",
    "derivative",
    "divide",
    "integral",
    "inverse",
    "laurent_valuation_profile",
    "multiply",
    "power",
    "principal_part",
    "ramify",
    "residue",
    "shift",
    "subtract",
    "truncate",
]
