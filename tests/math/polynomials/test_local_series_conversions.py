from __future__ import annotations

import json

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.polynomials.local_series import (
    TruncatedLaurentWindow,
    from_power_series,
    to_power_series,
)
from jacobian.math.polynomials.local_series._tools import TOOLS
from jacobian.math.polynomials.series._models import TruncatedSeries


def _q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def test_power_series_roundtrips_through_normalized_laurent_window() -> None:
    source = TruncatedSeries(
        variable="x",
        truncation_order=4,
        coefficients=(_q(0), _q(2), _q(3), _q(0)),
    )

    laurent = from_power_series(source)
    assert (laurent.place, laurent.center) == ("FINITE", _q(0))
    assert (laurent.valuation_lower, laurent.precision) == (1, 4)
    assert laurent.coefficients == (_q(2), _q(3), _q(0))

    converted = to_power_series(laurent)
    assert converted.status == "CONVERTED"
    assert converted.result == source
    assert converted.first_negative_exponent is None
    assert type(converted).model_validate_json(converted.model_dump_json()) == converted


def test_zero_and_known_zero_negative_slots_convert_without_claiming_a_pole() -> None:
    zero = TruncatedSeries(
        variable="x", truncation_order=3, coefficients=(_q(0), _q(0), _q(0))
    )
    assert from_power_series(zero).valuation_lower == 0

    source = TruncatedLaurentWindow(
        variable="x",
        center=_q(0),
        valuation_lower=-2,
        precision=3,
        coefficients=(_q(0), _q(0), _q(5), _q(0), _q(0)),
    )
    converted = to_power_series(source)
    assert converted.status == "CONVERTED"
    assert converted.result is not None
    assert converted.result.coefficients == (_q(5), _q(0), _q(0))


def test_pole_returns_structural_obstruction_without_truncating() -> None:
    source = TruncatedLaurentWindow(
        variable="x",
        center=_q(0),
        valuation_lower=-2,
        precision=2,
        coefficients=(_q(3), _q(0), _q(1), _q(2)),
    )
    result = to_power_series(source)
    assert result.status == "HAS_NEGATIVE_EXPONENTS"
    assert result.result is None
    assert result.first_negative_exponent == -2
    assert result.source == source


def test_conversion_requires_origin_parent_and_power_series_envelopes() -> None:
    nonzero_center = TruncatedLaurentWindow(
        variable="x",
        center=_q(1),
        valuation_lower=0,
        precision=1,
        coefficients=(_q(1),),
    )
    with pytest.raises(OperationDomainValidationError):
        to_power_series(nonzero_center)

    at_infinity = TruncatedLaurentWindow(
        variable="x",
        place="INFINITY",
        center=_q(0),
        valuation_lower=0,
        precision=1,
        coefficients=(_q(1),),
    )
    with pytest.raises(OperationDomainValidationError):
        to_power_series(at_infinity)

    too_wide = TruncatedSeries(
        variable="x",
        truncation_order=513,
        coefficients=tuple(_q(0) for _ in range(513)),
    )
    with pytest.raises(OperationResourceAdmissionError):
        from_power_series(too_wide)


def test_conversion_operations_are_published_and_examples_execute() -> None:
    operations = {
        item.operation_id: item
        for item in TOOLS
        if item.operation_id
        in {
            "local_series.from_power_series.compute",
            "local_series.to_power_series.compute",
        }
    }
    assert set(operations) == {
        "local_series.from_power_series.compute",
        "local_series.to_power_series.compute",
    }
    for operation in operations.values():
        request = operation.request_type.model_validate_json(
            json.dumps(operation.examples[0].input)
        )
        operation.run(request)
