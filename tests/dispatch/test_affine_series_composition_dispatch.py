"""Native and public affine composition share the full linear envelope."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationResourceAdmissionError
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import compose
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    MAX_TRUNCATION_ORDER,
    SeriesComposeResult,
    TruncatedSeries,
)


@pytest.mark.parametrize("order", (MAX_TRUNCATION_ORDER + 1, MAX_TRUNCATE_SOURCE_ORDER))
def test_public_affine_composition_round_trips_and_truncates(order: int) -> None:
    q = CanonicalRational.from_fraction
    zero = q(Fraction(0))
    large = 10**MAX_RATIONAL_DIGITS - 1
    slope = (
        Fraction(large, large - 2)
        if order == MAX_TRUNCATE_SOURCE_ORDER
        else Fraction(-5, 7)
    )
    values = (
        [Fraction(large - 4, large - 6)] * (order - 1)
        if order == MAX_TRUNCATE_SOURCE_ORDER
        else [Fraction((i % 5) - 2, 7) for i in range(1, order)]
    )
    outer = TruncatedSeries(
        variable="x",
        truncation_order=order,
        coefficients=(q(Fraction(2, 3)), q(slope)) + (zero,) * (order - 2),
    )
    inner = TruncatedSeries(
        variable="x",
        truncation_order=order,
        coefficients=(zero, *(q(value) for value in values)),
    )
    catalog = Catalog.open()
    native = compose(outer, inner)
    public = invoke_operation(
        "formal_series.rational.compose.compute",
        {
            "outer": outer.model_dump(mode="json"),
            "inner": inner.model_dump(mode="json"),
        },
        catalog,
    )
    decoded = SeriesComposeResult.model_validate_json(json.dumps(public.output))
    assert decoded == native
    expected = [Fraction(2, 3)] + [
        slope * value.as_fraction() for value in inner.coefficients[1:]
    ]
    assert [value.as_fraction() for value in decoded.result.coefficients] == expected
    prefix = invoke_operation(
        "formal_series.rational.truncate.compute",
        {
            "series": decoded.result.model_dump(mode="json"),
            "target_order": MAX_TRUNCATION_ORDER,
        },
        catalog,
    )
    assert prefix.output["result"]["truncation_order"] == MAX_TRUNCATION_ORDER


def test_public_nonlinear_composition_retains_the_original_boundary() -> None:
    order = MAX_TRUNCATION_ORDER + 1
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    outer = {
        "variable": "x",
        "truncation_order": order,
        "coefficients": [zero, zero, one] + [zero] * (order - 3),
    }
    inner = {
        "variable": "x",
        "truncation_order": order,
        "coefficients": [zero, one] + [zero] * (order - 2),
    }
    with pytest.raises(OperationResourceAdmissionError) as error:
        invoke_operation(
            "formal_series.rational.compose.compute",
            {"outer": outer, "inner": inner},
            Catalog.open(),
        )
    assert error.value.errors()[0]["type"] == "formal_power_series.input_order"
