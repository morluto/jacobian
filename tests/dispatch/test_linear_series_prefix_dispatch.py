"""Public linear operations accept the same complete source prefixes."""

import json
from fractions import Fraction

import pytest

from jacobian._exact import CanonicalRational as Q
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import add, derivative, subtract
from jacobian.math.polynomials.series._models import (
    MAX_RATIONAL_DIGITS,
    MAX_TRUNCATE_SOURCE_ORDER,
    SeriesArithmeticResult,
    SeriesDerivativeResult,
    TruncatedSeries,
)


@pytest.mark.parametrize("name", ("add", "subtract", "derivative"))
def test_public_linear_joint_maximum_round_trips_and_hands_off(name: str) -> None:
    n = MAX_TRUNCATE_SOURCE_ORDER
    p = 10**MAX_RATIONAL_DIGITS - 1
    a, b = Fraction(p, p - 2), Fraction(p - 4, p - 6)
    left = TruncatedSeries(
        variable="x", truncation_order=n, coefficients=(Q.from_fraction(a),) * n
    )
    right = TruncatedSeries(
        variable="x", truncation_order=n, coefficients=(Q.from_fraction(b),) * n
    )
    catalog = Catalog.open()
    native: SeriesArithmeticResult | SeriesDerivativeResult
    if name == "derivative":
        payload = left.model_dump(mode="json")
        native = derivative(left)
        expected = [k * a for k in range(1, n)]
    else:
        payload = {
            "left": left.model_dump(mode="json"),
            "right": right.model_dump(mode="json"),
        }
        native = add(left, right) if name == "add" else subtract(left, right)
        expected = [a + b if name == "add" else a - b] * n
    public = invoke_operation(
        f"formal_series.rational.{name}.compute", payload, catalog
    )
    decoded = (
        SeriesDerivativeResult if name == "derivative" else SeriesArithmeticResult
    ).model_validate_json(json.dumps(public.output))
    assert decoded == native
    assert [value.as_fraction() for value in decoded.result.coefficients] == expected
    prefix = invoke_operation(
        "formal_series.rational.truncate.compute",
        {"series": decoded.result.model_dump(mode="json"), "target_order": 512},
        catalog,
    )
    assert (
        TruncatedSeries.model_validate_json(
            json.dumps(prefix.output["result"])
        ).coefficients
        == decoded.result.coefficients[:512]
    )


def test_public_derivative_reuses_actual_inverse_output_and_adds_zero() -> None:
    n = 2048
    zero = {"num": "0", "den": "1"}
    source = {
        "variable": "x",
        "truncation_order": n,
        "coefficients": [
            {"num": "1", "den": "1"},
            {"num": "-1", "den": "1"},
            *([zero] * (n - 2)),
        ],
    }
    catalog = Catalog.open()
    produced = invoke_operation(
        "formal_series.rational.inverse.compute", source, catalog
    )
    result = invoke_operation(
        "formal_series.rational.derivative.compute", produced.output["result"], catalog
    )
    decoded = SeriesDerivativeResult.model_validate_json(json.dumps(result.output))
    assert [q.as_fraction() for q in decoded.result.coefficients] == list(range(1, n))
    added = invoke_operation(
        "formal_series.rational.add.compute",
        {
            "left": result.output["result"],
            "right": {
                "variable": "x",
                "truncation_order": n - 1,
                "coefficients": [zero] * (n - 1),
            },
        },
        catalog,
    )
    assert added.output["result"] == result.output["result"]
