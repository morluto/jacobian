"""Direct-dispatch parity for the restored composition admission boundary."""

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.series import compose
from jacobian.math.polynomials.series._models import (
    SeriesComposeResult,
    TruncatedSeries,
)


def test_native_and_public_composition_agree_on_restored_request() -> None:
    def series(values: list[Fraction]) -> TruncatedSeries:
        return TruncatedSeries(
            variable="q",
            truncation_order=32,
            coefficients=tuple(
                CanonicalRational.from_fraction(v)
                for v in [*values, *([Fraction()] * (32 - len(values)))]
            ),
        )

    outer = series([Fraction(3), Fraction(2), Fraction(-1)])
    inner = series([Fraction(), Fraction(10**200, 3), Fraction(10**200, 7)])
    native = compose(outer, inner)
    result = invoke_operation(
        "formal_series.rational.compose.compute",
        {
            "outer": outer.model_dump(mode="json"),
            "inner": inner.model_dump(mode="json"),
        },
        Catalog.open(),
    )
    assert result.output == native.model_dump(mode="json")
    assert SeriesComposeResult.model_validate_json(json.dumps(result.output)) == native
