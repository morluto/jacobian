"""The rational degree-eight circle envelope through its public operation."""

from __future__ import annotations

import json
import random
from fractions import Fraction

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.unit_circle._sup_norm_models import (
    UnitCircleSupNormSquaredResult,
)


def test_independent_32_digit_denominators_survive_dispatch_and_json() -> None:
    catalog = Catalog.open()
    operation_id = "polynomial.unit_circle.sup_norm_squared.compute"
    descriptor = catalog.inspect(operation_id)
    assert descriptor is not None
    description = descriptor.input_schema["properties"]["polynomial"]["description"]
    assert "32 decimal digits" in description
    assert "denominator-aware" in description
    rng = random.Random(2768)
    terms = []
    real_total = imaginary_total = Fraction(0)
    for exponent in range(8, -1, -1):
        real = Fraction(rng.randrange(10**31, 10**32), rng.randrange(10**31, 10**32))
        imaginary = Fraction(
            rng.randrange(10**31, 10**32), rng.randrange(10**31, 10**32)
        )
        real_total += real
        imaginary_total += imaginary
        terms.append(
            {
                "exponent": exponent,
                "coefficient": {
                    "real": {"num": str(real.numerator), "den": str(real.denominator)},
                    "imaginary": {
                        "num": str(imaginary.numerator),
                        "den": str(imaginary.denominator),
                    },
                },
            }
        )
    response = invoke_operation(operation_id, {"polynomial": {"terms": terms}}, catalog)
    result = UnitCircleSupNormSquaredResult.model_validate_json(
        json.dumps(response.output)
    )
    assert len(result.critical_points) == 14
    assert (
        result.transformed_numerator[0].as_fraction()
        == real_total**2 + imaginary_total**2
    )
    assert (
        UnitCircleSupNormSquaredResult.model_validate_json(result.model_dump_json())
        == result
    )
