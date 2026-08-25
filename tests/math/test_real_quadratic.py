"""Admission tests for exact real-quadratic scalar operations."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.real_quadratic import (
    RealQuadraticOrderRequest,
    RealQuadraticValue,
)


def _value(rational: int) -> RealQuadraticValue:
    return RealQuadraticValue(
        rational_part=CanonicalRational.from_fraction(Fraction(rational)),
        radical_coefficient=CanonicalRational(num="0", den="1"),
        radicand=2,
    )


def test_order_preflights_a_difference_that_cannot_be_returned() -> None:
    maximum_component = 10**256 - 1

    with pytest.raises(ValidationError) as error:
        RealQuadraticOrderRequest(
            left=_value(maximum_component),
            right=_value(-maximum_component),
        )
    assert error.value.errors()[0]["type"] == "real_quadratic.difference_bound_exceeded"
