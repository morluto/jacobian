"""Admission tests for exact real-quadratic scalar operations."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticOrderValue,
    RealQuadraticValue,
    real_quadratic_order,
)
from jacobian.math.number_theory.arithmetic._real_quadratic import (
    RealQuadraticOrderRequest,
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


def test_native_order_api_accepts_canonical_values_without_a_wire_request() -> None:
    result = real_quadratic_order(_value(3), _value(1))

    assert result.order == "GT"
    assert result.difference.rational_part.as_fraction() == 2


def test_native_order_api_retains_shared_field_admission() -> None:
    with pytest.raises(ValueError, match="comparison requires one shared radicand"):
        real_quadratic_order(
            _value(1),
            RealQuadraticValue(
                rational_part=CanonicalRational(num="0", den="1"),
                radical_coefficient=CanonicalRational(num="0", den="1"),
                radicand=3,
            ),
        )


def test_order_declaration_projects_its_wire_request_to_the_native_kernel() -> None:
    from jacobian.math.number_theory.arithmetic._real_quadratic import (
        REAL_QUADRATIC_OPERATIONS,
    )

    tools = {tool.operation_id: tool for tool in REAL_QUADRATIC_OPERATIONS}
    left, right = _value(3), _value(1)
    order_tool = cast(
        MathTool[RealQuadraticOrderRequest, RealQuadraticOrderValue],
        tools["arithmetic.real_quadratic.order.compute"],
    )
    result = order_tool.run(RealQuadraticOrderRequest(left=left, right=right))

    assert result == real_quadratic_order(left, right)
