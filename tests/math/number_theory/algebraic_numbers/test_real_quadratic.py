"""Admission tests for exact real-quadratic scalar operations."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import MathTool
from jacobian.math.number_theory.algebraic_numbers.quadratic import (
    RealQuadraticEmbeddingProfile,
    RealQuadraticOrderValue,
    RealQuadraticValue,
    real_quadratic_embeddings,
    real_quadratic_order,
)
from jacobian.math.number_theory.arithmetic._real_quadratic import (
    RealQuadraticEmbeddingRequest,
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

    request = RealQuadraticOrderRequest(
        left=_value(maximum_component),
        right=_value(-maximum_component),
    )
    with pytest.raises(ValueError, match="difference"):
        real_quadratic_order(request.left, request.right)


def test_native_order_api_accepts_canonical_values_without_a_wire_request() -> None:
    result = real_quadratic_order(_value(3), _value(1))

    assert result.order == "GT"
    assert result.difference.rational_part.as_fraction() == 2


@pytest.mark.parametrize(
    ("field", "forged_value", "error_type"),
    (
        (
            "difference",
            {
                "rational_part": {"num": "1", "den": "1"},
                "radical_coefficient": {"num": "0", "den": "1"},
                "radicand": 2,
            },
            "real_quadratic.difference_mismatch",
        ),
        ("order", "LT", "real_quadratic.order_mismatch"),
        ("sign_basis", "RADICAL_ONLY", "real_quadratic.order_mismatch"),
        (
            "sign_certificate",
            {
                "rational_part_squared": {"num": "1", "den": "1"},
                "radical_part_squared": {"num": "0", "den": "1"},
                "magnitude_order": "GT",
            },
            "real_quadratic.sign_certificate_mismatch",
        ),
    ),
)
def test_order_result_rejects_forged_source_bound_fields(
    field: str, forged_value: object, error_type: str
) -> None:
    result = real_quadratic_order(_value(3), _value(1))
    payload = result.model_dump(mode="json")

    with pytest.raises(ValidationError) as exc_info:
        RealQuadraticOrderValue.model_validate({**payload, field: forged_value})

    assert exc_info.value.errors()[0]["type"] == error_type


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


def test_declarations_project_wire_requests_to_native_kernels() -> None:
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

    element = RealQuadraticValue(
        rational_part=CanonicalRational(num="1", den="1"),
        radical_coefficient=CanonicalRational(num="1", den="1"),
        radicand=2,
    )
    embedding_tool = cast(
        MathTool[RealQuadraticEmbeddingRequest, RealQuadraticEmbeddingProfile],
        tools["arithmetic.real_quadratic.embeddings.compute"],
    )
    embedding_result = embedding_tool.run(
        RealQuadraticEmbeddingRequest(element=element)
    )
    assert embedding_result == real_quadratic_embeddings(element)
