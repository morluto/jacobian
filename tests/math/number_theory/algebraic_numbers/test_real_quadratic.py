"""Admission tests for exact real-quadratic scalar operations."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import pytest

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
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
    with pytest.raises(OperationResourceAdmissionError, match="difference") as error:
        real_quadratic_order(request.left, request.right)
    assert error.value.errors()[0]["type"] == "real_quadratic.difference_bound_exceeded"
    assert error.value.errors()[0]["loc"] == ()


def test_native_order_api_accepts_canonical_values_without_a_wire_request() -> None:
    result = real_quadratic_order(_value(3), _value(1))

    assert result.order == "GT"
    assert result.difference.rational_part.as_fraction() == 2


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        (
            "difference",
            {
                "rational_part": {"num": "1", "den": "1"},
                "radical_coefficient": {"num": "0", "den": "1"},
                "radicand": 2,
            },
        ),
        ("order", "LT"),
        ("sign_basis", "RADICAL_ONLY"),
        (
            "sign_certificate",
            {
                "rational_part_squared": {"num": "1", "den": "1"},
                "radical_part_squared": {"num": "0", "den": "1"},
                "magnitude_order": "GT",
            },
        ),
    ),
)
def test_order_result_parsing_retains_structural_source_context_only(
    field: str, forged_value: object
) -> None:
    result = real_quadratic_order(_value(3), _value(1))
    payload = result.model_dump(mode="json")

    parsed = RealQuadraticOrderValue.model_validate({**payload, field: forged_value})
    assert parsed.left.radicand == parsed.right.radicand == parsed.difference.radicand


def test_native_order_api_retains_shared_field_admission() -> None:
    with pytest.raises(
        OperationDomainValidationError, match="comparison requires one shared radicand"
    ) as error:
        real_quadratic_order(
            _value(1),
            RealQuadraticValue(
                rational_part=CanonicalRational(num="0", den="1"),
                radical_coefficient=CanonicalRational(num="0", den="1"),
                radicand=3,
            ),
        )
    assert error.value.errors()[0]["type"] == "real_quadratic.radicand_mismatch"
    assert error.value.errors()[0]["loc"] == ("right", "radicand")


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
