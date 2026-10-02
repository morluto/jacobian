"""Lie derivative distinguishes mathematical invalidity from non-completion."""

import asyncio
import json
from typing import Any

import pytest
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.dispatch import invoke_operation
from jacobian.math.geometry.differential._models import RationalLieDerivativeRequest
from jacobian.math.geometry.differential.operations import lie_derivative
from jacobian.mcp.server import create_server
from mcp import Client


def _tensor(power: int, *, axis: str = "x", scalar: bool = False) -> dict[str, Any]:
    def monomial(exponent: int) -> dict[str, Any]:
        return {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [exponent]}
            ]
        }

    return {
        "coordinate_axis": [axis],
        "variance": [] if scalar else ["CONTRAVARIANT"],
        "components": [
            {
                "variables": [axis],
                "numerator": monomial(0),
                "denominator": monomial(power),
            }
        ],
        "retained_nonzero_denominators": [monomial(power)],
    }


@pytest.mark.parametrize(
    "resource", (False, True), ids=("axis-mismatch", "output-bound")
)
def test_lie_derivative_native_dispatch_mcp_errors_and_recovery(resource: bool) -> None:
    operation_id = "differential_geometry.rational_tensor.lie_derivative.compute"
    # L_(x^-64 d/dx) x^-64 = -64 x^-129: a genuine output-bound
    # refusal, independent of cancellation-aware self-bracket admission.
    payload = {
        "vector_field": _tensor(64),
        "tensor": _tensor(64, axis="x" if resource else "y", scalar=True),
    }
    request = RationalLieDerivativeRequest.model_validate_json(json.dumps(payload))
    expected_error = (
        OperationResourceAdmissionError if resource else OperationDomainValidationError
    )
    with pytest.raises(expected_error) as native:
        lie_derivative(request.vector_field, request.tensor)
    assert type(native.value) is expected_error
    with pytest.raises(expected_error) as dispatch:
        invoke_operation(operation_id, payload, Catalog.open())
    assert dispatch.value.errors() == native.value.errors()

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            rejected = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload}
            )
            assert rejected.is_error
            content = rejected.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == (
                "RESOURCE_ADMISSION_REJECTED" if resource else "INVALID_REQUEST"
            )
            assert diagnostic["stage"] == (
                "resource_admission" if resource else "operation_validation"
            )
            assert diagnostic["errors"] == [
                {
                    "location": list(error["loc"]),
                    "code": error["type"],
                    "message": error["msg"],
                }
                for error in native.value.errors()
            ]
            accepted = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": {
                        "vector_field": _tensor(31),
                        "tensor": _tensor(32, scalar=True),
                    },
                },
            )
            assert not accepted.is_error
            assert accepted.structured_content is not None
            # Independently, x^-31 * d(x^-32)/dx = -32 x^-64.
            component = accepted.structured_content["output"]["lie_derivative"][
                "components"
            ][0]
            assert component["numerator"]["terms"] == [
                {"coefficient": {"num": "-32", "den": "1"}, "exponents": [0]}
            ]
            assert component["denominator"]["terms"] == [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [64]}
            ]

    asyncio.run(scenario())
