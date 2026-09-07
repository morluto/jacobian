"""Real owner diagnostics survive native, dispatch, and live MCP calls."""

import asyncio
import json

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.number_theory.algebraic_numbers.quadratic import real_quadratic_order
from jacobian.math.number_theory.arithmetic._real_quadratic import (
    RealQuadraticOrderRequest,
)
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize(
    "overflow", (False, True), ids=("shared-field", "difference-bound")
)
def test_real_quadratic_rejection_and_recovery(overflow: bool) -> None:
    operation_id = "arithmetic.real_quadratic.order.compute"
    payload = {
        "left": {
            "rational_part": {"num": "0", "den": "1"},
            "radical_coefficient": {"num": "3", "den": "8"},
            "radicand": 2,
        },
        "right": {
            "rational_part": {"num": "1", "den": "2"},
            "radical_coefficient": {"num": "1", "den": "20"},
            "radicand": 3,
        },
    }
    valid = RealQuadraticOrderRequest.model_validate_json(
        json.dumps({**payload, "left": {**payload["left"], "radicand": 3}})
    )
    if overflow:
        payload = valid.model_dump(mode="json")
        payload["left"]["rational_part"] = {"num": "9" * 256, "den": "1"}
        payload["right"]["rational_part"] = {"num": "-" + "9" * 256, "den": "1"}
    request = RealQuadraticOrderRequest.model_validate_json(json.dumps(payload))
    with pytest.raises(OperationDomainValidationError) as native:
        real_quadratic_order(request.left, request.right)
    with pytest.raises(OperationDomainValidationError) as dispatch:
        invoke_operation(operation_id, payload, Catalog.open())
    assert dispatch.value.errors() == native.value.errors()

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=True) as client:
            with pytest.raises(MCPError) as rejected:
                await client.call_tool(
                    "math.run", {"operation_id": operation_id, "payload": payload}
                )
            assert rejected.value.code == INVALID_PARAMS
            assert rejected.value.data["code"] == (
                "RESOURCE_ADMISSION_REJECTED" if overflow else "INVALID_REQUEST"
            )
            assert rejected.value.data["stage"] == (
                "resource_admission" if overflow else "operation_validation"
            )
            assert rejected.value.data["errors"] == [
                {
                    "location": list(error["loc"]),
                    "code": error["type"],
                    "message": error["msg"],
                }
                for error in native.value.errors()
            ]
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": operation_id,
                    "payload": valid.model_dump(mode="json"),
                },
            )
            output = result.structured_content["output"]
            assert output["order"] == "GT"
            assert output["difference"]["rational_part"] == {"num": "-1", "den": "2"}
            assert output["difference"]["radical_coefficient"] == {
                "num": "13",
                "den": "40",
            }
            assert output["sign_certificate"]["radical_part_squared"] == {
                "num": "507",
                "den": "1600",
            }

    asyncio.run(scenario())
