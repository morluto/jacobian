"""Polynomial remainder-only requests compose without a discarded primitive."""

import asyncio
import json

import pytest
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.mcp.server import create_server
from mcp import Client


@pytest.mark.parametrize("kind", ("partial_fractions", "logarithmic_differential"))
@pytest.mark.parametrize("degree", (1, 63))
def test_polynomial_remainder_public_roundtrip_and_primitive_refusal(
    kind: str, degree: int
) -> None:
    source = {
        "domain": "QQ",
        "variables": ["t"],
        "numerator": {
            "terms": [
                {"coefficient": {"num": "-1", "den": "9" * 128}, "exponents": [degree]}
            ]
        },
        "denominator": {
            "terms": [{"coefficient": {"num": "1", "den": "1"}, "exponents": [0]}]
        },
    }
    operation = f"rational_function.{kind}.compute"
    catalog = Catalog.open()
    output = invoke_operation(operation, {"function": source}, catalog).output
    declaration = catalog.operation(operation)
    assert declaration is not None
    assert (
        declaration.result_type.model_validate_json(json.dumps(output)).model_dump(
            mode="json"
        )
        == output
    )
    assert output["terms"] == []
    if kind == "partial_fractions":
        assert output["polynomial_part"]["polynomial"] == source["numerator"]
        assert output["reconstructed"] == source
        assert output["factors"] == []
        assert output["hermite_agreement"] is True
        remainder = output["hermite_remainder"]
    else:
        assert output["source"] == source
        remainder = output["reconstructed"]
    assert remainder["variables"] == ["t"]
    assert remainder["numerator"]["terms"] == []

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.hermite_reduction.compute",
                    "payload": {"function": source},
                },
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert diagnostic["stage"] == "resource_admission"
            assert (
                diagnostic["errors"][0]["code"] == "polynomial.hermite_reduction_budget"
            )
            result = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": {"function": source}}
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == output

    asyncio.run(scenario())
