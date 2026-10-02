"""Actual public reduction chains keep the original request's admitted source."""

import asyncio
import json

import pytest
from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.polynomials.values import RationalFunction
from jacobian.mcp.server import create_server
from mcp import Client


def _function(
    num: list[tuple[str, str, int]], den: list[tuple[str, str, int]]
) -> dict[str, object]:
    def polynomial(terms: list[tuple[str, str, int]]) -> dict[str, object]:
        return {
            "terms": [
                {"coefficient": {"num": n, "den": d}, "exponents": [e]}
                for n, d, e in terms
            ]
        }

    return {
        "domain": "QQ",
        "variables": ["x"],
        "numerator": polynomial(num),
        "denominator": polynomial(den),
    }


@pytest.mark.parametrize("kind", ("logarithmic_differential", "formal_antiderivative"))
def test_public_derived_remainder_exactness_and_unchanged_source_limits(
    kind: str,
) -> None:
    source = _function(
        [("1", "99", 1)],
        [("1", "1", 3), ("-17", "1", 2), ("63", "1", 1), ("81", "1", 0)],
    )
    expected_h = _function(
        [("1", "990", 0)], [("1", "1", 2), ("-8", "1", 1), ("-9", "1", 0)]
    )
    expected_r = _function([("-1", "110", 0)], [("1", "1", 1), ("-9", "1", 0)])
    catalog = Catalog.open()
    hermite = invoke_operation(
        "rational_function.hermite_reduction.compute", {"function": source}, catalog
    ).output
    assert hermite["remainder"] == expected_h
    assert hermite["rational_part"] == expected_r
    assert (
        RationalFunction.model_validate_json(
            json.dumps(hermite["remainder"])
        ).model_dump(mode="json")
        == expected_h
    )
    operation = f"rational_function.{kind}.compute"
    output = invoke_operation(operation, {"function": source}, catalog).output
    logarithmic = (
        output if kind == "logarithmic_differential" else output["logarithmic_part"]
    )
    assert logarithmic["source"] == logarithmic["reconstructed"] == expected_h
    assert len(logarithmic["terms"]) == 2
    if kind == "formal_antiderivative":
        assert output["source"] == source
        assert output["rational_part"] == expected_r
    declaration = catalog.operation(operation)
    assert declaration is not None
    assert (
        declaration.result_type.model_validate_json(json.dumps(output)).model_dump(
            mode="json"
        )
        == output
    )

    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            # A fresh public request still owns its own two-digit source budget;
            # this fix reuses admitted facts only inside the original request.
            refused = await client.call_tool(
                "math.run",
                {
                    "operation_id": "rational_function.partial_fractions.compute",
                    "payload": {"function": hermite["remainder"]},
                },
            )
            assert refused.is_error
            content = refused.content[0]
            assert isinstance(content, TextContent)
            diagnostic = json.loads(
                content.text.removeprefix("Error executing tool math.run: ")
            )
            assert diagnostic["code"] == "RESOURCE_ADMISSION_REJECTED"
            assert (
                diagnostic["errors"][0]["code"] == "polynomial.partial_fraction_budget"
            )
            result = await client.call_tool(
                "math.run", {"operation_id": operation, "payload": {"function": source}}
            )
            assert not result.is_error
            assert result.structured_content is not None
            assert result.structured_content["output"] == output

    asyncio.run(scenario())
