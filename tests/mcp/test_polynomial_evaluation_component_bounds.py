"""Large exact identity evaluations and true overflow through the MCP boundary."""

import asyncio
import json

from mcp.types import TextContent

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.mcp.server import create_server
from mcp import Client


def test_identity_component_bounds_and_square_overflow_through_mcp() -> None:
    async def scenario() -> None:
        magnitude = 10 ** (MAX_CANONICAL_RATIONAL_DIGITS - 1)
        identity_point = CanonicalRational(num=magnitude, den=magnitude + 1)
        overflow_point = CanonicalRational(
            num=1, den=10 ** (MAX_CANONICAL_RATIONAL_DIGITS // 2)
        )
        async with Client(create_server(), raise_exceptions=False) as client:
            for operation_id in (
                "polynomial.rational.compute.evaluate",
                "polynomial.map.evaluate",
            ):
                for exponent, point in ((1, identity_point), (2, overflow_point)):
                    wire_point = point.model_dump(mode="json")
                    payload = {
                        "polynomial": {
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [exponent],
                                    }
                                ]
                            },
                        },
                        "point": wire_point
                        if operation_id == "polynomial.rational.compute.evaluate"
                        else {"variables": ["x"], "values": [wire_point]},
                    }
                    result = await client.call_tool(
                        "math.run", {"operation_id": operation_id, "payload": payload}
                    )
                    if exponent == 1:
                        assert not result.is_error
                        assert result.structured_content is not None
                        assert (
                            result.structured_content["output"]["value"] == wire_point
                        )
                    else:
                        assert result.is_error
                        content = result.content[0]
                        assert isinstance(content, TextContent)
                        failure = json.loads(
                            content.text.removeprefix("Error executing tool math.run: ")
                        )
                        assert failure["code"] == "INVALID_REQUEST"

    asyncio.run(scenario())
