"""Live MCP inspection, normalized exact results, refusal, and recovery."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def test_mcp_rational_ingress_schema_composition_and_recovery() -> None:
    async def scenario() -> None:
        async with Client(create_server(), raise_exceptions=False) as client:
            inspected = await client.call_tool(
                "math.find", {"operation_id": "matrix.determinant.compute"}
            )
            assert not inspected.is_error
            assert inspected.structured_content is not None
            schema = inspected.structured_content["operation"]["input_schema"]
            matrix = schema["properties"]["matrix"]
            assert matrix["properties"]["column_count"]["maximum"] == 128
            denominator = matrix["properties"]["entries"]["items"]["items"][
                "properties"
            ]["den"]
            assert denominator["type"] == "string"
            assert denominator["pattern"].startswith("^-?")
            for numerator, denominator, expected in [
                ("2", "-4", {"num": "-1", "den": "2"}),
                ("0", "-5", {"num": "0", "den": "1"}),
            ]:
                result = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "matrix.determinant.compute",
                        "payload": {
                            "matrix": {
                                "entries": [[{"num": numerator, "den": denominator}]]
                            }
                        },
                    },
                )
                assert not result.is_error
                assert result.structured_content is not None
                assert result.structured_content["output"]["determinant"] == expected
            for numerator, denominator in [
                ("1", "0"),
                ("2" + "0" * 256, "4" + "0" * 256),
            ]:
                refused = await client.call_tool(
                    "math.run",
                    {
                        "operation_id": "matrix.determinant.compute",
                        "payload": {
                            "matrix": {
                                "entries": [[{"num": numerator, "den": denominator}]]
                            }
                        },
                    },
                )
                assert refused.is_error
                assert isinstance(refused.content[0], TextContent)
                text = refused.content[0].text
                diagnostic = json.loads(text[text.index("{") :])
                assert diagnostic["code"] == "INVALID_REQUEST"
            recovered = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.rational_linear_system.solve",
                    "payload": {
                        "matrix": {"entries": [[{"num": "2", "den": "4"}]]},
                        "rhs": [{"num": "-6", "den": "-8"}],
                    },
                },
            )
            assert not recovered.is_error
            assert recovered.structured_content is not None
            assert recovered.structured_content["output"]["solution"] == [
                {"num": "3", "den": "2"}
            ]

    asyncio.run(scenario())
