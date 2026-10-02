"""Discovery and precise errors guide complete set-table recovery over MCP."""

import asyncio
import json

from mcp.types import TextContent

from jacobian.mcp.server import create_server
from mcp import Client


def test_live_set_function_duplicate_diagnostic_and_recovery() -> None:
    async def scenario() -> None:
        operation_id = "combinatorics.set_function.evaluate"
        async with Client(create_server(), raise_exceptions=False) as client:
            found = await client.call_tool("math.find", {"operation_id": operation_id})
            assert not found.is_error
            assert found.structured_content is not None
            definitions = found.structured_content["operation"]["input_schema"]["$defs"]
            entries_schema = definitions["SetFunction"]["properties"]["entries"]
            assert "exactly once" in entries_schema["description"]
            entries = json.loads(json.dumps(entries_schema["examples"][0]))
            entries[1]["subset"] = []
            payload = {
                "function": {"ground_set_size": 1, "entries": entries},
                "subset": [0],
            }
            refused = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload}
            )
            assert refused.is_error
            assert isinstance(refused.content[0], TextContent)
            text = refused.content[0].text
            diagnostic = json.loads(text[text.index("{") :])
            assert diagnostic["code"] == "INVALID_REQUEST"
            error = diagnostic["errors"][0]
            assert error["code"] == "submodular_opt.table_subsets_not_unique"
            assert error["location"] == ["function", "entries"]
            assert "entries[1]" in error["message"]
            assert "entries[0]" in error["message"]
            assert "missing subset [0]" in error["message"]
            entries[1]["subset"] = [0]
            accepted = await client.call_tool(
                "math.run", {"operation_id": operation_id, "payload": payload}
            )
            assert not accepted.is_error
            assert accepted.structured_content is not None
            assert accepted.structured_content["output"]["value"] == {
                "num": "1",
                "den": "1",
            }

    asyncio.run(scenario())
