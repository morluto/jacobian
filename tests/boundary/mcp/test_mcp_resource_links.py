from __future__ import annotations

import asyncio
import json
from pathlib import Path

from jacobian.adapters.mcp.server import create_server

MATH_TOOL_NAMES = {"math.find", "math.run"}
MCP_TOOL_NAMES = MATH_TOOL_NAMES


def test_mcp_inline_results_do_not_emit_resource_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "integer.compute.gcd",
                    "payload": {"left": "84", "right": "30"},
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content["artifact_uris"] == []
            assert [
                block for block in result.content if block.type == "resource_link"
            ] == []

    asyncio.run(scenario())


def test_mcp_materialized_results_emit_readable_native_resource_links(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(tmp_path), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "capability_id": "sat.cnf.materialize",
                    "payload": {
                        "variable_names": ["x"],
                        "clauses": [[1]],
                    },
                },
            )
            assert isinstance(result.structured_content, dict)
            artifact_uris = result.structured_content["artifact_uris"]
            links = [block for block in result.content if block.type == "resource_link"]
            assert [str(link.uri) for link in links] == artifact_uris
            assert [link.name for link in links] == artifact_uris
            assert all(link.mime_type == "application/json" for link in links)

            resource = await client.read_resource(links[0].uri)
            envelope = json.loads(resource.contents[0].text)
            assert envelope["artifact_uri"] == artifact_uris[0]
            assert envelope["payload"]["clauses"] == [{"literals": [1]}]

    asyncio.run(scenario())
