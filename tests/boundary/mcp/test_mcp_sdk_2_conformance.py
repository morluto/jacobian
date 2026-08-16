"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect

from mcp.types.methods import serialize_server_result

import jacobian.adapters.mcp.server as server_module
from jacobian.adapters.mcp.server import create_server
from jacobian.adapters.mcp.tools import math_run
from jacobian.contracts.operations import OperationCatalogSnapshot, OperationResult


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.0.0"
    assert not inspect.iscoroutinefunction(math_run)


def test_mcp_v2_uses_sdk_typed_tools_lifespan_and_structured_resources(
    monkeypatch,
) -> None:
    monkeypatch.delattr(server_module, "Context", raising=False)

    async def scenario() -> None:
        from mcp import Client

        server = create_server()
        assert not hasattr(server_module, "Context")
        assert hasattr(server, "list_tools") and hasattr(server, "call_tool")
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == {"math.find", "math.run"}

            invoke = next(tool for tool in listed.tools if tool.name == "math.run")
            assert set(invoke.input_schema["properties"]) == {
                "operation_id",
                "payload",
            }
            assert invoke.output_schema == OperationResult.model_json_schema()
            assert invoke.annotations is not None
            assert invoke.annotations.read_only_hint is True
            assert invoke.annotations.idempotent_hint is False

            find = next(tool for tool in listed.tools if tool.name == "math.find")
            assert find.annotations is not None
            assert find.annotations.read_only_hint is True
            assert find.annotations.idempotent_hint is True
            assert set(find.input_schema["properties"]) == {"request"}
            assert find.input_schema["properties"]["request"]["discriminator"] == {
                "mapping": {
                    "browse": "#/$defs/OperationBrowseRequest",
                    "inspect": "#/$defs/OperationInspectRequest",
                    "search": "#/$defs/OperationSearchRequest",
                },
                "propertyName": "op",
            }
            assert find.output_schema["type"] == "object"

            browse = await client.call_tool(
                "math.find",
                {"request": {"op": "browse", "domain": "matrix", "limit": 1}},
            )
            assert isinstance(browse.structured_content, dict)
            assert browse.structured_content["kind"] == "browse"

            serialized_tools = serialize_server_result(
                "tools/list",
                "2026-07-28",
                listed.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
            assert serialized_tools["tools"][0]["outputSchema"]["type"] == "object"

            invalid_request = await client.call_tool(
                "math.find", {"unknown_key": "rejected"}
            )
            assert invalid_request.is_error is True
            assert invalid_request.content, "error responses must carry diagnostic text"
            assert any(
                "request" in item.text
                for item in invalid_request.content
                if getattr(item, "text", None) is not None
            ), "error text must identify the missing request field"

            contract_result = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "matrix.determinant.compute",
                    }
                },
            )
            assert isinstance(contract_result.structured_content, dict)
            contract = contract_result.structured_content
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "matrix.determinant.compute",
                    "payload": contract["operation"]["examples"][0]["input"],
                },
            )
            assert isinstance(result.structured_content, dict)
            assert result.structured_content == OperationResult.model_validate(
                result.structured_content
            ).model_dump(mode="json")
            assert "output" in result.structured_content
            assert "determinant" in result.structured_content["output"]

            catalog = await client.read_resource("operation://catalog")
            snapshot = OperationCatalogSnapshot.model_validate_json(
                catalog.contents[0].text
            )
            assert snapshot.operations

    asyncio.run(scenario())
