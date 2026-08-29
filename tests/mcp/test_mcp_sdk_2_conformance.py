"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
from typing import Any

import pytest
from mcp.types import ContentBlock, TextContent, TextResourceContents
from mcp.types.methods import serialize_server_result
from pydantic import model_serializer

import jacobian.mcp.server as server_module
from jacobian._execution import current_request_execution
from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationCatalogSnapshot
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server, create_server
from jacobian.mcp.tools import run_direct_math_tool


def _content_text(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text


def test_mcp_sdk_is_exactly_pinned_and_v2_bindings_are_used() -> None:
    assert importlib.metadata.version("mcp") == "2.1.0"
    assert not inspect.iscoroutinefunction(run_direct_math_tool)


def test_direct_tool_does_not_resolve_the_catalog_at_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operation declaration is bound while compiling the immutable surface."""

    catalog = Catalog.open()
    server = _build_server(state=AppState(operation_catalog=catalog))

    def unexpected_public_lookup(operation_id: str) -> Any:
        raise AssertionError(f"unexpected public lookup: {operation_id}")

    monkeypatch.setattr(catalog, "operation", unexpected_public_lookup)

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "integer.compute.extended_gcd",
                {"left": "84", "right": "30"},
            )
        assert result.structured_content is not None
        assert result.structured_content["gcd"] == "6"

    asyncio.run(scenario())


def test_direct_tool_serializes_once_inside_the_request_envelope() -> None:
    observed_envelopes: list[bool] = []

    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

        @model_serializer(mode="plain")
        def serialize(self) -> dict[str, int]:
            observed_envelopes.append(current_request_execution() is not None)
            return {"value": self.value}

    operation = MathTool(
        operation_id="test.mcp.serialization_envelope",
        title="Serialization envelope sentinel",
        description="Exercises direct result serialization.",
        request_type=Request,
        result_type=Result,
        run=lambda request: Result(value=request.value),
    )
    server = _build_server(state=AppState(operation_catalog=Catalog((operation,))))

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(operation.operation_id, {"value": 7})
        assert result.structured_content == {"value": 7}

    asyncio.run(scenario())
    assert observed_envelopes == [True]


def test_direct_tool_encloses_logarithm_on_a_positive_box() -> None:
    """A valid Arb enclosure must cross the MCP worker boundary as a result."""

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "interval.expression.box_enclosure.compute",
                {
                    "expression": {
                        "op": "log",
                        "children": [{"op": "var", "variable": "x"}],
                    },
                    "box": {
                        "variables": ["x"],
                        "intervals": [
                            {
                                "lower": {"num": "1", "den": "1"},
                                "upper": {"num": "2", "den": "1"},
                            }
                        ],
                    },
                    "precision_bits": 1024,
                },
            )
            assert isinstance(result.structured_content, dict)
            output = result.structured_content
            assert output["status"] == "ENCLOSED"
            assert output["lower"] is not None and output["upper"] is not None

    asyncio.run(scenario())


def test_direct_tool_encloses_logarithm_second_jet_on_a_positive_box() -> None:
    """The parallel Arb enclosure must also cross the MCP boundary as a result."""

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "interval.expression.second_jet_enclosure.compute",
                {
                    "expression": {
                        "op": "log",
                        "children": [{"op": "var", "variable": "x"}],
                    },
                    "box": {
                        "variables": ["x"],
                        "intervals": [
                            {
                                "lower": {"num": "1", "den": "1"},
                                "upper": {"num": "2", "den": "1"},
                            }
                        ],
                    },
                    "precision_bits": 1024,
                },
            )
            assert isinstance(result.structured_content, dict)
            output = result.structured_content
            assert output["status"] == "ENCLOSED"
            assert output["value"] is not None
            assert len(output["gradient"]) == 1
            assert len(output["hessian"]) == 1

    asyncio.run(scenario())


def test_direct_tool_projects_unexpected_operation_failures() -> None:
    """An owner crash must not escape the MCP worker as a TaskGroup failure."""

    from jacobian._models import StrictModel
    from jacobian.catalog.builtins import BUILTIN_TOOLS

    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    def crashing_kernel(_request: Request) -> Result:
        raise RuntimeError("private backend failure")

    operation = MathTool(
        operation_id="test.mcp.crashing_kernel",
        title="Crashing kernel sentinel",
        description="Exercises MCP execution-failure projection.",
        request_type=Request,
        result_type=Result,
        run=crashing_kernel,
    )
    server = _build_server(
        state=AppState(operation_catalog=Catalog((*BUILTIN_TOOLS, operation)))
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(
                "test.mcp.crashing_kernel",
                {"value": 1},
            )
        assert result.is_error is True
        assert result.structured_content is None
        text = _content_text(result.content[0]) if result.content else ""
        assert text == (
            "Error executing tool test.mcp.crashing_kernel: operation execution failed"
        )
        assert "private backend failure" not in text

    asyncio.run(scenario())


def test_production_surface_omits_generic_discovery_and_execution_tools() -> None:
    catalog = Catalog.open()
    server = create_server()

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == {
            item.operation_id for item in catalog.snapshot().operations
        }
        assert "math.find" not in {tool.name for tool in listed.tools}
        assert "math.run" not in {tool.name for tool in listed.tools}

    asyncio.run(scenario())


def test_mcp_v2_uses_sdk_typed_tools_lifespan_and_structured_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(server_module, "Context", raising=False)
    expected_catalog = Catalog.open().snapshot()

    async def scenario() -> None:
        from mcp import Client

        server = _build_server(
            state=AppState(operation_catalog=Catalog.open()),
            include_math_find=True,
        )
        assert not hasattr(server_module, "Context")
        assert hasattr(server, "list_tools") and hasattr(server, "call_tool")
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()
            public_ids = {item.operation_id for item in expected_catalog.operations}
            assert {tool.name for tool in listed.tools} == {"math.find", *public_ids}

            invoke = next(
                tool
                for tool in listed.tools
                if tool.name == "matrix.determinant.compute"
            )
            descriptor = next(
                item
                for item in expected_catalog.operations
                if item.operation_id == "matrix.determinant.compute"
            )
            direct_input_schema = dict(invoke.input_schema)
            assert direct_input_schema.pop("examples") == [descriptor.examples[0].input]
            assert direct_input_schema == descriptor.input_schema
            assert invoke.output_schema == descriptor.output_schema
            assert invoke.description is not None
            assert "Canonical argument example:" in invoke.description
            assert '"num":"0"' in invoke.description
            assert invoke.annotations is not None
            assert invoke.annotations.read_only_hint is True
            assert invoke.annotations.idempotent_hint is True

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
            assert find.output_schema is not None
            assert find.output_schema["type"] == "object"
            search_schema = find.input_schema["$defs"]["OperationSearchRequest"]
            browse_schema = find.input_schema["$defs"]["OperationBrowseRequest"]
            assert "namespace" in search_schema["properties"]
            assert "domain" not in search_schema["properties"]
            assert "namespace" in browse_schema["properties"]
            assert "domain" not in browse_schema["properties"]

            browse = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "browse",
                        "namespace": "matrix",
                        "limit": 1,
                    }
                },
            )
            assert isinstance(browse.structured_content, dict)
            assert browse.structured_content["kind"] == "browse"
            assert (
                browse.structured_content["catalog_resource"] == "operation://catalog"
            )
            assert "truncated" not in browse.structured_content

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
                if isinstance(item, TextContent)
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
                "matrix.determinant.compute",
                contract["operation"]["examples"][0]["input"],
            )
            assert isinstance(result.structured_content, dict)
            assert "determinant" in result.structured_content

            catalog = await client.read_resource("operation://catalog")
            content = catalog.contents[0]
            assert isinstance(content, TextResourceContents)
            resource_snapshot = OperationCatalogSnapshot.model_validate_json(
                content.text
            )
            assert resource_snapshot.operations

    asyncio.run(scenario())
