"""Executable conformance checks for the pinned MCP Python SDK boundary."""

from __future__ import annotations

import asyncio
import json

from mcp.types import ContentBlock, TextContent, TextResourceContents
from mcp.types.methods import serialize_server_result

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationCatalogSnapshot, OperationResult
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server, create_server


def _content_text(block: ContentBlock) -> str:
    assert isinstance(block, TextContent)
    return block.text


def test_math_run_encloses_logarithm_on_a_positive_box() -> None:
    """A valid Arb enclosure must cross the MCP worker boundary as a result."""

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "interval.expression.box_enclosure.compute",
                    "payload": {
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
                },
            )
            assert isinstance(result.structured_content, dict)
            output = result.structured_content["output"]
            assert output["status"] == "ENCLOSED"
            assert output["lower"] is not None and output["upper"] is not None

    asyncio.run(scenario())


def test_math_run_encloses_logarithm_second_jet_on_a_positive_box() -> None:
    """The parallel Arb enclosure must also cross the MCP boundary as a result."""

    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "interval.expression.second_jet_enclosure.compute",
                    "payload": {
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
                },
            )
            assert isinstance(result.structured_content, dict)
            output = result.structured_content["output"]
            assert output["status"] == "ENCLOSED"
            assert output["value"] is not None
            assert len(output["gradient"]) == 1
            assert len(output["hessian"]) == 1

    asyncio.run(scenario())


def test_math_run_projects_unexpected_operation_failures() -> None:
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
                "math.run",
                {
                    "operation_id": "test.mcp.crashing_kernel",
                    "payload": {"value": 1},
                },
            )
        assert result.is_error is True
        assert result.structured_content is None
        text = _content_text(result.content[0]) if result.content else ""
        assert text == "Error executing tool math.run: operation execution failed"
        assert "private backend failure" not in text

    asyncio.run(scenario())


def test_math_run_projects_forged_character_as_invalid_request() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=False) as client:
            error = await client.call_tool(
                "math.run",
                {
                    "operation_id": "dirichlet_character.principal.value.compute",
                    "payload": {
                        "character": {
                            "modulus": 4,
                            "unit_residues": [1],
                            "values": [0, 1, 0, 0],
                        },
                        "integer": "1",
                    },
                },
            )

        assert error.is_error is True
        diagnostic = json.loads(
            _content_text(error.content[0]).removeprefix(
                "Error executing tool math.run: "
            )
        )
        assert diagnostic["code"] == "INVALID_REQUEST"
        assert diagnostic["stage"] == "operation_validation"
        assert diagnostic["errors"] == [
            {
                "location": ["character", "unit_residues"],
                "code": "dirichlet_character.unit_residues_mismatch",
                "message": (
                    "unit residues must be the complete canonical unit group modulo modulus"
                ),
            }
        ]

    asyncio.run(scenario())


def test_math_run_rejects_the_unsupported_degree_six_splitting_field() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            result = await client.call_tool(
                "math.run",
                {
                    "operation_id": "number_field.polynomial.splitting_field.compute",
                    "payload": {
                        "polynomial": {
                            "variables": ["x"],
                            "polynomial": {
                                "terms": [
                                    {
                                        "coefficient": {"num": "1", "den": "1"},
                                        "exponents": [6],
                                    },
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "exponents": [1],
                                    },
                                    {
                                        "coefficient": {"num": "-1", "den": "1"},
                                        "exponents": [0],
                                    },
                                ]
                            },
                        }
                    },
                },
            )

        assert result.is_error
        assert result.structured_content is None

    asyncio.run(scenario())


def test_math_run_rejects_overbound_character_values_as_invalid_request() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=False) as client:
            error = await client.call_tool(
                "math.run",
                {
                    "operation_id": "dirichlet_character.value.compute",
                    "payload": {
                        "character": {
                            "group": {
                                "modulus": 3,
                                "unit_residues": [1, 2],
                                "character_count": 2,
                                "invariant_factors": [2],
                                "generators": [2],
                                "generator_orders": [2],
                                "unit_coordinates": [[0], [1]],
                                "exponent": 2,
                            },
                            "coordinates": [0],
                        },
                        "integer": "1" * 257,
                    },
                },
            )

        assert error.is_error is True
        diagnostic = json.loads(
            _content_text(error.content[0]).removeprefix(
                "Error executing tool math.run: "
            )
        )
        assert diagnostic["code"] == "INVALID_REQUEST"
        assert (
            diagnostic["errors"][0]["code"] == "dirichlet_character.integer_digit_bound"
        )

    asyncio.run(scenario())


def test_mcp_v2_uses_sdk_typed_tools_lifespan_and_structured_resources() -> None:
    async def scenario() -> None:
        from mcp import Client

        server = create_server()
        async with Client(server, raise_exceptions=True) as client:
            listed = await client.list_tools()

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
            assert set(find.input_schema["properties"]) == {
                "query",
                "operation_id",
                "namespace",
                "limit",
                "cursor",
                "search_mode",
            }
            assert not find.input_schema.get("required")
            query_schema = find.input_schema["properties"]["query"]
            need_description = query_schema["anyOf"][0]["description"]
            assert "established mathematical names" in need_description
            assert "full scalar, batch, or exhaustive scope" in need_description
            assert "requested result" in need_description
            operation_id_schema = find.input_schema["properties"]["operation_id"]
            assert (
                "Exact public operation ID"
                in operation_id_schema["anyOf"][0]["description"]
            )
            limit_schema = find.input_schema["properties"]["limit"]
            assert limit_schema["default"] is None
            assert any(item.get("maximum") == 20 for item in limit_schema["anyOf"])
            assert find.output_schema is not None
            assert find.output_schema["type"] == "object"

            serialized_tools = serialize_server_result(
                "tools/list",
                "2026-07-28",
                listed.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
            serialized_by_name = {
                tool["name"]: tool for tool in serialized_tools["tools"]
            }
            assert (
                serialized_by_name["math.run"]["outputSchema"]
                == OperationResult.model_json_schema()
            )

            invalid_request = await client.call_tool(
                "math.find", {"unknown_key": "rejected"}
            )
            assert invalid_request.is_error is True
            assert invalid_request.content, "error responses must carry diagnostic text"
            assert any(
                "query or operation_id" in item.text
                for item in invalid_request.content
                if isinstance(item, TextContent)
            ), "error text must identify the required discovery selector"

            flat_query = await client.call_tool(
                "math.find",
                {"query": "polynomial symbolic expand", "limit": 2},
            )
            assert flat_query.is_error is False
            assert isinstance(flat_query.structured_content, dict)
            assert flat_query.structured_content["kind"] == "matches"
            assert len(flat_query.structured_content["matches"]) <= 2

            blank_need = await client.call_tool("math.find", {"query": "   "})
            assert blank_need.is_error is True
            assert any(
                "non-whitespace" in item.text
                for item in blank_need.content
                if isinstance(item, TextContent)
            )

            both_selectors = await client.call_tool(
                "math.find",
                {
                    "query": "exact determinant",
                    "operation_id": "matrix.determinant.compute",
                },
            )
            assert both_selectors.is_error is True
            assert any(
                "exactly one" in item.text
                for item in both_selectors.content
                if isinstance(item, TextContent)
            )

            inspect_with_search_option = await client.call_tool(
                "math.find",
                {"operation_id": "matrix.determinant.compute", "limit": 2},
            )
            assert inspect_with_search_option.is_error is True
            assert any(
                "only valid with query" in item.text
                for item in inspect_with_search_option.content
                if isinstance(item, TextContent)
            )

            legacy_wrapper = await client.call_tool(
                "math.find",
                {
                    "request": {
                        "op": "inspect",
                        "operation_id": "matrix.determinant.compute",
                    }
                },
            )
            assert legacy_wrapper.is_error is True

            contract_result = await client.call_tool(
                "math.find",
                {
                    "operation_id": "matrix.determinant.compute",
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
            content = catalog.contents[0]
            assert isinstance(content, TextResourceContents)
            snapshot = OperationCatalogSnapshot.model_validate_json(content.text)
            assert snapshot.operations

    asyncio.run(scenario())
