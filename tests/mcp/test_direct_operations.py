"""Direct catalog-derived MCP operation registration and invocation."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Self

from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, TextContent
from pydantic import model_validator

from jacobian._execution import (
    OperationExecutionCancelledError,
    current_request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool
from jacobian.mcp.direct_tools import _operation_input_schema, _operation_tool_name
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server, create_server

_FIXED_TOOLS = {"math.find", "math.run"}


def _operations(*operation_ids: str) -> tuple[MathTool[Any, Any], ...]:
    catalog = Catalog.open()
    operations: list[MathTool[Any, Any]] = []
    for operation_id in operation_ids:
        operation = catalog.operation(operation_id)
        assert operation is not None
        operations.append(operation)
    return tuple(operations)


def _server(*operation_ids: str) -> Any:
    return _build_server(
        state=AppState(operation_catalog=Catalog(_operations(*operation_ids)))
    )


def test_every_catalog_operation_is_one_direct_identity_named_tool() -> None:
    async def scenario() -> None:
        from mcp import Client

        catalog_ids = {
            descriptor.operation_id
            for descriptor in Catalog.open().snapshot().operations
        }
        async with Client(create_server(), raise_exceptions=True) as client:
            listed_ids = {tool.name for tool in (await client.list_tools()).tools}

        assert listed_ids == catalog_ids | _FIXED_TOOLS
        assert all(
            _operation_tool_name(operation_id) == operation_id
            for operation_id in catalog_ids
        )

    asyncio.run(scenario())


def test_direct_schemas_are_owner_contracts_with_only_mcp_root_projection() -> None:
    operation_ids = (
        "integer.compute.extended_gcd",
        "matrix.determinant.compute",
        "topology.simplicial_complex.canonicalize",
    )
    operations = _operations(*operation_ids)

    async def scenario() -> None:
        from mcp import Client

        async with Client(_server(*operation_ids), raise_exceptions=True) as client:
            listed = {tool.name: tool for tool in (await client.list_tools()).tools}

        for operation in operations:
            direct = listed[operation.operation_id]
            assert direct.input_schema == _operation_input_schema(operation)
            assert direct.output_schema == operation.result_type.model_json_schema()
            assert direct.annotations is not None
            assert direct.annotations.read_only_hint is True
            assert direct.annotations.idempotent_hint is True
            assert "operation_id" not in direct.input_schema.get("properties", {})
            assert "payload" not in direct.input_schema.get("properties", {})

        root_union = listed["topology.simplicial_complex.canonicalize"].input_schema
        assert root_union["type"] == "object"
        owner_union = operations[-1].request_type.model_json_schema()
        assert {
            key: value for key, value in root_union.items() if key != "type"
        } == owner_union

    asyncio.run(scenario())


def test_catalog_membership_automatically_changes_the_direct_tool_list() -> None:
    first = "integer.compute.extended_gcd"
    second = "matrix.determinant.compute"

    async def listed(server: Any) -> set[str]:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            return {tool.name for tool in (await client.list_tools()).tools}

    only_first = asyncio.run(listed(_server(first)))
    both = asyncio.run(listed(_server(first, second)))

    assert only_first == _FIXED_TOOLS | {first}
    assert both == only_first | {second}


def test_direct_calls_return_owner_results_without_dispatch_envelopes() -> None:
    operation_ids = (
        "integer.compute.extended_gcd",
        "matrix.determinant.compute",
    )
    determinant = _operations(operation_ids[1])[0]

    async def scenario() -> None:
        from mcp import Client

        async with Client(_server(*operation_ids), raise_exceptions=True) as client:
            gcd = await client.call_tool(
                "integer.compute.extended_gcd",
                {"left": "84", "right": "30"},
            )
            matrix = await client.call_tool(
                determinant.operation_id,
                dict(determinant.examples[0].input),
            )

        assert gcd.structured_content == {
            "gcd": "6",
            "left_coefficient": "-1",
            "right_coefficient": "3",
        }
        assert matrix.structured_content == {"determinant": {"num": "-6", "den": "1"}}
        gcd_content = gcd.content[0]
        matrix_content = matrix.content[0]
        assert isinstance(gcd_content, TextContent)
        assert isinstance(matrix_content, TextContent)
        assert json.loads(gcd_content.text) == gcd.structured_content
        assert json.loads(matrix_content.text) == matrix.structured_content
        assert "operation_id" not in gcd.structured_content
        assert "output" not in gcd.structured_content

    asyncio.run(scenario())


def test_direct_parsing_and_execution_share_one_request_envelope() -> None:
    observed_envelopes: list[bool] = []

    class Request(StrictModel):
        value: int

        @model_validator(mode="after")
        def record_parsing_envelope(self) -> Self:
            observed_envelopes.append(current_request_execution() is not None)
            return self

    class Result(StrictModel):
        value: int

    def run(request: Request) -> Result:
        observed_envelopes.append(current_request_execution() is not None)
        return Result(value=request.value)

    operation = MathTool(
        operation_id="test.direct.envelope",
        title="Direct request envelope",
        description="Exercises request-scoped direct parsing and execution.",
        request_type=Request,
        result_type=Result,
        run=run,
    )
    server = _build_server(state=AppState(operation_catalog=Catalog((operation,))))

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(operation.operation_id, {"value": 7})

        assert result.structured_content == {"value": 7}

    asyncio.run(scenario())
    assert observed_envelopes == [True, True]


def test_direct_calls_preserve_strict_and_domain_invalid_params() -> None:
    operation_ids = (
        "integer.compute.extended_gcd",
        "universal_algebra.term.evaluate.compute",
    )
    universal = _operations(operation_ids[1])[0]
    invalid_domain = copy.deepcopy(universal.examples[0].input)
    invalid_domain["assignment"] = [0]
    secret = "do-not-reflect-this-caller-value"

    async def scenario() -> None:
        from mcp import Client

        async with Client(_server(*operation_ids), raise_exceptions=True) as client:
            try:
                await client.call_tool(
                    "integer.compute.extended_gcd",
                    {"left": "84", "right": "30", "private": secret},
                )
            except MCPError as invalid:
                assert invalid.code == INVALID_PARAMS
                assert invalid.data["errors"] == [
                    {
                        "location": ["private"],
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    }
                ]
                assert secret not in invalid.message
                assert secret not in str(invalid.data)
            else:  # pragma: no cover - regression assertion
                raise AssertionError("extra direct argument was accepted")

            try:
                await client.call_tool(
                    "integer.compute.extended_gcd",
                    {"left": 1.5, "right": "30"},
                )
            except MCPError as noncanonical:
                assert noncanonical.code == INVALID_PARAMS
                assert noncanonical.data["errors"] == [
                    {
                        "location": [],
                        "code": "canonicalization_error",
                        "message": "JSON floating-point numbers are not allowed",
                    }
                ]
            else:  # pragma: no cover - regression assertion
                raise AssertionError("floating-point direct argument was accepted")

            try:
                await client.call_tool(universal.operation_id, invalid_domain)
            except MCPError as domain:
                assert domain.code == INVALID_PARAMS
                assert domain.data["errors"] == [
                    {
                        "location": ["assignment"],
                        "code": "universal_algebra.assignment_coverage",
                        "message": "assignment must cover exactly the referenced variables",
                    }
                ]
            else:  # pragma: no cover - regression assertion
                raise AssertionError("domain-invalid direct argument was accepted")

    asyncio.run(scenario())


def test_direct_calls_do_not_expose_unexpected_owner_failures() -> None:
    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    def crash(_request: Request) -> Result:
        raise RuntimeError("private backend value")

    operation = MathTool(
        operation_id="test.direct.crash",
        title="Direct crash sentinel",
        description="Exercises direct MCP failure projection.",
        request_type=Request,
        result_type=Result,
        run=crash,
    )
    server = _build_server(state=AppState(operation_catalog=Catalog((operation,))))

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(operation.operation_id, {"value": 1})

        assert result.is_error is True
        assert result.structured_content is None
        assert result.content and isinstance(result.content[0], TextContent)
        assert result.content[0].text == (
            "Error executing tool test.direct.crash: operation execution failed"
        )
        assert "private backend value" not in result.content[0].text

    asyncio.run(scenario())


def test_direct_calls_preserve_owner_cancellation_diagnosis() -> None:
    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    def cancel(_request: Request) -> Result:
        raise OperationExecutionCancelledError("private cancellation detail")

    operation = MathTool(
        operation_id="test.direct.cancel",
        title="Direct cancellation sentinel",
        description="Exercises direct MCP cancellation projection.",
        request_type=Request,
        result_type=Result,
        run=cancel,
    )
    server = _build_server(state=AppState(operation_catalog=Catalog((operation,))))

    async def scenario() -> None:
        from mcp import Client

        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(operation.operation_id, {"value": 1})

        assert result.is_error is True
        assert result.structured_content is None
        assert result.content and isinstance(result.content[0], TextContent)
        assert result.content[0].text == (
            "Error executing tool test.direct.cancel: operation cancelled"
        )
        assert "private cancellation detail" not in result.content[0].text
        assert "operation execution failed" not in result.content[0].text

    asyncio.run(scenario())
