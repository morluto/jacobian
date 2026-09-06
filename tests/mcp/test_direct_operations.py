"""Experimental direct catalog projection used by frozen evaluations."""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any, Self

import pytest
from mcp.shared.exceptions import MCPError
from mcp.types import INVALID_PARAMS, TextContent
from pydantic import model_validator

from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    current_request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.mcp.direct_tools import (
    _operation_input_schema,
    _operation_tool_name,
    direct_operation_tools,
)
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server, create_server
from jacobian.mcp.tools import _invalid_request_error

_FIXED_TOOLS = {"math.find", "math.run"}


def _operations(*operation_ids: str) -> tuple[MathTool[Any, Any], ...]:
    catalog = Catalog.open()
    operations: list[MathTool[Any, Any]] = []
    for operation_id in operation_ids:
        operation = catalog.operation(operation_id)
        assert operation is not None
        operations.append(operation)
    return tuple(operations)


def _direct_server(catalog: Catalog) -> Any:
    return _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog),
    )


def _server(*operation_ids: str) -> Any:
    return _direct_server(Catalog(_operations(*operation_ids)))


def test_production_server_does_not_eagerly_expose_catalog_operations() -> None:
    async def scenario() -> None:
        from mcp import Client

        async with Client(create_server(), raise_exceptions=True) as client:
            listed_ids = {tool.name for tool in (await client.list_tools()).tools}

        assert listed_ids == _FIXED_TOOLS

    asyncio.run(scenario())


def test_experimental_direct_projection_preserves_identity_names() -> None:
    catalog_ids = {
        descriptor.operation_id for descriptor in Catalog.open().snapshot().operations
    }

    assert all(
        _operation_tool_name(operation_id) == operation_id
        for operation_id in catalog_ids
    )


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

        topology_schema = listed[
            "topology.simplicial_complex.canonicalize"
        ].input_schema
        assert topology_schema["type"] == "object"
        assert topology_schema == operations[-1].request_type.model_json_schema()

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


def test_direct_adaptive_call_preserves_typed_arb_domain_uncertainty() -> None:
    operation_id = "interval.expression.adaptive_range_enclosure.compute"
    operation = _operations(operation_id)[0]
    payload = {
        "expression": {
            "op": "log",
            "children": [
                {
                    "op": "add",
                    "children": [
                        {"op": "var", "variable": "x"},
                        {
                            "op": "const",
                            "value": {"num": "1", "den": "1" + "0" * 127},
                        },
                    ],
                }
            ],
        },
        "box": {
            "variables": ["x"],
            "intervals": [
                {
                    "lower": {"num": "0", "den": "1"},
                    "upper": {"num": "1", "den": "1"},
                }
            ],
        },
        "precision_bits": 32,
        "maximum_precision_bits": 32,
        "target_width": {"num": "100", "den": "1"},
        "max_leaves": 1,
        "max_depth": 8,
        "max_evaluations": 1,
        "wall_seconds": 30,
    }
    request = operation.request_type.model_validate(payload)
    expected = operation.run(request).model_dump(mode="json")

    async def scenario() -> None:
        from mcp import Client

        async with Client(_server(operation_id), raise_exceptions=True) as client:
            result = await client.call_tool(operation_id, payload)

        assert result.structured_content == expected
        assert result.structured_content["disposition"] == {
            "status": "DOMAIN_UNPROVEN",
            "reason": "MAX_LEAVES",
        }
        assert result.structured_content["enclosure"] is None
        assert result.structured_content["leaves"][0]["status"] == "DOMAIN_UNPROVEN"

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
    server = _direct_server(Catalog((operation,)))

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
    server = _direct_server(Catalog((operation,)))

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
    server = _direct_server(Catalog((operation,)))

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


@pytest.mark.parametrize(
    ("exception", "code"),
    (
        (OperationExecutionTimeoutError("private timeout detail"), "OPERATION_TIMEOUT"),
        (
            OperationExecutionCancelledError("private cancellation detail"),
            "OPERATION_CANCELLED",
        ),
    ),
)
def test_math_run_preserves_bounded_operation_failure_context(
    exception: Exception, code: str
) -> None:
    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    def fail(_request: Request) -> Result:
        raise exception

    operation = MathTool(
        operation_id="test.context.failure",
        title="Context failure sentinel",
        description="Exercises bounded math.run execution diagnostics.",
        request_type=Request,
        result_type=Result,
        run=fail,
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            _direct_server(Catalog((operation,))), raise_exceptions=False
        ) as client:
            result = await client.call_tool(
                "math.run",
                {"operation_id": operation.operation_id, "payload": {"value": 1}},
            )

        assert result.is_error is True
        assert result.content and isinstance(result.content[0], TextContent)
        diagnostic = json.loads(
            result.content[0].text.removeprefix("Error executing tool math.run: ")
        )
        assert diagnostic == {
            "code": code,
            "operation_id": operation.operation_id,
            "stage": "operation_execution",
        }
        assert "private" not in result.content[0].text

    asyncio.run(scenario())


def test_math_run_reports_resource_admission_separately_from_invalid_payload() -> None:
    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int

    def reject(_request: Request) -> Result:
        raise OperationResourceAdmissionError(
            location=("value",),
            code="test.budget_exceeded",
            message="test work exceeds the 10-unit budget",
        )

    operation = MathTool(
        operation_id="test.resource.reject",
        title="Resource rejection sentinel",
        description="Exercises math.run resource-admission recovery.",
        request_type=Request,
        result_type=Result,
        run=reject,
    )

    async def scenario() -> None:
        from mcp import Client

        async with Client(
            _direct_server(Catalog((operation,))), raise_exceptions=True
        ) as client:
            with pytest.raises(MCPError) as error:
                await client.call_tool(
                    "math.run",
                    {"operation_id": operation.operation_id, "payload": {"value": 1}},
                )

        assert error.value.code == INVALID_PARAMS
        assert error.value.data["code"] == "RESOURCE_ADMISSION_REJECTED"
        assert error.value.data["stage"] == "resource_admission"
        assert error.value.data["operation_id"] == operation.operation_id
        assert "correct the fields" not in error.value.data["hint"]

    asyncio.run(scenario())


def test_budget_named_mathematical_precondition_remains_invalid_payload() -> None:
    error = OperationDomainValidationError(
        location=("matrix",),
        code="matrix.budget_exceeded",
        message="matrix must be square",
    )

    projected = _invalid_request_error("matrix.determinant.compute", error)

    assert projected.data["code"] == "INVALID_REQUEST"
