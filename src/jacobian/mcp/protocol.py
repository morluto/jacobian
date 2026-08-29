"""Transport-neutral MCP projection over the admitted operation catalog."""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver.resources import FunctionResource
from mcp.server.mcpserver.tools import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.types import ToolAnnotations
from pydantic import ConfigDict

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationCatalogSnapshot
from jacobian.mcp.guidance import MATH_FIND_DESCRIPTION
from jacobian.mcp.runtime import AppState
from jacobian.mcp.tools import math_find, run_direct_math_tool

_DIRECT_CONTEXT_ARGUMENT = "_jacobian_mcp_context"
_DIRECT_CALLER_CONTEXT_ARGUMENT = "__jacobian_mcp_context_supplied"


class _DirectArguments(ArgModelBase):
    """Pass raw direct-tool arguments to the one strict owner parser.

    ``Tool.parameters`` publishes the selected owner's exact request schema. The
    SDK argument adapter deliberately retains the raw mapping so Jacobian's
    strict JSON canonicalization and owner model run exactly once in the tool
    handler, where their diagnostics can be bounded before projection.
    """

    model_config = ConfigDict(extra="allow")

    def model_dump_one_level(self) -> dict[str, Any]:
        arguments = dict(self.__pydantic_extra__ or {})
        if _DIRECT_CONTEXT_ARGUMENT in arguments:
            # Tool.run injects the real context after this adapter validates the
            # caller mapping. Preserve a marker so invoke() can put the reserved
            # caller key back through Jacobian's strict owner parser.
            arguments[_DIRECT_CALLER_CONTEXT_ARGUMENT] = True
        return arguments


def _direct_tool(
    operation: MathTool[Any, Any],
) -> Tool:
    """Compile one immutable declaration into one direct typed MCP tool."""

    def invoke(**arguments: Any) -> Any:
        ctx = arguments.pop(_DIRECT_CONTEXT_ARGUMENT)
        if arguments.pop(_DIRECT_CALLER_CONTEXT_ARGUMENT, False):
            arguments[_DIRECT_CONTEXT_ARGUMENT] = True
        return run_direct_math_tool(operation, arguments, ctx=ctx)

    metadata = FuncMetadata(
        arg_model=_DirectArguments,
        output_model=operation.result_type,
        output_schema=operation.result_type.model_json_schema(),
    )
    parameters = operation.request_type.model_json_schema()
    if parameters.get("type") != "object":
        raise TypeError(
            f"{operation.operation_id} request schema is not an MCP object schema"
        )
    description = operation.description
    if operation.examples:
        canonical_example = dict(operation.examples[0].input)
        parameters["examples"] = [canonical_example]
        rendered_example = json.dumps(
            canonical_example,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        description = f"{description}\nCanonical argument example: `{rendered_example}`"
    return Tool(
        fn=invoke,
        name=operation.operation_id,
        title=operation.title,
        description=description,
        parameters=parameters,
        fn_metadata=metadata,
        is_async=False,
        context_kwarg=_DIRECT_CONTEXT_ARGUMENT,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )


def _math_find_tool() -> Tool:
    return Tool.from_function(
        math_find,
        name="math.find",
        title="Search installed Jacobian mathematical vocabulary",
        description=MATH_FIND_DESCRIPTION,
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
        structured_output=True,
    )


def compile_tools(
    state: AppState,
    *,
    include_math_find: bool = False,
) -> tuple[Tool, ...]:
    """Compile one deterministic MCP tool list from one immutable catalog."""

    catalog = state.operation_catalog
    operations = tuple(
        _required_operation(catalog, descriptor.operation_id)
        for descriptor in catalog.snapshot().operations
    )
    direct_tools = tuple(_direct_tool(operation) for operation in operations)
    discovery_tools = (_math_find_tool(),) if include_math_find else ()
    return (*discovery_tools, *direct_tools)


def _required_operation(catalog: Catalog, operation_id: str) -> MathTool[Any, Any]:
    operation = catalog.operation(operation_id)
    if operation is None:  # pragma: no cover - immutable Catalog invariant
        raise RuntimeError(f"catalog snapshot lost operation {operation_id}")
    return operation


def compile_resources(
    state: AppState,
) -> tuple[FunctionResource, ...]:
    """Compile immutable catalog resources for one server instance."""

    def operation_catalog() -> OperationCatalogSnapshot:
        return state.operation_catalog.snapshot()

    return (
        FunctionResource.from_function(
            operation_catalog,
            uri="operation://catalog",
            name="operation-catalog",
            description=(
                "Installed model-facing operations and their compact schemas."
            ),
            mime_type="application/json",
        ),
    )


__all__ = ["compile_resources", "compile_tools"]
