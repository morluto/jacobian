"""Catalog-derived direct MCP tools for owner-local mathematical operations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.server.mcpserver.utilities.func_metadata import ArgModelBase, FuncMetadata
from mcp.shared.exceptions import MCPError
from mcp.shared.tool_name_validation import validate_tool_name
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import ValidationError

from jacobian._execution import (
    OperationExecutionCancelledError,
    current_request_execution,
    request_execution,
)
from jacobian.canonical import CanonicalizationError, encode_strict_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import MathTool, OperationDomainValidationError
from jacobian.dispatch import (
    OperationExecutionTimeoutError,
    OperationRequestValidationError,
    parse_operation_input,
)
from jacobian.mcp.runtime import AppState, _authorize
from jacobian.mcp.tools import (
    _invalid_request_error,
    _request_cancellation,
)
from jacobian.process import bounded_process_cancellation

_FIXED_TOOL_NAMES = frozenset({"math.find", "math.run"})


@dataclass(frozen=True, slots=True)
class _RawDirectArguments:
    """One SDK argument result retaining the unparsed direct request object."""

    payload: dict[str, Any]

    def model_dump_one_level(self) -> dict[str, dict[str, Any]]:
        return {"payload": self.payload}


def direct_operation_tools(catalog: Catalog) -> list[Tool]:
    """Build one direct SDK tool for every declaration in ``catalog``."""

    tools: list[Tool] = []
    for descriptor in catalog.snapshot().operations:
        operation = catalog.operation(descriptor.operation_id)
        if operation is None:  # pragma: no cover - snapshot and lookup are atomic
            raise RuntimeError(
                f"catalog declaration disappeared for {descriptor.operation_id}"
            )
        tools.append(_direct_operation_tool(operation, catalog))
    return tools


def _direct_operation_tool(
    operation: MathTool[Any, Any],
    catalog: Catalog,
) -> Tool:
    operation_id = _operation_tool_name(operation.operation_id)
    binding = catalog._binding(operation_id)
    if binding is None:  # pragma: no cover - catalog iteration and binding are atomic
        raise RuntimeError(f"catalog binding disappeared for {operation_id}")

    def execute(
        payload: dict[str, Any],
        *,
        ctx: Context[AppState, Any],
    ) -> CallToolResult:
        _authorize(ctx)
        cancellation = _request_cancellation(ctx)
        try:
            started = time.monotonic()
            with bounded_process_cancellation(cancellation), request_execution(started):
                if cancellation.is_set():
                    raise ToolError("operation cancelled before execution")
                try:
                    request = parse_operation_input(operation.request_type, payload)
                except (CanonicalizationError, ValidationError) as cause:
                    raise OperationRequestValidationError(cause) from cause
                if cancellation.is_set():
                    raise ToolError("operation cancelled before execution")
                result = binding.run(request)
                _require_active_deadline("before result serialization")
                structured_content = result.model_dump(mode="json", by_alias=True)
                content = encode_strict_json(structured_content).decode("utf-8")
                _require_active_deadline("during result serialization")
                return CallToolResult(
                    content=[TextContent(type="text", text=content)],
                    structured_content=structured_content,
                )
        except (OperationRequestValidationError, OperationDomainValidationError) as exc:
            raise _invalid_request_error(operation_id, exc) from exc
        except OperationExecutionTimeoutError as exc:
            raise ToolError("operation execution deadline expired") from exc
        except OperationExecutionCancelledError as exc:
            raise ToolError("operation cancelled") from exc
        except (MCPError, ToolError):
            raise
        except Exception as exc:
            raise ToolError("operation execution failed") from exc

    metadata = FuncMetadata(
        arg_model=_direct_argument_model(operation),
        output_model=operation.result_type,
    )
    return Tool(
        fn=execute,
        name=operation_id,
        title=operation.title,
        description=operation.description,
        parameters=_operation_input_schema(operation),
        fn_metadata=metadata,
        is_async=False,
        context_kwarg="ctx",
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )


def _direct_argument_model(
    operation: MathTool[Any, Any],
) -> type[ArgModelBase]:
    """Adapt the SDK's argument hook to Jacobian's strict JSON parser."""

    def model_validate(
        cls: type[Any],
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> _RawDirectArguments:
        del cls, args, kwargs
        # MCP tool arguments have an object root. Keep this hook deliberately
        # non-validating so owner parsing occurs inside Jacobian's complete
        # request-scoped execution envelope in ``execute``.
        return _RawDirectArguments(payload=cast(dict[str, Any], value))

    argument_model = type(
        f"{operation.request_type.__name__}DirectMCPArguments",
        (ArgModelBase,),
        {
            "__module__": __name__,
            "model_validate": classmethod(model_validate),
        },
    )
    return cast(type[ArgModelBase], argument_model)


def _operation_input_schema(operation: MathTool[Any, Any]) -> dict[str, Any]:
    """Project an owner request schema into MCP's required object root."""

    schema = cast(dict[str, Any], operation.request_type.model_json_schema())
    if schema.get("type") == "object":
        return schema
    if not _schema_is_object(schema, root=schema):
        raise TypeError(
            f"{operation.operation_id} request schema is not an MCP object schema"
        )
    return {**schema, "type": "object"}


def _schema_is_object(schema: dict[str, Any], *, root: dict[str, Any]) -> bool:
    if schema.get("type") == "object":
        return True
    reference = schema.get("$ref")
    if isinstance(reference, str) and reference.startswith("#/$defs/"):
        definition = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        return isinstance(definition, dict) and _schema_is_object(definition, root=root)
    variants = schema.get("anyOf")
    return (
        isinstance(variants, list)
        and bool(variants)
        and all(
            isinstance(variant, dict) and _schema_is_object(variant, root=root)
            for variant in variants
        )
    )


def _operation_tool_name(operation_id: str) -> str:
    """Keep the operation ID unchanged when it is a conforming MCP tool name."""

    if operation_id in _FIXED_TOOL_NAMES:
        raise ValueError(f"operation ID conflicts with fixed MCP tool: {operation_id}")
    validation = validate_tool_name(operation_id)
    if not validation.is_valid:
        raise ValueError(f"operation ID is not an MCP-safe tool name: {operation_id}")
    return operation_id


def _require_active_deadline(stage: str) -> None:
    execution = current_request_execution()
    if (
        execution is not None
        and execution.deadline is not None
        and time.monotonic() >= execution.deadline
    ):
        raise OperationExecutionTimeoutError(f"request deadline expired {stage}")


__all__ = ["direct_operation_tools"]
