"""Opt-in local-host delivery of complete results through temporary SDK resources."""

from __future__ import annotations

import inspect
import json
import math
import secrets
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, Literal, cast

import anyio
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, ResourceLink, TextContent
from mcp_types.methods import serialize_server_result
from pydantic import BaseModel, ConfigDict, Field, RootModel

from jacobian.catalog.models import OperationResult
from jacobian.mcp.models import OperationFindResponse


class _RetainedReference(BaseModel):
    """Serving metadata, never an operation's mathematical value or input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    uri: str = Field(
        min_length=82, max_length=82, pattern=r"^jacobian-result://[0-9a-f]{64}$"
    )
    byte_count: int = Field(ge=1, le=64 * 1024 * 1024)
    expires_at: datetime
    preview: str = Field(max_length=512)
    representation: Literal["structured_result", "tool_result"]


class RetainedResultReference(_RetainedReference):
    kind: Literal["retained_result"] = "retained_result"


class RetainedErrorReference(_RetainedReference):
    kind: Literal["retained_error"] = "retained_error"


class DeliveredRunResult(RootModel[OperationResult | RetainedResultReference]):
    """An inline exact result or a reference to its complete serialized bytes."""

    model_config = ConfigDict(json_schema_extra={"type": "object"})


class DeliveredFindResult(RootModel[OperationFindResponse | RetainedResultReference]):
    """Inline discovery/inspection or a reference to the complete contract."""

    model_config = ConfigDict(json_schema_extra={"type": "object"})


@dataclass(frozen=True, slots=True)
class _RetainedResult:
    text: str
    byte_count: int
    deadline: float


class ResultDelivery:
    """One local host's finite, expiring result storage; no mathematical state.

    The inline budget measures UTF-8 JSON of CallToolResult, excluding JSON-RPC
    framing. Counts, retained bytes and lifetime are independently bounded.
    Existing references are never evicted early to make room for another call.
    """

    def __init__(
        self,
        inline_result_bytes: int,
        *,
        max_retained_bytes: int = 64 * 1024 * 1024,
        max_results: int = 16,
        lifetime_seconds: float = 300,
    ) -> None:
        if (
            type(inline_result_bytes) is not int
            or not 4096 <= inline_result_bytes <= 1048576
        ):
            raise ValueError("inline_result_bytes must lie in [4096, 1048576]")
        if (
            type(max_retained_bytes) is not int
            or not 1 <= max_retained_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("retained bytes must lie in [1, 67108864]")
        if type(max_results) is not int or not 1 <= max_results <= 16:
            raise ValueError("retained result count must lie in [1, 16]")
        if (
            isinstance(lifetime_seconds, bool)
            or not math.isfinite(lifetime_seconds)
            or not 0 < lifetime_seconds <= 300
        ):
            raise ValueError("retained lifetime must lie in (0, 300] seconds")
        self.inline_result_bytes = inline_result_bytes
        self.max_retained_bytes = max_retained_bytes
        self.max_results = max_results
        self.lifetime_seconds = lifetime_seconds
        self._results: dict[str, _RetainedResult] = {}
        self._retained_bytes = 0
        self._active_sessions = 0
        self._lock = threading.RLock()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Clear host storage only after its final SDK session shuts down."""

        with self._lock:
            self._active_sessions += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_sessions -= 1
                if self._active_sessions == 0:
                    self.clear()

    def _expire_locked(self) -> None:
        now = time.monotonic()
        for token, result in tuple(self._results.items()):
            if result.deadline <= now:
                self._retained_bytes -= result.byte_count
                del self._results[token]

    @property
    def retained_count(self) -> int:
        with self._lock:
            self._expire_locked()
            return len(self._results)

    @property
    def retained_bytes(self) -> int:
        with self._lock:
            self._expire_locked()
            return self._retained_bytes

    def clear(self) -> None:
        with self._lock:
            self._results.clear()
            self._retained_bytes = 0

    async def expire_periodically(self) -> None:
        while True:
            await anyio.sleep(min(30, self.lifetime_seconds))
            with self._lock:
                self._expire_locked()

    def read(self, token: str) -> str:
        if len(token) != 64 or any(
            character not in "0123456789abcdef" for character in token
        ):
            raise ValueError("retained result is unavailable: invalid reference")
        with self._lock:
            self._expire_locked()
            result = self._results.get(token)
            if result is None:
                raise ValueError(
                    "retained result is unavailable: expired, unknown, or host restarted"
                )
            return result.text

    async def __call__(
        self, ctx: ServerRequestContext[Any, Any], call_next: CallNext
    ) -> HandlerResult:
        if ctx.request is not None:
            raise MCPError(
                code=-32600, message="temporary result delivery is local-only"
            )
        result = await call_next(ctx)
        if ctx.method == "tools/call":
            serialized = result if isinstance(result, dict) else None
            # The SDK's terminal handler serializes models before unwinding
            # context middleware; an inner middleware may also return a model.
            if isinstance(result, dict) and "content" in result:
                result = CallToolResult.model_validate(result)
            if isinstance(result, CallToolResult):
                projected = await self.project(
                    result,
                    protocol_version=ctx.protocol_version,
                    serialized_result=serialized,
                )
                return _wire(projected, ctx.protocol_version)
        return result

    async def project(
        self,
        result: CallToolResult,
        *,
        protocol_version: str | None = None,
        serialized_result: dict[str, Any] | None = None,
    ) -> CallToolResult:
        await anyio.lowlevel.checkpoint()
        try:
            projected, candidate = await anyio.to_thread.run_sync(
                self._prepare,
                result,
                protocol_version,
                serialized_result,
                abandon_on_cancel=True,
            )
            await anyio.lowlevel.checkpoint()
            if candidate is not None:
                self._publish(*candidate)
            return projected
        except ToolError as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))], is_error=True
            )

    def _prepare(
        self,
        result: CallToolResult,
        protocol_version: str | None,
        serialized_result: dict[str, Any] | None,
    ) -> tuple[CallToolResult, tuple[str, _RetainedResult] | None]:
        structured = result.structured_content
        envelope = result.is_error or structured is None
        value = (
            (
                serialized_result
                if serialized_result is not None
                else _wire(result, protocol_version)
            )
            if envelope
            else structured
        )
        if not isinstance(value, dict):
            raise _delivery_error("the tool response has no JSON object to retain")
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        byte_count = len(text.encode("utf-8"))
        inline = (
            result
            if envelope
            else result.model_copy(
                update={"content": [TextContent(type="text", text=text)]}
            )
        )
        if (
            byte_count <= self.inline_result_bytes
            and _wire_size(inline, protocol_version) <= self.inline_result_bytes
        ):
            return inline, None
        if byte_count > self.max_retained_bytes:
            raise _delivery_error(
                "the complete result exceeds temporary retained-result capacity"
            )
        token = secrets.token_hex(32)
        uri = "jacobian-result://" + token
        reference_type = (
            RetainedErrorReference if result.is_error else RetainedResultReference
        )
        deadline = time.monotonic() + self.lifetime_seconds
        reference = reference_type(
            uri=uri,
            byte_count=byte_count,
            expires_at=datetime.now(UTC) + timedelta(seconds=self.lifetime_seconds),
            preview=(
                "Operation failed. Retrieve the complete SDK error response for diagnostic details; no mathematical conclusion follows."
                if result.is_error
                else _preview(value)
            ),
            representation="tool_result" if envelope else "structured_result",
        )
        metadata = reference.model_dump(mode="json")
        projected = CallToolResult(
            content=[
                TextContent(
                    type="text", text=json.dumps(metadata, separators=(",", ":"))
                ),
                ResourceLink(
                    type="resource_link",
                    uri=uri,
                    name="jacobian-result",
                    mime_type="application/json",
                    size=byte_count,
                    description="Complete JSON; retrieve before using the mathematical result.",
                ),
            ],
            structured_content=metadata,
            is_error=result.is_error,
            _meta=result.meta,
        )
        if _wire_size(projected, protocol_version) > self.inline_result_bytes:
            raise _delivery_error(
                "the configured budget cannot carry a result reference"
            )
        return projected, (token, _RetainedResult(text, byte_count, deadline))

    def _publish(self, token: str, retained: _RetainedResult) -> None:
        with self._lock:
            self._expire_locked()
            if retained.deadline <= time.monotonic():
                raise _delivery_error(
                    "the temporary reference expired before publication"
                )
            if token in self._results:
                raise _delivery_error("could not allocate a unique temporary reference")
            if (
                len(self._results) >= self.max_results
                or self._retained_bytes + retained.byte_count > self.max_retained_bytes
            ):
                raise _delivery_error("temporary retained-result capacity is exhausted")
            self._results[token] = retained
            self._retained_bytes += retained.byte_count


def _wire(result: CallToolResult, protocol_version: str | None) -> dict[str, Any]:
    value = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    if protocol_version is not None:
        return serialize_server_result("tools/call", protocol_version, value)
    return value


def _wire_size(result: CallToolResult, protocol_version: str | None) -> int:
    return len(
        json.dumps(
            _wire(result, protocol_version), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    )


def _delivery_error(message: str) -> ToolError:
    return ToolError(
        json.dumps(
            {
                "code": "RESULT_DELIVERY_FAILED",
                "stage": "result_delivery",
                "message": message,
                "hint": "Retrieve existing results before expiry; retry after expiry or configure a larger local-host delivery budget. No mathematical conclusion follows from this failure.",
            },
            separators=(",", ":"),
        )
    )


def _preview(value: dict[str, Any]) -> str:
    if isinstance(value.get("output"), dict) and isinstance(
        value.get("operation_id"), str
    ):
        output = value["output"]
        fields = ", ".join(
            f"{key}: {len(item)} entries" if isinstance(item, list) else key
            for key, item in tuple(output.items())[:8]
        )
        summary = f"Exact result for {value['operation_id']}. {fields}. Retrieve the complete JSON before composition."
    elif isinstance(value.get("operation"), dict):
        operation = value["operation"]
        schema = operation.get("input_schema")
        properties = schema.get("properties") if isinstance(schema, dict) else None
        fields = (
            ", ".join(str(key) for key in properties)
            if isinstance(properties, dict)
            else ""
        )
        summary = f"Full contract: {operation.get('title', '')}. Input fields: {fields[:160]}. {operation.get('description', '')}"
    else:
        summary = "Tool response retained as complete JSON. Retrieve it before using the result."
    # This is explicitly an incomplete description, never a truncated math value.
    return summary.encode("ascii", "backslashreplace")[:512].decode("ascii")


def with_delivery_schema(
    function: Callable[..., Awaitable[BaseModel]],
    output_model: type[BaseModel],
) -> Callable[..., Awaitable[BaseModel]]:
    """Preserve the typed SDK input signature and advertise both delivery forms."""

    @wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> BaseModel:
        return await function(*args, **kwargs)

    signature = inspect.signature(function, eval_str=True)
    cast(Any, wrapped).__signature__ = signature.replace(return_annotation=output_model)
    return wrapped
