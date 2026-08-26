"""MCP-level executable contracts for every advertised catalog example.

Covers the transport projection (`src/jacobian/mcp/tools.py:91 math_run`)
in addition to the dispatch path (`tests/integration/catalog/test_builtin_examples.py:27`).
Sequential replay establishes that every published example returns a mathematical
value rather than a host-level ExceptionGroup. The dedicated concurrent case below
proves that the server serializes mathematical kernel execution. See
https://github.com/morluto/jacobian/issues/2720.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.server import create_server
from mcp import Client

_CATALOG = Catalog.open()


def test_mcp_advertised_examples_execute_as_typed_results() -> None:
    async def scenario() -> None:
        catalog = Catalog.open()
        failures: list[str] = []
        async with Client(create_server(), raise_exceptions=False) as client:
            for descriptor in catalog.snapshot().operations:
                operation = catalog.operation(descriptor.operation_id)
                assert operation is not None
                assert operation.examples, (
                    f"{descriptor.operation_id} must advertise an example"
                )
                for example in operation.examples:
                    payload = dict(example.input)
                    result = await client.call_tool(
                        "math.run",
                        {"operation_id": descriptor.operation_id, "payload": payload},
                    )
                    if result.is_error:
                        text = (
                            result.content[0].text if result.content else "<no content>"
                        )
                        failures.append(
                            f"{descriptor.operation_id} {example.name}: is_error {text[:500]}"
                        )
                        continue
                    structured = result.structured_content
                    if not isinstance(structured, dict) or "output" not in structured:
                        failures.append(
                            f"{descriptor.operation_id} {example.name}: missing output {structured}"
                        )
                        continue
                    output = structured["output"]
                    try:
                        validated = operation.result_type.model_validate(output)
                    except Exception as exc:
                        failures.append(
                            f"{descriptor.operation_id} {example.name}: result validation {exc} output={json.dumps(output)[:500]}"
                        )
                        continue
                    if validated.model_dump(mode="json") != output:
                        failures.append(
                            f"{descriptor.operation_id} {example.name}: round-trip mismatch"
                        )
                    if structured.get("operation_id") != descriptor.operation_id:
                        failures.append(
                            f"{descriptor.operation_id}: operation_id mismatch in MCP output"
                        )
                    if (
                        not isinstance(structured.get("runtime_ms"), int)
                        or structured["runtime_ms"] < 0
                    ):
                        failures.append(
                            f"{descriptor.operation_id}: missing/invalid runtime_ms"
                        )
        assert not failures, "MCP example replay failures:\n" + "\n".join(failures)

    asyncio.run(scenario())


def test_mcp_host_failures_are_not_exception_groups() -> None:
    """A broken operation must return ToolError, not a host ExceptionGroup."""

    from pydantic import Field

    from jacobian._models import StrictModel
    from jacobian.catalog.builtins import BUILTIN_TOOLS
    from jacobian.catalog.models import MathTool

    class BoomRequest(StrictModel):
        x: int = Field(ge=0)

    class BoomResult(StrictModel):
        y: int

    def boom(_request: BoomRequest) -> BoomResult:
        raise RuntimeError("synthetic boom")

    boom_tool = MathTool(
        operation_id="test.synthetic.boom",
        title="Synthetic boom",
        description="Always raises.",
        request_type=BoomRequest,
        result_type=BoomResult,
        run=boom,
    )

    catalog = Catalog((*BUILTIN_TOOLS, boom_tool))

    from jacobian.mcp.runtime import AppState
    from jacobian.mcp.server import _build_server

    server = _build_server(state=AppState(operation_catalog=catalog))

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(
                "math.run",
                {"operation_id": "test.synthetic.boom", "payload": {"x": 1}},
            )
            # Must be a bounded tool error, not an unhandled ExceptionGroup / crash.
            assert result.is_error is True
            assert result.structured_content is None
            text = result.content[0].text if result.content else ""
            assert "synthetic boom" in text
            assert len(text) < 5000

    asyncio.run(scenario())


def test_mcp_concurrent_math_runs_return_serialized_results() -> None:
    """Concurrent clients must not enter one server's kernels simultaneously."""

    from jacobian._models import StrictModel
    from jacobian.catalog.builtins import BUILTIN_TOOLS
    from jacobian.catalog.models import MathTool
    from jacobian.mcp.runtime import AppState
    from jacobian.mcp.server import _build_server

    class Request(StrictModel):
        value: int

    class Result(StrictModel):
        value: int
        concurrent_kernel_calls: int

    active_calls = 0
    active_calls_lock = threading.Lock()

    def serial_kernel(request: Request) -> Result:
        nonlocal active_calls
        with active_calls_lock:
            active_calls += 1
            concurrent_kernel_calls = active_calls
        try:
            # The SDK runs sync tools in worker threads. This pause turns a
            # missing server gate into an observable, incorrect result.
            time.sleep(0.025)
            return Result(
                value=request.value,
                concurrent_kernel_calls=concurrent_kernel_calls,
            )
        finally:
            with active_calls_lock:
                active_calls -= 1

    tool = MathTool(
        operation_id="test.concurrent.serial_kernel",
        title="Concurrent execution sentinel",
        description="Reports concurrent mathematical-kernel calls.",
        request_type=Request,
        result_type=Result,
        run=serial_kernel,
    )
    catalog = Catalog((*BUILTIN_TOOLS, tool))
    server = _build_server(state=AppState(operation_catalog=catalog))

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            results = await asyncio.gather(
                *(
                    client.call_tool(
                        "math.run",
                        {
                            "operation_id": "test.concurrent.serial_kernel",
                            "payload": {"value": value},
                        },
                    )
                    for value in range(12)
                )
            )

        assert all(not result.is_error for result in results)
        outputs = [result.structured_content["output"] for result in results]
        assert outputs == [
            {"value": value, "concurrent_kernel_calls": 1} for value in range(12)
        ]

    asyncio.run(scenario())
