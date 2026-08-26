"""MCP-level executable contracts for every advertised catalog example.

Covers the transport projection (`src/jacobian/mcp/tools.py:91 math_run`)
in addition to the dispatch path (`tests/integration/catalog/test_builtin_examples.py:27`).
Sequential replay avoids the known concurrent-worker segfault while still
establishing that every published example returns a mathematical value rather
than a host-level ExceptionGroup. See https://github.com/morluto/jacobian/issues/2713.
"""

from __future__ import annotations

import asyncio
import json

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
