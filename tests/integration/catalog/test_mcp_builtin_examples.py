"""MCP-level executable contracts for every advertised catalog example.

Covers the transport projection (`src/jacobian/mcp/tools.py:91 math_run`)
in addition to the dispatch path (`tests/integration/catalog/test_builtin_examples.py:27`).
The catalog sweep is sequential so a failure identifies one advertised
invocation; the MCP journey separately proves independent requests can overlap.
"""

from __future__ import annotations

import asyncio
import json
import shutil

from mcp.types import TextContent

from jacobian.catalog.catalog import Catalog
from jacobian.mcp.server import create_server
from mcp import Client

_CATALOG = Catalog.open()
_SINGULAR_OPERATION_IDS = frozenset(
    {
        "polynomial.ideal.minimal_primes.compute",
        "polynomial.ideal.quotient.compute",
        "polynomial.ideal.radical.compute",
        "polynomial.ideal.saturation.compute",
        "polynomial.map.generic_degree.compute",
    }
)


def test_mcp_available_advertised_examples_execute_as_typed_results() -> None:
    async def scenario() -> None:
        catalog = Catalog.open()
        failures: list[str] = []
        async with Client(create_server(), raise_exceptions=False) as client:
            for descriptor in catalog.snapshot().operations:
                if (
                    descriptor.operation_id in _SINGULAR_OPERATION_IDS
                    and shutil.which("Singular") is None
                ):
                    continue
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
                        first = result.content[0] if result.content else None
                        text = (
                            first.text
                            if isinstance(first, TextContent)
                            else "<no content>"
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


def test_mcp_unexpected_operation_fault_uses_the_sdk_failure_path() -> None:
    """Unexpected operation faults are not recast as mathematical tool errors."""

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
            # Expected mathematical failures are returned by their owning operation.
            # A programming fault reaches the MCP SDK's installed failure path rather
            # than being mislabeled as a generic Jacobian ToolError.
            assert result.is_error is True
            assert result.structured_content is None
            first = result.content[0] if result.content else None
            text = first.text if isinstance(first, TextContent) else ""
            assert text == "Error executing tool math.run: operation execution failed"
            assert len(text) < 5000

    asyncio.run(scenario())
