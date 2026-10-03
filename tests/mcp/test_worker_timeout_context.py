"""Both SDK entry points retain source-authored context from a real worker."""

import asyncio
import json
import logging
from typing import Any

import pytest
from mcp.types import TextContent
from tests.support.chromatic_timeout import (
    chromatic_timeout_request,
    chromatic_timeout_worker_output,
    patch_chromatic_clock,
)

from jacobian._execution import TimeoutOwner, bind_request_deadline, request_execution
from jacobian.catalog.catalog import Catalog
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    CHROMATIC_BIPARTITION_OPERATION,
    ChromaticBipartitionRequest,
    ChromaticBipartitionResult,
)
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from jacobian.process import BoundedProcessResult
from mcp import Client


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    (
        "wall_seconds",
        "outer_deadline",
        "enclosing_deadline",
        "retains_context",
        "uses_outer_owner",
    ),
    [
        (5, None, None, True, False),
        (120, None, None, True, False),
        (5, 101, None, False, True),
        (5, 105, None, False, True),
        (5, 110, None, True, False),
        (5, None, 101, False, False),
        (5, None, 105, False, False),
        (5, None, 110, True, False),
        (5, 110, 101, False, False),
        (5, 101, 110, False, True),
        (5, 101, 101, False, True),
        (5, 105, 105, False, True),
        (5, 110, 110, True, False),
    ],
)
@pytest.mark.parametrize("supervisor_timeout", [False, True])
@pytest.mark.parametrize(
    "outer_owner", [TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.BACKEND_TIMEOUT]
)
def test_worker_timeout_context_through_sdk(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    direct: bool,
    wall_seconds: int,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
    uses_outer_owner: bool,
    supervisor_timeout: bool,
    outer_owner: TimeoutOwner,
) -> None:
    request = chromatic_timeout_request(wall_seconds)
    clock = [100.0]
    completed_calls = 0

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        nonlocal completed_calls
        completed_calls += 1
        input_bytes = kwargs["input_bytes"]
        assert isinstance(input_bytes, bytes)
        clock[0] = 100.5
        output, solver_timeouts = chromatic_timeout_worker_output(
            monkeypatch, input_bytes
        )
        assert len(solver_timeouts) == 1
        assert json.loads(output)["configured_seconds"] == wall_seconds
        assert len(output) <= 160
        return BoundedProcessResult(0, output, b"", False, False, supervisor_timeout)

    native_process = process_owner.find_chromatic_bipartition

    def run_with_enclosing_deadline(
        source: ChromaticBipartitionRequest,
    ) -> ChromaticBipartitionResult:
        # Bind the actual parent request context only during the synchronous
        # operation, leaving the live SDK's event-loop clock unchanged.
        with monkeypatch.context() as execution_patch:
            patch_chromatic_clock(execution_patch, clock)
            with request_execution(
                100, outer_deadline=outer_deadline, timeout_owner=outer_owner
            ):
                if enclosing_deadline is not None:
                    bind_request_deadline(enclosing_deadline)
                return native_process(source)

    monkeypatch.setattr(process_owner, "run_bounded_process", complete)
    monkeypatch.setattr(
        process_owner, "find_chromatic_bipartition", run_with_enclosing_deadline
    )
    operation = CHROMATIC_BIPARTITION_OPERATION
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    payload = request.model_dump(mode="json")

    async def scenario() -> Any:
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                operation.operation_id if direct else "math.run",
                payload
                if direct
                else {"operation_id": operation.operation_id, "payload": payload},
            )

    with caplog.at_level(logging.ERROR, logger="jacobian.mcp.tools"):
        result = asyncio.run(scenario())
    assert result.is_error
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    expected: dict[str, object] = {
        "code": "OPERATION_TIMEOUT",
        "message": "operation deadline expired",
        "operation_id": operation.operation_id,
        "stage": "operation_execution",
        "timeout_owner": outer_owner.value if uses_outer_owner else "operation_wall",
        "deterministic_work_remains_fixed": False,
    }
    if retains_context:
        expected["configured_seconds"] = wall_seconds
        expected["adjustable_field_path"] = ["resource_budget", "wall_seconds"]
    assert json.loads(text[text.index("{") :]) == expected
    assert completed_calls == 1
    assert not [
        record
        for record in caplog.records
        if record.name == "jacobian.mcp.tools" and record.exc_info
    ]
