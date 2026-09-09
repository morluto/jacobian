"""Both SDK entry points expose sanitized failures and retain private tracebacks."""

import asyncio
import importlib
import json
import logging
from typing import Any, Never

import pytest
from mcp.types import TextContent

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.catalog.catalog import Catalog
from jacobian.math.logic import _sat
from jacobian.mcp.direct_tools import direct_operation_tools
from jacobian.mcp.runtime import AppState
from jacobian.mcp.server import _build_server
from mcp import Client


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("operation_id", ["sat.solve", "smt.solve", "smt.unsat_core"])
def test_sdk_solver_entry_points_accept_extended_wall_time(
    direct: bool, operation_id: str
) -> None:
    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )
    payload = {**operation.examples[0].input, "timeout_ms": 120_000}

    async def scenario() -> Any:
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                operation_id if direct else "math.run",
                payload
                if direct
                else {"operation_id": operation_id, "payload": payload},
            )

    result = asyncio.run(scenario())
    assert not result.is_error
    assert result.structured_content is not None
    output = (
        result.structured_content if direct else result.structured_content["output"]
    )
    assert output["source"]["timeout_ms"] == 120_000


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    ("error", "code", "tracebacks"),
    [
        (RuntimeError("private failure marker"), None, 1),
        (
            OperationBackendError(BackendFailureReason.INVALID_OUTPUT),
            "OPERATION_FAILED",
            1,
        ),
        (
            OperationResourceExhaustedError(ExecutionResource.WORK),
            "RESOURCE_EXHAUSTED",
            0,
        ),
        (
            OperationExecutionTimeoutError("private failure marker"),
            "OPERATION_TIMEOUT",
            0,
        ),
        (
            OperationExecutionCancelledError("private failure marker"),
            "OPERATION_CANCELLED",
            0,
        ),
    ],
)
def test_sdk_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    direct: bool,
    error: Exception,
    code: str | None,
    tracebacks: int,
) -> None:
    def fail(*args: object, **kwargs: object) -> Never:
        raise error

    monkeypatch.setattr(_sat, "_run_sat_worker", fail)
    catalog = Catalog.open()
    operation = catalog.operation("sat.solve")
    assert operation is not None
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    async def scenario() -> Any:
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                "sat.solve" if direct else "math.run",
                operation.examples[0].input
                if direct
                else {
                    "operation_id": "sat.solve",
                    "payload": operation.examples[0].input,
                },
            )

    with caplog.at_level(logging.ERROR, logger="jacobian.mcp.tools"):
        result = asyncio.run(scenario())
    assert result.is_error
    assert result.structured_content is None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "private failure marker" not in text
    if code:
        data = json.loads(text[text.index("{") :])
        assert data["code"] == code
        assert data["operation_id"] == "sat.solve"
        assert data["stage"] == "operation_execution"
        assert data["message"]
        if code == "RESOURCE_EXHAUSTED":
            assert data["resource"] == "work"
            assert "timeout_ms does not increase" in data["hint"]
        if code == "OPERATION_TIMEOUT":
            assert data["timeout_owner"] == "operation_wall"
            assert data["deterministic_work_remains_fixed"] is False
    else:
        assert text.endswith("operation execution failed")
    records = [
        r for r in caplog.records if r.name == "jacobian.mcp.tools" and r.exc_info
    ]
    assert len(records) == tracebacks
    if code is None:
        assert "private failure marker" in caplog.text


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("operation_id", ["sat.solve", "smt.solve", "smt.unsat_core"])
def test_success_and_resource_errors_keep_distinct_sdk_shapes(
    monkeypatch: pytest.MonkeyPatch, direct: bool, operation_id: str
) -> None:
    import jsonschema

    from jacobian.math.logic import _sat, _smt, _unsat_core

    catalog = Catalog.open()
    operation = catalog.operation(operation_id)
    assert operation is not None
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    async def invoke() -> Any:
        async with Client(server, raise_exceptions=False) as client:
            return await client.call_tool(
                operation_id if direct else "math.run",
                operation.examples[0].input
                if direct
                else {
                    "operation_id": operation_id,
                    "payload": operation.examples[0].input,
                },
            )

    success = asyncio.run(invoke())
    assert not success.is_error
    assert success.structured_content is not None
    output = (
        success.structured_content if direct else success.structured_content["output"]
    )
    jsonschema.validate(
        output, operation.result_type.model_json_schema(mode="serialization")
    )
    assert output["outcome"] in {"SAT", "UNSAT"}

    def exhausted(*args: object, **kwargs: object) -> Never:
        raise OperationResourceExhaustedError(ExecutionResource.WORK)

    owner = {"sat.solve": _sat, "smt.solve": _smt, "smt.unsat_core": _unsat_core}[
        operation_id
    ]
    monkeypatch.setattr(owner, "run_bounded_process", exhausted)
    failed = asyncio.run(invoke())
    assert failed.is_error
    assert failed.structured_content is None
    assert isinstance(failed.content[0], TextContent)
    text = failed.content[0].text
    diagnostic = json.loads(text[text.index("{") :])
    assert diagnostic["resource"] == "work"
    if operation_id == "smt.unsat_core":
        assert "rlimit within its admitted range" in diagnostic["hint"]
    else:
        assert "fixed work allowance" in diagnostic["hint"]


OWNERS = [
    ("sat.solve", "jacobian.math.logic._sat"),
    ("smt.solve", "jacobian.math.logic._smt"),
    ("smt.unsat_core", "jacobian.math.logic._unsat_core"),
    *[
        (name, "jacobian.math.graphs.optimization._finite_optimization")
        for name in (
            "graph.domination.minimum.compute",
            "graph.induced_bipartite.maximum.compute",
            "graph.induced_forest.maximum.compute",
            "graph.induced_tree.maximum.compute",
            "graph.matching.maximal.minimum.compute",
        )
    ],
    (
        "graph.invariant.clique_number.compute",
        "jacobian.math.graphs.optimization._invariants",
    ),
    (
        "graph.invariant.chromatic_number.compute",
        "jacobian.math.graphs.optimization._chromatic_number",
    ),
    (
        "graph.invariant.independence_number.compute",
        "jacobian.math.graphs._independence_z3",
    ),
    (
        "hypergraph.independence_number.compute",
        "jacobian.math.combinatorics.finite_structures.hypergraphs._independence_z3",
    ),
]


@pytest.mark.parametrize(("operation_id", "module"), OWNERS)
@pytest.mark.parametrize("direct", [False, True])
def test_each_owner_startup_failure_is_visible_to_the_model(
    monkeypatch: pytest.MonkeyPatch,
    operation_id: str,
    module: str,
    direct: bool,
) -> None:
    operation = Catalog.open().operation(operation_id)
    assert operation is not None
    catalog = Catalog((operation,))
    server = _build_server(
        state=AppState(operation_catalog=catalog),
        evaluation_tools=direct_operation_tools(catalog) if direct else (),
    )

    def fail(*args: object, **kwargs: object) -> Never:
        raise OSError("private worker startup marker")

    monkeypatch.setattr(importlib.import_module(module), "run_bounded_process", fail)

    async def scenario() -> None:
        async with Client(server, raise_exceptions=False) as client:
            result = await client.call_tool(
                operation_id if direct else "math.run",
                operation.examples[0].input
                if direct
                else {
                    "operation_id": operation_id,
                    "payload": operation.examples[0].input,
                },
            )
        assert result.is_error
        assert result.structured_content is None
        assert isinstance(result.content[0], TextContent)
        text = result.content[0].text
        assert "private worker startup marker" not in text
        data = json.loads(text[text.index("{") :])
        assert data["code"] == "OPERATION_FAILED"
        assert data["operation_id"] == operation_id

    asyncio.run(scenario())
