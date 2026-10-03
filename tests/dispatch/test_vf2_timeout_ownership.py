"""VF2's local timing policy agrees across native and operation-ID execution."""

from __future__ import annotations

import time
from typing import Any

import pytest
from tests.support.vf2_timeout import VF2_PAYLOAD, TimeoutPath, patch_vf2_timeout

from jacobian import process
from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    TimeoutOwner,
    bind_request_deadline,
    current_request_execution,
    request_execution,
)
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.graphs.isomorphism._models import GraphIsomorphismRequest
from jacobian.math.graphs.isomorphism._tools import TOOLS
from jacobian.math.graphs.isomorphism._vf2_process import (
    decide_graph_isomorphism,
    verify_graph_isomorphism,
)
from jacobian.process import BoundedProcessResult

_OPERATION = TOOLS[0]
_CATALOG = Catalog((_OPERATION,))


@pytest.mark.parametrize("native", [False, True], ids=["dispatch", "native"])
@pytest.mark.parametrize("path", ["reserve", "prelaunch", "worker", "delivery"])
@pytest.mark.parametrize(
    ("outer", "bound", "outer_owner", "owner", "source_owns"),
    [
        (190.0, None, TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.OPERATION_WALL, True),
        (
            130.0,
            None,
            TimeoutOwner.CALLER_DEADLINE,
            TimeoutOwner.CALLER_DEADLINE,
            False,
        ),
        (
            130.0,
            None,
            TimeoutOwner.BACKEND_TIMEOUT,
            TimeoutOwner.BACKEND_TIMEOUT,
            False,
        ),
        (None, 130.0, TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.OPERATION_WALL, False),
        (
            160.0,
            None,
            TimeoutOwner.CALLER_DEADLINE,
            TimeoutOwner.CALLER_DEADLINE,
            False,
        ),
        (None, 160.0, TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.OPERATION_WALL, False),
        (
            140.0,
            130.0,
            TimeoutOwner.CALLER_DEADLINE,
            TimeoutOwner.OPERATION_WALL,
            False,
        ),
        (
            130.0,
            140.0,
            TimeoutOwner.BACKEND_TIMEOUT,
            TimeoutOwner.BACKEND_TIMEOUT,
            False,
        ),
        (
            130.0,
            130.0,
            TimeoutOwner.CALLER_DEADLINE,
            TimeoutOwner.CALLER_DEADLINE,
            False,
        ),
        (None, 190.0, TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.OPERATION_WALL, True),
    ],
    ids=[
        "source",
        "caller",
        "backend",
        "prior-operation",
        "source-caller-tie",
        "source-bound-tie",
        "prior-earlier",
        "outer-earlier",
        "outer-bound-tie",
        "source-before-prior",
    ],
)
def test_vf2_timeout_retains_limiting_owner(
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    path: TimeoutPath,
    outer: float | None,
    bound: float | None,
    outer_owner: TimeoutOwner,
    owner: TimeoutOwner,
    source_owns: bool,
) -> None:
    cutoff = min(value for value in (160.0, outer, bound) if value is not None)
    clock = [cutoff - 0.001 if path == "reserve" else 100.0]
    trace = patch_vf2_timeout(monkeypatch, clock, path)
    request = GraphIsomorphismRequest.model_validate(VF2_PAYLOAD)
    with request_execution(
        100.0, outer_deadline=outer, timeout_owner=outer_owner
    ) as parent:
        if bound is not None:
            bind_request_deadline(bound)
        inherited_deadline = parent.deadline
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            if native:
                decide_graph_isomorphism(request)
            else:
                invoke_operation(_OPERATION.operation_id, VF2_PAYLOAD, _CATALOG)
        assert caught.value.timeout_owner is owner
        assert caught.value.stage is OperationExecutionStage.OPERATION_EXECUTION
        has_source_configuration = source_owns and (
            path in {"reserve", "prelaunch"}
            or (path == "delivery" and bound == 190.0 and native)
        )
        assert caught.value.configured_seconds == (
            60 if has_source_configuration else None
        )
        assert caught.value.adjustable_field_path is None
        assert trace == (
            ["lease", "worker", "delivery"]
            if path == "delivery"
            else ["lease", "worker"]
            if path == "worker"
            else ["lease"]
        )
        assert current_request_execution() is parent
        if not native:
            assert parent.deadline == inherited_deadline
    assert current_request_execution() is None


@pytest.mark.parametrize("native", [False, True], ids=["dispatch", "native"])
def test_vf2_preserves_an_explicit_earlier_backend_timeout(
    monkeypatch: pytest.MonkeyPatch, native: bool
) -> None:
    failure = OperationExecutionTimeoutError(
        "backend's independent allowance expired",
        timeout_owner=TimeoutOwner.BACKEND_TIMEOUT,
        configured_seconds=0.01,
        adjustable_field_path=("backend", "timeout_seconds"),
    )

    def fail(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        raise failure

    monkeypatch.setattr(process, "run_bounded_process", fail)
    now = time.monotonic()
    with request_execution(now, outer_deadline=now + 30):
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            if native:
                decide_graph_isomorphism(
                    GraphIsomorphismRequest.model_validate(VF2_PAYLOAD)
                )
            else:
                invoke_operation(_OPERATION.operation_id, VF2_PAYLOAD, _CATALOG)
        assert caught.value is failure
        assert failure.timeout_owner is TimeoutOwner.BACKEND_TIMEOUT
        assert failure.configured_seconds == 0.01
        assert failure.adjustable_field_path == ("backend", "timeout_seconds")


def test_vf2_supervisor_cancellation_keeps_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            None, b"", b"", False, False, True, True
        ),
    )
    with (
        request_execution(time.monotonic()),
        pytest.raises(OperationExecutionCancelledError),
    ):
        invoke_operation(_OPERATION.operation_id, VF2_PAYLOAD, _CATALOG)


def test_vf2_keeps_ordinary_startup_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = OSError("worker could not start")

    def fail(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        raise failure

    monkeypatch.setattr(process, "run_bounded_process", fail)
    with pytest.raises(OperationBackendError) as caught:
        invoke_operation(_OPERATION.operation_id, VF2_PAYLOAD, _CATALOG)
    assert caught.value.reason is BackendFailureReason.STARTUP
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize("example", [0, 1], ids=["isomorphic", "non-isomorphic"])
def test_real_vf2_native_and_dispatch_results_match(example: int) -> None:
    payload = _OPERATION.examples[example].input
    request = GraphIsomorphismRequest.model_validate(payload)
    now = time.monotonic()
    with request_execution(now - 10, outer_deadline=now + 30):
        native = decide_graph_isomorphism(request)
    with request_execution(now - 10, outer_deadline=now + 30) as parent:
        projected = invoke_operation(_OPERATION.operation_id, payload, _CATALOG)
        assert current_request_execution() is parent
        assert parent.deadline == now + 30
    assert projected.output == native.model_dump(mode="json")
    assert native.status == ("ISOMORPHIC" if example == 0 else "NOT_ISOMORPHIC")
    assert verify_graph_isomorphism(native) is (example == 0)
