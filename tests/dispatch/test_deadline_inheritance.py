"""Nested dispatch retains timing while owning fresh delivery context."""

from __future__ import annotations

import time
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from jacobian import _execution, dispatch
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    TimeoutOwner,
    bind_request_deadline,
    current_request_cancellation,
    current_request_execution,
    request_cancellation,
    request_execution,
)
from jacobian._models import StrictModel
from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import execute_operation, invoke_operation
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    CHROMATIC_BIPARTITION_OPERATION,
    ChromaticBipartitionRequest,
    find_chromatic_bipartition,
)

_OPERATION_ID = CHROMATIC_BIPARTITION_OPERATION.operation_id
_PAYLOAD: dict[str, Any] = {
    "graph": {"vertices": [], "edges": []},
    "s": 1,
    "t": 1,
    "resource_budget": {"wall_seconds": 120},
}
_CATALOG = Catalog((CHROMATIC_BIPARTITION_OPERATION,))


@pytest.mark.parametrize("bound", [False, True], ids=["outer", "operation"])
def test_expired_native_and_dispatched_bounds_both_refuse(bound: bool) -> None:
    request = ChromaticBipartitionRequest.model_validate(_PAYLOAD)
    now = time.monotonic()
    for native in (True, False):
        with request_execution(
            now, outer_deadline=None if bound else now - 1
        ) as parent:
            if bound:
                bind_request_deadline(now - 1)
            with pytest.raises(OperationExecutionTimeoutError) as error:
                if native:
                    find_chromatic_bipartition(request)
                else:
                    invoke_operation(_OPERATION_ID, _PAYLOAD, _CATALOG)
            if not native:
                assert error.value.stage is OperationExecutionStage.REQUEST_PARSING
                assert error.value.timeout_owner is (
                    TimeoutOwner.OPERATION_WALL
                    if bound
                    else TimeoutOwner.CALLER_DEADLINE
                )
            assert current_request_execution() is parent
            assert parent.deadline == now - 1
    assert current_request_execution() is None


@pytest.mark.parametrize(
    ("outer_offset", "bound_offset", "owner"),
    [
        (60, None, TimeoutOwner.BACKEND_TIMEOUT),
        (None, 60, TimeoutOwner.OPERATION_WALL),
        (60, 60, TimeoutOwner.BACKEND_TIMEOUT),
        (60, 90, TimeoutOwner.BACKEND_TIMEOUT),
        (90, 60, TimeoutOwner.OPERATION_WALL),
    ],
    ids=["outer", "bound", "tied", "outer-earlier", "bound-earlier"],
)
def test_live_bound_reaches_real_owner_without_widening(
    outer_offset: int | None, bound_offset: int | None, owner: TimeoutOwner
) -> None:
    now = time.monotonic()
    original_start = now - 10
    outer = None if outer_offset is None else now + outer_offset
    with request_execution(
        original_start, outer_deadline=outer, timeout_owner=TimeoutOwner.BACKEND_TIMEOUT
    ) as parent:
        if bound_offset is not None:
            bind_request_deadline(now + bound_offset)
        inherited_deadline = parent.deadline

        def project(
            _operation_id: str, result: StrictModel, started: float
        ) -> StrictModel:
            child = current_request_execution()
            assert child is not None and child is not parent
            assert child.started_at == original_start
            assert child.outer_deadline == inherited_deadline
            assert child.deadline == inherited_deadline
            assert child.timeout_owner is owner
            assert started >= now
            return result

        result = execute_operation(_OPERATION_ID, _PAYLOAD, _CATALOG, projector=project)
        assert result.model_dump(mode="json")["status"] == "NO_SPLIT"
        assert current_request_execution() is parent
        assert parent.deadline == inherited_deadline
    assert current_request_execution() is None


@pytest.mark.parametrize(
    ("outer_offset", "bound_offset", "owner"),
    [
        (-1, -1, TimeoutOwner.BACKEND_TIMEOUT),
        (-1, -2, TimeoutOwner.OPERATION_WALL),
        (-2, -1, TimeoutOwner.BACKEND_TIMEOUT),
    ],
    ids=["tied", "bound-earlier", "outer-earlier"],
)
def test_expired_dispatch_preserves_winning_owner_and_parent(
    outer_offset: int, bound_offset: int, owner: TimeoutOwner
) -> None:
    now = time.monotonic()
    with request_execution(
        now - 10,
        outer_deadline=now + outer_offset,
        timeout_owner=TimeoutOwner.BACKEND_TIMEOUT,
    ) as parent:
        bind_request_deadline(now + bound_offset)
        inherited_deadline = parent.deadline
        with pytest.raises(OperationExecutionTimeoutError) as error:
            invoke_operation(_OPERATION_ID, _PAYLOAD, _CATALOG)
        assert error.value.timeout_owner is owner
        assert error.value.elapsed_seconds is not None
        assert error.value.elapsed_seconds >= 10
        assert current_request_execution() is parent
        assert parent.deadline == inherited_deadline
    assert current_request_execution() is None


def test_original_start_without_outer_bound_does_not_refresh_source_budget() -> None:
    request = ChromaticBipartitionRequest.model_validate(_PAYLOAD)
    original_start = time.monotonic() - 130
    for native in (True, False):
        with request_execution(original_start) as parent:
            with pytest.raises(OperationExecutionTimeoutError) as error:
                if native:
                    find_chromatic_bipartition(request)
                else:
                    invoke_operation(_OPERATION_ID, _PAYLOAD, _CATALOG)
            assert error.value.timeout_owner is TimeoutOwner.OPERATION_WALL
            assert current_request_execution() is parent
            if not native:
                assert parent.deadline is None
    assert current_request_execution() is None


def test_runtime_still_measures_only_this_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = time.monotonic()
    clock = iter((now, now + 0.125))
    # Replace only dispatch's clock reference, never the shared event-loop clock.
    monkeypatch.setattr(
        dispatch, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )
    with request_execution(now - 10, outer_deadline=now + 60) as parent:
        result = invoke_operation(_OPERATION_ID, _PAYLOAD, _CATALOG)
        assert result.output["status"] == "NO_SPLIT"
        assert result.runtime_ms == 125
        assert current_request_execution() is parent
        assert parent.deadline == now + 60


@pytest.mark.parametrize(
    ("child_deadline", "observed_at", "owner"),
    [
        (102.0, 103.0, TimeoutOwner.OPERATION_WALL),
        (102.0, 106.0, TimeoutOwner.OPERATION_WALL),
        (105.0, 106.0, TimeoutOwner.CALLER_DEADLINE),
        (108.0, 109.0, TimeoutOwner.CALLER_DEADLINE),
    ],
    ids=["child-first", "child-observed-late", "tied", "outer-first"],
)
def test_child_local_deadline_keeps_its_winning_owner(
    monkeypatch: pytest.MonkeyPatch,
    child_deadline: float,
    observed_at: float,
    owner: TimeoutOwner,
) -> None:
    catalog = Catalog.open()
    clock = 100.0
    local_time = SimpleNamespace(monotonic=lambda: clock)
    monkeypatch.setattr(_execution, "time", local_time)
    monkeypatch.setattr(dispatch, "time", local_time)

    def project(
        _operation_id: str, result: StrictModel, _started: float
    ) -> StrictModel:
        nonlocal clock
        assert result.model_dump(mode="json") == {
            "determinant": {"num": "-2", "den": "1"}
        }
        child = current_request_execution()
        assert child is not None and child.outer_deadline == 105.0
        assert child.deadline == 105.0
        bind_request_deadline(child_deadline)
        assert child.deadline == min(child_deadline, 105.0)
        clock = observed_at
        return result

    with request_execution(100.0, outer_deadline=105.0) as parent:
        with pytest.raises(OperationExecutionTimeoutError) as error:
            execute_operation(
                "matrix.determinant.compute",
                {
                    "matrix": {
                        "domain": "QQ",
                        "entries": [
                            [{"num": "1", "den": "1"}, {"num": "2", "den": "1"}],
                            [{"num": "3", "den": "1"}, {"num": "4", "den": "1"}],
                        ],
                    }
                },
                catalog,
                projector=project,
            )
        assert error.value.timeout_owner is owner
        assert error.value.stage is OperationExecutionStage.RESULT_PROJECTION
        assert error.value.elapsed_seconds == observed_at - 100.0
        assert current_request_execution() is parent
        assert parent.deadline == 105.0
    assert current_request_execution() is None


class _Progress:
    def __init__(self) -> None:
        self.values: list[int] = []

    def report(
        self, progress: int, *, total: int | None = None, message: str | None = None
    ) -> None:
        self.values.append(progress)


@pytest.mark.parametrize("explicit", [False, True], ids=["omitted", "explicit"])
def test_child_keeps_its_own_progress_and_cancellation_arguments(
    explicit: bool,
) -> None:
    parent_signal = Event()
    parent_signal.set()
    child_signal = Event()
    parent_progress = _Progress()
    child_progress = _Progress() if explicit else None
    now = time.monotonic()
    with request_execution(
        now - 10,
        outer_deadline=now + 60,
        cancellation_signal=parent_signal,
        progress_sink=parent_progress,
    ) as parent:

        def project(
            _operation_id: str, result: StrictModel, _started: float
        ) -> StrictModel:
            child = current_request_execution()
            assert child is not None and child is not parent
            assert child.cancellation_signal is (child_signal if explicit else None)
            assert child.progress_sink is child_progress
            return result

        result = execute_operation(
            _OPERATION_ID,
            _PAYLOAD,
            _CATALOG,
            projector=project,
            cancellation_signal=child_signal if explicit else None,
            progress_sink=child_progress,
        )
        assert result.model_dump(mode="json")["status"] == "NO_SPLIT"
        assert parent_progress.values == []
        if child_progress is not None:
            assert child_progress.values == [0, 0]
        assert current_request_execution() is parent
        assert parent.deadline == now + 60
        assert current_request_cancellation() is None
    assert current_request_execution() is None


@pytest.mark.parametrize("explicit", [False, True], ids=["legacy", "explicit-override"])
def test_legacy_cancellation_and_explicit_override_are_unchanged(
    explicit: bool,
) -> None:
    legacy_signal = Event()
    legacy_signal.set()
    child_signal = Event()
    now = time.monotonic()
    with request_execution(now, outer_deadline=now + 60) as parent:
        bind_request_deadline(now + 30)
        with request_cancellation(legacy_signal):
            if explicit:
                result = execute_operation(
                    _OPERATION_ID,
                    _PAYLOAD,
                    _CATALOG,
                    projector=lambda _id, result, _started: result,
                    cancellation_signal=child_signal,
                )
                assert result.model_dump(mode="json")["status"] == "NO_SPLIT"
            else:
                with pytest.raises(OperationExecutionCancelledError):
                    invoke_operation(_OPERATION_ID, _PAYLOAD, _CATALOG)
            assert current_request_cancellation() is legacy_signal
            assert current_request_execution() is parent
            assert parent.deadline == now + 30
    assert current_request_cancellation() is None
    assert current_request_execution() is None
