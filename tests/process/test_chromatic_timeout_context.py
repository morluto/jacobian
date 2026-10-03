"""Source-authored timeout context survives a checked production worker."""

import json

import pytest
from tests.support.chromatic_timeout import (
    chromatic_timeout_request,
    chromatic_timeout_worker_output,
    chromatic_worker_timeout,
    patch_chromatic_clock,
)

from jacobian._execution import (
    BackendFailureReason,
    OperationBackendError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    OperationPhaseLease,
    TimeoutOwner,
    bind_request_deadline,
    lease_operation_phases,
    request_checkpoint,
    request_execution,
    request_stage,
    require_execution_deadline,
)
from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
    ChromaticBipartitionResult,
    find_chromatic_bipartition,
)
from jacobian.process import BoundedProcessResult


@pytest.mark.parametrize("wall_seconds", [1, 5, 120])
def test_chromatic_kernel_context_survives_checked_parent(
    monkeypatch: pytest.MonkeyPatch, wall_seconds: int
) -> None:
    request, output = chromatic_worker_timeout(monkeypatch, wall_seconds)

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        assert kwargs["stdout_limit"] == (
            process_owner._chromatic_bipartition_worker_stdout_limit(request)
        )
        return BoundedProcessResult(0, output, b"", False, False, False)

    monkeypatch.setattr(process_owner, "run_bounded_process", complete)
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        find_chromatic_bipartition(request)
    assert caught.value.configured_seconds == wall_seconds
    assert caught.value.adjustable_field_path == ("resource_budget", "wall_seconds")
    assert caught.value.timeout_owner == "operation_wall"
    assert caught.value.elapsed_seconds is None
    assert caught.value.maximum_seconds is None

    assert json.loads(output) == {
        "kind": "execution_error",
        "stage": "operation_execution",
        "reason": "timeout",
        "configured_seconds": wall_seconds,
        "adjustable_field_path": ["resource_budget", "wall_seconds"],
    }
    # This is the largest actual source-authored frame, even at the legal wall
    # maximum. It fits the owner's smallest result envelope without more stdout.
    empty = ChromaticBipartitionRequest.model_validate(
        {"graph": {"vertices": [], "edges": []}, "s": 1, "t": 1}
    )
    assert len(output) <= 160
    assert len(output) <= process_owner._chromatic_bipartition_worker_stdout_limit(
        empty
    )
    assert len(output) <= process_owner._chromatic_bipartition_worker_stdout_limit(
        request
    )


def test_inherited_worker_checkpoint_does_not_claim_the_source_wall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, output = chromatic_worker_timeout(
        monkeypatch, wall_seconds=120, expired_parent_deadline=True
    )
    monkeypatch.setattr(
        process_owner,
        "run_bounded_process",
        lambda *args, **kwargs: BoundedProcessResult(
            0, output, b"", False, False, False
        ),
    )
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        find_chromatic_bipartition(request)
    assert caught.value.configured_seconds is None
    assert caught.value.adjustable_field_path is None
    assert json.loads(output) == {
        "kind": "execution_error",
        "stage": "operation_execution",
        "reason": "timeout",
    }


@pytest.mark.parametrize(
    ("outer_deadline", "enclosing_deadline", "retains_context", "uses_outer_owner"),
    [
        (None, None, True, False),
        (101, None, False, True),
        (105, None, False, True),
        (110, None, True, False),
        (None, 101, False, False),
        (None, 105, False, False),
        (None, 110, True, False),
        (110, 101, False, False),
        (101, 110, False, True),
        (110, 105, False, False),
        (101, 101, False, True),
        (105, 105, False, True),
        (110, 110, True, False),
    ],
)
@pytest.mark.parametrize("supervisor_timeout", [False, True])
@pytest.mark.parametrize(
    "outer_owner", [TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.BACKEND_TIMEOUT]
)
def test_only_a_strictly_limiting_source_wall_retains_recovery_context(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
    supervisor_timeout: bool,
    uses_outer_owner: bool,
    outer_owner: TimeoutOwner,
) -> None:
    clock = [100.0]
    patch_chromatic_clock(monkeypatch, clock)
    request = chromatic_timeout_request()
    completed_calls = 0

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        nonlocal completed_calls
        completed_calls += 1
        input_bytes = kwargs["input_bytes"]
        assert isinstance(input_bytes, bytes)
        # A fresh child start and the mandatory phase reserves shorten the
        # solver window even when the original source wall owns the deadline.
        clock[0] = 100.5
        output, solver_timeouts = chromatic_timeout_worker_output(
            monkeypatch, input_bytes
        )
        assert len(solver_timeouts) == 1
        assert 0 < solver_timeouts[0] < 4500
        assert json.loads(output)["configured_seconds"] == 5
        assert len(output) <= 160
        return BoundedProcessResult(0, output, b"", False, False, supervisor_timeout)

    monkeypatch.setattr(process_owner, "run_bounded_process", complete)
    with request_execution(
        100, outer_deadline=outer_deadline, timeout_owner=outer_owner
    ):
        if enclosing_deadline is not None:
            bind_request_deadline(enclosing_deadline)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            find_chromatic_bipartition(request)
    assert completed_calls == 1
    assert caught.value.timeout_owner is (
        outer_owner if uses_outer_owner else TimeoutOwner.OPERATION_WALL
    )
    assert caught.value.stage is OperationExecutionStage.OPERATION_EXECUTION
    assert caught.value.configured_seconds == (5 if retains_context else None)
    assert caught.value.adjustable_field_path == (
        ("resource_budget", "wall_seconds") if retains_context else None
    )
    assert caught.value.elapsed_seconds is None
    assert caught.value.maximum_seconds is None


@pytest.mark.parametrize(
    ("outer_deadline", "enclosing_deadline", "retains_context", "uses_outer_owner"),
    [
        (None, None, True, False),
        (104.995, None, False, True),
        (105, None, False, True),
        (110, None, True, False),
        (None, 104.995, False, False),
        (110, 104.995, False, False),
        (104.995, 110, False, True),
        (105, 105, False, True),
    ],
)
@pytest.mark.parametrize(
    "outer_owner", [TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.BACKEND_TIMEOUT]
)
def test_exhausted_phase_reserve_preserves_only_owned_configuration(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
    uses_outer_owner: bool,
    outer_owner: TimeoutOwner,
) -> None:
    patch_chromatic_clock(monkeypatch, [104.99])
    request = chromatic_timeout_request()

    def unexpected_worker(*args: object, **kwargs: object) -> BoundedProcessResult:
        raise AssertionError("an exhausted backend reserve must prevent worker launch")

    monkeypatch.setattr(process_owner, "run_bounded_process", unexpected_worker)
    with request_execution(
        100, outer_deadline=outer_deadline, timeout_owner=outer_owner
    ):
        if enclosing_deadline is not None:
            bind_request_deadline(enclosing_deadline)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            find_chromatic_bipartition(request)
    assert caught.value.configured_seconds == (5 if retains_context else None)
    assert caught.value.timeout_owner is (
        outer_owner if uses_outer_owner else TimeoutOwner.OPERATION_WALL
    )
    assert caught.value.stage is OperationExecutionStage.OPERATION_EXECUTION
    assert caught.value.elapsed_seconds == pytest.approx(4.99)
    # The shared lease failure has no field path; filtering must not invent one.
    assert caught.value.adjustable_field_path is None


@pytest.mark.parametrize(
    ("outer_deadline", "enclosing_deadline", "retains_context", "uses_outer_owner"),
    [
        (None, None, True, False),
        (101, None, False, True),
        (110, 101, False, False),
        (101, 110, False, True),
        (105, 105, False, True),
        (110, 110, True, False),
    ],
)
@pytest.mark.parametrize(
    "outer_owner", [TimeoutOwner.CALLER_DEADLINE, TimeoutOwner.BACKEND_TIMEOUT]
)
@pytest.mark.parametrize(
    "boundary",
    [
        "before_worker",
        "after_worker",
        "worker_checkpoint",
        "decode_checkpoint",
        "validation_checkpoint",
    ],
)
def test_local_timeout_paths_preserve_captured_owner_and_original_metadata(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
    uses_outer_owner: bool,
    outer_owner: TimeoutOwner,
    boundary: str,
) -> None:
    clock = [100.0]
    patch_chromatic_clock(monkeypatch, clock)
    request = ChromaticBipartitionRequest.model_validate(
        {
            "graph": {"vertices": [], "edges": []},
            "s": 1,
            "t": 1,
            "resource_budget": {"wall_seconds": 5},
        }
    )
    result = ChromaticBipartitionResult(
        graph=request.graph, s=1, t=1, status="NO_SPLIT", checked_partitions=0
    )
    completed_calls = 0

    def lease(
        wall_seconds: float, *, admitted_response_bytes: int, validation_work: int
    ) -> OperationPhaseLease:
        actual = lease_operation_phases(
            wall_seconds,
            admitted_response_bytes=admitted_response_bytes,
            validation_work=validation_work,
        )
        if boundary == "before_worker":
            clock[0] = actual.backend_deadline
        return actual

    def complete(*args: object, **kwargs: object) -> BoundedProcessResult:
        nonlocal completed_calls
        completed_calls += 1
        if boundary == "worker_checkpoint":
            clock[0] = 111
        return BoundedProcessResult(
            0,
            encode_worker_result_frame(result.model_dump(mode="json")),
            b"",
            False,
            False,
            False,
        )

    def checkpoint(stage: str) -> None:
        if boundary == "validation_checkpoint" and "response validation" in stage:
            clock[0] = 111
        request_checkpoint(stage)
        if boundary == "after_worker" and stage == "after chromatic bipartition worker":
            clock[0] = 111

    def require_deadline(deadline: float) -> None:
        if boundary == "decode_checkpoint":
            clock[0] = 111
        require_execution_deadline(deadline)

    monkeypatch.setattr(process_owner, "lease_operation_phases", lease)
    monkeypatch.setattr(process_owner, "run_bounded_process", complete)
    monkeypatch.setattr(process_owner, "request_checkpoint", checkpoint)
    monkeypatch.setattr(process_owner, "require_execution_deadline", require_deadline)
    with request_execution(
        100, outer_deadline=outer_deadline, timeout_owner=outer_owner
    ):
        if enclosing_deadline is not None:
            bind_request_deadline(enclosing_deadline)
        with (
            request_stage(OperationExecutionStage.RESULT_PROJECTION),
            pytest.raises(OperationExecutionTimeoutError) as caught,
        ):
            find_chromatic_bipartition(request)
    assert completed_calls == (0 if boundary == "before_worker" else 1)
    assert caught.value.timeout_owner is (
        outer_owner if uses_outer_owner else TimeoutOwner.OPERATION_WALL
    )
    checkpoint_timeout = boundary.endswith("checkpoint")
    assert caught.value.stage is (
        OperationExecutionStage.RESULT_PROJECTION
        if checkpoint_timeout
        else OperationExecutionStage.OPERATION_EXECUTION
    )
    assert caught.value.elapsed_seconds == (11 if checkpoint_timeout else None)
    assert caught.value.configured_seconds == (
        5 if retains_context and not checkpoint_timeout else None
    )
    assert caught.value.adjustable_field_path == (
        ("resource_budget", "wall_seconds")
        if retains_context and not checkpoint_timeout
        else None
    )
    assert caught.value.maximum_seconds is None


def test_chromatic_worker_startup_oserror_keeps_backend_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = chromatic_timeout_request()
    failure = OSError("private startup failure")

    def fail(*args: object, **kwargs: object) -> BoundedProcessResult:
        raise failure

    monkeypatch.setattr(process_owner, "run_bounded_process", fail)
    with pytest.raises(OperationBackendError) as caught:
        find_chromatic_bipartition(request)
    assert caught.value.reason is BackendFailureReason.STARTUP
    assert caught.value.__cause__ is failure
