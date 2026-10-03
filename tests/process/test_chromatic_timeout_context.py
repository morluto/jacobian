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
    OperationExecutionTimeoutError,
    bind_request_deadline,
    request_execution,
)
from jacobian.math.graphs.optimization import (
    _chromatic_bipartition_process as process_owner,
)
from jacobian.math.graphs.optimization._chromatic_bipartition import (
    ChromaticBipartitionRequest,
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
    ("outer_deadline", "enclosing_deadline", "retains_context"),
    [
        (None, None, True),
        (101, None, False),
        (105, None, False),
        (110, None, True),
        (None, 101, False),
        (None, 105, False),
        (None, 110, True),
        (110, 101, False),
        (101, 110, False),
        (110, 105, False),
    ],
)
@pytest.mark.parametrize("supervisor_timeout", [False, True])
def test_only_a_strictly_limiting_source_wall_retains_recovery_context(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
    supervisor_timeout: bool,
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
    with request_execution(100, outer_deadline=outer_deadline):
        if enclosing_deadline is not None:
            bind_request_deadline(enclosing_deadline)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            find_chromatic_bipartition(request)
    assert completed_calls == 1
    assert caught.value.configured_seconds == (5 if retains_context else None)
    assert caught.value.adjustable_field_path == (
        ("resource_budget", "wall_seconds") if retains_context else None
    )
    assert caught.value.elapsed_seconds is None
    assert caught.value.maximum_seconds is None


@pytest.mark.parametrize(
    ("outer_deadline", "enclosing_deadline", "retains_context"),
    [
        (None, None, True),
        (104.995, None, False),
        (105, None, False),
        (110, None, True),
        (None, 104.995, False),
    ],
)
def test_exhausted_phase_reserve_preserves_only_owned_configuration(
    monkeypatch: pytest.MonkeyPatch,
    outer_deadline: float | None,
    enclosing_deadline: float | None,
    retains_context: bool,
) -> None:
    patch_chromatic_clock(monkeypatch, [104.99])
    request = chromatic_timeout_request()

    def unexpected_worker(*args: object, **kwargs: object) -> BoundedProcessResult:
        raise AssertionError("an exhausted backend reserve must prevent worker launch")

    monkeypatch.setattr(process_owner, "run_bounded_process", unexpected_worker)
    with request_execution(100, outer_deadline=outer_deadline):
        if enclosing_deadline is not None:
            bind_request_deadline(enclosing_deadline)
        with pytest.raises(OperationExecutionTimeoutError) as caught:
            find_chromatic_bipartition(request)
    assert caught.value.configured_seconds == (5 if retains_context else None)
    # The shared lease failure has no field path; filtering must not invent one.
    assert caught.value.adjustable_field_path is None
