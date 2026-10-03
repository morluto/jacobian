"""Private IPC preserves classifications and rejects malformed error branches."""

import json
from collections.abc import Iterator
from typing import cast

import pytest

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
    TimeoutOwner,
)
from jacobian._worker_errors import (
    decode_worker_execution_error,
    worker_execution_errors,
)


@pytest.mark.parametrize(
    "error",
    [
        *[OperationBackendError(reason) for reason in BackendFailureReason],
        *[OperationResourceExhaustedError(resource) for resource in ExecutionResource],
        OperationExecutionCancelledError("private marker"),
        OperationExecutionTimeoutError(
            "private marker", stage=OperationExecutionStage.RESULT_PROJECTION
        ),
    ],
)
@pytest.mark.parametrize("preserve_context", [False, True])
def test_private_error_roundtrip(
    capsys: pytest.CaptureFixture[str],
    error: OperationBackendError
    | OperationResourceExhaustedError
    | OperationExecutionTimeoutError
    | OperationExecutionCancelledError,
    preserve_context: bool,
) -> None:
    with worker_execution_errors(preserve_timeout_context=preserve_context):
        raise error from RuntimeError("private cause marker")
    branch = json.loads(capsys.readouterr().out)
    with pytest.raises(type(error)) as caught:
        decode_worker_execution_error(branch)
    assert caught.value.stage == error.stage
    if isinstance(error, OperationBackendError):
        assert isinstance(caught.value, OperationBackendError)
        assert caught.value.reason == error.reason
        assert "private cause marker" in "".join(caught.value.__notes__)
    elif isinstance(error, OperationResourceExhaustedError):
        assert isinstance(caught.value, OperationResourceExhaustedError)
        assert caught.value.resource == error.resource
    assert "private" not in str(caught.value)


@pytest.mark.parametrize(
    "branch",
    [
        {},
        {"stage": None},
        {"stage": []},
        {"stage": "result_projection", "reason": "unknown"},
        {"stage": "operation_execution", "resource": "time"},
        {"stage": "operation_execution", "resource": "work", "extra": True},
        {"stage": "operation_execution", "reason": "startup", "diagnostic": 1},
        {"stage": "operation_execution", "reason": "startup", "diagnostic": "x" * 1025},
    ],
)
def test_malformed_execution_branch_is_backend_failure(
    branch: dict[str, object],
) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_worker_execution_error({"kind": "execution_error", **branch})
    assert caught.value.reason == "malformed_response"


def test_mathematical_unknown_is_not_an_execution_branch() -> None:
    decode_worker_execution_error({"outcome": "UNKNOWN", "detail": "inconclusive"})


@pytest.mark.parametrize(
    ("configured", "path"),
    [
        (None, None),
        (0, None),
        (0.125, None),
        (None, ("requests", 0, "wall_seconds")),
        (120, ("resource_budget", "wall_seconds")),
    ],
)
def test_opted_in_timeout_context_roundtrip(
    capsys: pytest.CaptureFixture[str],
    configured: float | None,
    path: tuple[str | int, ...] | None,
) -> None:
    with worker_execution_errors(preserve_timeout_context=True):
        raise OperationExecutionTimeoutError(
            "private marker",
            stage=OperationExecutionStage.RESULT_PROJECTION,
            configured_seconds=configured,
            adjustable_field_path=path,
            # The repair deliberately transports only the source-authored pair.
            timeout_owner=TimeoutOwner.BACKEND_TIMEOUT,
            elapsed_seconds=7,
            maximum_seconds=120,
            deterministic_work_remains_fixed=True,
        )
    branch = json.loads(capsys.readouterr().out)
    expected: dict[str, object] = {
        "kind": "execution_error",
        "stage": "result_projection",
        "reason": "timeout",
    }
    if configured is not None:
        expected["configured_seconds"] = configured
    if path is not None:
        expected["adjustable_field_path"] = list(path)
    assert branch == expected
    with pytest.raises(OperationExecutionTimeoutError) as caught:
        decode_worker_execution_error(branch)
    assert caught.value.configured_seconds == configured
    assert caught.value.adjustable_field_path == path
    assert caught.value.stage == OperationExecutionStage.RESULT_PROJECTION
    assert caught.value.timeout_owner == TimeoutOwner.OPERATION_WALL
    assert caught.value.elapsed_seconds is None
    assert caught.value.maximum_seconds is None
    assert caught.value.deterministic_work_remains_fixed is False
    assert "private" not in str(caught.value)


def test_other_workers_keep_their_compact_timeout_branch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with worker_execution_errors():
        raise OperationExecutionTimeoutError(
            "private marker",
            configured_seconds=120,
            adjustable_field_path=("resource_budget", "wall_seconds"),
        )
    output = capsys.readouterr().out
    assert output == (
        '{"kind":"execution_error","stage":"operation_execution","reason":"timeout"}\n'
    )


@pytest.mark.parametrize(
    "context",
    [
        {"configured_seconds": None},
        {"configured_seconds": True},
        {"configured_seconds": -1},
        {"configured_seconds": float("nan")},
        {"configured_seconds": float("inf")},
        {"configured_seconds": 10**1000},
        {"configured_seconds": "5"},
        {"configured_seconds": []},
        *[
            {"adjustable_field_path": value}
            for value in (
                None,
                "wall_seconds",
                [],
                [None],
                [True],
                [-1],
                [2**53],
                [0.5],
                [""],
                ["x" * 129],
                ["x"] * 33,
            )
        ],
        {"configured_seconds": 5, "extra": True},
        {"timeout_owner": "backend_timeout"},
        {"elapsed_seconds": 5},
        {"maximum_seconds": 120},
        {"deterministic_work_remains_fixed": True},
    ],
)
def test_malformed_timeout_context_is_backend_failure(
    context: dict[str, object],
) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_worker_execution_error(
            {
                "kind": "execution_error",
                "stage": "operation_execution",
                "reason": "timeout",
                **context,
            }
        )
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "branch",
    [{"reason": "cancelled"}, {"resource": "work"}],
)
def test_timeout_fields_do_not_expand_other_error_branches(
    branch: dict[str, str],
) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_worker_execution_error(
            {
                "kind": "execution_error",
                "stage": "operation_execution",
                "configured_seconds": 5,
                **branch,
            }
        )
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


def test_opted_in_encoder_refuses_invalid_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        pytest.raises(ValueError),
        worker_execution_errors(preserve_timeout_context=True),
    ):
        raise OperationExecutionTimeoutError(
            "private marker", configured_seconds=float("inf")
        )
    assert capsys.readouterr().out == ""


class _IntegerHooks(int):
    def __lt__(self, other: object) -> bool:
        raise AssertionError("integer comparison hook ran")

    def __ge__(self, other: object) -> bool:
        raise AssertionError("integer comparison hook ran")


class _FloatHooks(float):
    def __lt__(self, other: object) -> bool:
        raise AssertionError("float comparison hook ran")


class _StringHooks(str):
    def __len__(self) -> int:
        raise AssertionError("string length hook ran")


class _ListHooks(list[object]):
    def __len__(self) -> int:
        raise AssertionError("list length hook ran")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("list iteration hook ran")


class _TupleHooks(tuple[object, ...]):
    def __len__(self) -> int:
        raise AssertionError("tuple length hook ran")

    def __iter__(self) -> Iterator[object]:
        raise AssertionError("tuple iteration hook ran")


class _IterableHooks:
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("iterable hook ran")


@pytest.mark.parametrize(
    "context",
    [
        {"configured_seconds": _IntegerHooks(5)},
        {"configured_seconds": _FloatHooks(5)},
        {"adjustable_field_path": _ListHooks(["wall_seconds"])},
        {"adjustable_field_path": [_StringHooks("wall_seconds")]},
        {"adjustable_field_path": [_IntegerHooks(0)]},
    ],
)
def test_timeout_decode_refuses_subclasses_before_evaluating_hooks(
    context: dict[str, object],
) -> None:
    with pytest.raises(OperationBackendError) as caught:
        decode_worker_execution_error(
            {
                "kind": "execution_error",
                "stage": "operation_execution",
                "reason": "timeout",
                **context,
            }
        )
    assert caught.value.reason == BackendFailureReason.MALFORMED_RESPONSE


@pytest.mark.parametrize(
    "path",
    [
        _IterableHooks(),
        _TupleHooks(("wall_seconds",)),
        _ListHooks(["wall_seconds"]),
        "wall_seconds",
        (),
        ("wall_seconds",) * 33,
    ],
)
def test_timeout_encoder_checks_native_path_before_materializing_it(
    capsys: pytest.CaptureFixture[str], path: object
) -> None:
    with (
        pytest.raises(ValueError),
        worker_execution_errors(preserve_timeout_context=True),
    ):
        raise OperationExecutionTimeoutError(
            "private marker", adjustable_field_path=cast(tuple[str | int, ...], path)
        )
    assert capsys.readouterr().out == ""
