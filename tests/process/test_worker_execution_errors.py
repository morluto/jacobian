"""Private IPC preserves classifications and rejects malformed error branches."""

import json

import pytest

from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionStage,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
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
def test_private_error_roundtrip(
    capsys: pytest.CaptureFixture[str],
    error: OperationBackendError
    | OperationResourceExhaustedError
    | OperationExecutionTimeoutError
    | OperationExecutionCancelledError,
) -> None:
    with worker_execution_errors():
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
