"""Process-boundary behavior for contiguous-sum profiling."""

import pytest

from jacobian import process as process_runtime
from jacobian._execution import (
    BackendFailureReason,
    ExecutionResource,
    OperationBackendError,
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    OperationResourceExhaustedError,
)
from jacobian.math.number_theory._contiguous_sum import compute_contiguous_sum_profile
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
    _FACTORIZATION_WORKER_FILE_SIZE_BYTES,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _completed(
    *,
    returncode: int | None = 0,
    stdout: bytes = b"",
    stderr: bytes = b"",
    stdout_exceeded: bool = False,
    stderr_exceeded: bool = False,
    timed_out: bool = False,
    cancelled: bool = False,
) -> BoundedProcessResult:
    """Name the operational state instead of repeating result fields."""
    return BoundedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        stdout_exceeded=stdout_exceeded,
        stderr_exceeded=stderr_exceeded,
        timed_out=timed_out,
        cancelled=cancelled,
    )


def test_timed_out_high_magnitude_profile_is_an_operational_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def timed_out_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        return _completed(returncode=None, timed_out=True)

    monkeypatch.setattr(process_runtime, "run_bounded_process", timed_out_worker)

    with pytest.raises(OperationExecutionTimeoutError):
        compute_contiguous_sum_profile(
            ContiguousSumProfileRequest(
                lower_bound=1099511627776,
                upper_bound=1099511627776,
            )
        )
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-direct-factor-")


@pytest.mark.parametrize(
    ("completed", "error_type", "detail"),
    [
        (
            _completed(returncode=None, cancelled=True),
            OperationExecutionCancelledError,
            None,
        ),
        (
            _completed(returncode=None, stdout_exceeded=True),
            OperationResourceExhaustedError,
            ExecutionResource.OUTPUT,
        ),
        (
            _completed(returncode=-9),
            OperationResourceExhaustedError,
            ExecutionResource.MEMORY,
        ),
        (
            _completed(returncode=0xC0000005),
            OperationBackendError,
            BackendFailureReason.ABNORMAL_EXIT,
        ),
        (
            _completed(stdout=b"not json"),
            OperationBackendError,
            BackendFailureReason.MALFORMED_RESPONSE,
        ),
    ],
)
def test_worker_stop_reason_maps_to_operational_error(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
    error_type: type[Exception],
    detail: object,
) -> None:
    monkeypatch.setattr(
        process_runtime, "run_bounded_process", lambda *_args, **_kwargs: completed
    )

    with pytest.raises(error_type) as raised:
        compute_contiguous_sum_profile(
            ContiguousSumProfileRequest(
                lower_bound=1099511627776,
                upper_bound=1099511627776,
            )
        )
    if isinstance(raised.value, OperationResourceExhaustedError):
        assert raised.value.resource == detail
    if isinstance(raised.value, OperationBackendError):
        assert raised.value.reason == detail
