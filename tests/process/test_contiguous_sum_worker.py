"""Process-boundary behavior for contiguous-sum profiling."""

import re

import pytest

from jacobian import process as process_runtime
from jacobian.math.number_theory._contiguous_sum import compute_contiguous_sum_profile
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
    _FACTORIZATION_WORKER_FILE_SIZE_BYTES,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def test_timed_out_high_magnitude_profile_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def timed_out_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        return BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        )

    monkeypatch.setattr(process_runtime, "run_bounded_process", timed_out_worker)

    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound=1099511627776,
            upper_bound=1099511627776,
        )
    )

    assert result.status == "UNKNOWN"
    assert result.rows == ()
    assert result.diagnostic is not None
    assert result.diagnostic.failure == "WORKER_TIMEOUT"
    assert result.diagnostic.timeout_layer == "WORKER_WALL"
    assert result.diagnostic.elapsed_ms >= 0
    assert 0 < result.diagnostic.worker_timeout_ms <= 60_000
    assert result.diagnostic.budget_seconds == 60
    assert result.diagnostic.operation_version == "1"
    assert re.fullmatch(r"unknown|[0-9a-f]{40}", result.diagnostic.repository_revision)
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-direct-factor-")


@pytest.mark.parametrize(
    ("completed", "failure", "timeout_layer"),
    [
        (
            BoundedProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
                cancelled=True,
            ),
            "WORKER_CANCELLED",
            "REQUEST_CANCELLATION",
        ),
        (
            BoundedProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=True,
                stderr_exceeded=False,
                timed_out=False,
            ),
            "STDOUT_LIMIT_EXCEEDED",
            "OUTPUT_LIMIT",
        ),
        (
            BoundedProcessResult(
                returncode=-9,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
            ),
            "WORKER_RESOURCE_LIMIT",
            "PROCESS_RESOURCE",
        ),
        (
            BoundedProcessResult(
                returncode=0xC0000005,
                stdout=b"",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
            ),
            "WORKER_EXITED",
            "WORKER_EXIT",
        ),
        (
            BoundedProcessResult(
                returncode=0,
                stdout=b"not json",
                stderr=b"",
                stdout_exceeded=False,
                stderr_exceeded=False,
                timed_out=False,
            ),
            "MALFORMED_OUTPUT",
            "RESULT_VALIDATION",
        ),
    ],
)
def test_worker_stop_reason_is_retained_in_public_result(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
    failure: str,
    timeout_layer: str,
) -> None:
    monkeypatch.setattr(
        process_runtime, "run_bounded_process", lambda *_args, **_kwargs: completed
    )

    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound=1099511627776,
            upper_bound=1099511627776,
        )
    )

    assert result.status == "UNKNOWN"
    assert result.rows == ()
    assert result.diagnostic is not None
    assert result.diagnostic.failure == failure
    assert result.diagnostic.timeout_layer == timeout_layer
    assert result.diagnostic.returncode == completed.returncode
    assert result.diagnostic.elapsed_ms >= 0
    assert 0 < result.diagnostic.worker_timeout_ms <= 60_000
    assert result.diagnostic.budget_seconds == 60
    assert result.diagnostic.operation_version == "1"
    assert re.fullmatch(r"unknown|[0-9a-f]{40}", result.diagnostic.repository_revision)
