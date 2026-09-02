"""Process-boundary behavior for contiguous-sum profiling."""

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
            lower_bound="1099511627776",
            upper_bound="1099511627776",
        )
    )

    assert result.model_dump() == {
        "status": "UNKNOWN",
        "lower_bound": "1099511627776",
        "upper_bound": "1099511627776",
        "rows": (),
        "detail": "the bounded factorization worker did not establish the complete profile",
    }
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-direct-factor-")


@pytest.mark.parametrize(
    "completed",
    [
        BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
            cancelled=True,
        ),
        BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=True,
            stderr_exceeded=False,
            timed_out=False,
        ),
        BoundedProcessResult(
            returncode=-9,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
        BoundedProcessResult(
            returncode=0xC0000005,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
        BoundedProcessResult(
            returncode=0,
            stdout=b"not json",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        ),
    ],
)
def test_worker_stop_reason_is_hidden_from_public_result(
    monkeypatch: pytest.MonkeyPatch,
    completed: BoundedProcessResult,
) -> None:
    monkeypatch.setattr(
        process_runtime, "run_bounded_process", lambda *_args, **_kwargs: completed
    )

    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound="1099511627776",
            upper_bound="1099511627776",
        )
    )

    assert result.model_dump() == {
        "status": "UNKNOWN",
        "lower_bound": "1099511627776",
        "upper_bound": "1099511627776",
        "rows": (),
        "detail": "the bounded factorization worker did not establish the complete profile",
    }
