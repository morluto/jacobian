"""Process-boundary behavior for contiguous-sum profiling."""

import pytest

import jacobian.process as process
from jacobian.math.number_theory._contiguous_sum_models import (
    ContiguousSumProfileRequest,
)
from jacobian.math.number_theory._contiguous_sum_operations import (
    compute_contiguous_sum_profile,
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

    monkeypatch.setattr(process, "run_bounded_process", timed_out_worker)

    result = compute_contiguous_sum_profile(
        ContiguousSumProfileRequest(
            lower_bound="1099511627776",
            upper_bound="1099511627776",
        )
    )

    assert result.status == "UNKNOWN"
    assert result.rows == ()
    assert result.detail
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-direct-factor-")
