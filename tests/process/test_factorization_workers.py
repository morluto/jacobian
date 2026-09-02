"""Process-boundary behavior for integer-factorization workers."""

import pytest

from jacobian import process as process_runtime
from jacobian.math.number_theory._certification_models import (
    CertifiedFactorizationRequest,
)
from jacobian.math.number_theory._direct_factorization_models import (
    FactorizationRequest,
)
from jacobian.math.number_theory._factorization_kernels import (
    _FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
    _FACTORIZATION_WORKER_FILE_SIZE_BYTES,
    enumerate_divisors,
    factorize_certified,
    factorize_primes,
)
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _timed_out_worker(*_args: object, **_kwargs: object) -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=None,
        stdout=b"",
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=True,
    )


def test_timed_out_certified_factorization_raises_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_runtime, "run_bounded_process", _timed_out_worker)

    with pytest.raises(TimeoutError):
        factorize_certified(CertifiedFactorizationRequest(value="10403"))


def test_timed_out_direct_factorization_worker_raises_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_runtime, "run_bounded_process", _timed_out_worker)

    request = FactorizationRequest(value="12")
    with pytest.raises(TimeoutError):
        enumerate_divisors(request)
    with pytest.raises(TimeoutError):
        factorize_primes(request)


def test_factorization_workers_have_private_cwds_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: list[dict[str, object]] = []

    def timed_out_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.append(kwargs)
        return _timed_out_worker()

    monkeypatch.setattr(process_runtime, "run_bounded_process", timed_out_worker)

    with pytest.raises(TimeoutError):
        factorize_certified(CertifiedFactorizationRequest(value="10403"))
    with pytest.raises(TimeoutError):
        factorize_primes(FactorizationRequest(value="12"))
    assert len(recorded) == 2
    for invocation, prefix in zip(
        recorded,
        ("jacobian-certified-factor-", "jacobian-direct-factor-"),
        strict=True,
    ):
        assert invocation["resource_limits"] == ProcessResourceLimits(
            cpu_seconds=60,
            address_space_bytes=_FACTORIZATION_WORKER_ADDRESS_SPACE_BYTES,
            file_size_bytes=_FACTORIZATION_WORKER_FILE_SIZE_BYTES,
        )
        assert str(invocation["cwd"]).split("/")[-1].startswith(prefix)
