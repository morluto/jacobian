"""Operational failure contracts for the bounded cyclic-kernel batch."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

import jacobian.process as process
from jacobian._execution import OperationBackendError
from jacobian.canonical import encode_strict_json
from jacobian.math.matrices.cyclic_linear import _kernel_process
from jacobian.math.matrices.cyclic_linear._kernel_process import (
    run_cyclotomic_kernels,
)


class _PickleSideEffect:
    def __init__(self, target: Path) -> None:
        self.target = target

    def __reduce__(self) -> tuple[object, tuple[Path, str]]:
        return (Path.write_text, (self.target, "executed"))


def test_cyclic_kernel_rejects_an_unbound_worker_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unbound_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=pickle.dumps((b"bad", (_PickleSideEffect(tmp_path / "owned"),))),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", unbound_projection)

    with pytest.raises(OperationBackendError) as error:
        run_cyclotomic_kernels(((1, 1, (((1,),),), 1),), deadline=None)
    assert error.value.reason.value == "malformed_response"
    assert not (tmp_path / "owned").exists()


def test_cyclic_kernel_rejects_zero_denominator_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ((1, 1, (((1,),),), 1),)
    request_digest = hashlib.sha256(
        _kernel_process._encode_requests(request)
    ).hexdigest()
    response = encode_strict_json(
        {
            "digest": request_digest,
            "results": [[1, 1, None, [[[{"num": "1", "den": "0"}]]]]],
        }
    )

    def malformed_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=response,
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", malformed_projection)
    with pytest.raises(OperationBackendError) as error:
        _kernel_process.run_cyclotomic_kernels(request, deadline=None)
    assert error.value.reason.value == "malformed_response"


def test_cyclic_kernel_rejects_response_dimensions_not_bound_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ((1, 1, (((1,),),), 1),)
    request_digest = hashlib.sha256(
        _kernel_process._encode_requests(request)
    ).hexdigest()
    response = encode_strict_json(
        {
            "digest": request_digest,
            "results": [[999_999, 999_999, None, []]],
        }
    )

    def malformed_projection(
        *args: object, **kwargs: object
    ) -> process.BoundedProcessResult:
        return process.BoundedProcessResult(
            returncode=0,
            stdout=response,
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process, "run_bounded_process", malformed_projection)
    with pytest.raises(OperationBackendError) as error:
        _kernel_process.run_cyclotomic_kernels(request, deadline=None)
    assert error.value.reason.value == "malformed_response"
