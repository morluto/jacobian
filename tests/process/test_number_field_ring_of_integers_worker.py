"""Process-boundary behavior for number-field integral-basis workers."""

from __future__ import annotations

import hashlib
import time
from fractions import Fraction

import pytest

from jacobian import process as process_runtime
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    request_execution,
)
from jacobian.canonical import encode_strict_json, loads_strict_json
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldRingOfIntegersRequest,
)
from jacobian.math.number_theory.number_fields._ring_of_integers_process import (
    compute_nf_ring_of_integers,
)
from jacobian.math.number_theory.number_fields.values import (
    SimpleNumberFieldPresentation,
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


def _request() -> NumberFieldRingOfIntegersRequest:
    return NumberFieldRingOfIntegersRequest(
        field=SimpleNumberFieldPresentation(coefficients_descending=(1, 0, -5))
    )


def test_ring_worker_golden_field_returns_canonical_basis_members() -> None:
    result = compute_nf_ring_of_integers(_request())

    assert result.field_discriminant == 5
    assert [
        [coefficient.as_fraction() for coefficient in element.coefficients_ascending]
        for element in result.basis
    ] == [[Fraction(1), Fraction(0)], [Fraction(1, 2), Fraction(1, 2)]]
    assert all(element.presentation is result.field for element in result.basis)


def test_cancelled_ring_worker_is_an_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_runtime,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(returncode=None, cancelled=True),
    )

    with pytest.raises(OperationExecutionCancelledError):
        compute_nf_ring_of_integers(_request())


def test_timed_out_ring_worker_is_an_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_runtime,
        "run_bounded_process",
        lambda *_args, **_kwargs: _completed(returncode=None, timed_out=True),
    )

    with pytest.raises(OperationExecutionTimeoutError):
        compute_nf_ring_of_integers(_request())


def test_ring_worker_shares_prior_request_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_launch(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        raise AssertionError("expired request must not launch a worker")

    monkeypatch.setattr(process_runtime, "run_bounded_process", fail_to_launch)

    with (
        request_execution(time.monotonic() - 61),
        pytest.raises(OperationExecutionTimeoutError),
    ):
        compute_nf_ring_of_integers(_request())


def test_ring_worker_uses_private_cwd_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def complete_worker(command: object, **kwargs: object) -> BoundedProcessResult:
        recorded["command"] = command
        recorded.update(kwargs)
        input_bytes = kwargs["input_bytes"]
        assert isinstance(input_bytes, bytes)
        response = {
            "basis": [
                [
                    {"num": "1", "den": "1"},
                    {"num": "0", "den": "1"},
                ],
                [
                    {"num": "1", "den": "2"},
                    {"num": "1", "den": "2"},
                ],
            ],
            "discriminant": "5",
            "kind": "complete",
            "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        }
        return _completed(stdout=encode_strict_json(response))

    monkeypatch.setattr(process_runtime, "run_bounded_process", complete_worker)

    result = compute_nf_ring_of_integers(_request())

    assert result.field_discriminant == 5
    command = recorded["command"]
    assert isinstance(command, list)
    assert command[-1] == "--basis"
    raw_input = recorded["input_bytes"]
    assert isinstance(raw_input, bytes)
    payload = loads_strict_json(raw_input)
    assert isinstance(payload, dict)
    # Admission algebra now runs inside the worker, so the parent sends only
    # the source field.
    assert set(payload) == {"field"}
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=1024 * 1024 * 1024,
        file_size_bytes=1024 * 1024,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-number-field-")
