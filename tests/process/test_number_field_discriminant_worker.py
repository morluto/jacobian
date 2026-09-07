"""Process-boundary behavior for number-field discriminant workers."""

import hashlib

import pytest

from jacobian import process as process_runtime
from jacobian.math.number_theory.number_fields import (
    SimpleNumberFieldPresentation,
)
from jacobian.math.number_theory.number_fields import (
    _discriminant_process as number_field_operations,
)
from jacobian.math.number_theory.number_fields._discriminant_process import (
    compute_nf_discriminant,
)
from jacobian.math.number_theory.number_fields._models import NumberFieldRequest
from jacobian.process import BoundedProcessResult, ProcessResourceLimits


def _number_field(*coefficients: str) -> SimpleNumberFieldPresentation:
    return SimpleNumberFieldPresentation(
        coefficients_descending=tuple(int(coefficient) for coefficient in coefficients)
    )


def test_timed_out_number_field_worker_is_an_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        process_runtime,
        "run_bounded_process",
        lambda *_args, **_kwargs: BoundedProcessResult(
            returncode=None,
            stdout=b"",
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=True,
        ),
    )

    with pytest.raises(TimeoutError):
        compute_nf_discriminant(NumberFieldRequest(field=_number_field("1", "0", "-2")))


def test_number_field_worker_has_private_cwd_and_os_resource_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def complete_worker(*_args: object, **kwargs: object) -> BoundedProcessResult:
        recorded.update(kwargs)
        input_bytes = kwargs["input_bytes"]
        assert isinstance(input_bytes, bytes)
        return BoundedProcessResult(
            returncode=0,
            stdout=(
                b'{"discriminant":"8","kind":"complete","request_digest":"'
                + hashlib.sha256(input_bytes).hexdigest().encode()
                + b'"}'
            ),
            stderr=b"",
            stdout_exceeded=False,
            stderr_exceeded=False,
            timed_out=False,
        )

    monkeypatch.setattr(process_runtime, "run_bounded_process", complete_worker)

    result = compute_nf_discriminant(
        NumberFieldRequest(field=_number_field("1", "0", "-2"))
    )

    assert result.discriminant == 8
    assert recorded["resource_limits"] == ProcessResourceLimits(
        cpu_seconds=60,
        address_space_bytes=number_field_operations._WORKER_ADDRESS_SPACE_BYTES,
        file_size_bytes=number_field_operations._WORKER_FILE_SIZE_BYTES,
    )
    assert str(recorded["cwd"]).split("/")[-1].startswith("jacobian-number-field-")


def test_number_field_worker_start_failure_is_operational(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        raise OSError("worker unavailable")

    monkeypatch.setattr(process_runtime, "run_bounded_process", unavailable)

    with pytest.raises(RuntimeError):
        compute_nf_discriminant(NumberFieldRequest(field=_number_field("1", "0", "-2")))
