"""Process-boundary behavior for number-field discriminant workers."""

import hashlib
import time

import pytest

from jacobian import process as process_runtime
from jacobian._execution import OperationExecutionTimeoutError, request_execution
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


def test_number_field_budget_includes_prior_request_work() -> None:
    request = NumberFieldRequest(field=_number_field("1", "0", "-2"))
    with (
        request_execution(time.monotonic() - 61),
        pytest.raises(OperationExecutionTimeoutError, match="deadline expired"),
    ):
        compute_nf_discriminant(request)


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


def test_semiprime_discriminant_rejection_survives_the_stdout_envelope() -> None:
    """A worker rejection is budgeted, not just a worker answer.

    ``include_basis=False`` used to size stdout from the bare discriminant
    alone, so the larger typed ``rejected`` response was truncated, reported as
    ``stdout_exceeded``, and re-raised as a generic ``RuntimeError``. The
    motivating field must therefore return the declared domain error.
    """

    from jacobian.catalog.models import OperationDomainValidationError

    request = NumberFieldRequest(field=_number_field("1", "0", str(-100003 * 100019)))

    with pytest.raises(OperationDomainValidationError) as error:
        compute_nf_discriminant(request)

    assert error.value.errors()[0]["type"] == (
        "number_field.ring_of_integers_discriminant_factorization_bound"
    )


def test_discriminant_stdout_envelope_bounds_every_worker_response() -> None:
    """The advertised stdout ceiling covers both emitted response branches."""

    from jacobian.canonical import encode_strict_json
    from jacobian.catalog.models import (
        OperationDomainValidationError,
        OperationResourceAdmissionError,
    )
    from jacobian.math.number_theory.number_fields._integral_basis_process import (
        _worker_stdout_limit,
        worker_rejection,
    )

    field = _number_field("1", "0", str(-100003 * 100019))
    limit = _worker_stdout_limit(field, include_basis=False)
    rejections = (
        OperationDomainValidationError(
            location=("field",),
            code="number_field.ring_of_integers_discriminant_factorization_bound",
            message="x" * 4_096,
        ),
        OperationResourceAdmissionError(
            location=("field",),
            code="number_field.ring_of_integers_discriminant_output_bound",
            message="y" * 4_096,
        ),
    )
    for rejection in rejections:
        emitted = encode_strict_json(
            worker_rejection(rejection, request_digest="0" * 64)
        )
        assert len(emitted) <= limit
    # A rejection is larger than the bare discriminant it replaces, which is
    # precisely the envelope the discriminant-only route used to omit.
    discriminant_only = encode_strict_json(
        {
            "kind": "complete",
            "discriminant": "-9",
            "request_digest": "0" * 64,
        }
    )
    assert len(discriminant_only) < limit
