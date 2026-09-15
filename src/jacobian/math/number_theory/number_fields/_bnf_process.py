"""Killable PARI process boundary for exact class and unit group computation."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._execution import (
    OperationExecutionCancelledError,
    OperationExecutionTimeoutError,
    bind_request_deadline,
    current_request_execution,
    request_checkpoint,
    request_execution,
)
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.number_theory.number_fields._models import (
    NumberFieldClassGroupRequest,
    NumberFieldUnitGroupRequest,
)
from jacobian.math.number_theory.number_fields.values import (
    MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    SimpleNumberFieldElement,
    SimpleNumberFieldPresentation,
)

_WORKER = Path(__file__).resolve().with_name("_bnf_worker.py")
_WORKER_TIMEOUT_SECONDS = 120.0
_WORKER_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024
_WORKER_FILE_SIZE_BYTES = 1024 * 1024
_WORKER_STDERR_BYTES = 64 * 1024
_WORKER_REJECTION_CODE_CHARACTERS = 96
_WORKER_REJECTION_MESSAGE_CHARACTERS = 512

_BnfRequest = NumberFieldClassGroupRequest | NumberFieldUnitGroupRequest


@dataclass(frozen=True, slots=True)
class BnfWorkerResult:
    """The bounded worker projection retained by the parent process."""

    class_number: int
    abelian_invariants: tuple[int, ...]
    field_discriminant: int
    real_embedding_count: int
    complex_embedding_pair_count: int
    ideal_representatives: tuple[tuple[tuple[int, ...], ...], ...]
    rank: int
    torsion_order: int
    torsion_generator: tuple[CanonicalRational, ...]
    fundamental_units: tuple[tuple[CanonicalRational, ...], ...]


def _require_pari() -> None:
    try:
        import cypari  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise OperationResourceAdmissionError(
            location=("field",),
            code="number_field.pari_backend_unavailable",
            message="class and unit group computation requires the PARI backend",
        ) from exc


def run_bnf_worker(request: _BnfRequest) -> BnfWorkerResult:
    """Compute class and unit data in a request-owned killable worker."""

    _require_pari()
    execution = current_request_execution()
    if execution is None:
        with request_execution(time.monotonic()):
            return run_bnf_worker(request)

    owner_deadline = execution.started_at + _WORKER_TIMEOUT_SECONDS
    deadline = (
        min(execution.deadline, owner_deadline)
        if execution.deadline is not None
        else owner_deadline
    )
    bind_request_deadline(deadline)
    request_checkpoint("before number-field class-group preparation")

    payload = dict(request.model_dump(mode="json"))
    input_bytes = encode_strict_json(payload)
    stdout_limit = _worker_stdout_limit(request.field)
    try:
        with TemporaryDirectory(prefix="jacobian-bnf-") as worker_directory:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise OperationExecutionTimeoutError(
                    "request deadline expired before bnf worker launch"
                )
            from jacobian.process import (
                ProcessResourceLimits,
                run_bounded_process,
                worker_environment,
            )

            completed = run_bounded_process(
                [sys.executable, str(_WORKER)],
                input_bytes=input_bytes,
                timeout_seconds=remaining,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=stdout_limit,
                stderr_limit=_WORKER_STDERR_BYTES,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=math.ceil(_WORKER_TIMEOUT_SECONDS),
                    address_space_bytes=_WORKER_ADDRESS_SPACE_BYTES,
                    file_size_bytes=_WORKER_FILE_SIZE_BYTES,
                ),
                cwd=worker_directory,
            )
    except OperationExecutionTimeoutError:
        raise
    except OSError as exc:
        request_checkpoint("during bnf worker startup")
        raise RuntimeError("bounded bnf worker could not be started") from exc

    request_checkpoint("after number-field bnf worker")
    if completed.cancelled:
        raise OperationExecutionCancelledError("class/unit group computation cancelled")
    if completed.timed_out:
        raise OperationExecutionTimeoutError(
            "class/unit group computation timed out"
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        raise RuntimeError("bounded bnf worker did not establish its invariants")
    return _decode_worker_response(
        completed.stdout,
        request=request,
        input_bytes=input_bytes,
        stdout_limit=stdout_limit,
    )


def bounded_rejection_text(value: str, *, limit: int) -> str:
    """Return a rejection-text prefix whose strict-JSON encoding fits ``limit``."""

    cleaned = "".join(
        character
        if 0x20 <= ord(character) < 0x7F and character not in {'"', "\\"}
        else " "
        for character in value
    )
    return cleaned[:limit]


def worker_rejection(
    error: OperationDomainValidationError | OperationResourceAdmissionError,
    *,
    request_digest: str,
) -> dict[str, object]:
    """Project one typed admission failure into the bounded worker response."""

    detail = error.errors()[0]
    return {
        "kind": "rejected",
        "resource": isinstance(error, OperationResourceAdmissionError),
        "code": bounded_rejection_text(
            str(detail["type"]), limit=_WORKER_REJECTION_CODE_CHARACTERS
        ),
        "message": bounded_rejection_text(
            str(detail["msg"]), limit=_WORKER_REJECTION_MESSAGE_CHARACTERS
        ),
        "request_digest": request_digest,
    }


def _worker_rejection_envelope() -> dict[str, object]:
    return {
        "kind": "rejected",
        "resource": False,
        "code": "x" * _WORKER_REJECTION_CODE_CHARACTERS,
        "message": "x" * _WORKER_REJECTION_MESSAGE_CHARACTERS,
        "request_digest": "0" * 64,
    }


def _ratio_json() -> dict[str, str]:
    return {
        "num": "-" + "9" * MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
        "den": "9" * MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
    }


def _worker_stdout_limit(field: SimpleNumberFieldPresentation) -> int:
    degree = field.degree
    matrix = [[1] * degree for _ in range(degree)]
    unit = [_ratio_json() for _ in range(degree)]
    response: dict[str, object] = {
        "kind": "complete",
        "class_number": "9" * 64,
        "abelian_invariants": [9] * degree,
        "field_discriminant": "-" + "9" * 64,
        "real_embedding_count": degree,
        "complex_embedding_pair_count": degree,
        "ideal_representatives": [matrix for _ in range(degree)],
        "rank": degree,
        "torsion_order": 2 * degree,
        "torsion_generator": unit,
        "fundamental_units": [unit for _ in range(degree)],
        "request_digest": "0" * 64,
    }
    return max(
        len(encode_strict_json(response)),
        len(encode_strict_json(_worker_rejection_envelope())),
    )


class _WorkerRejectionError(Exception):
    def __init__(self, *, resource: bool, code: str, message: str) -> None:
        super().__init__(message)
        self.resource = resource
        self.code = code
        self.message = message


def _decode_worker_response(
    output: bytes,
    *,
    request: _BnfRequest,
    input_bytes: bytes,
    stdout_limit: int,
) -> BnfWorkerResult:
    try:
        response = loads_strict_json(
            output,
            limits=CanonicalLimits(
                max_input_bytes=stdout_limit, max_output_bytes=stdout_limit
            ),
        )
        if not isinstance(response, dict):
            raise ValueError("worker response must be an object")
        if response.get("request_digest") != hashlib.sha256(input_bytes).hexdigest():
            raise ValueError("worker response is not bound to its request")
        if response.get("kind") == "rejected":
            if set(response) != {
                "kind", "resource", "code", "message", "request_digest",
            }:
                raise ValueError("rejected worker response has invalid fields")
            code, message, resource = (
                response["code"], response["message"], response["resource"],
            )
            if (
                not isinstance(code, str)
                or not isinstance(message, str)
                or not isinstance(resource, bool)
            ):
                raise ValueError("rejected worker response has invalid values")
            raise _WorkerRejectionError(resource=resource, code=code, message=message)
        expected = {
            "kind",
            "class_number",
            "abelian_invariants",
            "field_discriminant",
            "real_embedding_count",
            "complex_embedding_pair_count",
            "ideal_representatives",
            "rank",
            "torsion_order",
            "torsion_generator",
            "fundamental_units",
            "request_digest",
        }
        if response.get("kind") != "complete" or set(response) != expected:
            raise ValueError("complete worker response has invalid fields")
        degree = request.field.degree
        class_number = parse_canonical_integer(
            _require_canonical(response["class_number"])
        )
        discriminant = parse_canonical_integer(
            _require_canonical(response["field_discriminant"])
        )
        invariants = tuple(int(value) for value in response["abelian_invariants"])
        representatives = tuple(
            _decode_matrix(matrix, degree=degree)
            for matrix in response["ideal_representatives"]
        )
        return BnfWorkerResult(
            class_number=class_number,
            abelian_invariants=invariants,
            field_discriminant=discriminant,
            real_embedding_count=int(response["real_embedding_count"]),
            complex_embedding_pair_count=int(response["complex_embedding_pair_count"]),
            ideal_representatives=representatives,
            rank=int(response["rank"]),
            torsion_order=int(response["torsion_order"]),
            torsion_generator=_decode_element(
                response["torsion_generator"], degree=degree
            ),
            fundamental_units=tuple(
                _decode_element(element, degree=degree)
                for element in response["fundamental_units"]
            ),
        )
    except _WorkerRejectionError as rejection:
        if rejection.resource:
            raise OperationResourceAdmissionError(
                location=("field",), code=rejection.code, message=rejection.message
            ) from rejection
        raise OperationDomainValidationError(
            location=("field",), code=rejection.code, message=rejection.message
        ) from rejection
    except (CanonicalizationError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("bounded bnf worker returned malformed output") from exc


def _require_canonical(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("worker integer must be a canonical string")
    parsed = parse_canonical_integer(value)
    if format_canonical_integer(parsed) != value:
        raise ValueError("worker integer is not canonical")
    return value


def _decode_matrix(raw: object, *, degree: int) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or len(raw) != degree:
        raise ValueError("worker ideal matrix has invalid row count")
    rows: list[tuple[int, ...]] = []
    for row in raw:
        if not isinstance(row, list) or len(row) != degree:
            raise ValueError("worker ideal matrix has invalid row width")
        rows.append(tuple(int(value) for value in row))
    return tuple(rows)


def _decode_element(raw: object, *, degree: int) -> tuple[CanonicalRational, ...]:
    if not isinstance(raw, list) or len(raw) != degree:
        raise ValueError("worker field element has invalid coordinate count")
    coordinates: list[CanonicalRational] = []
    for component in raw:
        if not isinstance(component, dict) or set(component) != {"num", "den"}:
            raise ValueError("worker field element has an invalid rational")
        num = parse_canonical_integer(_require_canonical(component["num"]))
        den = parse_canonical_integer(_require_canonical(component["den"]))
        rational = CanonicalRational(num=num, den=den)
        require_bounded_rational(
            rational,
            max_digits=MAX_SIMPLE_NUMBER_FIELD_ELEMENT_DIGITS,
            label="unit element",
        )
        coordinates.append(rational)
    return tuple(coordinates)


def bind_element(
    field: SimpleNumberFieldPresentation,
    coefficients: tuple[CanonicalRational, ...],
) -> SimpleNumberFieldElement:
    """Build one retained field element from trusted worker coordinates."""

    return SimpleNumberFieldElement.model_construct(
        presentation=field, coefficients_ascending=coefficients
    )


__all__ = [
    "BnfWorkerResult",
    "bind_element",
    "bounded_rejection_text",
    "run_bnf_worker",
    "worker_rejection",
]
