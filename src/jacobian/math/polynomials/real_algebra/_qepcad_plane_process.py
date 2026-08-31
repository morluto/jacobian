"""Killable process boundary for exact QEPCAD plane-component profiles."""

from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import TYPE_CHECKING, Literal

from pydantic import ValidationError

from jacobian._execution import OperationExecutionCancelledError
from jacobian.math.polynomials.real_algebra._plane_component_models import (
    IsolatedRealPlanePoint,
    PlaneComponentProfileRequest,
)
from jacobian.math.polynomials.real_algebra._qepcad_plane_protocol import (
    MAX_QEPCAD_WORKER_RESPONSE_BYTES,
    QEPCAD_PLANE_WORKER_RESPONSE_ADAPTER,
    PlaneSamplesValid,
    PlaneSampleWorkerRequest,
    QepcadPlaneWorkerComplete,
    QepcadPlaneWorkerInvalid,
    QepcadPlaneWorkerRejected,
    QepcadPlaneWorkerRequest,
    QepcadPlaneWorkerResponse,
)

if TYPE_CHECKING:
    from jacobian._models import StrictModel
    from jacobian.process import BoundedProcessResult

_WORKER = Path(__file__).resolve().with_name("_qepcad_plane_worker.py")
_ADDRESS_SPACE_BYTES = 1536 * 1024 * 1024
_FILE_SIZE_BYTES = 1024 * 1024
_STDOUT_BYTES = MAX_QEPCAD_WORKER_RESPONSE_BYTES
_STDERR_BYTES = 64 * 1024
_SAMPLE_STDOUT_BYTES = 64 * 1024

QepcadPlaneProcessStatus = Literal[
    "COMPUTED",
    "BACKEND_UNAVAILABLE",
    "TIMEOUT",
    "RESOURCE_LIMIT",
    "BACKEND_ERROR",
]
QepcadPlaneProcessReason = Literal[
    "SUPPORTED_QEPCAD_NOT_INSTALLED",
    "UNSUPPORTED_QEPCAD_VERSION",
    "QEPCAD_DEADLINE_EXPIRED",
    "QEPCAD_OUTPUT_LIMIT",
    "QEPCAD_CELL_LIMIT",
    "QEPCAD_INVALID_OUTPUT",
    "QEPCAD_EXECUTION_FAILED",
    "SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
    "SAMPLE_RECOGNITION_OUTPUT_LIMIT",
    "RESULT_OUTPUT_LIMIT",
    "SAMPLE_RECOGNITION_INVALID_OUTPUT",
    "SAMPLE_RECOGNITION_EXECUTION_FAILED",
]


@dataclass(frozen=True, slots=True)
class QepcadPlaneProcessOutcome:
    status: QepcadPlaneProcessStatus
    reason: QepcadPlaneProcessReason | None = None
    version: str | None = None
    projection: QepcadPlaneWorkerComplete | None = None
    canonical_samples: tuple[IsolatedRealPlanePoint, ...] | None = None


class QepcadPlaneSampleValidationError(ValueError):
    """A supplied point was not an admitted exact isolated sample."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _qepcad_root(executable: str) -> str | None:
    configured = os.environ.get("QEPCAD_ROOT")
    candidates = (
        Path(configured) if configured else None,
        Path(executable).resolve().parent.parent / "lib" / "qepcad",
        Path("/usr/lib/qepcad"),
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "default.qepcadrc").is_file():
            return str(candidate.resolve())
    return None


def _worker_outcome(
    response: QepcadPlaneWorkerResponse,
) -> QepcadPlaneProcessOutcome:
    if isinstance(response, QepcadPlaneWorkerComplete):
        return QepcadPlaneProcessOutcome(
            status="COMPUTED",
            version=response.version,
            projection=response,
        )
    if isinstance(response, QepcadPlaneWorkerInvalid):
        raise QepcadPlaneSampleValidationError(response.reason)
    if not isinstance(response, QepcadPlaneWorkerRejected):
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="QEPCAD_INVALID_OUTPUT",
        )
    reason = response.reason
    if reason == "UNSUPPORTED_QEPCAD_VERSION":
        status: QepcadPlaneProcessStatus = "BACKEND_UNAVAILABLE"
    elif reason == "QEPCAD_DEADLINE_EXPIRED":
        status = "TIMEOUT"
    elif reason in {"QEPCAD_OUTPUT_LIMIT", "QEPCAD_CELL_LIMIT"}:
        status = "RESOURCE_LIMIT"
    else:
        status = "BACKEND_ERROR"
    return QepcadPlaneProcessOutcome(
        status=status,
        reason=reason,
    )


def _run_worker(
    request: StrictModel,
    *,
    deadline: float,
    stdout_limit: int,
) -> BoundedProcessResult | None:
    from jacobian.process import (
        ProcessResourceLimits,
        run_bounded_process,
        worker_environment,
    )

    if deadline - monotonic() <= 0:
        return None
    payload = request.model_dump_json().encode("utf-8")
    with TemporaryDirectory(prefix="jacobian-qepcad-plane-") as worker_directory:
        remaining = deadline - monotonic()
        if remaining <= 0:
            return None
        return run_bounded_process(
            [sys.executable, str(_WORKER)],
            input_bytes=payload,
            timeout_seconds=remaining,
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=stdout_limit,
            stderr_limit=_STDERR_BYTES,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(remaining)),
                address_space_bytes=_ADDRESS_SPACE_BYTES,
                file_size_bytes=_FILE_SIZE_BYTES,
            ),
            cwd=worker_directory,
        )


def run_plane_sample_recognition(
    request: PlaneComponentProfileRequest,
    *,
    wall_seconds: float,
) -> QepcadPlaneProcessOutcome:
    """Recognize exact samples for a component profile that needs no CAD."""

    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError(
            "plane sample-recognition wall budget must be positive and finite"
        )
    deadline = monotonic() + wall_seconds
    worker_request = PlaneSampleWorkerRequest(samples=request.samples)
    try:
        completed = _run_worker(
            worker_request,
            deadline=deadline,
            stdout_limit=_SAMPLE_STDOUT_BYTES,
        )
    except OSError:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="SAMPLE_RECOGNITION_EXECUTION_FAILED",
        )
    if completed is None or completed.timed_out:
        return QepcadPlaneProcessOutcome(
            status="TIMEOUT",
            reason="SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during exact plane-sample recognition"
        )
    if completed.stdout_exceeded:
        return QepcadPlaneProcessOutcome(
            status="RESOURCE_LIMIT",
            reason="SAMPLE_RECOGNITION_OUTPUT_LIMIT",
        )
    if completed.stderr_exceeded or completed.returncode != 0:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="SAMPLE_RECOGNITION_EXECUTION_FAILED",
        )
    try:
        response = QEPCAD_PLANE_WORKER_RESPONSE_ADAPTER.validate_json(
            completed.stdout,
            strict=True,
        )
    except ValidationError:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="SAMPLE_RECOGNITION_INVALID_OUTPUT",
        )
    if monotonic() >= deadline:
        return QepcadPlaneProcessOutcome(
            status="TIMEOUT",
            reason="SAMPLE_RECOGNITION_DEADLINE_EXPIRED",
        )
    if isinstance(response, QepcadPlaneWorkerInvalid):
        raise QepcadPlaneSampleValidationError(response.reason)
    if not isinstance(response, PlaneSamplesValid):
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="SAMPLE_RECOGNITION_INVALID_OUTPUT",
        )
    return QepcadPlaneProcessOutcome(
        status="COMPUTED",
        canonical_samples=response.canonical_samples,
    )


def run_qepcad_plane_components(
    request: PlaneComponentProfileRequest,
    *,
    wall_seconds: float,
    canonical_samples: tuple[IsolatedRealPlanePoint, ...] | None = None,
) -> QepcadPlaneProcessOutcome:
    """Return one exact profile, or an explicit operational non-conclusion."""

    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("QEPCAD wall budget must be positive and finite")
    deadline = monotonic() + wall_seconds
    executable = shutil.which("qepcad")
    if executable is None:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_UNAVAILABLE",
            reason="SUPPORTED_QEPCAD_NOT_INSTALLED",
        )
    qepcad_root = _qepcad_root(executable)
    if qepcad_root is None:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_UNAVAILABLE",
            reason="SUPPORTED_QEPCAD_NOT_INSTALLED",
        )
    try:
        worker_request = QepcadPlaneWorkerRequest(
            executable=str(Path(executable).resolve()),
            qepcad_root=qepcad_root,
            deadline_monotonic=deadline,
            request=request,
            canonical_samples=canonical_samples,
        )
    except ValidationError:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="QEPCAD_INVALID_OUTPUT",
        )

    try:
        completed = _run_worker(
            worker_request,
            deadline=deadline,
            stdout_limit=_STDOUT_BYTES,
        )
    except OSError:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="QEPCAD_EXECUTION_FAILED",
        )

    if completed is None:
        return QepcadPlaneProcessOutcome(
            status="TIMEOUT",
            reason="QEPCAD_DEADLINE_EXPIRED",
        )
    if completed.cancelled:
        raise OperationExecutionCancelledError(
            "request cancelled during QEPCAD plane-component computation"
        )
    if completed.timed_out:
        return QepcadPlaneProcessOutcome(
            status="TIMEOUT",
            reason="QEPCAD_DEADLINE_EXPIRED",
        )
    if completed.stdout_exceeded:
        return QepcadPlaneProcessOutcome(
            status="RESOURCE_LIMIT",
            reason="QEPCAD_OUTPUT_LIMIT",
        )
    if completed.stderr_exceeded or completed.returncode != 0:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="QEPCAD_EXECUTION_FAILED",
        )
    try:
        response = QEPCAD_PLANE_WORKER_RESPONSE_ADAPTER.validate_json(
            completed.stdout,
            strict=True,
        )
    except ValidationError:
        return QepcadPlaneProcessOutcome(
            status="BACKEND_ERROR",
            reason="QEPCAD_INVALID_OUTPUT",
        )
    if monotonic() >= deadline:
        return QepcadPlaneProcessOutcome(
            status="TIMEOUT",
            reason="QEPCAD_DEADLINE_EXPIRED",
        )
    return _worker_outcome(response)


__all__ = [
    "QepcadPlaneProcessOutcome",
    "QepcadPlaneSampleValidationError",
    "run_plane_sample_recognition",
    "run_qepcad_plane_components",
]
