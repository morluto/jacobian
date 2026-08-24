"""Bounded child-process replay of generic-fiber Gröbner certificates."""

from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from jacobian.process import (
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

if TYPE_CHECKING:
    from jacobian.math.polynomials.maps._models import GenericFiberCertificate
    from jacobian.math.polynomials.maps.values import RationalPolynomialMap

MathematicalOutcome = Literal[
    "GENERICALLY_FINITE",
    "NOT_DOMINANT",
    "DOMINANT_NOT_GENERICALLY_FINITE",
]

ReplayStatus = Literal[
    "COMPUTED",
    "INVALID",
    "LIMIT_EXCEEDED",
    "TIMEOUT",
    "CANCELLED",
    "ERROR",
]

_REPLAY_STDOUT_LIMIT = 64_000
_REPLAY_STDERR_LIMIT = 64_000
_REPLAY_ADDRESS_SPACE_BYTES = 1024 * 1024 * 1024
_REPLAY_FILE_SIZE_BYTES = 1024 * 1024

_MATHEMATICAL_OUTCOMES = frozenset(
    {
        "GENERICALLY_FINITE",
        "NOT_DOMINANT",
        "DOMINANT_NOT_GENERICALLY_FINITE",
    }
)

_REPLAY_WORKER = """
import json
import sys

from jacobian.math.polynomials.maps._generic_degree import (
    GenericFiberReplayLimitError,
    validate_generic_fiber_certificate,
)
from jacobian.math.polynomials.maps._models import GenericFiberCertificate
from jacobian.math.polynomials.maps.values import RationalPolynomialMap

payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
source = RationalPolynomialMap.model_validate(payload["source"])
certificate = GenericFiberCertificate.model_validate(payload["certificate"])
try:
    outcome, degree = validate_generic_fiber_certificate(source, certificate)
except GenericFiberReplayLimitError:
    verdict = {"status": "LIMIT_EXCEEDED"}
except ValueError:
    verdict = {"status": "INVALID"}
else:
    verdict = {"status": "COMPUTED", "outcome": outcome, "degree": degree}
sys.stdout.write(json.dumps(verdict))
"""


@dataclass(frozen=True, slots=True)
class CertificateReplayResult:
    """One bounded certificate-replay verdict."""

    status: ReplayStatus
    outcome: MathematicalOutcome | None = None
    degree: int | None = None
    detail: str | None = None


def _parse_verdict(stdout: bytes) -> CertificateReplayResult:
    try:
        verdict = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return CertificateReplayResult(
            status="ERROR",
            detail="The certificate replay worker returned an unreadable verdict.",
        )
    if not isinstance(verdict, dict):
        return CertificateReplayResult(
            status="ERROR",
            detail="The certificate replay worker returned an unreadable verdict.",
        )
    status = verdict.get("status")
    if status == "INVALID":
        return CertificateReplayResult(status="INVALID")
    if status == "LIMIT_EXCEEDED":
        return CertificateReplayResult(status="LIMIT_EXCEEDED")
    if status != "COMPUTED":
        return CertificateReplayResult(
            status="ERROR",
            detail="The certificate replay worker returned an unknown verdict.",
        )
    outcome = verdict.get("outcome")
    degree = verdict.get("degree")
    if outcome not in _MATHEMATICAL_OUTCOMES or not (
        degree is None or type(degree) is int
    ):
        return CertificateReplayResult(
            status="ERROR",
            detail="The certificate replay worker returned a malformed verdict.",
        )
    return CertificateReplayResult(
        status="COMPUTED",
        outcome=outcome,
        degree=degree,
    )


def run_bounded_certificate_replay(
    source: RationalPolynomialMap,
    certificate: GenericFiberCertificate,
    *,
    wall_seconds: float,
) -> CertificateReplayResult:
    """Replay one exact certificate inside a killable bounded worker process."""

    payload = json.dumps(
        {
            "source": source.model_dump(mode="json"),
            "certificate": certificate.model_dump(mode="json"),
        }
    ).encode("utf-8")
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        completed = run_bounded_process(
            [sys.executable, "-c", _REPLAY_WORKER],
            input_bytes=payload,
            timeout_seconds=float(wall_seconds),
            environment=worker_environment(locale="C.UTF-8"),
            stdout_limit=_REPLAY_STDOUT_LIMIT,
            stderr_limit=_REPLAY_STDERR_LIMIT,
            resource_limits=ProcessResourceLimits(
                cpu_seconds=max(1, math.ceil(wall_seconds)),
                address_space_bytes=_REPLAY_ADDRESS_SPACE_BYTES,
                file_size_bytes=_REPLAY_FILE_SIZE_BYTES,
            ),
            platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
        )
    except OSError:
        return CertificateReplayResult(
            status="ERROR",
            detail="The certificate replay worker could not be started.",
        )
    if completed.cancelled:
        return CertificateReplayResult(
            status="CANCELLED",
            detail="Certificate replay was cancelled before producing a result.",
        )
    if completed.timed_out:
        return CertificateReplayResult(
            status="TIMEOUT",
            detail="Certificate replay exceeded the declared wall-time limit.",
        )
    if (
        completed.stdout_exceeded
        or completed.stderr_exceeded
        or completed.returncode != 0
    ):
        return CertificateReplayResult(
            status="LIMIT_EXCEEDED",
            detail=(
                "The generic-fiber certificate replay exceeded the declared "
                "computation envelope."
            ),
        )
    return _parse_verdict(completed.stdout)


__all__ = ["CertificateReplayResult", "run_bounded_certificate_replay"]
