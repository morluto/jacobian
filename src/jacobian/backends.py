"""Optional native runtime requirements and explicit environment diagnostics.

Importing this module never probes or installs a backend. Checks describe the
current environment, not mathematical correctness or future call completion.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from jacobian._models import StrictModel

SINGULAR_VERSION_MIN = 44000
SINGULAR_VERSION_MAX = 45000

BackendName = Literal["singular", "qepcad"]
BackendState = Literal["AVAILABLE", "MISSING", "UNSUPPORTED", "CHECK_FAILED"]


class BackendAvailability(StrictModel):
    """A bounded executable/version check, not an operation execution guarantee."""

    backend: BackendName
    status: BackendState
    required_version: str
    version: str | None = None
    detail: str
    installation: str


def backend_installation(backend: BackendName) -> str:
    """Describe provisioning in the environment that executes Jacobian."""

    version = _required_version(backend)
    return (
        f"Install {backend} {version} in the environment running Jacobian and put "
        f"{'Singular' if backend == 'singular' else 'qepcad'} on PATH. "
        f"On Debian/Ubuntu: sudo apt-get install {backend}; verify the version. "
        "For remote MCP, the server operator must install it. "
        "See https://github.com/morluto/jacobian/blob/main/docs/how-to/"
        "backend-requirements.md."
    )


def _required_version(backend: BackendName) -> str:
    if backend == "singular":
        return "4.4.x"
    if backend == "qepcad":
        return "B 1.74"
    raise ValueError(f"unknown optional backend: {backend}")


class BackendUnavailableError(RuntimeError):
    """An optional native runtime is absent, unsupported, or misconfigured."""

    def __init__(self, backend: BackendName, *, detail: str) -> None:
        self.backend = backend
        self.required_version = _required_version(backend)
        self.detail = detail
        self.installation = backend_installation(backend)
        super().__init__(f"{detail} {self.installation}")


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


def check_backend(backend: BackendName) -> BackendAvailability:
    """Check one optional runtime without importing the catalog or doing algebra.

    Uses at most five seconds for a version subprocess, with bounded output.
    No result is cached: installation and PATH can change between calls.
    """

    from jacobian.process import run_bounded_process, worker_environment

    required_version = _required_version(backend)

    def report(
        status: BackendState, detail: str, version: str | None = None
    ) -> BackendAvailability:
        return BackendAvailability(
            backend=backend,
            status=status,
            required_version=required_version,
            version=version,
            detail=detail,
            installation=backend_installation(backend),
        )

    executable = shutil.which("Singular" if backend == "singular" else "qepcad")
    if executable is None:
        return report("MISSING", "The executable was not found on PATH.")
    arguments = (
        [
            "-q",
            "-t",
            "--no-rc",
            "--no-shell",
            "--no-stdlib",
            "--execute",
            'system("version");quit;',
        ]
        if backend == "singular"
        else ["-v"]
    )
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-backend-check-") as directory:
            completed = run_bounded_process(
                [str(Path(executable).resolve()), *arguments],
                input_bytes=b"",
                timeout_seconds=5.0,
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=4096,
                stderr_limit=4096,
                cwd=directory,
            )
    except OSError:
        return report("CHECK_FAILED", "The executable could not be started.")
    if (
        completed.returncode != 0
        or completed.timed_out
        or completed.cancelled
        or completed.stdout_exceeded
        or completed.stderr_exceeded
    ):
        return report("CHECK_FAILED", "The bounded version check did not complete.")
    output = completed.stdout.decode("ascii", errors="replace").strip()
    if backend == "singular":
        if not re.fullmatch(r"[0-9]{5,6}", output):
            return report("CHECK_FAILED", "Unrecognized Singular version output.")
        numeric = int(output)
        major, remainder = divmod(numeric, 10000)
        minor, patch_code = divmod(remainder, 1000)
        version = f"{major}.{minor}.{patch_code // 100}"
        supported = SINGULAR_VERSION_MIN <= numeric < SINGULAR_VERSION_MAX
    else:
        match = re.search(r"Version B ([0-9]{1,3}\.[0-9]{1,3}),", output)
        if match is None:
            return report("CHECK_FAILED", "Unrecognized QEPCAD version output.")
        version = match.group(1)
        supported = version == "1.74"
    if not supported:
        return report("UNSUPPORTED", "The installed version is not supported.", version)
    if backend == "qepcad" and _qepcad_root(executable) is None:
        return report(
            "CHECK_FAILED",
            "QEPCAD support files were not found; set QEPCAD_ROOT to their directory.",
            version,
        )
    return report(
        "AVAILABLE",
        "Supported runtime detected; execution remains request-dependent.",
        version,
    )


__all__ = [
    "BackendAvailability",
    "BackendName",
    "BackendState",
    "BackendUnavailableError",
    "backend_installation",
    "check_backend",
]
