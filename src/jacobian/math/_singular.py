"""Shared hermetic runtime boundary for private Singular adapters."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from jacobian.process import (
    BoundedProcessResult,
    ProcessPlatformTools,
    ProcessResourceLimits,
    run_bounded_process,
    worker_environment,
)

SINGULAR_VERSION_MIN = 44_000
SINGULAR_VERSION_MAX = 45_000

_STDOUT_LIMIT = 512 * 1024
_STDERR_LIMIT = 64 * 1024
_SINGULAR_ARGUMENTS = (
    "-q",
    "-t",
    "--no-rc",
    "--no-shell",
    "--no-stdlib",
)


class UnsupportedSingularVersionError(ValueError):
    """The child identified an installed release outside the supported range."""


@dataclass(slots=True)
class SingularProtocolReader:
    """Read one line-oriented, adapter-owned Singular protocol."""

    lines: list[str]
    cursor: int = 0

    def pop(self) -> str:
        if self.cursor >= len(self.lines):
            raise ValueError("Singular output ended unexpectedly")
        line = self.lines[self.cursor]
        self.cursor += 1
        return line

    def expect(self, expected: str) -> None:
        if self.pop() != expected:
            raise ValueError(f"Singular output is missing {expected!r}")

    def finished(self) -> bool:
        return self.cursor == len(self.lines)


def singular_version_preamble(protocol_header: str) -> tuple[str, ...]:
    """Emit the protocol identity and reject unsupported releases before algebra."""

    return (
        f'print("{protocol_header}");',
        'int jacobian_runtime_version=system("version");',
        "print(jacobian_runtime_version);",
        f"if (jacobian_runtime_version < {SINGULAR_VERSION_MIN})",
        "{",
        "  quit;",
        "}",
        f"if (jacobian_runtime_version >= {SINGULAR_VERSION_MAX})",
        "{",
        "  quit;",
        "}",
    )


def read_singular_version(
    reader: SingularProtocolReader,
    *,
    protocol_header: str,
) -> int:
    """Parse and enforce the numeric release from one protocol preamble."""

    reader.expect(protocol_header)
    try:
        version = int(reader.pop())
    except ValueError as exc:
        raise ValueError("Singular output has an invalid numeric version") from exc
    if not SINGULAR_VERSION_MIN <= version < SINGULAR_VERSION_MAX:
        raise UnsupportedSingularVersionError(
            "installed Singular release is outside the supported range"
        )
    return version


def format_singular_version(version: int) -> str:
    """Format Singular's numeric capability version for private diagnostics."""

    major, remainder = divmod(version, 10_000)
    minor, patch_code = divmod(remainder, 1_000)
    patch = patch_code // 100
    return f"{major}.{minor}.{patch}"


def run_bounded_singular(
    source: bytes,
    *,
    wall_seconds: int,
) -> BoundedProcessResult | None:
    """Run one request-scoped Singular program, or return ``None`` if unavailable."""

    executable = shutil.which("Singular")
    if executable is None:
        return None
    executable = str(Path(executable).resolve())
    prlimit = shutil.which("prlimit")
    if prlimit is not None:
        prlimit = str(Path(prlimit).resolve())
    try:
        with tempfile.TemporaryDirectory(prefix="jacobian-singular-") as directory:
            return run_bounded_process(
                [executable, *_SINGULAR_ARGUMENTS],
                input_bytes=source,
                timeout_seconds=float(wall_seconds),
                environment=worker_environment(locale="C.UTF-8"),
                stdout_limit=_STDOUT_LIMIT,
                stderr_limit=_STDERR_LIMIT,
                resource_limits=ProcessResourceLimits(
                    cpu_seconds=wall_seconds,
                    address_space_bytes=1024 * 1024 * 1024,
                    file_size_bytes=1024 * 1024,
                ),
                platform_tools=ProcessPlatformTools(prlimit_executable=prlimit),
                cwd=directory,
            )
    except OSError:
        return None


__all__ = [
    "SingularProtocolReader",
    "UnsupportedSingularVersionError",
    "format_singular_version",
    "read_singular_version",
    "run_bounded_singular",
    "singular_version_preamble",
]
