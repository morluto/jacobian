"""Probe a pinned CGAL 6.2 exact-Delaunay spike executable."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import re
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
)

from benchmarks.tooling.spike_utils import (
    default_runner,
    owned_fixture_path,
    sha256_bytes,
)

PIN_PATH = owned_fixture_path(
    __file__,
    "tests/fixtures/providers/cgal/cgal_delaunay_pin.json",
    "cgal_delaunay_pin.json",
)
ADAPTER_SOURCE = owned_fixture_path(
    __file__,
    "tests/fixtures/providers/cgal/cgal_delaunay_spike.cpp",
    "cgal_delaunay_spike.cpp",
)
_SOURCE_MEMBERS = (
    "CGAL-6.2/include/CGAL/version.h",
    "CGAL-6.2/include/CGAL/Delaunay_triangulation_2.h",
)
_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
ProcessRunner = Callable[[ToolCommandRequest], ToolCommandResult]


class CgalSpikeError(RuntimeError):
    def __init__(self, status: str, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_pin(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CgalSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The CGAL spike pin is unavailable."
        ) from exc
    if not isinstance(payload, dict) or payload.get("contract") != (
        "jacobian.cgal-delaunay-spike/v1"
    ):
        raise CgalSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The CGAL spike pin is malformed."
        )
    required = {
        "contract",
        "provider",
        "version",
        "download_url",
        "archive_sha256",
        "manual_url",
        "license",
        "supported_gnu_compiler_minimum",
        "adapter_source_sha256",
        "reproductions",
    }
    license_info = payload.get("license")
    reproductions = payload.get("reproductions")
    unique = reproductions.get("unique") if isinstance(reproductions, dict) else None
    cocircular = (
        reproductions.get("cocircular") if isinstance(reproductions, dict) else None
    )
    if (
        set(payload) != required
        or any(
            not isinstance(payload.get(key), str)
            for key in required - {"license", "reproductions"}
        )
        or not isinstance(license_info, dict)
        or set(license_info) != {"commercial_alternative", "open_source_id", "package"}
        or type(license_info.get("commercial_alternative")) is not bool
        or not all(
            isinstance(license_info.get(key), str)
            for key in ("open_source_id", "package")
        )
        or not isinstance(reproductions, dict)
        or set(reproductions) != {"unique", "cocircular"}
        or not isinstance(unique, dict)
        or set(unique)
        != {
            "command",
            "expected_output",
            "expected_output_sha256",
            "scope",
            "site_count",
        }
        or not isinstance(cocircular, dict)
        or set(cocircular) != {"command", "expected_output", "expected_output_sha256"}
        or not all(
            isinstance(case.get("command"), list)
            and all(isinstance(part, str) for part in case["command"])
            and isinstance(case.get("expected_output"), str)
            and isinstance(case.get("expected_output_sha256"), str)
            for case in (unique, cocircular)
        )
        or not isinstance(unique.get("scope"), str)
        or type(unique.get("site_count")) is not int
    ):
        raise CgalSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The CGAL spike pin is malformed."
        )
    return payload


def _resolve_file(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise OSError
    except OSError as exc:
        raise CgalSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            f"The explicitly selected {role} file is unavailable.",
        ) from exc
    return resolved


def _inspect_source_archive(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, "CGAL source archive")
    digest = _sha256_file(resolved)
    if digest != pin["archive_sha256"]:
        raise CgalSpikeError(
            "REJECTED",
            "SOURCE_VERSION_MISMATCH",
            "The CGAL source archive does not match the frozen 6.2 digest.",
        )
    try:
        with tarfile.open(resolved, mode="r:xz") as archive:
            contents: dict[str, bytes] = {}
            for name in _SOURCE_MEMBERS:
                member = archive.getmember(name)
                if not member.isfile() or member.size > 128 * 1024:
                    raise ValueError("source identity member is invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source identity member is unreadable")
                contents[name] = stream.read()
    except (KeyError, OSError, lzma.LZMAError, tarfile.TarError, ValueError) as exc:
        raise CgalSpikeError(
            "REJECTED",
            "SOURCE_ARCHIVE_MALFORMED",
            "The pinned CGAL source archive could not be inspected safely.",
        ) from exc
    if (
        b"#define CGAL_VERSION 6.2" not in contents[_SOURCE_MEMBERS[0]]
        or b"SPDX-License-Identifier: GPL-3.0-or-later OR LicenseRef-Commercial"
        not in contents[_SOURCE_MEMBERS[1]]
    ):
        raise CgalSpikeError(
            "REJECTED",
            "SOURCE_METADATA_MISMATCH",
            "CGAL version or package-license metadata differs from the pin.",
        )
    return {
        "archive": str(resolved),
        "archive_sha256": digest,
        "download_url": pin["download_url"],
        "manual_url": pin["manual_url"],
    }


def _run_checked(
    runner: ProcessRunner,
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> bytes:
    completed = runner(
        ToolCommandRequest(
            executable=command[0],
            arguments=tuple(command[1:]),
            stdin_bytes=b"",
            timeout_seconds=timeout_seconds,
            environment=_ENVIRONMENT,
            cwd=str(cwd.resolve()),
            stdout_limit_bytes=32_768,
            stderr_limit_bytes=16_384,
        )
    )
    if completed.status is ToolCommandStatus.START_FAILED:
        raise CgalSpikeError(
            "ERROR", "PROVIDER_LAUNCH_ERROR", "The CGAL spike could not be launched."
        )
    if completed.status is ToolCommandStatus.CANCELLED:
        raise CgalSpikeError(
            "CANCELLED", "PROVIDER_CANCELLED", "The CGAL spike was cancelled."
        )
    if completed.status is ToolCommandStatus.TIMED_OUT:
        raise CgalSpikeError("TIMEOUT", "PROVIDER_TIMEOUT", "The CGAL spike timed out.")
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise CgalSpikeError(
            "ERROR", "PROVIDER_OUTPUT_LIMIT", "The CGAL spike exceeded output bounds."
        )
    if completed.exit_code != 0:
        raise CgalSpikeError(
            "ERROR", "PROVIDER_CRASH", "The CGAL spike exited unsuccessfully."
        )
    return completed.stdout


def _parse_version(output: bytes, pin: Mapping[str, Any]) -> dict[str, str]:
    try:
        lines = output.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise CgalSpikeError(
            "ERROR", "PROVIDER_OUTPUT_MALFORMED", "The version output is not ASCII."
        ) from exc
    expected = f"{pin['contract']} CGAL {pin['version']}"
    if len(lines) != 3 or lines[0] != expected:
        raise CgalSpikeError(
            "REJECTED",
            "PROVIDER_VERSION_MISMATCH",
            "The executable does not expose the pinned CGAL spike protocol.",
        )
    fields = {}
    for line, name in zip(lines[1:], ("compiler", "boost"), strict=True):
        prefix = name + " "
        if not line.startswith(prefix) or len(line) == len(prefix):
            raise CgalSpikeError(
                "ERROR",
                "PROVIDER_OUTPUT_MALFORMED",
                "The CGAL toolchain probe is malformed.",
            )
        fields[name] = line[len(prefix) :]
    return fields


def _compiler_support(compiler: str, minimum: str) -> str:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", compiler)
    if match is None:
        return "UNKNOWN_COMPILER"
    observed = tuple(int(part) for part in match.groups())
    required = tuple(int(part) for part in minimum.split("."))
    return "SUPPORTED" if observed >= required else "BELOW_DOCUMENTED_SUPPORT_FLOOR"


def run_spike(
    *,
    executable: Path,
    source_archive: Path,
    cwd: Path,
    timeout_seconds: float = 5,
    runner: ProcessRunner = default_runner,
    pin_path: Path = PIN_PATH,
    adapter_source: Path = ADAPTER_SOURCE,
) -> dict[str, Any]:
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise CgalSpikeError(
                "ERROR", "INVALID_TIMEOUT", "The CGAL spike timeout must be positive."
            )
        pin = _load_pin(pin_path)
        source = _inspect_source_archive(source_archive, pin)
        resolved = _resolve_file(executable, "CGAL spike executable")
        adapter = _resolve_file(adapter_source, "CGAL spike adapter source")
        adapter_digest = _sha256_file(adapter)
        if adapter_digest != pin["adapter_source_sha256"]:
            raise CgalSpikeError(
                "REJECTED",
                "ADAPTER_SOURCE_MISMATCH",
                "The CGAL adapter source does not match the frozen pin.",
            )
        toolchain = _parse_version(
            _run_checked(
                runner,
                [str(resolved), "--version"],
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            ),
            pin,
        )
        observed: dict[str, str] = {}
        for name in ("unique", "cocircular"):
            case = pin["reproductions"][name]
            output = _run_checked(
                runner,
                [str(resolved), *case["command"]],
                cwd=cwd,
                timeout_seconds=timeout_seconds,
            )
            try:
                decoded = output.decode("ascii")
            except UnicodeDecodeError as exc:
                raise CgalSpikeError(
                    "ERROR",
                    "PROVIDER_OUTPUT_MALFORMED",
                    "CGAL returned non-ASCII reproduction output.",
                ) from exc
            if (
                decoded != case["expected_output"]
                or sha256_bytes(output) != case["expected_output_sha256"]
            ):
                raise CgalSpikeError(
                    "REJECTED",
                    "REPRODUCTION_MISMATCH",
                    f"The CGAL {name} reproduction differs from the frozen result.",
                )
            observed[name] = sha256_bytes(output)
        compiler_support = _compiler_support(
            toolchain["compiler"],
            pin["supported_gnu_compiler_minimum"],
        )
        limitations = [
            "the reproduction does not authorize CGAL as an independent checker",
            "source, adapter, and executable digests are measured but lack build attestation",
        ]
        if compiler_support == "BELOW_DOCUMENTED_SUPPORT_FLOOR":
            limitations.append(
                "the local compiler is below the CGAL 6.2 documented GNU support floor"
            )
        elif compiler_support == "UNKNOWN_COMPILER":
            limitations.append(
                "the local compiler version could not be compared with the CGAL 6.2 support floor"
            )
        return {
            "contract": pin["contract"],
            "status": "COMPLETED",
            "conclusion": "SPIKE_PASSED_PRODUCTION_DEFERRED",
            "assurance": "OBSERVED_PROVIDER_BEHAVIOR",
            "provider": {
                "name": pin["provider"],
                "version": pin["version"],
                "install_tier": "T2",
                "license": pin["license"],
                "distribution_decision": "DO_NOT_DISTRIBUTE_WITH_MIT_CORE",
                "deployment": (
                    "operator-installed subprocess after license and toolchain review"
                ),
                "source": source,
                "adapter_source_sha256": adapter_digest,
                "executable": str(resolved),
                "executable_sha256": _sha256_file(resolved),
                "toolchain": {
                    **toolchain,
                    "documented_gnu_minimum": pin["supported_gnu_compiler_minimum"],
                    "support": compiler_support,
                },
            },
            "reproductions": {
                "unique": {
                    **pin["reproductions"]["unique"],
                    "observed_output_sha256": observed["unique"],
                },
                "cocircular": {
                    **pin["reproductions"]["cocircular"],
                    "observed_output_sha256": observed["cocircular"],
                },
            },
            "checker_feasibility": {
                "decision": "REVISE",
                "independent_exact_replay": "FEASIBLE_WITH_STDLIB_RATIONAL_ARITHMETIC",
                "open_obligations": [
                    "bind canonical rational sites, duplicate policy, and degeneracy mode",
                    "verify triangle orientation, edge incidence, non-crossing, and hull coverage independently",
                    "replay exact empty-circumcircle predicates for every triangle and site",
                    "reject cocircular input under REQUIRE_UNIQUE or specify a canonical tie-break",
                    "keep any Voronoi outcome in a separate operation and artifact",
                ],
            },
            "operation_ids_registered": [],
            "limitations": limitations,
        }
    except CgalSpikeError as exc:
        return {
            "contract": "jacobian.cgal-delaunay-spike/v1",
            "status": exc.status,
            "conclusion": "NO_CONCLUSION",
            "diagnostic": {"code": exc.code, "detail": exc.detail},
            "operation_ids_registered": [],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_spike(
        executable=args.executable,
        source_archive=args.source_archive,
        cwd=args.cwd,
        timeout_seconds=args.timeout_seconds,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
