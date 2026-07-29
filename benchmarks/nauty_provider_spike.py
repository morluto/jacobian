"""Probe pinned nauty gtools without registering production capabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jacobian.bounded_process import BoundedProcessResult, run_bounded_process

PIN_PATH = Path(__file__).with_name("nauty_provider_pin.json")
_HELP_MARKERS = {
    "geng": (
        b"Usage: geng ",
        b"Generate all graphs of a specified class.",
        b"-q    : suppress auxiliary output",
    ),
    "labelg": (
        b"Usage: labelg ",
        b"Canonically label a file of graphs or digraphs.",
        b"-q  suppress auxiliary information",
    ),
}
_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
}
_SOURCE_MEMBERS = (
    "nauty2_9_3/COPYRIGHT",
    "nauty2_9_3/gtools.h",
)

ProcessRunner = Callable[..., BoundedProcessResult]


class NautySpikeError(RuntimeError):
    """A typed non-conclusion from the optional-provider spike."""

    def __init__(self, status: str, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _load_pin(path: Path = PIN_PATH) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NautySpikeError(
            "ERROR",
            "INVALID_SPIKE_PIN",
            "The checked-in nauty spike pin is unavailable or malformed.",
        ) from exc
    required = {
        "archive_sha256",
        "canonicalization",
        "contract",
        "download_url",
        "license_id",
        "manual_url",
        "provider",
        "reproduction",
        "version",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise NautySpikeError(
            "ERROR",
            "INVALID_SPIKE_PIN",
            "The checked-in nauty spike pin is malformed.",
        )
    return payload


def _resolve_file(path: Path, *, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise OSError
    except OSError as exc:
        raise NautySpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            f"The explicitly selected {role} file is unavailable.",
        ) from exc
    return resolved


def _inspect_source_archive(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, role="source archive")
    digest = _sha256_file(resolved)
    if digest != pin["archive_sha256"]:
        raise NautySpikeError(
            "REJECTED",
            "SOURCE_VERSION_MISMATCH",
            "The nauty source archive does not match the frozen 2.9.3 digest.",
        )
    try:
        with tarfile.open(resolved, mode="r:gz") as archive:
            contents: dict[str, bytes] = {}
            for member_name in _SOURCE_MEMBERS:
                member = archive.getmember(member_name)
                if not member.isfile() or member.size > 128 * 1024:
                    raise ValueError("source identity member is invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source identity member is unreadable")
                contents[member_name] = stream.read()
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise NautySpikeError(
            "REJECTED",
            "SOURCE_ARCHIVE_MALFORMED",
            "The pinned nauty source archive could not be inspected safely.",
        ) from exc
    if (
        b"Licensed under the Apache License, Version 2.0"
        not in contents[_SOURCE_MEMBERS[0]]
        or b"nauty version 2.9.3" not in contents[_SOURCE_MEMBERS[1]]
    ):
        raise NautySpikeError(
            "REJECTED",
            "SOURCE_METADATA_MISMATCH",
            "The nauty source metadata does not match the frozen version and license.",
        )
    return {
        "archive": str(resolved),
        "archive_sha256": digest,
        "download_url": pin["download_url"],
        "license_id": pin["license_id"],
        "manual_url": pin["manual_url"],
    }


def _run_checked(
    runner: ProcessRunner,
    command: Sequence[str],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    stdout_limit: int,
) -> bytes:
    try:
        completed = runner(
            command,
            input_bytes=input_bytes,
            timeout_seconds=timeout_seconds,
            environment=_ENVIRONMENT,
            stdout_limit=stdout_limit,
            stderr_limit=16_384,
        )
    except OSError as exc:
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_LAUNCH_ERROR",
            "The nauty probe process could not be launched.",
        ) from exc
    if completed.cancelled:
        raise NautySpikeError(
            "CANCELLED",
            "PROVIDER_CANCELLED",
            "The nauty probe was cancelled before a conclusion.",
        )
    if completed.timed_out:
        raise NautySpikeError(
            "TIMEOUT",
            "PROVIDER_TIMEOUT",
            "The nauty probe timed out before a conclusion.",
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_LIMIT",
            "The nauty probe exceeded its bounded output allowance.",
        )
    if completed.returncode != 0:
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_CRASH",
            "The nauty probe process exited unsuccessfully.",
        )
    return completed.stdout


def _strict_ascii_lines(payload: bytes, *, expected_width: int) -> tuple[str, ...]:
    try:
        decoded = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The nauty provider returned non-ASCII graph6 output.",
        ) from exc
    if not decoded.endswith("\n"):
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The nauty provider returned an unterminated graph6 record.",
        )
    lines = tuple(decoded.splitlines())
    if not lines or any(
        len(line) != expected_width
        or any(not 63 <= ord(character) <= 126 for character in line)
        for line in lines
    ):
        raise NautySpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The nauty provider returned malformed bounded graph6 output.",
        )
    return lines


def _probe_help(
    runner: ProcessRunner,
    executable: Path,
    *,
    tool: str,
    timeout_seconds: float,
) -> None:
    output = _run_checked(
        runner,
        [str(executable), "-help"],
        input_bytes=b"",
        timeout_seconds=timeout_seconds,
        stdout_limit=16_384,
    )
    if any(marker not in output for marker in _HELP_MARKERS[tool]):
        raise NautySpikeError(
            "REJECTED",
            "PROVIDER_FEATURE_MISMATCH",
            f"The selected {tool} executable does not expose the pinned gtools interface.",
        )


def _success_report(
    *,
    pin: Mapping[str, Any],
    source: Mapping[str, Any],
    geng: Path,
    labelg: Path,
    generated: bytes,
    canonicalized: bytes,
) -> dict[str, Any]:
    return {
        "contract": pin["contract"],
        "status": "COMPLETED",
        "conclusion": "SPIKE_PASSED_PRODUCTION_DEFERRED",
        "assurance": "OBSERVED_PROVIDER_BEHAVIOR",
        "provider": {
            "name": pin["provider"],
            "version": pin["version"],
            "install_tier": "T2",
            "deployment": "operator-installed external executables; do not vendor",
            "source": dict(source),
            "executables": {
                "geng": {
                    "path": str(geng),
                    "sha256": _sha256_file(geng),
                },
                "labelg": {
                    "path": str(labelg),
                    "sha256": _sha256_file(labelg),
                },
            },
        },
        "reproduction": {
            **pin["reproduction"],
            "observed_count": len(generated.splitlines()),
            "observed_output_sha256": _sha256_bytes(generated),
        },
        "canonicalization": {
            **pin["canonicalization"],
            "observed_output_sha256": _sha256_bytes(canonicalized),
            "isomorphic_inputs_converged": len(set(canonicalized.splitlines())) == 1,
        },
        "checker_feasibility": {
            "canonical_label": {
                "decision": "REVISE",
                "open_obligations": [
                    "bind input graph semantics, output canonical bytes, and any vertex permutation in one artifact",
                    "define canonical minimality independently of the nauty search implementation",
                    "use an operator-authorized checker that does not call nauty",
                ],
            },
            "nonisomorphic_generation": {
                "decision": "REVISE",
                "open_obligations": [
                    "make geng scope, res/mod partitioning, paging, and truncation explicit",
                    "separate pairwise non-isomorphism from exhaustive completeness",
                    "provide independent coverage evidence before claiming a complete enumeration",
                ],
            },
        },
        "capability_ids_registered": [],
        "limitations": [
            "the spike observes exact bounded outputs but does not establish mathematical verification",
            "the executable digests are measured but not cryptographically derived from the source archive",
            "no provider or checker is authorized by this benchmark",
        ],
    }


def run_spike(
    *,
    geng: Path,
    labelg: Path,
    source_archive: Path,
    timeout_seconds: float = 5,
    runner: ProcessRunner = run_bounded_process,
    pin_path: Path = PIN_PATH,
) -> dict[str, Any]:
    """Run the frozen probe and return a JSON-safe success or non-conclusion."""

    try:
        if timeout_seconds <= 0:
            raise NautySpikeError(
                "ERROR",
                "INVALID_TIMEOUT",
                "The nauty spike timeout must be positive.",
            )
        pin = _load_pin(pin_path)
        source = _inspect_source_archive(source_archive, pin)
        resolved_geng = _resolve_file(geng, role="geng executable")
        resolved_labelg = _resolve_file(labelg, role="labelg executable")
        _probe_help(
            runner,
            resolved_geng,
            tool="geng",
            timeout_seconds=timeout_seconds,
        )
        _probe_help(
            runner,
            resolved_labelg,
            tool="labelg",
            timeout_seconds=timeout_seconds,
        )

        generated = _run_checked(
            runner,
            [str(resolved_geng), "-q", "4"],
            input_bytes=b"",
            timeout_seconds=timeout_seconds,
            stdout_limit=4096,
        )
        generated_lines = _strict_ascii_lines(generated, expected_width=2)
        reproduction = pin["reproduction"]
        if (
            list(generated_lines) != reproduction["expected_graph6"]
            or _sha256_bytes(generated) != reproduction["expected_output_sha256"]
        ):
            raise NautySpikeError(
                "REJECTED",
                "REPRODUCTION_MISMATCH",
                "geng did not reproduce the frozen complete n=4 output.",
            )

        canonical_input = (
            "\n".join(pin["canonicalization"]["input_graph6"]) + "\n"
        ).encode("ascii")
        canonicalized = _run_checked(
            runner,
            [str(resolved_labelg), "-q"],
            input_bytes=canonical_input,
            timeout_seconds=timeout_seconds,
            stdout_limit=4096,
        )
        canonical_lines = _strict_ascii_lines(canonicalized, expected_width=2)
        canonicalization = pin["canonicalization"]
        if (
            list(canonical_lines) != canonicalization["expected_output_graph6"]
            or _sha256_bytes(canonicalized)
            != canonicalization["expected_output_sha256"]
        ):
            raise NautySpikeError(
                "REJECTED",
                "CANONICALIZATION_MISMATCH",
                "labelg did not reproduce the frozen isomorphic-pair output.",
            )
        return _success_report(
            pin=pin,
            source=source,
            geng=resolved_geng,
            labelg=resolved_labelg,
            generated=generated,
            canonicalized=canonicalized,
        )
    except NautySpikeError as exc:
        return {
            "contract": "jacobian.nauty-provider-spike/v1",
            "status": exc.status,
            "conclusion": "NO_CONCLUSION",
            "diagnostic": {
                "code": exc.code,
                "detail": exc.detail,
            },
            "capability_ids_registered": [],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geng", type=Path, required=True)
    parser.add_argument("--labelg", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_spike(
        geng=args.geng,
        labelg=args.labelg,
        source_archive=args.source_archive,
        timeout_seconds=args.timeout_seconds,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
