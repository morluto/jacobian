"""Probe pinned cddlib/pycddlib exact H/V conversion without registration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks.tooling.command_runner import (
    ToolCommandRequest,
    ToolCommandResult,
    ToolCommandStatus,
    run_tool_command,
)

PIN_PATH = Path(__file__).with_name("pin.json")
ADAPTER_SOURCE = Path(__file__)
# Worker re-exec replaces the process environment; keep the image PYTHONPATH so
# `benchmarks.tooling.command_runner` remains importable inside --worker mode.
_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "PYTHONPATH": "/opt",
}
_KIND_ORDER = {
    "EQUALITY": 0,
    "INEQUALITY": 1,
    "VERTEX": 2,
    "RAY": 3,
    "LINEALITY": 4,
}
ProcessRunner = Callable[..., ToolCommandResult]
_WORKER_ERROR_PREFIX = b"JACOBIAN_SPIKE_ERROR "


def _default_runner(
    command: Sequence[str],
    *,
    input_bytes: bytes,
    timeout_seconds: float,
    environment: Mapping[str, str],
    stdout_limit: int,
    stderr_limit: int,
) -> ToolCommandResult:
    request = ToolCommandRequest(
        executable=command[0],
        arguments=tuple(command[1:]),
        environment=environment,
        cwd=str(Path.cwd()),
        timeout_seconds=timeout_seconds,
        stdin_bytes=input_bytes,
        stdout_limit_bytes=stdout_limit,
        stderr_limit_bytes=stderr_limit,
    )
    return run_tool_command(request)


class CddlibSpikeError(RuntimeError):
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


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _load_pin(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CddlibSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The cddlib spike pin is unavailable."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != "jacobian.cddlib-hv-spike/v1"
    ):
        raise CddlibSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The cddlib spike pin is malformed."
        )
    required = {
        "contract",
        "provider",
        "versions",
        "sources",
        "reproduction",
        "adapter_source_sha256",
    }
    versions = payload.get("versions")
    sources = payload.get("sources")
    reproduction = payload.get("reproduction")
    scope = reproduction.get("scope") if isinstance(reproduction, dict) else None
    scope_valid = (
        isinstance(scope, dict)
        and set(scope)
        == {
            "ambient_dimension",
            "case_count",
            "covers",
            "exact_input",
            "max_homogeneous_rows_per_case",
        }
        and type(scope.get("ambient_dimension")) is int
        and type(scope.get("case_count")) is int
        and type(scope.get("max_homogeneous_rows_per_case")) is int
        and isinstance(scope.get("exact_input"), str)
        and isinstance(scope.get("covers"), list)
        and all(isinstance(item, str) for item in scope["covers"])
    )
    sources_valid = isinstance(sources, dict) and set(sources) == {
        "cddlib",
        "pycddlib",
    }
    if sources_valid:
        for source in sources.values():
            if (
                not isinstance(source, dict)
                or set(source)
                != {
                    "download_url",
                    "archive_sha256",
                    "tag",
                    "tag_commit",
                    "license_id",
                    "identity_members",
                }
                or not all(
                    isinstance(source.get(key), str)
                    for key in (
                        "download_url",
                        "archive_sha256",
                        "tag",
                        "tag_commit",
                        "license_id",
                    )
                )
                or not isinstance(source.get("identity_members"), dict)
                or not source["identity_members"]
            ):
                sources_valid = False
                break
            for member_name, member in source["identity_members"].items():
                if (
                    not isinstance(member_name, str)
                    or not isinstance(member, dict)
                    or set(member) != {"sha256", "max_bytes", "required_ascii_markers"}
                    or not isinstance(member.get("sha256"), str)
                    or type(member.get("max_bytes")) is not int
                    or not isinstance(member.get("required_ascii_markers"), list)
                    or not all(
                        isinstance(marker, str)
                        for marker in member["required_ascii_markers"]
                    )
                ):
                    sources_valid = False
                    break
    if (
        set(payload) != required
        or not isinstance(payload.get("provider"), str)
        or not isinstance(payload.get("adapter_source_sha256"), str)
        or not isinstance(versions, dict)
        or set(versions) != {"cddlib", "pycddlib"}
        or not all(isinstance(value, str) for value in versions.values())
        or not sources_valid
        or not isinstance(reproduction, dict)
        or set(reproduction)
        != {
            "scope",
            "cases",
            "exact_arithmetic_probe",
            "expected_mathematical_output_sha256",
        }
        or not scope_valid
        or not isinstance(reproduction.get("cases"), list)
        or not isinstance(reproduction.get("exact_arithmetic_probe"), str)
        or not isinstance(reproduction.get("expected_mathematical_output_sha256"), str)
    ):
        raise CddlibSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The cddlib spike pin is malformed."
        )
    return payload


def _resolve_file(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise OSError
    except OSError as exc:
        raise CddlibSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            f"The explicitly selected {role} file is unavailable.",
        ) from exc
    return resolved


def _resolve_interpreter(path: Path) -> Path:
    selected = path.expanduser().absolute()
    try:
        target = selected.resolve(strict=True)
        if not selected.is_file() or not target.is_file():
            raise OSError
    except OSError as exc:
        raise CddlibSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            "The explicitly selected pycddlib Python interpreter is unavailable.",
        ) from exc
    return selected


def _inspect_archive(
    path: Path,
    *,
    role: str,
    source_pin: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = _resolve_file(path, role)
    digest = _sha256_file(resolved)
    if digest != source_pin["archive_sha256"]:
        raise CddlibSpikeError(
            "REJECTED",
            "SOURCE_VERSION_MISMATCH",
            f"The {role} does not match its frozen digest.",
        )
    try:
        with tarfile.open(resolved, mode="r:gz") as archive:
            contents: dict[str, bytes] = {}
            for member_name, expected in source_pin["identity_members"].items():
                member = archive.getmember(member_name)
                if not member.isfile() or member.size > expected["max_bytes"]:
                    raise ValueError("source identity member is invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source identity member is unreadable")
                contents[member_name] = stream.read()
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise CddlibSpikeError(
            "REJECTED",
            "SOURCE_ARCHIVE_MALFORMED",
            f"The pinned {role} could not be inspected safely.",
        ) from exc

    for member_name, payload in contents.items():
        expected = source_pin["identity_members"][member_name]
        if _sha256_bytes(payload) != expected["sha256"]:
            raise CddlibSpikeError(
                "REJECTED",
                "SOURCE_METADATA_MISMATCH",
                f"The pinned identity member {member_name} differs.",
            )
        for marker in expected.get("required_ascii_markers", []):
            if marker.encode("ascii") not in payload:
                raise CddlibSpikeError(
                    "REJECTED",
                    "SOURCE_METADATA_MISMATCH",
                    f"The pinned identity member {member_name} lost a required marker.",
                )
    return {
        "archive": str(resolved),
        "archive_sha256": digest,
        "download_url": source_pin["download_url"],
        "tag": source_pin["tag"],
        "tag_commit": source_pin["tag_commit"],
        "license_id": source_pin["license_id"],
        "identity_members": source_pin["identity_members"],
    }


def _as_fraction(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("exact values must be integers or rational strings")
    return Fraction(value)


def _validate_cases(pin: Mapping[str, Any]) -> list[dict[str, Any]]:
    reproduction = pin.get("reproduction")
    cases = reproduction.get("cases") if isinstance(reproduction, dict) else None
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise CddlibSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The bounded H/V cases are invalid."
        )
    observed_ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for case in cases:
        try:
            case_id = case["case_id"]
            representation = case["input"]["representation"]
            rows = case["input"]["homogeneous_rows"]
            linearity_rows = case["input"]["linearity_rows"]
            ambient_dimension = case["ambient_dimension"]
        except (KeyError, TypeError) as exc:
            raise CddlibSpikeError(
                "ERROR", "INVALID_SPIKE_PIN", "A bounded H/V case is malformed."
            ) from exc
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in observed_ids
            or representation not in {"H", "V"}
            or not isinstance(ambient_dimension, int)
            or not 1 <= ambient_dimension <= 4
            or not isinstance(rows, list)
            or not 1 <= len(rows) <= 16
            or not isinstance(linearity_rows, list)
            or any(
                not isinstance(index, int) or index < 0 or index >= len(rows)
                for index in linearity_rows
            )
            or len(set(linearity_rows)) != len(linearity_rows)
        ):
            raise CddlibSpikeError(
                "ERROR", "INVALID_SPIKE_PIN", "A bounded H/V case is invalid."
            )
        exact_rows: list[list[Fraction]] = []
        try:
            for row in rows:
                if not isinstance(row, list) or len(row) != ambient_dimension + 1:
                    raise ValueError
                exact_rows.append([_as_fraction(value) for value in row])
        except (ValueError, ZeroDivisionError) as exc:
            raise CddlibSpikeError(
                "ERROR",
                "INVALID_SPIKE_PIN",
                "A bounded H/V row is not an exact homogeneous row.",
            ) from exc
        observed_ids.add(case_id)
        validated.append(
            {
                "case_id": case_id,
                "ambient_dimension": ambient_dimension,
                "representation": representation,
                "rows": exact_rows,
                "linearity_rows": set(linearity_rows),
            }
        )
    if {case["representation"] for case in validated} != {"H", "V"}:
        raise CddlibSpikeError(
            "ERROR",
            "INVALID_SPIKE_PIN",
            "The spike must exercise both H-to-V and V-to-H conversion.",
        )
    return validated


def _integer_normalize(row: Sequence[Fraction], *, sign_free: bool) -> list[Fraction]:
    denominator_lcm = 1
    for value in row:
        denominator_lcm = math.lcm(denominator_lcm, value.denominator)
    integers = [
        value.numerator * (denominator_lcm // value.denominator) for value in row
    ]
    divisor = 0
    for value in integers:
        divisor = math.gcd(divisor, abs(value))
    if divisor == 0:
        return [Fraction(0) for _ in row]
    integers = [value // divisor for value in integers]
    if sign_free:
        first = next((value for value in integers if value), 0)
        if first < 0:
            integers = [-value for value in integers]
    return [Fraction(value) for value in integers]


def _rank(rows: Sequence[Sequence[Fraction]]) -> int:
    if not rows:
        return 0
    matrix = [list(row) for row in rows]
    row_count = len(matrix)
    column_count = len(matrix[0])
    pivot_row = 0
    for column in range(column_count):
        pivot = next(
            (index for index in range(pivot_row, row_count) if matrix[index][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[pivot_row], matrix[pivot] = matrix[pivot], matrix[pivot_row]
        scale = matrix[pivot_row][column]
        matrix[pivot_row] = [value / scale for value in matrix[pivot_row]]
        for index in range(row_count):
            if index == pivot_row or not matrix[index][column]:
                continue
            factor = matrix[index][column]
            matrix[index] = [
                value - factor * basis
                for value, basis in zip(matrix[index], matrix[pivot_row], strict=True)
            ]
        pivot_row += 1
        if pivot_row == row_count:
            break
    return pivot_row


def _fraction_strings(values: Sequence[Fraction]) -> list[str]:
    return [str(value) for value in values]


def _summarize(
    representation: str,
    rows: Sequence[Sequence[Fraction]],
    linearity_rows: set[int],
    ambient_dimension: int,
) -> dict[str, Any]:
    if any(len(row) != ambient_dimension + 1 for row in rows):
        raise ValueError("wrong homogeneous row width")
    entries: list[tuple[str, list[Fraction]]] = []
    if representation == "H":
        for index, row in enumerate(rows):
            kind = "EQUALITY" if index in linearity_rows else "INEQUALITY"
            normalized = _integer_normalize(row, sign_free=kind == "EQUALITY")
            if not any(normalized):
                raise ValueError("zero H row")
            entries.append((kind, normalized))
        equality_normals = [row[1:] for kind, row in entries if kind == "EQUALITY"]
        affine_dimension = ambient_dimension - _rank(equality_normals)
    elif representation == "V":
        vertices: list[list[Fraction]] = []
        directions: list[list[Fraction]] = []
        for index, row in enumerate(rows):
            leading = row[0]
            if index in linearity_rows:
                if leading:
                    raise ValueError("lineality row has nonzero homogenizing value")
                kind = "LINEALITY"
                normalized = _integer_normalize(row, sign_free=True)
            elif leading == 1:
                kind = "VERTEX"
                normalized = list(row)
            elif not leading:
                kind = "RAY"
                normalized = _integer_normalize(row, sign_free=False)
            else:
                raise ValueError("V row is not normalized to homogenizing value 0 or 1")
            if not any(normalized):
                raise ValueError("zero V row")
            entries.append((kind, normalized))
            if kind == "VERTEX":
                vertices.append(normalized[1:])
            else:
                directions.append(normalized[1:])
        base = vertices[0] if vertices else [Fraction(0)] * ambient_dimension
        directions.extend(
            [
                [value - origin for value, origin in zip(vertex, base, strict=True)]
                for vertex in vertices[1:]
            ]
        )
        affine_dimension = _rank(directions)
    else:
        raise ValueError("unknown representation")

    entries.sort(
        key=lambda item: (
            _KIND_ORDER[item[0]],
            tuple(item[1]),
        )
    )
    return {
        "representation": representation,
        "ambient_dimension": ambient_dimension,
        "affine_dimension": affine_dimension,
        "homogeneous_convention": (
            "H:[b,a] means b+a*x>=0; H linearity means equality; "
            "V:[1,x] vertex; V:[0,d] ray; V linearity means span(d)"
        ),
        "homogeneous_rows": [
            {"kind": kind, "row": _fraction_strings(row)} for kind, row in entries
        ],
        "equalities": [
            _fraction_strings(row) for kind, row in entries if kind == "EQUALITY"
        ],
        "inequalities": [
            _fraction_strings(row) for kind, row in entries if kind == "INEQUALITY"
        ],
        "vertices": [
            _fraction_strings(row[1:]) for kind, row in entries if kind == "VERTEX"
        ],
        "rays": [_fraction_strings(row[1:]) for kind, row in entries if kind == "RAY"],
        "lineality": [
            _fraction_strings(row[1:]) for kind, row in entries if kind == "LINEALITY"
        ],
    }


def _parse_summary_rows(summary: Mapping[str, Any]) -> list[tuple[str, list[Fraction]]]:
    try:
        return [
            (entry["kind"], [Fraction(value) for value in entry["row"]])
            for entry in summary["homogeneous_rows"]
        ]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise CddlibSpikeError(
            "REJECTED",
            "PROVIDER_OUTPUT_MALFORMED",
            "The provider returned a malformed exact H/V summary.",
        ) from exc


def _expected_mathematical(pin: Mapping[str, Any]) -> dict[str, Any]:
    cases = _validate_cases(pin)
    case_pins = {item["case_id"]: item for item in pin["reproduction"]["cases"]}
    expected_cases = []
    for case in cases:
        expected_output = case_pins[case["case_id"]]["expected_output"]
        output_rows = [
            [_as_fraction(value) for value in row]
            for row in expected_output["homogeneous_rows"]
        ]
        output_summary = _summarize(
            expected_output["representation"],
            output_rows,
            set(expected_output["linearity_rows"]),
            case["ambient_dimension"],
        )
        input_summary = _summarize(
            case["representation"],
            case["rows"],
            case["linearity_rows"],
            case["ambient_dimension"],
        )
        expected_cases.append(
            {
                "case_id": case["case_id"],
                "conversion": (
                    f"{case['representation']}_TO_{expected_output['representation']}"
                ),
                "input": input_summary,
                "output": output_summary,
                "same_provider_roundtrip": {
                    "status": "MATCH",
                    "result": input_summary,
                    "independent": False,
                },
            }
        )
    probe = pin["reproduction"]["exact_arithmetic_probe"]
    return {
        "contract": pin["contract"],
        "provider": pin["provider"],
        "versions": pin["versions"],
        "exact_arithmetic": {
            "module": "cdd.gmp",
            "number_type": "fractions.Fraction",
            "large_fraction_input": probe,
            "large_fraction_output": probe,
            "roundtrip_exact": True,
        },
        "cases": expected_cases,
    }


def _independent_soundness(
    case: Mapping[str, Any], output: Mapping[str, Any]
) -> dict[str, Any]:
    """Check one containment direction using only stdlib Fraction arithmetic."""
    input_representation = case["representation"]
    input_summary = _summarize(
        input_representation,
        case["rows"],
        case["linearity_rows"],
        case["ambient_dimension"],
    )
    output_rows = _parse_summary_rows(output)
    checks = 0

    if input_representation == "H":
        constraints = _parse_summary_rows(input_summary)
        for output_kind, output_row in output_rows:
            if output_kind not in {"VERTEX", "RAY", "LINEALITY"}:
                raise CddlibSpikeError(
                    "REJECTED",
                    "INDEPENDENT_REPLAY_MISMATCH",
                    "H-to-V output contains a non-generator row.",
                )
            point_or_direction = output_row[1:]
            for constraint_kind, constraint in constraints:
                value = sum(
                    (
                        coefficient * coordinate
                        for coefficient, coordinate in zip(
                            constraint[1:], point_or_direction, strict=True
                        )
                    ),
                    constraint[0] if output_kind == "VERTEX" else Fraction(0),
                )
                if constraint_kind == "EQUALITY" or output_kind == "LINEALITY":
                    accepted = value == 0
                else:
                    accepted = value >= 0
                checks += 1
                if not accepted:
                    raise CddlibSpikeError(
                        "REJECTED",
                        "INDEPENDENT_REPLAY_MISMATCH",
                        "A provider generator violates an input H row.",
                    )
        direction_summary = output
    else:
        generators = _parse_summary_rows(input_summary)
        for output_kind, output_row in output_rows:
            if output_kind not in {"EQUALITY", "INEQUALITY"}:
                raise CddlibSpikeError(
                    "REJECTED",
                    "INDEPENDENT_REPLAY_MISMATCH",
                    "V-to-H output contains a non-constraint row.",
                )
            for generator_kind, generator in generators:
                value = sum(
                    (
                        coefficient * coordinate
                        for coefficient, coordinate in zip(
                            output_row[1:], generator[1:], strict=True
                        )
                    ),
                    output_row[0] if generator_kind == "VERTEX" else Fraction(0),
                )
                if output_kind == "EQUALITY" or generator_kind == "LINEALITY":
                    accepted = value == 0
                else:
                    accepted = value >= 0
                checks += 1
                if not accepted:
                    raise CddlibSpikeError(
                        "REJECTED",
                        "INDEPENDENT_REPLAY_MISMATCH",
                        "An output H row excludes an input generator.",
                    )
        direction_summary = input_summary

    if output.get("affine_dimension") != direction_summary["affine_dimension"]:
        raise CddlibSpikeError(
            "REJECTED",
            "INDEPENDENT_REPLAY_MISMATCH",
            "The provider affine dimension disagrees with exact rank replay.",
        )
    return {
        "case_id": case["case_id"],
        "status": "SOUNDNESS_MATCH",
        "exact_constraint_generator_checks": checks,
        "affine_dimension": output["affine_dimension"],
        "imports_provider": False,
        "completeness": "NOT_ESTABLISHED",
    }


def _run_checked(
    runner: ProcessRunner,
    command: Sequence[str],
    *,
    timeout_seconds: float,
) -> bytes:
    completed = runner(
        command,
        input_bytes=b"",
        timeout_seconds=timeout_seconds,
        environment=_ENVIRONMENT,
        stdout_limit=128 * 1024,
        stderr_limit=16 * 1024,
    )
    if completed.status is ToolCommandStatus.START_FAILED:
        raise CddlibSpikeError(
            "ERROR",
            "PROVIDER_LAUNCH_ERROR",
            "The pycddlib spike could not be launched.",
        )
    if completed.status is ToolCommandStatus.CANCELLED:
        raise CddlibSpikeError(
            "CANCELLED", "PROVIDER_CANCELLED", "The pycddlib spike was cancelled."
        )
    if completed.status is ToolCommandStatus.TIMED_OUT:
        raise CddlibSpikeError(
            "TIMEOUT", "PROVIDER_TIMEOUT", "The pycddlib spike timed out."
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise CddlibSpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_LIMIT",
            "The pycddlib spike exceeded output bounds.",
        )
    if completed.exit_code != 0:
        if completed.stderr.startswith(_WORKER_ERROR_PREFIX):
            try:
                worker_error = json.loads(
                    completed.stderr[len(_WORKER_ERROR_PREFIX) :].decode("ascii")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                worker_error = None
            if (
                isinstance(worker_error, dict)
                and set(worker_error) == {"status", "code", "detail"}
                and all(isinstance(value, str) for value in worker_error.values())
            ):
                raise CddlibSpikeError(
                    worker_error["status"],
                    worker_error["code"],
                    worker_error["detail"],
                )
        raise CddlibSpikeError(
            "ERROR", "PROVIDER_CRASH", "The pycddlib spike exited unsuccessfully."
        )
    return completed.stdout


def _parse_provider_output(output: bytes, pin: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(output.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CddlibSpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The pycddlib output is not canonical ASCII JSON.",
        ) from exc
    if not isinstance(payload, dict) or _canonical_json(payload) != output:
        raise CddlibSpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The pycddlib output is not one canonical JSON object.",
        )
    if (
        payload.get("contract") != pin["contract"]
        or payload.get("provider") != pin["provider"]
        or payload.get("versions") != pin["versions"]
    ):
        raise CddlibSpikeError(
            "REJECTED",
            "PROVIDER_VERSION_MISMATCH",
            "The interpreter does not expose the pinned cddlib spike protocol.",
        )
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("pycddlib") != pin["versions"]["pycddlib"]
        or runtime.get("number_type") != "fractions.Fraction"
        or not isinstance(runtime.get("python"), str)
        or not runtime["python"].startswith("3.12.")
        or not isinstance(runtime.get("gmp_module_sha256"), str)
        or not isinstance(runtime.get("distribution_record_sha256"), str)
    ):
        raise CddlibSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The worker is not the pinned CPython 3.12 cdd.gmp runtime.",
        )
    mathematical = {key: value for key, value in payload.items() if key != "runtime"}
    reproduction = pin["reproduction"]
    if (
        mathematical != _expected_mathematical(pin)
        or _sha256_bytes(_canonical_json(mathematical))
        != reproduction["expected_mathematical_output_sha256"]
    ):
        raise CddlibSpikeError(
            "REJECTED",
            "REPRODUCTION_MISMATCH",
            "pycddlib did not reproduce the frozen exact H/V conversions.",
        )
    return payload


def run_spike(
    *,
    python_executable: Path,
    cddlib_source_archive: Path,
    pycddlib_source_archive: Path,
    timeout_seconds: float = 10,
    runner: ProcessRunner = _default_runner,
    pin_path: Path = PIN_PATH,
    adapter_source: Path = ADAPTER_SOURCE,
) -> dict[str, Any]:
    """Run the bounded provider reproduction and independent soundness replay."""
    try:
        pin = _load_pin(pin_path)
        resolved_python = _resolve_interpreter(python_executable)
        cases = _validate_cases(pin)
        cddlib_source = _inspect_archive(
            cddlib_source_archive,
            role="cddlib source archive",
            source_pin=pin["sources"]["cddlib"],
        )
        pycddlib_source = _inspect_archive(
            pycddlib_source_archive,
            role="pycddlib source archive",
            source_pin=pin["sources"]["pycddlib"],
        )
        resolved_adapter = _resolve_file(adapter_source, "cddlib spike adapter")
        adapter_digest = _sha256_file(resolved_adapter)
        if adapter_digest != pin["adapter_source_sha256"]:
            raise CddlibSpikeError(
                "REJECTED",
                "ADAPTER_SOURCE_MISMATCH",
                "The cddlib adapter source differs from the frozen digest.",
            )
        output = _run_checked(
            runner,
            [
                str(resolved_python),
                str(resolved_adapter),
                "--worker",
                "--pin",
                str(pin_path.resolve()),
            ],
            timeout_seconds=timeout_seconds,
        )
        provider_output = _parse_provider_output(output, pin)
        by_id = {case["case_id"]: case for case in cases}
        independent = [
            _independent_soundness(
                by_id[observed["case_id"]],
                observed["output"],
            )
            for observed in provider_output["cases"]
        ]
        return {
            "contract": pin["contract"],
            "status": "COMPLETED",
            "conclusion": "SPIKE_PASSED_PRODUCTION_DEFERRED",
            "assurance": "OBSERVED_EXACT_PROVIDER_BEHAVIOR_WITH_SOUNDNESS_REPLAY",
            "provider": {
                "name": pin["provider"],
                "versions": pin["versions"],
                "install_tier": "T1",
                "deployment": (
                    "operator-installed source build of GPL cddlib plus pycddlib"
                ),
                "distribution_decision": "GPL_OPTIONAL_PROVIDER_NOT_CORE_DEPENDENCY",
                "sources": {
                    "cddlib": cddlib_source,
                    "pycddlib": pycddlib_source,
                },
                "adapter_source_sha256": adapter_digest,
                "python_executable": str(resolved_python),
                "runtime": provider_output["runtime"],
            },
            "reproduction": {
                "scope": pin["reproduction"]["scope"],
                "provider_output_sha256": _sha256_bytes(output),
                "exact_arithmetic": provider_output["exact_arithmetic"],
                "cases": provider_output["cases"],
            },
            "independent_replay": {
                "algorithm": "STDLIB_FRACTION_CONSTRAINT_GENERATOR_REPLAY",
                "status": "SOUNDNESS_MATCH",
                "cases": independent,
                "completeness": "NOT_ESTABLISHED",
            },
            "checker_feasibility": {
                "decision": "REVISE",
                "soundness_replay": "DEMONSTRATED_FOR_BOUNDED_CASES",
                "same_provider_roundtrip": "EVIDENCE_ONLY_NOT_INDEPENDENT",
                "open_obligations": [
                    "freeze separate typed H and V artifacts with affine-hull semantics",
                    "bind ambient and affine dimensions plus homogeneous normalization",
                    "separate output soundness from reverse-containment completeness",
                    "add independent Farkas, extremality, and lineality certificates",
                    "authorize a checker package independent of cddlib and pycddlib",
                    "measure the installed cddlib shared-library identity at runtime",
                    "add adversarial omitted-facet, omitted-ray, sign, and rebinding cases",
                ],
            },
            "capability_ids_registered": [],
            "limitations": [
                "all four cases are answer-visible bounded reproductions",
                "same-provider round trips do not independently establish completeness",
                "stdlib replay proves only that returned generators/constraints are sound",
                "pycddlib exposes no cddlib runtime version API",
                "GPL source builds remain operator-installed and outside core dependencies",
            ],
        }
    except CddlibSpikeError as exc:
        return {
            "contract": "jacobian.cddlib-hv-spike/v1",
            "status": exc.status,
            "conclusion": "NO_CONCLUSION",
            "diagnostic": {"code": exc.code, "detail": exc.detail},
            "capability_ids_registered": [],
        }


def _worker(pin_path: Path) -> int:
    """Run only inside the explicitly selected pycddlib interpreter."""
    pin = _load_pin(pin_path)
    cases = _validate_cases(pin)
    try:
        import importlib.metadata

        import cdd.gmp as cdd
    except ImportError as exc:
        raise CddlibSpikeError(
            "UNAVAILABLE",
            "PROVIDER_IMPORT_ERROR",
            "The pycddlib cdd.gmp module is unavailable.",
        ) from exc
    installed_version = importlib.metadata.version("pycddlib")
    if installed_version != pin["versions"]["pycddlib"]:
        sys.stdout.buffer.write(
            _canonical_json(
                {
                    "contract": pin["contract"],
                    "provider": pin["provider"],
                    "versions": {
                        **pin["versions"],
                        "pycddlib": installed_version,
                    },
                }
            )
        )
        return 0

    observed_cases: list[dict[str, Any]] = []
    for case in cases:
        representation = (
            cdd.RepType.INEQUALITY
            if case["representation"] == "H"
            else cdd.RepType.GENERATOR
        )
        matrix = cdd.matrix_from_array(
            case["rows"],
            lin_set=case["linearity_rows"],
            rep_type=representation,
        )
        polyhedron = cdd.polyhedron_from_matrix(matrix)
        output_matrix = cdd.copy_output(polyhedron)
        cdd.matrix_canonicalize(output_matrix)
        output_representation = "V" if case["representation"] == "H" else "H"
        output_summary = _summarize(
            output_representation,
            output_matrix.array,
            set(output_matrix.lin_set),
            case["ambient_dimension"],
        )
        roundtrip_polyhedron = cdd.polyhedron_from_matrix(output_matrix)
        roundtrip_matrix = cdd.copy_output(roundtrip_polyhedron)
        cdd.matrix_canonicalize(roundtrip_matrix)
        roundtrip_summary = _summarize(
            case["representation"],
            roundtrip_matrix.array,
            set(roundtrip_matrix.lin_set),
            case["ambient_dimension"],
        )
        input_summary = _summarize(
            case["representation"],
            case["rows"],
            case["linearity_rows"],
            case["ambient_dimension"],
        )
        observed_cases.append(
            {
                "case_id": case["case_id"],
                "conversion": f"{case['representation']}_TO_{output_representation}",
                "input": input_summary,
                "output": output_summary,
                "same_provider_roundtrip": {
                    "status": (
                        "MATCH" if roundtrip_summary == input_summary else "MISMATCH"
                    ),
                    "result": roundtrip_summary,
                    "independent": False,
                },
            }
        )

    huge = Fraction(pin["reproduction"]["exact_arithmetic_probe"])
    probe_matrix = cdd.matrix_from_array(
        [[huge, Fraction(1)]],
        rep_type=cdd.RepType.INEQUALITY,
    )
    exact_probe = probe_matrix.array[0][0]
    gmp_module = Path(cdd.__file__).resolve()
    distribution = importlib.metadata.distribution("pycddlib")
    record_path = next(
        (
            Path(distribution.locate_file(item))
            for item in distribution.files or ()
            if item.name == "RECORD"
        ),
        None,
    )
    if record_path is None or not record_path.is_file():
        raise CddlibSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The installed pycddlib distribution RECORD is unavailable.",
        )
    payload = {
        "contract": pin["contract"],
        "provider": pin["provider"],
        "versions": pin["versions"],
        "exact_arithmetic": {
            "module": "cdd.gmp",
            "number_type": "fractions.Fraction",
            "large_fraction_input": str(huge),
            "large_fraction_output": str(exact_probe),
            "roundtrip_exact": exact_probe == huge,
        },
        "cases": observed_cases,
        "runtime": {
            "python": sys.version.split()[0],
            "pycddlib": installed_version,
            "number_type": f"{cdd.NumberType.__module__}.{cdd.NumberType.__name__}",
            "gmp_module_sha256": _sha256_file(gmp_module),
            "distribution_record_sha256": _sha256_file(record_path),
        },
    }
    sys.stdout.buffer.write(_canonical_json(payload))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--cddlib-source-archive", type=Path)
    parser.add_argument("--pycddlib-source-archive", type=Path)
    parser.add_argument("--pin", type=Path, default=PIN_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        try:
            return _worker(args.pin)
        except CddlibSpikeError as exc:
            payload = {
                "status": exc.status,
                "code": exc.code,
                "detail": exc.detail,
            }
            sys.stderr.buffer.write(_WORKER_ERROR_PREFIX + _canonical_json(payload))
            return 64
    if (
        args.python_executable is None
        or args.cddlib_source_archive is None
        or args.pycddlib_source_archive is None
        or args.output is None
    ):
        parser.error(
            "--python-executable, --cddlib-source-archive, "
            "--pycddlib-source-archive, and --output are required"
        )
    report = run_spike(
        python_executable=args.python_executable,
        cddlib_source_archive=args.cddlib_source_archive,
        pycddlib_source_archive=args.pycddlib_source_archive,
        pin_path=args.pin,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
