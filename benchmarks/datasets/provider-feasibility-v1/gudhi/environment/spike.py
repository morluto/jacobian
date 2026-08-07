"""Probe a pinned GUDHI persistent-homology optional-provider candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
import zipfile
import zlib
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
_SOURCE_ROOT = "gudhi-devel-tags-gudhi-release-3.13.0"
_SOURCE_MEMBERS = {
    "simplex_tree": (f"{_SOURCE_ROOT}/src/Simplex_tree/include/gudhi/Simplex_tree.h"),
    "persistent_cohomology": (
        f"{_SOURCE_ROOT}/src/Persistent_cohomology/include/gudhi/"
        "Persistent_cohomology.h"
    ),
}
# Worker re-exec replaces the process environment; keep the image PYTHONPATH so
# `benchmarks.tooling.command_runner` remains importable inside --worker mode.
_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "PYTHONPATH": "/opt",
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


class GudhiSpikeError(RuntimeError):
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
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise GudhiSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            "A selected GUDHI spike artifact could not be read.",
        ) from exc
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
        raise GudhiSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The GUDHI spike pin is unavailable."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != "jacobian.gudhi-persistence-spike/v1"
    ):
        raise GudhiSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The GUDHI spike pin is malformed."
        )
    required = {
        "contract",
        "provider",
        "version",
        "documentation_url",
        "licensing_url",
        "adapter_source_sha256",
        "source",
        "wheel",
        "reproduction",
    }
    source = payload.get("source")
    wheel = payload.get("wheel")
    reproduction = payload.get("reproduction")
    module_licenses = (
        source.get("module_licenses") if isinstance(source, dict) else None
    )
    module_licenses_valid = (
        isinstance(module_licenses, dict)
        and set(module_licenses) == set(_SOURCE_MEMBERS)
        and all(
            isinstance(module, dict)
            and set(module) == {"header_sha256", "license_id"}
            and all(isinstance(value, str) for value in module.values())
            for module in module_licenses.values()
        )
    )
    if (
        set(payload) != required
        or any(
            not isinstance(payload.get(key), str)
            for key in required - {"source", "wheel", "reproduction"}
        )
        or not isinstance(source, dict)
        or set(source)
        != {"download_url", "archive_sha256", "tag", "tag_commit", "module_licenses"}
        or not all(
            isinstance(source.get(key), str)
            for key in ("download_url", "archive_sha256", "tag", "tag_commit")
        )
        or not module_licenses_valid
        or not isinstance(wheel, dict)
        or set(wheel)
        != {
            "download_url",
            "filename",
            "sha256",
            "metadata_member",
            "license_member",
            "license_sha256",
        }
        or not all(isinstance(value, str) for value in wheel.values())
        or not isinstance(reproduction, dict)
        or set(reproduction)
        != {
            "scope",
            "coefficient_prime",
            "simplices",
            "expected_provider_output",
            "expected_mathematical_output_sha256",
        }
        or not isinstance(reproduction.get("scope"), str)
        or type(reproduction.get("coefficient_prime")) is not int
        or not isinstance(reproduction.get("simplices"), list)
        or not isinstance(reproduction.get("expected_provider_output"), dict)
        or not isinstance(reproduction.get("expected_mathematical_output_sha256"), str)
    ):
        raise GudhiSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The GUDHI spike pin is malformed."
        )
    return payload


def _resolve_file(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise OSError
    except OSError as exc:
        raise GudhiSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            f"The explicitly selected {role} file is unavailable.",
        ) from exc
    return resolved


def _resolve_interpreter(path: Path) -> Path:
    """Keep a venv launcher path while validating its resolved target."""
    selected = path.expanduser().absolute()
    try:
        target = selected.resolve(strict=True)
        if not selected.is_file() or not target.is_file():
            raise OSError
    except OSError as exc:
        raise GudhiSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            "The explicitly selected GUDHI Python interpreter is unavailable.",
        ) from exc
    return selected


def _inspect_source_archive(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, "GUDHI source archive")
    digest = _sha256_file(resolved)
    source_pin = pin["source"]
    if digest != source_pin["archive_sha256"]:
        raise GudhiSpikeError(
            "REJECTED",
            "SOURCE_VERSION_MISMATCH",
            "The GUDHI source archive does not match the frozen 3.13.0 digest.",
        )
    try:
        with tarfile.open(resolved, mode="r:gz") as archive:
            contents: dict[str, bytes] = {}
            for role, member_name in _SOURCE_MEMBERS.items():
                member = archive.getmember(member_name)
                if not member.isfile() or member.size > 512 * 1024:
                    raise ValueError("source identity member is invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source identity member is unreadable")
                contents[role] = stream.read()
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise GudhiSpikeError(
            "REJECTED",
            "SOURCE_ARCHIVE_MALFORMED",
            "The pinned GUDHI source archive could not be inspected safely.",
        ) from exc

    module_pins = source_pin["module_licenses"]
    for role, payload in contents.items():
        expected = module_pins[role]
        if (
            _sha256_bytes(payload) != expected["header_sha256"]
            or b"released under MIT" not in payload[:1024]
            or expected["license_id"] != "MIT"
        ):
            raise GudhiSpikeError(
                "REJECTED",
                "SOURCE_METADATA_MISMATCH",
                f"The GUDHI {role} module license differs from the pin.",
            )
    return {
        "archive": str(resolved),
        "archive_sha256": digest,
        "download_url": source_pin["download_url"],
        "tag": source_pin["tag"],
        "tag_commit": source_pin["tag_commit"],
        "module_licenses": module_pins,
    }


def _inspect_wheel(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, "GUDHI CPython 3.12 wheel")
    wheel_pin = pin["wheel"]
    digest = _sha256_file(resolved)
    if resolved.name != wheel_pin["filename"] or digest != wheel_pin["sha256"]:
        raise GudhiSpikeError(
            "REJECTED",
            "WHEEL_VERSION_MISMATCH",
            "The GUDHI wheel filename or digest differs from the CPython 3.12 pin.",
        )
    try:
        with zipfile.ZipFile(resolved) as archive:
            license_payload = archive.read(wheel_pin["license_member"])
            metadata = archive.read(wheel_pin["metadata_member"])
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise GudhiSpikeError(
            "REJECTED",
            "WHEEL_MALFORMED",
            "The pinned GUDHI wheel could not be inspected safely.",
        ) from exc
    if (
        _sha256_bytes(license_payload) != wheel_pin["license_sha256"]
        or not license_payload.startswith(b"MIT License\n")
        or b"Name: gudhi\n" not in metadata
        or f"Version: {pin['version']}\n".encode() not in metadata
        or b"Requires-Python: >=3.10\n" not in metadata
    ):
        raise GudhiSpikeError(
            "REJECTED",
            "WHEEL_METADATA_MISMATCH",
            "The GUDHI wheel metadata or license differs from the pin.",
        )
    return {
        "path": str(resolved),
        "filename": resolved.name,
        "sha256": digest,
        "download_url": wheel_pin["download_url"],
        "python_tag": "cp312",
        "platform_tag": "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
        "license_id": "MIT",
        "license_member": wheel_pin["license_member"],
        "license_sha256": wheel_pin["license_sha256"],
    }


def _validate_reproduction(pin: Mapping[str, Any]) -> list[dict[str, Any]]:
    reproduction = pin["reproduction"]
    simplices = reproduction["simplices"]
    if (
        reproduction.get("coefficient_prime") not in {2, 3, 5, 7}
        or not isinstance(simplices, list)
        or not 1 <= len(simplices) <= 64
    ):
        raise GudhiSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The frozen persistence case is invalid."
        )
    ids: set[str] = set()
    vertices_to_rank: dict[tuple[int, ...], int] = {}
    previous_value: Fraction | None = None
    for expected_rank, item in enumerate(simplices):
        try:
            simplex_id = item["simplex_id"]
            vertices = tuple(item["vertices"])
            rank = item["rank"]
            value = Fraction(item["exact_value"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise GudhiSpikeError(
                "ERROR", "INVALID_SPIKE_PIN", "A frozen simplex is malformed."
            ) from exc
        if (
            not isinstance(simplex_id, str)
            or not simplex_id
            or simplex_id in ids
            or rank != expected_rank
            or not vertices
            or tuple(sorted(set(vertices))) != vertices
            or (previous_value is not None and value <= previous_value)
        ):
            raise GudhiSpikeError(
                "ERROR", "INVALID_SPIKE_PIN", "The frozen filtration is not canonical."
            )
        for face_index in range(len(vertices)):
            face = vertices[:face_index] + vertices[face_index + 1 :]
            if face and face not in vertices_to_rank:
                raise GudhiSpikeError(
                    "ERROR",
                    "INVALID_SPIKE_PIN",
                    "A frozen simplex appears before one of its faces.",
                )
        ids.add(simplex_id)
        vertices_to_rank[vertices] = rank
        previous_value = value
    return simplices


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
        stdout_limit=64 * 1024,
        stderr_limit=16 * 1024,
    )
    if completed.status is ToolCommandStatus.START_FAILED:
        raise GudhiSpikeError(
            "ERROR", "PROVIDER_LAUNCH_ERROR", "The GUDHI spike could not be launched."
        )
    if completed.status is ToolCommandStatus.CANCELLED:
        raise GudhiSpikeError(
            "CANCELLED", "PROVIDER_CANCELLED", "The GUDHI spike was cancelled."
        )
    if completed.status is ToolCommandStatus.TIMED_OUT:
        raise GudhiSpikeError(
            "TIMEOUT", "PROVIDER_TIMEOUT", "The GUDHI spike timed out."
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise GudhiSpikeError(
            "ERROR", "PROVIDER_OUTPUT_LIMIT", "The GUDHI spike exceeded output bounds."
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
                raise GudhiSpikeError(
                    worker_error["status"],
                    worker_error["code"],
                    worker_error["detail"],
                )
        raise GudhiSpikeError(
            "ERROR", "PROVIDER_CRASH", "The GUDHI spike exited unsuccessfully."
        )
    return completed.stdout


def _parse_provider_output(output: bytes, pin: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(output.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GudhiSpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The GUDHI provider output is not canonical ASCII JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise GudhiSpikeError(
            "ERROR", "PROVIDER_OUTPUT_MALFORMED", "The GUDHI output is not an object."
        )
    if (
        payload.get("contract") != pin["contract"]
        or payload.get("provider") != pin["provider"]
        or payload.get("version") != pin["version"]
    ):
        raise GudhiSpikeError(
            "REJECTED",
            "PROVIDER_VERSION_MISMATCH",
            "The interpreter does not expose the pinned GUDHI spike protocol.",
        )
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("gudhi") != pin["version"]
        or not isinstance(runtime.get("python"), str)
        or not runtime["python"].startswith("3.12.")
        or not isinstance(runtime.get("numpy"), str)
        or not runtime["numpy"]
    ):
        raise GudhiSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The GUDHI worker is not running in the pinned CPython 3.12 boundary.",
        )
    expected = pin["reproduction"]["expected_provider_output"]
    mathematical_output = {
        key: value for key, value in payload.items() if key != "runtime"
    }
    if (
        mathematical_output != expected
        or _sha256_bytes(_canonical_json(mathematical_output))
        != pin["reproduction"]["expected_mathematical_output_sha256"]
    ):
        raise GudhiSpikeError(
            "REJECTED",
            "REPRODUCTION_MISMATCH",
            "GUDHI did not reproduce the frozen exact-rank persistence pairing.",
        )
    return payload


def _mod_inverse(value: int, prime: int) -> int:
    return pow(value, -1, prime)


def _independent_reduction(
    simplices: Sequence[Mapping[str, Any]], prime: int
) -> dict[str, Any]:
    """Reduce the boundary matrix without importing or calling GUDHI."""
    index_by_vertices = {
        tuple(item["vertices"]): index for index, item in enumerate(simplices)
    }
    reduced_columns: list[dict[int, int]] = []
    pivot_to_column: dict[int, int] = {}
    ledger: list[dict[str, Any]] = []

    for column_index, item in enumerate(simplices):
        vertices = tuple(item["vertices"])
        column: dict[int, int] = {}
        for removed in range(len(vertices)):
            face = vertices[:removed] + vertices[removed + 1 :]
            if not face:
                continue
            row = index_by_vertices[face]
            coefficient = 1 if removed % 2 == 0 else prime - 1
            column[row] = coefficient
        while column and max(column) in pivot_to_column:
            pivot = max(column)
            previous = reduced_columns[pivot_to_column[pivot]]
            scale = column[pivot] * _mod_inverse(previous[pivot], prime) % prime
            for row, coefficient in previous.items():
                updated = (column.get(row, 0) - scale * coefficient) % prime
                if updated:
                    column[row] = updated
                else:
                    column.pop(row, None)
        pivot = max(column) if column else None
        if pivot is not None:
            pivot_to_column[pivot] = column_index
        reduced_columns.append(column)
        ledger.append(
            {
                "column_simplex_id": item["simplex_id"],
                "pivot_simplex_id": (
                    simplices[pivot]["simplex_id"] if pivot is not None else None
                ),
                "reduced_entries": [
                    {
                        "simplex_id": simplices[row]["simplex_id"],
                        "coefficient": coefficient,
                    }
                    for row, coefficient in sorted(column.items())
                ],
            }
        )

    pairs: list[dict[str, Any]] = []
    for birth_index, item in enumerate(simplices):
        if reduced_columns[birth_index]:
            continue
        death_index = pivot_to_column.get(birth_index)
        pair = {
            "dimension": len(item["vertices"]) - 1,
            "birth_simplex_id": item["simplex_id"],
            "birth_rank": item["rank"],
            "birth_exact_value": item["exact_value"],
            "death": {"kind": "INFINITE"},
        }
        if death_index is not None:
            death = simplices[death_index]
            pair["death"] = {
                "kind": "FINITE",
                "simplex_id": death["simplex_id"],
                "rank": death["rank"],
                "exact_value": death["exact_value"],
            }
        pairs.append(pair)
    pairs.sort(
        key=lambda pair: (
            pair["dimension"],
            pair["birth_rank"],
            pair["death"].get("rank", len(simplices)),
        )
    )
    return {
        "algorithm": "STDLIB_MOD_P_BOUNDARY_COLUMN_REDUCTION",
        "coefficient_prime": prime,
        "pairs": pairs,
        "reduced_columns": ledger,
    }


def run_spike(
    *,
    python_executable: Path,
    wheel: Path,
    source_archive: Path,
    timeout_seconds: float = 10,
    runner: ProcessRunner = _default_runner,
    pin_path: Path = PIN_PATH,
    adapter_source: Path = ADAPTER_SOURCE,
) -> dict[str, Any]:
    """Run the bounded provider reproduction and independent exact replay."""
    try:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise GudhiSpikeError(
                "ERROR",
                "INVALID_TIMEOUT",
                "The GUDHI spike timeout must be finite and positive.",
            )
        pin = _load_pin(pin_path)
        resolved_python = _resolve_interpreter(python_executable)
        source = _inspect_source_archive(source_archive, pin)
        wheel_identity = _inspect_wheel(wheel, pin)
        simplices = _validate_reproduction(pin)
        resolved_adapter = _resolve_file(adapter_source, "GUDHI spike adapter")
        adapter_digest = _sha256_file(resolved_adapter)
        if adapter_digest != pin["adapter_source_sha256"]:
            raise GudhiSpikeError(
                "REJECTED",
                "ADAPTER_SOURCE_MISMATCH",
                "The GUDHI adapter source differs from the frozen digest.",
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
        prime = pin["reproduction"]["coefficient_prime"]
        independent = _independent_reduction(simplices, prime)
        if independent["pairs"] != provider_output["pairs"]:
            raise GudhiSpikeError(
                "REJECTED",
                "INDEPENDENT_REPLAY_MISMATCH",
                "Independent modular reduction rejects the GUDHI pairing.",
            )
        return {
            "contract": pin["contract"],
            "status": "COMPLETED",
            "conclusion": "SPIKE_PASSED_PRODUCTION_DEFERRED",
            "assurance": "OBSERVED_PROVIDER_BEHAVIOR_WITH_INDEPENDENT_REPLAY",
            "provider": {
                "name": pin["provider"],
                "version": pin["version"],
                "install_tier": "T1",
                "deployment": "operator-installed optional CPython wheel",
                "distribution_decision": "MIT_COMPATIBLE_OPTIONAL_PROVIDER",
                "source": source,
                "wheel": wheel_identity,
                "adapter_source_sha256": adapter_digest,
                "python_executable": str(resolved_python),
                "runtime": provider_output["runtime"],
            },
            "reproduction": {
                "scope": pin["reproduction"]["scope"],
                "coefficient_prime": prime,
                "rank_transport": "UNIQUE_INTEGER_RANKS_NOT_EXACT_VALUES",
                "exact_value_source": "FROZEN_CANONICAL_INPUT",
                "provider_output_sha256": _sha256_bytes(output),
                "pairs": provider_output["pairs"],
                "filtration": provider_output["filtration"],
            },
            "independent_replay": {
                **independent,
                "status": "MATCH",
                "imports_provider": False,
            },
            "checker_feasibility": {
                "decision": "REVISE",
                "independent_mod_p_replay": "DEMONSTRATED_FOR_BOUNDED_CASE",
                "open_obligations": [
                    "move the checker to an operator-authorized provider-independent package",
                    "bind the canonical filtered-complex artifact, prime, dimensions, and tie order",
                    "preserve provider pairings and independent reduced-column evidence as artifacts",
                    "validate exact birth/death values only from bound input simplex identifiers",
                    "represent essential intervals with a typed sentinel rather than float infinity",
                    "add adversarial pairing, rank, value, and artifact-binding mutation tests",
                ],
            },
            "capability_ids_registered": [],
            "limitations": [
                "GUDHI receives integer ranks because its Python filtration API uses float",
                "the provider is a producer and is not authorized as its own checker",
                "the spike records a wheel digest, not an installed RECORD runtime digest",
                f"only one bounded F_{prime} case was reproduced",
            ],
        }
    except GudhiSpikeError as exc:
        return {
            "contract": "jacobian.gudhi-persistence-spike/v1",
            "status": exc.status,
            "conclusion": "NO_CONCLUSION",
            "diagnostic": {"code": exc.code, "detail": exc.detail},
            "capability_ids_registered": [],
        }


def _worker(pin_path: Path) -> int:
    """Run only inside the explicitly selected provider interpreter."""
    pin = _load_pin(pin_path)
    simplices = _validate_reproduction(pin)
    try:
        import gudhi
        import numpy
    except ImportError as exc:
        raise GudhiSpikeError(
            "UNAVAILABLE", "PROVIDER_IMPORT_ERROR", "GUDHI or NumPy is unavailable."
        ) from exc
    if gudhi.__version__ != pin["version"]:
        payload = {
            "contract": pin["contract"],
            "provider": pin["provider"],
            "version": gudhi.__version__,
        }
        sys.stdout.buffer.write(_canonical_json(payload))
        return 0

    tree = gudhi.SimplexTree()
    for item in simplices:
        tree.insert(item["vertices"], filtration=float(item["rank"]))
    observed_filtration = []
    by_vertices = {tuple(item["vertices"]): item for item in simplices}
    for vertices, rank_as_float in tree.get_filtration():
        canonical = tuple(sorted(vertices))
        item = by_vertices.get(canonical)
        if (
            item is None
            or not rank_as_float.is_integer()
            or int(rank_as_float) != item["rank"]
        ):
            raise GudhiSpikeError(
                "REJECTED",
                "FILTRATION_TRANSPORT_MISMATCH",
                "GUDHI changed the unique integer-rank filtration.",
            )
        observed_filtration.append(
            {"simplex_id": item["simplex_id"], "rank": item["rank"]}
        )
    tree.persistence(
        homology_coeff_field=pin["reproduction"]["coefficient_prime"],
        min_persistence=-1,
        persistence_dim_max=True,
    )
    pairs = []
    for birth_vertices, death_vertices in tree.persistence_pairs():
        birth = by_vertices[tuple(sorted(birth_vertices))]
        pair = {
            "dimension": len(birth["vertices"]) - 1,
            "birth_simplex_id": birth["simplex_id"],
            "birth_rank": birth["rank"],
            "birth_exact_value": birth["exact_value"],
            "death": {"kind": "INFINITE"},
        }
        if death_vertices:
            death = by_vertices[tuple(sorted(death_vertices))]
            pair["death"] = {
                "kind": "FINITE",
                "simplex_id": death["simplex_id"],
                "rank": death["rank"],
                "exact_value": death["exact_value"],
            }
        pairs.append(pair)
    pairs.sort(
        key=lambda pair: (
            pair["dimension"],
            pair["birth_rank"],
            pair["death"].get("rank", len(simplices)),
        )
    )
    payload = {
        "contract": pin["contract"],
        "filtration": observed_filtration,
        "pairs": pairs,
        "provider": pin["provider"],
        "runtime": {
            "gudhi": gudhi.__version__,
            "numpy": numpy.__version__,
            "python": sys.version.split()[0],
        },
        "version": pin["version"],
    }
    sys.stdout.buffer.write(_canonical_json(payload))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--pin", type=Path, default=PIN_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args(argv)
    if args.worker:
        try:
            return _worker(args.pin)
        except GudhiSpikeError as exc:
            payload = {
                "status": exc.status,
                "code": exc.code,
                "detail": exc.detail,
            }
            sys.stderr.buffer.write(_WORKER_ERROR_PREFIX + _canonical_json(payload))
            return 64
    if (
        args.python_executable is None
        or args.wheel is None
        or args.source_archive is None
        or args.output is None
    ):
        parser.error(
            "--python-executable, --wheel, --source-archive, and --output are required"
        )
    report = run_spike(
        python_executable=args.python_executable,
        wheel=args.wheel,
        source_archive=args.source_archive,
        pin_path=args.pin,
    )
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
