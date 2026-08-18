"""Probe Regina as an isolated producer for bounded 3-manifold outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import tarfile
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from benchmarks.tooling.spike_utils import (
    canonical_json,
    default_runner,
    sha256_bytes,
)
from tools.command_runner import (
    ToolCommandResult,
    ToolCommandStatus,
)

PIN_PATH = Path(__file__).with_name("pin.json")
ADAPTER_SOURCE = Path(__file__)
_SOURCE_ROOT = "regina-7.4.1"
_SOURCE_MEMBERS = {
    "license": f"{_SOURCE_ROOT}/LICENSE.txt",
    "core": f"{_SOURCE_ROOT}/engine/regina-core.h",
    "triangulation": (f"{_SOURCE_ROOT}/engine/triangulation/dim3/triangulation3.h"),
    "normal_surfaces": f"{_SOURCE_ROOT}/engine/surface/normalsurfaces.h",
}
# Worker re-exec replaces the process environment; keep the image PYTHONPATH so
# `tools.command_runner` remains importable inside --worker mode.
_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC",
    "PYTHONPATH": "/opt",
}
_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")
ProcessRunner = Callable[..., ToolCommandResult]
_WORKER_ERROR_PREFIX = b"JACOBIAN_SPIKE_ERROR "


class ReginaSpikeError(RuntimeError):
    """A typed non-conclusion from the optional-provider spike."""

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
        raise ReginaSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The Regina spike pin is unavailable."
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("contract") != "jacobian.regina-outcomes-spike/v1"
    ):
        raise ReginaSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The Regina spike pin is malformed."
        )
    required = {
        "contract",
        "provider",
        "distribution_version",
        "engine_version",
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
    if (
        set(payload) != required
        or any(
            not isinstance(payload.get(key), str)
            for key in required - {"source", "wheel", "reproduction"}
        )
        or not isinstance(source, dict)
        or set(source)
        != {
            "download_url",
            "signed_checksums_url",
            "archive_sha256",
            "members",
        }
        or not all(
            isinstance(source.get(key), str)
            for key in ("download_url", "signed_checksums_url", "archive_sha256")
        )
        or not isinstance(source.get("members"), dict)
        or set(source["members"]) != set(_SOURCE_MEMBERS)
        or not all(
            isinstance(member, dict)
            and set(member) == {"path", "sha256"}
            and all(isinstance(value, str) for value in member.values())
            for member in source["members"].values()
        )
        or not isinstance(wheel, dict)
        or set(wheel)
        != {
            "download_url",
            "filename",
            "sha256",
            "metadata_member",
            "metadata_sha256",
            "wheel_member",
            "wheel_sha256",
            "license_members",
        }
        or not all(
            isinstance(wheel.get(key), str) for key in set(wheel) - {"license_members"}
        )
        or not isinstance(wheel.get("license_members"), list)
        or not all(isinstance(value, str) for value in wheel["license_members"])
        or not isinstance(reproduction, dict)
        or set(reproduction)
        != {
            "scope",
            "cases",
            "normal_surface_case_id",
            "expected_provider_output",
            "expected_mathematical_output_sha256",
        }
        or not isinstance(reproduction.get("scope"), str)
        or not isinstance(reproduction.get("cases"), list)
        or not isinstance(reproduction.get("normal_surface_case_id"), str)
        or not isinstance(reproduction.get("expected_provider_output"), dict)
        or not isinstance(reproduction.get("expected_mathematical_output_sha256"), str)
    ):
        raise ReginaSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The Regina spike pin is malformed."
        )
    return payload


def _resolve_file(path: Path, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise OSError
    except OSError as exc:
        raise ReginaSpikeError(
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
        raise ReginaSpikeError(
            "UNAVAILABLE",
            "PROVIDER_FILE_UNAVAILABLE",
            "The explicitly selected Regina Python interpreter is unavailable.",
        ) from exc
    return selected


def _inspect_source_archive(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, "Regina source archive")
    source_pin = pin["source"]
    digest = _sha256_file(resolved)
    if digest != source_pin["archive_sha256"]:
        raise ReginaSpikeError(
            "REJECTED",
            "SOURCE_VERSION_MISMATCH",
            "The Regina source archive differs from the frozen 7.4.1 digest.",
        )
    try:
        with tarfile.open(resolved, mode="r:gz") as archive:
            contents: dict[str, bytes] = {}
            for role, member_name in _SOURCE_MEMBERS.items():
                member = archive.getmember(member_name)
                if not member.isfile() or member.size > 2 * 1024 * 1024:
                    raise ValueError("source identity member is invalid")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("source identity member is unreadable")
                contents[role] = stream.read()
    except (KeyError, OSError, tarfile.TarError, ValueError) as exc:
        raise ReginaSpikeError(
            "REJECTED",
            "SOURCE_ARCHIVE_MALFORMED",
            "The pinned Regina source archive could not be inspected safely.",
        ) from exc
    expected = source_pin["members"]
    if any(
        sha256_bytes(payload) != expected[role]["sha256"]
        for role, payload in contents.items()
    ) or not all(
        b"GNU General Public License" in contents[role][:2048]
        for role in ("core", "triangulation", "normal_surfaces")
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "SOURCE_METADATA_MISMATCH",
            "The Regina source or license slice differs from the pin.",
        )
    return {
        "archive": str(resolved),
        "archive_sha256": digest,
        "download_url": source_pin["download_url"],
        "signed_checksums_url": source_pin["signed_checksums_url"],
        "license_id": "GPL-2.0-or-later",
        "members": expected,
    }


def _inspect_wheel(path: Path, pin: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolve_file(path, "Regina CPython 3.12 wheel")
    wheel_pin = pin["wheel"]
    digest = _sha256_file(resolved)
    if resolved.name != wheel_pin["filename"] or digest != wheel_pin["sha256"]:
        raise ReginaSpikeError(
            "REJECTED",
            "WHEEL_VERSION_MISMATCH",
            "The Regina wheel filename or digest differs from the pin.",
        )
    try:
        with zipfile.ZipFile(resolved) as archive:
            metadata = archive.read(wheel_pin["metadata_member"])
            wheel_metadata = archive.read(wheel_pin["wheel_member"])
            license_members = sorted(
                name
                for name in archive.namelist()
                if ".dist-info/" in name and "license" in name.lower()
            )
    except (KeyError, OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ReginaSpikeError(
            "REJECTED",
            "WHEEL_MALFORMED",
            "The pinned Regina wheel could not be inspected safely.",
        ) from exc
    if (
        sha256_bytes(metadata) != wheel_pin["metadata_sha256"]
        or sha256_bytes(wheel_metadata) != wheel_pin["wheel_sha256"]
        or b"Name: regina\n" not in metadata
        or f"Version: {pin['distribution_version']}\n".encode() not in metadata
        or b"License: GPLv2+\n" not in metadata
        or b"Requires-Python: >=3.6\n" not in metadata
        or license_members != wheel_pin["license_members"]
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "WHEEL_METADATA_MISMATCH",
            "The Regina wheel metadata or license inventory differs from the pin.",
        )
    return {
        "path": str(resolved),
        "filename": resolved.name,
        "sha256": digest,
        "download_url": wheel_pin["download_url"],
        "python_tag": "cp312",
        "platform_tag": "manylinux_2_28_x86_64",
        "license_metadata": "GPLv2+",
        "license_members": license_members,
        "license_file_present": bool(license_members),
    }


def _validate_reproduction(pin: Mapping[str, Any]) -> list[dict[str, Any]]:
    reproduction = pin.get("reproduction")
    if not isinstance(reproduction, dict):
        raise ReginaSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The Regina reproduction is missing."
        )
    cases = reproduction.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8:
        raise ReginaSpikeError(
            "ERROR", "INVALID_SPIKE_PIN", "The Regina case list is malformed."
        )
    identifiers: set[str] = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or set(case) != {"case_id", "iso_sig", "maximum_tetrahedra"}
            or not isinstance(case["case_id"], str)
            or not case["case_id"]
            or case["case_id"] in identifiers
            or not isinstance(case["iso_sig"], str)
            or not 1 <= len(case["iso_sig"]) <= 128
            or type(case["maximum_tetrahedra"]) is not int
            or not 1 <= case["maximum_tetrahedra"] <= 8
        ):
            raise ReginaSpikeError(
                "ERROR", "INVALID_SPIKE_PIN", "A Regina case is malformed."
            )
        identifiers.add(case["case_id"])
    normal_case = reproduction.get("normal_surface_case_id")
    if normal_case not in identifiers:
        raise ReginaSpikeError(
            "ERROR",
            "INVALID_SPIKE_PIN",
            "The normal-surface case is not in the frozen case list.",
        )
    return cases


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
        stdout_limit=256 * 1024,
        stderr_limit=32 * 1024,
    )
    if completed.status is ToolCommandStatus.START_FAILED:
        raise ReginaSpikeError(
            "ERROR", "PROVIDER_LAUNCH_ERROR", "The Regina spike could not be launched."
        )
    if completed.status is ToolCommandStatus.CANCELLED:
        raise ReginaSpikeError(
            "CANCELLED", "PROVIDER_CANCELLED", "The Regina spike was cancelled."
        )
    if completed.status is ToolCommandStatus.TIMED_OUT:
        raise ReginaSpikeError(
            "TIMEOUT", "PROVIDER_TIMEOUT", "The Regina spike timed out."
        )
    if completed.stdout_exceeded or completed.stderr_exceeded:
        raise ReginaSpikeError(
            "ERROR", "PROVIDER_OUTPUT_LIMIT", "The Regina spike exceeded output bounds."
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
                raise ReginaSpikeError(
                    worker_error["status"],
                    worker_error["code"],
                    worker_error["detail"],
                )
        raise ReginaSpikeError(
            "ERROR", "PROVIDER_CRASH", "The Regina spike exited unsuccessfully."
        )
    return completed.stdout


def _parse_provider_output(output: bytes, pin: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(output.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReginaSpikeError(
            "ERROR",
            "PROVIDER_OUTPUT_MALFORMED",
            "The Regina provider output is not canonical ASCII JSON.",
        ) from exc
    if not isinstance(payload, dict):
        raise ReginaSpikeError(
            "ERROR", "PROVIDER_OUTPUT_MALFORMED", "The Regina output is not an object."
        )
    runtime = payload.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("distribution") != pin["distribution_version"]
        or runtime.get("engine") != pin["engine_version"]
        or not isinstance(runtime.get("python"), str)
        or not runtime["python"].startswith("3.12.")
        or runtime.get("wheel_sha256") != pin["wheel"]["sha256"]
        or type(runtime.get("verified_runtime_files")) is not int
        or runtime["verified_runtime_files"] < 1
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The worker is not running in the pinned Regina CPython boundary.",
        )
    mathematical = {key: value for key, value in payload.items() if key != "runtime"}
    expected = pin["reproduction"]["expected_provider_output"]
    if (
        mathematical != expected
        or sha256_bytes(canonical_json(mathematical))
        != pin["reproduction"]["expected_mathematical_output_sha256"]
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "REPRODUCTION_MISMATCH",
            "Regina did not reproduce the frozen bounded outcomes.",
        )
    return payload


class _UnionFind:
    def __init__(self, values: Sequence[tuple[int, int, tuple[int, ...]]]) -> None:
        self.parent = {value: value for value in values}

    def find(
        self, value: tuple[int, int, tuple[int, ...]]
    ) -> tuple[int, int, tuple[int, ...]]:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(
        self,
        left: tuple[int, int, tuple[int, ...]],
        right: tuple[int, int, tuple[int, ...]],
    ) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _validate_gluing_shape(tetrahedra: Any, gluings: Any) -> None:
    if (
        type(tetrahedra) is not int
        or not 1 <= tetrahedra <= 8
        or not isinstance(gluings, list)
        or len(gluings) != tetrahedra
        or any(not isinstance(row, list) or len(row) != 4 for row in gluings)
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "INDEPENDENT_REPLAY_MISMATCH",
            "The Regina gluing table has an invalid bounded shape.",
        )


def _gluing_is_malformed(gluing: Any) -> bool:
    return not isinstance(gluing, dict) or set(gluing) != {
        "tetrahedron",
        "facet",
        "permutation",
    }


def _gluing_out_of_scope(
    gluing: Mapping[str, Any], facet: int, tetrahedra: int
) -> bool:
    target = gluing["tetrahedron"]
    target_facet = gluing["facet"]
    permutation = gluing["permutation"]
    return (
        type(target) is not int
        or not 0 <= target < tetrahedra
        or type(target_facet) is not int
        or not 0 <= target_facet < 4
        or not isinstance(permutation, list)
        or sorted(permutation) != [0, 1, 2, 3]
        or permutation[facet] != target_facet
    )


def _gluing_not_reciprocal(
    gluings: list,
    source: int,
    facet: int,
    target: int,
    target_facet: int,
    permutation: list,
) -> bool:
    reverse = gluings[target][target_facet]
    return (
        not isinstance(reverse, dict)
        or reverse.get("tetrahedron") != source
        or reverse.get("facet") != facet
        or not isinstance(reverse.get("permutation"), list)
        or any(
            reverse["permutation"][permutation[vertex]] != vertex for vertex in range(4)
        )
    )


def _validate_facet_gluings(gluings: list, tetrahedra: int) -> None:
    for source, row in enumerate(gluings):
        for facet, gluing in enumerate(row):
            if gluing is None:
                continue
            if _gluing_is_malformed(gluing):
                raise ReginaSpikeError(
                    "REJECTED",
                    "INDEPENDENT_REPLAY_MISMATCH",
                    "A Regina facet gluing is malformed.",
                )
            if _gluing_out_of_scope(gluing, facet, tetrahedra):
                raise ReginaSpikeError(
                    "REJECTED",
                    "INDEPENDENT_REPLAY_MISMATCH",
                    "A Regina facet gluing lies outside the replay scope.",
                )
            if _gluing_not_reciprocal(
                gluings,
                source,
                facet,
                gluing["tetrahedron"],
                gluing["facet"],
                gluing["permutation"],
            ):
                raise ReginaSpikeError(
                    "REJECTED",
                    "INDEPENDENT_REPLAY_MISMATCH",
                    "The Regina facet gluings are not reciprocal.",
                )


def _compute_f_vector(gluings: list, tetrahedra: int) -> list[int]:
    nodes = [
        (tetrahedron, dimension, vertices)
        for tetrahedron in range(tetrahedra)
        for dimension in range(3)
        for vertices in combinations(range(4), dimension + 1)
    ]
    components = _UnionFind(nodes)
    for source, row in enumerate(gluings):
        for facet, gluing in enumerate(row):
            if gluing is None:
                continue
            target = gluing["tetrahedron"]
            permutation = gluing["permutation"]
            for dimension in range(3):
                for vertices in combinations(
                    [vertex for vertex in range(4) if vertex != facet],
                    dimension + 1,
                ):
                    image = tuple(sorted(permutation[vertex] for vertex in vertices))
                    components.union(
                        (source, dimension, vertices),
                        (target, dimension, image),
                    )
    f_vector = [
        len({components.find(node) for node in nodes if node[1] == dimension})
        for dimension in range(3)
    ]
    f_vector.append(tetrahedra)
    return f_vector


def _replay_triangulation(case: Mapping[str, Any]) -> dict[str, Any]:
    tetrahedra = case.get("tetrahedra")
    gluings = case.get("facet_gluings")
    _validate_gluing_shape(tetrahedra, gluings)
    _validate_facet_gluings(gluings, tetrahedra)
    f_vector = _compute_f_vector(gluings, tetrahedra)
    if case.get("f_vector") != f_vector:
        raise ReginaSpikeError(
            "REJECTED",
            "INDEPENDENT_REPLAY_MISMATCH",
            "Independent face-quotient replay rejects the Regina f-vector.",
        )
    return {
        "case_id": case.get("case_id"),
        "computed_f_vector": f_vector,
        "facet_gluing_involution": "MATCH",
    }


def _replay_normal_surfaces(
    normal: Mapping[str, Any],
    case_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    case_id = normal.get("triangulation_case_id")
    case = case_by_id.get(case_id) if isinstance(case_id, str) else None
    surfaces = normal.get("surfaces")
    if (
        case is None
        or normal.get("coordinate_system") != "STANDARD_7N"
        or normal.get("list_kind") != "VERTEX_EMBEDDED_ONLY"
        or not isinstance(surfaces, list)
        or normal.get("surface_count") != len(surfaces)
        or len(surfaces) > 128
    ):
        raise ReginaSpikeError(
            "REJECTED",
            "INDEPENDENT_REPLAY_MISMATCH",
            "The normal-surface ledger is malformed.",
        )
    tetrahedra = case["tetrahedra"]
    for surface in surfaces:
        if not isinstance(surface, dict) or set(surface) != {
            "compact",
            "connected",
            "coordinates",
            "euler_characteristic",
            "has_real_boundary",
            "orientable",
        }:
            raise ReginaSpikeError(
                "REJECTED",
                "INDEPENDENT_REPLAY_MISMATCH",
                "A normal-surface record is malformed.",
            )
        raw = surface["coordinates"]
        if (
            not isinstance(raw, list)
            or len(raw) != 7 * tetrahedra
            or any(
                not isinstance(value, str)
                or _INTEGER.fullmatch(value) is None
                or len(value) > 256
                for value in raw
            )
        ):
            raise ReginaSpikeError(
                "REJECTED",
                "INDEPENDENT_REPLAY_MISMATCH",
                "A normal coordinate vector is noncanonical or out of bounds.",
            )
        coordinates = [int(value) for value in raw]
        if not any(coordinates) or math.gcd(*coordinates) != 1:
            raise ReginaSpikeError(
                "REJECTED",
                "INDEPENDENT_REPLAY_MISMATCH",
                "A normal coordinate vector is not primitive.",
            )
        if any(
            sum(value > 0 for value in coordinates[offset + 4 : offset + 7]) > 1
            for offset in range(0, len(coordinates), 7)
        ):
            raise ReginaSpikeError(
                "REJECTED",
                "INDEPENDENT_REPLAY_MISMATCH",
                "A normal surface violates the quadrilateral constraints.",
            )
    return {
        "triangulation_case_id": case_id,
        "surface_count": len(surfaces),
        "nonnegative_primitive_vectors": "MATCH",
        "quadrilateral_constraints": "MATCH",
    }


def _independent_replay(payload: Mapping[str, Any]) -> dict[str, Any]:
    cases = payload.get("cases")
    normal = payload.get("normal_surfaces")
    if not isinstance(cases, list) or not isinstance(normal, dict):
        raise ReginaSpikeError(
            "REJECTED",
            "INDEPENDENT_REPLAY_MISMATCH",
            "The Regina mathematical output is incomplete.",
        )
    triangulations = [_replay_triangulation(case) for case in cases]
    by_id = {
        case["case_id"]: case
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("case_id"), str)
    }
    normal_replay = _replay_normal_surfaces(normal, by_id)
    return {
        "status": "PARTIAL_MATCH",
        "imports_provider": False,
        "triangulations": triangulations,
        "normal_surface_local_constraints": normal_replay,
        "not_established": [
            "isomorphism-signature canonicality",
            "integral homology from independently reconstructed chain matrices",
            "normal matching equations, vertex extremality, or enumeration completeness",
            "portable certificates for sphere or ball recognition",
        ],
    }


def run_spike(
    *,
    python_executable: Path,
    wheel: Path,
    source_archive: Path,
    timeout_seconds: float = 20,
    runner: ProcessRunner = default_runner,
    pin_path: Path = PIN_PATH,
    adapter_source: Path = ADAPTER_SOURCE,
) -> dict[str, Any]:
    """Run the bounded Regina reproduction and partial independent replay."""
    try:
        pin = _load_pin(pin_path)
        _validate_reproduction(pin)
        resolved_python = _resolve_interpreter(python_executable)
        source = _inspect_source_archive(source_archive, pin)
        wheel_identity = _inspect_wheel(wheel, pin)
        resolved_adapter = _resolve_file(adapter_source, "Regina spike adapter")
        adapter_digest = _sha256_file(resolved_adapter)
        if adapter_digest != pin["adapter_source_sha256"]:
            raise ReginaSpikeError(
                "REJECTED",
                "ADAPTER_SOURCE_MISMATCH",
                "The Regina adapter source differs from the frozen digest.",
            )
        output = _run_checked(
            runner,
            [
                str(resolved_python),
                str(resolved_adapter),
                "--worker",
                "--pin",
                str(pin_path.resolve()),
                "--wheel",
                wheel_identity["path"],
            ],
            timeout_seconds=timeout_seconds,
        )
        provider_output = _parse_provider_output(output, pin)
        mathematical = {
            key: value for key, value in provider_output.items() if key != "runtime"
        }
        replay = _independent_replay(mathematical)
        return {
            "contract": pin["contract"],
            "status": "COMPLETED",
            "conclusion": "SPIKE_PASSED_PRODUCTION_DEFERRED",
            "assurance": "OBSERVED_PROVIDER_BEHAVIOR_WITH_PARTIAL_INDEPENDENT_REPLAY",
            "provider": {
                "name": pin["provider"],
                "distribution_version": pin["distribution_version"],
                "engine_version": pin["engine_version"],
                "install_tier": "T1",
                "deployment": "operator-installed isolated CPython wheel",
                "distribution_decision": (
                    "GPL_OPTIONAL_PROVIDER_OPERATOR_APPROVAL_REQUIRED"
                ),
                "source": source,
                "wheel": wheel_identity,
                "adapter_source_sha256": adapter_digest,
                "python_executable": str(resolved_python),
                "runtime": provider_output["runtime"],
            },
            "reproduction": {
                "scope": pin["reproduction"]["scope"],
                "provider_output_sha256": sha256_bytes(output),
                "cases": provider_output["cases"],
                "normal_surfaces": provider_output["normal_surfaces"],
            },
            "independent_replay": replay,
            "outcome_gates": {
                "triangulation_materialize": {
                    "decision": "REVISE",
                    "evidence": "facet involution and quotient f-vector replayed",
                    "open_obligation": (
                        "define domain-owned gluing semantics and bind or independently "
                        "recompute canonical isomorphism signatures"
                    ),
                },
                "three_manifold_homology": {
                    "decision": "REVISE",
                    "evidence": "frozen H_1 groups reproduced",
                    "open_obligation": (
                        "expose cellular chain matrices and reuse independent certified "
                        "Smith replay instead of trusting invariant factors alone"
                    ),
                },
                "normal_surface_enumeration": {
                    "decision": "REVISE",
                    "evidence": (
                        "bounded vectors, primitivity, and quadrilateral constraints replayed"
                    ),
                    "open_obligation": (
                        "check matching equations, vertex or fundamental extremality, "
                        "enumeration completeness, and algorithm scope independently"
                    ),
                },
                "sphere_ball_recognition": {
                    "decision": "RESEARCH_ONLY",
                    "evidence": "frozen exact Regina decisions reproduced",
                    "open_obligation": (
                        "define separate decision contracts and portable certificates; "
                        "provider booleans alone cannot support VERIFIED"
                    ),
                },
            },
            "operation_ids_registered": [],
            "limitations": [
                "the PyPI wheel metadata has no bundled dist-info license file",
                "the wheel distribution is 7.4.1 while the engine reports major.minor 7.4",
                "normal-surface enumeration may be extremely slow beyond tiny cases",
                "all reproductions are answer-visible and are not held-out evaluation",
            ],
        }
    except ReginaSpikeError as exc:
        return {
            "contract": "jacobian.regina-outcomes-spike/v1",
            "status": exc.status,
            "conclusion": "NO_CONCLUSION",
            "diagnostic": {"code": exc.code, "detail": exc.detail},
            "operation_ids_registered": [],
        }


def _group_payload(group: Any) -> dict[str, Any]:
    return {
        "free_rank": group.rank(),
        "torsion_invariant_factors": [
            str(group.invariantFactor(index))
            for index in range(group.countInvariantFactors())
        ],
    }


def _gluing_payload(triangulation: Any) -> list[list[dict[str, Any] | None]]:
    result: list[list[dict[str, Any] | None]] = []
    for tetrahedron_index in range(triangulation.size()):
        tetrahedron = triangulation.tetrahedron(tetrahedron_index)
        row: list[dict[str, Any] | None] = []
        for facet in range(4):
            adjacent = tetrahedron.adjacentTetrahedron(facet)
            if adjacent is None:
                row.append(None)
                continue
            permutation = tetrahedron.adjacentGluing(facet)
            row.append(
                {
                    "tetrahedron": adjacent.index(),
                    "facet": permutation[facet],
                    "permutation": [permutation[index] for index in range(4)],
                }
            )
        result.append(row)
    return result


def _verify_installed_runtime(wheel_path: Path) -> int:
    distribution = importlib.metadata.distribution("regina")
    installed = {
        str(item).replace("\\", "/"): Path(distribution.locate_file(item))
        for item in distribution.files or ()
    }
    verified = 0
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for member in archive.namelist():
                if member.endswith("/") or ".dist-info/" in member:
                    continue
                if not member.split("/", 1)[0].startswith("regina"):
                    continue
                installed_path = installed.get(member)
                if installed_path is None or not installed_path.is_file():
                    raise ReginaSpikeError(
                        "REJECTED",
                        "PROVIDER_RUNTIME_MISMATCH",
                        "The installed Regina runtime is missing a pinned wheel member.",
                    )
                if _sha256_file(installed_path) != sha256_bytes(archive.read(member)):
                    raise ReginaSpikeError(
                        "REJECTED",
                        "PROVIDER_RUNTIME_MISMATCH",
                        "The installed Regina runtime differs from the pinned wheel.",
                    )
                verified += 1
    except (OSError, RuntimeError, zipfile.BadZipFile, zlib.error) as exc:
        raise ReginaSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The installed Regina runtime could not be bound to the pinned wheel.",
        ) from exc
    if verified < 1:
        raise ReginaSpikeError(
            "REJECTED",
            "PROVIDER_RUNTIME_MISMATCH",
            "The pinned wheel contains no Regina runtime files.",
        )
    return verified


def _worker(pin_path: Path, wheel_path: Path) -> int:
    """Run only inside the explicitly selected Regina interpreter."""
    pin = _load_pin(pin_path)
    cases = _validate_reproduction(pin)
    try:
        import regina
    except ImportError as exc:
        raise ReginaSpikeError(
            "UNAVAILABLE", "PROVIDER_IMPORT_ERROR", "Regina is unavailable."
        ) from exc
    distribution_version = importlib.metadata.version("regina")
    engine_version = regina.versionString()
    verified_runtime_files = _verify_installed_runtime(wheel_path)
    payload_cases = []
    triangulations: dict[str, Any] = {}
    for case in cases:
        triangulation = regina.Triangulation3.fromIsoSig(case["iso_sig"])
        if triangulation.size() > case["maximum_tetrahedra"]:
            raise ReginaSpikeError(
                "REJECTED",
                "REPRODUCTION_SCOPE_EXCEEDED",
                "A frozen Regina triangulation exceeds its tetrahedron bound.",
            )
        triangulations[case["case_id"]] = triangulation
        payload_cases.append(
            {
                "case_id": case["case_id"],
                "input_iso_sig": case["iso_sig"],
                "canonical_iso_sig": triangulation.isoSig(),
                "tetrahedra": triangulation.size(),
                "f_vector": [
                    triangulation.countFaces(dimension) for dimension in range(4)
                ],
                "valid": triangulation.isValid(),
                "closed": triangulation.isClosed(),
                "connected": triangulation.isConnected(),
                "orientable": triangulation.isOrientable(),
                "homology_1": _group_payload(triangulation.homology()),
                "recognition": {
                    "is_three_sphere": triangulation.isSphere(),
                    "is_three_ball": triangulation.isBall(),
                },
                "facet_gluings": _gluing_payload(triangulation),
            }
        )

    normal_case_id = pin["reproduction"]["normal_surface_case_id"]
    normal_triangulation = triangulations[normal_case_id]
    normal_list = regina.NormalSurfaces(
        normal_triangulation,
        regina.NormalCoords.Standard,
        regina.NormalList.Vertex | regina.NormalList.EmbeddedOnly,
    )
    surfaces = []
    for surface in normal_list:
        coordinates = []
        for tetrahedron in range(normal_triangulation.size()):
            coordinates.extend(
                str(surface.triangles(tetrahedron, vertex)) for vertex in range(4)
            )
            coordinates.extend(
                str(surface.quads(tetrahedron, quad_type)) for quad_type in range(3)
            )
        surfaces.append(
            {
                "coordinates": coordinates,
                "euler_characteristic": int(str(surface.eulerChar())),
                "orientable": surface.isOrientable(),
                "compact": surface.isCompact(),
                "connected": surface.isConnected(),
                "has_real_boundary": surface.hasRealBoundary(),
            }
        )
    surfaces.sort(key=lambda surface: surface["coordinates"])
    payload = {
        "contract": pin["contract"],
        "provider": pin["provider"],
        "distribution_version": distribution_version,
        "cases": payload_cases,
        "normal_surfaces": {
            "triangulation_case_id": normal_case_id,
            "coordinate_system": "STANDARD_7N",
            "list_kind": "VERTEX_EMBEDDED_ONLY",
            "observed_algorithm_flags": str(normal_list.algorithm()),
            "surface_count": len(surfaces),
            "surfaces": surfaces,
        },
        "runtime": {
            "distribution": distribution_version,
            "engine": engine_version,
            "python": sys.version.split()[0],
            "wheel_sha256": _sha256_file(wheel_path),
            "verified_runtime_files": verified_runtime_files,
        },
    }
    sys.stdout.buffer.write(canonical_json(payload))
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
        if args.wheel is None:
            parser.error("--wheel is required in worker mode")
        try:
            return _worker(args.pin, args.wheel)
        except ReginaSpikeError as exc:
            payload = {
                "status": exc.status,
                "code": exc.code,
                "detail": exc.detail,
            }
            sys.stderr.buffer.write(_WORKER_ERROR_PREFIX + canonical_json(payload))
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
