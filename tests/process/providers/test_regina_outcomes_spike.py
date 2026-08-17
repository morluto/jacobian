from __future__ import annotations

import copy
import json
import runpy
import tarfile
import zipfile
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from tests.process.providers._spike_support import (
    _canonical,
    _result,
    _runner,
    _sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPIKE = runpy.run_path(
    str(PROJECT_ROOT / "tests" / "fixtures" / "providers" / "regina" / "spike.py")
)
BASE_PIN = json.loads(
    (
        PROJECT_ROOT / "tests" / "fixtures" / "providers" / "regina" / "pin.json"
    ).read_text(encoding="utf-8")
)
RunSpike = Callable[..., dict[str, Any]]
RUN_SPIKE = cast(RunSpike, SPIKE["run_spike"])


def _provider_output(
    mathematical: dict[str, Any] | None = None,
    *,
    distribution_version: str = "7.4.1",
    engine_version: str = "7.4",
    python_version: str = "3.12.13",
    wheel_sha256: str | None = None,
) -> bytes:
    payload = {
        **(mathematical or BASE_PIN["reproduction"]["expected_provider_output"]),
        "runtime": {
            "distribution": distribution_version,
            "engine": engine_version,
            "python": python_version,
            "wheel_sha256": wheel_sha256 or BASE_PIN["wheel"]["sha256"],
            "verified_runtime_files": 1,
        },
    }
    return _canonical(payload)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "regina-python"
    adapter = tmp_path / "regina-spike.py"
    python.touch()
    adapter.write_text("# fixture\n", encoding="utf-8")

    source_payloads = {
        "license": b"fixture GPL license\n",
        "core": b"GNU General Public License\nfixture core\n",
        "triangulation": b"GNU General Public License\nfixture triangulation\n",
        "normal_surfaces": b"GNU General Public License\nfixture surfaces\n",
    }
    source = tmp_path / "regina-7.4.1.tar.gz"
    members = {
        "license": "regina-7.4.1/LICENSE.txt",
        "core": "regina-7.4.1/engine/regina-core.h",
        "triangulation": ("regina-7.4.1/engine/triangulation/dim3/triangulation3.h"),
        "normal_surfaces": "regina-7.4.1/engine/surface/normalsurfaces.h",
    }
    with tarfile.open(source, mode="w:gz") as archive:
        for role, name in members.items():
            payload = source_payloads[role]
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))

    metadata = (
        b"Name: regina\nVersion: 7.4.1\nLicense: GPLv2+\nRequires-Python: >=3.6\n"
    )
    wheel_metadata = b"Wheel-Version: 1.0\nTag: cp312-cp312-manylinux_2_28_x86_64\n"
    wheel = tmp_path / BASE_PIN["wheel"]["filename"]
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(BASE_PIN["wheel"]["metadata_member"], metadata)
        archive.writestr(BASE_PIN["wheel"]["wheel_member"], wheel_metadata)

    pin = {
        **BASE_PIN,
        "adapter_source_sha256": _sha256(adapter.read_bytes()),
        "source": {
            **BASE_PIN["source"],
            "archive_sha256": _sha256(source.read_bytes()),
            "members": {
                role: {
                    "path": members[role],
                    "sha256": _sha256(source_payloads[role]),
                }
                for role in members
            },
        },
        "wheel": {
            **BASE_PIN["wheel"],
            "sha256": _sha256(wheel.read_bytes()),
            "metadata_sha256": _sha256(metadata),
            "wheel_sha256": _sha256(wheel_metadata),
        },
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    return python, wheel, source, adapter, pin_path


def test_regina_spike_replays_partial_evidence_and_defers_production(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner(
            [_result(stdout=_provider_output(wheel_sha256=_sha256(wheel.read_bytes())))]
        ),
    )

    assert report["status"] == "COMPLETED"
    assert report["conclusion"] == "SPIKE_PASSED_PRODUCTION_DEFERRED"
    assert report["provider"]["install_tier"] == "T1"
    assert report["provider"]["distribution_decision"] == (
        "GPL_OPTIONAL_PROVIDER_OPERATOR_APPROVAL_REQUIRED"
    )
    assert report["independent_replay"]["status"] == "PARTIAL_MATCH"
    assert len(report["independent_replay"]["triangulations"]) == 4
    assert (
        report["independent_replay"]["normal_surface_local_constraints"][
            "surface_count"
        ]
        == 4
    )
    assert {gate["decision"] for gate in report["outcome_gates"].values()} == {
        "REVISE",
        "RESEARCH_ONLY",
    }
    assert report["operation_ids_registered"] == []


def test_absent_regina_is_an_explicit_non_conclusion(tmp_path: Path) -> None:
    _python, wheel, source, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=tmp_path / "absent-regina-python",
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
    )

    assert report["status"] == "UNAVAILABLE"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "PROVIDER_FILE_UNAVAILABLE"
    assert report["operation_ids_registered"] == []


def test_source_and_wheel_mismatch_fail_before_execution(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)
    source.write_bytes(b"not the pinned source")
    source_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
    )
    assert source_report["status"] == "REJECTED"
    assert source_report["diagnostic"]["code"] == "SOURCE_VERSION_MISMATCH"

    python, wheel, source, adapter, pin = _fixture(tmp_path / "wheel")
    wheel.write_bytes(b"not the pinned wheel")
    wheel_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
    )
    assert wheel_report["status"] == "REJECTED"
    assert wheel_report["diagnostic"]["code"] == "WHEEL_VERSION_MISMATCH"


def test_malformed_source_and_wheel_are_typed_non_conclusions(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    source.write_bytes(b"not-a-tar")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["source"]["archive_sha256"] = _sha256(source.read_bytes())
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    source_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
    )

    python, wheel, source, adapter, pin_path = _fixture(tmp_path / "wheel")
    wheel.write_bytes(b"not-a-wheel")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["wheel"]["sha256"] = _sha256(wheel.read_bytes())
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    wheel_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
    )

    assert (source_report["status"], source_report["diagnostic"]["code"]) == (
        "REJECTED",
        "SOURCE_ARCHIVE_MALFORMED",
    )
    assert (wheel_report["status"], wheel_report["diagnostic"]["code"]) == (
        "REJECTED",
        "WHEEL_MALFORMED",
    )
    assert source_report["operation_ids_registered"] == []
    assert wheel_report["operation_ids_registered"] == []


def test_runtime_malformed_output_timeout_and_crash_fail_closed(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)
    wrong_runtime = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner(
            [_result(stdout=_provider_output(distribution_version="7.3.1.1"))]
        ),
    )
    malformed = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=b"partial Regina output\n")]),
    )
    timed_out = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(timed_out=True)]),
    )
    crashed = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(returncode=9)]),
    )

    assert wrong_runtime["diagnostic"]["code"] == "PROVIDER_RUNTIME_MISMATCH"
    assert malformed["diagnostic"]["code"] == "PROVIDER_OUTPUT_MALFORMED"
    assert (timed_out["status"], timed_out["diagnostic"]["code"]) == (
        "TIMEOUT",
        "PROVIDER_TIMEOUT",
    )
    assert (crashed["status"], crashed["diagnostic"]["code"]) == (
        "ERROR",
        "PROVIDER_CRASH",
    )
    assert all(
        report["conclusion"] == "NO_CONCLUSION"
        for report in (wrong_runtime, malformed, timed_out, crashed)
    )


def test_independent_replay_rejects_a_reciprocal_gluing_forgery(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    mathematical = copy.deepcopy(BASE_PIN["reproduction"]["expected_provider_output"])
    mathematical["cases"][0]["facet_gluings"][0][0]["tetrahedron"] = 0
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["reproduction"]["expected_provider_output"] = mathematical
    pin["reproduction"]["expected_mathematical_output_sha256"] = _sha256(
        _canonical(mathematical)
    )
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
        runner=_runner(
            [
                _result(
                    stdout=_provider_output(
                        mathematical,
                        wheel_sha256=_sha256(wheel.read_bytes()),
                    )
                )
            ]
        ),
    )

    assert report["status"] == "REJECTED"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "INDEPENDENT_REPLAY_MISMATCH"


def test_incomplete_pin_is_a_typed_non_conclusion(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    del pin["source"]["members"]
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
    )

    assert report["status"] == "ERROR"
    assert report["diagnostic"]["code"] == "INVALID_SPIKE_PIN"


def test_worker_import_failure_preserves_unavailable_status(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    worker_error = _canonical(
        {
            "status": "UNAVAILABLE",
            "code": "PROVIDER_IMPORT_ERROR",
            "detail": "Regina is unavailable.",
        }
    )

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
        runner=_runner(
            [
                _result(
                    stderr=b"JACOBIAN_SPIKE_ERROR " + worker_error,
                    returncode=64,
                )
            ]
        ),
    )

    assert report["status"] == "UNAVAILABLE"
    assert report["diagnostic"]["code"] == "PROVIDER_IMPORT_ERROR"


def test_runtime_must_bind_to_the_pinned_wheel(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    payload = json.loads(_provider_output())
    payload["runtime"]["wheel_sha256"] = "sha256:" + "0" * 64

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
        runner=_runner([_result(stdout=_canonical(payload))]),
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "PROVIDER_RUNTIME_MISMATCH"
    assert report["operation_ids_registered"] == []
