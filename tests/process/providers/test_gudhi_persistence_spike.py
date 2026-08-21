from __future__ import annotations

import json
import tarfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from benchmarks.tooling.providers.gudhi import run_spike
from tests.process.providers._spike_support import (
    _canonical,
    _ExpectedRunner,
    _request,
    _result,
    _run_expected,
    _runner,
    _sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _base_pin() -> dict[str, Any]:
    path = PROJECT_ROOT / "tests/fixtures/providers/gudhi/pin.json"
    return json.loads(path.read_text(encoding="utf-8"))


def RUN_SPIKE(**kwargs: Any) -> dict[str, Any]:  # noqa: N802
    runner = kwargs.get("runner")
    if not isinstance(runner, _ExpectedRunner):
        return run_spike(cwd=PROJECT_ROOT, **kwargs)
    python = Path(kwargs["python_executable"])
    adapter = Path(
        kwargs.get(
            "adapter_source", PROJECT_ROOT / "benchmarks/tooling/providers/gudhi.py"
        )
    )
    pin = Path(kwargs["pin_path"])
    expected = _request(
        python,
        (str(adapter.resolve()), "--worker", "--pin", str(pin.resolve())),
        environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC", "PYTHONPATH": "/opt"},
        cwd=PROJECT_ROOT,
        timeout_seconds=float(kwargs.get("timeout_seconds", 10)),
        stdout_limit_bytes=64 * 1024,
        stderr_limit_bytes=16 * 1024,
    )
    return _run_expected(
        lambda: run_spike(cwd=PROJECT_ROOT, **kwargs), runner, (expected,)
    )


def _provider_output(
    mathematical: dict[str, Any] | None = None,
    *,
    gudhi_version: str = "3.13.0",
    python_version: str = "3.12.13",
) -> bytes:
    payload = {
        **(mathematical or _base_pin()["reproduction"]["expected_provider_output"]),
        "runtime": {
            "gudhi": gudhi_version,
            "numpy": "2.4.1",
            "python": python_version,
        },
    }
    return _canonical(payload)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "gudhi-python"
    adapter = tmp_path / "gudhi-spike.py"
    python.touch()
    adapter.write_text("# fixture\n", encoding="utf-8")

    source = tmp_path / "gudhi-3.13.0-source.tar.gz"
    source_payloads = {
        (
            "gudhi-devel-tags-gudhi-release-3.13.0/src/Simplex_tree/"
            "include/gudhi/Simplex_tree.h"
        ): b"/* released under MIT */\n",
        (
            "gudhi-devel-tags-gudhi-release-3.13.0/src/Persistent_cohomology/"
            "include/gudhi/Persistent_cohomology.h"
        ): b"/* released under MIT */\n",
    }
    with tarfile.open(source, mode="w:gz") as archive:
        for name, payload in source_payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))

    license_payload = b"MIT License\nfixture\n"
    metadata = b"Name: gudhi\nVersion: 3.13.0\nRequires-Python: >=3.10\n"
    wheel = tmp_path / _base_pin()["wheel"]["filename"]
    with zipfile.ZipFile(wheel, mode="w") as archive:
        archive.writestr(_base_pin()["wheel"]["license_member"], license_payload)
        archive.writestr(_base_pin()["wheel"]["metadata_member"], metadata)

    pin = {
        **_base_pin(),
        "adapter_source_sha256": _sha256(adapter.read_bytes()),
        "source": {
            **_base_pin()["source"],
            "archive_sha256": _sha256(source.read_bytes()),
            "module_licenses": {
                "simplex_tree": {
                    "header_sha256": _sha256(
                        source_payloads[
                            (
                                "gudhi-devel-tags-gudhi-release-3.13.0/"
                                "src/Simplex_tree/include/gudhi/Simplex_tree.h"
                            )
                        ]
                    ),
                    "license_id": "MIT",
                },
                "persistent_cohomology": {
                    "header_sha256": _sha256(
                        source_payloads[
                            (
                                "gudhi-devel-tags-gudhi-release-3.13.0/"
                                "src/Persistent_cohomology/include/gudhi/"
                                "Persistent_cohomology.h"
                            )
                        ]
                    ),
                    "license_id": "MIT",
                },
            },
        },
        "wheel": {
            **_base_pin()["wheel"],
            "sha256": _sha256(wheel.read_bytes()),
            "license_sha256": _sha256(license_payload),
        },
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    return python, wheel, source, adapter, pin_path


def test_persistence_spike_rehydrates_exact_values_and_defers_production(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=_provider_output())]),
    )

    assert report["status"] == "COMPLETED"
    assert report["conclusion"] == "SPIKE_PASSED_PRODUCTION_DEFERRED"
    assert report["provider"]["install_tier"] == "T1"
    assert report["provider"]["distribution_decision"] == (
        "MIT_COMPATIBLE_OPTIONAL_PROVIDER"
    )
    assert report["reproduction"]["rank_transport"] == (
        "UNIQUE_INTEGER_RANKS_NOT_EXACT_VALUES"
    )
    assert len(report["reproduction"]["pairs"]) == 6
    assert report["reproduction"]["pairs"][0]["birth_exact_value"] == "-2/3"
    assert report["reproduction"]["pairs"][0]["death"] == {"kind": "INFINITE"}
    assert report["independent_replay"]["status"] == "MATCH"
    assert len(report["independent_replay"]["reduced_columns"]) == 11
    assert report["checker_feasibility"]["decision"] == "REVISE"
    assert report["operation_ids_registered"] == []


def test_absent_provider_is_an_explicit_non_conclusion(tmp_path: Path) -> None:
    _python, wheel, source, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=tmp_path / "absent-python",
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
    source.write_bytes(b"not the pinned archive")

    source_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
    )
    assert source_report["status"] == "REJECTED"
    assert source_report["diagnostic"]["code"] == "SOURCE_VERSION_MISMATCH"

    _python, wheel, source, adapter, pin = _fixture(tmp_path / "wheel-case")
    wheel.write_bytes(b"not the pinned wheel")
    wheel_report = RUN_SPIKE(
        python_executable=_python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
    )
    assert wheel_report["status"] == "REJECTED"
    assert wheel_report["diagnostic"]["code"] == "WHEEL_VERSION_MISMATCH"


def test_protocol_runtime_and_malformed_output_fail_closed(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)
    wrong_protocol = {
        **_base_pin()["reproduction"]["expected_provider_output"],
        "version": "3.12.0",
    }
    version_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=_provider_output(wrong_protocol))]),
    )
    assert version_report["status"] == "REJECTED"
    assert version_report["diagnostic"]["code"] == "PROVIDER_VERSION_MISMATCH"

    runtime_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=_provider_output(python_version="3.11.9"))]),
    )
    assert runtime_report["status"] == "REJECTED"
    assert runtime_report["diagnostic"]["code"] == "PROVIDER_RUNTIME_MISMATCH"

    malformed_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=b"{not-json\n")]),
    )
    assert malformed_report["status"] == "ERROR"
    assert malformed_report["diagnostic"]["code"] == "PROVIDER_OUTPUT_MALFORMED"


def test_timeout_and_crash_are_distinct_non_conclusions(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin = _fixture(tmp_path)

    timeout = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(returncode=None, timed_out=True)]),
    )
    assert timeout["status"] == "TIMEOUT"
    assert timeout["diagnostic"]["code"] == "PROVIDER_TIMEOUT"

    crash = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(returncode=7)]),
    )
    assert crash["status"] == "ERROR"
    assert crash["diagnostic"]["code"] == "PROVIDER_CRASH"


def test_independent_replay_rejects_a_self_consistent_forged_pin(
    tmp_path: Path,
) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    forged = json.loads(json.dumps(pin["reproduction"]["expected_provider_output"]))
    forged["pairs"][1]["death"] = {
        "kind": "FINITE",
        "simplex_id": "e12",
        "rank": 5,
        "exact_value": "7/13",
    }
    pin["reproduction"]["expected_provider_output"] = forged
    pin["reproduction"]["expected_mathematical_output_sha256"] = _sha256(
        _canonical(forged)
    )
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
        runner=_runner([_result(stdout=_provider_output(forged))]),
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "INDEPENDENT_REPLAY_MISMATCH"


def test_malformed_archives_do_not_escape_as_exceptions(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    source.write_bytes(b"bad gzip")
    pin["source"]["archive_sha256"] = _sha256(source.read_bytes())
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    source_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
    )
    assert source_report["status"] == "REJECTED"
    assert source_report["diagnostic"]["code"] == "SOURCE_ARCHIVE_MALFORMED"

    python, wheel, source, adapter, pin_path = _fixture(tmp_path / "wheel")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    wheel.write_bytes(b"bad zip")
    pin["wheel"]["sha256"] = _sha256(wheel.read_bytes())
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    wheel_report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
    )
    assert wheel_report["status"] == "REJECTED"
    assert wheel_report["diagnostic"]["code"] == "WHEEL_MALFORMED"


def test_incomplete_pin_is_a_typed_non_conclusion(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    del pin["wheel"]["metadata_member"]
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
            "detail": "GUDHI or NumPy is unavailable.",
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


def test_non_finite_timeout_is_rejected_before_launch(tmp_path: Path) -> None:
    python, wheel, source, adapter, pin_path = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=python,
        wheel=wheel,
        source_archive=source,
        adapter_source=adapter,
        pin_path=pin_path,
        timeout_seconds=float("inf"),
    )

    assert report["status"] == "ERROR"
    assert report["diagnostic"]["code"] == "INVALID_TIMEOUT"
