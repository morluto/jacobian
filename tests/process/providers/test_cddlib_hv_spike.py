from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any

from tests.fixtures.providers.cddlib.spike import _expected_mathematical, run_spike
from tests.process.providers._spike_support import (
    _canonical,
    _result,
    _runner,
    _sha256,
)
from tools.command_runner import ToolCommandResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_PIN = json.loads(
    (
        PROJECT_ROOT / "tests" / "fixtures" / "providers" / "cddlib" / "pin.json"
    ).read_text(encoding="utf-8")
)

EXPECTED_MATHEMATICAL = _expected_mathematical(BASE_PIN)


RUN_SPIKE = run_spike


def _provider_output(
    mathematical: dict[str, Any] | None = None,
    *,
    python_version: str = "3.12.13",
    pycddlib_version: str = "3.0.2",
) -> bytes:
    payload = {
        **(mathematical or EXPECTED_MATHEMATICAL),
        "runtime": {
            "distribution_record_sha256": "sha256:" + "1" * 64,
            "gmp_module_sha256": "sha256:" + "2" * 64,
            "number_type": "fractions.Fraction",
            "pycddlib": pycddlib_version,
            "python": python_version,
        },
    }
    return _canonical(payload)


def _archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, mode="w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    python = tmp_path / "pycddlib-python"
    adapter = tmp_path / "cddlib-spike.py"
    python.touch()
    adapter.write_text("# fixture\n", encoding="utf-8")

    cdd_members = {
        name: "\n".join(expected["required_ascii_markers"]).encode("ascii")
        for name, expected in BASE_PIN["sources"]["cddlib"]["identity_members"].items()
    }
    pycdd_members = {
        name: "\n".join(expected["required_ascii_markers"]).encode("ascii")
        for name, expected in BASE_PIN["sources"]["pycddlib"][
            "identity_members"
        ].items()
    }
    cdd_source = tmp_path / "cddlib-0.94n.tar.gz"
    pycdd_source = tmp_path / "pycddlib-3.0.2.tar.gz"
    _archive(cdd_source, cdd_members)
    _archive(pycdd_source, pycdd_members)

    pin = {
        **BASE_PIN,
        "adapter_source_sha256": _sha256(adapter.read_bytes()),
        "sources": {
            "cddlib": {
                **BASE_PIN["sources"]["cddlib"],
                "archive_sha256": _sha256(cdd_source.read_bytes()),
                "identity_members": {
                    name: {
                        **BASE_PIN["sources"]["cddlib"]["identity_members"][name],
                        "sha256": _sha256(payload),
                    }
                    for name, payload in cdd_members.items()
                },
            },
            "pycddlib": {
                **BASE_PIN["sources"]["pycddlib"],
                "archive_sha256": _sha256(pycdd_source.read_bytes()),
                "identity_members": {
                    name: {
                        **BASE_PIN["sources"]["pycddlib"]["identity_members"][name],
                        "sha256": _sha256(payload),
                    }
                    for name, payload in pycdd_members.items()
                },
            },
        },
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    return python, cdd_source, pycdd_source, adapter, pin_path


def test_exact_hv_spike_keeps_production_deferred(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
        runner=_runner([_result(stdout=_provider_output())]),
        pin_path=pin,
        adapter_source=adapter,
    )

    assert report["status"] == "COMPLETED"
    assert report["conclusion"] == "SPIKE_PASSED_PRODUCTION_DEFERRED"
    assert report["provider"]["distribution_decision"] == (
        "GPL_OPTIONAL_PROVIDER_NOT_CORE_DEPENDENCY"
    )
    assert report["reproduction"]["exact_arithmetic"]["roundtrip_exact"] is True
    assert [
        item["exact_constraint_generator_checks"]
        for item in report["independent_replay"]["cases"]
    ] == [4, 6, 4, 6]
    assert report["independent_replay"]["completeness"] == "NOT_ESTABLISHED"
    assert report["checker_feasibility"]["decision"] == "REVISE"
    assert report["operation_ids_registered"] == []


def test_absent_provider_is_an_explicit_non_conclusion(tmp_path: Path) -> None:
    report = RUN_SPIKE(
        python_executable=tmp_path / "absent-python",
        cddlib_source_archive=tmp_path / "absent-cddlib.tar.gz",
        pycddlib_source_archive=tmp_path / "absent-pycddlib.tar.gz",
    )

    assert report == {
        "contract": "jacobian.cddlib-hv-spike/v1",
        "status": "UNAVAILABLE",
        "conclusion": "NO_CONCLUSION",
        "diagnostic": {
            "code": "PROVIDER_FILE_UNAVAILABLE",
            "detail": (
                "The explicitly selected pycddlib Python interpreter is unavailable."
            ),
        },
        "operation_ids_registered": [],
    }


def test_source_mismatch_fails_before_execution(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin = _fixture(tmp_path)
    cdd_source.write_bytes(b"wrong source")
    calls = 0

    def runner(*_args: object, **_kwargs: object) -> ToolCommandResult:
        nonlocal calls
        calls += 1
        return _result(stdout=_provider_output())

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
        runner=runner,
        pin_path=pin,
        adapter_source=adapter,
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "SOURCE_VERSION_MISMATCH"
    assert calls == 0


def test_protocol_runtime_and_malformed_output_fail_closed(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin = _fixture(tmp_path)
    common = {
        "python_executable": python,
        "cddlib_source_archive": cdd_source,
        "pycddlib_source_archive": pycdd_source,
        "pin_path": pin,
        "adapter_source": adapter,
    }

    malformed = RUN_SPIKE(runner=_runner([_result(stdout=b"not-json")]), **common)
    wrong_version = {
        **EXPECTED_MATHEMATICAL,
        "versions": {"cddlib": "0.94n", "pycddlib": "9.9.9"},
    }
    version = RUN_SPIKE(
        runner=_runner([_result(stdout=_provider_output(wrong_version))]), **common
    )
    runtime = RUN_SPIKE(
        runner=_runner([_result(stdout=_provider_output(python_version="3.11.99"))]),
        **common,
    )

    assert malformed["status"] == "ERROR"
    assert malformed["diagnostic"]["code"] == "PROVIDER_OUTPUT_MALFORMED"
    assert version["status"] == "REJECTED"
    assert version["diagnostic"]["code"] == "PROVIDER_VERSION_MISMATCH"
    assert runtime["status"] == "REJECTED"
    assert runtime["diagnostic"]["code"] == "PROVIDER_RUNTIME_MISMATCH"


def test_timeout_and_crash_are_distinct_non_conclusions(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin = _fixture(tmp_path)
    common = {
        "python_executable": python,
        "cddlib_source_archive": cdd_source,
        "pycddlib_source_archive": pycdd_source,
        "pin_path": pin,
        "adapter_source": adapter,
    }

    timeout = RUN_SPIKE(
        runner=_runner([_result(returncode=None, timed_out=True)]), **common
    )
    crash = RUN_SPIKE(runner=_runner([_result(returncode=7)]), **common)

    assert timeout["status"] == "TIMEOUT"
    assert timeout["diagnostic"]["code"] == "PROVIDER_TIMEOUT"
    assert crash["status"] == "ERROR"
    assert crash["diagnostic"]["code"] == "PROVIDER_CRASH"


def test_independent_replay_rejects_self_consistent_unsound_output(
    tmp_path: Path,
) -> None:
    python, cdd_source, pycdd_source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pin["reproduction"]["cases"][0]["expected_output"] = {
        "representation": "V",
        "homogeneous_rows": [["1", "0", "0"], ["0", "1", "0"]],
        "linearity_rows": [],
    }
    forged = _expected_mathematical(pin)
    pin["reproduction"]["expected_mathematical_output_sha256"] = _sha256(
        _canonical(forged)
    )
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
        runner=_runner([_result(stdout=_provider_output(forged))]),
        pin_path=pin_path,
        adapter_source=adapter,
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "INDEPENDENT_REPLAY_MISMATCH"


def test_incomplete_pin_is_a_typed_non_conclusion(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    del pin["sources"]["cddlib"]["identity_members"]
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
        adapter_source=adapter,
        pin_path=pin_path,
    )

    assert report["status"] == "ERROR"
    assert report["diagnostic"]["code"] == "INVALID_SPIKE_PIN"


def test_worker_import_failure_preserves_unavailable_status(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin_path = _fixture(tmp_path)
    worker_error = _canonical(
        {
            "status": "UNAVAILABLE",
            "code": "PROVIDER_IMPORT_ERROR",
            "detail": "The pycddlib cdd.gmp module is unavailable.",
        }
    )

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
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
    assert report["operation_ids_registered"] == []


def test_malformed_archives_do_not_escape_as_exceptions(tmp_path: Path) -> None:
    python, cdd_source, pycdd_source, adapter, pin_path = _fixture(tmp_path)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    pycdd_source.write_bytes(b"not-a-tar")
    pin["sources"]["pycddlib"]["archive_sha256"] = _sha256(pycdd_source.read_bytes())
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        python_executable=python,
        cddlib_source_archive=cdd_source,
        pycddlib_source_archive=pycdd_source,
        pin_path=pin_path,
        adapter_source=adapter,
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "SOURCE_ARCHIVE_MALFORMED"
