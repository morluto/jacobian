from __future__ import annotations

import hashlib
import json
import lzma
import runpy
import tarfile
from collections.abc import Callable, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, cast

from jacobian.bounded_process import BoundedProcessResult

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPIKE = runpy.run_path(str(PROJECT_ROOT / "benchmarks" / "cgal_delaunay_spike.py"))
BASE_PIN = json.loads(
    (PROJECT_ROOT / "benchmarks" / "cgal_delaunay_pin.json").read_text(encoding="utf-8")
)
RunSpike = Callable[..., dict[str, Any]]
RUN_SPIKE = cast(RunSpike, SPIKE["run_spike"])


def _result(
    *,
    stdout: bytes = b"",
    returncode: int | None = 0,
    timed_out: bool = False,
) -> BoundedProcessResult:
    return BoundedProcessResult(
        returncode=returncode,
        stdout=stdout,
        stderr=b"",
        stdout_exceeded=False,
        stderr_exceeded=False,
        timed_out=timed_out,
    )


def _runner(
    outcomes: Sequence[BoundedProcessResult],
) -> Callable[..., BoundedProcessResult]:
    remaining = iter(outcomes)

    def run(*_args: object, **_kwargs: object) -> BoundedProcessResult:
        return next(remaining)

    return run


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    executable = tmp_path / "cgal-spike"
    adapter = tmp_path / "cgal-spike.cpp"
    executable.touch()
    adapter.write_text("// fixture\n", encoding="utf-8")
    archive = tmp_path / "CGAL-6.2-library.tar.xz"
    with tarfile.open(archive, mode="w:xz") as bundle:
        members = (
            (
                "CGAL-6.2/include/CGAL/version.h",
                b"#define CGAL_VERSION 6.2\n",
            ),
            (
                "CGAL-6.2/include/CGAL/Delaunay_triangulation_2.h",
                (
                    b"SPDX-License-Identifier: GPL-3.0-or-later "
                    b"OR LicenseRef-Commercial\n"
                ),
            ),
        )
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            bundle.addfile(info, BytesIO(payload))
    pin = {
        **BASE_PIN,
        "archive_sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
        "adapter_source_sha256": (
            "sha256:" + hashlib.sha256(adapter.read_bytes()).hexdigest()
        ),
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    return executable, archive, adapter, pin_path


def _successes() -> list[BoundedProcessResult]:
    return [
        _result(
            stdout=(
                b"jacobian.cgal-delaunay-spike/v1 CGAL 6.2\n"
                b"compiler 13.3.0\n"
                b"boost 1_79\n"
            )
        ),
        _result(stdout=BASE_PIN["reproductions"]["unique"]["expected_output"].encode()),
        _result(
            stdout=BASE_PIN["reproductions"]["cocircular"]["expected_output"].encode()
        ),
    ]


def test_exact_delaunay_spike_passes_but_defers_production(tmp_path: Path) -> None:
    executable, archive, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner(_successes()),
    )

    assert report["status"] == "COMPLETED"
    assert report["conclusion"] == "SPIKE_PASSED_PRODUCTION_DEFERRED"
    assert report["provider"]["license"]["open_source_id"] == "GPL-3.0-or-later"
    assert report["provider"]["distribution_decision"] == (
        "DO_NOT_DISTRIBUTE_WITH_MIT_CORE"
    )
    assert report["provider"]["toolchain"]["support"] == "SUPPORTED"
    assert report["checker_feasibility"]["decision"] == "REVISE"
    assert report["capability_ids_registered"] == []


def test_absent_provider_is_an_explicit_non_conclusion(tmp_path: Path) -> None:
    _executable, archive, adapter, pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        executable=tmp_path / "absent-cgal-spike",
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
    )

    assert report["status"] == "UNAVAILABLE"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "PROVIDER_FILE_UNAVAILABLE"
    assert report["capability_ids_registered"] == []


def test_source_version_mismatch_fails_before_execution(tmp_path: Path) -> None:
    executable, archive, adapter, _pin = _fixture(tmp_path)

    report = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "SOURCE_VERSION_MISMATCH"


def test_protocol_version_and_malformed_result_fail_closed(tmp_path: Path) -> None:
    executable, archive, adapter, pin = _fixture(tmp_path)
    wrong_version = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(stdout=b"CGAL 6.1\n")]),
    )
    malformed = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner(
            [
                _successes()[0],
                _result(stdout=b"partial triangulation\n"),
            ]
        ),
    )

    assert wrong_version["diagnostic"]["code"] == "PROVIDER_VERSION_MISMATCH"
    assert malformed["diagnostic"]["code"] == "REPRODUCTION_MISMATCH"
    assert wrong_version["conclusion"] == malformed["conclusion"] == "NO_CONCLUSION"


def test_timeout_and_crash_are_distinct_non_conclusions(tmp_path: Path) -> None:
    executable, archive, adapter, pin = _fixture(tmp_path)
    timed_out = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(timed_out=True)]),
    )
    crashed = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin,
        runner=_runner([_result(returncode=9)]),
    )

    assert (timed_out["status"], timed_out["diagnostic"]["code"]) == (
        "TIMEOUT",
        "PROVIDER_TIMEOUT",
    )
    assert (crashed["status"], crashed["diagnostic"]["code"]) == (
        "ERROR",
        "PROVIDER_CRASH",
    )


def test_malformed_xz_archive_does_not_escape_as_an_exception(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "cgal-spike"
    adapter = tmp_path / "adapter.cpp"
    archive = tmp_path / "CGAL-6.2-library.tar.xz"
    executable.touch()
    adapter.touch()
    archive.write_bytes(lzma.compress(b"not a tar archive"))
    pin = {
        **BASE_PIN,
        "archive_sha256": "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest(),
    }
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    report = RUN_SPIKE(
        executable=executable,
        source_archive=archive,
        adapter_source=adapter,
        pin_path=pin_path,
    )

    assert report["status"] == "REJECTED"
    assert report["diagnostic"]["code"] == "SOURCE_ARCHIVE_MALFORMED"
