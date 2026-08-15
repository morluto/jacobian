from __future__ import annotations

import hashlib
import json
import math
import runpy
import tarfile
from collections.abc import Callable, Sequence
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest
from benchmarks.tooling.command_runner import ToolCommandResult, ToolCommandStatus

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SPIKE = runpy.run_path(
    str(PROJECT_ROOT / "tests" / "fixtures" / "providers" / "nauty" / "spike.py")
)
PIN = json.loads(
    (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "providers"
        / "nauty"
        / "nauty_provider_pin.json"
    ).read_text(encoding="utf-8")
)
RunSpike = Callable[..., dict[str, Any]]
RUN_SPIKE = cast(RunSpike, SPIKE["run_spike"])

GENG_HELP = (
    b"Usage: geng \n"
    b"Generate all graphs of a specified class.\n"
    b"-q    : suppress auxiliary output\n"
)
LABELG_HELP = (
    b"Usage: labelg \n"
    b"Canonically label a file of graphs or digraphs.\n"
    b"-q  suppress auxiliary information\n"
)
GENG_OUTPUT = ("\n".join(PIN["reproduction"]["expected_graph6"]) + "\n").encode()
LABELG_OUTPUT = (
    "\n".join(PIN["canonicalization"]["expected_output_graph6"]) + "\n"
).encode()


def _result(
    *,
    stdout: bytes = b"",
    returncode: int | None = 0,
    timed_out: bool = False,
    cancelled: bool = False,
    stdout_exceeded: bool = False,
) -> ToolCommandResult:
    if cancelled:
        status = ToolCommandStatus.CANCELLED
    elif timed_out:
        status = ToolCommandStatus.TIMED_OUT
    elif stdout_exceeded:
        status = ToolCommandStatus.OUTPUT_LIMIT_EXCEEDED
    else:
        status = ToolCommandStatus.EXITED
    return ToolCommandResult(
        status=status,
        exit_code=returncode,
        stdout=stdout,
        stderr=b"",
        stdout_exceeded=stdout_exceeded,
    )


def _runner(
    outcomes: Sequence[ToolCommandResult],
) -> Callable[..., ToolCommandResult]:
    remaining = iter(outcomes)

    def run(*_args: object, **_kwargs: object) -> ToolCommandResult:
        return next(remaining)

    return run


def _source_archive(tmp_path: Path, *, pin_digest: bool = True) -> Path:
    archive_path = tmp_path / "nauty2_9_3.tar.gz"
    copyright_notice = b"Licensed under the Apache License, Version 2.0 (the License)\n"
    version_header = b"/* nauty version 2.9.3 */\n"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, payload in (
            ("nauty2_9_3/COPYRIGHT", copyright_notice),
            ("nauty2_9_3/gtools.h", version_header),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))
    if pin_digest:
        digest = "sha256:" + hashlib.sha256(archive_path.read_bytes()).hexdigest()
        PIN["archive_sha256"] = digest
    return archive_path


def _files(tmp_path: Path) -> tuple[Path, Path]:
    geng = tmp_path / "geng"
    labelg = tmp_path / "labelg"
    geng.touch()
    labelg.touch()
    return geng, labelg


def _pin_file(tmp_path: Path, pin: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(PIN if pin is None else pin), encoding="utf-8")
    return path


def _successful_outcomes() -> list[ToolCommandResult]:
    return [
        _result(stdout=GENG_HELP),
        _result(stdout=LABELG_HELP),
        _result(stdout=GENG_OUTPUT),
        _result(stdout=LABELG_OUTPUT),
    ]


def test_spike_records_exact_reproduction_and_defers_production(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=_runner(_successful_outcomes()),
        pin_path=_pin_file(tmp_path),
    )

    assert report["status"] == "COMPLETED"
    assert report["conclusion"] == "SPIKE_PASSED_PRODUCTION_DEFERRED"
    assert report["provider"]["version"] == "2.9.3"
    assert report["provider"]["source"]["license_id"] == "Apache-2.0"
    assert report["reproduction"]["observed_count"] == 11
    assert report["canonicalization"]["isomorphic_inputs_converged"] is True
    assert report["checker_feasibility"]["canonical_label"]["decision"] == "REVISE"
    assert (
        report["checker_feasibility"]["nonisomorphic_generation"]["decision"]
        == "REVISE"
    )
    assert report["operation_ids_registered"] == []


def test_absent_explicit_executable_is_an_isolated_non_conclusion(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    _geng, labelg = _files(tmp_path)

    report = RUN_SPIKE(
        geng=tmp_path / "absent-geng",
        labelg=labelg,
        source_archive=archive,
        pin_path=_pin_file(tmp_path),
    )

    assert report == {
        "contract": "jacobian.nauty-provider-spike/v1",
        "status": "UNAVAILABLE",
        "conclusion": "NO_CONCLUSION",
        "diagnostic": {
            "code": "PROVIDER_FILE_UNAVAILABLE",
            "detail": "The explicitly selected geng executable file is unavailable.",
        },
        "operation_ids_registered": [],
    }


def test_source_version_mismatch_fails_closed_before_execution(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path, pin_digest=False)
    geng, labelg = _files(tmp_path)

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
    )

    assert report["status"] == "REJECTED"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "SOURCE_VERSION_MISMATCH"


def test_invalid_timeout_fails_before_provider_inspection(tmp_path: Path) -> None:
    report = RUN_SPIKE(
        geng=tmp_path / "absent-geng",
        labelg=tmp_path / "absent-labelg",
        source_archive=tmp_path / "absent-nauty.tar.gz",
        timeout_seconds=0,
    )

    assert report["status"] == "ERROR"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "INVALID_TIMEOUT"


@pytest.mark.parametrize("timeout_seconds", [math.inf, math.nan])
def test_nonfinite_timeout_fails_before_provider_inspection(
    tmp_path: Path,
    timeout_seconds: float,
) -> None:
    report = RUN_SPIKE(
        geng=tmp_path / "absent-geng",
        labelg=tmp_path / "absent-labelg",
        source_archive=tmp_path / "absent-nauty.tar.gz",
        timeout_seconds=timeout_seconds,
    )

    assert report["status"] == "ERROR"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "INVALID_TIMEOUT"


def test_malformed_nested_pin_fails_as_json_safe_non_conclusion(tmp_path: Path) -> None:
    pin = deepcopy(PIN)
    pin["reproduction"] = {"command": ["geng", "-q", "4"]}

    report = RUN_SPIKE(
        geng=tmp_path / "absent-geng",
        labelg=tmp_path / "absent-labelg",
        source_archive=tmp_path / "absent-nauty.tar.gz",
        pin_path=_pin_file(tmp_path, pin),
    )

    assert report["status"] == "ERROR"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "INVALID_SPIKE_PIN"


def test_spike_executes_the_command_profile_recorded_by_the_pin(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)
    pin = deepcopy(PIN)
    pin["reproduction"]["command"] = ["geng", "--frozen-profile", "4"]
    commands: list[list[str]] = []
    outcomes = iter(_successful_outcomes())

    def runner(command: Sequence[str], **_kwargs: object) -> ToolCommandResult:
        commands.append(list(command))
        return next(outcomes)

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=runner,
        pin_path=_pin_file(tmp_path, pin),
    )

    assert report["status"] == "COMPLETED"
    assert commands[2] == [str(geng.resolve()), "--frozen-profile", "4"]


def test_changed_executable_digest_rejects_the_success_report(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)
    outcomes = iter(_successful_outcomes())
    call_count = 0

    def runner(_command: Sequence[str], **_kwargs: object) -> ToolCommandResult:
        nonlocal call_count
        result = next(outcomes)
        call_count += 1
        if call_count == 4:
            labelg.write_bytes(b"replaced after execution")
        return result

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=runner,
        pin_path=_pin_file(tmp_path),
    )

    assert report["status"] == "REJECTED"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "EXECUTABLE_CHANGED"


def test_malformed_generation_output_is_not_a_partial_result(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)
    outcomes = [
        _result(stdout=GENG_HELP),
        _result(stdout=LABELG_HELP),
        _result(stdout=b"not-graph6\n"),
    ]

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=_runner(outcomes),
        pin_path=_pin_file(tmp_path),
    )

    assert report["status"] == "ERROR"
    assert report["conclusion"] == "NO_CONCLUSION"
    assert report["diagnostic"]["code"] == "PROVIDER_OUTPUT_MALFORMED"


def test_timeout_and_crash_remain_distinct_non_conclusions(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)
    pin_path = _pin_file(tmp_path)

    timed_out = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=_runner([_result(timed_out=True)]),
        pin_path=pin_path,
    )
    crashed = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=_runner([_result(returncode=9)]),
        pin_path=pin_path,
    )

    assert (timed_out["status"], timed_out["diagnostic"]["code"]) == (
        "TIMEOUT",
        "PROVIDER_TIMEOUT",
    )
    assert (crashed["status"], crashed["diagnostic"]["code"]) == (
        "ERROR",
        "PROVIDER_CRASH",
    )
    assert timed_out["conclusion"] == crashed["conclusion"] == "NO_CONCLUSION"
