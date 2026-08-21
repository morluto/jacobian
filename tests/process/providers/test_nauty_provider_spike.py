from __future__ import annotations

import hashlib
import json
import math
import tarfile
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from benchmarks.tooling.providers.nauty import run_spike
from tests.process.providers._spike_support import (
    _ExpectedRunner,
    _request,
    _result,
    _run_expected,
    _runner,
)
from tools.command_runner import ToolCommandRequest, ToolCommandResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _base_pin() -> dict[str, Any]:
    path = PROJECT_ROOT / "tests/fixtures/providers/nauty/nauty_provider_pin.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


def _geng_output() -> bytes:
    return ("\n".join(_base_pin()["reproduction"]["expected_graph6"]) + "\n").encode()


def _labelg_output() -> bytes:
    return (
        "\n".join(_base_pin()["canonicalization"]["expected_output_graph6"]) + "\n"
    ).encode()


def RUN_SPIKE(**kwargs: Any) -> dict[str, Any]:  # noqa: N802
    runner = kwargs.get("runner")
    if not isinstance(runner, _ExpectedRunner):
        return run_spike(cwd=PROJECT_ROOT, **kwargs)
    pin = json.loads(Path(kwargs["pin_path"]).read_text(encoding="utf-8"))
    geng = Path(kwargs["geng"])
    labelg = Path(kwargs["labelg"])
    timeout = float(kwargs.get("timeout_seconds", 5))
    canonical_input = (
        "\n".join(pin["canonicalization"]["input_graph6"]) + "\n"
    ).encode("ascii")
    requests = (
        _request(
            geng,
            ("-help",),
            environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            cwd=PROJECT_ROOT,
            timeout_seconds=timeout,
            stdout_limit_bytes=16_384,
            stderr_limit_bytes=16_384,
        ),
        _request(
            labelg,
            ("-help",),
            environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            cwd=PROJECT_ROOT,
            timeout_seconds=timeout,
            stdout_limit_bytes=16_384,
            stderr_limit_bytes=16_384,
        ),
        _request(
            geng,
            tuple(pin["reproduction"]["command"][1:]),
            environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            cwd=PROJECT_ROOT,
            timeout_seconds=timeout,
            stdout_limit_bytes=4096,
            stderr_limit_bytes=16_384,
        ),
        _request(
            labelg,
            tuple(pin["canonicalization"]["command"][1:]),
            environment={"LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
            cwd=PROJECT_ROOT,
            timeout_seconds=timeout,
            stdin_bytes=canonical_input,
            stdout_limit_bytes=4096,
            stderr_limit_bytes=16_384,
        ),
    )
    return _run_expected(
        lambda: run_spike(cwd=PROJECT_ROOT, **kwargs), runner, requests
    )


def _source_archive(tmp_path: Path) -> Path:
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
    return archive_path


def _files(tmp_path: Path) -> tuple[Path, Path]:
    geng = tmp_path / "geng"
    labelg = tmp_path / "labelg"
    geng.touch()
    labelg.touch()
    return geng, labelg


def _pin_file(tmp_path: Path, pin: dict[str, Any] | None = None) -> Path:
    path = tmp_path / "pin.json"
    payload = _base_pin() if pin is None else pin
    archive = tmp_path / "nauty2_9_3.tar.gz"
    if pin is None and archive.exists():
        payload["archive_sha256"] = (
            "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _successful_outcomes() -> list[ToolCommandResult]:
    return [
        _result(stdout=GENG_HELP),
        _result(stdout=LABELG_HELP),
        _result(stdout=_geng_output()),
        _result(stdout=_labelg_output()),
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
    archive = _source_archive(tmp_path)
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
    pin = deepcopy(_base_pin())
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
    pin = deepcopy(_base_pin())
    pin["archive_sha256"] = "sha256:" + hashlib.sha256(archive.read_bytes()).hexdigest()
    pin["reproduction"]["command"] = ["geng", "--frozen-profile", "4"]
    runner = _runner(_successful_outcomes())

    report = RUN_SPIKE(
        geng=geng,
        labelg=labelg,
        source_archive=archive,
        runner=runner,
        pin_path=_pin_file(tmp_path, pin),
    )

    assert report["status"] == "COMPLETED"
    assert [runner.requests[2].executable, *runner.requests[2].arguments] == [
        str(geng.resolve()),
        "--frozen-profile",
        "4",
    ]


def test_changed_executable_digest_rejects_the_success_report(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    geng, labelg = _files(tmp_path)
    runner = _runner(_successful_outcomes())

    def mutate_after_last_call(call_count: int, _request: ToolCommandRequest) -> None:
        if call_count == 4:
            labelg.write_bytes(b"replaced after execution")

    runner.on_call = mutate_after_last_call

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
