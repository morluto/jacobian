from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from benchmarks.tooling.providers import cddlib, cgal, gudhi, nauty, regina
from benchmarks.tooling.spike_utils import default_runner
from tests.process.providers._spike_support import _request, _result, _runner


def _expected(tmp_path: Path):
    executable = tmp_path / "provider"
    executable.touch()
    return _request(
        executable,
        ("--probe", "value"),
        environment={"LANG": "C", "TZ": "UTC"},
        cwd=tmp_path,
        timeout_seconds=5,
        stdin_bytes=b"input",
        stdout_limit_bytes=1024,
        stderr_limit_bytes=512,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request, tmp_path: replace(request, arguments=("--wrong",)),
        lambda request, tmp_path: replace(request, environment={"LANG": "en_US"}),
        lambda request, tmp_path: replace(request, cwd=str(tmp_path.parent)),
        lambda request, tmp_path: replace(request, timeout_seconds=6),
        lambda request, tmp_path: replace(request, stdout_limit_bytes=2048),
    ],
)
def test_strict_runner_rejects_mutated_request_fields(tmp_path: Path, mutation) -> None:
    expected = _expected(tmp_path)
    runner = _runner([_result()])
    runner.expect((expected,))

    with pytest.raises(AssertionError, match="provider command 1 mismatch"):
        runner(mutation(expected, tmp_path))


def test_strict_runner_rejects_reordered_and_extra_commands(tmp_path: Path) -> None:
    first = _expected(tmp_path)
    second = replace(first, arguments=("--second",))
    runner = _runner([_result(), _result()])
    runner.expect((first, second))

    with pytest.raises(AssertionError, match="provider command 1 mismatch"):
        runner(second)

    runner = _runner([_result()])
    runner.expect((first,))
    assert runner(first) == _result()
    with pytest.raises(AssertionError, match="unexpected provider command"):
        runner(second)


def test_strict_runner_rejects_unconsumed_expectations(tmp_path: Path) -> None:
    runner = _runner([_result(), _result()])
    runner.expect((_expected(tmp_path), replace(_expected(tmp_path), arguments=("x",))))
    runner(_expected(tmp_path))

    with pytest.raises(AssertionError, match="1 expected provider command"):
        runner.assert_finished()


@pytest.mark.parametrize(
    ("module", "error_type", "extra_arguments"),
    [
        (cddlib, cddlib.CddlibSpikeError, {}),
        (cgal, cgal.CgalSpikeError, {}),
        (gudhi, gudhi.GudhiSpikeError, {}),
        (nauty, nauty.NautySpikeError, {"input_bytes": b"", "stdout_limit": 4096}),
        (regina, regina.ReginaSpikeError, {}),
    ],
)
def test_provider_runner_translates_an_absent_working_directory(
    tmp_path: Path,
    module: Any,
    error_type: type[Exception],
    extra_arguments: dict[str, Any],
) -> None:
    absent_cwd = tmp_path / "absent"

    with pytest.raises(error_type) as caught:
        module._run_checked(
            default_runner,
            (sys.executable, "-c", "pass"),
            cwd=absent_cwd,
            timeout_seconds=1,
            **extra_arguments,
        )

    assert cast(Any, caught.value).code == "PROVIDER_LAUNCH_ERROR"
