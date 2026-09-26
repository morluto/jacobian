"""Behavioral coverage for the JUnit and worker timing report."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from tools import test_timing_report as reporter
from tools.command_runner import ToolCommandStatus, run_operator_command

ROOT = Path(__file__).parents[2]


def test_report_orders_slowest_cases_and_exposes_worker_skew(tmp_path: Path) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        """<?xml version=\"1.0\"?>
<testsuites><testsuite><testcase classname=\"math.graphs\" name=\"fast\" time=\"0.2\" />
<testcase classname=\"math.graphs\" name=\"slow\" time=\"1.5\" /></testsuite></testsuites>
""",
        encoding="utf-8",
    )
    timing = tmp_path / "timing.json"
    timing.write_text(
        """{"version": 1, "wall_seconds": 3.0, "workers": [
{"id": "gw0", "call_seconds": 1.0, "call_count": 1},
{"id": "gw1", "call_seconds": 2.0, "call_count": 1}]}""",
        encoding="utf-8",
    )

    summary = reporter.build_summary(junit=junit, timing=timing, limit=10)
    output = StringIO()
    reporter.write_summary(summary, stream=output)

    assert summary.test_count == 2
    assert summary.slowest[0].node_id == "math.graphs::slow"
    assert "call-time skew 2.00x" in output.getvalue()
    assert "Non-call wall remainder: 1.000s" in output.getvalue()


def test_junit_only_report_marks_worker_distribution_unavailable(
    tmp_path: Path,
) -> None:
    junit = tmp_path / "pytest.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="only" time="0.1" />'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )

    summary = reporter.build_summary(junit=junit, timing=None, limit=10)
    output = StringIO()
    reporter.write_summary(summary, stream=output)

    assert "Worker timing: unavailable" in output.getvalue()


def test_phase_timing_round_trip_exposes_collection_and_fixture_costs(
    tmp_path: Path,
) -> None:
    timing = tmp_path / "timing.json"
    junit = tmp_path / "pytest.xml"
    test_file = tmp_path / "test_timing_phases.py"
    test_file.write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def value():\n"
        "    yield 1\n"
        "def test_fixture(value):\n"
        "    assert value == 1\n",
        encoding="utf-8",
    )

    completed = run_operator_command(
        "uv",
        (
            "run",
            "--locked",
            "python",
            "-m",
            "pytest",
            "-p",
            "tools.pytest_timing",
            f"--jacobian-timing-json={timing}",
            f"--junitxml={junit}",
            "-q",
            str(test_file),
        ),
        cwd=ROOT,
        timeout_seconds=60,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
    )
    assert completed.status is ToolCommandStatus.EXITED
    assert completed.exit_code == 0, completed.stderr.decode(errors="replace")

    summary = reporter.build_summary(junit=junit, timing=timing, limit=10)
    output = StringIO()
    reporter.write_summary(summary, stream=output)
    assert len(summary.workers) == 1
    worker = summary.workers[0]

    assert summary.test_count == 1
    assert worker.collection_seconds is not None
    assert worker.setup_seconds is not None
    assert worker.teardown_seconds is not None
    assert "collection (including test imports):" in output.getvalue()
    assert "setup:" in output.getvalue()
    assert "teardown:" in output.getvalue()
    assert "not a measurement of startup alone" in output.getvalue()
