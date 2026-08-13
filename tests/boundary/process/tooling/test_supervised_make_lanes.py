from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _make_dry_run(*args: str) -> str:
    completed = subprocess.run(
        [
            "make",
            "--dry-run",
            "--no-print-directory",
            "TESTS=",
            "PYTEST_ARGS=",
            *args,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_harbor_execution_check_stays_out_of_the_full_verifier_corpus() -> None:
    output = _make_dry_run("harbor-execution-check")

    assert "tools/check_harbor_dataset.py --check" in output
    assert "tools/check_benchmark_contracts.py" in output
    assert "tests/unit/tooling/test_harbor*.py" in output
    assert "benchmarks/validation" not in output
    assert "harbor-oracle" not in output


def test_process_lane_dry_run_uses_lifecycle_and_signal_timeout() -> None:
    output = _make_dry_run("test-process")

    assert "tools/pytest_lifecycle.py" in output
    assert "--name process" in output
    assert "-n 2" in output
    assert "--timeout=120" in output
    assert "--timeout-method=signal" in output
    assert "tests/boundary/process" in output


def test_mcp_lane_dry_run_uses_lifecycle() -> None:
    output = _make_dry_run("test-mcp")

    assert "tools/pytest_lifecycle.py" in output
    assert "--name mcp" in output
    assert "tests/boundary/mcp" in output


def test_lean_lane_dry_run_is_serial_and_supervised() -> None:
    output = _make_dry_run("test-lean")

    assert "tools/pytest_lifecycle.py" in output
    assert "--name lean" in output
    assert "--timeout=300" in output
    assert "--timeout-method=signal" in output
    assert "tests/boundary/providers/lean" in output
    assert " -n " not in output


def test_supervised_lane_forwards_pytest_arguments() -> None:
    relative = Path(__file__).relative_to(ROOT).as_posix()
    output = _make_dry_run(
        "test-process",
        f"TESTS={relative}",
        "PYTEST_ARGS=-k target_test --junitxml=pytest.xml",
    )

    assert relative in output
    assert "-k target_test" in output
    assert "--junitxml=pytest.xml" in output


def test_ordinary_unit_lane_invokes_pytest_directly() -> None:
    output = _make_dry_run("test-unit")

    assert "pytest_lifecycle.py" not in output
    assert "pytest --timeout=10" in output
    assert "tests/unit" in output


def test_explicit_lean_module_still_collects() -> None:
    representative = "tests/boundary/providers/lean/test_lean_residual_contracts.py"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "pytest",
            "--collect-only",
            "-q",
            representative,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr

    assert representative in output
