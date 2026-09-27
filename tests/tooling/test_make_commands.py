"""Owner-local CI policy tests split from test_ci_execution_policy.py."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

import pytest
from tools.command_runner import ToolCommandStatus, run_operator_command

ROOT = Path(__file__).parents[2]


def _make_dry_run(*arguments: str) -> str:
    result = run_operator_command(
        "make",
        (
            "--no-print-directory",
            "--dry-run",
            f"UV_RUN={shutil.which('uv')} run --locked",
            "VALIDATION_LOCK=echo tools/with_validation_lock.py",
            *arguments,
        ),
        cwd=ROOT,
        timeout_seconds=15,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
    )
    assert result.status is ToolCommandStatus.EXITED
    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    return result.stdout.decode()


def test_exhaustive_local_reproduction_includes_exhaustive_marker_lane() -> None:
    output = _make_dry_run("_test-full")

    for target in (
        "test-math",
        "test-property",
        "_test-exhaustive",
        "_test-scale",
        "test-catalog-examples",
    ):
        assert target in output
    assert output.index("test-math") < output.index("_test-exhaustive")


def test_full_reproduction_uses_validation_lock() -> None:
    output = _make_dry_run("test-full")

    assert "tools/with_validation_lock.py run --target test-full" in output


def test_full_catalog_examples_split_singular_backend_ownership() -> None:
    output = _make_dry_run("test-catalog-examples")

    assert "tests/integration/catalog/test_builtin_examples.py" in output
    assert "not singular_catalog_example" in output


def test_focused_math_lane_skips_validation_lock() -> None:
    output = _make_dry_run(
        "test-math",
        "MATH_WORKERS=0",
        "TESTS=tests/math/logic/test_tools.py",
    )

    assert "tools/with_validation_lock.py" not in output
    assert "pytest -n 0" in output
    assert "tests/math/logic/test_tools.py" in output


def test_exhaustive_and_harbor_broad_commands_use_validation_lock() -> None:
    for target in (
        "test-exhaustive",
        "harbor-check-all",
        "harbor-host-validation",
        "harbor-oracle-all",
    ):
        output = _make_dry_run(target)
        assert f"tools/with_validation_lock.py run --target {target}" in output


def test_scoped_handoff_requires_explicit_static_paths() -> None:
    output = _make_dry_run(
        "handoff-scoped",
        "LANE=tooling",
        "TESTS=tests/tooling/test_make_commands.py",
        "PATHS=tests/tooling/test_make_commands.py",
    )

    assert "ruff check tests/tooling/test_make_commands.py" in output
    assert "ruff format --check tests/tooling/test_make_commands.py" in output
    assert "mypy tests/tooling/test_make_commands.py" in output
    assert 'make test-tooling TESTS="tests/tooling/test_make_commands.py"' in output


def test_affected_validation_is_a_public_planner_backed_default() -> None:
    affected = _make_dry_run("affected", "AFFECTED_BASE=origin/main")
    plan = _make_dry_run("affected-plan", "AFFECTED_BASE=origin/main")

    assert 'python tools/affected_validation.py --base "origin/main"' in affected
    assert 'python tools/affected_validation.py --base "origin/main" --dry-run' in plan


def test_timing_report_is_a_public_read_only_diagnostic() -> None:
    output = _make_dry_run("test-timings", "JUNIT=pytest.xml", "TIMING=timing.json")

    assert 'tools/test_timing_report.py --junit "pytest.xml"' in output
    assert '--timing "timing.json"' in output


def test_broad_commands_share_the_nonblocking_validation_lease() -> None:
    for target in ("check", "check-all"):
        output = _make_dry_run(target)
        assert f"tools/with_validation_lock.py run --target {target}" in output


@pytest.mark.parametrize(
    ("target", "workers", "timeout", "root"),
    [
        ("test-catalog", 2, 30, "tests/catalog"),
        ("test-dispatch", 2, 120, "tests/dispatch"),
        ("test-cli", 2, 30, "tests/cli"),
        ("test-tooling", 2, 30, "tests/tooling"),
        ("test-integration", 1, 120, "tests/integration"),
    ],
)
def test_owner_lanes_keep_worker_bounds_and_collection_roots(
    target: str, workers: int, timeout: int, root: str
) -> None:
    output = _make_dry_run(target)

    assert f"pytest -n {workers} --dist worksteal --timeout={timeout}" in output
    assert root in output
    assert '-m "not property and not exhaustive and not scale"' in output


@pytest.mark.parametrize("workers", [0, 1, 2, 4])
@pytest.mark.parametrize("target", ["test-math", "test-focused"])
def test_math_worker_control_preserves_focused_selection(
    workers: int, target: str
) -> None:
    selector = "tests/math/lattices/test_rectangular_hnf.py"
    result = run_operator_command(
        "make",
        (
            "--no-print-directory",
            "-s",
            "-n",
            target,
            "LANE=math",
            f"MATH_WORKERS={workers}",
            f"TESTS={selector}",
            "PYTEST_ARGS=",
        ),
        cwd=ROOT,
        timeout_seconds=10,
        stdout_limit_bytes=64 * 1024,
        stderr_limit_bytes=64 * 1024,
    )
    assert result.status is ToolCommandStatus.EXITED
    assert result.exit_code == 0, result.stderr
    commands = result.stdout.decode().replace("\\\n", " ").splitlines()
    pytest_commands = [
        shlex.split(line) for line in commands if line.startswith("uv run")
    ]
    assert pytest_commands == [
        [
            "uv",
            "run",
            "--locked",
            "pytest",
            "-n",
            str(workers),
            "--dist",
            "worksteal",
            "--timeout=120",
            "-m",
            "not property and not exhaustive and not scale",
            selector,
            "--durations=10",
        ]
    ]


def test_scale_lane_is_marker_selected_and_validation_locked() -> None:
    output = _make_dry_run("_test-scale")

    assert "-m scale" in output


def test_scale_public_command_uses_validation_lock() -> None:
    output = _make_dry_run("test-scale")

    assert "tools/with_validation_lock.py run --target test-scale" in output


def test_harbor_plan_forwards_explicit_changed_paths(tmp_path: Path) -> None:
    import sys

    from tools.command_runner import ToolCommandStatus, run_operator_command

    shim = tmp_path / "harbor-plan-shim.py"
    shim.write_text(
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "print('ARGS:', ' '.join(args))\n"
        "paths = pathlib.Path(args[args.index('--paths-file') + 1])\n"
        "print('CHANGED_PATHS:', paths.read_text(encoding='utf-8').strip())\n"
        "pathlib.Path(args[args.index('--output') + 1]).write_text('{}', encoding='utf-8')\n",
        encoding="utf-8",
    )
    result = run_operator_command(
        "make",
        (
            "--no-print-directory",
            "UV_RUN=" + str(shutil.which("uv")) + " run --locked",
            "harbor-plan",
            "PATHS=Makefile",
            f"HARBOR_PYTHON={sys.executable} {shim}",
        ),
        cwd=ROOT,
        timeout_seconds=15,
        stdout_limit_bytes=4 * 1024 * 1024,
        stderr_limit_bytes=1024 * 1024,
    )
    assert result.status is ToolCommandStatus.EXITED
    assert result.exit_code == 0, result.stderr.decode(errors="replace")
    output = result.stdout.decode()

    assert "CHANGED_PATHS: Makefile" in output
    assert ".github/scripts/plan-benchmarks" in output
    assert "--paths-file" in output
