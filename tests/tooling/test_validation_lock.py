from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import tools.with_validation_lock as validation_lock
from tools.with_validation_lock import main

ROOT = Path(__file__).parents[2]
LOCK = ROOT / "tools" / "with_validation_lock.py"


def _worktree(tmp_path: Path) -> Path:
    (tmp_path / "Makefile").write_text("# test worktree\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_status_reports_free_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))

    assert main(["status"]) == 0


def test_second_broad_run_is_rejected_while_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    holder = subprocess.Popen(
        [
            sys.executable,
            str(LOCK),
            "run",
            "--target",
            "test-full",
            "--",
            sys.executable,
            "-c",
            "open('held', 'w').write('1'); import time; time.sleep(2)",
        ],
        cwd=tmp_path,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (tmp_path / "held").exists():
                break
            time.sleep(0.05)
        else:
            raise AssertionError("holder did not take the validation lock")
        with pytest.raises(SystemExit, match="already running broad validation"):
            main(["run", "--target", "test-exhaustive", "--", "true"])
    finally:
        holder.wait(timeout=15)
    assert holder.returncode == 0


def test_timed_out_command_releases_the_validation_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    monkeypatch.setattr(validation_lock, "_VALIDATION_TIMEOUT_SECONDS", 0.2)

    result = main(
        [
            "run",
            "--target",
            "test-timeout",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ]
    )

    assert result != 0
    assert main(["status"]) == 0


def test_run_forwards_pytest_arguments_to_the_locked_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    monkeypatch.setenv("PYTEST_ARGS", "--junitxml=pytest.xml")

    result = main(
        [
            "run",
            "--target",
            "test-scale",
            "--",
            sys.executable,
            "-c",
            "from pathlib import Path; import os; Path('args').write_text(os.environ['PYTEST_ARGS'])",
        ]
    )

    assert result == 0
    assert (tmp_path / "args").read_text(encoding="utf-8") == "--junitxml=pytest.xml"
