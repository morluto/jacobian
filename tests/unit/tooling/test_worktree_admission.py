from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tools.worktree_admission import TOKEN_ENV, main

ROOT = Path(__file__).parents[3]
ADMISSION = ROOT / "tools" / "worktree_admission.py"


def _worktree(tmp_path: Path) -> Path:
    (tmp_path / "Makefile").write_text("# test worktree\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_status_reports_free_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    assert main(["status"]) == 0


def test_nested_make_reenters_with_unforgeable_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    assert (
        main(
            [
                "run",
                "--target",
                "test-all-ci",
                "--",
                sys.executable,
                str(ADMISSION),
                "run",
                "--target",
                "nested",
                "--",
                "true",
            ]
        )
        == 0
    )


def test_second_exhaustive_run_is_rejected_while_lease_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(_worktree(tmp_path))
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    holder = subprocess.Popen(
        [
            sys.executable,
            str(ADMISSION),
            "run",
            "--target",
            "test-all-ci",
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
            raise AssertionError("holder did not take the validation lease")
        with pytest.raises(SystemExit, match="already running exhaustive validation"):
            main(["run", "--target", "test-exhaustive", "--", "true"])
    finally:
        holder.wait(timeout=15)
    assert holder.returncode == 0
    assert TOKEN_ENV not in os.environ
