from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from tools.restack_feature_branch import restack


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_restack_reports_unique_and_duplicate_subjects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "restack@example.test")
    _git(repo, "config", "user.name", "Restack")
    (repo / "README").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "shared subject")
    _git(repo, "checkout", "-b", "feature")
    (repo / "README").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-m", "shared subject")
    (repo / "UNIQUE").write_text("leaf\n", encoding="utf-8")
    _git(repo, "add", "UNIQUE")
    _git(repo, "commit", "-m", "unique leaf work")

    assert restack(cwd=repo, parent="main", feature="feature") == 0
    output = capsys.readouterr().out
    assert "unique commits: 2" in output
    assert "shared subject" in output
    assert "unique leaf work" in output
    assert "does not rewrite published branches" in output
