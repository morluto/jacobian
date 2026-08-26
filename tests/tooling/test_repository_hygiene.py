from pathlib import Path

import pytest
from tools import check_repository_hygiene


def test_rejects_artifacts_and_conflict_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "pr-audit").mkdir()
    (tmp_path / "pr-audit/report.json").write_text("{}")
    (tmp_path / "jacobian_scale_audit_report.md").write_text("stale\n")
    (tmp_path / ".agents-tmp-unresolved.sh").write_text("#!/bin/sh\n")
    (tmp_path / "guide.md").write_text("<<<<<<< HEAD\n=======\n>>>>>>> main\n")
    monkeypatch.setattr(
        check_repository_hygiene,
        "tracked_paths",
        lambda root: (
            Path("pr-audit/report.json"),
            Path("jacobian_scale_audit_report.md"),
            Path(".agents-tmp-unresolved.sh"),
            Path("guide.md"),
        ),
    )

    assert check_repository_hygiene.check(tmp_path) == (
        "pr-audit/report.json: forbidden local artifact",
        "jacobian_scale_audit_report.md: forbidden local artifact",
        ".agents-tmp-unresolved.sh: forbidden local artifact",
        "guide.md:1: unresolved conflict marker",
        "guide.md:2: unresolved conflict marker",
        "guide.md:3: unresolved conflict marker",
    )


def test_allows_literal_markers_only_in_fixture_and_vendored_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = tmp_path / "tests/fixtures/merge-conflict.txt"
    vendor = tmp_path / "vendor/merge-conflict.txt"
    document = tmp_path / "docs/guide.md"
    for path in (fixture, vendor, document):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("<<<<<<< HEAD\n=======\n>>>>>>> main\n", encoding="utf-8")
    monkeypatch.setattr(
        check_repository_hygiene,
        "tracked_paths",
        lambda root: (
            Path("tests/fixtures/merge-conflict.txt"),
            Path("vendor/merge-conflict.txt"),
            Path("docs/guide.md"),
        ),
    )

    assert check_repository_hygiene.check(tmp_path) == (
        "docs/guide.md:1: unresolved conflict marker",
        "docs/guide.md:2: unresolved conflict marker",
        "docs/guide.md:3: unresolved conflict marker",
    )
