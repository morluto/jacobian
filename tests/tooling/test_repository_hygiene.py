from pathlib import Path

from tools import check_repository_hygiene


def test_rejects_artifacts_and_conflict_markers(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "pr-audit").mkdir()
    (tmp_path / "pr-audit/report.json").write_text("{}")
    (tmp_path / "guide.md").write_text("<<<<<<< HEAD\n")
    monkeypatch.setattr(
        check_repository_hygiene,
        "tracked_paths",
        lambda root: (Path("pr-audit/report.json"), Path("guide.md")),
    )

    assert check_repository_hygiene.check(tmp_path) == (
        "pr-audit/report.json: forbidden local artifact",
        "guide.md:1: unresolved conflict marker",
    )
