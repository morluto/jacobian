from __future__ import annotations

from pathlib import Path

from tools.test_architecture.lanes import owner_for, owners

ROOT = Path(__file__).resolve().parents[3]


def test_every_test_module_has_one_semantic_lane() -> None:
    test_files = sorted((ROOT / "tests").rglob("test_*.py"))
    for path in test_files:
        relative = path.relative_to(ROOT).as_posix()
        claimed = owners(relative)
        assert len(claimed) == 1, f"{relative} has lanes {claimed}"
        assert owner_for(relative) == claimed[0]


def test_obsolete_catch_all_directories_are_absent() -> None:
    for name in (
        "integration",
        "contract",
        "checkers",
        "reference",
        "end_to_end",
        "helpers",
        "fixtures",
    ):
        assert not (ROOT / "tests" / name).exists()
