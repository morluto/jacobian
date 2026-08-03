"""Tests for observation task-selection normalization and explicit-task path validation."""

from __future__ import annotations

from pathlib import Path

from benchmarks.tooling.observation_selection import (
    normalize_selection,
    validate_explicit_task_path,
)
from benchmarks.validation.observation_results_support import _DIGEST

# ---------------------------------------------------------------------------
# Regression for #338: strict dataset/task selection normalization
# ---------------------------------------------------------------------------


def _selection_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    known = {"case": _DIGEST, "other": "sha256:" + "b" * 64}
    task_dirs: dict[str, Path] = {}
    for name in known:
        task_dir = tmp_path / name
        task_dir.mkdir()
        (task_dir / "task.toml").write_text("version = '1'\n", encoding="utf-8")
        task_dirs[name] = task_dir
    return known, task_dirs, tmp_path


def test_selection_rejects_mixed_datasets_and_tasks(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {
        "datasets": [{"path": "x", "task_names": ["case"]}],
        "tasks": [{"path": "case"}],
    }

    selected, mode, _eval, failures = normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path, root=tmp_path
    )

    assert mode == "mixed"
    assert selected == []
    assert any("not both" in f for f in failures)


def test_selection_rejects_implicit_fallback(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"n_attempts": 1}

    selected, mode, _eval, failures = normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path, root=tmp_path
    )

    assert mode == "implicit-fallback"
    assert selected == []
    assert any("implicit fallback" in f for f in failures)


def test_selection_rejects_unknown_task_name(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"datasets": [{"path": "x", "task_names": ["nonexistent"]}]}

    _selected, _mode, _eval, failures = normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path, root=tmp_path
    )

    assert any("unknown task name" in f for f in failures)


def test_selection_rejects_empty_datasets_and_empty_task_names(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)

    _s, _m, _e, empty_failures = normalize_selection(
        {"datasets": []},
        known=known,
        task_dirs=task_dirs,
        dataset_path=dataset_path,
        root=tmp_path,
    )
    assert any("non-empty array" in f for f in empty_failures)

    _s, _m, _e, names_failures = normalize_selection(
        {"datasets": [{"path": "x", "task_names": []}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=dataset_path,
        root=tmp_path,
    )
    assert any("non-empty array" in f for f in names_failures)


def test_selection_optional_task_names_selects_all_known(tmp_path: Path) -> None:
    known, task_dirs, dataset_path = _selection_fixture(tmp_path)
    job = {"datasets": [{"path": "x"}]}

    selected, mode, _eval, failures = normalize_selection(
        job, known=known, task_dirs=task_dirs, dataset_path=dataset_path, root=tmp_path
    )

    assert failures == []
    assert mode == "dataset-task-names"
    assert selected == ["case", "other"]


def test_selection_explicit_tasks_reject_outside_dataset(tmp_path: Path) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "task.toml").write_text("version = '1'\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    _s, _m, _e, failures = normalize_selection(
        {"tasks": [{"path": "outside"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=dataset_path,
        root=tmp_path,
    )

    assert any("outside the dataset" in f for f in failures)


def test_selection_explicit_tasks_reject_unknown_path(tmp_path: Path) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    bogus = tmp_path / "bogus"
    bogus.mkdir()
    (bogus / "task.toml").write_text("version = '1'\n", encoding="utf-8")

    _s, _m, _e, failures = normalize_selection(
        {"tasks": [{"path": "bogus"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )

    assert any("not a known task" in f for f in failures)


def test_selection_explicit_tasks_reject_absolute_and_traversal(tmp_path: Path) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)

    _s, _m, _e, abs_failures = normalize_selection(
        {"tasks": [{"path": "/etc/passwd"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )
    assert any("must be relative" in f for f in abs_failures)

    _s, _m, _e, trav_failures = normalize_selection(
        {"tasks": [{"path": "../case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )
    assert any("traverse" in f for f in trav_failures)


def test_selection_explicit_tasks_reject_missing_manifest(tmp_path: Path) -> None:
    known, _task_dirs, _ = _selection_fixture(tmp_path)
    no_manifest = tmp_path / "case"
    # Reuse the "case" dir from the fixture but remove its manifest.
    (no_manifest / "task.toml").unlink()
    task_dirs = {"case": no_manifest}

    _s, _m, _e, failures = normalize_selection(
        {"tasks": [{"path": "case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )

    assert any("missing its manifest" in f for f in failures)


def test_selection_explicit_tasks_reject_escaping_symlink(tmp_path: Path) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "case")

    _s, _m, _e, failures = normalize_selection(
        {"tasks": [{"path": "link"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )

    assert any("symlink" in f for f in failures)


def test_selection_explicit_tasks_accept_valid_path(tmp_path: Path) -> None:
    known, task_dirs, _ = _selection_fixture(tmp_path)

    selected, mode, _eval, failures = normalize_selection(
        {"tasks": [{"path": "case"}]},
        known=known,
        task_dirs=task_dirs,
        dataset_path=tmp_path,
        root=tmp_path,
    )

    assert failures == []
    assert mode == "explicit-tasks"
    assert selected == ["case"]


# ---------------------------------------------------------------------------
# Explicit task path: absolute and traversal (path hygiene, #343)
# ---------------------------------------------------------------------------


def test_explicit_task_path_rejects_absolute(tmp_path: Path) -> None:
    _short, failures = validate_explicit_task_path(
        "/etc/passwd", dataset_path=None, task_dirs={}, root=tmp_path
    )

    assert _short is None
    assert any("must be relative" in f for f in failures)


def test_explicit_task_path_rejects_traversal(tmp_path: Path) -> None:
    _short, failures = validate_explicit_task_path(
        "../escape", dataset_path=None, task_dirs={}, root=tmp_path
    )

    assert _short is None
    assert any("traverse" in f for f in failures)
