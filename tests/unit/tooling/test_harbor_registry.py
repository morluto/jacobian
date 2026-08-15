"""Harbor registry and suite-membership policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.tooling.errors import HarborSuiteError
from benchmarks.tooling.harbor_suite import get_suite, load_registry
from tests.unit.tooling.harbor_suite_support import (
    _make_canonical_task,
    _make_dataset_entry,
    _make_minimal_task,
    _write_registry,
    _write_suite_toml,
    patch_harbor_root,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def patched_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return patch_harbor_root(monkeypatch, tmp_path)


def test_load_registry_returns_unique_well_formed_datasets() -> None:
    suites = load_registry()
    ids = {s.id for s in suites}
    assert ids
    assert len(ids) == len(suites)
    assert all(suite.dataset_name.startswith("jacobian/") for suite in suites)
    assert all(
        suite.id == suite.dataset_name.removeprefix("jacobian/") for suite in suites
    )
    assert all(suite.path.is_dir() for suite in suites)
    assert all(suite.tasks_dir.is_dir() for suite in suites)


def test_load_registry_rejects_wrong_schema_version(
    tmp_path: Path, patched_root: Path
) -> None:
    reg = tmp_path / "registry.toml"
    reg.write_text('schema_version = "99"\ndatasets = []')
    with pytest.raises(HarborSuiteError, match="schema_version"):
        load_registry(reg)


def test_load_registry_fails_closed_on_missing_suite_toml(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    ds_path.mkdir()
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    with pytest.raises(HarborSuiteError, match=r"suite\.toml"):
        load_registry(reg)


def test_suite_loads_tasks_when_suite_toml_exists(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    _make_canonical_task(tmp_path)
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "assurance_ceiling": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suites = load_registry(reg)
    assert len(suites[0].tasks) == 1


# ---------------------------------------------------------------------------
# Suite parsing
# ---------------------------------------------------------------------------


def test_suite_parses_tasks_without_assurance_ceiling(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    _make_canonical_task(tmp_path)
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suite = load_registry(reg)[0]
    assert suite.tasks[0].name == "jacobian/test-v1-a"
    assert suite.tasks[0].required_provider == "core"


def test_suite_allows_empty_tasks(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    ds_path.mkdir()
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(ds_path / "suite.toml")
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suite = load_registry(reg)[0]
    assert suite.tasks == ()


def test_suite_rejects_task_path_outside_tasks_root(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/missing-task",
                "assurance_ceiling": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])
    with pytest.raises(HarborSuiteError, match="Harbor task is missing"):
        load_registry(reg)


def test_suite_rejects_symlinked_task_path(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    _make_canonical_task(tmp_path)
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[{"id": "jacobian/test-v1-a", "assurance_ceiling": "COMPUTED"}],
    )
    member = ds_path / "members" / "test-v1-a.toml"
    target = tmp_path / "member-target.toml"
    member.replace(target)
    member.symlink_to(target)
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])
    with pytest.raises(HarborSuiteError, match="symlink"):
        load_registry(reg)


def test_suite_rejects_noncanonical_task_id(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    ds_path.mkdir()
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/nested/task",
                "assurance_ceiling": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    with pytest.raises(HarborSuiteError, match="invalid canonical task id"):
        load_registry(reg)


def test_registry_rejects_nested_canonical_task_bundle(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(ds_path / "suite.toml")
    _make_minimal_task(
        ds_path / "algebra" / "nested-task",
        task_id="jacobian/nested-task",
    )
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])

    with pytest.raises(HarborSuiteError, match=r"direct children"):
        load_registry(reg)


def test_registry_rejects_incomplete_canonical_task_directory(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(ds_path / "suite.toml")
    (ds_path / "incomplete-task").mkdir(parents=True)
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])

    with pytest.raises(HarborSuiteError, match=r"non-task directory"):
        load_registry(reg)


def test_registry_rejects_unowned_direct_task_bundle(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    _make_canonical_task(tmp_path)
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(ds_path / "suite.toml")
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])

    with pytest.raises(HarborSuiteError, match=r"not assigned in members"):
        load_registry(reg)


def test_committed_examples_suite_allows_empty_tasks() -> None:
    suite = get_suite("jacobian/examples-v1")
    assert suite.tasks == ()


def test_committed_agent_workflow_suite_has_dataset_local_tasks() -> None:
    suite = get_suite("jacobian/mathematical-benchmarks-v1")
    assert suite.tasks
    assert all(
        task.path.parent
        == ROOT / "benchmarks" / "datasets" / "mathematical-benchmarks-v1"
        for task in suite.tasks
    )
