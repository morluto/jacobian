"""Harbor task topology and visibility policy tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.tooling.harbor_suite import (
    validate_task_topology,
    validate_task_visibility,
)
from tests.unit.tooling.harbor_suite_support import (
    _make_suite_with_task,
    patch_harbor_root,
)


@pytest.fixture
def patched_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    return patch_harbor_root(monkeypatch, tmp_path)


def test_validate_task_topology_passes_for_minimal_task(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    assert validate_task_topology(suite, task) == []


def test_validate_task_topology_reports_missing_readme(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (task / "README.md").unlink()
    failures = validate_task_topology(suite, task)
    assert any("README.md" in f for f in failures)


def test_validate_task_topology_reports_missing_metadata_field(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    task_toml = (task / "task.toml").read_text()
    task_toml = task_toml.replace('evaluation_kind = "workflow"\n', "")
    (task / "task.toml").write_text(task_toml)
    failures = validate_task_topology(suite, task)
    assert any("metadata" in f.lower() for f in failures)


def test_validate_task_topology_reports_unknown_environment_profile(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    unknown = dataclasses.replace(suite.tasks[0], environment_profile="not-a-profile")
    suite = dataclasses.replace(suite, tasks=(unknown,))

    failures = validate_task_topology(suite, task)

    assert any("unknown environment profile" in failure for failure in failures)


def test_validate_task_topology_reports_workflow_fixture_digest_drift(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    task_toml = (
        (task / "task.toml")
        .read_text()
        .replace(
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
            "sha256:" + "0" * 64,
        )
    )
    (task / "task.toml").write_text(task_toml)
    failures = validate_task_topology(suite, task)
    assert any("fixture_digest" in f for f in failures)


def test_validate_task_topology_reports_env_dockerfile_copies_solution(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY solution/solve.sh /app/\n"
    )
    failures = validate_task_topology(suite, task)
    assert any("hidden" in f.lower() for f in failures)


def test_validate_task_topology_forbids_root_input_json(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (task / "input.json").write_text("{}")
    failures = validate_task_topology(suite, task)
    assert any("input.json" in f for f in failures)


def test_validate_task_topology_forbids_raw_interpreter_caches(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    cache = task / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "verifier.cpython-312.pyc").write_bytes(b"cache")
    failures = validate_task_topology(suite, task)
    assert any("cache" in f for f in failures)


def test_validate_task_topology_ignores_gitignored_interpreter_caches(
    tmp_path: Path, patched_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.tooling.harbor_suite as harbor_suite

    monkeypatch.setattr(
        harbor_suite.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    suite, task = _make_suite_with_task(tmp_path)
    cache = task / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "verifier.cpython-312.pyc").write_bytes(b"cache")

    assert validate_task_topology(suite, task) == []


def test_validate_task_topology_does_not_ignore_tracked_interpreter_caches(
    tmp_path: Path, patched_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.tooling.harbor_suite as harbor_suite

    calls: list[list[str]] = []

    def check_tracked_cache(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(harbor_suite.subprocess, "run", check_tracked_cache)
    suite, task = _make_suite_with_task(tmp_path)
    cache = task / "tests" / "__pycache__"
    cache.mkdir()
    (cache / "verifier.cpython-312.pyc").write_bytes(b"cache")

    failures = validate_task_topology(suite, task)

    assert any("raw interpreter cache" in failure for failure in failures)
    assert all("--no-index" not in args for args in calls)


def test_validate_task_visibility_detects_host_path(
    tmp_path: Path, patched_root: Path
) -> None:
    _suite, task = _make_suite_with_task(tmp_path)
    (task / "instruction.md").write_text("Read /home/user/secret.txt for the answer.")
    failures = validate_task_visibility(task)
    assert any("host path" in f for f in failures)


def test_validate_task_visibility_detects_secret(
    tmp_path: Path, patched_root: Path
) -> None:
    _suite, task = _make_suite_with_task(tmp_path)
    (task / "instruction.md").write_text('api_key = "supersecretkey12345"')
    failures = validate_task_visibility(task)
    assert any("secret" in f for f in failures)


def test_validate_task_visibility_rejects_oracle_named_files(
    tmp_path: Path, patched_root: Path
) -> None:
    _suite, task = _make_suite_with_task(tmp_path)
    (task / "environment" / "expected.json").write_text("{}")
    failures = validate_task_visibility(task)
    assert any("Oracle/verifier" in f for f in failures)


def test_validate_task_visibility_clean_for_normal_task(
    tmp_path: Path, patched_root: Path
) -> None:
    _suite, task = _make_suite_with_task(tmp_path)
    assert validate_task_visibility(task) == []


# ---------------------------------------------------------------------------
# Verifier support
# ---------------------------------------------------------------------------
