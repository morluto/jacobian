"""Focused unit tests for the Harbor suite infrastructure module."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest
import tomli_w
from benchmarks.tooling.harbor_suite import (
    DIGEST_PREFIX,
    MODEL_PLACEHOLDER,
    TASK_SCHEMA_VERSION,
    HarborSuiteError,
    Suite,
    check_dataset_manifest,
    check_verifier_support,
    expected_dataset_manifest,
    get_suite,
    load_registry,
    render_job_config,
    validate_task_topology,
    validate_task_visibility,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def patched_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect harbor_suite.ROOT to tmp_path so _resolve accepts test paths."""

    import benchmarks.tooling.harbor_suite as hs

    monkeypatch.setattr(hs, "ROOT", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _write_registry(
    tmp_path: Path,
    datasets: list[dict],
) -> Path:
    registry = {"schema_version": "1", "datasets": datasets}
    path = tmp_path / "registry.toml"
    path.write_text(tomli_w.dumps(registry))
    return path


def _make_dataset_entry(ds_id: str, ds_path: Path, **overrides) -> dict:
    entry = {
        "id": ds_id,
        "directory": str(ds_path),
        "evaluation_kind": "test",
        "scored": False,
        "publication_status": "local",
        "required_provider": "core",
        "runtime_profile": "core",
        "title": "Test",
        "purpose": "Test purpose.",
        "claim_class": "test",
        "answer_visibility": "public",
        "default_execution_profile": "oracle-only",
        "jobs": {"oracle": "jobs/oracle.json"},
    }
    entry.update(overrides)
    return entry


def _write_suite_toml(
    path: Path,
    *,
    ds_id: str = "jacobian/test-v1",
    tasks: list[dict] | None = None,
) -> None:
    raw: dict = {
        "schema_version": "1",
        "dataset": {
            "id": ds_id,
            "version": "1.0.0",
            "title": "Test",
            "purpose": "Test purpose.",
        },
    }
    if tasks:
        raw["tasks"] = tasks
    path.write_text(tomli_w.dumps(raw))


def _make_minimal_task(root: Path, *, task_id: str = "jacobian/test-v1-a") -> Path:
    """Create a minimal valid task at ``root`` (the task directory itself)."""

    task = root
    env = task / "environment"
    tests = task / "tests"
    sol = task / "solution"
    env.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    sol.mkdir(parents=True, exist_ok=True)
    (task / "README.md").write_text("# Task A")
    (task / "instruction.md").write_text("Do the task.")
    (task / "task.toml").write_text(
        textwrap.dedent(
            f"""
            schema_version = "{TASK_SCHEMA_VERSION}"
            artifacts = ["/app/submission.json"]
            [task]
            name = "{task_id}"
            version = "1.0.0"
            description = "Test task."
            [metadata]
            evaluation_kind = "workflow"
            domain = "test"
            field = "test"
            assurance_ceiling = "COMPUTED"
            answer_visibility = "hidden"
            provenance_class = "hand-designed"
            fixture_digest = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            required_provider = "core"
            [agent]
            timeout_sec = 60.0
            [verifier]
            timeout_sec = 60.0
            environment_mode = "separate"
            """
        ).strip()
        + "\n"
    )
    (env / "Dockerfile").write_text(
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY input.json submission_schema.json /app/\n"
    )
    (env / "input.json").write_text("{}")
    (env / "submission_schema.json").write_text('{"type": "object"}')
    (tests / "test.sh").write_text("#!/bin/sh\npython /tests/verifier.py\n")
    verifier_source = "print('ok')\n"
    (tests / "verifier.py").write_text(verifier_source)
    verifier_checksum = hashlib.sha256(verifier_source.encode()).hexdigest()
    (tests / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        f'LABEL jacobian.checksum="{verifier_checksum}"\n'
        "COPY test.sh verifier.py verifier_support.py /tests/\n"
    )
    (tests / "verifier_support.py").write_text("# vendored support\n")
    (sol / "solve.sh").write_text("#!/bin/sh\necho done\n")
    return task


def test_load_registry_parses_all_datasets() -> None:
    suites = load_registry()
    ids = {s.id for s in suites}
    assert ids == {
        "agent-workflow-v1",
        "examples-v1",
        "performance-v1",
        "provider-feasibility-v1",
        "public-reproductions-v1",
        "research-diagnostics-v1",
    }


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
    (ds_path / "tasks" / "a").mkdir(parents=True)
    (ds_path / "tasks" / "a" / "solution").mkdir(parents=True)
    _make_minimal_task(ds_path / "tasks" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/a",
                "maximum_assurance": "COMPUTED",
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


def test_suite_parses_tasks_with_maximum_assurance(
    tmp_path: Path, patched_root: Path
) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "tasks" / "a").mkdir(parents=True)
    (ds_path / "tasks" / "a" / "solution").mkdir(parents=True)
    _make_minimal_task(ds_path / "tasks" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/a",
                "maximum_assurance": "COMPUTED",
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
    assert suite.tasks[0].maximum_assurance == "COMPUTED"
    assert suite.tasks[0].required_provider == "core"


def test_suite_parses_verified_ceiling(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "tasks" / "a").mkdir(parents=True)
    (ds_path / "tasks" / "a" / "solution").mkdir(parents=True)
    _make_minimal_task(ds_path / "tasks" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/a",
                "maximum_assurance": "VERIFIED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suite = load_registry(reg)[0]
    assert suite.tasks[0].maximum_assurance == "VERIFIED"


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
    task = ds_path / "outside"
    _make_minimal_task(task)
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "../test-v1/outside",
                "maximum_assurance": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])
    with pytest.raises(HarborSuiteError, match="below"):
        load_registry(reg)


def test_suite_rejects_symlinked_task_path(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    real = ds_path / "tasks" / "real"
    _make_minimal_task(real)
    alias = ds_path / "tasks" / "alias"
    alias.symlink_to(real, target_is_directory=True)
    (ds_path / "jobs").mkdir(parents=True)
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/alias",
                "maximum_assurance": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(tmp_path, [_make_dataset_entry("jacobian/test-v1", ds_path)])
    with pytest.raises(HarborSuiteError, match="symlink"):
        load_registry(reg)


def test_suite_parses_nested_task_paths(tmp_path: Path, patched_root: Path) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "tasks" / "x" / "y" / "z" / "a").mkdir(parents=True)
    (ds_path / "tasks" / "x" / "y" / "z" / "a" / "solution").mkdir(parents=True)
    _make_minimal_task(ds_path / "tasks" / "x" / "y" / "z" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/x/y/z/a",
                "maximum_assurance": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suite = load_registry(reg)[0]
    assert suite.tasks[0].path == (ds_path / "tasks" / "x" / "y" / "z" / "a").resolve()


# ---------------------------------------------------------------------------
# Dataset manifest generation
# ---------------------------------------------------------------------------


def test_expected_dataset_manifest_has_header_and_tasks(
    tmp_path: Path, patched_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds_path = tmp_path / "test-v1"
    (ds_path / "tasks" / "a").mkdir(parents=True)
    (ds_path / "tasks" / "a" / "solution").mkdir(parents=True)
    _make_minimal_task(ds_path / "tasks" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/a",
                "maximum_assurance": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    suite = load_registry(reg)[0]
    monkeypatch.setattr(
        "benchmarks.tooling.harbor_suite.task_digest", lambda p: "a" * 64
    )
    manifest = expected_dataset_manifest(suite)
    assert "[dataset]" in manifest
    assert 'name = "jacobian/test-v1"' in manifest
    assert "[[tasks]]" in manifest
    assert f"{DIGEST_PREFIX}{'a' * 64}" in manifest


def test_check_dataset_manifest_reports_missing_manifest(
    tmp_path: Path, patched_root: Path
) -> None:
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
    failures = check_dataset_manifest(suite)
    assert len(failures) == 1
    assert "missing" in failures[0].lower() or "stale" in failures[0].lower()


# ---------------------------------------------------------------------------
# Job rendering
# ---------------------------------------------------------------------------


def test_render_job_config_resolves_model() -> None:
    config = {
        "agents": [{"name": "codex", "model_name": MODEL_PLACEHOLDER}],
        "tasks": [{"path": "some/path"}],
    }
    rendered = render_job_config(config, model="gpt-5")
    assert rendered["agents"][0]["model_name"] == "gpt-5"


def test_render_job_config_rejects_unresolved_placeholders() -> None:
    config = {
        "agents": [{"name": "codex", "model_name": MODEL_PLACEHOLDER}],
        "tasks": [{"path": "${OTHER_VAR}"}],
    }
    with pytest.raises(ValueError, match="unresolved"):
        render_job_config(config, model="gpt-5")


def test_render_job_config_rejects_missing_agents() -> None:
    config = {"tasks": [{"path": "x"}]}
    with pytest.raises(ValueError, match="agents"):
        render_job_config(config, model="gpt-5")


def test_render_job_config_rejects_no_placeholder() -> None:
    config = {"agents": [{"name": "codex", "model_name": "already-set"}]}
    with pytest.raises(ValueError, match="does not contain"):
        render_job_config(config, model="gpt-5")


# ---------------------------------------------------------------------------
# Topology and visibility
# ---------------------------------------------------------------------------


def _make_suite_with_task(tmp_path: Path) -> tuple[Suite, Path]:
    ds_path = tmp_path / "test-v1"
    (ds_path / "tasks" / "a").mkdir(parents=True)
    task = _make_minimal_task(ds_path / "tasks" / "a")
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    canonical = tmp_path / "benchmarks" / "tooling"
    canonical.mkdir(parents=True)
    (canonical / "verifier_support.py").write_text("# vendored support\n")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[
            {
                "id": "jacobian/test-v1-a",
                "path": "tasks/a",
                "maximum_assurance": "COMPUTED",
                "required_provider": "core",
            }
        ],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    return load_registry(reg)[0], task


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


def test_check_verifier_support_passes_when_all_copies_match(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    support = "# canonical support\n"
    (patched_root / "benchmarks" / "tooling" / "verifier_support.py").write_text(
        support
    )
    (task / "tests" / "verifier_support.py").write_text(support)
    assert check_verifier_support(suite) == []


def test_check_verifier_support_reports_drift(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    (patched_root / "benchmarks" / "tooling" / "verifier_support.py").write_text(
        "# canonical\n"
    )
    (task / "tests" / "verifier_support.py").write_text("# drifted\n")
    failures = check_verifier_support(suite)
    assert any("differs" in f for f in failures)


def test_check_verifier_support_uses_repository_canonical_copy(
    tmp_path: Path, patched_root: Path
) -> None:
    suite, _task = _make_suite_with_task(tmp_path)
    # No dataset-root verifier_support.py is needed: the repository-owned
    # canonical copy is the only source of truth.
    assert check_verifier_support(suite) == []


# ---------------------------------------------------------------------------
# Integration with committed datasets
# ---------------------------------------------------------------------------


def test_committed_performance_suite_has_four_tasks() -> None:
    suite = get_suite("jacobian/performance-v1")
    assert len(suite.tasks) == 4
    assert all(t.maximum_assurance == "COMPUTED" for t in suite.tasks)
    assert all(t.required_provider == "core" for t in suite.tasks)


def test_committed_provider_suite_has_six_tasks() -> None:
    suite = get_suite("jacobian/provider-feasibility-v1")
    assert len(suite.tasks) == 6


def test_committed_examples_suite_allows_empty_tasks() -> None:
    suite = get_suite("jacobian/examples-v1")
    assert suite.tasks == ()


def test_committed_agent_workflow_suite_has_26_tasks() -> None:
    suite = get_suite("jacobian/agent-workflow-v1")
    assert len(suite.tasks) == 26
