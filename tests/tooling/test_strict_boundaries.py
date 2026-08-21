"""Focused tests for the strict typed-boundary module and its loader integrations.

These cover the PR3 strict-configuration front door: closed ``extra="forbid"``
Pydantic models with strict scalars that validate authored config *before* any
semantic ``.get``/iteration, returning field-path/code diagnostics through the
existing ``HarborSuiteError`` convention, plus the atomic allow_apt fix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import tomli_w
from benchmarks.tooling.benchmark_job_models import (
    HarborJobDatasetEntry,
    HarborJobSelection,
)
from benchmarks.tooling.harbor_suite import (
    HarborSuiteError,
    load_environment_profiles,
    validate_task_topology,
)
from benchmarks.tooling.harbor_task_contract import (
    TaskEnvironmentSection,
    TaskManifestSections,
)
from benchmarks.tooling.heldout_plan_models import HeldoutRunPlan
from benchmarks.tooling.heldout_runner import _validated_plan
from benchmarks.tooling.strict_boundaries import (
    format_strict_errors,
    raise_strict_model,
    strict_model_failures,
)
from pydantic import ValidationError

from tests.tooling.harbor_suite_support import (
    _make_suite_with_task,
)

# ---------------------------------------------------------------------------
# Strict boundary models
# ---------------------------------------------------------------------------


def _json_digest(value: object) -> str:
    """Compute the canonical plan digest used by the held-out run plan loader."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_task_environment_section_rejects_extra_fields() -> None:
    failures = strict_model_failures(
        TaskEnvironmentSection,
        {"network_mode": "no-network", "gpu": True},
        label="env",
    )
    assert any("extra" in f and "env.gpu" in f for f in failures)


def test_task_environment_section_rejects_non_str_network_mode() -> None:
    failures = strict_model_failures(
        TaskEnvironmentSection,
        {"network_mode": 1},
        label="env",
    )
    assert any("env.network_mode" in f for f in failures)


def test_task_manifest_sections_requires_task_section() -> None:
    failures = strict_model_failures(
        TaskManifestSections,
        {"schema_version": "1.4"},
        label="task.toml",
    )
    assert any("task.toml.task" in f for f in failures)


def test_task_manifest_sections_rejects_string_timeout() -> None:
    failures = strict_model_failures(
        TaskManifestSections,
        {
            "schema_version": "1.4",
            "task": {"name": "x", "version": "1.0.0"},
            "agent": {"timeout_sec": "60"},
        },
        label="task.toml",
    )
    assert any("task.toml.agent.timeout_sec" in f for f in failures)


def test_harbor_job_selection_rejects_extra_field_in_dataset_entry() -> None:
    failures = strict_model_failures(
        HarborJobSelection,
        {"datasets": [{"path": "p", "evil": True}]},
        label="job.json",
    )
    assert any("job.json.datasets" in f for f in failures)


def test_harbor_job_selection_rejects_non_str_path() -> None:
    failures = strict_model_failures(
        HarborJobSelection,
        {"tasks": [{"path": 1}]},
        label="job.json",
    )
    assert any("job.json.tasks" in f for f in failures)


def test_harbor_job_dataset_entry_accepts_valid_selection() -> None:
    entry = HarborJobDatasetEntry.model_validate(
        {"path": "p", "task_names": ["a", "b"]}
    )
    assert entry.path == "p"
    assert entry.task_names == ["a", "b"]


def test_heldout_run_plan_rejects_invalid_condition() -> None:
    failures = strict_model_failures(
        HeldoutRunPlan,
        {
            "schema_version": "2",
            "harbor_version": "0.20.0",
            "pair_count": 1,
            "budget": {
                "max_tokens": 1,
                "max_cost_usd": 1.0,
                "enforcement": "x",
                "missing_accounting": "x",
                "overage": "x",
            },
            "runs": [{"pair_id": "p", "condition": "C3", "job": "j", "jobs_dir": "d"}],
            "plan_digest": "sha256:" + "0" * 64,
        },
        label="run-plan.json",
    )
    assert any("run-plan.json.runs" in f and "C3" not in f for f in failures)


def test_heldout_run_plan_rejects_extra_top_level_field() -> None:
    failures = strict_model_failures(
        HeldoutRunPlan,
        {
            "schema_version": "2",
            "harbor_version": "0.20.0",
            "pair_count": 1,
            "budget": {
                "max_tokens": 1,
                "max_cost_usd": 1.0,
                "enforcement": "x",
                "missing_accounting": "x",
                "overage": "x",
            },
            "runs": [{"pair_id": "p", "condition": "C1", "job": "j", "jobs_dir": "d"}],
            "plan_digest": "sha256:" + "0" * 64,
            "rogue": True,
        },
        label="run-plan.json",
    )
    assert any("rogue" in f for f in failures)


def test_format_strict_errors_carries_field_path_and_code() -> None:
    try:
        TaskEnvironmentSection.model_validate({"network_mode": 1})
    except ValidationError as exc:
        failures = format_strict_errors(exc, label="lbl")
        assert failures
        assert all(f.startswith("lbl.") for f in failures)
        assert all("(" in f and ")" in f for f in failures)
    else:
        pytest.fail("expected ValidationError")


def test_raise_strict_model_raises_harbor_suite_error() -> None:
    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        raise_strict_model(
            TaskEnvironmentSection,
            {"network_mode": 1},
            label="lbl",
        )


# ---------------------------------------------------------------------------
# harbor_suite integration: strict task/environment sections
# ---------------------------------------------------------------------------


def test_topology_reports_strict_failure_for_extra_environment_field(
    tmp_path: Path, synthetic_harbor_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    task_toml = (task / "task.toml").read_text()
    task_toml = task_toml.replace(
        "[environment]\n",
        "[environment]\nrogue_field = true\n",
    )
    (task / "task.toml").write_text(task_toml)
    failures = validate_task_topology(suite, task)
    assert any("task.toml.environment.rogue_field" in f for f in failures)


def test_topology_reports_strict_failure_for_non_float_agent_timeout(
    tmp_path: Path, synthetic_harbor_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    task_toml = (task / "task.toml").read_text()
    task_toml = task_toml.replace("timeout_sec = 60.0\n", 'timeout_sec = "slow"\n', 1)
    (task / "task.toml").write_text(task_toml)
    failures = validate_task_topology(suite, task)
    assert any("task.toml.agent.timeout_sec" in f for f in failures)


def test_topology_passes_for_valid_minimal_task(
    tmp_path: Path, synthetic_harbor_root: Path
) -> None:
    suite, task = _make_suite_with_task(tmp_path)
    assert validate_task_topology(suite, task) == []


# ---------------------------------------------------------------------------
# harbor_suite integration: allow_apt strict bool
# ---------------------------------------------------------------------------


def _write_profiles(tmp_path: Path, allow_apt_value: object) -> Path:
    path = tmp_path / "environment-profiles.toml"
    path.write_text(
        tomli_w.dumps(
            {
                "schema_version": "1",
                "profiles": {
                    "p": {
                        "agent_image": "python:3.12-slim@sha256:" + "0" * 64,
                        "verifier_image": "python:3.12-slim@sha256:" + "0" * 64,
                        "allow_apt": allow_apt_value,
                    }
                },
            }
        )
    )
    return path


def test_load_environment_profiles_accepts_real_bool(tmp_path: Path) -> None:
    path = _write_profiles(tmp_path, True)
    profiles = load_environment_profiles(path)
    assert profiles["p"].allow_apt is True


def test_load_environment_profiles_rejects_non_bool_allow_apt(tmp_path: Path) -> None:
    path = _write_profiles(tmp_path, "true")
    with pytest.raises(HarborSuiteError, match="allow_apt must be a bool"):
        load_environment_profiles(path)


def test_load_environment_profiles_rejects_missing_allow_apt(tmp_path: Path) -> None:
    path = tmp_path / "environment-profiles.toml"
    path.write_text(
        tomli_w.dumps(
            {
                "schema_version": "1",
                "profiles": {
                    "p": {
                        "agent_image": "python:3.12-slim@sha256:" + "0" * 64,
                        "verifier_image": "python:3.12-slim@sha256:" + "0" * 64,
                    }
                },
            }
        )
    )
    with pytest.raises(HarborSuiteError, match="allow_apt must be a bool"):
        load_environment_profiles(path)


# ---------------------------------------------------------------------------
# heldout_runner integration: strict run-plan validation before side effects
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, runs: list[dict]) -> Path:
    plan = {
        "schema_version": "3",
        "manifest_digest": "sha256:" + "a" * 64,
        "pair_count": 1,
        "budget": {
            "max_tokens": 100,
            "max_cost_usd": 2.0,
            "enforcement": "PAIR_BOUNDARY_POST_RUN",
            "missing_accounting": "INCOMPLETE",
            "overage": "INCOMPLETE",
        },
        "runs": runs,
    }
    plan["plan_digest"] = _json_digest(plan)
    path = tmp_path / "run-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_heldout_cost_budget_accepts_integer_json_number(tmp_path: Path) -> None:
    path = _write_plan(
        tmp_path,
        [
            {"pair_id": "p", "condition": condition, "job": "j", "jobs_dir": "d"}
            for condition in ("C1", "C2")
        ],
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    value["budget"]["max_cost_usd"] = 2
    value["plan_digest"] = _json_digest(value)

    parsed = HeldoutRunPlan.model_validate(value)

    assert parsed.budget.max_cost_usd == 2.0


def test_validated_plan_rejects_run_missing_pair_id(tmp_path: Path) -> None:
    path = _write_plan(
        tmp_path,
        [{"condition": "C1", "job": "j", "jobs_dir": "d"}],
    )
    with pytest.raises(
        HarborSuiteError, match="strict configuration validation failed"
    ):
        _validated_plan(path)


def test_validated_plan_rejects_extra_run_field(tmp_path: Path) -> None:
    path = _write_plan(
        tmp_path,
        [
            {
                "pair_id": "p",
                "condition": "C1",
                "job": "j",
                "jobs_dir": "d",
                "rogue": True,
            }
        ],
    )
    with pytest.raises(HarborSuiteError, match="rogue"):
        _validated_plan(path)


def test_validated_plan_accepts_full_rendered_entry(tmp_path: Path) -> None:
    common = {
        "pair_id": "p",
        "pair_index": 0,
        "task": "t",
        "repetition": 1,
        "job": "j",
        "runtime_snapshot": "s",
    }
    path = _write_plan(
        tmp_path,
        [
            {
                **common,
                "condition": "C1",
                "jacobian_enabled": False,
                "jobs_dir": "d-c1",
            },
            {
                **common,
                "condition": "C2",
                "jacobian_enabled": True,
                "jobs_dir": "d-c2",
            },
        ],
    )
    plan, _digest, pair_ids = _validated_plan(path)
    assert pair_ids == ["p"]
    assert plan["runs"][0]["condition"] == "C1"
