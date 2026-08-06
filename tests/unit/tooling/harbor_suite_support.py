"""Fixture builders shared by Harbor suite policy tests."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path

import pytest
import tomli_w
from benchmarks.tooling.harbor_suite import (
    TASK_SCHEMA_VERSION,
    EnvironmentProfile,
    Suite,
    load_registry,
)


def patch_harbor_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect harbor_suite.ROOT to tmp_path so _resolve accepts test paths."""

    import benchmarks.tooling.harbor_suite as hs

    monkeypatch.setattr(hs, "ROOT", tmp_path)
    profile = EnvironmentProfile(
        name="test-profile",
        agent_image="python:3.12-slim",
        verifier_image="python:3.12-slim",
        allow_apt=False,
    )
    monkeypatch.setattr(
        hs, "load_environment_profiles", lambda: {profile.name: profile}
    )
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
        "schema_version": "2",
        "dataset": {
            "id": ds_id,
            "title": "Test",
            "purpose": "Test purpose.",
        },
    }
    path.write_text(tomli_w.dumps(raw))
    members = path.parent / "members"
    members.mkdir(exist_ok=True)
    if tasks:
        for entry in tasks:
            task_id = str(entry["id"]).removeprefix("jacobian/")
            member = {
                "schema_version": "2",
                "task_id": task_id,
                "task_name": f"jacobian/{task_id}",
                "evaluation_kind": "workflow",
                "domain": "test",
                "primary_domain": "algebra",
                "field": "test",
                "provenance_class": "hand-designed",
                "provenance_ref": "unit-test fixture",
                "assurance_ceiling": entry["assurance_ceiling"],
                "required_provider": entry.get("required_provider", "core"),
                "environment_profile": "test-profile",
                "verifier_contract_version": "1",
                "evaluation_owner": ds_id,
            }
            member_name = task_id.replace("/", "-")
            (members / f"{member_name}.toml").write_text(tomli_w.dumps(member))


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
            primary_domain = "algebra"
            field = "test"
            assurance_ceiling = "COMPUTED"
            answer_visibility = "hidden"
            provenance_class = "hand-designed"
            fixture_digest = "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
            required_provider = "core"
            [agent]
            timeout_sec = 60.0
            [environment]
            network_mode = "no-network"
            [verifier]
            timeout_sec = 60.0
            environment_mode = "separate"
            [verifier.environment]
            network_mode = "no-network"
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


def _make_canonical_task(tmp_path: Path, *, task_id: str = "test-v1-a") -> Path:
    (tmp_path / "test-v1").mkdir(parents=True, exist_ok=True)
    return _make_minimal_task(
        tmp_path / "test-v1" / task_id,
        task_id=f"jacobian/{task_id}",
    )


def _make_suite_with_task(tmp_path: Path) -> tuple[Suite, Path]:
    ds_path = tmp_path / "test-v1"
    task = _make_canonical_task(tmp_path)
    (ds_path / "jobs").mkdir()
    (ds_path / "jobs" / "oracle.json").write_text("{}")
    _write_suite_toml(
        ds_path / "suite.toml",
        tasks=[{"id": "jacobian/test-v1-a", "assurance_ceiling": "COMPUTED"}],
    )
    reg = _write_registry(
        tmp_path,
        [_make_dataset_entry("jacobian/test-v1", ds_path)],
    )
    return load_registry(reg)[0], task
