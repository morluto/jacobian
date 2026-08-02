from __future__ import annotations

import json
import tomllib
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[2]
DATASET = ROOT / "benchmarks" / "datasets" / "agent-workflow-v1"
REQUIRED_METADATA = {
    "evaluation_kind",
    "domain",
    "field",
    "assurance_ceiling",
    "answer_visibility",
    "provenance_class",
    "fixture_digest",
    "required_provider",
}


def task_dirs() -> list[Path]:
    return sorted(ref.path for ref in get_suite("agent-workflow-v1").tasks)


def test_agent_workflow_v1_has_flat_tasks_and_authoritative_members() -> None:
    task_paths = task_dirs()
    assert task_paths
    member_ids = {path.stem for path in (DATASET / "members").glob("*.toml")}
    assert member_ids == {path.name for path in task_paths}
    assert not (DATASET / "dataset.toml").exists()

    for task in task_paths:
        config = tomllib.loads((task / "task.toml").read_text())
        metadata = config["metadata"]
        member = tomllib.loads((DATASET / "members" / f"{task.name}.toml").read_text())
        assert member["task_name"] == config["task"]["name"]
        assert member["evaluation_owner"] == "jacobian/agent-workflow-v1"
        assert member["verifier_contract_version"] == "1"
        assert set(metadata) >= REQUIRED_METADATA
        assert metadata["domain"] == "mathematical-sciences"
        assert metadata["required_provider"] == "core"
        assert metadata["fixture_digest"].startswith("sha256:")
        assert (task / "README.md").is_file()
        assert (task / "instruction.md").is_file()
        environment_dockerfile = task / "environment" / "Dockerfile"
        assert environment_dockerfile.is_file()
        assert (task / "environment" / "input.json").is_file()
        assert (task / "environment" / "submission_schema.json").is_file()
        assert (task / "solution").is_dir()
        assert (task / "tests" / "Dockerfile").is_file()
        assert (task / "tests" / "test.sh").is_file()
        assert (task / "tests" / "verifier.py").is_file()
        assert (task / "tests" / "verifier_support.py").is_file()
        assert not (task / "input.json").exists()
        assert not (task / "metadata.json").exists()
        assert not (task / "environment" / "metadata.json").exists()
        assert "metadata.json" not in environment_dockerfile.read_text()
        input_data = json.loads((task / "environment" / "input.json").read_text())
        assert input_data["task_id"] == config["task"]["name"]
        assert len(metadata["fixture_digest"]) == len("sha256:") + 64
        instruction = (task / "instruction.md").read_text().lower()
        assert "capability_id" not in instruction
        assert "toolbox" not in instruction
        assert "jacobian" not in instruction
        for path in task.rglob("*"):
            assert not path.is_symlink()
