from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[3]
DATASET = ROOT / "benchmarks" / "datasets" / "agent-workflow-v1"
TASKS = DATASET / "tasks"
EXPECTED_TASKS = {
    "autoformalization-semantic-audit",
    "calendar-good-days-audit",
    "distinct-sum-pairing-optimum",
    "divisibility-construction-witness",
    "euler-line-symbolic-certificate",
    "finite-magma-countermodel",
    "grounded-premise-proof",
    "graph-counterexample",
    "graph-artifact-composition",
    "finite-partition",
    "sat-witness",
    "rational-linear-solution",
    "hermite-normal-form",
    "log-exponent-recovery",
    "log-inequality-meta-audit",
    "matrix-square-zero-counterexample",
    "metric-tsp-proof-repair",
    "modular-cubic-obstruction",
    "natural-subtraction-proof-repair",
    "nondifferentiable-maximum-construction",
    "polynomial-normalization",
    "polynomial-map-collision",
    "polynomial-tail-counterexample",
    "random-function-expectation-audit",
    "subspace-direct-sum-counterexample",
    "well-total-domination-counterexample",
}
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
    return sorted(path.parent for path in TASKS.rglob("task.toml"))


def test_agent_workflow_v1_has_nested_tasks_and_generated_manifest() -> None:
    manifest = tomllib.loads((DATASET / "dataset.toml").read_text())
    assert manifest["dataset"]["name"] == "jacobian/agent-workflow-v1"
    task_paths = task_dirs()
    assert {path.name for path in task_paths} == EXPECTED_TASKS
    manifest_names = {entry["name"] for entry in manifest["tasks"]}
    assert manifest_names == {
        f"jacobian/agent-workflow-v1-{name}" for name in EXPECTED_TASKS
    }

    for task in task_paths:
        config = tomllib.loads((task / "task.toml").read_text())
        metadata = config["metadata"]
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
