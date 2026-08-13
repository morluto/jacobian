from __future__ import annotations

import json
import tomllib
from pathlib import Path

from benchmarks.tooling.harbor_suite import get_suite

ROOT = Path(__file__).parents[2]
DATASET = ROOT / "benchmarks" / "datasets" / "mathematical-benchmarks-v1"
REQUIRED_METADATA = {
    "evaluation_kind",
    "domain",
    "primary_domain",
    "field",
    "assurance_ceiling",
    "answer_visibility",
    "provenance_class",
    "fixture_digest",
    "required_provider",
}
ASSURANCE_ORDER = ("UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED")


def task_dirs() -> list[Path]:
    return sorted(ref.path for ref in get_suite("mathematical-benchmarks-v1").tasks)


def test_mathematical_benchmarks_v1_has_flat_tasks_and_authoritative_members() -> None:
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
        assert member["evaluation_owner"] == "jacobian/mathematical-benchmarks-v1"
        assert member["verifier_contract_version"] == "1"
        assert set(metadata) >= REQUIRED_METADATA
        assert metadata["domain"] == "mathematical-sciences"
        assert metadata["primary_domain"]
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
        submission_schema = json.loads(
            (task / "environment" / "submission_schema.json").read_text()
        )
        # The submission schema accepts the full assurance vocabulary; the
        # scoreable subset is declared in the task-owned public_contract.json.
        assert submission_schema["properties"]["claimed_assurance"]["enum"] == list(
            ASSURANCE_ORDER
        )
        public_contract = json.loads(
            (task / "tests" / "public_contract.json").read_text()
        )
        advertised_assurances = public_contract["allowed_assurance"]
        ceiling_index = ASSURANCE_ORDER.index(metadata["assurance_ceiling"])
        assert metadata["assurance_ceiling"] in advertised_assurances
        assert all(
            assurance in ASSURANCE_ORDER[: ceiling_index + 1]
            for assurance in advertised_assurances
        )
        assert input_data["task_id"] == config["task"]["name"]
        assert len(metadata["fixture_digest"]) == len("sha256:") + 64
        instruction = (task / "instruction.md").read_text().lower()
        assert "operation_id" not in instruction
        assert "toolbox" not in instruction
        assert "jacobian" not in instruction
        for path in task.rglob("*"):
            assert not path.is_symlink()
