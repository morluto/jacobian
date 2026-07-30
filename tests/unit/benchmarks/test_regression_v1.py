from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).parents[3]
DATASET = ROOT / "benchmarks" / "regression-v1"
TASKS = DATASET / "tasks"
EXPECTED_TASKS = {
    "graph-counterexample",
    "graph-artifact-composition",
    "finite-partition",
    "sat-witness",
    "rational-linear-solution",
    "hermite-normal-form",
    "polynomial-normalization",
    "polynomial-map-collision",
}
VERIFICATION_RECORD_TASKS = {
    "finite-partition",
    "hermite-normal-form",
    "polynomial-normalization",
    "polynomial-map-collision",
    "sat-witness",
}


def test_regression_v1_is_a_frozen_eight_task_dataset() -> None:
    manifest = tomllib.loads((DATASET / "dataset.toml").read_text())
    assert manifest["dataset"]["name"] == "jacobian/regression-v1"
    assert {
        task["name"].rsplit("/", 1)[-1].removeprefix("regression-v1-")
        for task in manifest["tasks"]
    } == EXPECTED_TASKS

    task_dirs = {path.name for path in TASKS.iterdir() if path.is_dir()}
    assert task_dirs == EXPECTED_TASKS

    verifier_dockerfile_sizes = []
    for task_name in sorted(EXPECTED_TASKS):
        task = TASKS / task_name
        spec = tomllib.loads((task / "task.toml").read_text())
        assert spec["schema_version"] == "1.4"
        assert spec["task"]["name"] == f"jacobian/regression-v1-{task_name}"
        assert spec["environment"]["network_mode"] == "no-network"
        assert spec["verifier"]["environment"]["network_mode"] == "no-network"

        input_bytes = (task / "input.json").read_bytes()
        assert input_bytes == (task / "environment" / "input.json").read_bytes()
        assert input_bytes == (task / "tests" / "input.json").read_bytes()
        json.loads(input_bytes)
        input_digest = "sha256:" + hashlib.sha256(input_bytes).hexdigest()
        metadata = json.loads((task / "metadata.json").read_text())
        assert metadata["case_version"] == "regression-v1"
        assert metadata["fixture_digest"] == input_digest
        assert (
            json.loads((task / "environment" / "metadata.json").read_text()) == metadata
        )
        assert spec["metadata"]["fixture_digest"] == input_digest
        assert metadata["upstream"] is None

        instruction = (task / "instruction.md").read_text()
        submission_schema = json.loads(
            (task / "environment" / "submission_schema.json").read_text()
        )
        assert submission_schema["type"] == "object"
        assert submission_schema["additionalProperties"] is False
        assert set(submission_schema["required"]) == {
            "task_id",
            "conclusion",
            "result",
            "claimed_assurance",
            "scope",
            "completeness",
            "evidence",
            "limitations",
        }
        expected = json.loads((task / "tests" / "expected.json").read_text())
        build_expected = task / "tests" / f"{task_name}-expected.json"
        if build_expected.exists():
            assert json.loads(build_expected.read_text()) == expected
        if task_name in VERIFICATION_RECORD_TASKS:
            assert "verification_record_uri" in submission_schema["properties"]
            assert submission_schema["then"]["required"] == ["verification_record_uri"]
            record_schema = json.loads(
                (task / "environment" / "verification_record_schema.json").read_text()
            )
            Draft202012Validator.check_schema(record_schema)
            solution_record = json.loads(
                (task / "solution" / "verification-record.json").read_text()
            )
            Draft202012Validator(record_schema).validate(solution_record)
            record_name = (
                "authorized_records.json"
                if task_name == "sat-witness"
                else "authorized_record.json"
            )
            assert (task / "tests" / record_name).is_file()
            assert expected["maximum_assurance"] == "VERIFIED"
            assert "verification_record_schema.json" in instruction
        else:
            assert "verification_record_uri" not in submission_schema["properties"]
            assert expected["maximum_assurance"] == "COMPUTED"
            if task_name == "rational-linear-solution":
                assert not (task / "tests" / "authorized_record.json").exists()
        assert "submission_schema.json" in instruction
        assert "evidence/answer.txt" in instruction
        assert "capability_id" not in instruction
        assert "agent-specific" not in instruction.lower()
        assert "jacobian" not in instruction.lower()
        assert "toolbox" not in instruction.lower()

        for dockerfile in (
            task / "environment" / "Dockerfile",
            task / "tests" / "Dockerfile",
        ):
            assert "@sha256:" in dockerfile.read_text()
        environment_dockerfile = (task / "environment" / "Dockerfile").read_text()
        assert "submission_schema.json" in environment_dockerfile
        if task_name in VERIFICATION_RECORD_TASKS:
            assert "verification_record_schema.json" in environment_dockerfile
        tests_dockerfile = (task / "tests" / "Dockerfile").read_text()
        verifier_dockerfile_sizes.append((task / "tests" / "Dockerfile").stat().st_size)
        verifier_digest = hashlib.sha256(
            (task / "tests" / "verifier.py").read_bytes()
        ).hexdigest()
        assert f'jacobian.checksum="{verifier_digest}"' in tests_dockerfile
        assert "COPY . /tests" not in tests_dockerfile
        assert f'== "{task_name}"' in tests_dockerfile
        build_input = task / "tests" / f"{task_name}-input.json"
        if build_input.exists():
            assert build_input.read_bytes() == input_bytes

    assert len(verifier_dockerfile_sizes) == len(set(verifier_dockerfile_sizes))
