from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from benchmarks.tooling.trajectory_value_study import (
    TrajectoryValueStudySpec,
    _codex_arguments,
    _submission_schema,
    load_spec,
    run_study,
)
from benchmarks.tooling.trajectory_value_study_verifier import verify_workspace
from jsonschema import Draft202012Validator
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "benchmarks/config/trajectory-value-study-v1.json"


def _task(spec: TrajectoryValueStudySpec, task_id: str):
    return next(task for task in spec.tasks if task.task_id == task_id)


def _task_payload(task) -> dict[str, object]:
    return {
        "schema_version": "1",
        "task_id": task.task_id,
        "task_group": task.task_group,
        "task_family": task.task_family,
        "kind": task.kind,
        "statement": task.statement,
        "payload": task.payload,
    }


def _verify(tmp_path: Path, task, answer: object):
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "submission.json").write_text(
        json.dumps({"task_id": task.task_id, "answer": answer}),
        encoding="utf-8",
    )
    return verify_workspace(payload, tmp_path)


def test_frozen_spec_covers_four_families_and_repeated_rollouts() -> None:
    spec = load_spec(SPEC_PATH)

    assert spec.model.model_id == "gpt-5.4-mini"
    assert spec.model.reasoning_effort == "medium"
    assert spec.repetitions_per_task == 4
    assert len({task.task_family for task in spec.tasks}) == 4
    assert spec.reasoning_log_mode == "REQUIRED"
    assert spec.training_performed is False
    assert spec.scorer_intervention is False
    assert spec.exact_resume_supported is False


def test_committed_study_schema_matches_authoritative_contract() -> None:
    path = (
        ROOT
        / "docs/reference/evaluations/schemas/trajectory-value-study-v1.schema.json"
    )

    assert json.loads(path.read_text(encoding="utf-8")) == (
        TrajectoryValueStudySpec.model_json_schema(mode="validation")
    )


@pytest.mark.parametrize(
    ("task_id", "answer"),
    [
        (
            "integer-bezout-01",
            {"gcd": "99", "left_coefficient": "21", "right_coefficient": "-100"},
        ),
        ("matrix-determinant-01", {"determinant": "4312753"}),
        (
            "polynomial-gcd-bezout-01",
            {
                "gcd": ["1", "1", "1"],
                "left_bezout": ["-1/23", "4/23", "3/46"],
                "right_bezout": ["-13/46", "7/46", "-5/46", "-3/46"],
            },
        ),
        (
            "graph-independent-set-01",
            {"vertices": ["c", "e", "h", "j", "l"], "optimum": 5},
        ),
    ],
)
def test_clean_room_verifier_accepts_exact_alternate_witnesses(
    tmp_path: Path, task_id: str, answer: object
) -> None:
    spec = load_spec(SPEC_PATH)

    result = _verify(tmp_path, _task(spec, task_id), answer)

    assert result["acceptance"] == "ACCEPTED"
    assert result["input_binding_valid"] is True
    assert result["artifact_binding_valid"] is True
    assert result["clean_room"] is True
    assert all(result["checks"].values())


def test_verifier_rejects_wrong_bound_answer_without_losing_bindings(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "matrix-determinant-01")

    result = _verify(tmp_path, task, {"determinant": "0"})

    assert result["acceptance"] == "REJECTED"
    assert result["input_binding_valid"] is True
    assert result["artifact_binding_valid"] is True
    assert result["checks"]["exact_relation"] is False


def test_verifier_treats_missing_artifact_as_inconclusive(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    (tmp_path / "task.json").write_text(json.dumps(payload), encoding="utf-8")

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "INCONCLUSIVE"
    assert result["artifact_binding_valid"] is False


def test_verifier_rejects_substituted_task_binding(tmp_path: Path) -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "integer-bezout-01")
    payload = _task_payload(task)
    substituted = json.loads(json.dumps(payload))
    substituted["payload"]["left"] = "1"
    (tmp_path / "task.json").write_text(json.dumps(substituted), encoding="utf-8")
    (tmp_path / "submission.json").write_text("{}", encoding="utf-8")

    result = verify_workspace(payload, tmp_path)

    assert result["acceptance"] == "INCONCLUSIVE"
    assert result["input_binding_valid"] is False
    assert result["artifact_binding_valid"] is True


def test_task_owned_submission_schemas_validate_known_witnesses() -> None:
    spec = load_spec(SPEC_PATH)
    task = _task(spec, "graph-independent-set-01")
    schema = _submission_schema(task)

    Draft202012Validator.check_schema(schema)
    assert Draft202012Validator(schema).is_valid(
        {
            "task_id": task.task_id,
            "answer": {"vertices": ["c", "e", "h", "j", "l"], "optimum": 5},
        }
    )
    assert not Draft202012Validator(schema).is_valid(
        {
            "task_id": task.task_id,
            "answer": {"vertices": ["c", "c"], "optimum": 2},
        }
    )


def test_study_contract_is_closed_and_requires_frozen_kind_order() -> None:
    payload = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TrajectoryValueStudySpec.model_validate(payload)

    payload.pop("unexpected")
    payload["tasks"] = list(reversed(payload["tasks"]))
    with pytest.raises(ValidationError, match="four frozen kinds in order"):
        TrajectoryValueStudySpec.model_validate(payload)


def test_codex_command_binds_exact_model_isolation_and_required_server(
    tmp_path: Path,
) -> None:
    spec = load_spec(SPEC_PATH)
    arguments = _codex_arguments(
        workspace=tmp_path,
        spec=spec,
        mcp_url="http://127.0.0.1:8765/mcp",
        prompt="frozen prompt",
    )

    assert arguments[arguments.index("-m") + 1] == "gpt-5.4-mini"
    assert 'model_reasoning_effort="medium"' in arguments
    assert 'mcp_servers.jacobian.url="http://127.0.0.1:8765/mcp"' in arguments
    assert "--ephemeral" in arguments
    assert "--ignore-user-config" in arguments
    assert "--ignore-rules" in arguments
    assert arguments[arguments.index("-s") + 1] == "workspace-write"
    assert arguments[-1] == "frozen prompt"


def test_clean_room_verifier_has_only_standard_library_imports() -> None:
    path = ROOT / "benchmarks/tooling/trajectory_value_study_verifier.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert roots <= {
        "__future__",
        "fractions",
        "hashlib",
        "itertools",
        "json",
        "math",
        "pathlib",
        "re",
        "stat",
        "typing",
    }


def test_model_execution_is_explicitly_opt_in(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="without --execute"):
        run_study(SPEC_PATH, tmp_path / "results", execute=False)
