import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/necklace-burnside-certificate"
)
TASK_NAME = "necklace-burnside-certificate"

LIMITATIONS = ["FINITE_LENGTH_16_INSTANCE", "NO_GENERAL_ENUMERATION_THEOREM"]


def load_verifier():
    saved_path = sys.path[:]
    saved_modules = dict(sys.modules)
    try:
        sys.path.insert(0, str(TASK / "tests"))
        spec = importlib.util.spec_from_file_location(
            "necklace_verifier", TASK / "tests/verifier.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


def _digest(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / TASK_NAME / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK / "solution" / "submission.json").read_text())
    for descriptor in submission["witness"]:
        evidence_path = Path(descriptor["path"])
        destination = app / evidence_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        fixture = TASK / "solution" / evidence_path.name
        if fixture.is_file():
            shutil.copy2(fixture, destination)
        descriptor["sha256"] = _digest(destination)
    _write_json(app / "submission.json", submission)
    return TASK, app, logs


def _evidence_object(result: dict) -> dict:
    return {
        "schema_version": "1",
        "task_id": "jacobian/necklace-burnside-certificate",
        "result": result,
    }


def _rewrite(app: Path, submission: dict, result: dict | None = None) -> None:
    """Rebind the JSON witness object and submission digest in concert."""
    bound_result = result if result is not None else submission["result"]
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(
        json.dumps(
            _evidence_object(bound_result), sort_keys=True, separators=(",", ":")
        )
    )
    submission["witness"][0]["sha256"] = _digest(evidence_path)
    _write_json(app / "submission.json", submission)


def test_independent_orbit_derivation():
    result = load_verifier().derive()
    assert result["valid_labelled_words"] == 2206
    assert result["burnside_numerator"] == 2816
    assert result["orbit_count"] == 88
    assert len(result["canonical_representatives"]) == 88


def test_wraparound_and_reflection_are_material():
    verifier = load_verifier()
    assert not verifier.valid(tuple(map(int, "0010101010101010")))
    result = verifier.derive()
    assert result["reflection_fixed_counts"] == [42, 26] * 8


def test_corrupt_fixed_count_or_orbit_representative_is_rejected():
    verifier = load_verifier()
    result = verifier.derive()
    result["rotation_fixed_counts"][0] -= 1
    assert not verifier.matches(result)
    result = verifier.derive()
    result["canonical_representatives"].pop()
    assert not verifier.matches(result)


def test_canonical_solution_receives_full_reward(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0
    assert result.reward == 1.0


def test_rejects_boolean_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [bool(x) for x in result["rotation_fixed_counts"]]
    submission["result"] = result
    _rewrite(app, submission, result)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_rejects_float_fixed_counts(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    result = dict(submission["result"])
    result["rotation_fixed_counts"] = [
        float(x) for x in result["rotation_fixed_counts"]
    ]
    submission["result"] = result
    _rewrite(app, submission, result)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_witness_result_must_match_submission_result(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    corrupted["orbit_count"] = 999
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)

    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_extra_envelope_field_is_rejected(tmp_path: Path) -> None:
    """An extra envelope field breaks the schema-validated submission;
    correctness and witness_validity fall to zero."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["extra_field"] = "unexpected"
    _write_json(app / "submission.json", submission)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_float_in_result_is_rejected(tmp_path: Path) -> None:
    """A float where an integer is required is a schema violation."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    corrupted["orbit_count"] = 88.0
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


@pytest.mark.parametrize(
    "corruption",
    (
        "empty",
        "duplicate",
        "non_binary",
    ),
)
def test_representative_schema_violations_are_rejected(
    tmp_path: Path, corruption: str
) -> None:
    """The result-shape check enforces the advertised representative schema:
    at least one entry, all unique, and every entry matching ^[01]{16}$."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    corrupted = dict(submission["result"])
    reps = list(corrupted["canonical_representatives"])
    if corruption == "empty":
        corrupted["canonical_representatives"] = []
    elif corruption == "duplicate":
        corrupted["canonical_representatives"] = [reps[0], reps[0]]
    else:
        corrupted["canonical_representatives"] = ["x" * 16]
    submission["result"] = corrupted
    _rewrite(app, submission, corrupted)
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 0.0
    assert result.reward == 0.0


def test_input_tamper_gates_reward(tmp_path: Path) -> None:
    """When the workspace input is altered, mathematical correctness remains
    independently evaluated (the result is still canonical) while the
    witness binding is gated to zero by the failed input binding."""
    task, app, logs = _case(tmp_path)
    (app / "input.json").write_text("{}")
    result = _run_verifier(task, app, logs)
    assert result.details["correctness"] == 1.0
    assert result.reward == 0.0
    assert result.reward == 0.0


def test_oversized_witness_with_valid_digest_is_accepted(tmp_path: Path) -> None:
    """No task-local witness-size cap: a large but well-formed and
    digest-bound witness object with the correct result still receives
    full witness_validity and reward. The JSON is padded with whitespace
    so the parsed object still binds the result."""
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    obj = _evidence_object(submission["result"])
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    padded = payload.replace(",", ", " + " " * (2 * 1024 * 1024 // 10))
    evidence_path.write_text(padded)
    submission["witness"][0]["sha256"] = _digest(evidence_path)
    _write_json(app / "submission.json", submission)
    result = _run_verifier(task, app, logs)
    assert result.reward == 1.0
    assert result.details["correctness"] == 1.0
    assert result.reward == 1.0
