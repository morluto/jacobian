from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

from benchmarks.validation._verifier_child import run_verifier_in_child
from benchmarks.validation.mathematical_benchmarks_v1._verifier import _run_verifier
from jsonschema import Draft202012Validator

TASK = "cubic-image-classification"
TASK_PATH = Path(__file__).resolve().parents[3] / (
    "benchmarks/datasets/mathematical-benchmarks-v1/cubic-image-classification"
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _case(tmp_path: Path):
    root = tmp_path / TASK / "computed"
    app = root / "app"
    logs = root / "logs"
    app.mkdir(parents=True)
    logs.mkdir(parents=True)
    shutil.copy2(TASK_PATH / "environment" / "input.json", app / "input.json")
    submission = json.loads((TASK_PATH / "solution" / "submission.json").read_text())
    _write_json(app / "submission.json", submission)
    return TASK_PATH, app, logs


def _rewrite(app: Path, submission: dict) -> None:
    _write_json(app / "submission.json", submission)


def test_accepts_reordered_covered_residues(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["families"][0]["covered_residues"].reverse()
    submission["result"]["image_residues_mod_9"].reverse()
    submission["result"]["excluded_residues_mod_9"].reverse()
    _rewrite(app, submission)

    accepted = _run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_affine_wire_shape_requires_exactly_two_components(tmp_path: Path) -> None:
    for affine in [[], [1], [1, 0, 2]]:
        task, app, logs = _case(tmp_path / str(len(affine)))
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        submission["result"]["families"][0]["A"] = affine
        schema = json.loads((task / "environment/submission_schema.json").read_text())
        assert not Draft202012Validator(schema).is_valid(submission)
        path.write_text(json.dumps(submission))
        assert run_verifier_in_child(task=task, app=app, logs=logs).reward == 0.0


def test_rejects_duplicate_covered_residue(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["families"][0]["covered_residues"] = [1, 1, 4, 7]
    _rewrite(app, submission)

    rejected = _run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_family_envelope_accepts_alternate_constructions_and_rejects_overflow(
    tmp_path: Path,
) -> None:
    for count, accepted in [(18, True), (19, False)]:
        task, app, logs = _case(tmp_path / str(count))
        path = app / "submission.json"
        submission = json.loads(path.read_text())
        families = submission["result"]["families"]
        families.extend(
            deepcopy(families[i % len(families)]) for i in range(count - len(families))
        )
        # The form is symmetric in A and C; all affine identities survive.
        for family in families:
            family["A"], family["C"] = family["C"], family["A"]
        schema = json.loads((task / "environment/submission_schema.json").read_text())
        assert Draft202012Validator(schema).is_valid(submission) is accepted
        path.write_text(json.dumps(submission))
        assert run_verifier_in_child(task=task, app=app, logs=logs).reward == float(
            accepted
        )
