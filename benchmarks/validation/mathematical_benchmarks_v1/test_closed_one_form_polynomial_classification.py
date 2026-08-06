from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VERIFIER = (
    ROOT
    / "benchmarks/datasets/mathematical-benchmarks-v1/closed-one-form-polynomial-classification/tests/verifier.py"
)


def _module():
    sys.path.insert(0, str(VERIFIER.parent))
    spec = importlib.util.spec_from_file_location("closed_form_verifier", VERIFIER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correct_constraint_rank_and_wrong_published_rank() -> None:
    v = _module()
    assert v.rank(v.TARGET) == 2
    wrong = [[0, -1, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 1, 0, 0, 0, 0]]
    assert v.rank(v.TARGET + wrong) > 2


def test_derivative_replays_potential() -> None:
    v = _module()
    terms = [
        {"coefficient": "1", "x_power": 2, "y_power": 1},
        {"coefficient": "1", "x_power": 1, "y_power": 2},
    ]
    assert v.derivative(terms, 0) == {(1, 1): Fraction(2), (0, 2): Fraction(1)}
    assert v.derivative(terms, 1) == {(2, 0): Fraction(1), (1, 1): Fraction(2)}


def test_dependent_basis_is_detected() -> None:
    v = _module()
    basis = [[1 if i == j else 0 for i in range(10)] for j in range(7)]
    basis.append(basis[0])
    assert v.rank(basis) == 7


def test_verifier_rejects_short_and_surplus_basis_rows(tmp_path: Path) -> None:
    from benchmarks.validation.mathematical_benchmarks_v1 import support

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "short-basis"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["basis"] = [
        [1 if i == j else 0 for i in range(8)] for j in range(8)
    ]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "surplus-basis"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["basis"].append(submission["result"]["basis"][0])
    submission["result"]["potentials"].append(submission["result"]["potentials"][0])
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_verifier_rejects_boolean_matrix_entries(tmp_path: Path) -> None:
    from benchmarks.validation.mathematical_benchmarks_v1 import support

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "boolean-entry"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["constraints"][0][4] = True
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_verifier_checks_derivation_evidence_and_scope(tmp_path: Path) -> None:
    from benchmarks.validation.mathematical_benchmarks_v1 import support

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "empty-evidence"
    )
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence" / "answer.txt"
    evidence.write_text("unrelated prose\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["evidence_validity"] == 0.0
    assert rejected["reward"] == 0.0

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "bad-scope"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["scope"] = "not a closed polynomial one-form; degree at most three on R2"
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected["scope_accuracy"] == 0.0
    assert rejected["reward"] == 0.0


def test_verifier_separates_math_from_envelope_and_limitation(tmp_path: Path) -> None:
    from benchmarks.validation.mathematical_benchmarks_v1 import support

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "mismatched-claim"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["conclusion"] = "UNSUPPORTED"
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 0.0
    assert result["reward"] == 0.0

    task, app, logs = support._prepare_case(
        tmp_path, "closed-one-form-polynomial-classification", "bad-limitation"
    )
    submission = json.loads((app / "submission.json").read_text())
    submission["limitations"] = ["No limitations."]
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["correctness"] == 1.0
    assert result["assurance_calibration"] == 0.0
    assert result["reward"] == 0.0
