from __future__ import annotations

import json
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "subset-incidence-determinant"


def test_rejects_boolean_diagonal_weights(tmp_path: Path) -> None:
    """Boolean ``true`` must not spoof integer 1 in diagonal_weights."""
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["diagonal_weights"] = [
        True if w == 1 else w for w in submission["result"]["diagonal_weights"]
    ]
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_mask_order(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["mask_order"][0] = True
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_trace_fields(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["trace"][0]["n"] = True
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_boolean_sample_n(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["sample_n"] = True
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_accepts_reversed_same_cardinality_mask_order(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "reversed-masks")
    submission = json.loads((app / "submission.json").read_text())
    grouped: dict[int, list[tuple[int, int]]] = {}
    for mask, weight in zip(
        submission["result"]["mask_order"],
        submission["result"]["diagonal_weights"],
        strict=True,
    ):
        grouped.setdefault(mask.bit_count(), []).append((mask, weight))
    order: list[int] = []
    weights: list[int] = []
    for cardinality in sorted(grouped):
        for mask, weight in reversed(grouped[cardinality]):
            order.append(mask)
            weights.append(weight)
    submission["result"]["mask_order"] = order
    submission["result"]["diagonal_weights"] = weights
    _fixtures._write_json(app / "submission.json", submission)
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_accepts_typed_general_formulas(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == 1.0


def test_rejects_wrong_general_determinant_formula(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["general_determinant"]["otherwise"] = 1
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
