from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "metamath-syllogism-repair"


def _tamper(app: Path, submission: dict, mutation: str) -> None:
    actions = {
        "proof": lambda: submission["result"]["repaired_proof"].__setitem__(6, "a1i"),
        "positions": lambda: submission["result"].__setitem__(
            "changed_positions", [5, 9]
        ),
        "trace": lambda: submission["result"]["trace"][6].__setitem__(
            "stack_depth", submission["result"]["trace"][6]["stack_depth"] + 1
        ),
        "substitution": lambda: submission["result"]["trace"][9][
            "substitution"
        ].__setitem__("u", ["u"]),
        "target": lambda: submission["result"]["final_expression"].__setitem__(-2, "v"),
        "legacy_field": lambda: submission.__setitem__("legacy_metadata", True),
        "extra_field": lambda: submission["result"].__setitem__("unexpected", True),
    }
    actions[mutation]()
    _fixtures._write_json(app / "submission.json", submission)


def test_metamath_repair_accepts_oracle(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_metamath_repair_accepts_unordered_positions(
    tmp_path: Path,
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["changed_positions"] = [9, 6]
    _fixtures._write_json(app / "submission.json", submission)

    accepted = _verifier._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutation",
    [
        "proof",
        "positions",
        "trace",
        "substitution",
        "target",
        "legacy_field",
        "extra_field",
    ],
)
def test_metamath_repair_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _tamper(app, submission, mutation)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_metamath_repair_rejects_frozen_input_tamper(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["target"][-2] = "v"
    _fixtures._write_json(app / "input.json", frozen)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
