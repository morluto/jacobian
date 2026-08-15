from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import (
    _fixtures,
    _verifier,
)

TASK = "metamath-syllogism-repair"


def _bind(app: Path, submission: dict) -> None:
    _fixtures._bind_result_evidence(app, submission)
    _fixtures._write_json(app / "submission.json", submission)


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
        "witness": lambda: submission["witness"][0].__setitem__(
            "sha256", "sha256:" + "0" * 64
        ),
        "extra_field": lambda: submission["result"].__setitem__("unexpected", True),
    }
    actions[mutation]()
    if mutation == "witness":
        _fixtures._write_json(app / "submission.json", submission)
    else:
        _bind(app, submission)


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
    _bind(app, submission)

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
        "witness",
        "extra_field",
    ],
)
def test_metamath_repair_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _tamper(app, submission, mutation)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


@pytest.mark.parametrize(
    "witness_text",
    [
        lambda result: (
            "A proof explanation.\n"
            + "RESULT_JSON: "
            + json.dumps(result, sort_keys=True, separators=(",", ":"))
            + "\nRESULT_JSON: {}\n"
        ),
        lambda _result: "RESULT_JSON: {}\n",
    ],
)
def test_metamath_repair_rejects_malformed_witness(
    tmp_path: Path, witness_text: Callable[[dict], str]
) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(witness_text(submission["result"]))
    submission["witness"][0]["sha256"] = _fixtures._digest(evidence_path)
    _fixtures._write_json(app / "submission.json", submission)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.reward == 0.0
    assert rejected.reward == 0.0


def test_metamath_repair_rejects_symlinked_witness(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    real_path = app / "other.txt"
    real_path.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    evidence_path.symlink_to(real_path)

    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_metamath_repair_rejects_frozen_input_tamper(tmp_path: Path) -> None:
    task, app, logs = _fixtures._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["target"][-2] = "v"
    _fixtures._write_json(app / "input.json", frozen)
    rejected = _verifier._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
