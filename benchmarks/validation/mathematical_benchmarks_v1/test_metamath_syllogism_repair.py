from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "metamath-syllogism-repair"


def _bind(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


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
        "assurance": lambda: submission.__setitem__("claimed_assurance", "VERIFIED"),
        "evidence": lambda: submission["evidence"][0].__setitem__(
            "sha256", "sha256:" + "0" * 64
        ),
        "limitations": lambda: submission.__setitem__("limitations", ["unbounded"]),
        "extra_field": lambda: submission["result"].__setitem__("unexpected", True),
    }
    actions[mutation]()
    if mutation != "evidence":
        _bind(app, submission)
    else:
        support._write_json(app / "submission.json", submission)


def test_metamath_repair_accepts_oracle(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_metamath_repair_accepts_unverified_and_unordered_positions(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "UNVERIFIED"
    submission["result"]["changed_positions"] = [9, 6]
    _bind(app, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted.reward == pytest.approx(1.0)


def test_metamath_repair_keeps_diagnostics_for_unsupported_assurance(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    submission["claimed_assurance"] = "VERIFIED"
    _bind(app, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 1.0
    assert rejected.details["scope_accuracy"] == 1.0
    assert rejected.details["assurance_calibration"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize(
    "mutation",
    [
        "proof",
        "positions",
        "trace",
        "substitution",
        "target",
        "assurance",
        "evidence",
        "limitations",
        "extra_field",
    ],
)
def test_metamath_repair_rejects_tampering(tmp_path: Path, mutation: str) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    _tamper(app, submission, mutation)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


@pytest.mark.parametrize(
    "evidence_text",
    [
        lambda result: (
            "RESULT_JSON: "
            + json.dumps(result, sort_keys=True, separators=(",", ":"))
            + "\n"
        ),
        lambda result: (
            "A proof explanation.\n"
            + "RESULT_JSON: "
            + json.dumps(result, sort_keys=True, separators=(",", ":"))
            + "\nRESULT_JSON: {}\n"
        ),
        lambda result: (
            "This unrelated text proves nothing.\nRESULT_JSON: "
            + json.dumps(result, sort_keys=True, separators=(",", ":"))
            + "\n"
        ),
    ],
)
def test_metamath_repair_rejects_weak_evidence(
    tmp_path: Path, evidence_text: Callable[[dict], str]
) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text(evidence_text(submission["result"]))
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_metamath_repair_rejects_symlinked_evidence(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    evidence_path = app / "evidence" / "answer.txt"
    real_path = app / "other.txt"
    real_path.write_bytes(evidence_path.read_bytes())
    evidence_path.unlink()
    evidence_path.symlink_to(real_path)

    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_metamath_repair_rejects_frozen_input_tamper(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    frozen = json.loads((app / "input.json").read_text())
    frozen["target"][-2] = "v"
    support._write_json(app / "input.json", frozen)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0
