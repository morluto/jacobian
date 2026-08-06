from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "marginal-joint-product-audit"


def _run(tmp_path: Path, mutate=None):
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    if mutate is not None:
        mutate(submission)
        support._bind_result_evidence(app, submission)
        support._write_json(app / "submission.json", submission)
    return support._run_verifier(task, app, logs)


def _product(entries):
    masses = defaultdict(Fraction)
    for entry in entries:
        masses[entry["x"] * entry["y"]] += Fraction(entry["mass"])
    return [
        {"value": value, "mass": str(mass)}
        for value, mass in sorted(masses.items())
        if mass
    ]


def test_oracle_passes(tmp_path: Path) -> None:
    assert _run(tmp_path)["reward"] == 1.0


def test_plain_digest_bound_evidence_needs_no_private_marker(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    submission = json.loads((app / "submission.json").read_text())
    evidence = app / "evidence/answer.txt"
    evidence.write_text("Finite coupling certificate supplied in submission.json.\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence)
    support._write_json(app / "submission.json", submission)
    result = support._run_verifier(task, app, logs)
    assert result["evidence_validity"] == 1.0
    assert result["reward"] == 1.0


def test_visible_input_tamper_preserves_math_diagnostic(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(tmp_path, TASK, "computed")
    (app / "input.json").write_text("{}")
    result = support._run_verifier(task, app, logs)
    assert result["input_binding"] == 0.0
    assert result["correctness"] == 1.0
    assert result["reward"] == 0.0


def test_zero_mass_product_entries_are_normalized(tmp_path: Path) -> None:
    def mutate(submission):
        entries = submission["result"]["limit_product_distribution"]
        present = {entry["value"] for entry in entries}
        attainable = {
            left * right for left in (-3, -1, 2, 5) for right in (-3, -1, 2, 5)
        }
        missing = min(attainable - present)
        entries.append({"value": missing, "mass": "0"})
        entries.sort(key=lambda entry: entry["value"])

    assert _run(tmp_path, mutate)["reward"] == 1.0


def test_accepts_alternative_nonproduct_coupling(tmp_path: Path) -> None:
    def mutate(submission):
        entries = json.loads(json.dumps(submission["result"]["prelimit_joint"]))
        submission["result"]["limit_joint"] = entries
        by_pair = {(entry["x"], entry["y"]): entry for entry in entries}
        by_pair[-3, -3]["mass"] = "3/200"
        by_pair[-3, -1]["mass"] = "3/200"
        by_pair[-1, -3]["mass"] = "3/200"
        by_pair[-1, -1]["mass"] = "9/200"
        submission["result"]["limit_product_distribution"] = _product(entries)
        pre = {
            entry["value"]: Fraction(entry["mass"])
            for entry in submission["result"]["prelimit_product_distribution"]
        }
        lim = {
            entry["value"]: Fraction(entry["mass"])
            for entry in submission["result"]["limit_product_distribution"]
        }
        submission["result"]["witness_product_value"] = next(
            value
            for value in set(pre) | set(lim)
            if pre.get(value, 0) != lim.get(value, 0)
        )

    assert _run(tmp_path, mutate)["reward"] == 1.0


def test_rejects_prelimit_dependence(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["prelimit_joint"][0]["mass"] = "1/50"

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_wrong_limit_marginal(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["limit_joint"][0]["mass"] = "1/5"

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_corrupted_product_pushforward(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["limit_product_distribution"][0]["mass"] = "1/10"

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_noncanonical_mass(tmp_path: Path) -> None:
    def mutate(submission):
        submission["result"]["prelimit_joint"][0]["mass"] = "2/200"

    assert _run(tmp_path, mutate)["reward"] == 0.0


def test_rejects_false_verified_claim(tmp_path: Path) -> None:
    def mutate(submission):
        submission["claimed_assurance"] = "VERIFIED"

    result = _run(tmp_path, mutate)
    assert result["false_certification"] is True
    assert result["reward"] == 0.0
