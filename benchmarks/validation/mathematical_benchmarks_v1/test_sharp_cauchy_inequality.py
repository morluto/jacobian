from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "sharp-cauchy-inequality"


def test_sharp_cauchy_inequality_accepts_symbolic_certificate(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    (app / "evidence" / "inequality-certificate.json").write_bytes(
        (task / "solution" / "inequality-certificate.json").read_bytes()
    )
    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_sharp_cauchy_inequality_rejects_corrupted_residual(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    evidence_path = app / "evidence" / "inequality-certificate.json"
    evidence = json.loads(
        (task / "solution" / "inequality-certificate.json").read_text()
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["certificate"]["residual"][0]["coefficient"] = 2
    evidence["result"] = submission["result"]
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_sharp_cauchy_inequality_rejects_nonsharp_constant(tmp_path: Path) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    evidence_path = app / "evidence" / "inequality-certificate.json"
    evidence = json.loads(
        (task / "solution" / "inequality-certificate.json").read_text()
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["constant"] = {"numerator": 19, "denominator": 10}
    evidence["result"] = submission["result"]
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["correctness"] == 0.0
    assert rejected["reward"] == 0.0


def test_sharp_cauchy_inequality_accepts_cauchy_composition_mode(
    tmp_path: Path,
) -> None:
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(task / "tests"))
    try:
        verifier = runpy.run_path(str(task / "tests" / "verifier.py"))
    finally:
        sys.path.remove(str(task / "tests"))
        sys.modules.pop("verifier_support", None)
        sys.dont_write_bytecode = original_dont_write_bytecode
    expected = verifier["_expected"]()

    def encode(poly):
        return [
            {"exponents": list(monomial), "coefficient": coefficient}
            for monomial, coefficient in sorted(poly.items())
        ]

    keys = (
        "d",
        "a2",
        "x2",
        "across",
        "xcross",
        "gram_residual",
        "gram_sos",
        "total_a_square",
        "total_x_square",
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["proof_mode"] = "CS_COMPOSITION"
    submission["result"]["certificate"] = {key: encode(expected[key]) for key in keys}
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "inequality-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_sharp_cauchy_inequality_rejects_checked_assurance_above_ceiling(
    tmp_path: Path,
) -> None:
    """CHECKED is above the COMPUTED ceiling and must force reward to zero."""
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    (app / "evidence" / "inequality-certificate.json").write_bytes(
        (task / "solution" / "inequality-certificate.json").read_bytes()
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["claimed_assurance"] = "CHECKED"
    evidence_path = app / "evidence" / "inequality-certificate.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["result"] = submission["result"]
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    rejected = support._run_verifier(task, app, logs)
    assert rejected["reward"] == 0.0


def test_sharp_cauchy_inequality_accepts_unreduced_rationals(tmp_path: Path) -> None:
    """Schema-valid unreduced rationals in the witness must receive full
    reward because Fraction normalizes equivalent representations."""
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    witness = submission["result"]["equality_witness"]
    for key in ("a", "b", "c", "x", "y", "z"):
        n = witness[key]["numerator"]
        d = witness[key]["denominator"]
        witness[key] = {"numerator": n * 2, "denominator": d * 2}
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "inequality-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)


def test_sharp_cauchy_inequality_accepts_direct_sos_mode(tmp_path: Path) -> None:
    """A strategy-neutral DIRECT_SOS certificate with any valid sum-of-squares
    decomposition must be accepted."""
    task, app, logs = support._prepare_case(
        tmp_path, "sharp-cauchy-inequality", "computed"
    )
    original_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(task / "tests"))
    try:
        verifier = runpy.run_path(str(task / "tests" / "verifier.py"))
    finally:
        sys.path.remove(str(task / "tests"))
        sys.modules.pop("verifier_support", None)
        sys.dont_write_bytecode = original_dont_write_bytecode
    expected = verifier["_expected"]()

    def encode(poly):
        return [
            {"exponents": list(monomial), "coefficient": coefficient}
            for monomial, coefficient in sorted(poly.items())
        ]

    a, b, c, x, y, z = (verifier["_var"](i) for i in range(6))
    factor_polys = [
        verifier["_add"]((1, a), (-1, x)),
        verifier["_add"]((1, b), (-1, y)),
        verifier["_add"]((1, c), (-1, z)),
    ]
    submission_path = app / "submission.json"
    submission = json.loads(submission_path.read_text())
    submission["result"]["proof_mode"] = "DIRECT_SOS"
    submission["result"]["certificate"] = {
        "d": encode(expected["d"]),
        "u": encode(expected["u"]),
        "v": encode(expected["v"]),
        "residual": encode(expected["residual"]),
        "constraint_residual": encode(expected["constraint_residual"]),
        "sos_factors": [encode(f) for f in factor_polys],
    }
    evidence = {
        "schema_version": "1",
        "task_id": submission["task_id"],
        "result": submission["result"],
        "limitations": submission["limitations"],
    }
    evidence_path = app / "evidence" / "inequality-certificate.json"
    support._write_json(evidence_path, evidence)
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(submission_path, submission)

    accepted = support._run_verifier(task, app, logs)
    assert accepted["correctness"] == 1.0
    assert accepted["reward"] == pytest.approx(1.0)
