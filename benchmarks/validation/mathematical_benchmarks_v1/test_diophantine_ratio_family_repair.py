from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest
from benchmarks.validation.mathematical_benchmarks_v1 import support

TASK = "diophantine-ratio-family-repair"


def _case(tmp_path: Path):
    return support._prepare_case(tmp_path, TASK, "computed")


def _rewrite(app: Path, submission: dict) -> None:
    support._bind_result_evidence(app, submission)
    support._write_json(app / "submission.json", submission)


def _compose_with_shift(poly: list[int], shift: int = 1) -> list[int]:
    result = [0] * len(poly)
    for degree, coefficient in enumerate(poly):
        for power in range(degree + 1):
            result[power] += (
                coefficient * math.comb(degree, power) * (shift ** (degree - power))
            )
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _probe(family: dict, t: int) -> dict:
    def evaluate(poly: list[int]) -> int:
        value = 0
        for coefficient in reversed(poly):
            value = value * t + coefficient
        return value

    x = evaluate(family["x"])
    y = evaluate(family["y"])
    ratio = evaluate(family["ratio"])
    divisor = x * x - x * y + y * y
    return {
        "t": t,
        "x": x,
        "y": y,
        "divisor": divisor,
        "multiple": x * y * (x * y - 1),
        "ratio": [ratio, 1],
    }


def test_accepts_alternative_polynomial_parameterization(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    family = submission["result"]["family"]
    for name, polynomial in family.items():
        family[name] = _compose_with_shift(polynomial)
    submission["result"]["probes"] = [_probe(family, t) for t in (2, 4, 6)]
    _rewrite(app, submission)
    accepted = support._run_verifier(task, app, logs)
    assert accepted.details["correctness"] == 1.0
    assert accepted.reward == pytest.approx(1.0)


def test_rejects_ratio_not_bound_to_pair(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["family"]["ratio"] = [1]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_probe_count_above_public_maximum(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    family = submission["result"]["family"]
    submission["result"]["probes"] = [_probe(family, t) for t in range(2, 9)]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["protocol_compliance"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_polynomial_outside_public_bounds(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["family"]["a"] = [0] * 13
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_preserves_math_diagnostic_for_envelope_error(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["completeness"] = "PARTIAL"
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["protocol_compliance"] == 0.0
    assert rejected.details["correctness"] == 0.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_evidence_json_with_json_type_coercion(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    result = json.loads(json.dumps(submission["result"]))
    result["probes"][0]["x"] = True
    evidence_path.write_text("RESULT_JSON: " + json.dumps(result) + "\n")
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_deeply_nested_evidence_without_crashing(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    evidence_path.write_text("RESULT_JSON: " + ("[" * 10000) + ("]" * 10000))
    submission["evidence"][0]["sha256"] = support._digest(evidence_path)
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_non_regular_visible_input_without_blocking(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    app_input = app / "input.json"
    app_input.unlink()
    os.mkfifo(app_input)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.reward == 0.0


def test_rejects_oversized_evidence_before_hashing(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    evidence_path = app / "evidence" / "answer.txt"
    with evidence_path.open("wb") as stream:
        stream.truncate(1_048_577)
    submission["evidence"][0]["sha256"] = "sha256:" + "0" * 64
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 1.0
    assert rejected.details["evidence_validity"] == 0.0
    assert rejected.reward == 0.0


@pytest.mark.parametrize(
    ("path", "replacement"),
    [("quotient", [-1, 0, 1]), ("divisibility_quotient", [1]), ("ratio", "t")],
)
def test_rejects_corrupted_family(
    tmp_path: Path, path: str, replacement: object
) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["family"][path] = replacement
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_false_vieta_integrality(tmp_path: Path) -> None:
    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["source_audit"]["status_for_d_ge_2"] = "INTEGER"
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0


def test_rejects_empty_audit_tokens_as_protocol_failure(tmp_path: Path) -> None:
    """Empty audit strings violate minLength:1 and must fail protocol."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["result"]["source_audit"]["invalid_step"] = ""
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["protocol_compliance"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_zero_valued_probe_multiple(tmp_path: Path) -> None:
    """A probe with multiple=0 violates the schema minimum of 1."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    # Force x=y=1 at t=2 so multiple = 1*1*(1-1) = 0.
    family = submission["result"]["family"]
    family["a"] = [1]
    family["b"] = [1]
    family["d"] = [1]
    family["x"] = [1]
    family["y"] = [1]
    family["norm"] = [1]
    family["square_congruence_factor"] = [1]
    family["quotient"] = [0]
    family["divisibility_quotient"] = [0]
    family["ratio"] = [1]
    submission["result"]["probes"] = [_probe(family, 2)]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_malformed_evidence_descriptor_as_protocol_failure(
    tmp_path: Path,
) -> None:
    """A null evidence descriptor must fail protocol, not just evidence validity."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    submission["evidence"] = [None]
    support._write_json(app / "submission.json", submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["protocol_compliance"] == 0.0
    assert rejected.reward == 0.0


def test_rejects_sign_equivalent_decomposition_failing_auxiliary_identities(
    tmp_path: Path,
) -> None:
    """The sign-equivalent a=-t^2, b=-1, d=t-t^3 preserves x, y, ratio
    but fails the published auxiliary congruence identities."""

    task, app, logs = _case(tmp_path)
    submission = json.loads((app / "submission.json").read_text())
    family = submission["result"]["family"]
    # a = -t^2, b = -1, d = t - t^3
    family["a"] = [0, 0, -1]
    family["b"] = [-1]
    family["d"] = [0, 1, 0, -1]
    # x = d*a = (t - t^3)(-t^2) = t^5 - t^3
    family["x"] = [0, 0, -1, 0, 0, 1]
    # y = d*b = (t - t^3)(-1) = t^3 - t
    family["y"] = [0, -1, 0, 1]
    # norm = a^2 - a*b + b^2 = t^4 - t^2 + 1
    family["norm"] = [1, 0, -1, 0, 1]
    # ratio = x/y = (t^5 - t^3)/(t^3 - t) = t^2
    family["ratio"] = [0, 0, 1]
    # square_congruence_factor and quotient will fail the auxiliary identities
    # regardless of what we put, so set them to match the canonical values
    family["square_congruence_factor"] = [-1, 0, 1]
    family["quotient"] = [-1, 0, -1, 0, 1]
    # divisibility_quotient: x*y*(x*y-1) / (x^2 - x*y + y^2)
    # x*y = (t^5 - t^3)(t^3 - t) = t^8 - 2*t^6 + t^4
    # This is the same as the canonical family's x*y, so the divisibility
    # quotient is the same.
    family["divisibility_quotient"] = [0, 0, -1, 0, -1, 0, 1]
    submission["result"]["probes"] = [_probe(family, t) for t in (2, 3, 5)]
    _rewrite(app, submission)
    rejected = support._run_verifier(task, app, logs)
    assert rejected.details["correctness"] == 0.0
    assert rejected.reward == 0.0
