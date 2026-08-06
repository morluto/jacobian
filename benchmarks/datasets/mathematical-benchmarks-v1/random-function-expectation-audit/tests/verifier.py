import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def q(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]{1,40}|/[1-9][0-9]*)?", value)
        is None
    ):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return parsed


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    math_contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        verification_record="optional",
    )

    n = x["domain_size"]
    self_hit = Fraction(2 * n - 1, n * n)
    other_hit = Fraction(n - 1, n * n)
    squared_sum = sum(
        (target - source) ** 2
        for source in range(1, n + 1)
        for target in range(1, n + 1)
    )
    expectation = other_hit * squared_sum
    valid = (
        math_contract
        and set(r)
        == {
            "self_hit_probability",
            "other_hit_probability",
            "ordered_squared_difference_sum",
            "expected_value",
        }
        and type(r.get("ordered_squared_difference_sum")) is int
        and q(r.get("self_hit_probability")) == self_hit
        and q(r.get("other_hit_probability")) == other_hit
        and r.get("ordered_squared_difference_sum") == squared_sum
        and q(r.get("expected_value")) == expectation
        and expectation != 2025
    )
    math_correct = bool(valid)
    correct = bool(contract and math_correct)
    good = bool(math_contract and evidence_matches_result(s["evidence"], r))
    scope = bool(math_contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(math_contract and s["claimed_assurance"] == e["maximum_assurance"])
    false = bool(isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED")
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
