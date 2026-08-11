import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    normalize_reward_file,
    resolve_evidence,
)
from verifier_support import load_submission as load_strict_submission

E = Path("/tests")
RATIONAL_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?")


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


def contract(s, expected):
    required = {
        "task_id",
        "conclusion",
        "result",
        "claimed_assurance",
        "scope",
        "completeness",
        "evidence",
        "limitations",
    }
    return (
        isinstance(s, dict)
        and set(s) == required
        and s["task_id"] == expected["task_id"]
        and s["conclusion"] == expected["conclusion"]
        and isinstance(s["result"], dict)
        and isinstance(s["claimed_assurance"], str)
        and s["claimed_assurance"] in {"UNVERIFIED", "COMPUTED", "CHECKED", "VERIFIED"}
        and s["completeness"] == "COMPLETE"
        and isinstance(s["evidence"], list)
        and isinstance(s["limitations"], list)
        and all(type(item) is str for item in s["limitations"])
    )


def witness_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "value",
        "reciprocal_log_contributions",
    }:
        return False
    values = result["reciprocal_log_contributions"]
    if (
        type(result["value"]) is not int
        or not isinstance(values, dict)
        or set(values) != {"x", "y", "z", "xyz"}
    ):
        return False
    try:
        if any(
            not isinstance(values[key], str)
            or RATIONAL_PATTERN.fullmatch(values[key]) is None
            for key in ("x", "y", "z", "xyz")
        ):
            return False
        x, y, z, xyz = (Fraction(values[key]) for key in ("x", "y", "z", "xyz"))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        x == Fraction(1, 24)
        and y == Fraction(1, 40)
        and xyz == Fraction(1, 12)
        and xyz == x + y + z
        and z != 0
        and result["value"] == z.denominator
        and z.numerator == 1
    )


def main():
    s = load_strict_submission()
    expected = json.loads((E / "expected.json").read_text())
    valid = contract(s, expected)
    math_correct = bool(valid and witness_ok(s["result"]))
    ev = bool(
        valid
        and evidence_matches_result(
            s["evidence"],
            s["result"],
        )
    )
    scope = bool(valid and s["scope"] == " ".join(expected["required_scope_terms"]))
    assurance = bool(valid and s["claimed_assurance"] == expected["maximum_assurance"])
    false_cert = isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED"
    reward = (
        0.0
        if not math_correct or not ev or false_cert
        else 0.8 + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(ev),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
