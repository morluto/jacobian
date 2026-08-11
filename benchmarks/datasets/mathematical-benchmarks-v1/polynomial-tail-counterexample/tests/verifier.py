import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    normalize_reward_file,
    resolve_evidence,
)
from verifier_support import load_submission as load_strict_submission

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


def parse_fraction(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def evaluate(coefficients, x):
    value = Fraction(0)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def witness_ok(result):
    keys = {"p_coefficients", "q_coefficients", "p_roots", "q_roots", "x1", "x2"}
    if not isinstance(result, dict) or set(result) != keys:
        return False
    if not all(
        isinstance(result[key], list)
        for key in ("p_coefficients", "q_coefficients", "p_roots", "q_roots")
    ):
        return False
    try:
        p = [parse_fraction(x) for x in result["p_coefficients"]]
        q = [parse_fraction(x) for x in result["q_coefficients"]]
        proots = [parse_fraction(x) for x in result["p_roots"]]
        qroots = [parse_fraction(x) for x in result["q_roots"]]
        x1, x2 = parse_fraction(result["x1"]), parse_fraction(result["x2"])
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if len(p) < 3 or len(q) < 2 or not p[0] or not q[0]:
        return False
    if not (len(p) - 1 > len(q) - 1 > 1 and p[0] > q[0] > 0):
        return False
    if len(proots) != len(p) - 1 or len(qroots) != len(q) - 1:
        return False
    if len(set(proots)) != len(proots) or len(set(qroots)) != len(qroots):
        return False
    if any(evaluate(p, root) for root in proots) or any(
        evaluate(q, root) for root in qroots
    ):
        return False
    largest_root = max(proots + qroots)
    if not (largest_root <= x1 and Fraction(0) <= x1 < x2):
        return False
    return evaluate(p, x1) - evaluate(q, x1) >= evaluate(p, x2) - evaluate(q, x2)


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
