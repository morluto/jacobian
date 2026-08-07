import itertools
import json
from pathlib import Path

from verifier_support import evidence_list_is_bound, resolve_evidence
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


def determinant(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def witness_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "generators",
        "dependence_coefficients",
    }:
        return False
    vectors = result["generators"]
    coefficients = result["dependence_coefficients"]
    if (
        not isinstance(vectors, list)
        or len(vectors) != 4
        or any(not isinstance(v, list) or len(v) != 3 for v in vectors)
        or any(type(x) is not int for v in vectors for x in v)
        or any(all(x == 0 for x in v) for v in vectors)
        or not isinstance(coefficients, list)
        or len(coefficients) != 4
        or any(type(x) is not int or x == 0 for x in coefficients)
    ):
        return False
    if any(
        determinant(*(vectors[i] for i in triple)) == 0
        for triple in itertools.combinations(range(4), 3)
    ):
        return False
    return all(
        sum(coefficients[i] * vectors[i][j] for i in range(4)) == 0 for j in range(3)
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


if __name__ == "__main__":
    main()
