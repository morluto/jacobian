import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATION = "Eight exact levels replay the general formula but do not machine-prove the infinite limit or the Erdős problem."


def q(text):
    if (
        not isinstance(text, str)
        or re.fullmatch(r"(?:0|1|[1-9][0-9]*/[1-9][0-9]*)", text) is None
    ):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "base",
        "family",
        "count_formula",
        "levels",
        "lower_density",
        "upper_density",
        "lower_density_positive",
        "natural_density_exists",
        "semantic_relation",
    }:
        return False
    b = result.get("base")
    if (
        type(b) is not int
        or b not in range(2, 10)
        or result.get("family") != "ALTERNATING_GEOMETRIC_BLOCKS"
        or result.get("count_formula") != "(b^(2m+2)-1)/(b+1)"
    ):
        return False
    expected = []
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        expected.append(
            {
                "level": m,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": str(Fraction(count, high)),
                "excluded_density": str(Fraction(count, low)),
            }
        )
    return (
        result.get("levels") == expected
        and q(result.get("lower_density")) == Fraction(1, b + 1)
        and q(result.get("upper_density")) == Fraction(b, b + 1)
        and result.get("lower_density_positive") is True
        and result.get("natural_density_exists") is False
        and result.get("semantic_relation") == "FORMALIZED_PREDICATE_STRICTLY_STRONGER"
    )


def evidence_ok(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    path = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text()
        lines = [x for x in text.splitlines() if x.startswith("RESULT_JSON:")]
        return (
            len(lines) == 1
            and json.loads(lines[0].removeprefix("RESULT_JSON:").strip()) == result
            and all(
                w in text.lower()
                for w in ("lower density", "natural density", "subsequence", "finite")
            )
        )
    except (OSError, UnicodeError, ValueError):
        return False


def main():
    e = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    r = s.get("result") if isinstance(s, dict) else None
    math_ok = bool(workspace_input_is_bound() and valid_result(r))
    ev = bool(isinstance(s, dict) and evidence_ok(s.get("evidence"), r))
    scope = bool(
        isinstance(s, dict)
        and s.get("scope")
        == "parameterized-geometric-block-family-with-eight-replayed-levels"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == [LIMITATION]
    )
    assurance = bool(isinstance(s, dict) and s.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(s, verification_record_bound=False)
    correct = bool(contract and math_ok and ev and scope and not false)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "correctness": float(math_ok),
                "evidence_validity": float(ev),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
