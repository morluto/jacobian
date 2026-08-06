import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
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
    levels = result.get("levels")
    exact_integer_levels = bool(
        isinstance(levels, list)
        and len(levels) == 8
        and all(
            isinstance(row, dict)
            and all(
                type(row.get(field)) is int
                for field in (
                    "level",
                    "included_endpoint",
                    "excluded_endpoint",
                    "cumulative_count",
                )
            )
            for row in levels
        )
    )
    return (
        exact_integer_levels
        and result.get("levels") == expected
        and q(result.get("lower_density")) == Fraction(1, b + 1)
        and q(result.get("upper_density")) == Fraction(b, b + 1)
        and result.get("lower_density_positive") is True
        and result.get("natural_density_exists") is False
        and result.get("semantic_relation") == "FORMALIZED_PREDICATE_STRICTLY_STRONGER"
    )


def evidence_ok(evidence):
    # The structured endpoint certificate is replayed independently.  Keep the
    # text artifact requirement to the public digest-binding contract.
    return evidence_list_is_bound(evidence)


def main():
    e = json.loads((T / "expected.json").read_text())
    input_binding = workspace_input_is_bound()
    s = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    r = s.get("result") if isinstance(s, dict) else None
    math_ok = valid_result(r)
    ev = bool(isinstance(s, dict) and evidence_ok(s.get("evidence")))
    scope = bool(
        isinstance(s, dict)
        and s.get("scope")
        == "parameterized-geometric-block-family-with-eight-replayed-levels"
        and s.get("completeness") == "COMPLETE"
        and s.get("limitations") == [LIMITATION]
    )
    assurance = bool(isinstance(s, dict) and s.get("claimed_assurance") == "COMPUTED")
    false = false_verified_claim(s, verification_record_bound=False)
    correct = bool(
        contract and input_binding and math_ok and ev and scope and not false
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
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
