import json
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
LIMITATION = (
    "The checker validates the recurrence-difference identity and affine "
    "valuation arithmetic conditional on frozen doubling identities, but "
    "does not derive those identities or replay the Lean kernel proof."
)
Affine = tuple[int, int]


def _load_frozen_input() -> dict[str, Any]:
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _affine(value: object) -> Affine | None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        return None
    return value[0], value[1]


def _affines(value: object, count: int) -> list[Affine] | None:
    if not isinstance(value, list) or len(value) != count:
        return None
    parsed = [_affine(item) for item in value]
    return (
        None
        if any(item is None for item in parsed)
        else [item for item in parsed if item is not None]
    )


def _add(left: Affine, right: Affine) -> Affine:
    return left[0] + right[0], left[1] + right[1]


def _scale(value: Affine, scalar: int) -> Affine:
    return value[0] * scalar, value[1] * scalar


def _shift(value: Affine) -> Affine:
    """Return f(k+1) for f(k)=a*k+b."""

    return value[0], value[0] + value[1]


def _strictly_above(upper: Affine, lower: Affine) -> bool:
    """Check upper(k)>lower(k) for every integer k>=1."""

    slope = upper[0] - lower[0]
    intercept = upper[1] - lower[1]
    return slope >= 0 and slope + intercept > 0


def _base_is_valid(base: object) -> bool:
    if not isinstance(base, dict) or set(base) != {"b", "u", "P", "valuations"}:
        return False
    b = [0]
    for _ in range(2):
        b.append(2 * b[-1] ** 2 + b[-1] + 1)
    u = [2 * item for item in b]
    p = [1]
    for index in range(2):
        p.append(p[-1] * (2 * u[index] + 1))
    valuations = base["valuations"]
    return bool(
        base["b"] == b
        and base["u"] == u
        and base["P"] == p
        and valuations == {"u2": 3, "P2_minus_1": 2, "P2_plus_1": 1}
        and u[2] == 8
        and p[2] == 5
    )


def _difference_identity_is_valid(value: object) -> bool:
    if value != {"left": "f(a)-f(b)", "right_factors": ["a-b", "a+b+1"]}:
        return False
    # Both sides have degree at most two in each variable; exact evaluation on
    # this grid independently catches any coefficient mismatch.
    for a in range(-3, 4):
        for b in range(-3, 4):
            left = (a * a + a + 2) - (b * b + b + 2)
            if left != (a - b) * (a + b + 1):
                return False
    return True


def _induction_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "hypotheses",
        "sub_one_term_lower_bounds",
        "add_one_reason",
        "u_term_lower_bounds",
        "successor",
    }:
        return False
    hypotheses = _affines(value["hypotheses"], 3)
    sub_terms = _affines(value["sub_one_term_lower_bounds"], 2)
    u_terms = _affines(value["u_term_lower_bounds"], 2)
    successor = _affines(value["successor"], 3)
    if None in (hypotheses, sub_terms, u_terms, successor):
        return False
    assert hypotheses is not None and sub_terms is not None
    assert u_terms is not None and successor is not None
    if hypotheses != [(1, 2), (1, 1), (0, 1)]:
        return False

    expected_sub = [_add(hypotheses[1], hypotheses[2]), _add(hypotheses[0], (0, 1))]
    expected_u = [_add(hypotheses[0], hypotheses[2]), _scale(hypotheses[0], 2)]
    expected_successor = [expected_u[0], expected_sub[0], (0, 1)]
    shifted_hypotheses = [_shift(item) for item in hypotheses]
    return bool(
        sub_terms == expected_sub
        and _strictly_above(sub_terms[1], sub_terms[0])
        and value["add_one_reason"]
        == "P_2n_minus_1_divisible_by_8_implies_P_2n_plus_1_has_v2_1"
        and u_terms == expected_u
        and _strictly_above(u_terms[1], u_terms[0])
        and successor == expected_successor == shifted_hypotheses
    )


def _target_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "u_difference_term_lower_bounds",
        "u_difference",
        "scale_v2",
        "b_difference",
        "divides_exponent",
        "not_divides_exponent",
    }:
        return False
    terms = _affines(value["u_difference_term_lower_bounds"], 2)
    u_difference = _affine(value["u_difference"])
    b_difference = _affine(value["b_difference"])
    divides = _affine(value["divides_exponent"])
    not_divides = _affine(value["not_divides_exponent"])
    if None in (terms, u_difference, b_difference, divides, not_divides):
        return False
    assert terms is not None and u_difference is not None
    assert b_difference is not None
    assert divides is not None and not_divides is not None
    hypothesis_u = (1, 2)
    hypothesis_p_minus = (1, 1)
    expected_terms = [_add(hypothesis_u, hypothesis_p_minus), _scale(hypothesis_u, 2)]
    expected_b = (expected_terms[0][0], expected_terms[0][1] - 1)
    return bool(
        terms == expected_terms
        and _strictly_above(terms[1], terms[0])
        and u_difference == terms[0]
        and value["scale_v2"] == 1
        and b_difference == expected_b
        and divides == expected_b
        and not_divides == _add(expected_b, (0, 1))
    )


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    frozen_source = source.get("source", {})
    if (
        frozen_source.get("revision") != "2653cded72f5112acdc935b4f674711a780af95d"
        or frozen_source.get("problem_path") != "Putnam2025/A6/problem.lean"
        or frozen_source.get("solution_path") != "Putnam2025/A6/solution.lean"
        or not isinstance(result, dict)
        or set(result)
        != {
            "base",
            "difference_identity",
            "doubling_identities",
            "valuation_induction",
            "target_transfer",
            "finite_testing_role",
        }
    ):
        return False
    identities = result["doubling_identities"]
    return bool(
        _base_is_valid(result["base"])
        and _difference_identity_is_valid(result["difference_identity"])
        and identities
        == {
            "taylor": "u_2n-2*u_n=u_n*(P_n-1)+u_n^2*K",
            "product": "P_2n=P_n*(P_n+2*u_n*Delta)",
        }
        and _induction_is_valid(result["valuation_induction"])
        and _target_is_valid(result["target_transfer"])
        and result["finite_testing_role"] == "SANITY_ONLY_NOT_UNIVERSAL_PROOF"
    )


def _evidence_is_valid(evidence: object, result: object) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or not evidence:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = target.read_text().casefold()
        return all(
            fragment in text
            for fragment in (
                "f(a)-f(b)",
                "(3,2,1)",
                "successor",
                "valuation 2k+3",
                "b-difference",
                "finite recurrence values are sanity checks only",
            )
        )
    except (OSError, UnicodeError):
        return False


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = data.get("result")
    math_correct = bool(contract and _result_is_valid(result, _load_frozen_input()))
    evidence_valid = bool(
        math_correct and _evidence_is_valid(data.get("evidence"), result)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(contract and LIMITATION in data.get("limitations", []))
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
