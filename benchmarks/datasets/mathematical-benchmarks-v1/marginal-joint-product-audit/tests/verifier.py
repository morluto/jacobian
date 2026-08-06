import json
import re
from collections import defaultdict
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
SUPPORT = (-3, -1, 2, 5)
MARGINAL = (Fraction(1, 10), Fraction(1, 5), Fraction(3, 10), Fraction(2, 5))
LIMITATION = (
    "This exact finite-law countermodel does not machine-verify a general "
    "weak-convergence theorem or disambiguate the original prose."
)


def canonical_fraction(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"(?:0|1|[1-9][0-9]*/[1-9][0-9]*)", value) is None
    ):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    if parsed < 0 or parsed > 1 or str(parsed) != value:
        return None
    return parsed


def parse_joint(entries):
    expected_pairs = [(x, y) for x in SUPPORT for y in SUPPORT]
    if not isinstance(entries, list) or len(entries) != len(expected_pairs):
        return None
    table = {}
    for entry, pair in zip(entries, expected_pairs, strict=True):
        if not isinstance(entry, dict) or set(entry) != {"x", "y", "mass"}:
            return None
        if (entry.get("x"), entry.get("y")) != pair:
            return None
        mass = canonical_fraction(entry.get("mass"))
        if mass is None:
            return None
        table[pair] = mass
    return table if sum(table.values()) == 1 else None


def marginals(table):
    left = {x: sum(table[x, y] for y in SUPPORT) for x in SUPPORT}
    right = {y: sum(table[x, y] for x in SUPPORT) for y in SUPPORT}
    return left, right


def target_marginal():
    return dict(zip(SUPPORT, MARGINAL, strict=True))


def product_pushforward(table):
    distribution = defaultdict(Fraction)
    for (x, y), mass in table.items():
        distribution[x * y] += mass
    return {value: mass for value, mass in sorted(distribution.items()) if mass}


def parse_product(entries):
    if not isinstance(entries, list) or not entries:
        return None
    distribution = {}
    prior = None
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"value", "mass"}:
            return None
        value = entry.get("value")
        mass = canonical_fraction(entry.get("mass"))
        if type(value) is not int or mass is None:
            return None
        if prior is not None and value <= prior:
            return None
        distribution[value] = mass
        prior = value
    if sum(distribution.values()) != 1:
        return None
    return {value: mass for value, mass in distribution.items() if mass}


def evidence_matches(evidence):
    # The full finite laws and product pushforwards are independently replayed.
    # The public evidence contract promises only one bound text artifact.
    return evidence_list_is_bound(evidence)


def result_is_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "diagnosis",
        "sequence_model",
        "support",
        "prelimit_joint",
        "limit_joint",
        "prelimit_product_distribution",
        "limit_product_distribution",
        "witness_product_value",
        "missing_assumption",
    }:
        return False
    if (
        result.get("diagnosis") != "MISSING_JOINT_LAW_CONTROL"
        or result.get("sequence_model") != "CONSTANT_IN_N"
    ):
        return False
    if result.get("support") != list(SUPPORT):
        return False
    if result.get("missing_assumption") not in {
        "JOINT_CONVERGENCE",
        "LIMIT_PAIR_INDEPENDENCE",
    }:
        return False
    prelimit = parse_joint(result.get("prelimit_joint"))
    limit = parse_joint(result.get("limit_joint"))
    if prelimit is None or limit is None:
        return False
    target = target_marginal()
    if marginals(prelimit) != (target, target) or marginals(limit) != (target, target):
        return False
    independent = {(x, y): target[x] * target[y] for x in SUPPORT for y in SUPPORT}
    if prelimit != independent or limit == independent:
        return False
    prelimit_product = product_pushforward(prelimit)
    limit_product = product_pushforward(limit)
    if parse_product(result.get("prelimit_product_distribution")) != prelimit_product:
        return False
    if parse_product(result.get("limit_product_distribution")) != limit_product:
        return False
    witness = result.get("witness_product_value")
    return type(witness) is int and prelimit_product.get(
        witness, 0
    ) != limit_product.get(witness, 0)


def main():
    expected = json.loads((T / "expected.json").read_text())
    input_bound = workspace_input_is_bound()
    submission = load_submission(W / "submission.json", require_input_binding=False)
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = result_is_valid(result)
    evidence_ok = bool(
        isinstance(submission, dict) and evidence_matches(submission.get("evidence"))
    )
    scope_ok = bool(
        isinstance(submission, dict)
        and submission.get("scope")
        == "frozen-four-point-marginal-and-submitted-couplings"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == [LIMITATION]
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(
        contract
        and input_bound
        and math_ok
        and evidence_ok
        and scope_ok
        and not false_cert
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_bound),
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance_ok,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
