import itertools
import json
import math
import re
from pathlib import Path

from verifier_support import (
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

E = Path("/tests")
INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
ALLOWED = frozenset({"COMPUTED"})


def _integer(value):
    if not isinstance(value, str) or len(value.lstrip("-")) > 128:
        raise ValueError
    if INTEGER.fullmatch(value) is None or str(int(value)) != value:
        raise ValueError
    return int(value)


def _is_perfect(residues, modulus):
    if len(residues) != len(set(residues)):
        return False
    if modulus != len(residues) * (len(residues) - 1) + 1:
        return False
    differences = [
        (left - right) % modulus
        for left in residues
        for right in residues
        if left != right
    ]
    return sorted(differences) == list(range(1, modulus))


def _finite_core(value, frozen, expected):
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "integer_set",
        "ordered_differences",
        "sidon",
        "fixed_order_checks",
        "universal_obstruction_replayed",
    }:
        return False
    try:
        elements = [_integer(item) for item in value["integer_set"]]
    except (TypeError, ValueError):
        return False
    if value["integer_set"] != frozen.get("integer_set") or len(elements) != len(
        set(elements)
    ):
        return False
    ordered = [
        {
            "minuend": str(left),
            "subtrahend": str(right),
            "difference": str(left - right),
        }
        for left in elements
        for right in elements
        if left != right
    ]
    differences = [int(item["difference"]) for item in ordered]
    if (
        value["schema_version"] != "1"
        or value["ordered_differences"] != ordered
        or value["sidon"] is not True
        or len(set(differences)) != len(differences)
        or value["universal_obstruction_replayed"] is not False
    ):
        return False
    checks = value["fixed_order_checks"]
    orders = expected["orders"]
    if not isinstance(checks, list) or len(checks) != len(orders):
        return False
    for check, order, candidate_count in zip(
        checks, orders, expected["candidate_counts"], strict=True
    ):
        if not isinstance(check, dict) or set(check) != {
            "target_order",
            "modulus",
            "base_residues",
            "candidate_space_size",
            "decision",
            "coverage",
        }:
            return False
        modulus = order * (order - 1) + 1
        base = sorted({element % modulus for element in elements})
        additional = order - len(base)
        calculated_count = math.comb(modulus - len(base), additional)
        if (
            check
            != {
                "target_order": order,
                "modulus": modulus,
                "base_residues": base,
                "candidate_space_size": calculated_count,
                "decision": "DOES_NOT_EXTEND",
                "coverage": "ALL_CANDIDATES",
            }
            or calculated_count != candidate_count
        ):
            return False
        pool = [residue for residue in range(modulus) if residue not in set(base)]
        if any(
            _is_perfect(sorted((*base, *extra)), modulus)
            for extra in itertools.combinations(pool, additional)
        ):
            return False
    return True


def _result(value):
    return value == {
        "evidence_path": "evidence/finite-core.json",
        "sidon": True,
        "fixed_orders_checked": [5, 6, 7],
        "public_universal_result": (
            "A_IS_NOT_CONTAINED_IN_ANY_FINITE_PERFECT_DIFFERENCE_SET"
        ),
        "universal_obstruction_replayed": False,
    }


def main():
    submission = load_submission()
    frozen = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    evidence = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/finite-core.json"
        )
        if contract
        else None
    )
    evidence_valid = bool(evidence is not None)
    math_correct = bool(
        contract
        and _result(submission["result"])
        and _finite_core(evidence, frozen, expected)
    )
    scope = bool(contract and submission["scope"] == expected["scope"])
    assurance = bool(
        contract and submission["claimed_assurance"] == expected["maximum_assurance"]
    )
    limitations = bool(
        contract
        and submission["limitations"]
        == [
            "PUBLIC_UNIVERSAL_OBSTRUCTION_NOT_REPLAYED",
            "FINITE_ORDERS_DO_NOT_PROVE_UNIVERSAL_RESULT",
        ]
    )
    false_certification = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0
        if (
            not math_correct
            or not evidence_valid
            or not scope
            or not assurance
            or not limitations
            or false_certification
        )
        else 0.7
        + 0.1 * evidence_valid
        + 0.1 * scope
        + 0.05 * assurance
        + 0.05 * limitations
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "limitation_accuracy": float(limitations),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
