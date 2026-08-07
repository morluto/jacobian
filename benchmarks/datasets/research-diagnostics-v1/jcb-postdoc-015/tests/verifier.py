import itertools
import json
import math
import re
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    read_evidence_json,
    strict_submission_contract,
)

E = Path("/tests")
INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
ALLOWED = frozenset({"COMPUTED"})
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024


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


def _ordered_difference_key(row):
    if not isinstance(row, dict) or "minuend" not in row or "subtrahend" not in row:
        raise ValueError
    minuend = row["minuend"]
    subtrahend = row["subtrahend"]
    if type(minuend) is not str or type(subtrahend) is not str:
        raise ValueError
    return (_integer(minuend), _integer(subtrahend))


def _normalize_differences(submitted, expected_rows):
    """Return sorted unique submitted differences, or None if malformed."""

    if not isinstance(submitted, list):
        return None
    try:
        normalized = sorted(submitted, key=_ordered_difference_key)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(normalized) != len({_ordered_difference_key(row) for row in normalized}):
        return None
    if normalized != sorted(expected_rows, key=_ordered_difference_key):
        return None
    return normalized


def _normalize_fixed_order_checks(checks, orders):
    """Return sorted unique fixed-order checks, or None if malformed."""

    if not isinstance(checks, list) or len(checks) != len(orders):
        return None
    if any(
        not isinstance(row, dict) or type(row.get("target_order")) is not int
        for row in checks
    ):
        return None
    try:
        normalized = sorted(checks, key=lambda row: row["target_order"])
        keys = {row["target_order"] for row in normalized}
    except (TypeError, KeyError, ValueError, OverflowError):
        return None
    if len(normalized) != len(keys):
        return None
    return normalized


def _fixed_order_check_is_valid(check, order, candidate_count, elements):
    """Validate one fixed-order extension check against independent replay."""

    if not isinstance(check, dict) or set(check) != {
        "target_order",
        "modulus",
        "base_residues",
        "candidate_space_size",
        "decision",
        "coverage",
    }:
        return False
    if (
        type(check["target_order"]) is not int
        or type(check["modulus"]) is not int
        or type(check["candidate_space_size"]) is not int
        or not isinstance(check["base_residues"], list)
        or not all(type(r) is int for r in check["base_residues"])
    ):
        return False
    modulus = order * (order - 1) + 1
    base = sorted({element % modulus for element in elements})
    additional = order - len(base)
    calculated_count = math.comb(modulus - len(base), additional)
    # Compare base_residues as a duplicate-free set; the public schema
    # requires only unique integer items and the instruction treats these
    # residues as a set, so order must not affect equality.
    submitted_residues = check["base_residues"]
    if (
        check["target_order"] != order
        or check["modulus"] != modulus
        or len(submitted_residues) != len(set(submitted_residues))
        or sorted(submitted_residues) != base
        or check["candidate_space_size"] != calculated_count
        or check["decision"] != "DOES_NOT_EXTEND"
        or check["coverage"] != "ALL_CANDIDATES"
        or calculated_count != candidate_count
    ):
        return False
    pool = [residue for residue in range(modulus) if residue not in set(base)]
    return not any(
        _is_perfect(sorted((*base, *extra)), modulus)
        for extra in itertools.combinations(pool, additional)
    )


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
    if _normalize_differences(value["ordered_differences"], ordered) is None:
        return False
    if (
        value["schema_version"] != "1"
        or value["sidon"] is not True
        or len(set(differences)) != len(differences)
        or value["universal_obstruction_replayed"] is not False
    ):
        return False
    normalized_checks = _normalize_fixed_order_checks(
        value["fixed_order_checks"], expected["orders"]
    )
    if normalized_checks is None:
        return False
    for check, order, candidate_count in zip(
        normalized_checks,
        sorted(expected["orders"]),
        sorted(expected["candidate_counts"]),
        strict=True,
    ):
        if not _fixed_order_check_is_valid(check, order, candidate_count, elements):
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


def _safe_evidence(submission):
    """Read digest-bound evidence independently of protocol compliance."""

    if not isinstance(submission, dict):
        return None
    evidence = submission.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return None
    return read_evidence_json(
        evidence[0],
        expected_path="evidence/finite-core.json",
        max_bytes=MAX_EVIDENCE_BYTES,
    )


def main():
    submission = load_submission_raw()
    frozen = json.loads((E / "input.json").read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    evidence = _safe_evidence(submission)
    finite_core_valid = bool(
        evidence is not None and _finite_core(evidence, frozen, expected)
    )
    evidence_valid = bool(evidence is not None and finite_core_valid)
    math_correct = bool(
        isinstance(submission, dict)
        and isinstance(submission.get("result"), dict)
        and _result(submission["result"])
        and finite_core_valid
    )
    scope = bool(
        isinstance(submission, dict) and submission.get("scope") == expected["scope"]
    )
    assurance = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(
        isinstance(submission, dict)
        and submission.get("limitations")
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
            not contract
            or not math_correct
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
