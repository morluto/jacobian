import itertools
import json
import math
from fractions import Fraction
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
    "The checker replays exact algebraic certificates and a finite group-action "
    "count; Chebotarev's density theorem is a declared trusted theorem."
)


def _load_frozen_input() -> dict[str, Any]:
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        payload = frozen.read_bytes()
        if workspace.read_bytes() != payload:
            return {}
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _trim(poly: list[int]) -> list[int]:
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _mul_mod(left: list[int], right: list[int], prime: int) -> list[int]:
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = (result[i + j] + a * b) % prime
    return _trim(result)


def _eval_mod(poly: list[int], value: int, prime: int) -> int:
    total = 0
    for coefficient in reversed(poly):
        total = (total * value + coefficient) % prime
    return total


def _shift_plus_one(poly: list[int]) -> list[int]:
    result = [0] * len(poly)
    for degree, coefficient in enumerate(poly):
        for power in range(degree + 1):
            result[power] += coefficient * math.comb(degree, power)
    return _trim(result)


def _factorization_certificate(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"prime", "linear", "cubic"}:
        return False
    prime = value["prime"]
    linear = value["linear"]
    cubic = value["cubic"]
    if (
        type(prime) is not int
        or prime not in {3, 5, 7, 17}
        or not isinstance(linear, list)
        or not isinstance(cubic, list)
        or len(linear) != 2
        or len(cubic) != 4
        or any(type(item) is not int for item in linear + cubic)
        or linear[-1] % prime != 1
        or cubic[-1] % prime != 1
    ):
        return False
    target = [1, -4, 0, 0, 1]
    if _mul_mod(linear, cubic, prime) != [item % prime for item in target]:
        return False
    return all(_eval_mod(cubic, root, prime) != 0 for root in range(prime))


def _fixed_point_count() -> tuple[int, dict[str, int]]:
    counts = {
        "identity": 0,
        "transposition": 0,
        "three_cycle": 0,
        "fixed_point_free": 0,
    }
    fixed_total = 0
    for permutation in itertools.permutations(range(4)):
        fixed = sum(permutation[i] == i for i in range(4))
        if fixed:
            fixed_total += 1
        if fixed == 4:
            counts["identity"] += 1
        elif fixed == 2:
            counts["transposition"] += 1
        elif fixed == 1:
            counts["three_cycle"] += 1
        else:
            counts["fixed_point_free"] += 1
    return fixed_total, counts


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    required = {
        "shifted_polynomial",
        "eisenstein_prime",
        "discriminant",
        "discriminant_factorization",
        "frobenius_factorization",
        "galois_group",
        "cycle_count",
        "density",
        "encoded_answer",
        "source_defects",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or source.get("source", {}).get("row_sha256")
        != "sha256:26404f5cbfbeac8a8e02fce8369d781fb5424ebb99cf45fb7905994b665a6efd"
        or source.get("problem", {}).get("polynomial_coefficients_ascending")
        != [1, -4, 0, 0, 1]
        or result["shifted_polynomial"] != [-2, 0, 6, 4, 1]
        or result["eisenstein_prime"] != 2
        or result["discriminant"] != -6656
        or result["discriminant_factorization"]
        != {
            "sign": -1,
            "prime_powers": [{"prime": 2, "exponent": 9}, {"prime": 13, "exponent": 1}],
        }
        or not _factorization_certificate(result["frobenius_factorization"])
        or result["galois_group"] != "S4"
    ):
        return False
    # Independently check the shifted polynomial and Eisenstein hypotheses.
    source_poly = source["problem"]["polynomial_coefficients_ascending"]
    shifted = result["shifted_polynomial"]
    if shifted != _shift_plus_one(source_poly):
        return False
    if shifted[-1] != 1 or any(coefficient % 2 for coefficient in shifted[:-1]):
        return False
    if shifted[0] % 4 == 0:
        return False
    # For the depressed monic quartic x^4+q*x+r, Delta=256*r^3-27*q^4.
    q = source_poly[1]
    r = source_poly[0]
    if result["discriminant"] != 256 * r**3 - 27 * q**4:
        return False
    fixed_total, counts = _fixed_point_count()
    cycle_count = result["cycle_count"]
    if cycle_count != {**counts, "with_fixed_point": fixed_total, "group_order": 24}:
        return False
    return bool(
        Fraction(*result["density"]) == Fraction(fixed_total, 24) == Fraction(5, 8)
        and result["encoded_answer"] == 508
        and set(result["source_defects"])
        == {
            "MOD_2_IRREDUCIBILITY_FALSE",
            "DISCRIMINANT_VALUE_FALSE",
            "DOUBLE_TRANSPOSITIONS_MISCOUNTED_AS_FIXED_POINT_ELEMENTS",
            "DENSITY_AND_ENCODED_ANSWER_FALSE",
        }
    )


def _evidence_matches_result(evidence: object, result: dict[str, Any]) -> bool:
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
        markers = [
            line.removeprefix("RESULT_JSON:").strip()
            for line in target.read_text().splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        return len(markers) == 1 and json.loads(markers[0]) == result
    except (OSError, UnicodeError, ValueError):
        return False


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = data.get("result")
    math_correct = bool(contract and _result_is_valid(result, source))
    evidence_valid = bool(
        math_correct
        and isinstance(result, dict)
        and _evidence_matches_result(data.get("evidence"), result)
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
