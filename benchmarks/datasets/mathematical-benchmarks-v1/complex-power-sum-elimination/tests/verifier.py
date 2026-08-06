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
    "The checker replays exact polynomial and quadratic-field arithmetic; "
    "it is not an external proof assistant or a general complex-algebra prover."
)
Poly = tuple[Fraction, ...]
Quad = tuple[Fraction, Fraction]


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


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    n, d = value["numerator"], value["denominator"]
    if type(n) is not int or type(d) is not int or d <= 0 or math.gcd(abs(n), d) != 1:
        return None
    return Fraction(n, d)


def _poly(value: object) -> Poly | None:
    if not isinstance(value, list) or not value:
        return None
    coefficients = tuple(_rational(item) for item in value)
    if any(item is None for item in coefficients):
        return None
    exact = tuple(item for item in coefficients if item is not None)
    if len(exact) > 1 and exact[-1] == 0:
        return None
    return exact


def _padd(left: Poly, right: Poly) -> Poly:
    result = [Fraction()] * max(len(left), len(right))
    for index, value in enumerate(left):
        result[index] += value
    for index, value in enumerate(right):
        result[index] += value
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _pmul(left: Poly, right: Poly) -> Poly:
    result = [Fraction()] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _pscale(poly: Poly, scalar: Fraction) -> Poly:
    return tuple(value * scalar for value in poly)


def _power_sum_polynomials(product: Poly) -> dict[str, Poly]:
    s: Poly = (Fraction(), Fraction(1))
    values = [(Fraction(2),), s]
    for _ in range(2, 7):
        values.append(
            _padd(
                _pmul(s, values[-1]),
                _pscale(_pmul(product, values[-2]), Fraction(-1)),
            )
        )
    return {str(index): values[index] for index in range(2, 7)}


def _qadd(left: Quad, right: Quad) -> Quad:
    return left[0] + right[0], left[1] + right[1]


def _qmul(left: Quad, right: Quad) -> Quad:
    return left[0] * right[0] + 17 * left[1] * right[1], left[0] * right[1] + left[
        1
    ] * right[0]


def _qscale(value: Quad, scalar: Fraction) -> Quad:
    return value[0] * scalar, value[1] * scalar


def _qpow(value: Quad, exponent: int) -> Quad:
    result = (Fraction(1), Fraction())
    for _ in range(exponent):
        result = _qmul(result, value)
    return result


def _qnorm(value: Quad) -> Fraction:
    return value[0] * value[0] - 17 * value[1] * value[1]


def _surd(value: object) -> Quad | None:
    if not isinstance(value, dict) or set(value) != {"rational", "sqrt17"}:
        return None
    rational = _rational(value["rational"])
    radical = _rational(value["sqrt17"])
    return None if rational is None or radical is None else (rational, radical)


def _branch_is_valid(branch: object) -> int | None:
    if not isinstance(branch, dict) or set(branch) != {
        "sqrt17_sign",
        "sum",
        "product",
        "target",
        "denominator_norms",
    }:
        return None
    sign = branch["sqrt17_sign"]
    if type(sign) is not int or sign not in {-1, 1}:
        return None
    s = _surd(branch["sum"])
    p = _surd(branch["product"])
    target = _surd(branch["target"])
    if s is None or p is None or target is None:
        return None
    if (s, p, target) != (
        (Fraction(5), Fraction(sign)),
        (Fraction(11), Fraction(3 * sign)),
        (Fraction(10), Fraction(2 * sign)),
    ):
        return None

    norms = branch["denominator_norms"]
    if (
        norms != {"s": 8, "s_minus_12": 32, "five_s_minus_44": -64}
        or _qnorm(s) != 8
        or _qnorm(_qadd(s, (Fraction(-12), Fraction()))) != 32
        or _qnorm(_qadd(_qscale(s, Fraction(5)), (Fraction(-44), Fraction()))) != -64
    ):
        return None

    power_sums: list[Quad] = [(Fraction(2), Fraction()), s]
    for _ in range(2, 7):
        power_sums.append(
            _qadd(
                _qmul(s, power_sums[-1]),
                _qscale(_qmul(p, power_sums[-2]), Fraction(-1)),
            )
        )
    zero: Quad = (Fraction(), Fraction())
    checks = (
        all(power_sums[index] != zero for index in (1, 3, 5)),
        power_sums[2] == _qscale(power_sums[1], Fraction(4)),
        power_sums[4] == _qscale(power_sums[3], Fraction(2)),
        _qmul(target, power_sums[5]) == power_sums[6],
        _qadd(
            _qadd(_qpow(target, 2), _qscale(target, Fraction(-20))),
            (Fraction(32), Fraction()),
        )
        == zero,
    )
    if not all(checks):
        return None
    return sign


def _submitted_powers_match(value: object, expected: dict[str, Poly]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == set(expected)
        and all(_poly(value[key]) == polynomial for key, polynomial in expected.items())
    )


def _recurrence_is_valid(value: object, expected_powers: dict[str, Poly]) -> bool:
    if value is None:
        return True
    expected_product: Poly = (Fraction(), Fraction(-2), Fraction(1, 2))
    return bool(
        isinstance(value, dict)
        and set(value) == {"seed", "product_polynomial", "power_sums"}
        and value["seed"] == ["2", "s"]
        and _poly(value["product_polynomial"]) == expected_product
        and _submitted_powers_match(value["power_sums"], expected_powers)
    )


def _elimination_is_valid(value: object, expected_powers: dict[str, Poly]) -> bool:
    if (
        not isinstance(value, dict)
        or not {"sum_polynomial", "target_polynomial"} <= set(value)
        or not set(value)
        <= {"sum_polynomial", "hypothesis_factorization", "target_polynomial"}
    ):
        return False
    expected_factorization = "A4-2*A3=-(s^2/2)*(s^2-10*s+8)"
    factorization = value.get("hypothesis_factorization")
    difference = _padd(
        expected_powers["4"], _pscale(expected_powers["3"], Fraction(-2))
    )
    factorized = _pmul(
        (Fraction(), Fraction(), Fraction(-1, 2)),
        (Fraction(8), Fraction(-10), Fraction(1)),
    )
    return bool(
        _poly(value["sum_polynomial"]) == (Fraction(8), Fraction(-10), Fraction(1))
        and _poly(value["target_polynomial"])
        == (Fraction(32), Fraction(-20), Fraction(1))
        and factorization in {None, expected_factorization}
        and difference == factorized
    )


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    frozen_source = source.get("source", {})
    if (
        frozen_source.get("revision") != "ac6b9ff5614ce8454c03d8c03bff571b91f6d31a"
        or frozen_source.get("row") != 1672
        or frozen_source.get("row_sha256")
        != "sha256:3c90414d1940aeff2f696b02fa5757ecd506180128e5ffd0b2db681e4c8d3f51"
        or not isinstance(result, dict)
        or not {"elimination", "branches", "achievability"} <= set(result)
        or not set(result) <= {"recurrence", "elimination", "branches", "achievability"}
    ):
        return False
    expected_product: Poly = (Fraction(), Fraction(-2), Fraction(1, 2))
    expected_powers = _power_sum_polynomials(expected_product)
    if not _recurrence_is_valid(
        result.get("recurrence"), expected_powers
    ) or not _elimination_is_valid(result["elimination"], expected_powers):
        return False

    branches = result["branches"]
    if not isinstance(branches, list) or len(branches) != 2:
        return False
    signs = [_branch_is_valid(branch) for branch in branches]
    if set(signs) != {-1, 1}:
        return False
    achievability = result["achievability"]
    return isinstance(achievability, dict) and achievability == {
        "construction": "TAKE_BOTH_COMPLEX_ROOTS",
        "quadratic": "T^2-s*T+p",
        "branch_coverage": "BOTH_AND_ONLY_BOTH",
    }


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
        text = target.read_text().casefold()
        return all(
            fragment in text
            for fragment in (
                "power sum recurrence",
                "s^2-10s+8",
                "r^2-20r+32",
                "both branches",
                "norm",
            )
        )
    except (OSError, UnicodeError):
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
