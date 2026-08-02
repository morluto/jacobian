from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "This exact certificate covers only the frozen dividend and divisor family; it is not a general polynomial-factorization result."


def trim(p: list[Fraction]) -> list[Fraction]:
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def add(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * max(len(p), len(q))
    for i, value in enumerate(p):
        out[i] += value
    for i, value in enumerate(q):
        out[i] += value
    return trim(out)


def scale(p: list[Fraction], c: Fraction) -> list[Fraction]:
    return trim([c * value for value in p])


def shift(p: list[Fraction]) -> list[Fraction]:
    return [Fraction(0), *p]


def divmod_poly(
    numerator: list[Fraction], denominator: list[Fraction]
) -> tuple[list[Fraction], list[Fraction]]:
    rem = trim(numerator[:])
    den = trim(denominator[:])
    if den == [0]:
        raise ZeroDivisionError
    quotient = [Fraction(0)] * max(1, len(rem) - len(den) + 1)
    while rem != [0] and len(rem) >= len(den):
        degree = len(rem) - len(den)
        coefficient = rem[-1] / den[-1]
        quotient[degree] += coefficient
        for i, value in enumerate(den):
            rem[i + degree] -= coefficient * value
        trim(rem)
    return trim(quotient), trim(rem)


def monic_gcd(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    left, right = trim(p[:]), trim(q[:])
    while right != [0]:
        _, remainder = divmod_poly(left, right)
        left, right = right, remainder
    return scale(left, Fraction(1, 1) / left[-1])


def symbolic_remainder() -> tuple[list[Fraction], list[Fraction]]:
    # A residue is c0(a) + c1(a)x. Multiplication by x uses x^2=x-a.
    c0, c1 = [Fraction(1)], [Fraction(0)]
    powers: list[tuple[list[Fraction], list[Fraction]]] = []
    for _ in range(14):
        powers.append((c0, c1))
        c0, c1 = scale(shift(c1), Fraction(-1)), add(c0, c1)
    r0 = add(add(scale(powers[0][0], Fraction(90)), powers[1][0]), powers[13][0])
    r1 = add(add(scale(powers[0][1], Fraction(90)), powers[1][1]), powers[13][1])
    return trim(r0), trim(r1)


def multiply(p: list[int], q: list[int]) -> list[int]:
    out = [0] * (len(p) + len(q) - 1)
    for i, left in enumerate(p):
        for j, right in enumerate(q):
            out[i + j] += left * right
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def integers(value: Any) -> list[int] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(type(x) is not int for x in value)
    ):
        return None
    return value


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
    try:
        frozen_input_ok = (
            not (WORKSPACE / "input.json").is_symlink()
            and (WORKSPACE / "input.json").read_bytes()
            == (TESTS / "input.json").read_bytes()
        )
    except OSError:
        frozen_input_ok = False
    result = data.get("result", {})
    r0 = (
        integers(result.get("remainder_constant")) if isinstance(result, dict) else None
    )
    r1 = integers(result.get("remainder_x")) if isinstance(result, dict) else None
    gcd = integers(result.get("common_gcd")) if isinstance(result, dict) else None
    quotient = integers(result.get("quotient")) if isinstance(result, dict) else None
    parameter = result.get("parameter") if isinstance(result, dict) else None

    computed_r0, computed_r1 = symbolic_remainder()
    computed_gcd = monic_gcd(computed_r0, computed_r1)
    # Uniqueness and the parameter are consequences of the recomputed gcd,
    # not hard-coded expectations.  A monic linear gcd c0 + c1*a (c1 != 0)
    # has exactly one rational root a = -c0/c1; that root is the unique
    # integer parameter, and its integrality is what licenses UNIQUE_PARAMETER.
    gcd_is_linear = (
        len(computed_gcd) == 2 and computed_gcd[1] != 0
    )
    derived_parameter = (
        -computed_gcd[0] / computed_gcd[1] if gcd_is_linear else None
    )
    parameter_is_integer = (
        derived_parameter is not None
        and derived_parameter.denominator == 1
    )
    derived_parameter_int = (
        int(derived_parameter.numerator) if parameter_is_integer else None
    )
    symbolic_ok = (
        r0 is not None
        and r1 is not None
        and [Fraction(x) for x in r0] == computed_r0
        and [Fraction(x) for x in r1] == computed_r1
        and gcd is not None
        and [Fraction(x) for x in gcd] == computed_gcd
        and gcd_is_linear
        and parameter_is_integer
    )
    quotient_ok = (
        type(parameter) is int
        and parameter == derived_parameter_int
        and quotient is not None
        and multiply([parameter, -1, 1], quotient)
        == [90, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )
    math_correct = bool(contract and frozen_input_ok and symbolic_ok and quotient_ok)
    evidence_valid = bool(
        math_correct
        and evidence_list_is_bound(
            data.get("evidence"), expected_path="evidence/answer.txt"
        )
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(contract and LIMITATION in data.get("limitations", []))
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    passed = bool(
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
                "reward": float(passed),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
