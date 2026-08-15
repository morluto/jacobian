from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "This exact certificate covers only the frozen dividend and divisor family; it is not a general polynomial-factorization result."
RESULT_KEYS = frozenset(
    {"parameter", "remainder_constant", "remainder_x", "common_gcd", "quotient"}
)


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


def derive_unique_parameter(gcd: list[Fraction]) -> int | None:
    """Derive the unique integer parameter from a monic linear gcd c0 + c1*a.

    A monic linear gcd with c1 != 0 has exactly one rational root a = -c0/c1.
    Return that root as an int when it is integral, otherwise None.  Nonlinear
    or constant gcds also return None: they cannot certify a *unique* integer
    parameter.  Extracted from ``main`` so the derivation logic can be exercised
    with controlled gcds whose root is not the canonical value 2.
    """
    if len(gcd) != 2 or gcd[1] == 0:
        return None
    root = -gcd[0] / gcd[1]
    if root.denominator != 1:
        return None
    return int(root.numerator)


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
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or (isinstance(x, float) and not x.is_integer())
            for x in value
        )
    ):
        return None
    return [int(x) for x in value]


def integer_value(value: Any) -> int | None:
    """Normalize a JSON number accepted by the schema's ``integer`` type."""

    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def load_regular_submission() -> dict[str, Any] | None:
    """Parse the submission, rejecting symlinked submission artifacts.

    The shared ``load_submission`` helper follows symlinks; this local wrapper
    requires the declared submission path to be a regular file rooted under
    the workspace so an agent cannot alias it to another workspace file.
    Deeply nested JSON can raise ``RecursionError`` from ``json.loads``; catch
    it here so a malformed submission scores zero instead of crashing the
    verifier without writing ``reward.json``.
    """
    path = WORKSPACE / "submission.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1_048_576:
            return None
        root = WORKSPACE.resolve()
        target = path.resolve(strict=True)
        if not target.is_relative_to(root):
            return None
    except OSError:
        return None
    try:
        return load_submission(path)
    except RecursionError:
        return None


def main() -> None:
    submission = load_regular_submission()
    protocol_ok = submission is not None
    data = submission if protocol_ok else {}
    result = data.get("result", {})
    result_typed = result if isinstance(result, dict) else {}
    result_schema_ok = isinstance(result, dict) and set(result) == RESULT_KEYS
    r0 = integers(result_typed.get("remainder_constant"))
    r1 = integers(result_typed.get("remainder_x"))
    gcd = integers(result_typed.get("common_gcd"))
    quotient = integers(result_typed.get("quotient"))
    parameter = result_typed.get("parameter")
    parameter_value = integer_value(parameter)

    computed_r0, computed_r1 = symbolic_remainder()
    computed_gcd = monic_gcd(computed_r0, computed_r1)
    gcd_is_linear = len(computed_gcd) == 2 and computed_gcd[1] != 0
    derived_parameter_int = derive_unique_parameter(computed_gcd)
    parameter_is_integer = derived_parameter_int is not None
    symbolic_ok = (
        r0 is not None
        and r1 is not None
        and trim([Fraction(x) for x in r0]) == computed_r0
        and trim([Fraction(x) for x in r1]) == computed_r1
        and gcd is not None
        and trim([Fraction(x) for x in gcd]) == computed_gcd
        and gcd_is_linear
        and parameter_is_integer
    )
    quotient_ok = (
        parameter_value is not None
        and parameter_value == derived_parameter_int
        and quotient is not None
        and trim(multiply([parameter_value, -1, 1], trim(quotient)))
        == [90, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    )
    math_correct = bool(
        protocol_ok and result_schema_ok and symbolic_ok and quotient_ok
    )
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
