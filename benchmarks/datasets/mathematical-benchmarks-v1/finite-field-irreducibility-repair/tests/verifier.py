from __future__ import annotations

import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
CONCLUSION = "BAD_REDUCTION_DIAGNOSED_AND_IRREDUCIBILITY_REPAIRED"


def _exact_int(value: object) -> bool:
    """Accept only a real int, rejecting bool which is an int subclass."""

    return type(value) is int


def _int_list(value: object, *, min_len: int, max_len: int) -> bool:
    """Accept a list of real ints with bounded cardinality, rejecting bools."""

    return (
        isinstance(value, list)
        and min_len <= len(value) <= max_len
        and all(type(x) is int for x in value)
    )


def _trim(a: list[int], p: int) -> list[int]:
    a = [x % p for x in a]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def _divmod(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]:
    a = _trim(a, p)
    b = _trim(b, p)
    if b == [0]:
        raise ZeroDivisionError
    q = [0] * max(1, len(a) - len(b) + 1)
    inv = pow(b[-1], -1, p)
    while a != [0] and len(a) >= len(b):
        shift = len(a) - len(b)
        coefficient = a[-1] * inv % p
        q[shift] = coefficient
        for i, value in enumerate(b):
            a[i + shift] = (a[i + shift] - coefficient * value) % p
        a = _trim(a, p)
    return _trim(q, p), a


def _gcd(a: list[int], b: list[int], p: int) -> list[int]:
    while _trim(b, p) != [0]:
        _, remainder = _divmod(a, b, p)
        a, b = b, remainder
    a = _trim(a, p)
    inv = pow(a[-1], -1, p)
    return _trim([x * inv for x in a], p)


def _mul_mod(a: list[int], b: list[int], modulus: list[int], p: int) -> list[int]:
    product = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            product[i + j] = (product[i + j] + x * y) % p
    return _divmod(product, modulus, p)[1]


def _pow_x(exponent: int, modulus: list[int], p: int) -> list[int]:
    result, base = [1], [0, 1]
    while exponent:
        if exponent & 1:
            result = _mul_mod(result, base, modulus, p)
        base = _mul_mod(base, base, modulus, p)
        exponent //= 2
    return _trim(result, p)


def _prime(p: int) -> bool:
    return p >= 2 and all(p % d for d in range(2, int(p**0.5) + 1))


def _source_is_bound() -> bool:
    try:
        hidden = (TESTS / "input.json").read_bytes()
        data = json.loads(hidden)
        return (WORKSPACE / "input.json").read_bytes() == hidden and data["source"][
            "revision"
        ] == "71595210590450202b7b69225bc07e9e01b13c5c"
    except (OSError, ValueError, KeyError):
        return False


def _result(value: object) -> bool:
    required = {
        "bad_prime",
        "bad_reduction",
        "bad_factor",
        "bad_factor_power",
        "repair_prime",
        "p2_remainder",
        "p4_remainder",
        "rabin_gcd_degree",
        "rational_conclusion",
    }
    if not isinstance(value, dict) or set(value) != required:
        return False
    if (
        not _exact_int(value["bad_prime"])
        or value["bad_prime"] != 2
        or not _int_list(value["bad_reduction"], min_len=5, max_len=5)
        or value["bad_reduction"] != [1, 0, 0, 0, 1]
        or not _int_list(value["bad_factor"], min_len=2, max_len=2)
        or value["bad_factor"] != [1, 1]
        or not _exact_int(value["bad_factor_power"])
        or value["bad_factor_power"] != 4
    ):
        return False
    p = value["repair_prime"]
    if not _exact_int(p) or not 3 <= p <= 97 or not _prime(p):
        return False
    modulus = [1 % p, (-4) % p, 0, 0, 1]
    p2 = _pow_x(p * p, modulus, p)
    p4 = _pow_x(p**4, modulus, p)
    submitted_p2 = value["p2_remainder"]
    submitted_p4 = value["p4_remainder"]
    if not _int_list(submitted_p2, min_len=1, max_len=4):
        return False
    if not _int_list(submitted_p4, min_len=1, max_len=4):
        return False
    x = [0, 1]
    difference = _trim(
        [
            (p2[i] if i < len(p2) else 0) - (x[i] if i < len(x) else 0)
            for i in range(max(len(p2), 2))
        ],
        p,
    )
    gcd = _gcd(modulus, difference, p)
    return bool(
        _trim(submitted_p2, p) == p2
        and _trim(submitted_p4, p) == p4
        and p4 == x
        and gcd == [1]
        and _exact_int(value["rabin_gcd_degree"])
        and value["rabin_gcd_degree"] == 0
        and value["rational_conclusion"] == "IRREDUCIBLE_OVER_Q"
    )


def _evaluate(submission: object) -> dict[str, float | bool]:
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(_source_is_bound() and _result(data.get("result")))
    reward = float(math_correct)
    return {
        "correctness": float(math_correct),
        "reward": reward,
    }


def main() -> None:
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_evaluate(load_submission()), sort_keys=True) + "\n"
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
