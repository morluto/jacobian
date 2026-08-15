import json
import math
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def _load_frozen_input():
    try:
        workspace, frozen = W / "input.json", E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _trim(poly):
    result = list(poly)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _integer_list(value):
    if not isinstance(value, list) or not value:
        return None
    normalized = [_integer(item) for item in value]
    return normalized if all(item is not None for item in normalized) else None


def _mul(left, right):
    result = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return _trim(result)


def _divide_exact(dividend, divisor):
    remainder = _trim(dividend)
    divisor = _trim(divisor)
    if not divisor or divisor[-1] not in {1, -1} or len(remainder) < len(divisor):
        return None
    quotient = [0] * (len(remainder) - len(divisor) + 1)
    while len(remainder) >= len(divisor) and remainder != [0]:
        offset = len(remainder) - len(divisor)
        lead = remainder[-1] // divisor[-1]
        if lead * divisor[-1] != remainder[-1]:
            return None
        quotient[offset] += lead
        for index, coefficient in enumerate(divisor):
            remainder[index + offset] -= lead * coefficient
        remainder = _trim(remainder)
    return _trim(quotient) if remainder == [0] else None


def _cyclotomic(order, cache):
    if order in cache:
        return cache[order]
    polynomial = [-1] + [0] * (order - 1) + [1]
    for divisor in range(1, order):
        if order % divisor == 0:
            polynomial = _divide_exact(polynomial, _cyclotomic(divisor, cache))
            if polynomial is None:
                return None
    cache[order] = polynomial
    return polynomial


def _parse_factors(factors, frozen):
    if not isinstance(factors, list) or not factors or len(factors) > 8:
        return None
    parsed, total = [], 0
    for factor in factors:
        if not isinstance(factor, dict) or set(factor) != {"order", "multiplicity"}:
            return None
        order, multiplicity = factor["order"], factor["multiplicity"]
        order = _integer(order)
        multiplicity = _integer(multiplicity)
        if (
            order is None
            or multiplicity is None
            or not 1 < order <= frozen.get("maximum_cyclotomic_order", 0)
            or not 1 <= multiplicity <= 8
        ):
            return None
        parsed.append((order, multiplicity))
        total += multiplicity
    if total > frozen.get("maximum_total_multiplicity", 0):
        return None
    return parsed


def _cyclotomic_product(parsed, leading):
    cache = {1: [-1, 1]}
    product = [leading]
    for order, multiplicity in sorted(parsed):
        factor = _cyclotomic(order, cache)
        if factor is None or factor != list(reversed(factor)):
            return None
        for _ in range(multiplicity):
            product = _mul(product, factor)
    return product


def _coefficients_match(
    product, expanded, reciprocal, coefficients, p_at_one, reciprocal_scalar, conclusion
):
    return bool(
        product == coefficients
        and expanded == product
        and reciprocal == list(reversed(product))
        and product == list(reversed(product))
        and p_at_one == sum(product)
        and p_at_one != 0
        and reciprocal_scalar == 1
        and conclusion == "INVERSION_CLOSED_WITH_EQUAL_MULTIPLICITIES"
    )


def _result_is_valid(result, frozen):
    required = {
        "leading_coefficient",
        "factors",
        "expanded_coefficients",
        "reciprocal_coefficients",
        "p_at_one",
        "reciprocal_scalar",
        "root_orbit_conclusion",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    leading = _integer(result["leading_coefficient"])
    p_at_one = _integer(result["p_at_one"])
    reciprocal_scalar = _integer(result["reciprocal_scalar"])
    if leading is None or leading == 0:
        return False
    parsed = _parse_factors(result["factors"], frozen)
    if parsed is None:
        return False
    product = _cyclotomic_product(parsed, leading)
    if product is None:
        return False
    expanded = _integer_list(result["expanded_coefficients"])
    reciprocal = _integer_list(result["reciprocal_coefficients"])
    coefficients = frozen.get("coefficients")
    return _coefficients_match(
        product,
        expanded,
        reciprocal,
        coefficients,
        p_at_one,
        reciprocal_scalar,
        result["root_orbit_conclusion"],
    )


def main():
    submission, frozen = load_submission(), _load_frozen_input()
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(
        isinstance(submission, dict) and _result_is_valid(data.get("result"), frozen)
    )
    correct = math_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
