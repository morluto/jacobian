import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
ZERO = (0, 0, 0, 0)


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_polynomial(value, maximum_degree):
    if not isinstance(value, list) or len(value) > 70:
        return None
    result = {}
    for term in value:
        if not isinstance(term, dict) or set(term) != {"exponents", "coefficient"}:
            return None
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 4
            or any(type(item) is not int or item < 0 for item in exponents)
            or sum(exponents) > maximum_degree
        ):
            return None
        try:
            coefficient = Fraction(term["coefficient"])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        if coefficient.denominator != 1:
            return None
        exponent = tuple(exponents)
        if (
            coefficient == 0
            or str(coefficient) != term["coefficient"]
            or exponent in result
        ):
            return None
        result[exponent] = coefficient
    return result


def _add(left, right):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _mul(left, right):
    result = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(x + y for x, y in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _variable(index, degree=1):
    exponent = [0, 0, 0, 0]
    exponent[index] = degree
    return {tuple(exponent): Fraction(1)}


def _generators_and_target():
    linear = {}
    quadratic = {}
    target = {}
    for index in range(4):
        linear = _add(linear, _variable(index))
        quadratic = _add(quadratic, _variable(index, 2))
        target = _add(target, _variable(index, 4))
    target[(1, 1, 1, 1)] = Fraction(4)
    return linear, quadratic, target


def _result_is_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "variables",
        "generator_multipliers",
        "identity_conclusion",
        "divisibility_conclusion",
    }:
        return False
    if result["variables"] != frozen.get("variables") or result["variables"] != [
        "a",
        "b",
        "c",
        "d",
    ]:
        return False
    multipliers = result["generator_multipliers"]
    if not isinstance(multipliers, list) or len(multipliers) != 2:
        return False
    parsed = [
        _parse_polynomial(value, frozen.get("maximum_multiplier_degree"))
        for value in multipliers
    ]
    if any(value is None for value in parsed):
        return False
    linear, quadratic, target = _generators_and_target()
    reconstructed = _add(_mul(linear, parsed[0]), _mul(quadratic, parsed[1]))
    return bool(
        reconstructed == target
        and result["identity_conclusion"] == "TARGET_IN_HYPOTHESIS_IDEAL"
        and result["divisibility_conclusion"] == "TARGET_DIVISIBLE_BY_N"
    )


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    frozen = _load_frozen_input()
    math_correct = bool(submission and _result_is_valid(data.get("result"), frozen))
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(math_correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
