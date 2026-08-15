import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")


def add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for exponent, coefficient in polynomial.items():
            result[exponent] = result.get(exponent, Fraction()) + coefficient
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def scale(polynomial, scalar):
    return {
        exponent: coefficient * scalar
        for exponent, coefficient in polynomial.items()
        if coefficient * scalar
    }


def multiply(left, right):
    result = {}
    for a, ca in left.items():
        for b, cb in right.items():
            exponent = tuple(x + y for x, y in zip(a, b, strict=True))
            result[exponent] = result.get(exponent, Fraction()) + ca * cb
    return {
        exponent: coefficient for exponent, coefficient in result.items() if coefficient
    }


def monomial(exponents, coefficient=1):
    return {tuple(exponents): Fraction(coefficient)}


def parse_polynomial(value):
    if not isinstance(value, list) or not value:
        return None
    result = {}
    for term in value:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            return None
        try:
            coefficient = Fraction(term["coefficient"])
        except (ValueError, ZeroDivisionError):
            return None
        if str(coefficient) != term["coefficient"] or not coefficient:
            return None
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 3
            or any(type(x) is not int or x < 0 for x in exponents)
        ):
            return None
        key = tuple(exponents)
        if key in result:
            return None
        result[key] = coefficient
    return result


def expected_polynomials():
    p = monomial((1, 0, 0))
    q = monomial((0, 1, 0))
    s = monomial((0, 0, 1))
    even = add(multiply(p, p), scale(q, -1))
    odd_coefficient = add(p, q)
    norm = add(
        multiply(even, even),
        scale(multiply(multiply(odd_coefficient, odd_coefficient), s), -1),
        scale(multiply(odd_coefficient, multiply(s, s)), -2),
        scale(multiply(multiply(s, s), s), -1),
    )
    return {
        "p_numerator": monomial((0, 1, 0)),
        "p_denominator": monomial((1, 0, 0)),
        "q_numerator": add(
            scale(monomial((0, 2, 0)), 13), scale(monomial((1, 0, 1)), -12)
        ),
        "q_denominator": monomial((2, 0, 0)),
        "d_delta_inverse": monomial((1, 1, 0)),
        "d2_delta_numerator": add(
            scale(monomial((1, 2, 0)), 13), scale(monomial((1, 0, 1)), -1)
        ),
        "d2_delta_denominator": monomial((0, 0, 0), 12),
        "s_forward": add(monomial((0, 0, 3)), scale(monomial((1, 0, 0)), -1)),
        "delta_inverse": add(monomial((0, 3, 0)), scale(monomial((0, 0, 1)), -1)),
        "norm_polynomial": norm,
    }


def valid_result(result):
    if not isinstance(result, dict) or set(result) != set(expected_polynomials()):
        return False
    parsed = {name: parse_polynomial(value) for name, value in result.items()}
    return (
        all(value is not None for value in parsed.values())
        and parsed == expected_polynomials()
    )


def main():
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    input_bound = workspace_input_is_bound()
    result = data.get("result")
    mathematical = bool(
        isinstance(submission, dict) and input_bound and valid_result(result)
    )
    correct = mathematical
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(mathematical),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
