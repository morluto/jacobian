import json
from fractions import Fraction
from math import factorial
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _component(value):
    if not isinstance(value, dict):
        return None
    numerator, denominator = value.get("num"), value.get("den")
    if isinstance(numerator, str) and isinstance(denominator, str):
        if not numerator.lstrip("-").isdigit() or not denominator.isdigit():
            return None
        numerator, denominator = int(numerator), int(denominator)
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


def _complex(value):
    if not isinstance(value, dict):
        return None
    real = _component(value.get("real"))
    imag = _component(value.get("imaginary"))
    if real is None or imag is None:
        return None
    return (real, imag)


def _parse_poly(value):
    if not isinstance(value, dict):
        return None
    count = value.get("variable_count")
    terms = value.get("terms")
    if type(count) is not int or count < 1 or not isinstance(terms, list):
        return None
    parsed = {}
    for term in terms:
        if not isinstance(term, dict):
            return None
        coefficient = _complex(term.get("coefficient"))
        exponents = term.get("exponents")
        if (
            coefficient is None
            or not isinstance(exponents, list)
            or len(exponents) != count
        ):
            return None
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            return None
        key = tuple(exponents)
        parsed[key] = (
            parsed.get(key, (Fraction(0), Fraction(0)))[0] + coefficient[0],
            parsed.get(key, (Fraction(0), Fraction(0)))[1] + coefficient[1],
        )
    return parsed, count


def _mul(left, right, count):
    result = {}
    for lexp, lcoeff in left.items():
        for rexp, rcoeff in right.items():
            exponents = tuple(lexp[i] + rexp[i] for i in range(count))
            real = lcoeff[0] * rcoeff[0] - lcoeff[1] * rcoeff[1]
            imag = lcoeff[0] * rcoeff[1] + lcoeff[1] * rcoeff[0]
            current = result.get(exponents, (Fraction(0), Fraction(0)))
            result[exponents] = (current[0] + real, current[1] + imag)
    return {key: value for key, value in result.items() if value != (0, 0)}


def _pow(poly, count, order):
    result = {(0,) * count: (Fraction(1), Fraction(0))}
    for _ in range(order):
        result = _mul(result, poly, count)
    return result


def _gaussian_moment(exponents):
    moment = Fraction(1)
    for exponent in exponents:
        if exponent % 2:
            return Fraction(0)
        half = exponent // 2
        moment *= Fraction(factorial(exponent), (2**half) * factorial(half))
    return moment


def _expected_moment(x):
    parsed = _parse_poly(x.get("polynomial"))
    order = x.get("order")
    if parsed is None or type(order) is not int or order < 1:
        return None
    poly, count = parsed
    powered = _pow(poly, count, order)
    real = Fraction(0)
    imag = Fraction(0)
    for exponents, coefficient in powered.items():
        weight = _gaussian_moment(exponents)
        real += coefficient[0] * weight
        imag += coefficient[1] * weight
    return real, imag


def _math(s, x):
    result = s.get("result") or {}
    moment = result.get("moment") if isinstance(result, dict) else None
    expected = _expected_moment(x)
    if not isinstance(moment, dict) or expected is None:
        return False
    submitted_real = _component(moment.get("real"))
    submitted_imag = _component(moment.get("imaginary"))
    if submitted_real is None or submitted_imag is None:
        return False
    return submitted_real == expected[0] and submitted_imag == expected[1]


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    math_correct = _math(s, x) if protocol_ok else False
    reward = float(protocol_ok and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
