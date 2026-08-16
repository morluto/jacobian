import json
from fractions import Fraction
from itertools import permutations
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _rational(value):
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


def _parse_poly(terms, variable_count):
    if not isinstance(terms, list):
        return None
    parsed = {}
    for term in terms:
        if not isinstance(term, dict):
            return None
        coefficient = _rational(term.get("coefficient"))
        exponents = term.get("exponents")
        if (
            coefficient is None
            or not isinstance(exponents, list)
            or len(exponents) != variable_count
        ):
            return None
        if any(type(exponent) is not int or exponent < 0 for exponent in exponents):
            return None
        key = tuple(exponents)
        parsed[key] = parsed.get(key, Fraction(0)) + coefficient
    return {key: value for key, value in parsed.items() if value}


def _differentiate(poly, index):
    result = {}
    for exponents, coefficient in poly.items():
        power = exponents[index]
        if power == 0:
            continue
        lowered = list(exponents)
        lowered[index] -= 1
        key = tuple(lowered)
        result[key] = result.get(key, Fraction(0)) + coefficient * power
    return {key: value for key, value in result.items() if value}


def _mul(left, right):
    result = {}
    for lexp, lcoeff in left.items():
        for rexp, rcoeff in right.items():
            key = tuple(a + b for a, b in zip(lexp, rexp, strict=True))
            result[key] = result.get(key, Fraction(0)) + lcoeff * rcoeff
    return {key: value for key, value in result.items() if value}


def _scale(poly, value):
    if not value:
        return {}
    return {key: coefficient * value for key, coefficient in poly.items()}


def _add(left, right):
    result = dict(left)
    for key, coefficient in right.items():
        total = result.get(key, Fraction(0)) + coefficient
        if total:
            result[key] = total
        else:
            result.pop(key, None)
    return result


def _det(matrix, variable_count):
    size = len(matrix)
    result = {}
    for perm in permutations(range(size)):
        inversions = sum(
            perm[i] > perm[j] for i in range(size) for j in range(i + 1, size)
        )
        term = {(0,) * variable_count: Fraction(1)}
        empty = False
        for row, col in enumerate(perm):
            if not matrix[row][col]:
                empty = True
                break
            term = _mul(term, matrix[row][col])
        if empty:
            continue
        sign = -1 if inversions % 2 else 1
        result = _add(result, _scale(term, sign))
    return result


def _expected_jacobian(x):
    mapping = x.get("map")
    if not isinstance(mapping, dict) or mapping.get("domain") != "QQ":
        return None
    variables = mapping.get("variables")
    coordinates = mapping.get("coordinates")
    if not isinstance(variables, list) or not isinstance(coordinates, list):
        return None
    if any(not isinstance(name, str) for name in variables):
        return None
    if len(variables) != len(coordinates):
        return None
    count = len(variables)
    polys = []
    for coordinate in coordinates:
        if not isinstance(coordinate, dict):
            return None
        parsed = _parse_poly(coordinate.get("terms"), count)
        if parsed is None:
            return None
        polys.append(parsed)
    jacobian = [
        [_differentiate(poly, index) for index in range(count)] for poly in polys
    ]
    return count, _det(jacobian, count)


def _is_nonzero_constant(poly, variable_count):
    zero = (0,) * variable_count
    return poly.keys() == {zero} and poly[zero] != 0


def _math(s, x):
    result = s.get("result") or {}
    if not isinstance(result, dict):
        return False
    expected = _expected_jacobian(x)
    if expected is None:
        return False
    variable_count, determinant = expected
    keller = _is_nonzero_constant(determinant, variable_count)
    if result.get("keller_condition_verified") is not keller:
        return False
    submitted_det = result.get("determinant")
    if not isinstance(submitted_det, dict):
        return False
    submitted = _parse_poly(submitted_det.get("terms"), variable_count)
    if submitted is None:
        return False
    return submitted == determinant


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
