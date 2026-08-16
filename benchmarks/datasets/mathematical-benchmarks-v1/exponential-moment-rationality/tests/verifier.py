import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
ONE_EXP = (0, 0, 0, 0)
MAX_SUBMISSION_BYTES = 1_048_576


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


def _load_bounded_submission():
    path = W / "submission.json"
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return None
    except OSError:
        return None
    try:
        return load_submission(path)
    except RecursionError:
        return None


def _add(left, right):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _scale(poly, scalar):
    return {
        exponent: coefficient * scalar
        for exponent, coefficient in poly.items()
        if coefficient * scalar
    }


def _mul(left, right):
    result = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _pow(poly, exponent):
    result = {ONE_EXP: Fraction(1)}
    for _ in range(exponent):
        result = _mul(result, poly)
    return result


def _monomial(exponents, coefficient=1):
    return {tuple(exponents): Fraction(coefficient)}


def _parse_coefficient(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _parse_polynomial(value, maximum_degree):
    if not isinstance(value, list) or not value or len(value) > 70:
        return None
    result = {}
    for term in value:
        if not isinstance(term, dict) or set(term) != {"exponents", "coefficient"}:
            return None
        exponents = term["exponents"]
        if (
            not isinstance(exponents, list)
            or len(exponents) != 4
            or any(type(x) is not int or x < 0 for x in exponents)
            or sum(exponents) > maximum_degree
        ):
            return None
        coefficient = _parse_coefficient(term["coefficient"])
        exponent = tuple(exponents)
        if coefficient is None or coefficient == 0 or exponent in result:
            return None
        result[exponent] = coefficient
    return result


def _evaluate(poly, substitutions):
    result = {}
    for exponents, coefficient in poly.items():
        term = {ONE_EXP: coefficient}
        for index, exponent in enumerate(exponents):
            term = _mul(term, _pow(substitutions[index], exponent))
        result = _add(result, term)
    return result


def _generic_moments():
    moments = []
    for degree in range(5):
        xr = _monomial((degree, 0, 1, 0))
        ys = _monomial((0, degree, 0, 1))
        moments.append(_add(xr, ys))
    return moments


def _singular_moments():
    return [_monomial((degree, 0, 1, 0), 2) for degree in range(5)]


def _formula_valid(
    formula, substitutions, target, maximum_degree, *, denominator_must_vanish=False
):
    if not isinstance(formula, dict) or set(formula) != {"numerator", "denominator"}:
        return False
    numerator = _parse_polynomial(formula["numerator"], maximum_degree)
    denominator = _parse_polynomial(formula["denominator"], maximum_degree)
    if numerator is None or denominator is None:
        return False
    evaluated_numerator = _evaluate(numerator, substitutions)
    evaluated_denominator = _evaluate(denominator, substitutions)
    if not evaluated_denominator:
        return denominator_must_vanish
    if denominator_must_vanish:
        return False
    return evaluated_numerator == _mul(evaluated_denominator, target)


def _is_nonzero_scalar_multiple(poly, template):
    if not poly or not template or set(poly) != set(template):
        return False
    scale = poly[next(iter(template))] / template[next(iter(template))]
    return scale != 0 and all(
        value == scale * template[key] for key, value in poly.items()
    )


def _generic_denominator_is_valid(poly, delta):
    """Accept delta and the advertised positive A-factor variants.

    A=e^x+e^y is strictly positive, so multiplying the generic denominator by
    a bounded power of A does not change its nonzero branch. The verifier
    still checks the exact polynomial identity after substitution.
    """
    factor = {(1, 0, 0, 0): Fraction(1)}
    candidates = [delta]
    for _ in range(1, 4):
        candidates.append(_mul(delta, factor))
        factor = _mul(factor, {(1, 0, 0, 0): Fraction(1)})
    return any(_is_nonzero_scalar_multiple(poly, candidate) for candidate in candidates)


def _result_is_valid(result, frozen):
    required = {
        "variables",
        "generic_formula",
        "singular_formula",
        "branch_partition",
        "rationality_conclusion",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or result["variables"] != ["A", "B", "C", "D"]
    ):
        return False
    maximum_degree = frozen.get("maximum_formula_degree")
    if type(maximum_degree) is not int or maximum_degree < 0:
        return False
    generic = _generic_moments()
    singular = _singular_moments()
    generic_formula = result["generic_formula"]
    if not _formula_valid(generic_formula, generic[:4], generic[4], maximum_degree):
        return False
    generic_denominator = _parse_polynomial(
        generic_formula["denominator"], maximum_degree
    )
    delta = {
        (1, 0, 1, 0): Fraction(1),
        (0, 2, 0, 0): Fraction(-1),
    }
    if generic_denominator is None or not _generic_denominator_is_valid(
        generic_denominator, delta
    ):
        return False
    if not _formula_valid(
        result["singular_formula"], singular[:4], singular[4], maximum_degree
    ):
        return False
    return (
        result["branch_partition"]
        == ["GENERIC_DENOMINATOR_NONZERO", "RANK_ONE_X_EQUALS_Y"]
        and result["rationality_conclusion"] == "E_RATIONAL_IN_BOTH_BRANCHES"
    )


def main():
    submission, frozen = _load_bounded_submission(), _load_frozen_input()
    protocol_ok = submission is not None
    math_correct = bool(
        protocol_ok and _result_is_valid(submission.get("result"), frozen)
    )
    reward = aggregate_reward(
        correctness=math_correct,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
