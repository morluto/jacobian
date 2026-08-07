import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
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


def _parse_polynomial(value, maximum_degree):
    if not isinstance(value, list) or not value or len(value) > 70:
        return None
    result, order = {}, []
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
        try:
            coefficient = Fraction(term["coefficient"])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        exponent = tuple(exponents)
        if (
            coefficient == 0
            or str(coefficient) != term["coefficient"]
            or exponent in result
        ):
            return None
        result[exponent] = coefficient
        order.append(exponent)
    return result if order == sorted(order) else None


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


def _evidence_matches(evidence):
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    descriptor = evidence[0]
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "sha256"}
        or descriptor.get("path") != "evidence/answer.txt"
        or not isinstance(descriptor.get("sha256"), str)
    ):
        return False
    target = W / "evidence" / "answer.txt"
    try:
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size > 1_048_576
        ):
            return False
    except OSError:
        return False
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(descriptor, expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text().casefold()
    except (OSError, UnicodeError):
        return False
    return bool(
        len(text) >= 180
        and all(
            word in text for word in ("delta", "nonzero", "generic", "singular", "x=y")
        )
        and re.search(r"\brational\b", text)
    )


def main():
    submission, frozen = _load_bounded_submission(), _load_frozen_input()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result_is_valid(submission.get("result"), frozen))
    evidence_valid = bool(
        contract and math_correct and _evidence_matches(submission.get("evidence"))
    )
    scope = submission.get("scope") if isinstance(submission, dict) else None
    scope_text = scope.casefold() if isinstance(scope, str) else ""
    scope_correct = bool(
        contract
        and all(
            term in scope_text
            for term in ("two", "atom", "exponential", "generic", "rank")
        )
        and not re.search(
            r"\b(?:not|without|exclude|excluding|except|omit)\b[^.]{0,60}\b(?:generic|rank|two[- ]atom)\b",
            scope_text,
        )
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = (
        submission.get("limitations", []) if isinstance(submission, dict) else []
    )
    limitation_correct = False
    if contract and isinstance(limitations, list):
        combined = " ".join(
            item.casefold() for item in limitations if isinstance(item, str)
        )
        negative_pattern = (
            r"\b(?:not|no|without|does not|doesn't|lacks?)\b[^.]{0,80}"
            r"\b(?:proof(?:[- ]assistant)?|machine|formal(?:ly)?)\b"
        )
        negative = re.search(negative_pattern, combined)
        remainder = re.sub(negative_pattern, "", combined)
        limitation_correct = bool(
            negative
            and re.search(r"\b(?:proof|machine|formal)", combined)
            and not re.search(
                r"\b(?:machine|formal(?:ly)?|proof[- ]assistant)\b[^.]{0,60}"
                r"\b(?:verified|checked|proof)\b",
                remainder,
            )
        )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract and math_correct and limitation_correct and not false_certification
    )
    reward = (
        0.0
        if not correct or not evidence_valid
        else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
