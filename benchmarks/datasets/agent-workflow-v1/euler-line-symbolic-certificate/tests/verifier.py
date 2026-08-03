import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
ZERO = {}
ONE = {(0, 0, 0): Fraction(1)}


def q(value):
    if (
        not isinstance(value, str)
        or re.fullmatch(r"-?[0-9]{1,20}(?:/[0-9]{1,20})?", value) is None
    ):
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return parsed


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def poly_add(left, right):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def poly_scale(poly, coefficient):
    return {
        exponent: coefficient * value
        for exponent, value in poly.items()
        if coefficient * value
    }


def poly_mul(left, right):
    result = {}
    for a_exp, a_coefficient in left.items():
        for b_exp, b_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(a_exp, b_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + a_coefficient * b_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def parse_poly(terms):
    if not isinstance(terms, list):
        return None
    result = {}
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            return None
        coefficient = q(term["coefficient"])
        exponents = term["exponents"]
        if (
            coefficient is None
            or not isinstance(exponents, list)
            or len(exponents) != 3
            or any(
                not isinstance(value, int) or value < 0 or value > 4
                for value in exponents
            )
        ):
            return None
        exponent = tuple(exponents)
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def parse_function(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = parse_poly(value["numerator"])
    denominator = parse_poly(value["denominator"])
    if numerator is None or denominator is None or len(denominator) != 1:
        return None
    exponent, coefficient = next(iter(denominator.items()))
    if coefficient == 0 or exponent[0] != 0 or exponent[1] != 0:
        return None
    return numerator, denominator


def rf_constant(value):
    return (poly_scale(ONE, Fraction(value)), ONE)


def rf_variable(index, scale=1):
    exponent = [0, 0, 0]
    exponent[index] = 1
    return ({tuple(exponent): Fraction(scale)}, ONE)


def rf_add(left, right):
    return (
        poly_add(
            poly_mul(left[0], right[1]),
            poly_mul(right[0], left[1]),
        ),
        poly_mul(left[1], right[1]),
    )


def rf_scale(value, coefficient):
    return (poly_scale(value[0], Fraction(coefficient)), value[1])


def rf_sub(left, right):
    return rf_add(left, rf_scale(right, -1))


def rf_mul(left, right):
    return poly_mul(left[0], right[0]), poly_mul(left[1], right[1])


def rf_is_zero(value):
    return not value[0]


def point_add(left, right):
    return rf_add(left[0], right[0]), rf_add(left[1], right[1])


def point_sub(left, right):
    return rf_sub(left[0], right[0]), rf_sub(left[1], right[1])


def point_scale(point, coefficient):
    return rf_scale(point[0], coefficient), rf_scale(point[1], coefficient)


def dot(left, right):
    return rf_add(rf_mul(left[0], right[0]), rf_mul(left[1], right[1]))


def squared_distance(left, right):
    delta = point_sub(left, right)
    return dot(delta, delta)


def point_is_zero(point):
    return rf_is_zero(point[0]) and rf_is_zero(point[1])


def parse_point(value):
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        return None
    x_coordinate = parse_function(value["x"])
    y_coordinate = parse_function(value["y"])
    if x_coordinate is None or y_coordinate is None:
        return None
    return x_coordinate, y_coordinate


def main():
    submission = load_submission()
    input_data = json.loads(next(E.glob("*input*.json")).read_text())
    expected = json.loads((E / "expected.json").read_text())
    result = submission.get("result") if isinstance(submission, dict) else None
    result = result if isinstance(result, dict) else {}
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    math_contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="optional",
    )

    coordinate_data = result.get("coordinates")
    coordinate_data = coordinate_data if isinstance(coordinate_data, dict) else {}
    points = {name: parse_point(coordinate_data.get(name)) for name in ("O", "G", "H")}
    relation = result.get("relation_coefficients")
    coefficients = (
        [q(value) for value in relation] if isinstance(relation, list) else []
    )
    input_contract = input_data == json.loads(next(E.glob("*input*.json")).read_text())

    zero = (rf_constant(0), rf_constant(0))
    a = zero
    b = (rf_variable(0, 2), rf_constant(0))
    c = (rf_variable(1, 2), rf_variable(2, 2))
    valid = bool(
        input_contract
        and math_contract
        and set(result) == {"coordinates", "relation_coefficients"}
        and isinstance(coordinate_data, dict)
        and set(coordinate_data) == {"O", "G", "H"}
        and isinstance(relation, list)
        and len(relation) == 3
        and all(point is not None for point in points.values())
        and len(coefficients) == 3
        and all(coefficient is not None for coefficient in coefficients)
        and coefficients[0] != 0
        and coefficients[0] / 2 == coefficients[1] / -3 == coefficients[2]
    )
    if valid:
        o, g, h = points["O"], points["G"], points["H"]
        assert o is not None and g is not None and h is not None
        relation_point = point_add(
            point_add(point_scale(o, coefficients[0]), point_scale(g, coefficients[1])),
            point_scale(h, coefficients[2]),
        )
        valid = (
            rf_is_zero(rf_sub(squared_distance(o, a), squared_distance(o, b)))
            and rf_is_zero(rf_sub(squared_distance(o, b), squared_distance(o, c)))
            and point_is_zero(
                point_sub(point_scale(g, 3), point_add(point_add(a, b), c))
            )
            and rf_is_zero(dot(point_sub(c, b), point_sub(h, a)))
            and rf_is_zero(dot(point_sub(c, a), point_sub(h, b)))
            and point_is_zero(relation_point)
        )

    math_correct = bool(valid)
    correct = bool(contract and math_correct)
    good = bool(contract and evidence_matches_result(submission["evidence"], result))
    scope = bool(
        contract and submission["scope"] == " ".join(expected["required_scope_terms"])
    )
    assurance = bool(
        contract and submission["claimed_assurance"] == expected["maximum_assurance"]
    )
    false = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "VERIFIED"
    )
    reward = (
        0
        if not correct or false
        else 0.7 * correct + 0.1 * good + 0.1 * scope + 0.1 * assurance
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )


if __name__ == "__main__":
    main()
