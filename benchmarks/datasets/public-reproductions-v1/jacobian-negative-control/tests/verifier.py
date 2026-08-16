import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _json_int(value):
    if type(value) is int:
        return value
    raise TypeError("rational components must be JSON integers")


def _fraction(value, *, require_json_int: bool):
    if not isinstance(value, dict) or "num" not in value or "den" not in value:
        raise TypeError("expected a rational object")
    if require_json_int:
        return Fraction(_json_int(value["num"]), _json_int(value["den"]))
    return Fraction(int(value["num"]), int(value["den"]))


def _point(values, *, require_json_int: bool):
    if not isinstance(values, list):
        raise TypeError("expected a point array")
    return tuple(
        _fraction(value, require_json_int=require_json_int) for value in values
    )


def _evaluate_map(polynomial_map, point):
    if polynomial_map.get("domain") != "QQ":
        raise ValueError("the verifier only accepts the frozen QQ map")
    coordinates = polynomial_map["coordinates"]
    if len(coordinates) != len(point):
        raise ValueError("map dimension does not match the point")
    values = []
    for coordinate in coordinates:
        total = Fraction(0)
        for term in coordinate["terms"]:
            exponents = term["exponents"]
            if len(exponents) != len(point) or any(
                type(exponent) is not int or exponent < 0 for exponent in exponents
            ):
                raise ValueError("invalid monomial exponents")
            monomial = _fraction(term["coefficient"], require_json_int=False)
            for value, exponent in zip(point, exponents, strict=True):
                monomial *= value**exponent
            total += monomial
        values.append(total)
    return tuple(values)


def _math(s, x):
    r = s.get("result", {})
    if not isinstance(r, dict) or set(r) != {"collision"}:
        return False
    c = r.get("collision")
    if not isinstance(c, dict):
        return False
    try:
        polynomial_map = x["map"]
        first = _point(c["first_point"], require_json_int=True)
        second = _point(c["second_point"], require_json_int=True)
        first_image = _point(c["first_image"], require_json_int=True)
        second_image = _point(c["second_image"], require_json_int=True)
        frozen_first = _point(x["first_point"], require_json_int=False)
        frozen_second = _point(x["second_point"], require_json_int=False)
        if len(first) != len(second) or len(first) != len(polynomial_map["variables"]):
            return False
        if {first, second} != {frozen_first, frozen_second}:
            return False
        expected_first_image = _evaluate_map(polynomial_map, first)
        expected_second_image = _evaluate_map(polynomial_map, second)
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        ZeroDivisionError,
        OverflowError,
    ):
        return False
    return (
        first != second
        and first_image == expected_first_image
        and second_image == expected_second_image
        and first_image == second_image
    )


def main():
    destination = Path("/logs/verifier/reward.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        s = load_submission()
        protocol_ok = s is not None
        x = json.loads(next(E.glob("*input*.json")).read_text())
        math_correct = _math(s, x) if protocol_ok else False
    except Exception:
        protocol_ok = False
        math_correct = False
    destination.write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(protocol_ok and math_correct),
            }
        )
    )
    normalize_reward_file(destination)


if __name__ == "__main__":
    main()
