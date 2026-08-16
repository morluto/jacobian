import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _fraction(value):
    return Fraction(int(value["num"]), int(value["den"]))


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
            monomial = _fraction(term["coefficient"])
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
        first = tuple(_fraction(value) for value in c["first_point"])
        second = tuple(_fraction(value) for value in c["second_point"])
        first_image = tuple(_fraction(value) for value in c["first_image"])
        second_image = tuple(_fraction(value) for value in c["second_image"])
        if len(first) != len(second) or len(first) != len(polynomial_map["variables"]):
            return False
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return False
    expected_first_image = _evaluate_map(polynomial_map, first)
    expected_second_image = _evaluate_map(polynomial_map, second)
    return (
        first != second
        and first_image == expected_first_image
        and second_image == expected_second_image
        and first_image == second_image
    )


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
