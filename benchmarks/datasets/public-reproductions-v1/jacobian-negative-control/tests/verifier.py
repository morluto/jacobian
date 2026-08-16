import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission, normalize_reward_file

W = Path("/app")
E = Path("/tests")


def _math(s, x):
    r = s.get("result", {})
    if not isinstance(r, dict) or set(r) != {
        "both_points_map_to_claimed_image",
        "noninvertibility_verified",
    }:
        return False
    try:
        polynomial_map = x["map"]
        first = tuple(_fraction(value) for value in x["first_point"])
        second = tuple(_fraction(value) for value in x["second_point"])
        claimed = tuple(_fraction(value) for value in x["claimed_image"])
        if len(first) != len(second) or len(first) != len(polynomial_map["variables"]):
            return False
        first_image = _evaluate_map(polynomial_map, first)
        second_image = _evaluate_map(polynomial_map, second)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
        return False
    both_map_to_claimed = first_image == claimed and second_image == claimed
    noninvertibility_verified = first != second and first_image == second_image
    return (
        r["both_points_map_to_claimed_image"] is both_map_to_claimed
        and r["noninvertibility_verified"] is noninvertibility_verified
    )


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
