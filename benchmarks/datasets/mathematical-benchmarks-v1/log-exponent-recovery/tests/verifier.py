import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission as load_strict_submission
from verifier_support import (
    normalize_reward_file,
)

E = Path("/tests")


def _fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError
    return Fraction(numerator, denominator)


def witness_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "value",
        "reciprocal_log_contributions",
    }:
        return False
    values = result["reciprocal_log_contributions"]
    if (
        type(result["value"]) is not int
        or not isinstance(values, dict)
        or set(values) != {"x", "y", "z", "xyz"}
    ):
        return False
    try:
        x, y, z, xyz = (_fraction(values[key]) for key in ("x", "y", "z", "xyz"))
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        x == Fraction(1, 24)
        and y == Fraction(1, 40)
        and xyz == Fraction(1, 12)
        and xyz == x + y + z
        and z != 0
        and result["value"] == z.denominator
        and z.numerator == 1
    )


def main():
    s = load_strict_submission()
    valid = isinstance(s, dict)
    result = s.get("result") if valid else None
    math_correct = bool(witness_ok(result))
    reward = float(math_correct)
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
