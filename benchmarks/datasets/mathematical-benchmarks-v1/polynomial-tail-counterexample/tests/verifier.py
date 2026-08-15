import json
from fractions import Fraction
from pathlib import Path

from verifier_support import load_submission as load_strict_submission
from verifier_support import (
    normalize_reward_file,
)

E = Path("/tests")


def parse_fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError
    return Fraction(numerator, denominator)


def evaluate(coefficients, x):
    value = Fraction(0)
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def witness_ok(result):
    keys = {"p_coefficients", "q_coefficients", "p_roots", "q_roots", "x1", "x2"}
    if not isinstance(result, dict) or set(result) != keys:
        return False
    if not all(
        isinstance(result[key], list)
        for key in ("p_coefficients", "q_coefficients", "p_roots", "q_roots")
    ):
        return False
    try:
        p = [parse_fraction(x) for x in result["p_coefficients"]]
        q = [parse_fraction(x) for x in result["q_coefficients"]]
        proots = [parse_fraction(x) for x in result["p_roots"]]
        qroots = [parse_fraction(x) for x in result["q_roots"]]
        x1, x2 = parse_fraction(result["x1"]), parse_fraction(result["x2"])
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    if len(p) < 3 or len(q) < 2 or not p[0] or not q[0]:
        return False
    if not (len(p) - 1 > len(q) - 1 > 1 and p[0] > q[0] > 0):
        return False
    if len(proots) != len(p) - 1 or len(qroots) != len(q) - 1:
        return False
    if len(set(proots)) != len(proots) or len(set(qroots)) != len(qroots):
        return False
    if any(evaluate(p, root) for root in proots) or any(
        evaluate(q, root) for root in qroots
    ):
        return False
    largest_root = max(proots + qroots)
    if not (largest_root <= x1 and Fraction(0) <= x1 < x2):
        return False
    return evaluate(p, x1) - evaluate(q, x1) >= evaluate(p, x2) - evaluate(q, x2)


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
