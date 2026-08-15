import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def q(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        parsed = Fraction(numerator, denominator)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    return (
        parsed
        if parsed.numerator == numerator and parsed.denominator == denominator
        else None
    )


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    input_binding = workspace_input_is_bound()
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}

    n = x["domain_size"]
    self_hit = Fraction(2 * n - 1, n * n)
    other_hit = Fraction(n - 1, n * n)
    squared_sum = sum(
        (target - source) ** 2
        for source in range(1, n + 1)
        for target in range(1, n + 1)
    )
    expectation = other_hit * squared_sum
    math_ok = bool(
        isinstance(s, dict)
        and set(r)
        == {
            "self_hit_probability",
            "other_hit_probability",
            "ordered_squared_difference_sum",
            "expected_value",
        }
        and type(r.get("ordered_squared_difference_sum")) is int
        and q(r.get("self_hit_probability")) == self_hit
        and q(r.get("other_hit_probability")) == other_hit
        and r.get("ordered_squared_difference_sum") == squared_sum
        and q(r.get("expected_value")) == expectation
        and expectation != 2025
    )
    reward = float(math_ok and input_binding)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
