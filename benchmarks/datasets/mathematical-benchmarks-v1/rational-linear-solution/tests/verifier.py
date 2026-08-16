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


def q(v):
    if not isinstance(v, dict) or set(v) != {"numerator", "denominator"}:
        return None
    numerator = v["numerator"]
    denominator = v["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    input_binding = workspace_input_is_bound()
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}
    sol = r.get("solution", {})
    values = {k: q(v) for k, v in sol.items()} if isinstance(sol, dict) else {}
    math_ok = bool(
        isinstance(s, dict)
        and set(values) == set(x["variables"])
        and all(v is not None for v in values.values())
        and all(
            sum(
                (
                    Fraction(a) * values[var]
                    for a, var in zip(row["coefficients"], x["variables"], strict=True)
                )
            )
            == row["rhs"]
            for row in x["equations"]
        )
    )
    reward = float(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
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
