import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def q(v):
    if not isinstance(v, str):
        return None
    try:
        value = Fraction(v)
    except (ValueError, TypeError, ZeroDivisionError):
        return None
    canonical = str(value.numerator)
    if value.denominator != 1:
        canonical += f"/{value.denominator}"
    return value if v == canonical else None


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
                Fraction(a) * values[var]
                for a, var in zip(row["coefficients"], x["variables"], strict=True)
            )
            == row["rhs"]
            for row in x["equations"]
        )
    )
    ev_ok = bool(isinstance(s, dict) and witness_list_is_bound(s.get("witness")))
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=bool(input_binding and s is not None),
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
