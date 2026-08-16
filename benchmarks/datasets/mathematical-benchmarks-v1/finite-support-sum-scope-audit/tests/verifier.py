import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W, E = Path("/app"), Path("/tests")


def _rat(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    result = Fraction(value["numerator"], value["denominator"])
    return result


def _truncated(n):
    value = Fraction(1)
    for k in range(1, n + 1):
        value *= Fraction(2 * k * k + 1, k * k)
    return value / math.factorial(n)


def _result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "n",
        "tail_singletons",
        "summand_values",
        "partial_sum_lower_bound",
        "truncated_checkpoints",
        "ratio_threshold",
        "ratio_bound",
    }:
        return False
    n, tails, values = result["n"], result["tail_singletons"], result["summand_values"]
    if (
        type(n) is not int
        or not 4 <= n <= 12
        or not isinstance(tails, list)
        or not 6 <= len(tails) <= 12
        or len(set(tails)) != len(tails)
        or any(type(m) is not int or not n < m <= 100 for m in tails)
    ):
        return False
    if (
        not isinstance(values, list)
        or len(values) != len(tails)
        or any(_rat(v) != 1 for v in values)
        or result["partial_sum_lower_bound"] != len(tails)
    ):
        return False
    checks = result["truncated_checkpoints"]
    if not isinstance(checks, list) or not 3 <= len(checks) <= 8:
        return False
    ns = []
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != {"n", "value"}
            or type(check["n"]) is not int
            or not 2 <= check["n"] <= 20
            or _rat(check["value"]) != _truncated(check["n"])
        ):
            return False
        ns.append(check["n"])
    if (
        len(set(ns)) != len(ns)
        or result["ratio_threshold"] != 2
        or _rat(result["ratio_bound"]) != Fraction(3, 4)
    ):
        return False
    # The exact ratio is (2+1/(n+1)^2)/(n+1), decreasing for n>=2.
    return Fraction(2 * 3 * 3 + 1, 3 * 3 * 3) <= Fraction(3, 4)


def _frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and json.loads(raw).get("task_id")
            == "jacobian/finite-support-sum-scope-audit"
        )
    except (OSError, ValueError):
        return False


def main():
    submission = load_submission()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(_result_ok(result) and _frozen_ok())
    correct = math_ok
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
