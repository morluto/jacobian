import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
)

W, T = Path("/app"), Path("/tests")
MAX_INPUT_BYTES = 16 * 1024 * 1024


def frozen() -> bool:
    visible = W / "input.json"
    frozen_path = T / "input.json"
    try:
        if not (
            is_regular_bounded_file(visible, max_bytes=MAX_INPUT_BYTES)
            and is_regular_bounded_file(frozen_path, max_bytes=MAX_INPUT_BYTES)
        ):
            return False
        return visible.read_bytes() == frozen_path.read_bytes()
    except (OSError, MemoryError):
        return False


def rat(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    n, d = value.get("numerator"), value.get("denominator")
    if type(n) is not int or type(d) is not int or d <= 0:
        return None
    return Fraction(n, d)


def valid(result: object) -> bool:
    keys = {
        "particular_coefficient",
        "homogeneous_base",
        "difference_delta_coefficient",
        "positive_delta_bad_parity",
        "negative_delta_bad_parity",
        "dominance_base",
        "checkpoints",
        "a0",
        "reciprocal",
    }
    if not isinstance(result, dict) or set(result) != keys:
        return False
    if not (
        rat(result["particular_coefficient"]) == Fraction(1, 9)
        and type(result["homogeneous_base"]) is int
        and result["homogeneous_base"] == -7
        and type(result["difference_delta_coefficient"]) is int
        and result["difference_delta_coefficient"] == -8
        and result["positive_delta_bad_parity"] == "EVEN"
        and result["negative_delta_bad_parity"] == "ODD"
        and rat(result["dominance_base"]) == Fraction(7, 2)
        and rat(result["a0"]) == Fraction(1, 9)
        and rat(result["reciprocal"]) == 9
    ):
        return False
    checkpoints = result["checkpoints"]
    if not isinstance(checkpoints, list) or not 4 <= len(checkpoints) <= 10:
        return False
    ns = []
    for item in checkpoints:
        if not isinstance(item, dict) or set(item) != {"n", "a_n", "difference"}:
            return False
        n = item["n"]
        if (
            type(n) is not int
            or not 1 <= n <= 30
            or rat(item["a_n"]) != Fraction(2**n, 9)
            or rat(item["difference"]) != Fraction(2**n, 9)
        ):
            return False
        ns.append(n)
    # Independently replay the closed form and recurrence beyond submitted points.
    for n in range(41):
        if Fraction(2 ** (n + 1), 9) != 2**n - 7 * Fraction(2**n, 9):
            return False
    return len(ns) == len(set(ns))


def main() -> None:
    submission = load_submission(W / "submission.json")
    math_ok = bool(
        frozen() and isinstance(submission, dict) and valid(submission.get("result"))
    )
    correct = bool(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": 1.0 if correct else 0.0,
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
