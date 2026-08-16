import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
DEFECTS = {
    "PRIME_POWER_CLASSIFICATION_FALSE",
    "COPRIMALITY_DOES_NOT_IMPLY_DIVISIBILITY",
    "N_MINUS_ONE_NOT_DIVISOR_OF_N_SQUARED",
    "CLAIMED_N8_WITNESSES_NOT_DIVISORS",
    "PUBLISHED_COUNT_WRONG",
}


def frozen():
    return workspace_input_is_bound()


def divisors_in_interval(n):
    return [d for d in range(n // 2 + 1, n) if n * n % d == 0 and 2 * d > n]


def expected_bits():
    flags = [bool(divisors_in_interval(n)) for n in range(1, 2026)]
    packed = bytearray((len(flags) + 7) // 8)
    for i, flag in enumerate(flags):
        if flag:
            packed[i // 8] |= 1 << (i % 8)
    return flags, packed.hex()


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "corrected_count",
        "membership_bitmap_hex",
        "witnesses",
        "nonmember_counterexamples",
        "defects",
    }:
        return False
    flags, bitmap = expected_bits()
    if (
        type(r["corrected_count"]) is not int
        or r["corrected_count"] != sum(flags)
        or r["corrected_count"] != 827
        or type(r["membership_bitmap_hex"]) is not str
        or r["membership_bitmap_hex"] != bitmap
    ):
        return False
    counterexamples = r["nonmember_counterexamples"]
    defects = r["defects"]
    if (
        not isinstance(counterexamples, list)
        or len(counterexamples) != 3
        or any(type(value) is not int for value in counterexamples)
        or len(set(counterexamples)) != len(counterexamples)
        or set(counterexamples) != {3, 5, 8}
        or any(divisors_in_interval(n) for n in (3, 5, 8))
        or not isinstance(defects, list)
        or len(defects) != len(DEFECTS)
        or any(type(value) is not str for value in defects)
        or len(set(defects)) != len(defects)
        or set(defects) != DEFECTS
    ):
        return False
    witnesses = r["witnesses"]
    if not isinstance(witnesses, list) or not 10 <= len(witnesses) <= 30:
        return False
    pairs = []
    for item in witnesses:
        if not isinstance(item, dict) or set(item) != {"n", "d"}:
            return False
        n, d = item["n"], item["d"]
        if (
            type(n) is not int
            or type(d) is not int
            or not 1 <= n <= 2025
            or d not in divisors_in_interval(n)
        ):
            return False
        x, y = n + d, n + n * n // d
        if not (x < y < 2 * x and x * y == n * (x + y)):
            return False
        pairs.append((n, d))
    return len(pairs) == len(set(pairs))


def main():
    s = load_submission(W / "submission.json")
    protocol_ok = s is not None
    input_bound = frozen()
    math_ok = bool(protocol_ok and input_bound and valid(s.get("result")))
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": 1.0 if math_ok else 0.0,
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
