import itertools
import json
import math
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W, T = Path("/app"), Path("/tests")


def derive():
    histogram = [0] * 5
    for permutation in itertools.permutations(range(1, 9)):
        fixed = sum(permutation[value - 1] == value for value in (2, 4, 6, 8))
        histogram[fixed] += 1
    terms = [(-1) ** j * math.comb(4, j) * math.factorial(8 - j) for j in range(5)]
    return {
        "signed_inclusion_terms": terms,
        "inclusion_sum": sum(terms),
        "exact_even_fixed_histogram": histogram,
    }


def matches(result):
    return result == derive()


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def main():
    submission = load_submission(W / "submission.json")
    protocol_ok = submission is not None
    input_bound = frozen()
    math_ok = bool(protocol_ok and input_bound and matches(submission.get("result")))
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
