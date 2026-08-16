import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def _maximum_size(n):
    edges = [(a, b) for a in range(1, n + 1) for b in range(a + 1, n + 1) if a + b <= n]
    best = 0

    def search(index, used, sums, count):
        nonlocal best
        if count + (len(edges) - index) <= best:
            return
        if index == len(edges):
            best = max(best, count)
            return
        a, b = edges[index]
        total = a + b
        if a not in used and b not in used and total not in sums:
            search(index + 1, used | {a, b}, sums | {total}, count + 1)
        search(index + 1, used, sums, count)

    search(0, set(), set(), 0)
    return best


def _valid(result, source):
    if not isinstance(result, dict) or set(result) != {"pairs"}:
        return False
    n = source.get("n")
    pairs = result.get("pairs")
    if not isinstance(n, int) or not isinstance(pairs, list):
        return False
    used = set()
    actual_sums = []
    previous = None
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool) for value in pair
            )
        ):
            return False
        a, b = pair
        if (
            not (1 <= a < b <= n)
            or a in used
            or b in used
            or (previous is not None and pair <= previous)
        ):
            return False
        used.update(pair)
        actual_sums.append(a + b)
        previous = pair
    return bool(
        len(set(actual_sums)) == len(actual_sums)
        and all(total <= n for total in actual_sums)
        and len(pairs) == _maximum_size(n)
    )


def main():
    submission = load_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    protocol_ok = submission is not None
    math_correct = bool(protocol_ok and _valid(submission.get("result"), source))
    reward = float(protocol_ok and math_correct)
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
