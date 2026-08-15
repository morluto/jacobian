import json
from pathlib import Path

from verifier_support import load_submission as load_strict_submission
from verifier_support import (
    normalize_reward_file,
)

E = Path("/tests")


def matrix_ok(result):
    if not isinstance(result, dict) or set(result) != {"matrix"}:
        return False
    a = result["matrix"]
    if (
        not isinstance(a, list)
        or len(a) != 2
        or any(not isinstance(row, list) or len(row) != 2 for row in a)
        or any(type(x) is not int for row in a for x in row)
    ):
        return False
    if all(x == 0 for row in a for x in row) or a[0][1] == a[1][0]:
        return False
    square = [
        [sum(a[i][k] * a[k][j] for k in range(2)) for j in range(2)] for i in range(2)
    ]
    return square == [[0, 0], [0, 0]]


def main():
    s = load_strict_submission()
    valid = isinstance(s, dict)
    result = s.get("result") if valid else None
    math_correct = bool(matrix_ok(result))
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
