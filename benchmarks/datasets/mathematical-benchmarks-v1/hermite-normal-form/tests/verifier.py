import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(2)) for j in range(2)] for i in range(2)
    ]


def det(a):
    return a[0][0] * a[1][1] - a[0][1] * a[1][0]


def hnf(a):
    return a == [[2, 0], [0, 4]]


def integer_matrix(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(
            isinstance(row, list)
            and len(row) == 2
            and all(type(entry) is int for entry in row)
            for row in value
        )
    )


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    x = json.loads(next(E.glob("*input*.json")).read_text())
    r = data.get("result") if isinstance(data.get("result"), dict) else {}
    h = r.get("normal_form")
    u = r.get("transformation")
    math_ok = bool(
        protocol_ok
        and integer_matrix(h)
        and integer_matrix(u)
        and h == [[2, 0], [0, 4]]
        and mul(u, x["matrix"]) == h
        and abs(det(u)) == 1
        and hnf(h)
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
