import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    sha256_uri,
)

W = Path("/app")
E = Path("/tests")


def _digest(path):
    return sha256_uri(path)


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
    ev_ok = False
    witness = data.get("witness")
    if protocol_ok and isinstance(witness, list) and witness:
        ev_ok = True
        for i in witness:
            if (
                not isinstance(i, dict)
                or not isinstance(i.get("path"), str)
                or not isinstance(i.get("sha256"), str)
            ):
                ev_ok = False
                continue
            p = Path(i["path"])
            t = (W / p).resolve()
            ev_ok &= (
                isinstance(i, dict)
                and not p.is_absolute()
                and p == Path("evidence/answer.txt")
                and ".." not in p.parts
                and not (W / p).is_symlink()
                and t.is_relative_to(W.resolve())
                and t.is_file()
            )
            if ev_ok:
                ev_ok &= i.get("sha256") == _digest(t)
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
