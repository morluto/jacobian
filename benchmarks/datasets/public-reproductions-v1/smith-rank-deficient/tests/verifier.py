import json
import math
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _math(s, x, e):
    r = s.get("result", {})
    if (
        not isinstance(r, dict)
        or set(r) != {"rank", "invariant_factors"}
        or type(r.get("rank")) is not int
    ):
        return False
    ifs = r.get("invariant_factors")
    if not isinstance(ifs, list) or any(type(value) is not int for value in ifs):
        return False
    matrix = x.get("matrix")
    try:
        rows = matrix["entries"]
        row_count = matrix["row_count"]
        column_count = matrix["column_count"]
        if (
            type(row_count) is not int
            or type(column_count) is not int
            or len(rows) != row_count
            or any(len(row) != column_count for row in rows)
        ):
            return False
        entries = [[int(value) for value in row] for row in rows]
    except (KeyError, TypeError, ValueError):
        return False
    entry_gcd = math.gcd(*(abs(value) for row in entries for value in row))
    minors = [
        abs(
            entries[first][0] * entries[second][1]
            - entries[first][1] * entries[second][0]
        )
        for first in range(row_count)
        for second in range(first + 1, row_count)
    ]
    minor_gcd = math.gcd(*minors)
    rank = 2 if minor_gcd else 1 if entry_gcd else 0
    invariant_factors = (
        []
        if rank == 0
        else [entry_gcd]
        if rank == 1
        else [entry_gcd, minor_gcd // entry_gcd]
    )
    return r["rank"] == rank and ifs == invariant_factors


def main():
    s = load_submission()
    protocol_ok = s is not None
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    math_correct = _math(s, x, e) if protocol_ok else False
    correct = bool(protocol_ok and math_correct)
    good = bool(protocol_ok and witness_list_is_bound(s["witness"]))
    reward = aggregate_reward(
        correctness=correct, witness_validity=good, protocol_ok=protocol_ok
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(good),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
