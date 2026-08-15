import json
import math
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W, T = Path("/app"), Path("/tests")


def frozen():
    try:
        return (W / "input.json").read_bytes() == (
            T / "input.json"
        ).read_bytes() and not (W / "input.json").is_symlink()
    except OSError:
        return False


def frontier():
    rows = []
    for n in range(1, 46, 2):
        budget = 2025 - n * n
        m = (math.isqrt(1 + 4 * budget) - 1) // 2
        while m * (m + 1) > budget:
            m -= 1
        rows.append(
            {
                "odd_count": n,
                "max_even_count": m,
                "minimum_sum": m * (m + 1) + n * n,
                "objective": 5 * m + 7 * n,
            }
        )
    return rows


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(exact_value(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                exact_value(value, target)
                for value, target in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def frontier_matches(submitted, expected):
    if not isinstance(submitted, list) or len(submitted) != len(expected):
        return False
    expected_by_odd = {row["odd_count"]: row for row in expected}
    seen = set()
    for row in submitted:
        if not isinstance(row, dict) or "odd_count" not in row:
            return False
        key = row["odd_count"]
        if key in seen or key not in expected_by_odd:
            return False
        seen.add(key)
        if not exact_value(row, expected_by_odd[key]):
            return False
    return True


def valid(r):
    if not isinstance(r, dict) or set(r) != {
        "even_numbers",
        "odd_numbers",
        "objective",
        "frontier",
    }:
        return False
    evens, odds = r["even_numbers"], r["odd_numbers"]
    if not isinstance(evens, list) or not isinstance(odds, list):
        return False
    if any(type(x) is not int or x <= 0 or x % 2 for x in evens) or any(
        type(x) is not int or x <= 0 or x % 2 != 1 for x in odds
    ):
        return False
    if (
        len(evens) != len(set(evens))
        or len(odds) != len(set(odds))
        or sum(evens) + sum(odds) != 2025
    ):
        return False
    objective = 5 * len(evens) + 7 * len(odds)
    rows = frontier()
    return (
        type(r["objective"]) is int
        and r["objective"] == objective == 384
        and frontier_matches(r["frontier"], rows)
        and max(row["objective"] for row in rows) == 384
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    s = load_submission(W / "submission.json")
    input_bound = frozen()
    math_ok = bool(isinstance(s, dict) and valid(s.get("result")))
    witness = s.get("witness") if isinstance(s, dict) else None
    witness_obj = (
        read_evidence_json(
            witness[0], expected_path="evidence/distinct-parity-certificate.json"
        )
        if isinstance(witness, list) and witness
        else None
    )
    witness_ok = bool(
        math_ok
        and input_bound
        and witness_obj
        and {"schema_version", "task_id", "result"} <= set(witness_obj)
        and witness_obj.get("schema_version") == "1"
        and witness_obj.get("task_id") == expected["task_id"]
        and json_value_equal(witness_obj.get("result"), s.get("result"))
    )
    correct = bool(input_bound and math_ok and witness_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(witness_ok),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
