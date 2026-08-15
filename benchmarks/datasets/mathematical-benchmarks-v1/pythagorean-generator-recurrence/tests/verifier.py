import json
import math
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")


def expected_stage(index, m, n):
    a, b, c = 2 * m * n, m * m - n * n, m * m + n * n
    return {
        "stage": index,
        "m": m,
        "n": n,
        "a": a,
        "b": b,
        "c": c,
        "q": m * m - 2 * m * n - n * n,
        "gcd": math.gcd(m, n),
        "parity_opposite": (m - n) % 2 == 1,
    }


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


def _stage_valid(stage, expected, previous_q):
    """Validate one recurrence stage against its expected values."""
    if not exact_value(stage, expected):
        return False
    if expected["gcd"] != 1 or not expected["parity_opposite"]:
        return False
    if abs(expected["q"]) != 1:
        return False
    if expected["a"] ** 2 + expected["b"] ** 2 != expected["c"] ** 2:
        return False
    if abs(expected["a"] - expected["b"]) != 1:
        return False
    return previous_q is None or expected["q"] == -previous_q


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "transform_matrix",
        "transform_determinant",
        "invariant_multiplier",
        "stages",
    }:
        return False
    if (
        not exact_value(result["transform_matrix"], [[2, 1], [1, 0]])
        or type(result["transform_determinant"]) is not int
        or result["transform_determinant"] != -1
        or type(result["invariant_multiplier"]) is not int
        or result["invariant_multiplier"] != -1
    ):
        return False
    stages = result.get("stages")
    if not isinstance(stages, list) or len(stages) != 8:
        return False
    first = stages[0]
    if not isinstance(first, dict):
        return False
    m, n = first.get("m"), first.get("n")
    if type(m) is not int or type(n) is not int:
        return False
    if not (2 <= m <= 100 and 1 <= n < m):
        return False
    previous_q = None
    for index, stage in enumerate(stages):
        expected = expected_stage(index, m, n)
        if not _stage_valid(stage, expected, previous_q):
            return False
        previous_q = expected["q"]
        m, n = 2 * m + n, m
    return True


def result_shape_valid(result):
    """Check the result has the correct keys, scalar types, and schema range
    constraints without semantic equality, so schema violations are reported
    as protocol failures rather than only as mathematical incorrectness."""
    if not isinstance(result, dict):
        return False
    if set(result) != {
        "transform_matrix",
        "transform_determinant",
        "invariant_multiplier",
        "stages",
    }:
        return False
    if (
        not isinstance(result["transform_matrix"], list)
        or len(result["transform_matrix"]) != 2
        or not all(
            isinstance(row, list)
            and len(row) == 2
            and all(type(entry) is int for entry in row)
            for row in result["transform_matrix"]
        )
    ):
        return False
    if type(result["transform_determinant"]) is not int:
        return False
    if type(result["invariant_multiplier"]) is not int:
        return False
    stages = result["stages"]
    if not isinstance(stages, list) or len(stages) != 8:
        return False
    return all(
        isinstance(s, dict)
        and set(s) == {"stage", "m", "n", "a", "b", "c", "q", "gcd", "parity_opposite"}
        and type(s["stage"]) is int
        and 0 <= s["stage"] <= 7
        and type(s["m"]) is int
        and s["m"] >= 1
        and type(s["n"]) is int
        and s["n"] >= 1
        and type(s["a"]) is int
        and s["a"] >= 1
        and type(s["b"]) is int
        and s["b"] >= 1
        and type(s["c"]) is int
        and s["c"] >= 1
        and type(s["q"]) is int
        and type(s["gcd"]) is int
        and type(s["parity_opposite"]) is bool
        for s in stages
    )


def frozen():
    return workspace_input_is_bound(W / "input.json", tests=T)


def main():
    submission = load_submission(W / "submission.json", require_input_binding=False)
    protocol_ok = submission is not None and workspace_input_is_bound(
        W / "input.json", tests=T
    )
    math_ok = bool(submission is not None and valid_result(submission.get("result")))
    reward = float(protocol_ok and math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
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
