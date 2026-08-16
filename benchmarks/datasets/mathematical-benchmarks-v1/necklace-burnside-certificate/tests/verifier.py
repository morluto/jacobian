import itertools
import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
N = 16


def valid(word):
    return all(
        not (word[i] == word[(i + 1) % N] == word[(i + 2) % N]) for i in range(N)
    )


def rotation(word, k):
    return word[k:] + word[:k]


def reflection(word, k):
    return tuple(word[(k - i) % N] for i in range(N))


def derive():
    words = [word for word in itertools.product((0, 1), repeat=N) if valid(word)]
    rotations = [sum(rotation(word, k) == word for word in words) for k in range(N)]
    reflections = [sum(reflection(word, k) == word for word in words) for k in range(N)]
    representatives = sorted(
        {
            "".join(
                map(
                    str,
                    min(
                        [rotation(word, k) for k in range(N)]
                        + [reflection(word, k) for k in range(N)]
                    ),
                )
            )
            for word in words
        }
    )
    return {
        "valid_labelled_words": len(words),
        "rotation_fixed_counts": rotations,
        "reflection_fixed_counts": reflections,
        "burnside_numerator": sum(rotations + reflections),
        "orbit_count": len(representatives),
        "canonical_representatives": representatives,
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


def matches(result):
    return exact_value(result, derive())


def result_shape_valid(result):
    """Check the result has the correct keys and scalar types without
    semantic equality, so schema violations are reported as protocol
    failures rather than only as mathematical incorrectness."""
    if not isinstance(result, dict):
        return False
    if set(result) != {
        "valid_labelled_words",
        "rotation_fixed_counts",
        "reflection_fixed_counts",
        "burnside_numerator",
        "orbit_count",
        "canonical_representatives",
    }:
        return False
    if (
        type(result["valid_labelled_words"]) is not int
        or result["valid_labelled_words"] < 1
    ):
        return False
    if (
        type(result["burnside_numerator"]) is not int
        or result["burnside_numerator"] < 1
    ):
        return False
    if type(result["orbit_count"]) is not int or result["orbit_count"] < 1:
        return False
    rfc = result["rotation_fixed_counts"]
    refc = result["reflection_fixed_counts"]
    if (
        not isinstance(rfc, list)
        or len(rfc) != 16
        or not all(type(x) is int and x >= 0 for x in rfc)
    ):
        return False
    if (
        not isinstance(refc, list)
        or len(refc) != 16
        or not all(type(x) is int and x >= 0 for x in refc)
    ):
        return False
    reps = result["canonical_representatives"]
    # Validate that every entry is a hashable string before constructing the
    # set so an unhashable JSON value (e.g. {}) fails closed instead of
    # raising an uncaught TypeError before reward.json is written.
    return (
        isinstance(reps, list)
        and len(reps) >= 1
        and all(type(r) is str and len(r) == 16 and set(r) <= {"0", "1"} for r in reps)
        and len(set(reps)) == len(reps)
        and reps == sorted(reps)
    )


def frozen():
    return workspace_input_is_bound(W / "input.json", tests=T)


def main():
    submission = load_submission(W / "submission.json", require_input_binding=False)
    result = submission.get("result") if isinstance(submission, dict) else None
    input_bound = frozen()
    shape_ok = bool(isinstance(result, dict) and result_shape_valid(result))
    math_ok = bool(shape_ok and isinstance(result, dict) and matches(result))
    correct = bool(input_bound and math_ok)
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
