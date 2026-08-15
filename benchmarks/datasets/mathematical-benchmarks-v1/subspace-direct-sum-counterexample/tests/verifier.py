import itertools
import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    normalize_reward_file,
    resolve_evidence,
)
from verifier_support import load_submission as load_strict_submission

E = Path("/tests")


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def determinant(a, b, c):
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def witness_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "generators",
        "dependence_coefficients",
    }:
        return False
    vectors = result["generators"]
    coefficients = result["dependence_coefficients"]
    if (
        not isinstance(vectors, list)
        or len(vectors) != 4
        or any(not isinstance(v, list) or len(v) != 3 for v in vectors)
        or any(type(x) is not int for v in vectors for x in v)
        or any(all(x == 0 for x in v) for v in vectors)
        or not isinstance(coefficients, list)
        or len(coefficients) != 4
        or any(type(x) is not int or x == 0 for x in coefficients)
    ):
        return False
    if any(
        determinant(*(vectors[i] for i in triple)) == 0
        for triple in itertools.combinations(range(4), 3)
    ):
        return False
    return all(
        sum(coefficients[i] * vectors[i][j] for i in range(4)) == 0 for j in range(3)
    )


def main():
    s = load_strict_submission()
    valid = isinstance(s, dict)
    result = s.get("result") if valid else None
    math_correct = bool(witness_ok(result))
    ev = bool(valid and evidence_matches_result(s.get("witness"), result))
    reward = float(math_correct and ev)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(ev),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
