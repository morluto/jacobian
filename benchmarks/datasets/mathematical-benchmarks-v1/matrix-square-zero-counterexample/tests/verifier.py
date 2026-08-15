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
