import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W, E = Path("/app"), Path("/tests")


def _matrix(value, n):
    if not isinstance(value, list) or len(value) != n:
        return None
    if any(
        not isinstance(row, list)
        or len(row) != n
        or any(type(x) is not int or x not in (0, 1) for x in row)
        for row in value
    ):
        return None
    return [row[:] for row in value]


def _rank(matrix):
    a, rows, cols = [row[:] for row in matrix], len(matrix), len(matrix[0])
    rank = 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if a[r][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        for r in range(rows):
            if r != rank and a[r][col]:
                a[r] = [x ^ y for x, y in zip(a[r], a[rank], strict=True)]
        rank += 1
    return rank


def _result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "dimension",
        "defects",
        "pattern",
        "low_rank_completion",
        "full_rank_completion",
    }:
        return False
    n = result["dimension"]
    if (
        type(n) is not int
        or not 8 <= n <= 14
        or result["defects"]
        != [
            "BAND_SUPPORT_DOES_NOT_IMPLY_SYMMETRY",
            "EXISTENCE_OF_FULL_RANK_COMPLETION_DOES_NOT_LOWER_BOUND_MINIMUM",
        ]
    ):
        return False
    pattern = _matrix(result["pattern"], n)
    low = _matrix(result["low_rank_completion"], n)
    high = _matrix(result["full_rank_completion"], n)
    if pattern is None or low is None or high is None:
        return False
    forced = [(i, j) for i in range(n) for j in range(n) if pattern[i][j]]
    band = all(abs(i - j) < 3 for i, j in forced)
    asymmetric = any(pattern[i][j] != pattern[j][i] for i in range(n) for j in range(n))
    respects = all(low[i][j] == high[i][j] == 1 for i, j in forced)
    return (
        len(forced) >= n + 1
        and band
        and asymmetric
        and respects
        and _rank(low) == 1
        and _rank(high) == n
    )


def _frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and json.loads(raw).get("task_id")
            == "jacobian/gf2-matrix-completion-quantifier-audit"
        )
    except (OSError, ValueError):
        return False


def _evidence_ok(evidence, result, expected):
    if not evidence:
        return False
    if set(evidence) != {"schema_version", "task_id", "result"}:
        return False
    if evidence["schema_version"] != "1":
        return False
    if evidence["task_id"] != expected["task_id"]:
        return False
    # Validate the evidence result with the same strict matrix checks used for
    # the submission result, then require recursive JSON equality. Python ``==``
    # admits ``True == 1`` and ``1.0 == 1``, so a certificate that replaces
    # integers with booleans or floats must be rejected by type-aware checks.
    if not _result_ok(evidence["result"]):
        return False
    return json_value_equal(evidence["result"], result)


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    protocol_ok = submission is not None
    result = submission.get("result") if protocol_ok else None
    math_ok = bool(protocol_ok and _result_ok(result) and _frozen_ok())
    evidence = (
        read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/matrix-completion.json",
            max_bytes=16 * 1024 * 1024,
        )
        if protocol_ok
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
        else None
    )
    evidence_ok = bool(_evidence_ok(evidence, result, expected))
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=evidence_ok,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(evidence_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
