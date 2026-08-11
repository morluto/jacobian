import json
from pathlib import Path

from verifier_support import (
    ASSURANCE_LEVELS,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
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


def _evidence_ok(evidence, result, expected, submission):
    if not evidence:
        return False
    if set(evidence) != {"schema_version", "task_id", "result", "limitations"}:
        return False
    if evidence["schema_version"] != "1":
        return False
    if evidence["task_id"] != expected["task_id"]:
        return False
    # Validate the evidence result with the same strict matrix checks used for
    # the submission result, then require byte-for-byte equality. Python ``==``
    # admits ``True == 1`` and ``1.0 == 1``, so a certificate that replaces
    # integers with booleans or floats must be rejected by type-aware checks
    # rather than by equality alone.
    if not _result_ok(evidence["result"]):
        return False
    if evidence["result"] != result:
        return False
    return evidence["limitations"] == submission.get("limitations")


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    # Structural envelope validity accepts any assurance level, any
    # schema-allowed conclusion, and any schema-allowed completeness so
    # that a false VERIFIED claim, an abstention, or a partial submission
    # corrupts only assurance calibration / protocol, not the independent
    # correctness, evidence, and scope diagnostics.
    allowed_conclusions = frozenset({expected["conclusion"], "INSUFFICIENT_EVIDENCE"})
    allowed_completeness = frozenset({"COMPLETE", "PARTIAL", "UNKNOWN"})
    structure_valid = any(
        strict_submission_contract(
            submission,
            task_id=expected["task_id"],
            conclusion=conclusion,
            completeness=completeness,
            allowed_assurances=ASSURANCE_LEVELS,
            verification_record="forbidden",
        )
        for conclusion in allowed_conclusions
        for completeness in allowed_completeness
    )
    protocol = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    is_submission = isinstance(submission, dict)
    result = submission.get("result") if is_submission else None
    math_ok = bool(_result_ok(result) and _frozen_ok())
    evidence = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/matrix-completion.json",
            max_bytes=16 * 1024 * 1024,
        )
        if structure_valid
        and is_submission
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        else None
    )
    evidence_ok = bool(
        structure_valid and _evidence_ok(evidence, result, expected, submission)
    )
    scope_ok = bool(
        structure_valid
        and is_submission
        and submission.get("scope") == "SUBMITTED_FINITE_GF2_PATTERN_AND_COMPLETIONS"
        and submission.get("limitations")
        == [
            "SOURCE_2025_DIMENSION_OPTIMUM_NOT_ASSESSED",
            "ONLY_SUBMITTED_FINITE_COMPLETIONS_CHECKED",
        ]
    )
    assurance_ok = bool(protocol and submission.get("claimed_assurance") == "COMPUTED")
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = math_ok and evidence_ok and scope_ok and not false_cert
    reward = 0.0
    if correct and assurance_ok and protocol:
        reward = 1.0
    elif correct and protocol:
        reward = 0.9
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol),
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": reward,
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
