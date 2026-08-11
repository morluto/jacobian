import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

E = Path("/tests")
MAX_SUBMISSION_BYTES = 1_048_576


def _is_int(value):
    """Accept JSON integers but reject Python booleans (True == 1)."""
    return type(value) is int


def _load_bounded_submission():
    path = Path("/app/submission.json")
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_SUBMISSION_BYTES
        ):
            return None
    except OSError:
        return None
    try:
        return load_submission(path)
    except RecursionError:
        return None


def _fraction(text, *, canonical=True):
    if not isinstance(text, str) or len(text) > 128:
        return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:/\d+)?|\d+\.\d+)", text):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return value if not canonical or str(value) == text else None


def _valid_levels(levels, start, end):
    if not isinstance(levels, list) or len(levels) != end - start + 1:
        return False
    rows = {}
    for row in levels:
        if not isinstance(row, dict) or not _is_int(row.get("level")):
            return False
        level = row["level"]
        if level in rows:
            return False
        rows[level] = row
    for expected_k in range(start, end + 1):
        row = rows.get(expected_k)
        if row is None:
            return False
        if not isinstance(row, dict) or set(row) != {
            "level",
            "interval_count",
            "event_mass",
            "index_start",
            "index_end",
        }:
            return False
        count = 2**expected_k
        if not (
            _is_int(row["level"])
            and row["level"] == expected_k
            and _is_int(row["interval_count"])
            and row["interval_count"] == count
            and _fraction(row["event_mass"], canonical=False) == Fraction(1, count)
            and _is_int(row["index_start"])
            and row["index_start"] == count
            and _is_int(row["index_end"])
            and row["index_end"] == 2 * count - 1
        ):
            return False
    return True


def _valid_probes(probes, start, end):
    if not isinstance(probes, list) or not 3 <= len(probes) <= 8:
        return False
    points = []
    for probe in probes:
        if not isinstance(probe, dict) or set(probe) != {"point", "hit_indices"}:
            return False
        point = _fraction(probe["point"], canonical=True)
        # Accept the full frozen space [0,1): zero is a valid probe with the
        # unique hit index 2^k at every level.
        if point is None or not 0 <= point < 1 or point in points:
            return False
        points.append(point)
        hit_indices = probe["hit_indices"]
        if not isinstance(hit_indices, list) or len(hit_indices) != end - start + 1:
            return False
        expected_hits = [
            2**k + (point.numerator * 2**k // point.denominator)
            for k in range(start, end + 1)
        ]
        if any(not _is_int(h) for h in hit_indices) or hit_indices != expected_hits:
            return False
    return True


def _valid_result(result, source):
    if not isinstance(result, dict) or set(result) != {
        "relationship",
        "levels",
        "probes",
        "probability_argument",
        "pointwise_argument",
        "research_scope",
    }:
        return False
    start = source["construction"]["level_start"]
    end = source["construction"]["level_end"]
    return bool(
        _valid_levels(result["levels"], start, end)
        and _valid_probes(result["probes"], start, end)
        and result["relationship"] == "IN_PROBABILITY_NOT_IMPLY_ALMOST_SURE"
        and result["probability_argument"]
        == {"event_mass_formula": "1/2^k", "limit": "ZERO"}
        and result["pointwise_argument"]
        == {"hit_count_per_level": 1, "miss_count_per_level": "AT_LEAST_ONE"}
        and result["research_scope"]
        == {
            "lean_theorem": "NOT_ELABORATED",
            "underlying_problem": "NOT_ADJUDICATED",
        }
    )


def _evidence_valid(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > 1_048_576:
            return False
        text = target.read_text()
    except (OSError, UnicodeError):
        return False
    # The evidence binds the exact checked result. The semantic arguments and
    # research-status boundary are closed fields in that result, rather than
    # brittle inferred properties of free-form prose.
    markers = [
        line.removeprefix("RESULT_JSON:").strip()
        for line in text.splitlines()
        if line.startswith("RESULT_JSON:")
    ]
    if len(markers) != 1:
        return False
    try:
        bound_result = json.loads(markers[0])
    except (ValueError, RecursionError):
        return False
    return isinstance(result, dict) and bound_result == result


def main():
    submission = _load_bounded_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_correct = bool(contract and _valid_result(result, source))
    evidence_valid = bool(
        contract and math_correct and _evidence_valid(data.get("evidence"), result)
    )
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract and data.get("limitations") == ["NO_OPEN_PROBLEM_OR_LEAN_CLAIM"]
    )
    false_certification = bool(
        false_verified_claim(submission, verification_record_bound=False)
        or (
            isinstance(data.get("claimed_assurance"), str)
            and data.get("claimed_assurance") not in {"UNVERIFIED", "COMPUTED"}
        )
    )
    correct = bool(
        contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    reward = 0.0 if not correct else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
