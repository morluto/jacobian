import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
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


def _fraction(value, *, canonical=True):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        parsed = Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    if canonical and (
        parsed.numerator != numerator or parsed.denominator != denominator
    ):
        return None
    return parsed


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


def main():
    submission = _load_bounded_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_correct = bool(isinstance(submission, dict) and _valid_result(result, source))
    correct = math_correct
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
