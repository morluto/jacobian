import json
from fractions import Fraction
from math import gcd
from pathlib import Path

from verifier_support import (
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
)

W, E = (Path("/app"), Path("/tests"))
MAX_INPUT_BYTES = 1048576
MAX_SUBMISSION_BYTES = 1048576


def _load_frozen():
    try:
        frozen_path = E / "input.json"
        workspace_path = W / "input.json"
        if not is_regular_bounded_file(
            frozen_path, max_bytes=MAX_INPUT_BYTES
        ) or not is_regular_bounded_file(workspace_path, max_bytes=MAX_INPUT_BYTES):
            return {}
        raw = frozen_path.read_bytes()
        if workspace_path.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _rat(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or (type(value["denominator"]) is not int)
        or (value["denominator"] <= 0)
        or (gcd(abs(value["numerator"]), value["denominator"]) != 1)
    ):
        return None
    return Fraction(value["numerator"], value["denominator"])


def _matrix(value):
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in value)
    ):
        return None
    parsed = [[_rat(item) for item in row] for row in value]
    return None if any(item is None for row in parsed for item in row) else parsed


def _mul(left, right):
    return [
        [
            sum((left[i][k] * right[k][j] for k in range(4)), Fraction())
            for j in range(4)
        ]
        for i in range(4)
    ]


def _transpose(matrix):
    return [[matrix[j][i] for j in range(4)] for i in range(4)]


def _det(matrix):
    work = [list(row) for row in matrix]
    value = Fraction(1)
    for column in range(len(work)):
        pivot = next(
            (row for row in range(column, len(work)) if work[row][column]), None
        )
        if pivot is None:
            return Fraction()
        if pivot != column:
            work[column], work[pivot] = (work[pivot], work[column])
            value = -value
        pivot_value = work[column][column]
        value *= pivot_value
        for row in range(column + 1, len(work)):
            scale = work[row][column] / pivot_value
            for item in range(column, len(work)):
                work[row][item] -= scale * work[column][item]
    return value


def _scalar_ok(value, frozen):
    if not isinstance(value, dict) or set(value) != {"y0", "c00_y", "m00", "objective"}:
        return False
    parsed = {key: _rat(item) for key, item in value.items()}
    source = frozen.get("scalar_inputs", {})
    expected = {key: _rat(item) for key, item in source.items()}
    return bool(
        all(item is not None for item in [*parsed.values(), *expected.values()])
        and parsed["y0"] == expected["y0"]
        and (parsed["c00_y"] == expected["c00_y"])
        and (parsed["objective"] == expected["objective"])
        and (parsed["m00"] == parsed["y0"] + parsed["c00_y"])
        and (parsed["m00"] < 0)
        and (parsed["objective"] > 0)
    )


def _positive_definite_ok(mode, certificate, matrix):
    if not isinstance(certificate, dict):
        return False
    if mode == "LDL":
        if set(certificate) != {"l", "d"}:
            return False
        lower = _matrix(certificate["l"])
        diagonal = certificate["d"]
        if lower is None or not isinstance(diagonal, list) or len(diagonal) != 4:
            return False
        pivots = [_rat(item) for item in diagonal]
        if any(item is None or item <= 0 for item in pivots):
            return False
        if any(
            lower[i][j] != (1 if i == j else 0) for i in range(4) for j in range(i, 4)
        ):
            return False
        diag_matrix = [
            [pivots[i] if i == j else Fraction() for j in range(4)] for i in range(4)
        ]
        return _mul(_mul(lower, diag_matrix), _transpose(lower)) == matrix
    if mode == "SYLVESTER":
        if set(certificate) != {"leading_principal_determinants"}:
            return False
        submitted = certificate["leading_principal_determinants"]
        if not isinstance(submitted, list) or len(submitted) != 4:
            return False
        determinants = [_rat(item) for item in submitted]
        expected = [_det([row[:size] for row in matrix[:size]]) for size in range(1, 5)]
        return (
            all(item is not None and item > 0 for item in determinants)
            and determinants == expected
        )
    return False


def _result_ok(result, frozen):
    if (
        not isinstance(result, dict)
        or set(result)
        != {"proof_mode", "scalar_replay", "positive_definite_certificate"}
        or frozen.get("full_certificate_shape")
        != {"scalar_coordinates": 1, "blocks": 6, "block_dimension": 31}
    ):
        return False
    matrix = _matrix(frozen.get("matrix"))
    return bool(
        matrix
        and _scalar_ok(result["scalar_replay"], frozen)
        and (result["proof_mode"] in frozen.get("proof_modes", []))
        and _positive_definite_ok(
            result["proof_mode"], result["positive_definite_certificate"], matrix
        )
    )


def main():
    submission_path = W / "submission.json"
    submission = (
        load_submission()
        if is_regular_bounded_file(submission_path, max_bytes=MAX_SUBMISSION_BYTES)
        else None
    )
    frozen = _load_frozen()
    protocol_ok = submission is not None
    math_correct = bool(protocol_ok and _result_ok(submission.get("result"), frozen))
    reward = float(math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps({"correctness": float(math_correct), "reward": reward})
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
