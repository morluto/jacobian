import json
from math import gcd
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    workspace_input_is_bound,
)

W, E = Path("/app"), Path("/tests")


def _load_frozen():
    try:
        raw = (E / "input.json").read_bytes()
        if (
            (W / "input.json").is_symlink()
            or (E / "input.json").is_symlink()
            or (W / "input.json").read_bytes() != raw
        ):
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _trim(poly):
    out = list(poly)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _poly(value):
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not int for item in value)
        or value != _trim(value)
    ):
        return None
    return value


def _padd(left, right, right_scale=1):
    size = max(len(left), len(right))
    return _trim(
        [
            (left[i] if i < len(left) else 0)
            + right_scale * (right[i] if i < len(right) else 0)
            for i in range(size)
        ]
    )


def _pmul(left, right):
    out = [0] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return _trim(out)


def _canonical(num, den):
    if den == [0]:
        return None
    common = 0
    for coefficient in [*num, *den]:
        common = gcd(common, abs(coefficient))
    if common != 1 or den[-1] < 0:
        return None
    return (num, den)


def _ratfun(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    num, den = _poly(value["numerator"]), _poly(value["denominator"])
    return None if num is None or den is None else _canonical(num, den)


def _add(left, right, right_scale=1):
    ln, ld = left
    rn, rd = right
    return (_padd(_pmul(ln, rd), _pmul(rn, ld), right_scale), _pmul(ld, rd))


def _equal(left, right):
    return _pmul(left[0], right[1]) == _pmul(right[0], left[1])


def _transform(value):
    num, den = value
    return (_padd(num, den, -1), num)


def _one_plus(value):
    return _add(([1], [1]), value)


def _det3(matrix):
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _result_ok(result, frozen):
    required = {
        "orbit",
        "right_hand_sides",
        "solution_values",
        "coefficient_matrix",
        "matrix_determinant",
        "solution_at_x",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or frozen.get("required_orbit_length") != 3
        or frozen.get("coefficient_domain") != "ZZ"
    ):
        return False
    orbit_raw = result["orbit"]
    rhs_raw = result["right_hand_sides"]
    values_raw = result["solution_values"]
    if (
        not isinstance(orbit_raw, list)
        or len(orbit_raw) != 3
        or not isinstance(rhs_raw, list)
        or len(rhs_raw) != 3
        or not isinstance(values_raw, list)
        or len(values_raw) != 3
    ):
        return False
    orbit = [_ratfun(item) for item in orbit_raw]
    rhs = [_ratfun(item) for item in rhs_raw]
    values = [_ratfun(item) for item in values_raw]
    at_x = _ratfun(result["solution_at_x"])
    if any(item is None for item in [*orbit, *rhs, *values, at_x]):
        return False
    x = ([0, 1], [1])
    if not _equal(orbit[0], x):
        return False
    if any(not _equal(_transform(orbit[i]), orbit[(i + 1) % 3]) for i in range(3)):
        return False
    if any(not _equal(rhs[i], _one_plus(orbit[i])) for i in range(3)):
        return False
    if any(not _equal(_add(values[i], values[(i + 1) % 3]), rhs[i]) for i in range(3)):
        return False
    matrix = result["coefficient_matrix"]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(type(entry) is not int for entry in row)
            for row in matrix
        )
    ):
        return False
    expected_matrix = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    return bool(
        matrix == expected_matrix
        and _det3(matrix) == result["matrix_determinant"] == 2
        and _equal(values[0], at_x)
    )


def main():
    submission, frozen = load_submission(), _load_frozen()
    input_binding = workspace_input_is_bound()
    expected = json.loads((E / "expected.json").read_text())
    math_ok = bool(
        submission is not None and _result_ok(submission.get("result"), frozen)
    )
    evidence = None
    if (
        submission is not None
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
    ):
        evidence = read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/functional-equation-certificate.json",
        )
    ev_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == expected["task_id"]
        and json_value_equal(evidence["result"], submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=bool(input_binding and submission is not None),
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
