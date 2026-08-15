import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576

RESULT_KEYS = frozenset(
    {
        "error_location",
        "D",
        "P",
        "Q",
        "W",
        "gram",
        "inverse_gram",
        "lagrangian_defect",
        "naive_P",
        "naive_Q",
        "corrected_first_projection",
        "corrected_second_projection",
    }
)


def load_frozen():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def canonical(value):
    if type(value) is not str:
        raise ValueError
    number = Fraction(value)
    expected = str(number.numerator)
    if number.denominator != 1:
        expected += f"/{number.denominator}"
    if value != expected or abs(number.numerator) > 50 or number.denominator > 20:
        raise ValueError
    return number


def matrix(value, rows, columns):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError
        result.append([canonical(item) for item in row])
    return result


def transpose(a):
    return [list(row) for row in zip(*a, strict=True)]


def multiply(a, b):
    bt = transpose(b)
    return [
        [sum(x * y for x, y in zip(row, col, strict=True)) for col in bt] for row in a
    ]


def add(a, b):
    return [
        [x + y for x, y in zip(arow, brow, strict=True)]
        for arow, brow in zip(a, b, strict=True)
    ]


def negate(a):
    return [[-value for value in row] for row in a]


def inverse_2x2(a):
    determinant = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if determinant == 0:
        raise ValueError
    return [
        [a[1][1] / determinant, -a[0][1] / determinant],
        [-a[1][0] / determinant, a[0][0] / determinant],
    ]


def nonzero(a):
    return any(value != 0 for row in a for value in row)


def certificate_valid(result, frozen):
    if not isinstance(result, dict) or set(result) != RESULT_KEYS:
        return False
    if not frozen or "frozen_claim" not in frozen:
        return False
    try:
        if result.get("error_location") != "ARBITRARY_D_ASSUMED_LAGRANGIAN":
            return False
        d = matrix(result.get("D"), 4, 2)
        p = matrix(result.get("P"), 2, 2)
        q = matrix(result.get("Q"), 2, 2)
        submitted_w = matrix(result.get("W"), 4, 2)
        submitted_g = matrix(result.get("gram"), 2, 2)
        submitted_n = matrix(result.get("inverse_gram"), 2, 2)
        submitted_l = matrix(result.get("lagrangian_defect"), 2, 2)
        submitted_np = matrix(result.get("naive_P"), 2, 2)
        submitted_nq = matrix(result.get("naive_Q"), 2, 2)
        submitted_c1 = matrix(result.get("corrected_first_projection"), 2, 2)
        submitted_c2 = matrix(result.get("corrected_second_projection"), 2, 2)
        j = [
            [Fraction(value) for value in row]
            for row in frozen["frozen_claim"]["standard_symplectic_matrix"]
        ]

        dt = transpose(d)
        g = multiply(dt, d)
        n = inverse_2x2(g)
        lagrangian_defect = multiply(multiply(dt, j), d)
        jdnq = multiply(multiply(multiply(j, d), n), q)
        rebuilt_w = add(multiply(d, p), jdnq)
        naive_p = multiply(multiply(n, dt), rebuilt_w)
        naive_q = negate(multiply(multiply(dt, j), rebuilt_w))
        corrected_1 = add(
            p,
            multiply(multiply(multiply(n, lagrangian_defect), n), q),
        )
        corrected_2 = add(negate(multiply(lagrangian_defect, p)), q)
    except (ValueError, ZeroDivisionError, TypeError, KeyError):
        return False

    return bool(
        nonzero(lagrangian_defect)
        and nonzero(p)
        and nonzero(q)
        and naive_p != p
        and naive_q != q
        and rebuilt_w == submitted_w
        and g == submitted_g
        and n == submitted_n
        and lagrangian_defect == submitted_l
        and naive_p == submitted_np == corrected_1 == submitted_c1
        and naive_q == submitted_nq == corrected_2 == submitted_c2
    )


def main():
    submission = load_submission()
    frozen = load_frozen()
    source = frozen.get("source") if isinstance(frozen, dict) else {}
    source_bound = bool(
        isinstance(source, dict)
        and source.get("revision") == "86c2b07ec545c0bd37feac10d4fc03675a85a6f6"
        and source.get("row_sha256")
        == "sha256:094bc10d13dd610b5f2a17f69203641a0cc05fbca5982df06d9e07c8d189a559"
        and source.get("license") == "CC 4.0"
    )
    result = submission.get("result") if isinstance(submission, dict) else {}
    correctness = bool(source_bound and certificate_valid(result, frozen))
    reward = float(correctness)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(correctness),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
