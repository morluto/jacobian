import itertools
import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
P = 5
EXPECTED_COLUMNS = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]


def tensor(dimension: int, products: dict[tuple[int, int], int]):
    value = [[[0] * dimension for _ in range(dimension)] for _ in range(dimension)]
    for (left, right), output in products.items():
        value[left][right][output] = 1
        value[right][left][output] = 1
    return value


def canonical_a():
    return tensor(5, {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 3): 3, (0, 4): 4, (3, 3): 4})


def canonical_b():
    return tensor(3, {(0, 0): 0, (1, 1): 1, (2, 2): 2})


def multiply(left, right, table):
    dimension = len(left)
    return [
        sum(
            left[i] * right[j] * table[i][j][k]
            for i in range(dimension)
            for j in range(dimension)
        )
        % P
        for k in range(dimension)
    ]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True)) % P


def algebra_maps(table, unit):
    dimension = len(unit)
    found = []
    for candidate in itertools.product(range(P), repeat=dimension):
        if dot(candidate, unit) != 1:
            continue
        if all(
            dot(candidate, table[i][j]) == candidate[i] * candidate[j] % P
            for i in range(dimension)
            for j in range(dimension)
        ):
            found.append(list(candidate))
    return found


def _is_int_vector(value, length, lo, hi):
    """Validate an exact-length integer vector before sorting or arithmetic."""

    return (
        isinstance(value, list)
        and len(value) == length
        and all(type(x) is int and lo <= x <= hi for x in value)
    )


def valid_morphism(columns, a_table, b_table, a_unit, b_unit):
    if not isinstance(columns, list) or len(columns) != 3:
        return False
    image_unit = [
        sum(b_unit[j] * columns[j][i] for j in range(3)) % P for i in range(5)
    ]
    if image_unit != a_unit:
        return False
    for i in range(3):
        for j in range(3):
            image_product = [
                sum(b_table[i][j][k] * columns[k][m] for k in range(3)) % P
                for m in range(5)
            ]
            if multiply(columns[i], columns[j], a_table) != image_product:
                return False
    return True


def _induced_point_map_ok(a_pts, b_pts, columns):
    induced = []
    for point in a_pts:
        pullback = [dot(point, column) for column in columns]
        if pullback not in b_pts:
            return None
        induced.append(b_pts.index(pullback))
    return induced


def _nilpotent_witness_ok(witness, a_table):
    vector = witness.get("vector") if isinstance(witness, dict) else None
    if not _is_int_vector(vector, 5, 0, P - 1) or vector == [0] * 5:
        return False
    power2 = multiply(vector, vector, a_table)
    power3 = multiply(power2, vector, a_table)
    if witness != {
        "vector": vector,
        "power2": power2,
        "power3": power3,
        "exact_order": 3,
    }:
        return False
    return not (power2 == [0] * 5 or power3 != [0] * 5)


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "field_prime",
        "a_unit",
        "b_unit",
        "a_multiplication",
        "b_multiplication",
        "morphism_columns",
        "a_points",
        "b_points",
        "induced_point_map",
        "nilpotent",
        "b_reduced",
    }:
        return False
    a_table, b_table = canonical_a(), canonical_b()
    a_unit, b_unit = [1, 1, 1, 0, 0], [1, 1, 1]
    a_points, b_points = algebra_maps(a_table, a_unit), algebra_maps(b_table, b_unit)
    if (
        result["field_prime"] != P
        or result["a_unit"] != a_unit
        or result["b_unit"] != b_unit
    ):
        return False
    if result["a_multiplication"] != a_table or result["b_multiplication"] != b_table:
        return False
    columns = result["morphism_columns"]
    if columns != EXPECTED_COLUMNS or not valid_morphism(
        columns, a_table, b_table, a_unit, b_unit
    ):
        return False
    a_pts = result["a_points"]
    b_pts = result["b_points"]
    if not (
        isinstance(a_pts, list)
        and isinstance(b_pts, list)
        and all(_is_int_vector(p, 5, 0, P - 1) for p in a_pts)
        and all(_is_int_vector(p, 3, 0, P - 1) for p in b_pts)
    ):
        return False
    if sorted(a_pts) != sorted(a_points) or sorted(b_pts) != sorted(b_points):
        return False
    induced = _induced_point_map_ok(a_pts, b_pts, columns)
    if (
        induced is None
        or result["induced_point_map"] != induced
        or sorted(induced) != [0, 1, 2]
    ):
        return False
    if not _nilpotent_witness_ok(result["nilpotent"], a_table):
        return False
    b_has_nilpotent = any(
        multiply(
            multiply(list(v), list(v), b_table),
            multiply(list(v), list(v), b_table),
            b_table,
        )
        == [0] * 3
        for v in itertools.product(range(P), repeat=3)
        if any(v)
    )
    return result["b_reduced"] is True and not b_has_nilpotent


def main():
    input_binding = workspace_input_is_bound()
    submission = load_submission(
        WORKSPACE / "submission.json", require_input_binding=False
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    mathematical = valid_result(result)
    (Path("/logs/verifier")).mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "input_binding": float(input_binding),
                "correctness": float(mathematical),
                "reward": float(input_binding and mathematical),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
