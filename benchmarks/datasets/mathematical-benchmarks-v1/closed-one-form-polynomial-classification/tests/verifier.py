from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

ORDER = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
TARGET = [[0, 0, 0, 0, 1, -2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, -3]]


def _integer(value):
    return type(value) is int


def _canonical_fraction(value):
    if not isinstance(value, str):
        return None
    try:
        fraction = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return fraction if str(fraction) == value else None


def _valid_vector(vector):
    return (
        isinstance(vector, list)
        and len(vector) == 10
        and all(_integer(value) for value in vector)
    )


def _valid_potential(potential):
    if not isinstance(potential, dict) or set(potential) != {"terms"}:
        return False
    terms = potential["terms"]
    if not isinstance(terms, list):
        return False
    seen = set()
    for term in terms:
        if not isinstance(term, dict) or set(term) != {
            "coefficient",
            "x_power",
            "y_power",
        }:
            return False
        coefficient = _canonical_fraction(term["coefficient"])
        x_power, y_power = term["x_power"], term["y_power"]
        if (
            coefficient is None
            or not _integer(x_power)
            or not _integer(y_power)
            or not 0 <= x_power <= 4
            or not 0 <= y_power <= 4
            or coefficient == 0
            or (x_power, y_power) in seen
        ):
            return False
        seen.add((x_power, y_power))
    return True


def rank(matrix):
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for i in range(len(rows)):
            if i != pivot_row and rows[i][column]:
                scale = rows[i][column]
                rows[i] = [
                    value - scale * p
                    for value, p in zip(rows[i], rows[pivot_row], strict=True)
                ]
        pivot_row += 1
    return pivot_row


def derivative(terms, axis):
    result = {}
    for term in terms:
        coefficient = Fraction(term["coefficient"])
        x_power, y_power = term["x_power"], term["y_power"]
        power = x_power if axis == 0 else y_power
        if not power:
            continue
        key = (x_power - 1, y_power) if axis == 0 else (x_power, y_power - 1)
        result[key] = result.get(key, Fraction()) + coefficient * power
    return {key: value for key, value in result.items() if value}


def valid_result(result):
    try:
        if not isinstance(result, dict) or set(result) != {
            "constraints",
            "rank",
            "dimension",
            "basis",
            "potentials",
        }:
            return False
        constraints, basis = result["constraints"], result["basis"]
        potentials = result["potentials"]
        if (
            not isinstance(constraints, list)
            or len(constraints) != 2
            or not all(_valid_vector(row) for row in constraints)
            or not isinstance(basis, list)
            or len(basis) != 8
            or not all(_valid_vector(vector) for vector in basis)
            or not isinstance(potentials, list)
            or len(potentials) != 8
            or not all(_valid_potential(potential) for potential in potentials)
            or not _integer(result["rank"])
            or not _integer(result["dimension"])
        ):
            return False
        valid = (
            result["rank"] == 2
            and result["dimension"] == 8
            and rank(constraints) == 2
            and rank(constraints + TARGET) == 2
            and rank(basis) == 8
            and all(
                all(sum(row[i] * vector[i] for i in range(10)) == 0 for row in TARGET)
                for vector in basis
            )
        )
        for vector, potential in zip(basis, potentials, strict=True):
            expected = {
                ORDER[i]: Fraction(value) for i, value in enumerate(vector) if value
            }
            valid = valid and derivative(potential["terms"], 0) == expected
            valid = valid and derivative(potential["terms"], 1) == {
                (y, x): value for (x, y), value in expected.items()
            }
        return valid
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        return False


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    data = submission if isinstance(submission, dict) else {}
    math_correct = bool(protocol_ok and valid_result(data.get("result")))
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
