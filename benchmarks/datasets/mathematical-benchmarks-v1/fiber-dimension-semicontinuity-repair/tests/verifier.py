from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from sympy import QQ, Matrix, Poly, groebner, symbols
from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
X, Y = symbols("x y")
TENSOR_REPAIR = "RIGHT_EXACTNESS_SUFFICES_RESIDUE_FIELD_NOT_FLAT_IN_GENERAL"
GLOBAL_REPAIR = "GLOBAL_FITTING_IDEAL_REPLACES_ARBITRARY_UNION"


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None


def _q(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None


def _poly(value: object) -> Poly | None:
    if not isinstance(value, dict) or set(value) != {"terms"}:
        return None
    terms = value.get("terms")
    if not isinstance(terms, list) or not 1 <= len(terms) <= 12:
        return None
    expression = 0
    seen: set[tuple[int, int]] = set()
    for term in terms:
        if not isinstance(term, dict) or set(term) != {"coefficient", "exponents"}:
            return None
        coefficient = _q(term.get("coefficient"))
        exponents = term.get("exponents")
        if (
            coefficient is None
            or coefficient == 0
            or not isinstance(exponents, list)
            or len(exponents) != 2
            or any(type(power) is not int or not 0 <= power <= 4 for power in exponents)
        ):
            return None
        key = (exponents[0], exponents[1])
        if key in seen:
            return None
        seen.add(key)
        expression += (
            QQ(coefficient.numerator, coefficient.denominator)
            * X ** key[0]
            * Y ** key[1]
        )
    polynomial = Poly(expression, X, Y, domain=QQ)
    return polynomial if not polynomial.is_zero else None


def _frozen_matrix(frozen: dict[str, Any]) -> Matrix | None:
    presentation = frozen.get("frozen_affine_presentation")
    rows = presentation.get("map_matrix") if isinstance(presentation, dict) else None
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or any(not isinstance(row, list) or len(row) != 3 for row in rows)
    ):
        return None
    entries = []
    try:
        for row in rows:
            entries.append([Poly(entry, X, Y, domain=QQ).as_expr() for entry in row])
    except (ValueError, TypeError, KeyError):
        return None
    return Matrix(entries)


def _expected_minors(frozen: dict[str, Any]) -> list[Poly] | None:
    matrix = _frozen_matrix(frozen)
    if matrix is None:
        return None
    return [
        Poly(matrix[:, [left, right]].det(), X, Y, domain=QQ)
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]


def _same_ideal(generators: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(generators, list) or not 2 <= len(generators) <= 6:
        return False
    submitted = [_poly(item) for item in generators]
    if any(item is None for item in submitted):
        return False
    try:
        expected = _expected_minors(frozen)
        if expected is None:
            return False
        submitted_basis = groebner(
            [item.as_expr() for item in submitted], X, Y, domain=QQ
        )
        expected_basis = groebner(
            [item.as_expr() for item in expected], X, Y, domain=QQ
        )
        return all(
            submitted_basis.reduce(item.as_expr())[1] == 0 for item in expected
        ) and all(expected_basis.reduce(item.as_expr())[1] == 0 for item in submitted)
    except (ValueError, TypeError, ZeroDivisionError):
        return False


def _frozen_q(value: object) -> Fraction | None:
    if type(value) is not str or any(marker in value for marker in ".eE"):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _expected_fiber_points(
    points: list[object],
) -> set[tuple[Fraction, Fraction]] | None:
    expected: set[tuple[Fraction, Fraction]] = set()
    for point in points:
        if not isinstance(point, dict):
            return None
        x_value, y_value = _frozen_q(point.get("x")), _frozen_q(point.get("y"))
        if x_value is None or y_value is None:
            return None
        expected.add((x_value, y_value))
    return expected


def _fiber_check_ok(
    check: object,
    seen: set[tuple[Fraction, Fraction]],
    expected_points: set[tuple[Fraction, Fraction]],
    matrix: Matrix,
    target_rank: int,
) -> bool:
    if not isinstance(check, dict) or set(check) != {
        "point",
        "matrix_rank",
        "cokernel_dimension",
    }:
        return False
    point = check.get("point")
    if not isinstance(point, dict) or set(point) != {"x", "y"}:
        return False
    x_value, y_value = _q(point.get("x")), _q(point.get("y"))
    if x_value is None or y_value is None:
        return False
    key = (x_value, y_value)
    if key in seen or key not in expected_points:
        return False
    seen.add(key)
    specialized = matrix.subs({X: x_value, Y: y_value})
    rank = specialized.rank()
    if type(check.get("matrix_rank")) is not int or check["matrix_rank"] != rank:
        return False
    return (
        type(check.get("cokernel_dimension")) is int
        and check["cokernel_dimension"] == target_rank - rank
    )


def _fiber_checks(value: object, frozen: dict[str, Any]) -> bool:
    points = frozen.get("fiber_points")
    matrix = _frozen_matrix(frozen)
    presentation = frozen.get("frozen_affine_presentation")
    target_rank = (
        presentation.get("cokernel_target_rank")
        if isinstance(presentation, dict)
        else None
    )
    if (
        not isinstance(value, list)
        or not isinstance(points, list)
        or len(value) != len(points)
        or matrix is None
        or type(target_rank) is not int
    ):
        return False
    expected_points = _expected_fiber_points(points)
    if expected_points is None:
        return False
    seen: set[tuple[Fraction, Fraction]] = set()
    for check in value:
        if not _fiber_check_ok(check, seen, expected_points, matrix, target_rank):
            return False
    return seen == expected_points


def _result(value: object, frozen: dict[str, Any]) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("tensor_repair") == TENSOR_REPAIR
        and value.get("global_repair") == GLOBAL_REPAIR
        and _same_ideal(value.get("ideal_generators"), frozen)
        and _fiber_checks(value.get("fiber_checks"), frozen)
    )


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    frozen_value = _load_json(TESTS / "input.json")
    frozen = frozen_value if isinstance(frozen_value, dict) else {}
    result = data.get("result")
    input_bound = workspace_input_is_bound()
    math_correct = bool(
        isinstance(submission, dict) and input_bound and _result(result, frozen)
    )
    correct = math_correct
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": float(correct),
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
