from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

from sympy import QQ, Matrix, Poly, groebner, symbols
from verifier_support import (
    MAX_SUBMISSION_BYTES,
    _public_submission_is_valid,
    false_verified_claim,
    is_regular_bounded_file,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE, TESTS = Path("/app"), Path("/tests")
X, Y = symbols("x y")
RATIONAL = re.compile(r"^-?(?:0|[1-9][0-9]{0,5})(?:/[1-9][0-9]{0,5})?$")
LIMITATION = "The verifier checks the frozen affine presentation and the two identified proof obligations; it does not formalize the full scheme-theoretic semicontinuity theorem."
TENSOR_REPAIR = "RIGHT_EXACTNESS_SUFFICES_RESIDUE_FIELD_NOT_FLAT_IN_GENERAL"
GLOBAL_REPAIR = "GLOBAL_FITTING_IDEAL_REPLACES_ARBITRARY_UNION"


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError, MemoryError):
        return None


def _submission() -> dict[str, Any] | None:
    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    value = _load_json(path)
    return value if isinstance(value, dict) else None


def _q(value: object) -> Fraction | None:
    if not isinstance(value, str) or RATIONAL.fullmatch(value) is None:
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


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


def _expected_fiber_points(
    points: list[object],
) -> set[tuple[Fraction, Fraction]] | None:
    expected: set[tuple[Fraction, Fraction]] = set()
    for point in points:
        if not isinstance(point, dict):
            return None
        x_value, y_value = _q(point.get("x")), _q(point.get("y"))
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


def _result_protocol(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "tensor_repair",
        "global_repair",
        "ideal_generators",
        "fiber_checks",
    }:
        return False
    generators = value.get("ideal_generators")
    if (
        not isinstance(generators, list)
        or not 2 <= len(generators) <= 6
        or any(_poly(item) is None for item in generators)
    ):
        return False
    checks = value.get("fiber_checks")
    if not isinstance(checks, list) or len(checks) != 4:
        return False
    return all(
        isinstance(check, dict)
        and set(check) == {"point", "matrix_rank", "cokernel_dimension"}
        and isinstance(check.get("point"), dict)
        and set(check["point"]) == {"x", "y"}
        and _q(check["point"].get("x")) is not None
        and _q(check["point"].get("y")) is not None
        and type(check.get("matrix_rank")) is int
        and type(check.get("cokernel_dimension")) is int
        for check in checks
    )


def _evidence_lines(result: dict[str, Any]) -> list[str] | None:
    checks = result.get("fiber_checks")
    generators = result.get("ideal_generators")
    if (
        not isinstance(checks, list)
        or len(checks) != 4
        or not isinstance(generators, list)
        or not 2 <= len(generators) <= 6
    ):
        return None
    rows: list[tuple[Fraction, Fraction, int]] = []
    for check in checks:
        if not isinstance(check, dict) or not isinstance(check.get("point"), dict):
            return None
        x_value, y_value = _q(check["point"].get("x")), _q(check["point"].get("y"))
        dimension = check.get("cokernel_dimension")
        if x_value is None or y_value is None or type(dimension) is not int:
            return None
        rows.append((x_value, y_value, dimension))
    rows.sort()
    return [
        "fiber-dimension-fitting-repair-v1",
        f"tensor-repair: {result.get('tensor_repair')}",
        f"global-repair: {result.get('global_repair')}",
        f"generator-count: {len(generators)}",
        "fiber-dimensions: "
        + ";".join(f"{x},{y}:{dimension}" for x, y, dimension in rows),
    ]


def _evidence(value: object, result: object) -> bool:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(result, dict):
        return False
    lines = _evidence_lines(result)
    if lines is None:
        return False
    encoded = ("\n".join(lines) + "\n").encode()
    path = resolve_evidence(
        value[0], expected_path="evidence/answer.txt", max_bytes=len(encoded)
    )
    try:
        return path is not None and path.read_bytes() == encoded
    except OSError:
        return False


def main() -> None:
    submission = _submission()
    data = submission if isinstance(submission, dict) else {}
    frozen_value = _load_json(TESTS / "input.json")
    frozen = frozen_value if isinstance(frozen_value, dict) else {}
    expected_value = _load_json(TESTS / "expected.json")
    expected = expected_value if isinstance(expected_value, dict) else {}
    result = data.get("result")
    input_bound = workspace_input_is_bound()
    envelope = strict_submission_contract(
        submission,
        task_id=str(expected.get("task_id")),
        conclusion=str(expected.get("conclusion")),
        allowed_assurances=frozenset({"COMPUTED"}),
        min_limitations=1,
        verification_record="forbidden",
    )
    protocol = bool(
        envelope
        and _public_submission_is_valid(submission)
        and _result_protocol(result)
    )
    math_correct = bool(_result(result, frozen))
    evidence_valid = _evidence(data.get("evidence"), result)
    scope_correct = bool(data.get("scope") == expected.get("required_scope"))
    assurance_correct = data.get("claimed_assurance") == "COMPUTED"
    completeness_correct = data.get("completeness") == "COMPLETE"
    limitations_correct = data.get("limitations") == [LIMITATION]
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = all(
        (
            input_bound,
            protocol,
            math_correct,
            evidence_valid,
            scope_correct,
            assurance_correct,
            completeness_correct,
            limitations_correct,
            not false_certification,
        )
    )
    out = Path("/logs/verifier")
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "input_binding": float(input_bound),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "completeness_accuracy": float(completeness_correct),
                "limitations_accuracy": float(limitations_correct),
                "protocol_compliance": float(protocol),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(out / "reward.json")


if __name__ == "__main__":
    main()
