"""Independent exact verifier for the additive-group action certificate."""

import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
DIMENSION = 5
LIMITATION = "The general local-finiteness theorem is not machine-formalized."


def _load_input() -> dict:
    try:
        visible = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if visible.is_symlink() or frozen.is_symlink():
            return {}
        payload = frozen.read_bytes()
        if visible.read_bytes() != payload:
            return {}
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if str(result) == value else None


def _xy_poly(value: object) -> dict[tuple[int, int], Fraction] | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 5:
        return None
    result: dict[tuple[int, int], Fraction] = {}
    previous = (-1, -1)
    for term in value:
        if not isinstance(term, dict) or set(term) != {
            "coefficient",
            "x_degree",
            "y_degree",
        }:
            return None
        x_degree, y_degree = term["x_degree"], term["y_degree"]
        coefficient = _rational(term["coefficient"])
        exponent = (x_degree, y_degree)
        if (
            not isinstance(x_degree, int)
            or isinstance(x_degree, bool)
            or not isinstance(y_degree, int)
            or isinstance(y_degree, bool)
            or x_degree < 0
            or y_degree < 0
            or x_degree + y_degree != 4
            or coefficient in (None, 0)
            or exponent <= previous
        ):
            return None
        result[exponent] = coefficient
        previous = exponent
    return result


def _t_poly(value: object) -> dict[int, Fraction] | None:
    if not isinstance(value, list) or len(value) > 5:
        return None
    result: dict[int, Fraction] = {}
    previous = -1
    for term in value:
        if not isinstance(term, dict) or set(term) != {"coefficient", "degree"}:
            return None
        degree = term["degree"]
        coefficient = _rational(term["coefficient"])
        if (
            not isinstance(degree, int)
            or isinstance(degree, bool)
            or not 0 <= degree <= 4
            or coefficient in (None, 0)
            or degree <= previous
        ):
            return None
        result[degree] = coefficient
        previous = degree
    return result


def _vector(poly: dict[tuple[int, int], Fraction]) -> list[Fraction]:
    return [poly.get((x_degree, 4 - x_degree), Fraction(0)) for x_degree in range(5)]


def _rank(matrix: list[list[Fraction]]) -> int:
    data = [row[:] for row in matrix]
    rank = 0
    for column in range(len(data[0])):
        pivot = next((r for r in range(rank, len(data)) if data[r][column]), None)
        if pivot is None:
            continue
        data[rank], data[pivot] = data[pivot], data[rank]
        scale = data[rank][column]
        data[rank] = [entry / scale for entry in data[rank]]
        for row in range(len(data)):
            if row != rank and data[row][column]:
                scale = data[row][column]
                data[row] = [
                    a - scale * b for a, b in zip(data[row], data[rank], strict=True)
                ]
        rank += 1
    return rank


def _action(
    poly: dict[tuple[int, int], Fraction],
) -> dict[tuple[int, int, int], Fraction]:
    result: dict[tuple[int, int, int], Fraction] = {}
    for (x_degree, y_degree), coefficient in poly.items():
        for t_degree in range(x_degree + 1):
            exponent = (x_degree - t_degree, y_degree + t_degree, t_degree)
            result[exponent] = result.get(
                exponent, Fraction(0)
            ) + coefficient * math.comb(x_degree, t_degree)
    return {key: value for key, value in result.items() if value}


def _represented_column(basis, matrix, column):
    result: dict[tuple[int, int, int], Fraction] = {}
    for row in range(DIMENSION):
        for (x_degree, y_degree), coefficient in basis[row].items():
            for t_degree, scalar in matrix[row][column].items():
                key = (x_degree, y_degree, t_degree)
                result[key] = result.get(key, Fraction(0)) + coefficient * scalar
    return {key: value for key, value in result.items() if value}


def _at_zero(poly: dict[int, Fraction]) -> Fraction:
    return poly.get(0, Fraction(0))


def _st_add(poly: dict[int, Fraction]) -> dict[tuple[int, int], Fraction]:
    result: dict[tuple[int, int], Fraction] = {}
    for degree, coefficient in poly.items():
        for s_degree in range(degree + 1):
            key = (s_degree, degree - s_degree)
            result[key] = result.get(key, Fraction(0)) + coefficient * math.comb(
                degree, s_degree
            )
    return {key: value for key, value in result.items() if value}


def _st_product(
    left: dict[int, Fraction], right: dict[int, Fraction]
) -> dict[tuple[int, int], Fraction]:
    return {
        (s_degree, t_degree): a * b
        for s_degree, a in left.items()
        for t_degree, b in right.items()
        if a * b
    }


def _st_sum(polys):
    result: dict[tuple[int, int], Fraction] = {}
    for poly in polys:
        for key, value in poly.items():
            result[key] = result.get(key, Fraction(0)) + value
            if not result[key]:
                del result[key]
    return result


def _certificate_valid(result: object, source: dict) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "basis",
        "f_coordinates",
        "action_matrix",
    }:
        return False
    if (
        source.get("required_basis_dimension") != DIMENSION
        or source.get("coefficient_domain") != "QQ"
    ):
        return False
    basis_raw, coordinates_raw, matrix_raw = (
        result["basis"],
        result["f_coordinates"],
        result["action_matrix"],
    )
    if not isinstance(basis_raw, list) or len(basis_raw) != DIMENSION:
        return False
    basis = [_xy_poly(poly) for poly in basis_raw]
    coordinates = (
        [_rational(value) for value in coordinates_raw]
        if isinstance(coordinates_raw, list)
        else []
    )
    if (
        any(poly is None for poly in basis)
        or len(coordinates) != DIMENSION
        or any(value is None for value in coordinates)
    ):
        return False
    basis = [poly for poly in basis if poly is not None]
    coordinates = [value for value in coordinates if value is not None]
    columns = [_vector(poly) for poly in basis]
    coefficient_matrix = [
        [columns[column][row] for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]
    if _rank(coefficient_matrix) != DIMENSION:
        return False
    frozen = _xy_poly(source.get("f"))
    if frozen is None:
        return False
    reconstructed = [
        sum(coordinates[j] * columns[j][i] for j in range(DIMENSION))
        for i in range(DIMENSION)
    ]
    if reconstructed != _vector(frozen):
        return False
    if (
        not isinstance(matrix_raw, list)
        or len(matrix_raw) != DIMENSION
        or any(not isinstance(row, list) or len(row) != DIMENSION for row in matrix_raw)
    ):
        return False
    matrix = [[_t_poly(entry) for entry in row] for row in matrix_raw]
    if any(entry is None for row in matrix for entry in row):
        return False
    matrix = [[entry for entry in row if entry is not None] for row in matrix]
    if any(
        _action(basis[j]) != _represented_column(basis, matrix, j)
        for j in range(DIMENSION)
    ):
        return False
    if any(
        _at_zero(matrix[i][j]) != Fraction(i == j)
        for i in range(DIMENSION)
        for j in range(DIMENSION)
    ):
        return False
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            left = _st_add(matrix[i][j])
            right = _st_sum(
                _st_product(matrix[i][k], matrix[k][j]) for k in range(DIMENSION)
            )
            if left != right:
                return False
    return True


def _evidence_valid(evidence: object, result: dict) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        markers = [
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        prose = " ".join(
            line
            for line in text.splitlines()
            if line.strip() and not line.startswith("RESULT_JSON:")
        )
        return (
            len(markers) == 1
            and json.loads(markers[0]) == result
            and len(prose.split()) >= 20
        )
    except (OSError, UnicodeError, ValueError):
        return False


def main() -> None:
    submission = load_submission()
    source = _load_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(contract and _certificate_valid(result, source))
    evidence_valid = bool(
        math_correct
        and isinstance(result, dict)
        and _evidence_valid(submission.get("evidence"), result)
    )
    scope_correct = bool(
        contract
        and submission.get("scope") == expected["required_scope"]
        and isinstance(submission.get("claimed_assurance"), str)
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract and LIMITATION in submission.get("limitations", [])
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
