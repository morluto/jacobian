"""Independent exact verifier for the additive-group action certificate."""

import json
import math
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    MAX_SUBMISSION_BYTES,
    false_verified_claim,
    is_regular_bounded_file,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
DIMENSION = 5
LIMITATION = "The general local-finiteness theorem is not machine-formalized."


def _limitations_valid(value: object) -> bool:
    if not (isinstance(value, list) and len(value) == 1 and isinstance(value[0], str)):
        return False
    text = value[0].casefold()
    if any(
        term in text
        for term in ("formally verified", "machine-proves", "general theorem is proved")
    ):
        return False
    return (
        any(
            term in text
            for term in (
                "local-finiteness",
                "local finiteness",
                "degree-four",
                "degree four",
            )
        )
        and any(term in text for term in ("theorem", "general result", "frozen action"))
        and any(
            term in text
            for term in (
                "not formal",
                "not machine",
                "not prove",
                "not verified",
                "only",
            )
        )
    )


_RATIONAL_PATTERN = re.compile(r"^-?(0|[1-9][0-9]*)(/[1-9][0-9]*)?$")
_EVIDENCE_FACTS = {
    "finite_expansion": (
        re.compile(r"\bfinite.{0,64}\bcoefficients?\b"),
        re.compile(r"\bcoefficients?.{0,64}\bfinite\b"),
        re.compile(r"\bfinite.{0,64}\bexpansion\b"),
    ),
    "insufficient_alone": (
        re.compile(r"\balone\b"),
        re.compile(r"\b(?:not|insufficient|fails?).{0,48}\b(?:prove|show|establish)\b"),
        re.compile(r"\bdoes\s+not\s+imply\b"),
    ),
    "action_law": (
        re.compile(r"\bcoaction\b"),
        re.compile(r"\bgroup[- ]law\b"),
        re.compile(r"r\s*\(\s*s\s*\+\s*t\s*\)"),
        re.compile(r"\bcomposition\b.{0,96}\b(?:law|identity)\b"),
        re.compile(r"\b(?:law|identity)\b.{0,96}\bcomposition\b"),
        re.compile(
            r"\baction\b.{0,96}\b(?:composition|group)\b"
            r".{0,32}\b(?:law|identity)\b"
        ),
    ),
    "invariance": (
        re.compile(r"\binvarian"),
        re.compile(r"\bstable\b"),
        re.compile(r"\bclosed\s+under\s+the\s+action\b"),
    ),
}
_EVIDENCE_CONTRADICTIONS = (
    re.compile(
        r"\bfinite\b.{0,64}\b(?:coefficient\s+)?expansion\b"
        r".{0,64}\b(?:alone\s+(?:proves|establishes|implies)|"
        r"by\s+itself\s+(?:proves|establishes|implies))\b"
    ),
    re.compile(r"\bcomposition\s+(?:law|identity)\b.{0,32}\b(?:fail|false)\b"),
    re.compile(r"\b(?:subspace|span)\b.{0,32}\bnot\s+invariant\b"),
)


def _load_frozen_input() -> dict:
    try:
        frozen = TESTS / "input.json"
        if frozen.is_symlink():
            return {}
        payload = frozen.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError, RecursionError, MemoryError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, str) or not _RATIONAL_PATTERN.match(value):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError, MemoryError):
        return None


def _canonical_rational(value: object) -> Fraction | None:
    result = _rational(value)
    if result is None:
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
        coefficient = _canonical_rational(term["coefficient"])
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
        coefficient = _canonical_rational(term["coefficient"])
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


def _basis_and_coordinates_ok(basis_raw, coordinates_raw):
    if not isinstance(basis_raw, list) or len(basis_raw) != DIMENSION:
        return None
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
        return None
    basis = [poly for poly in basis if poly is not None]
    coordinates = [value for value in coordinates if value is not None]
    return basis, coordinates


def _action_matrix_ok(matrix_raw):
    if (
        not isinstance(matrix_raw, list)
        or len(matrix_raw) != DIMENSION
        or any(not isinstance(row, list) or len(row) != DIMENSION for row in matrix_raw)
    ):
        return None
    matrix = [[_t_poly(entry) for entry in row] for row in matrix_raw]
    if any(entry is None for row in matrix for entry in row):
        return None
    return [[entry for entry in row if entry is not None] for row in matrix]


def _action_law_ok(basis, matrix):
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
    basis_and_coords = _basis_and_coordinates_ok(
        result["basis"], result["f_coordinates"]
    )
    if basis_and_coords is None:
        return False
    basis, coordinates = basis_and_coords
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
    matrix = _action_matrix_ok(result["action_matrix"])
    if matrix is None:
        return False
    return _action_law_ok(basis, matrix)


def _evidence_valid(evidence: object) -> bool:
    """Stream the public explanation and require both sides of its distinction."""

    if not isinstance(evidence, list) or len(evidence) != 1:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    matched = dict.fromkeys(_EVIDENCE_FACTS, False)
    contradicted = False
    carry = ""
    try:
        with target.open("r", encoding="utf-8") as stream:
            while chunk := stream.read(65_536):
                window = (carry + chunk).lower()
                contradicted = contradicted or any(
                    pattern.search(window) for pattern in _EVIDENCE_CONTRADICTIONS
                )
                for name, alternatives in _EVIDENCE_FACTS.items():
                    if not matched[name] and any(
                        pattern.search(window) for pattern in alternatives
                    ):
                        matched[name] = True
                carry = window[-256:]
    except (OSError, UnicodeError, MemoryError):
        return False
    return not contradicted and all(matched.values())


def _raw_submission() -> dict | None:
    """Parse the bounded envelope without conflating schema and math diagnostics."""

    path = WORKSPACE / "submission.json"
    if not is_regular_bounded_file(path, max_bytes=MAX_SUBMISSION_BYTES):
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError, MemoryError):
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    input_binding = workspace_input_is_bound()
    raw = _raw_submission()
    submission = load_submission(require_input_binding=False)
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = raw.get("result") if isinstance(raw, dict) else None
    math_correct = bool(_certificate_valid(result, source))
    evidence_valid = bool(
        isinstance(raw, dict) and _evidence_valid(raw.get("evidence"))
    )
    scope_correct = bool(
        isinstance(raw, dict) and raw.get("scope") == expected["required_scope"]
    )
    claimed_assurance = raw.get("claimed_assurance") if isinstance(raw, dict) else None
    assurance_correct = bool(
        isinstance(claimed_assurance, str)
        and claimed_assurance in {"UNVERIFIED", "COMPUTED"}
    )
    limitations_correct = bool(
        isinstance(raw, dict) and _limitations_valid(raw.get("limitations"))
    )
    false_certification = false_verified_claim(raw, verification_record_bound=False)
    correct = (
        math_correct
        and input_binding
        and contract
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
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "limitation_accuracy": float(limitations_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
