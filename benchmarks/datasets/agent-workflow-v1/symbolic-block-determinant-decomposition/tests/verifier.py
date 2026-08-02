import json
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
LIMITATION = (
    "The unrestricted n,k theorem and the source proof are not machine-checked."
)
VARIABLE_COUNT = 8
ZERO = {}


def _load_frozen_input() -> dict:
    try:
        workspace = WORKSPACE / "input.json"
        frozen = TESTS / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, str) or len(value) > 80:
        return None
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if str(parsed) == value else None


def _parse_matrix(value: object) -> list[list[Fraction]] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    matrix = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        parsed = [_rational(entry) for entry in row]
        if any(entry is None for entry in parsed):
            return None
        matrix.append(parsed)
    return matrix


def _constant(value: Fraction | int) -> dict:
    coefficient = Fraction(value)
    return {} if coefficient == 0 else {(0,) * VARIABLE_COUNT: coefficient}


def _variable(index: int) -> dict:
    exponent = [0] * VARIABLE_COUNT
    exponent[index] = 1
    return {tuple(exponent): Fraction(1)}


def _add(left: dict, right: dict) -> dict:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _mul(left: dict, right: dict) -> dict:
    result = {}
    for left_exp, left_coefficient in left.items():
        for right_exp, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp, strict=True))
            result[exponent] = (
                result.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _scale(poly: dict, scalar: Fraction | int) -> dict:
    return _mul(poly, _constant(scalar))


def _matrix_mul(left: list[list[dict]], right: list[list[dict]]) -> list[list[dict]]:
    return [
        [
            _sum_polys(_mul(left[i][k], right[k][j]) for k in range(len(right)))
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


def _sum_polys(values) -> dict:
    result = {}
    for value in values:
        result = _add(result, value)
    return result


def _kronecker_identity(matrix: list[list[Fraction]]) -> list[list[dict]]:
    result = [[{} for _ in range(6)] for _ in range(6)]
    for i in range(3):
        for j in range(3):
            for coordinate in range(2):
                result[2 * i + coordinate][2 * j + coordinate] = _constant(matrix[i][j])
    return result


def _symbolic_blocks() -> tuple[list[list[dict]], list[list[dict]]]:
    return (
        [[_variable(0), _variable(1)], [_variable(2), _variable(3)]],
        [[_variable(4), _variable(5)], [_variable(6), _variable(7)]],
    )


def _block_add(
    left: list[list[dict]], right: list[list[dict]], scale: int
) -> list[list[dict]]:
    return [
        [_add(left[i][j], _scale(right[i][j], scale)) for j in range(2)]
        for i in range(2)
    ]


def _source_matrix(a: list[list[dict]], b: list[list[dict]]) -> list[list[dict]]:
    result = [[{} for _ in range(6)] for _ in range(6)]
    for block_row in range(3):
        for block_col in range(3):
            block = a if block_row == block_col else b
            for i in range(2):
                for j in range(2):
                    result[2 * block_row + i][2 * block_col + j] = block[i][j]
    return result


def _block_at(matrix: list[list[dict]], index: int) -> list[list[dict]]:
    return [[matrix[2 * index + i][2 * index + j] for j in range(2)] for i in range(2)]


def _is_block_diagonal(matrix: list[list[dict]]) -> bool:
    for block_row in range(3):
        for block_col in range(3):
            if block_row == block_col:
                continue
            for i in range(2):
                for j in range(2):
                    if matrix[2 * block_row + i][2 * block_col + j] != ZERO:
                        return False
    return True


def _classify_channels(
    matrix: list[list[dict]],
    common: list[list[dict]],
    difference: list[list[dict]],
) -> list[str] | None:
    channels = []
    for index in range(3):
        block = _block_at(matrix, index)
        if block == common:
            channels.append("A+2B")
        elif block == difference:
            channels.append("A-B")
        else:
            return None
    return channels


def _symbolic_certificate_valid(result: object, source: dict) -> bool:
    required = {
        "basis_change",
        "basis_change_inverse",
        "channels",
        "determinant_identity",
        "invertibility_assumption",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    family = source.get("symbolic_family", {})
    if family.get("block_dimension") != 2 or family.get("block_count") != 3:
        return False
    basis = _parse_matrix(result["basis_change"])
    inverse = _parse_matrix(result["basis_change_inverse"])
    channels = result["channels"]
    if basis is None or inverse is None or not isinstance(channels, list):
        return False
    a, b = _symbolic_blocks()
    source_matrix = _source_matrix(a, b)
    left = _matrix_mul(
        _matrix_mul(_kronecker_identity(inverse), source_matrix),
        _kronecker_identity(basis),
    )
    identity = _matrix_mul(_kronecker_identity(inverse), _kronecker_identity(basis))
    expected_identity = [
        [_constant(1 if i == j else 0) for j in range(6)] for i in range(6)
    ]
    if identity != expected_identity:
        return False
    if not _is_block_diagonal(left):
        return False
    common = _block_add(a, b, 2)
    difference = _block_add(a, b, -1)
    derived_channels = _classify_channels(left, common, difference)
    return bool(
        derived_channels is not None
        and channels == derived_channels
        and channels == ["A+2B", "A-B", "A-B"]
        and result["determinant_identity"] == "det(C)=det(A-B)^2*det(A+2B)"
        and result["invertibility_assumption"] == "NOT_REQUIRED_FOR_POLYNOMIAL_IDENTITY"
    )


def _evidence_matches(evidence: object, result: dict) -> bool:
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
        prose = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.startswith("RESULT_JSON:")
        ]
        return bool(len(markers) == 1 and json.loads(markers[0]) == result and prose)
    except (OSError, UnicodeError, ValueError):
        return False


def main() -> None:
    submission = load_submission()
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_correct = bool(contract and _symbolic_certificate_valid(result, source))
    evidence_valid = bool(
        math_correct
        and isinstance(result, dict)
        and _evidence_matches(submission["evidence"], result)
    )
    scope_correct = bool(
        contract and submission.get("scope") == expected["required_scope"]
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
    correct = bool(
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
