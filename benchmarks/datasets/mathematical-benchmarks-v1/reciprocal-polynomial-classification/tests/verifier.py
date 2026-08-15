import json
import math
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    read_evidence_json,
)

W, E = Path("/app"), Path("/tests")


def _json_equal(left, right):
    """Compare two JSON values without Python's bool/int coercion.

    Python treats ``True == 1`` as equal, so a certificate that replaces an
    integer ``1`` with boolean ``true`` would pass ``==`` despite not being an
    exact copy.  Serializing both values to canonical JSON distinguishes them.
    """
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _integer_value(value):
    """Accept any schema-valid integral JSON number while rejecting booleans.

    JSON Schema's ``integer`` type accepts numbers with a zero fractional part
    (e.g. ``8.0``), so the verifier validates mathematical integrality rather
    than requiring Python's ``int`` representation. Booleans are rejected
    because ``True == 1`` would otherwise spoof a unit coefficient.
    """
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _load_frozen_input():
    try:
        workspace, frozen = W / "input.json", E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        raw = frozen.read_bytes()
        if workspace.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _parse_terms(value):
    if not isinstance(value, list) or not value:
        return None
    parsed = {}
    for term in value:
        if not isinstance(term, dict) or set(term) != {"exponent", "coefficient"}:
            return None
        exponent = _integer_value(term["exponent"])
        coefficient = _integer_value(term["coefficient"])
        if (
            exponent is None
            or exponent < 0
            or coefficient is None
            or coefficient == 0
            or exponent in parsed
        ):
            return None
        parsed[exponent] = coefficient
    return parsed


def _add(left, right, scale=1):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, 0) + scale * coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _mul(left, right):
    result = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = left_exponent + right_exponent
            result[exponent] = (
                result.get(exponent, 0) + left_coefficient * right_coefficient
            )
            if result[exponent] == 0:
                del result[exponent]
    return result


def _result_is_valid(result, frozen):
    required = {
        "m",
        "polynomial_terms",
        "quotient_terms",
        "reverse_terms",
        "reverse_quotient_constant",
        "degree",
        "quotient_degree",
        "cleared_identity_residual",
    }
    if not isinstance(result, dict) or set(result) != required:
        return False
    m = _integer_value(result["m"])
    bounds = frozen.get("family_index_bounds", {})
    if (
        m is None
        or not isinstance(bounds, dict)
        or not bounds.get("minimum") <= m <= bounds.get("maximum")
    ):
        return False
    polynomial = _parse_terms(result["polynomial_terms"])
    quotient = _parse_terms(result["quotient_terms"])
    reverse = _parse_terms(result["reverse_terms"])
    if polynomial is None or quotient is None or reverse is None:
        return False

    expected_polynomial = {2 * j + 1: (-1) ** j for j in range(m)}
    expected_quotient = {2 * j: (-1) ** j for j in range(m)}
    degree, quotient_degree = 2 * m - 1, 2 * m - 2
    expected_reverse = {
        quotient_degree - exponent: coefficient
        for exponent, coefficient in expected_quotient.items()
    }
    reverse_constant = (-1) ** (m - 1)

    geometric_left = _mul({0: 1, 2: 1}, quotient)
    geometric_right = {0: 1, 2 * m: -((-1) ** m)}
    reciprocal = {
        -exponent: coefficient for exponent, coefficient in polynomial.items()
    }
    cleared = _add(polynomial, reciprocal)
    cleared = _add(cleared, _mul({1: 1, -1: 1}, _mul(polynomial, reciprocal)), -1)

    reverse_quotient_constant = _integer_value(result["reverse_quotient_constant"])
    degree_value = _integer_value(result["degree"])
    quotient_degree_value = _integer_value(result["quotient_degree"])
    if (
        reverse_quotient_constant is None
        or degree_value is None
        or quotient_degree_value is None
    ):
        return False
    return bool(
        frozen.get("coefficient_domain") == "ZZ"
        and frozen.get("degree_bounds") == {"minimum": 11, "maximum": 39}
        and polynomial == expected_polynomial
        and quotient == expected_quotient
        and reverse == expected_reverse
        and reverse == {e: reverse_constant * c for e, c in quotient.items()}
        and reverse_quotient_constant == reverse_constant
        and degree_value == degree
        and quotient_degree_value == quotient_degree
        and geometric_left == geometric_right
        and cleared == {}
        and result["cleared_identity_residual"] == []
    )


def main():
    submission, frozen = load_submission(), _load_frozen_input()
    math_correct = bool(
        submission and _result_is_valid(submission.get("result"), frozen)
    )
    evidence = None
    if (
        submission
        and isinstance(submission.get("witness"), list)
        and len(submission["witness"]) == 1
    ):
        evidence = read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/classification-certificate.json",
        )
    evidence_valid = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == "reciprocal-polynomial-classification"
        and _json_equal(evidence["result"], submission.get("result"))
    )
    correct = bool(math_correct and evidence_valid)
    reward = float(correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(evidence_valid),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
