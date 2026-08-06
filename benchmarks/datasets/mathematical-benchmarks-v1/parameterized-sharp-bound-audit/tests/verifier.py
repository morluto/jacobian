import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
MAX_EVIDENCE_BYTES = 1_048_576
VARS = ("a", "b", "c", "d")
ZERO = (0, 0, 0, 0)
LIMITATION = (
    "The checker replays the declared exact certificate family; it is not a "
    "general inequality prover or external proof assistant."
)


def _load_frozen_input() -> dict[str, Any]:
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


def _clean(poly: dict[tuple[int, ...], Fraction]) -> dict[tuple[int, ...], Fraction]:
    return {
        monomial: coefficient for monomial, coefficient in poly.items() if coefficient
    }


def _add(*polys: dict[tuple[int, ...], Fraction]) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for poly in polys:
        for monomial, coefficient in poly.items():
            result[monomial] = result.get(monomial, Fraction()) + coefficient
    return _clean(result)


def _scale(
    poly: dict[tuple[int, ...], Fraction], value: Fraction
) -> dict[tuple[int, ...], Fraction]:
    return _clean(
        {monomial: value * coefficient for monomial, coefficient in poly.items()}
    )


def _mul(
    left: dict[tuple[int, ...], Fraction], right: dict[tuple[int, ...], Fraction]
) -> dict[tuple[int, ...], Fraction]:
    result: dict[tuple[int, ...], Fraction] = {}
    for lm, lc in left.items():
        for rm, rc in right.items():
            monomial = tuple(x + y for x, y in zip(lm, rm, strict=True))
            result[monomial] = result.get(monomial, Fraction()) + lc * rc
    return _clean(result)


def _const(value: Fraction | int) -> dict[tuple[int, ...], Fraction]:
    return {ZERO: Fraction(value)} if value else {}


def _var(name: str) -> dict[tuple[int, ...], Fraction]:
    exponent = [0, 0, 0, 0]
    exponent[VARS.index(name)] = 1
    return {tuple(exponent): Fraction(1)}


def _pow(
    poly: dict[tuple[int, ...], Fraction], exponent: int
) -> dict[tuple[int, ...], Fraction]:
    result = _const(1)
    for _ in range(exponent):
        result = _mul(result, poly)
    return result


def _sub(
    left: dict[tuple[int, ...], Fraction], right: dict[tuple[int, ...], Fraction]
) -> dict[tuple[int, ...], Fraction]:
    return _add(left, _scale(right, Fraction(-1)))


def _certificate_identities_hold(ordering: list[str]) -> bool:
    a, b, c, d = (_var(name) for name in VARS)
    variables = [a, b, c]
    p = _add(*variables)
    q = _add(_mul(a, b), _mul(b, c), _mul(c, a))
    r = _mul(_mul(a, b), c)
    cubes = _add(*(_pow(value, 3) for value in variables))

    tangent = _add(
        *(
            _mul(
                _pow(_sub(value, _const(Fraction(1, 3))), 2),
                _add(value, _const(Fraction(2, 3))),
            )
            for value in variables
        )
    )
    tangent_gap = _sub(cubes, _const(Fraction(1, 9)))
    tangent_constraint = _scale(_sub(p, _const(1)), Fraction(1, 3))
    if _sub(_sub(tangent_gap, tangent), tangent_constraint):
        return False

    schur = _add(_pow(p, 3), _scale(r, Fraction(9)), _scale(_mul(p, q), Fraction(-4)))
    x, y, z = (_var(name) for name in ordering)
    ordered_decomposition = _add(
        _mul(_pow(_sub(x, y), 2), _add(x, y, _scale(z, Fraction(-1)))),
        _mul(z, _mul(_sub(x, z), _sub(y, z))),
    )
    if _sub(schur, ordered_decomposition):
        return False

    f = _add(cubes, _mul(d, r))
    low_gap = _sub(f, _add(_const(Fraction(1, 9)), _scale(d, Fraction(1, 27))))
    transition_gap = _sub(
        _add(cubes, _scale(r, Fraction(15, 4))), _const(Fraction(1, 4))
    )
    transition_constraint = _scale(
        _mul(
            _sub(p, _const(1)),
            _add(_pow(p, 2), p, _const(1)),
        ),
        Fraction(1, 4),
    )
    if _sub(
        _sub(transition_gap, _scale(schur, Fraction(3, 4))),
        transition_constraint,
    ):
        return False
    weight = _scale(d, Fraction(4, 15))
    interpolation = _add(
        _mul(_sub(_const(1), weight), tangent_gap),
        _mul(weight, transition_gap),
    )
    if _sub(low_gap, interpolation):
        return False
    high_gap = _sub(f, _const(Fraction(1, 4)))
    high_decomposition = _add(transition_gap, _mul(_sub(d, _const(Fraction(15, 4))), r))
    return not _sub(high_gap, high_decomposition)


def _rational(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    if math.gcd(abs(numerator), denominator) != 1:
        return None
    return Fraction(numerator, denominator)


def _certificate_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "tangent_variables",
        "schur_ordering",
        "identity_scope",
    }:
        return False
    tangent_variables = value["tangent_variables"]
    ordering = value["schur_ordering"]
    return bool(
        isinstance(tangent_variables, list)
        and set(tangent_variables) == {"a", "b", "c"}
        and len(tangent_variables) == 3
        and isinstance(ordering, list)
        and set(ordering) == {"a", "b", "c"}
        and len(ordering) == 3
        and value["identity_scope"] == "a+b+c=1; a,b,c>0"
        and _certificate_identities_hold(ordering)
    )


def _symmetric_witness_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"a", "b", "c"}:
        return False
    values = [_rational(value[name]) for name in ("a", "b", "c")]
    if values != [Fraction(1, 3)] * 3:
        return False
    exact_values = cast(list[Fraction], values)
    product = exact_values[0] * exact_values[1] * exact_values[2]
    transition_value = sum(item**3 for item in exact_values) + Fraction(15, 4) * product
    return transition_value == Fraction(1, 4)


def _boundary_family_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "vanishing_variable",
        "other_variables",
        "parameter",
        "limit",
        "attained_for_positive_parameter",
    }:
        return False
    vanishing = value["vanishing_variable"]
    others = value["other_variables"]
    return bool(
        vanishing in {"a", "b", "c"}
        and isinstance(others, list)
        and len(others) == 2
        and set(others) == {"a", "b", "c"} - {vanishing}
        and value["parameter"] == "t->0+"
        and value["limit"] == "1/4"
        and value["attained_for_positive_parameter"] is False
    )


def _audit_is_valid(value: object) -> bool:
    expected_defects = {
        "SYMMETRIC_EQUALITY_OVERCLAIMED_FOR_ALL_D",
        "BOUNDARY_INFIMUM_MISLABELED_AS_MINIMUM",
        "TRANSITION_CASE_NOT_SEPARATED",
    }
    return bool(
        isinstance(value, dict)
        and set(value) == {"frozen_explanation_status", "defects"}
        and value["frozen_explanation_status"]
        == "PARTIALLY_CORRECT_BUT_SHARPNESS_JUSTIFICATION_DEFECTIVE"
        and isinstance(value["defects"], list)
        and len(value["defects"]) == 3
        and set(value["defects"]) == expected_defects
    )


def _result_is_valid(result: object, source: dict[str, Any]) -> bool:
    required = {
        "transition",
        "low_regime",
        "high_regime",
        "threshold_case",
        "certificate",
        "symmetric_witness",
        "boundary_family",
        "audit",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or source.get("problem", {}).get("frozen_answer") != "min(1/4, 1/9+d/27)"
        or source.get("source", {}).get("row_sha256")
        != "sha256:811d71c04f1be8345f05f1b0076af9189e8851eb2d98d5b4f557492eca0699fb"
    ):
        return False
    if _rational(result["transition"]) != Fraction(15, 4):
        return False

    expected_low = {
        "condition": "0<d<=15/4",
        "bound": "1/9+d/27",
        "interpolation": {"tangent_weight": "1-4*d/15", "schur_weight": "4*d/15"},
    }
    expected_high = {
        "condition": "d>=15/4",
        "bound": "1/4",
        "remainder_coefficient": "d-15/4",
        "attainment": "ATTAINED_ONLY_AT_THRESHOLD; INFIMUM_ONLY_ABOVE_THRESHOLD",
    }
    return bool(
        result["low_regime"] == expected_low
        and result["high_regime"] == expected_high
        and result["threshold_case"]
        == "BOTH_FORMULAS_AGREE_AND_SYMMETRIC_EQUALITY_IS_ATTAINED"
        and _certificate_is_valid(result["certificate"])
        and _symmetric_witness_is_valid(result["symmetric_witness"])
        and _boundary_family_is_valid(result["boundary_family"])
        and _audit_is_valid(result["audit"])
    )


def _evidence_matches_result(evidence: object, result: dict[str, Any]) -> bool:
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(evidence, list) or not evidence:
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        if target.stat().st_size > MAX_EVIDENCE_BYTES:
            return False
        text = target.read_text()
        markers = [
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        ]
        boundary_markers = [
            line.removeprefix("BOUNDARY_FAMILY_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("BOUNDARY_FAMILY_JSON:")
        ]
        return bool(
            len(markers) == 1
            and json.loads(markers[0]) == result
            and len(boundary_markers) == 1
            and json.loads(boundary_markers[0]) == result["boundary_family"]
        )
    except (OSError, UnicodeError, ValueError):
        return False


def main() -> None:
    submission = load_submission()
    submission_data = submission if isinstance(submission, dict) else {}
    source = _load_frozen_input()
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    result = submission_data.get("result")
    math_correct = bool(contract and _result_is_valid(result, source))
    evidence_valid = bool(
        math_correct
        and isinstance(result, dict)
        and _evidence_matches_result(submission_data.get("evidence"), result)
    )
    scope_correct = bool(
        contract and submission_data.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        contract
        and submission_data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(
        contract and LIMITATION in submission_data.get("limitations", [])
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
