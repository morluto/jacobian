import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

from verifier_support import (
    load_submission,
    normalize_reward_file,
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


def _fraction_value(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        return Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _affine_in_d(value: object, constant: Fraction, coefficient: Fraction) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "constant",
        "coefficient",
        "variable",
    }:
        return False
    return (
        value["variable"] == "d"
        and _fraction_value(value["constant"]) == constant
        and _fraction_value(value["coefficient"]) == coefficient
    )


def _simplex_scope(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"kind", "variables", "sum"}:
        return False
    variables = value["variables"]
    return (
        value["kind"] == "POSITIVE_SIMPLEX"
        and isinstance(variables, list)
        and set(variables) == {"a", "b", "c"}
        and len(variables) == 3
        and _fraction_value(value["sum"]) == Fraction(1)
    )


def _direction(value: object, variable: str, target: Fraction, side: str) -> bool:
    if not isinstance(value, dict) or set(value) != {"variable", "target", "side"}:
        return False
    return (
        value["variable"] == variable
        and _fraction_value(value["target"]) == target
        and value["side"] == side
    )


def _low_regime_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "condition",
        "bound",
        "interpolation",
    }:
        return False
    condition = value["condition"]
    interpolation = value["interpolation"]
    if not isinstance(condition, dict) or set(condition) != {
        "variable",
        "left",
        "left_closed",
        "right",
        "right_closed",
    }:
        return False
    if not isinstance(interpolation, dict) or set(interpolation) != {
        "tangent_weight",
        "schur_weight",
    }:
        return False
    return bool(
        condition["variable"] == "d"
        and _fraction_value(condition["left"]) == Fraction(0)
        and condition["left_closed"] is False
        and _fraction_value(condition["right"]) == Fraction(15, 4)
        and condition["right_closed"] is True
        and _affine_in_d(value["bound"], Fraction(1, 9), Fraction(1, 27))
        and _affine_in_d(interpolation["tangent_weight"], Fraction(1), Fraction(-4, 15))
        and _affine_in_d(interpolation["schur_weight"], Fraction(0), Fraction(4, 15))
    )


def _high_regime_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "condition",
        "bound",
        "remainder_coefficient",
        "attainment",
    }:
        return False
    condition = value["condition"]
    if not isinstance(condition, dict) or set(condition) != {
        "variable",
        "relation",
        "bound",
    }:
        return False
    return bool(
        condition["variable"] == "d"
        and condition["relation"] == "GE"
        and _fraction_value(condition["bound"]) == Fraction(15, 4)
        and _fraction_value(value["bound"]) == Fraction(1, 4)
        and _affine_in_d(value["remainder_coefficient"], Fraction(-15, 4), Fraction(1))
        and value["attainment"]
        == "ATTAINED_ONLY_AT_THRESHOLD; INFIMUM_ONLY_ABOVE_THRESHOLD"
    )


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
        and _simplex_scope(value["identity_scope"])
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
        and _direction(value["parameter"], "t", Fraction(0), "RIGHT")
        and _fraction_value(value["limit"]) == Fraction(1, 4)
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
    if _fraction_value(result["transition"]) != Fraction(15, 4):
        return False

    return bool(
        _low_regime_is_valid(result["low_regime"])
        and _high_regime_is_valid(result["high_regime"])
        and result["threshold_case"]
        == "BOTH_FORMULAS_AGREE_AND_SYMMETRIC_EQUALITY_IS_ATTAINED"
        and _certificate_is_valid(result["certificate"])
        and _symmetric_witness_is_valid(result["symmetric_witness"])
        and _boundary_family_is_valid(result["boundary_family"])
        and _audit_is_valid(result["audit"])
    )


def main() -> None:
    submission = load_submission()
    submission_data = submission if isinstance(submission, dict) else {}
    source = _load_frozen_input()
    input_binding = bool(source)
    result = submission_data.get("result")
    math_ok = bool(submission is not None and _result_is_valid(result, source))
    reward = float(math_ok and input_binding)

    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
