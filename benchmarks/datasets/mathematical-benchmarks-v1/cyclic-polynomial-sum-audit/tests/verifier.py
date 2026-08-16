import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

TESTS = Path("/tests")


def _fraction(value: object) -> Fraction | None:
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


def _frozen_fraction(value: object) -> Fraction | None:
    if type(value) is not str or any(marker in value for marker in ".eE"):
        return None
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def _poly_value(coefficients: list[int], value: Fraction) -> Fraction:
    result = Fraction()
    for coefficient in coefficients:
        result = result * value + coefficient
    return result


def _branch_is_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "sum",
        "pair_sum",
        "product",
        "product_consequence_lhs",
        "product_consequence_rhs",
        "residual",
    }:
        return False
    parsed = {key: _fraction(item) for key, item in value.items()}
    if any(item is None for item in parsed.values()):
        return False
    s = Fraction(3, 2)
    pair_sum = (s * s - (s + 18)) / 2
    product = s * pair_sum - 1
    lhs = (
        product * product
        - 6 * (pair_sum * pair_sum - 2 * product * s)
        + 36 * (s * s - 2 * pair_sum)
        - 216
    )
    rhs = product
    expected = {
        "sum": s,
        "pair_sum": pair_sum,
        "product": product,
        "product_consequence_lhs": lhs,
        "product_consequence_rhs": rhs,
        "residual": lhs - rhs,
    }
    return parsed == expected and lhs != rhs


def _roots_are_valid(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 2:
        return False
    expected = [
        (Fraction(-1, 2), Fraction(-1, 2), 17),
        (Fraction(-1, 2), Fraction(1, 2), 17),
    ]
    for item, (rational, coefficient, radicand) in zip(value, expected, strict=True):
        if not isinstance(item, dict) or set(item) != {
            "rational",
            "radical_coefficient",
            "radicand",
        }:
            return False
        if (
            _fraction(item["rational"]) != rational
            or _fraction(item["radical_coefficient"]) != coefficient
            or item["radicand"] != radicand
        ):
            return False
    return True


def _result_is_valid(result: object, source: dict[str, object]) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "necessary_polynomial",
        "proposed_evaluations",
        "proposed_classifications",
        "remaining_real_roots",
        "excluded_branch",
    }:
        return False
    coefficients = result["necessary_polynomial"]
    if coefficients != [2, -1, -11, 12]:
        return False
    # Primitive and square-free are checked explicitly, rather than trusted from labels.
    if math.gcd(*[abs(value) for value in coefficients]) != 1:
        return False
    proposed = source.get("adversarial_claimed_sums")
    if not isinstance(proposed, list):
        return False
    parsed = [_frozen_fraction(item) for item in proposed]
    if any(item is None for item in parsed):
        return False
    evaluations = [
        _poly_value(coefficients, item) for item in parsed if item is not None
    ]
    submitted = [_fraction(item) for item in result["proposed_evaluations"]]
    if submitted != evaluations:
        return False
    expected_classes = [
        "PASSES_NECESSARY_CONDITION" if item == 0 else "FAILS_NECESSARY_CONDITION"
        for item in evaluations
    ]
    return bool(
        result["proposed_classifications"] == expected_classes
        and _roots_are_valid(result["remaining_real_roots"])
        and _branch_is_valid(result["excluded_branch"])
    )


def main() -> None:
    _input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    source = json.loads((TESTS / "input.json").read_text())
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_correct = bool(
        isinstance(submission, dict) and _result_is_valid(result, source)
    )
    reward = float(_input_binding and math_correct)
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(_input_binding),
                "witness_validity": 1.0 if math_correct else 0.0,
                "reward": reward,
            }
        )
    )
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
