import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

T = Path("/tests")


def q(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        return None
    try:
        parsed = Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "base",
        "count_formula",
        "levels",
        "lower_density",
        "upper_density",
    }:
        return False
    b = result.get("base")
    formula = result.get("count_formula")
    if (
        type(b) is not int
        or b not in range(2, 10)
        or not isinstance(formula, dict)
        or set(formula)
        != {
            "base_variable",
            "level_variable",
            "numerator_exponent_coefficient",
            "numerator_exponent_offset",
            "numerator_constant",
            "denominator_offset",
        }
        or formula.get("base_variable") != "b"
        or formula.get("level_variable") != "m"
        or type(formula.get("numerator_exponent_coefficient")) is not int
        or type(formula.get("numerator_exponent_offset")) is not int
        or type(formula.get("numerator_constant")) is not int
        or type(formula.get("denominator_offset")) is not int
        or formula["denominator_offset"] + b == 0
    ):
        return False
    expected = {}
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        exponent = (
            formula["numerator_exponent_coefficient"] * m
            + formula["numerator_exponent_offset"]
        )
        if exponent < 0:
            return False
        formula_count = (b**exponent + formula["numerator_constant"]) // (
            b + formula["denominator_offset"]
        )
        if formula_count != count:
            return False
        expected[m] = (
            high,
            low,
            count,
            Fraction(count, high),
            Fraction(count, low),
        )
    levels = result.get("levels")
    if not isinstance(levels, list) or len(levels) != 8:
        return False
    submitted = {}
    for row in levels:
        if not isinstance(row, dict):
            return False
        level = row.get("level")
        included = q(row.get("included_density"))
        excluded = q(row.get("excluded_density"))
        if (
            type(level) is not int
            or level in submitted
            or type(row.get("included_endpoint")) is not int
            or type(row.get("excluded_endpoint")) is not int
            or type(row.get("cumulative_count")) is not int
            or included is None
            or excluded is None
        ):
            return False
        submitted[level] = (
            row["included_endpoint"],
            row["excluded_endpoint"],
            row["cumulative_count"],
            included,
            excluded,
        )
    return (
        submitted == expected
        and q(result.get("lower_density")) == Fraction(1, b + 1)
        and q(result.get("upper_density")) == Fraction(b, b + 1)
    )


def main():
    raw = load_submission_raw(require_input_binding=False)
    input_binding = workspace_input_is_bound()
    contract = submission_matches_public_schema(raw)
    r = raw.get("result") if isinstance(raw, dict) else None
    math_ok = valid_result(r)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "input_binding": float(input_binding),
                "correctness": float(math_ok),
                "reward": float(contract and input_binding and math_ok),
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
