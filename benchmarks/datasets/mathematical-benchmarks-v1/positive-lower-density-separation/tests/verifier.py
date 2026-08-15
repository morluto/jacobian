import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission_raw,
    normalize_reward_file,
    submission_matches_public_schema,
    workspace_input_is_bound,
)

T = Path("/tests")


def q(text):
    if (
        not isinstance(text, str)
        or re.fullmatch(r"(?:0|1|[1-9][0-9]*/[1-9][0-9]*)", text) is None
    ):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def valid_result(result):
    if not isinstance(result, dict) or set(result) != {
        "base",
        "family",
        "count_formula",
        "levels",
        "lower_density",
        "upper_density",
        "lower_density_positive",
        "natural_density_exists",
        "semantic_relation",
    }:
        return False
    b = result.get("base")
    if (
        type(b) is not int
        or b not in range(2, 10)
        or result.get("family") != "ALTERNATING_GEOMETRIC_BLOCKS"
        or result.get("count_formula") != "(b^(2m+2)-1)/(b+1)"
    ):
        return False
    expected = []
    for m in range(8):
        high, low = b ** (2 * m + 1), b ** (2 * m + 2)
        count = (low - 1) // (b + 1)
        expected.append(
            {
                "level": m,
                "included_endpoint": high,
                "excluded_endpoint": low,
                "cumulative_count": count,
                "included_density": str(Fraction(count, high)),
                "excluded_density": str(Fraction(count, low)),
            }
        )
    levels = result.get("levels")
    exact_integer_levels = bool(
        isinstance(levels, list)
        and len(levels) == 8
        and all(
            isinstance(row, dict)
            and all(
                type(row.get(field)) is int
                for field in (
                    "level",
                    "included_endpoint",
                    "excluded_endpoint",
                    "cumulative_count",
                )
            )
            for row in levels
        )
    )
    return (
        exact_integer_levels
        and sorted(result.get("levels"), key=lambda row: row["level"]) == expected
        and q(result.get("lower_density")) == Fraction(1, b + 1)
        and q(result.get("upper_density")) == Fraction(b, b + 1)
        and result.get("lower_density_positive") is True
        and result.get("natural_density_exists") is False
        and result.get("semantic_relation") == "FORMALIZED_PREDICATE_STRICTLY_STRONGER"
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
