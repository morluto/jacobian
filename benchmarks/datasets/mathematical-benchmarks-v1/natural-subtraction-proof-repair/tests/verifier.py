import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _fraction(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    if not re.fullmatch("-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value):
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def _result_fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError
    numerator = value["numerator"]
    denominator = value["denominator"]
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ValueError
    parsed = Fraction(numerator, denominator)
    return parsed


def _load_frozen_input():
    try:
        workspace = W / "input.json"
        frozen = E / "input.json"
        if workspace.is_symlink() or frozen.is_symlink():
            return {}
        frozen_bytes = frozen.read_bytes()
        if workspace.read_bytes() != frozen_bytes:
            return {}
        value = json.loads(frozen_bytes)
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _contains(tree, pattern):
    if tree == pattern:
        return True
    if not isinstance(tree, dict):
        return False
    return any(_contains(child, pattern) for child in tree.get("args", []))


def _repair_is_valid(result, source):
    if not isinstance(result, dict) or set(result) != {
        "failed_pattern_occurs",
        "basis_order",
        "multipliers",
        "derived_coefficients",
        "side_condition_used",
    }:
        return False
    rewrite = source.get("failed_rewrite", {})
    actual_occurrence = _contains(rewrite.get("target_ast"), rewrite.get("pattern_ast"))
    if result.get("failed_pattern_occurs") is not actual_occurrence:
        return False
    basis = source.get("equation_basis")
    if not isinstance(basis, list) or result.get("basis_order") != [
        row.get("id") for row in basis
    ]:
        return False
    if result.get("side_condition_used") != "b<=a" or "b<=a" not in source.get(
        "branch_context", {}
    ).get("hypotheses", []):
        return False
    try:
        multipliers = [_result_fraction(value) for value in result["multipliers"]]
        vectors = [[_fraction(value) for value in row["coefficients"]] for row in basis]
        claimed = [_result_fraction(value) for value in result["derived_coefficients"]]
        goal = [_fraction(value) for value in source["goal_coefficients"]]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    if len(multipliers) != len(vectors) or not vectors:
        return False
    width = len(goal)
    if any(len(vector) != width for vector in vectors) or len(claimed) != width:
        return False
    derived = [
        sum(
            (
                multiplier * vector[index]
                for multiplier, vector in zip(multipliers, vectors, strict=True)
            ),
            Fraction(0),
        )
        for index in range(width)
    ]
    return bool(not actual_occurrence and derived == claimed == goal)


def main():
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    source = _load_frozen_input()
    math_ok = bool(
        submission is not None and _repair_is_valid(submission.get("result"), source)
    )
    reward = float(math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
