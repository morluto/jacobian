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
MISMATCHES = {"OBJECTIVE_REPLACED", "BINARY_DOMAIN_RELAXED", "UNDECLARED_BUDGET_ADDED"}


def _mismatches_ok(mismatches):
    return (
        isinstance(mismatches, list)
        and len(mismatches) == 3
        and all(type(mismatch) is str for mismatch in mismatches)
        and set(mismatches) == MISMATCHES
    )


def _repair_method_ok(result):
    return (
        result.get("repair_method") == "EXACT_FRACTIONAL_RESIDUAL_CERTIFICATE"
        and type(result.get("maximum_residual_sum")) is int
        and result.get("maximum_residual_sum") == 0
    )


def _selected_indices_ok(selected, data):
    return (
        isinstance(selected, list)
        and all(type(index) is int for index in selected)
        and len(selected) == len(set(selected))
        and all(0 <= index < len(data["items"]) for index in selected)
    )


def _parse_ratio(ratio_text):
    if (
        not isinstance(ratio_text, str)
        or len(ratio_text) > 64
        or re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", ratio_text) is None
    ):
        return None
    try:
        ratio = Fraction(ratio_text)
    except (ValueError, ZeroDivisionError):
        return None
    return ratio if str(ratio) == ratio_text else None


def _residuals_ok(submitted, residuals):
    if not isinstance(submitted, list) or len(submitted) != len(residuals):
        return False
    if any(
        not isinstance(row, dict)
        or set(row) != {"index", "value"}
        or type(row["index"]) is not int
        or type(row["value"]) is not int
        for row in submitted
    ):
        return False
    submitted_by_index = {row["index"]: row["value"] for row in submitted}
    return len(submitted_by_index) == len(residuals) and submitted_by_index == dict(
        enumerate(residuals)
    )


def _positives_ok(result, positives, selected, constant, residuals):
    submitted_positives = result.get("positive_residual_indices")
    return (
        type(result.get("constant_residual")) is int
        and result.get("constant_residual") == constant
        and isinstance(submitted_positives, list)
        and all(type(index) is int for index in submitted_positives)
        and len(submitted_positives) == len(set(submitted_positives))
        and set(submitted_positives) == set(positives)
        and set(selected) == set(positives)
        and constant + sum(max(0, value) for value in residuals) == 0
    )


def valid_result(result, data):
    if not isinstance(result, dict) or set(result) != {
        "contract_mismatches",
        "selected_indices",
        "attained_ratio",
        "constant_residual",
        "item_residuals",
        "positive_residual_indices",
        "maximum_residual_sum",
        "repair_method",
    }:
        return False
    if not _mismatches_ok(result.get("contract_mismatches")):
        return False
    if not _repair_method_ok(result):
        return False
    selected = result.get("selected_indices")
    if not _selected_indices_ok(selected, data):
        return False
    ratio = _parse_ratio(result.get("attained_ratio"))
    if ratio is None:
        return False
    numerator = data["alpha"] + sum(data["items"][index]["t"] for index in selected)
    denominator = data["beta"] + sum(data["items"][index]["f"] for index in selected)
    if Fraction(numerator, denominator) != ratio:
        return False
    p, q = ratio.numerator, ratio.denominator
    constant = q * data["alpha"] - p * data["beta"]
    residuals = [q * item["t"] - p * item["f"] for item in data["items"]]
    if not _residuals_ok(result.get("item_residuals"), residuals):
        return False
    positives = [index for index, value in enumerate(residuals) if value > 0]
    return _positives_ok(result, positives, selected, constant, residuals)


def _array_preflight(raw):
    """Reject oversized index arrays before expensive schema validation."""

    if not isinstance(raw, dict):
        return True
    result = raw.get("result")
    if not isinstance(result, dict):
        return True
    return not any(
        isinstance(result.get(key), list) and len(result[key]) > 24
        for key in ("selected_indices", "positive_residual_indices", "item_residuals")
    )


def main():
    raw = load_submission_raw(require_input_binding=False)
    data = json.loads((T / "input.json").read_text())
    input_binding = workspace_input_is_bound()
    protocol_ok = _array_preflight(raw) and submission_matches_public_schema(raw)
    result = raw.get("result") if isinstance(raw, dict) else None
    math_ok = valid_result(result, data)
    reward = float(input_binding and protocol_ok and math_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "input_binding": float(input_binding),
                "protocol_compliance": float(protocol_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
