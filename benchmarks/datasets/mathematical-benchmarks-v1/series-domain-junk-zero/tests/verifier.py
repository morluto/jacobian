import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


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


def _canonical_fraction(value):
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


def _valid_blocks(blocks, q, start, end):
    if not isinstance(blocks, list) or len(blocks) != end - start + 1:
        return False
    for block, level in zip(blocks, range(start, end + 1), strict=True):
        if not isinstance(block, dict) or set(block) != {
            "level",
            "term_count",
            "upper_power_of_two",
            "block_sum_power_lower_bound",
        }:
            return False
        if block != {
            "level": level,
            "term_count": 2**level,
            "upper_power_of_two": 2 ** (level + 1),
            "block_sum_power_lower_bound": 2 ** ((q - 1) * level - 1),
        }:
            return False
    return True


def _valid_result(result, frozen):
    if not isinstance(result, dict) or set(result) != {
        "reciprocal_denominator",
        "real_part",
        "general_block_power_exponent",
        "blocks",
        "summability_status",
        "returned_value",
        "zero_classification",
        "critical_line_relation",
    }:
        return False
    bounds = frozen.get("denominator_bounds")
    levels = frozen.get("block_levels")
    q = result.get("reciprocal_denominator")
    if not (
        isinstance(bounds, list)
        and bounds == [3, 7]
        and isinstance(levels, list)
        and levels == [2, 10]
        and type(q) is int
        and bounds[0] <= q <= bounds[1]
    ):
        return False
    return bool(
        _canonical_fraction(result.get("real_part")) == Fraction(1, q)
        and result.get("general_block_power_exponent")
        == {"level_coefficient": q - 1, "constant": -1}
        and _valid_blocks(result.get("blocks"), q, levels[0], levels[1])
        and result.get("summability_status") == "DIVERGENT"
        and type(result.get("returned_value")) is int
        and result.get("returned_value") == 0
        and result.get("zero_classification") == "FALLBACK_ARTIFACT"
        and result.get("critical_line_relation") == "REAL_PART_NOT_ONE_HALF"
    )


def main():
    submission = load_submission()
    protocol_ok = submission is not None
    frozen = _load_frozen_input()
    math_correct = bool(protocol_ok and _valid_result(submission.get("result"), frozen))
    reward = aggregate_reward(
        correctness=math_correct,
        witness_validity=True,
        protocol_ok=protocol_ok,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
