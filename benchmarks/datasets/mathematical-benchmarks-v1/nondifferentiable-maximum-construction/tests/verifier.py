import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    load_submission,
    normalize_reward_file,
    witness_list_is_bound,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def _fraction(value):
    if not isinstance(value, str) or len(value) > 80:
        raise ValueError
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?", value) is None:
        raise ValueError
    parsed = Fraction(value)
    if str(parsed) != value:
        raise ValueError
    return parsed


def _valid_construction(result, source):
    if not isinstance(result, dict) or set(result) != {
        "peak",
        "left_slope",
        "right_slope",
        "left_value_at_join",
        "right_value_at_join",
        "left_derivative",
        "right_derivative",
    }:
        return False
    try:
        values = {key: _fraction(value) for key, value in result.items()}
    except (ValueError, ZeroDivisionError):
        return False
    if source.get("domain") != {"left": "-1", "right": "1", "join": "0"}:
        return False
    peak = values["peak"]
    left = values["left_slope"]
    right = values["right_slope"]
    return bool(
        values["left_value_at_join"] == peak
        and values["right_value_at_join"] == peak
        and values["left_derivative"] == left
        and values["right_derivative"] == right
        and left >= 0
        and right <= 0
        and left != right
        and peak - left <= peak
        and peak + right <= peak
    )


def main():
    submission = load_submission()
    input_binding = workspace_input_is_bound()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    math_ok = bool(
        submission is not None and _valid_construction(submission.get("result"), source)
    )
    ev_ok = bool(
        submission is not None
        and witness_list_is_bound(
            submission.get("witness"), expected_path="evidence/answer.txt"
        )
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=ev_ok,
        protocol_ok=bool(input_binding and submission is not None),
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(ev_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
