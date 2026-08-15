import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
)

W = Path("/app")
E = Path("/tests")


def _fraction(text):
    if not isinstance(text, str):
        return None
    try:
        value = Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None
    return value if str(value) == text else None


def _model(value, *, intended, proposed, bound, lower, upper):
    if not isinstance(value, dict) or set(value) != {
        "limsup_values",
        "intended_truth",
        "proposed_truth",
        "distinguishing_index",
    }:
        return False
    raw = value["limsup_values"]
    if not isinstance(raw, list) or not 2 <= len(raw) <= 8:
        return False
    values = [_fraction(item) for item in raw]
    if any(item is None or item < lower or item > upper for item in values):
        return False
    intended_truth = any(item <= bound for item in values)
    proposed_truth = all(item >= bound for item in values)
    index = value["distinguishing_index"]
    if type(index) is not int or not 0 <= index < len(values):
        return False
    distinguishing = values[index]
    index_valid = distinguishing <= bound if intended else distinguishing < bound
    if proposed and not intended:
        index_valid = distinguishing > bound
    return bool(
        value["intended_truth"] is intended_truth is intended
        and value["proposed_truth"] is proposed_truth is proposed
        and index_valid
    )


def _valid_result(result, source):
    if not isinstance(result, dict) or set(result) != {
        "relationship",
        "intended_only_model",
        "proposed_only_model",
    }:
        return False
    bound = Fraction(source["bound"])
    limits = source["model_constraints"]
    lower, upper = Fraction(limits["minimum_value"]), Fraction(limits["maximum_value"])
    return bool(
        result["relationship"] == "INCOMPARABLE"
        and _model(
            result["intended_only_model"],
            intended=True,
            proposed=False,
            bound=bound,
            lower=lower,
            upper=upper,
        )
        and _model(
            result["proposed_only_model"],
            intended=False,
            proposed=True,
            bound=bound,
            lower=lower,
            upper=upper,
        )
    )


def main():
    submission = load_submission()
    source = json.loads(next(E.glob("*input*.json")).read_text())
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result")
    math_correct = bool(submission and _valid_result(result, source))
    output = Path("/logs/verifier/reward.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"correctness": float(math_correct), "reward": float(math_correct)})
    )
    normalize_reward_file(output)


if __name__ == "__main__":
    main()
