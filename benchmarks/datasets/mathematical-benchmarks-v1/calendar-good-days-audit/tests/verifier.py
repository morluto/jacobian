import json
from pathlib import Path

from verifier_support import (
    json_value_equal,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    workspace_input_is_bound,
)

W = Path("/app")
E = Path("/tests")


def witness_matches_result(witness, result):
    if not isinstance(witness, list) or len(witness) != 1:
        return False
    target = resolve_evidence(witness[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json_value_equal(json.loads(marker), result) and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def concatenate(month, day):
    return int(f"{month}{day}")


def main():
    input_binding = workspace_input_is_bound()
    submission = load_submission(require_input_binding=False)
    x = json.loads(next(E.glob("*input*.json")).read_text())
    r = submission.get("result") if isinstance(submission, dict) else None
    r = r if isinstance(r, dict) else {}
    expected_dates = []
    for month_spec in x["months"]:
        month = month_spec["month"]
        for day in range(1, month_spec["days"] + 1):
            value = concatenate(month, day)
            if value % month == 0 and value % day == 0:
                expected_dates.append(
                    {"month": month, "day": day, "concatenated": value}
                )
    valid = (
        set(r) == {"count", "good_dates"}
        and type(r.get("count")) is int
        and isinstance(r.get("good_dates"), list)
        and all(
            isinstance(date, dict)
            and set(date) == {"month", "day", "concatenated"}
            and all(
                type(date[field]) is int for field in ("month", "day", "concatenated")
            )
            for date in r["good_dates"]
        )
        and r.get("count") == len(expected_dates)
        and r.get("good_dates") == expected_dates
        and len(expected_dates) != 15
    )
    math_correct = bool(valid)
    witness_ok = bool(
        isinstance(submission, dict)
        and witness_matches_result(submission.get("witness"), r)
    )
    reward = float(input_binding and math_correct and witness_ok)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "witness_validity": float(witness_ok),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
