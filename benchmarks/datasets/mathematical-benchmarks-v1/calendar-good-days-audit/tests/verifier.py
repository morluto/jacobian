import json
from pathlib import Path

from verifier_support import (
    load_submission,
    normalize_reward_file,
    workspace_input_is_bound,
)

E = Path("/tests")


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
    reward = float(input_binding and math_correct)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
