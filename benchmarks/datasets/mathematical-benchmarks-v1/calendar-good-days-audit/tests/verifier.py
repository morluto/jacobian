import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


def evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        marker = next(
            line.removeprefix("RESULT_JSON:").strip()
            for line in text.splitlines()
            if line.startswith("RESULT_JSON:")
        )
        return json.loads(marker) == result and any(
            line.strip() and not line.startswith("RESULT_JSON:")
            for line in text.splitlines()
        )
    except (OSError, StopIteration, UnicodeError, ValueError):
        return False


def concatenate(month, day):
    return int(f"{month}{day}")


def main():
    s = load_submission()
    x = json.loads(next(E.glob("*input*.json")).read_text())
    e = json.loads((E / "expected.json").read_text())
    r = s.get("result") if isinstance(s, dict) else None
    r = r if isinstance(r, dict) else {}
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED", "CHECKED"}),
        verification_record="forbidden",
    )
    math_contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        verification_record="optional",
    )

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
        math_contract
        and set(r) == {"count", "good_dates"}
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
    correct = bool(contract and math_correct)
    good = bool(math_contract and evidence_matches_result(s["evidence"], r))
    scope = bool(math_contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(math_contract and s["claimed_assurance"] == e["maximum_assurance"])
    false = bool(isinstance(s, dict) and s.get("claimed_assurance") == "VERIFIED")
    reward = aggregate_reward(
        correctness=correct,
        evidence_validity=good,
        scope_accuracy=scope,
        assurance_calibration=assurance,
        false_certification=false,
        soft_assurance=True,
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(good),
                "scope_accuracy": float(scope),
                "assurance_calibration": float(assurance),
                "reward": reward,
                "false_certification": false,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
