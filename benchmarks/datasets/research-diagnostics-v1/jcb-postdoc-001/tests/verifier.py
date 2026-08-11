import json
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    evidence_list_is_bound,
    load_submission,
    normalize_reward_file,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")
ALLOWED = frozenset({"UNVERIFIED", "COMPUTED"})


def _math(s, e):
    r = s.get("result", {})
    if not isinstance(r, dict):
        return False
    if r.get("expected_conclusion") != e["expected_conclusion"]:
        return False
    boundary = r.get("boundary_statement")
    if not isinstance(boundary, str) or len(boundary) == 0:
        return False
    summary = r.get("answer_visible_summary")
    if not isinstance(summary, str) or len(summary) == 0:
        return False
    want = e.get("expected_key_facts")
    facts = r.get("key_facts", {})
    if not isinstance(want, dict) or not isinstance(facts, dict):
        return False
    if not set(want).issubset(facts):
        return False
    return all(
        type(facts[key]) is str and type(value) is str and facts[key] == value
        for key, value in want.items()
    )


def main():
    s = load_submission()
    e = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        s,
        task_id=e["task_id"],
        conclusion=e["conclusion"],
        allowed_assurances=ALLOWED,
        verification_record="forbidden",
    )
    math_correct = _math(s, e) if contract else False
    correct = bool(contract and math_correct)
    good = bool(contract and evidence_list_is_bound(s["evidence"]))
    scope = bool(contract and s["scope"] == " ".join(e["required_scope_terms"]))
    assurance = bool(contract and s["claimed_assurance"] == e["maximum_assurance"])
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
