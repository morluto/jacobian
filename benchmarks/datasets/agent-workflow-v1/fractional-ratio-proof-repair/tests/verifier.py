import json
import re
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
MISMATCHES = {"OBJECTIVE_REPLACED", "BINARY_DOMAIN_RELAXED", "UNDECLARED_BUDGET_ADDED"}
LIMITATION = "The verifier certifies only the frozen exact instance; it does not machine-prove a general greedy theorem."


def evidence_ok(evidence, result):
    if not evidence_list_is_bound(evidence):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text()
        lines = [line for line in text.splitlines() if line.startswith("RESULT_JSON:")]
        return (
            len(lines) == 1
            and json.loads(lines[0].removeprefix("RESULT_JSON:").strip()) == result
            and all(
                word in text.lower()
                for word in ("objective", "binary", "budget", "residual")
            )
        )
    except (OSError, UnicodeError, ValueError):
        return False


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
    if (
        set(result.get("contract_mismatches", [])) != MISMATCHES
        or len(result["contract_mismatches"]) != 3
    ):
        return False
    if (
        result.get("repair_method") != "EXACT_FRACTIONAL_RESIDUAL_CERTIFICATE"
        or result.get("maximum_residual_sum") != 0
    ):
        return False
    selected = result.get("selected_indices")
    if (
        not isinstance(selected, list)
        or any(type(i) is not int for i in selected)
        or selected != sorted(set(selected))
        or any(i < 0 or i >= len(data["items"]) for i in selected)
    ):
        return False
    ratio_text = result.get("attained_ratio")
    if (
        not isinstance(ratio_text, str)
        or re.fullmatch(r"[1-9][0-9]*/[1-9][0-9]*", ratio_text) is None
    ):
        return False
    ratio = Fraction(ratio_text)
    if str(ratio) != ratio_text:
        return False
    numerator = data["alpha"] + sum(data["items"][i]["t"] for i in selected)
    denominator = data["beta"] + sum(data["items"][i]["f"] for i in selected)
    if Fraction(numerator, denominator) != ratio:
        return False
    p, q = ratio.numerator, ratio.denominator
    constant = q * data["alpha"] - p * data["beta"]
    residuals = [q * item["t"] - p * item["f"] for item in data["items"]]
    submitted = result.get("item_residuals")
    if not isinstance(submitted, list) or submitted != [
        {"index": i, "value": value} for i, value in enumerate(residuals)
    ]:
        return False
    positives = [i for i, value in enumerate(residuals) if value > 0]
    return (
        result.get("constant_residual") == constant
        and result.get("positive_residual_indices") == positives
        and selected == positives
        and constant + sum(max(0, value) for value in residuals) == 0
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    data = json.loads((T / "input.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"UNVERIFIED", "COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(workspace_input_is_bound() and valid_result(result, data))
    ev_ok = bool(
        isinstance(submission, dict) and evidence_ok(submission.get("evidence"), result)
    )
    scope_ok = bool(
        isinstance(submission, dict)
        and submission.get("scope") == "frozen-24-item-binary-fractional-ratio-instance"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == [LIMITATION]
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(contract and math_ok and ev_ok and scope_ok and not false_cert)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(bool(contract)),
                "correctness": float(math_ok),
                "evidence_validity": float(ev_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": 0 if not correct else 0.9 + 0.1 * assurance_ok,
                "false_certification": false_cert,
            }
        )
    )


if __name__ == "__main__":
    main()
