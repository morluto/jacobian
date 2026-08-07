import json
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    resolve_evidence,
    strict_submission_contract,
)

W = Path("/app")
E = Path("/tests")


def _complete_obstruction(result):
    if not isinstance(result, dict) or set(result) != {
        "modulus",
        "target_residue",
        "residue_cases",
    }:
        return False
    modulus = result["modulus"]
    target = result["target_residue"]
    cases = result["residue_cases"]
    if (
        type(modulus) is not int
        or modulus != 7
        or type(target) is not int
        or not 0 <= target < modulus
        or not isinstance(cases, list)
        or len(cases) != modulus
    ):
        return False

    observed = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "x_residue",
            "x_cube_residue",
            "lhs_residue",
        }:
            return False
        x = case["x_residue"]
        cube = case["x_cube_residue"]
        lhs = case["lhs_residue"]
        if (
            type(x) is not int
            or type(cube) is not int
            or type(lhs) is not int
            or not 0 <= x < modulus
            or x in observed
            or cube != pow(x, 3, modulus)
            or lhs != (4 * cube) % modulus
        ):
            return False
        observed[x] = lhs

    if set(observed) != set(range(modulus)) or target != 2003 % modulus:
        return False

    # Independently check every residue pair. This does not trust the submitted
    # shortcut that the y term vanishes for the selected modulus.
    return all(
        (4 * pow(x, 3, modulus) - 7 * pow(y, 3, modulus)) % modulus != target
        for x in range(modulus)
        for y in range(modulus)
    )


def _evidence_matches_result(evidence, result):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
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


def main():
    submission = load_submission()
    expected = json.loads((E / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _complete_obstruction(submission.get("result")))
    evidence_valid = bool(
        contract
        and _evidence_matches_result(submission["evidence"], submission["result"])
    )
    scope_correct = bool(
        contract and submission.get("scope") == expected["required_scope"]
    )
    assurance_correct = bool(
        contract
        and submission.get("claimed_assurance") == expected["maximum_assurance"]
    )
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = bool(
        contract and math_correct and scope_correct and not false_certification
    )
    reward = (
        0.0
        if not correct or not evidence_valid
        else 0.8 + 0.1 * scope_correct + 0.1 * assurance_correct
    )

    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    (Path("/logs/verifier/reward.json")).write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": reward,
                "false_certification": false_certification,
            }
        )
    )


if __name__ == "__main__":
    main()
