import json
import math
from pathlib import Path
from typing import Any

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)

WORKSPACE = Path("/app")
TESTS = Path("/tests")
LIMITATION = "The witness proves neither minimality nor claims about other exponents or numbers of summands."


def _load() -> dict[str, Any]:
    try:
        a, b = WORKSPACE / "input.json", TESTS / "input.json"
        if a.is_symlink() or b.is_symlink() or a.read_bytes() != b.read_bytes():
            return {}
        value = json.loads(b.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _result(result: object, frozen: dict[str, Any]) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "left_bases",
        "target_base",
        "left_fourth_powers",
        "left_sum",
        "right_fourth_power",
        "joint_gcd",
        "residue_checks",
        "claim_status",
    }:
        return False
    bases = result.get("left_bases")
    target = result.get("target_base")
    scope = frozen.get("search_scope")
    moduli = frozen.get("required_moduli")
    if (
        not isinstance(bases, list)
        or len(bases) != 3
        or any(type(x) is not int for x in bases)
        or type(target) is not int
        or not isinstance(scope, dict)
        or type(scope.get("maximum_base")) is not int
        or not isinstance(moduli, list)
        or any(type(p) is not int for p in moduli)
    ):
        return False
    bound = scope["maximum_base"]
    all_bases = [*bases, target]
    if bases != sorted(bases) or len(set(all_bases)) != 4:
        return False
    if any(x <= 0 or x > bound for x in all_bases):
        return False
    powers = [x**4 for x in bases]
    total = sum(powers)
    right = target**4
    expected_residues = [
        {"modulus": p, "left_residue": total % p, "right_residue": right % p}
        for p in moduli
    ]
    return (
        result.get("left_fourth_powers") == powers
        and result.get("left_sum") == total
        and result.get("right_fourth_power") == right
        and total == right
        and type(result.get("joint_gcd")) is int
        and result["joint_gcd"] == math.gcd(*bases, target) == 1
        and result.get("residue_checks") == expected_residues
        and result.get("claim_status") == "REFUTED_BY_COUNTEREXAMPLE"
    )


def _evidence(value: object) -> bool:
    if not evidence_list_is_bound(value, expected_path="evidence/answer.txt"):
        return False
    if not isinstance(value, list) or len(value) != 1:
        return False
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text().strip()
    except (OSError, UnicodeError):
        return False
    return len(text) >= 20


def main() -> None:
    submission = load_submission()
    data = submission if isinstance(submission, dict) else {}
    expected = json.loads((TESTS / "expected.json").read_text())
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        verification_record="forbidden",
    )
    math_correct = bool(contract and _result(data.get("result"), _load()))
    evidence_valid = bool(math_correct and _evidence(data.get("evidence")))
    scope_correct = bool(contract and data.get("scope") == expected["required_scope"])
    assurance_correct = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations_correct = bool(contract and LIMITATION in data.get("limitations", []))
    false_certification = false_verified_claim(
        submission, verification_record_bound=False
    )
    correct = (
        math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and limitations_correct
        and not false_certification
    )
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    (output / "reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_correct),
                "evidence_validity": float(evidence_valid),
                "scope_accuracy": float(scope_correct),
                "assurance_calibration": float(assurance_correct),
                "reward": float(correct),
                "false_certification": false_certification,
            }
        )
    )
    normalize_reward_file(output / "reward.json")


if __name__ == "__main__":
    main()
