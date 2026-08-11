import json
import math
import re
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
LIMITATION = "The checker validates the symbolic two-branch contract and bounded residue sanity suite, but does not replay a proof assistant proof of the universal theorem."


def _load_input() -> dict[str, Any]:
    try:
        raw = (TESTS / "input.json").read_bytes()
        if (WORKSPACE / "input.json").read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


def _witness(value: object, *, unit: bool) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "p",
        "d",
        "d_p",
        "C",
        "left_residue",
        "right_residue",
    }:
        return False
    if any(type(value[k]) is not int for k in value):
        return False
    p, d, dp, c = value["p"], value["d"], value["d_p"], value["C"]
    if not (
        3 <= p <= 43
        and _prime(p)
        and p % 2
        and 1 <= d <= 80
        and math.gcd(d, p - 1) == 1
        and dp == d % (p - 1)
        and 1 <= dp <= p - 2
    ):
        return False
    if (math.gcd(c, p) == 1) is not unit:
        return False
    left, right = pow(c, d, p), pow(c, dp, p)
    return (
        value["left_residue"] == left
        and value["right_residue"] == right
        and left == right
    )


def _bounded_sanity(source: dict[str, Any]) -> bool:
    bounds = source.get("sanity_bounds", {})
    if bounds != {"maximum_prime": 43, "maximum_exponent": 80}:
        return False
    for p in range(3, 44, 2):
        if not _prime(p):
            continue
        for d in range(1, 81):
            if math.gcd(d, p - 1) != 1:
                continue
            dp = d % (p - 1)
            if not 1 <= dp <= p - 2:
                return False
            for c in range(p):
                if pow(c, d, p) != pow(c, dp, p):
                    return False
    return True


def _result(value: object, source: dict[str, Any]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "diagnosis",
        "remainder_bounds",
        "unit_branch",
        "nonunit_branch",
        "domain_split",
        "unit_witness",
        "nonunit_witness",
        "finite_testing_role",
    }:
        return False
    provenance = source.get("source", {})
    return bool(
        provenance.get("revision") == "f5935720f176cedff4ecd8ebf83d1696e31cfac8"
        and provenance.get("row") == 7
        and provenance.get("source_id") == 780
        and value["diagnosis"]
        == {
            "unsafe_step": "NEGATIVE_POWER_REQUIRES_UNIT",
            "missing_domain": "C_CONGRUENT_ZERO_MOD_P",
        }
        and value["remainder_bounds"]
        == {"lower": 1, "upper": "p-2", "reason": "COPRIMALITY_EXCLUDES_ZERO_REMAINDER"}
        and value["unit_branch"]
        == {
            "condition": "gcd(C,p)=1",
            "quotient_relation": "d=d_p+k*(p-1)",
            "quotient_bound": "k>=0",
            "identity": "C^d=C^d_p*(C^(p-1))^k",
        }
        and value["nonunit_branch"]
        == {
            "condition": "p|C",
            "d_positive": True,
            "d_p_positive": True,
            "residues": [0, 0],
        }
        and value["domain_split"] == ["gcd(C,p)=1", "p|C"]
        and _witness(value["unit_witness"], unit=True)
        and _witness(value["nonunit_witness"], unit=False)
        and value["finite_testing_role"] == "SANITY_ONLY_NOT_UNIVERSAL_PROOF"
        and _bounded_sanity(source)
    )


def _evidence(value: object) -> bool:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not evidence_list_is_bound(value)
    ):
        return False
    assert isinstance(value, list)
    path = resolve_evidence(value[0], expected_path="evidence/answer.txt")
    if path is None:
        return False
    try:
        text = path.read_text().lower()
    except (OSError, UnicodeError):
        return False
    if not all(
        term in text
        for term in ("unit branch", "nonunit branch", "negative exponent", "computed")
    ):
        return False
    contradictions = (
        re.search(r"\bno\b[^.]{0,80}\bunit branch\b", text),
        re.search(
            r"\bnegative exponent\b[^.]{0,80}\b(?:always|everywhere|regardless)\b",
            text,
        ),
        re.search(
            r"\b(?:inverse|unit condition)\b[^.]{0,80}"
            r"\b(?:never|required nowhere)\b",
            text,
        ),
    )
    if any(contradictions):
        return False
    return bool(
        re.search(r"\b(?:gcd|coprime|unit group|unit condition)\b", text)
        and re.search(
            r"\b(?:p\s*\|\s*c|nonunit|zero modulo p|both positive powers)\b",
            text,
        )
        and re.search(r"\bd\s*=\s*d_p\s*\+\s*k\s*\(p-1\)|fermat\b", text)
    )


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
    correct = bool(contract and _result(data.get("result"), _load_input()))
    evidence = bool(correct and _evidence(data.get("evidence")))
    scope = bool(contract and data.get("scope") == expected["required_scope"])
    assurance = bool(
        contract and data.get("claimed_assurance") == expected["maximum_assurance"]
    )
    limitations = bool(
        contract
        and isinstance(data.get("limitations"), list)
        and any(
            isinstance(item, str)
            and "proof assistant" in item.casefold()
            and "not" in item.casefold()
            for item in data["limitations"]
        )
    )
    false_verified = false_verified_claim(submission, verification_record_bound=False)
    passed = bool(
        correct
        and evidence
        and scope
        and assurance
        and limitations
        and not false_verified
    )
    logs = Path("/logs/verifier")
    logs.mkdir(parents=True, exist_ok=True)
    result = {
        "correctness": float(correct),
        "evidence_validity": float(evidence),
        "scope_accuracy": float(scope),
        "assurance_calibration": float(assurance),
        "reward": float(passed),
        "false_certification": false_verified,
    }
    (logs / "reward.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    normalize_reward_file(logs / "reward.json")


if __name__ == "__main__":
    main()
