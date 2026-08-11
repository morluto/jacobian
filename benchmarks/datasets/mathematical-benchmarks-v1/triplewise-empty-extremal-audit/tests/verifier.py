import itertools
import json
import re
from pathlib import Path

from verifier_support import (
    evidence_list_is_bound,
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    resolve_evidence,
    strict_submission_contract,
)


def valid_family(n, raw_family):
    try:
        if type(n) is not int or not isinstance(raw_family, list):
            return False
        if any(
            not isinstance(item, list) or any(type(value) is not int for value in item)
            for item in raw_family
        ):
            return False
        family = [tuple(item) for item in raw_family]
        if len(family) != len(set(family)):
            return False
        if any(
            tuple(sorted(item)) != item or len(item) != len(set(item))
            for item in family
        ):
            return False
        if any(value < 0 or value >= n for item in family for value in item):
            return False
        if len(family) != 1 + n + n // 2:
            return False
        sets = [set(item) for item in family]
        if any(
            first & second & third
            for first, second, third in itertools.combinations(sets, 3)
        ):
            return False
        return all(sum(value in item for item in sets) <= 2 for value in range(n))
    except (TypeError, ValueError):
        return False


def _result_shape_is_valid(result):
    if not isinstance(result, dict) or set(result) != {
        "maximum_formula",
        "upper_bound_certificate",
        "constructions",
    }:
        return False
    certificate = result.get("upper_bound_certificate")
    if not isinstance(certificate, dict) or set(certificate) != {
        "element_frequency_cap",
        "singleton_cap",
        "nonsingleton_incidence_floor",
    }:
        return False
    constructions = result.get("constructions")
    if not isinstance(constructions, list) or len(constructions) != 3:
        return False
    return all(
        isinstance(item, dict)
        and set(item) == {"n", "family"}
        and type(item.get("n")) is int
        and isinstance(item.get("family"), list)
        for item in constructions
    )


def _math_claim_is_correct(result):
    if not _result_shape_is_valid(result):
        return False
    certificate = result["upper_bound_certificate"]
    constructions = result["constructions"]
    construction_map = {item["n"]: item["family"] for item in constructions}
    return bool(
        result["maximum_formula"] == "1+n+floor(n/2)"
        and certificate
        == {
            "element_frequency_cap": 2,
            "singleton_cap": "n",
            "nonsingleton_incidence_floor": 2,
        }
        and set(construction_map) == {7, 8, 11}
        and len(construction_map) == len(constructions)
        and all(valid_family(n, construction_map[n]) for n in construction_map)
    )


def _evidence_is_valid(evidence):
    if not evidence_list_is_bound(evidence, expected_path="evidence/answer.txt"):
        return False
    target = resolve_evidence(evidence[0], expected_path="evidence/answer.txt")
    if target is None:
        return False
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    folded = text.casefold()
    return bool(
        text.strip()
        and re.search(r"1\s*\+\s*n\s*\+\s*floor\s*\(\s*n\s*/\s*2\s*\)", folded)
        and re.search(
            r"element(?:s)?(?:'s)?\s+(?:frequency|occurrence|incidence)"
            r".{0,100}(?:at most|no more than|<=|≤)\s*(?:2|two)",
            folded,
        )
        and re.search(
            r"(?:remaining|non[- ]?singleton).{0,140}"
            r"(?:at least|minimum|>=|≥)\s*(?:2|two).{0,80}incidence",
            folded,
        )
        and re.search(
            r"(?:attain|attains|attained|match|matches|matching).{0,100}"
            r"(?:bound|maximum|formula)",
            folded,
        )
    )


def _scope_is_positive(scope):
    if not isinstance(scope, str):
        return False
    folded = scope.casefold()
    return bool(
        "distinct subsets" in folded
        and "triplewise empty intersection" in folded
        and not re.search(
            r"\b(?:not|no|without|never|doesn(?:t|'t)|does not)\b", folded
        )
    )


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/triplewise-empty-extremal-audit",
        conclusion="SOURCE_BOUND_REPAIRED",
        min_limitations=1,
        verification_record="forbidden",
    )
    data = submission if isinstance(submission, dict) else {}
    result = data.get("result", {})
    math_correct = _math_claim_is_correct(result)
    evidence_valid = _evidence_is_valid(data.get("evidence"))
    scope_correct = bool(contract and _scope_is_positive(data.get("scope")))
    assurance_correct = bool(contract and data.get("claimed_assurance") == "COMPUTED")
    false_certification = bool(
        false_verified_claim(submission, verification_record_bound=False)
    )
    reward = (
        1.0
        if contract
        and math_correct
        and evidence_valid
        and scope_correct
        and assurance_correct
        and not false_certification
        else 0.0
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
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
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
