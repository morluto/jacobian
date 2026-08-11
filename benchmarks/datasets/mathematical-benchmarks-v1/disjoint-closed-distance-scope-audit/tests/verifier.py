import json
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")


def rat(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    if (
        type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    return Fraction(value["numerator"], value["denominator"])


def _epsilon_witnesses_ok(eps, s, c):
    if not isinstance(eps, list) or len(eps) != 8:
        return False
    for k, witness in enumerate(eps, 2):
        if not isinstance(witness, dict) or set(witness) != {
            "epsilon",
            "index",
            "distance_squared",
        }:
            return False
        e, n, d = (
            rat(witness["epsilon"]),
            witness["index"],
            rat(witness["distance_squared"]),
        )
        if (
            e != Fraction(1, k)
            or type(n) is not int
            or n < 1
            or d != Fraction(s * s, (n + c) ** 2)
            or not d < e * e
        ):
            return False
    return True


def _separation_ok(sep, h):
    if not isinstance(sep, dict) or set(sep) != {
        "same_family_lower_bound_squared",
        "cross_family_vertical_nonzero",
        "closedness_reason",
    }:
        return False
    return (
        rat(sep["same_family_lower_bound_squared"]) == h * h
        and sep["cross_family_vertical_nonzero"] is True
        and sep["closedness_reason"] == "DISTINCT_INDICES_HAVE_HORIZONTAL_GAP"
    )


def result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "horizontal_step",
        "vertical_scale",
        "offset",
        "sample_indices",
        "distance_squared",
        "epsilon_witnesses",
        "separation_certificate",
        "formal_conclusion",
        "corrected_conclusion",
    }:
        return False
    h, s, c = result["horizontal_step"], result["vertical_scale"], result["offset"]
    if any(type(x) is not int for x in (h, s, c)) or not (
        2 <= h <= 20 and 1 <= s <= 20 and 2 <= c <= 20
    ):
        return False
    ns = result["sample_indices"]
    if (
        ns != sorted(ns)
        or len(ns) != 10
        or len(set(ns)) != 10
        or any(type(n) is not int or n < 1 for n in ns)
    ):
        return False
    distances = result["distance_squared"]
    if not isinstance(distances, list) or len(distances) != len(ns):
        return False
    expected = [Fraction(s * s, (n + c) ** 2) for n in ns]
    if [rat(x) for x in distances] != expected:
        return False
    if not _epsilon_witnesses_ok(result["epsilon_witnesses"], s, c):
        return False
    if not _separation_ok(result["separation_certificate"], h):
        return False
    return (
        result["formal_conclusion"] == "POSITIVE_DISTANCE"
        and result["corrected_conclusion"] == "SEPARATED_BUT_DISTANCE_INFIMUM_ZERO"
    )


def frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        data = json.loads(raw)
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and data["source_row"] == 18
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/disjoint-closed-distance-scope-audit",
        conclusion="DISJOINT_CLOSED_DOES_NOT_IMPLY_POSITIVE_DISTANCE",
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if contract else None
    math_ok = bool(result_ok(result) and frozen_ok())
    evidence = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/disjoint-closed-distance-audit.json",
        )
        if contract
        else None
    )
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and evidence["schema_version"] == "1"
        and evidence["task_id"] == submission["task_id"]
        and evidence["result"] == result
        and evidence["limitations"] == submission["limitations"]
    )
    scope_ok = bool(
        contract
        and submission["scope"] == "EXACT_PARAMETRIC_METRIC_COUNTERMODEL"
        and submission["limitations"]
        == [
            "TOPOLOGICAL_CLOSEDNESS_REPLAYED_FROM_INTEGER_HORIZONTAL_GAP",
            "LEAN_ELABORATION_NOT_ASSESSED",
        ]
    )
    assurance_ok = bool(contract and submission["claimed_assurance"] == "COMPUTED")
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    reward = (
        1.0
        if math_ok and evidence_ok and scope_ok and assurance_ok and not false_cert
        else 0.0
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "false_certification": false_cert,
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
