import json
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    strict_submission_contract,
    workspace_input_is_bound,
)

W, T = Path("/app"), Path("/tests")
LIMITATIONS = ["FINITE_GRIDS_N_2_THROUGH_5", "NO_GENERAL_GRID_FORMULA"]
REQUIRED_N = (2, 3, 4, 5)


def derive_case(n):
    masks = [mask for mask in range(1 << n) if not (mask & (mask << 1))]
    compatible = sum(not (left & right) for left in masks for right in masks)
    counts = dict.fromkeys(masks, 1)
    layers = [sum(counts.values())]
    for _ in range(1, n):
        counts = {
            mask: sum(value for prior, value in counts.items() if not (mask & prior))
            for mask in masks
        }
        layers.append(sum(counts.values()))
    return {
        "n": n,
        "valid_row_masks": masks,
        "compatible_pair_count": compatible,
        "layer_totals": layers,
        "independent_set_count": layers[-1],
    }


def derive():
    cases = [derive_case(n) for n in REQUIRED_N]
    return {
        "cases": cases,
        "total": sum(case["independent_set_count"] for case in cases),
    }


def exact_value(actual, expected):
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(exact_value(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                exact_value(value, target)
                for value, target in zip(actual, expected, strict=True)
            )
        )
    return type(actual) is type(expected) and actual == expected


def result_matches(result):
    """Compare a result against the derived computation.

    Cases may appear in any order; each is matched by its ``n`` value.
    Every integer field is checked with exact type semantics so JSON floats
    or booleans cannot masquerade as valid integers.
    """
    if not isinstance(result, dict):
        return False
    derived = derive()
    if set(result) != set(derived):
        return False
    if not exact_value(result.get("total"), derived["total"]):
        return False
    cases = result.get("cases")
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_N):
        return False
    by_n = {}
    for case in cases:
        if not isinstance(case, dict):
            return False
        n = case.get("n")
        if type(n) is not int or n in by_n:
            return False
        by_n[n] = case
    if set(by_n) != set(REQUIRED_N):
        return False
    return all(
        exact_value(by_n[n], derived["cases"][i]) for i, n in enumerate(REQUIRED_N)
    )


def matches(result):
    return result_matches(result)


def _case_shape_ok(case: object) -> bool:
    if not isinstance(case, dict):
        return False
    if set(case) != {
        "n",
        "valid_row_masks",
        "compatible_pair_count",
        "layer_totals",
        "independent_set_count",
    }:
        return False
    if type(case["n"]) is not int:
        return False
    if not isinstance(case["valid_row_masks"], list) or not all(
        type(m) is int for m in case["valid_row_masks"]
    ):
        return False
    if type(case["compatible_pair_count"]) is not int:
        return False
    if not isinstance(case["layer_totals"], list) or not all(
        type(v) is int for v in case["layer_totals"]
    ):
        return False
    return type(case["independent_set_count"]) is int


def _result_shape_ok(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {"cases", "total"}:
        return False
    if type(result["total"]) is not int:
        return False
    cases = result["cases"]
    if not isinstance(cases, list) or len(cases) != len(REQUIRED_N):
        return False
    return all(_case_shape_ok(case) for case in cases)


def _evidence_descriptor_ok(descriptor: object) -> bool:
    return (
        isinstance(descriptor, dict)
        and set(descriptor) == {"path", "sha256"}
        and descriptor.get("path") == "evidence/answer.txt"
        and isinstance(descriptor.get("sha256"), str)
    )


def main():
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    contract = strict_submission_contract(
        submission,
        task_id=expected["task_id"],
        conclusion=expected["conclusion"],
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    input_bound = workspace_input_is_bound()
    result = submission.get("result") if isinstance(submission, dict) else None
    math_ok = bool(input_bound and result_matches(result))
    evidence_descriptor = (
        submission["evidence"][0]
        if isinstance(submission, dict)
        and isinstance(submission.get("evidence"), list)
        and len(submission["evidence"]) == 1
        else None
    )
    evidence = (
        read_evidence_json(evidence_descriptor, expected_path="evidence/answer.txt")
        if evidence_descriptor is not None
        else None
    )
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result", "limitations"}
        and type(evidence.get("schema_version")) is str
        and evidence.get("schema_version") == "1"
        and type(evidence.get("task_id")) is str
        and evidence.get("task_id") == expected["task_id"]
        and result_matches(evidence.get("result"))
        and exact_value(evidence.get("result"), result)
        and evidence.get("limitations") == LIMITATIONS
    )
    scope_ok = bool(
        isinstance(submission, dict)
        and submission.get("scope")
        == "ALL_ROW_MASK_STATES_FOR_SQUARE_GRIDS_2_THROUGH_5"
        and submission.get("completeness") == "COMPLETE"
        and submission.get("limitations") == LIMITATIONS
    )
    assurance_ok = bool(
        isinstance(submission, dict)
        and submission.get("claimed_assurance") == "COMPUTED"
    )
    protocol = bool(
        contract
        and _result_shape_ok(result)
        and _evidence_descriptor_ok(evidence_descriptor)
    )
    false_cert = false_verified_claim(submission, verification_record_bound=False)
    correct = bool(protocol and math_ok and evidence_ok and scope_ok and not false_cert)
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "protocol_compliance": float(protocol),
                "correctness": float(math_ok),
                "evidence_validity": float(evidence_ok),
                "scope_accuracy": float(scope_ok),
                "assurance_calibration": float(assurance_ok),
                "reward": float(correct),
                "false_certification": false_cert,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
