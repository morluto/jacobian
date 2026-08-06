import json
from pathlib import Path

from verifier_support import (
    false_verified_claim,
    load_submission,
    read_evidence_json,
    strict_submission_contract,
)

W, E = Path("/app"), Path("/tests")


def topology_ok(n, raw):
    if not isinstance(raw, list) or not 5 <= len(raw) <= 64:
        return None
    opens = []
    for item in raw:
        if (
            not isinstance(item, list)
            or item != sorted(item)
            or len(set(item)) != len(item)
            or any(type(x) is not int or not 0 <= x < n for x in item)
        ):
            return None
        opens.append(frozenset(item))
    os = set(opens)
    universe = frozenset(range(n))
    if len(os) != len(opens) or frozenset() not in os or universe not in os:
        return None
    for a in os:
        for b in os:
            if a | b not in os or a & b not in os:
                return None
    return os


def is_t0(n, opens):
    return all(
        any((x in u) != (y in u) for u in opens)
        for x in range(n)
        for y in range(x + 1, n)
    )


def is_t2(n, opens):
    return all(
        any(x in u and y in v and not (u & v) for u in opens for v in opens)
        for x in range(n)
        for y in range(x + 1, n)
    )


def result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "factor_cardinalities",
        "bad_factor_topology",
        "empty_factor_index",
        "product_cardinality",
        "product_is_hausdorff",
        "bad_factor_is_t0",
        "bad_factor_is_hausdorff",
        "missing_assumption",
    }:
        return False
    cards = result["factor_cardinalities"]
    if (
        not isinstance(cards, list)
        or len(cards) != 3
        or any(type(x) is not int for x in cards)
    ):
        return False
    n, empty, other = cards
    if (
        not 4 <= n <= 7
        or empty != 0
        or not 2 <= other <= 5
        or result["empty_factor_index"] != 1
    ):
        return False
    opens = topology_ok(n, result["bad_factor_topology"])
    if opens is None:
        return False
    return (
        is_t0(n, opens)
        and not is_t2(n, opens)
        and result["product_cardinality"] == 0
        and result["product_is_hausdorff"] is True
        and result["bad_factor_is_t0"] is True
        and result["bad_factor_is_hausdorff"] is False
        and result["missing_assumption"] == "ALL_FACTORS_NONEMPTY"
    )


def frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        data = json.loads(raw)
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and data["source_row"] == 59
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/product-hausdorff-nonempty-scope-audit",
        conclusion="EMPTY_FACTOR_MASKS_NON_HAUSDORFF_FACTOR",
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if contract else None
    math_ok = bool(result_ok(result) and frozen_ok())
    evidence = (
        read_evidence_json(
            submission["evidence"][0],
            expected_path="evidence/product-hausdorff-audit.json",
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
        and submission["scope"] == "FINITE_TOPOLOGICAL_PRODUCT_SCOPE_COUNTERMODEL"
        and submission["limitations"]
        == ["FINITE_TOPOLOGIES_ONLY", "LEAN_ELABORATION_NOT_ASSESSED"]
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


if __name__ == "__main__":
    main()
