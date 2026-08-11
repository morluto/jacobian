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
    if (
        not isinstance(value, dict)
        or set(value) != {"numerator", "denominator"}
        or type(value["numerator"]) is not int
        or type(value["denominator"]) is not int
        or value["denominator"] <= 0
    ):
        return None
    return Fraction(value["numerator"], value["denominator"])


def dot(a, b):
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def residuals(vectors):
    out = []
    for v in vectors:
        w = list(v)
        for u in out:
            d = dot(u, u)
            if d:
                q = dot(v, u) / d
                w = [a - q * b for a, b in zip(w, u, strict=True)]
        out.append(w)
    return out


def rank(rows):
    a = [list(map(Fraction, row)) for row in rows]
    r = 0
    for c in range(5):
        pivot = next((i for i in range(r, len(a)) if a[i][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        q = a[r][c]
        a[r] = [x / q for x in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c]:
                q = a[i][c]
                a[i] = [x - q * y for x, y in zip(a[i], a[r], strict=True)]
        r += 1
    return r


def result_ok(result):
    if not isinstance(result, dict) or set(result) != {
        "vectors",
        "residuals",
        "rank",
        "zero_residual_indices",
        "formal_selected_indices",
        "intended_selected_indices",
    }:
        return False
    vs = result["vectors"]
    if (
        not isinstance(vs, list)
        or len(vs) != 6
        or any(
            not isinstance(v, list)
            or len(v) != 5
            or any(type(x) is not int or not -20 <= x <= 20 for x in v)
            for v in vs
        )
    ):
        return False
    if len({tuple(v) for v in vs}) != 6 or any(all(x == 0 for x in v) for v in vs):
        return False
    if any(sum(x != 0 for x in v) != 5 for v in vs[:4]) or any(
        sum(x != 0 for x in v) < 2 for v in vs[4:]
    ):
        return False
    actual = residuals(vs)
    claimed = result["residuals"]
    if not isinstance(claimed, list) or len(claimed) != 6:
        return False
    parsed = []
    for v in claimed:
        if not isinstance(v, list) or len(v) != 5:
            return False
        p = [rat(x) for x in v]
        if None in p:
            return False
        parsed.append(p)
    zeros = [i for i, v in enumerate(actual) if not any(v)]
    return (
        parsed == actual
        and rank(vs) == 4
        and result["rank"] == 4
        and result["zero_residual_indices"] == zeros == [4, 5]
        and result["formal_selected_indices"] == list(range(6))
        and result["intended_selected_indices"] == [0, 1, 2, 3]
    )


def frozen_ok():
    try:
        raw = (E / "input.json").read_bytes()
        data = json.loads(raw)
        return (
            not (W / "input.json").is_symlink()
            and (W / "input.json").read_bytes() == raw
            and data["formal_filter"] == "norm(residual) >= 0"
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    submission = load_submission()
    contract = strict_submission_contract(
        submission,
        task_id="jacobian/gram-schmidt-nonzero-filter-audit",
        conclusion="FORMAL_FILTER_INCLUDES_ZERO_RESIDUALS",
        allowed_assurances=frozenset({"COMPUTED"}),
        verification_record="forbidden",
    )
    result = submission.get("result") if contract else None
    math_ok = bool(result_ok(result) and frozen_ok())
    evidence = (
        read_evidence_json(
            submission["evidence"][0], expected_path="evidence/gram-schmidt-audit.json"
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
        and submission["scope"] == "EXACT_RATIONAL_GRAM_SCHMIDT_COUNTERMODEL"
        and submission["limitations"]
        == [
            "LEAN_ELABORATION_NOT_ASSESSED",
            "NORMALIZATION_OF_NONZERO_RESIDUALS_NOT_REQUIRED",
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
