import json
import math
from fractions import Fraction
from pathlib import Path

from verifier_support import (
    aggregate_reward,
    json_value_equal,
    load_submission,
    normalize_reward_file,
    read_evidence_json,
    workspace_input_is_bound,
)

W = Path("/app")
T = Path("/tests")


def frozen_contract() -> dict:
    try:
        a, t = W / "input.json", T / "input.json"
        raw = t.read_bytes()
        if a.is_symlink() or t.is_symlink() or a.read_bytes() != raw:
            return {}
        value = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return (
        value
        if isinstance(value, dict)
        and value.get("task_id") == "jacobian/rank-one-spectral-limit-certificate"
        else {}
    )


def rat(value: object) -> Fraction | None:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        return None
    n, d = value.get("numerator"), value.get("denominator")
    if type(n) is not int or type(d) is not int or d <= 0 or math.gcd(abs(n), d) != 1:
        return None
    return Fraction(n, d)


def _checkpoints_valid(checkpoints: object) -> list[int] | None:
    if not isinstance(checkpoints, list) or not 3 <= len(checkpoints) <= 8:
        return None
    ns = []
    for item in checkpoints:
        if not isinstance(item, dict) or set(item) != {"n", "reciprocal_sum", "root"}:
            return None
        n = item["n"]
        if type(n) is not int or not 4 <= n <= 30:
            return None
        expected_sum = sum((Fraction(1, k**3 - k) for k in range(2, n + 1)), Fraction())
        if (
            rat(item["reciprocal_sum"]) != expected_sum
            or rat(item["root"]) != 1 / expected_sum
        ):
            return None
        ns.append(n)
    return ns


def _root_formula_valid(value: object, ns: list[int]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "coefficient",
        "numerator_offsets",
        "denominator_offsets",
    }:
        return False
    coefficient = value["coefficient"]
    numerator_offsets = value["numerator_offsets"]
    denominator_offsets = value["denominator_offsets"]
    if (
        type(coefficient) is not int
        or not isinstance(numerator_offsets, list)
        or not isinstance(denominator_offsets, list)
        or len(numerator_offsets) != 2
        or len(denominator_offsets) != 2
        or not all(type(offset) is int for offset in numerator_offsets)
        or not all(type(offset) is int for offset in denominator_offsets)
    ):
        return False
    if (
        coefficient != 4
        or sorted(numerator_offsets) != [0, 1]
        or sorted(denominator_offsets) != [-1, 2]
    ):
        return False
    return all(
        Fraction(
            coefficient * math.prod(n + offset for offset in numerator_offsets),
            math.prod(n + offset for offset in denominator_offsets),
        )
        == 1 / sum((Fraction(1, k**3 - k) for k in range(2, n + 1)), Fraction())
        for n in ns
    )


def certificate_valid(result: object) -> bool:
    if not isinstance(result, dict) or set(result) != {
        "rank_one_sign",
        "partial_fraction",
        "checkpoints",
        "root_formula",
        "limit",
    }:
        return False
    pf = result["partial_fraction"]
    if not isinstance(pf, dict) or set(pf) != {
        "scale",
        "left_coefficient",
        "right_coefficient",
    }:
        return False
    scale = rat(pf["scale"])
    if (
        scale != Fraction(1, 2)
        or pf["left_coefficient"] != 1
        or pf["right_coefficient"] != -1
    ):
        return False
    # Independently replay the proposed partial fraction on a wider range than the submitted checkpoints.
    for k in range(2, 51):
        proposed = scale * (Fraction(1, k * (k - 1)) - Fraction(1, k * (k + 1)))
        if proposed != Fraction(1, k**3 - k):
            return False
    ns = _checkpoints_valid(result["checkpoints"])
    if ns is None:
        return False
    return bool(
        len(ns) == len(set(ns))
        and result["rank_one_sign"] == "DIAGONAL_MINUS_LAMBDA_ONES"
        and _root_formula_valid(result["root_formula"], ns)
        and rat(result["limit"]) == 4
    )


def main() -> None:
    frozen = frozen_contract()
    expected = json.loads((T / "expected.json").read_text())
    submission = load_submission(W / "submission.json")
    input_binding = workspace_input_is_bound()
    evidence = (
        read_evidence_json(
            submission["witness"][0],
            expected_path="evidence/spectral-certificate.json",
        )
        if isinstance(submission, dict)
        else None
    )
    math_ok = bool(
        submission is not None
        and frozen
        and certificate_valid(submission.get("result"))
    )
    evidence_ok = bool(
        evidence
        and set(evidence) == {"schema_version", "task_id", "result"}
        and evidence.get("schema_version") == "1"
        and evidence.get("task_id") == expected["task_id"]
        and json_value_equal(evidence.get("result"), submission.get("result"))
    )
    reward = aggregate_reward(
        correctness=math_ok,
        witness_validity=evidence_ok,
        protocol_ok=bool(input_binding and submission is not None),
    )
    Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
    Path("/logs/verifier/reward.json").write_text(
        json.dumps(
            {
                "correctness": float(math_ok),
                "witness_validity": float(evidence_ok),
                "input_binding": float(input_binding),
                "reward": reward,
            }
        )
    )
    normalize_reward_file(Path("/logs/verifier/reward.json"))


if __name__ == "__main__":
    main()
